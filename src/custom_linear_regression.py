import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed
from pysnptools.distreader import Bgen
from scipy.stats import chi2
from tqdm import tqdm
import numba
import copy
import time
import pdb
from firth_logistic import firth_logit_covars, firth_logit_svt
from scipy.special import logit
from joblib import Parallel, delayed


def preprocess_covars(covarFile, iid_fid):
    covars = pd.read_csv(covarFile, sep="\s+")
    covars = pd.merge(
        pd.DataFrame(iid_fid.astype("int"), columns=["FID", "IID"]),
        covars,
        on=["FID", "IID"],
    )
    for covar_col in covars.columns[2:]:
        covars[covar_col] = covars[covar_col].fillna(np.nanmean(covars[covar_col]))
    covars = covars[covars.columns[2:]]
    covars = (covars - covars.min(axis=0)) / (covars.max(axis=0) - covars.min(axis=0))
    covars["ALL_CONST"] = 1
    covars = np.array(covars.values, dtype="float32")
    return covars


def write_sumstats_file(bedFile, pheno_names, num_samples, afreq, beta, chisq, out):
    bim = pd.read_csv(
        bedFile + ".bim",
        sep="\s+",
        header=None,
        names=["#CHROM", "SNP", "GENPOS", "POS", "A1", "A2"],
    )
    bim["OBS_CT"] = num_samples
    bim["ALT_FREQS"] = afreq
    se = np.abs(beta) / np.sqrt(chisq)
    pval = chi2.sf(chisq, df=1)
    for pheno_no, pheno in enumerate(pheno_names):
        bim["BETA"] = -beta[pheno_no]
        bim["SE"] = se[pheno_no]
        bim["CHISQ"] = chisq[pheno_no]
        bim["P"] = pval[pheno_no]
        bim.to_hdf(
            "{0}.{1}.sumstats".format(out, pheno),
            key="sumstats",
            index=None,
            mode="w",
        )


def write_sumstats_file_bgen(
    snp_on_disk, pheno_names, num_samples, afreq, beta, chisq, out
):
    bim = pd.DataFrame()
    bim["#CHROM"] = np.array(snp_on_disk.pos[:, 0], dtype="int")
    bim["SNP"] = snp_on_disk.sid
    bim["GENPOS"] = snp_on_disk.pos[:, 1]
    bim["POS"] = snp_on_disk.pos[:, 2]
    bim["OBS_CT"] = num_samples
    bim["ALT_FREQS"] = afreq
    se = np.abs(beta) / np.sqrt(chisq)
    pval = chi2.sf(chisq, df=1)
    for pheno_no, pheno in enumerate(pheno_names):
        bim["BETA"] = -beta[pheno_no]
        bim["SE"] = se[pheno_no]
        bim["CHISQ"] = chisq[pheno_no]
        bim["P"] = pval[pheno_no]
        bim.to_hdf(
            "{0}.{1}.sumstats".format(out, pheno),
            key="sumstats",
            index=None,
            mode="w",
        )


def check_residuals_same_order(residualFileList):
    for chr_no in range(len(residualFileList)):
        df = pd.read_csv(residualFileList[chr_no], sep="\s+")
        if chr_no == 0:
            iid = df.IID.values
        else:
            assert (
                iid == df.IID.values
            ).all(), "Residuals file don't follow same order of FID, IID"


@numba.jit(nopython=True, parallel=True)
def MyLinRegr(X, Y, W, offset):
    """
    Author: Yiorgos + Hrushi
    DIY Linear regression with covariates and low memory footprint.
    X should be NxM, for sufficiently large M (e.g. one chromosome), and mean centered
    Y is multi-phenotype but should also be mean centered.
    W needs to be a NxC array with covariates including ones (the constant).
    Returns a DataFrame with estimated effect sizes, Chi-Sqr statistics, and p-values
    """

    ## Preprocess genotype first
    afreq = np.zeros(X.shape[1])
    for snp in numba.prange(X.shape[1]):
        isnan_at_snp = np.isnan(X[:, snp])
        freq = np.nansum(X[:, snp]) / np.sum(~isnan_at_snp)
        X[:, snp][isnan_at_snp] = 0
        X[:, snp][~isnan_at_snp] -= freq
        afreq[snp] = freq / 2

    N, M = X.shape
    beta, chisq = np.zeros((Y.shape[1], M)), np.zeros((Y.shape[1], M))
    K = np.linalg.inv(W.T.dot(W))
    temp = W.T.dot(X)
    var_X = np.array(
        [X[:, v].dot(X[:, v]) - temp[:, v].dot(K.dot(temp[:, v])) for v in range(M)]
    )
    Y = Y - offset
    y_hat = Y - W.dot(K.dot(W.T.dot(Y)))
    numerators = X.T.dot(y_hat)
    for pheno in numba.prange(Y.shape[1]):
        var_y = Y[:, pheno].dot(y_hat[:, pheno])
        beta[pheno] = numerators[:, pheno] / var_X
        chisq[pheno] = (N - W.shape[1]) / (var_y * var_X / numerators[:, pheno] ** 2)
    return beta, chisq, afreq


@numba.jit(nopython=True, parallel=True)
def MyLogRegr(X, Y, W, offset):
    """
    Author: Hrushi
    DIY Logistic regression with covariates and low memory footprint.
    X should be NxM, for sufficiently large M (e.g. one chromosome), and mean centered
    Y is multi-phenotype but should also be mean centered.
    W needs to be a NxC array with covariates including ones (the constant).
    offset is a NxP array with the offset for each phenotype
    Returns a DataFrame with estimated effect sizes, Chi-Sqr statistics, and p-values
    """

    ## Preprocess genotype first
    afreq = np.zeros(X.shape[1])
    for snp in numba.prange(X.shape[1]):
        isnan_at_snp = np.isnan(X[:, snp])
        freq = np.nansum(X[:, snp]) / np.sum(~isnan_at_snp)
        X[:, snp][isnan_at_snp] = 0
        X[:, snp][~isnan_at_snp] -= freq
        afreq[snp] = freq / 2

    N, M = X.shape
    beta, chisq = np.zeros((Y.shape[1], M)), np.zeros((Y.shape[1], M))
    K = np.linalg.inv(W.T.dot(W))
    Y1 = Y - offset
    y_hat = Y1 - W.dot(K.dot(W.T.dot(Y1)))
    offset = Y - y_hat
    Tau = offset * (1 - offset)
    for pheno in numba.prange(Y.shape[1]):
        g_hat = X - (
            W
            @ (
                np.linalg.inv((W.T * Tau[:, pheno]) @ W)
                @ (W.T @ (X.T * Tau[:, pheno]).T)
            )
        )
        numerators = g_hat.T @ (y_hat[:, pheno])
        denominator = np.array(
            [g_hat[:, v].dot(g_hat[:, v] * Tau[:, pheno]) for v in range(M)]
        )
        chisq[pheno] = numerators**2 / denominator
        ##beta for logistic regression
        beta[pheno] = numerators / denominator
    return beta, chisq, afreq


def firth_parallel(chisq_snp, beta_snp, geno_snp, Y, pred_covars, covars):
    chisq_out = chisq_snp.copy()
    beta_out = beta_snp.copy()
    pheno_mask = chi2.sf(chisq_snp, df=1) < 0.1  ### caution
    beta_firth, se, loglike_diff, iters = firth_logit_svt(
        geno_snp, Y[:, pheno_mask], pred_covars[:, pheno_mask], covars
    )
    chisq_out[pheno_mask] = 2 * loglike_diff
    beta_out[pheno_mask] = beta_firth
    return chisq_out, beta_out


def firth_null_parallel(offsetFile, Y, covar_effects, covars):
    offset = pd.read_csv(offsetFile, sep="\s+")
    offset = offset[offset.columns[2:]].values.astype("float32")
    random_effects = logit(offset) - covar_effects
    pred_covars, _, _ = firth_logit_covars(covars, Y, random_effects)
    return pred_covars


def get_unadjusted_test_statistics(
    bedFile,
    phenoFileList,
    offsetFileList,
    covarFile,
    out,
    unique_chrs,
    num_threads=-1,
    max_memory=1000,
    binary=False,
    firth=False,
):
    if num_threads >= 1:
        numba.set_num_threads(num_threads)

    snp_on_disk = Bed(bedFile, count_A1=True)
    chr_map = np.array(snp_on_disk.pos[:, 0], dtype="int")  ## chr_no

    offset = pd.read_csv(offsetFileList[0], sep="\s+")
    samples_dict = {}
    for i, fid in enumerate(snp_on_disk.iid[:, 0]):
        samples_dict[int(fid)] = i
    sample_indices = []
    for fid in offset.FID:
        sample_indices.append(samples_dict[int(fid)])
    iid_fid = snp_on_disk.iid[sample_indices]
    snp_on_disk = snp_on_disk[sample_indices, :]

    ## read covars and preprocess them
    covars = preprocess_covars(covarFile, iid_fid)
    assert len(covars) == len(iid_fid)

    ## calculate batch_size based on max_memory
    batch_size = int(max_memory * 1024 * 1024 / 8 / snp_on_disk.shape[0])
    beta_arr = np.zeros((offset.shape[1] - 2, len(chr_map)))
    chisq_arr = np.zeros((offset.shape[1] - 2, len(chr_map)))
    afreq_arr = np.zeros(len(chr_map))

    print("Running linear/logistic regression to get association")

    covar_effects = pd.read_csv(
        phenoFileList[0].split(".traits")[0] + ".covar_effects", sep="\s+"
    )
    covar_effects = covar_effects[covar_effects.columns[2:]].values.astype("float32")
    pheno = pd.read_csv(phenoFileList[0], sep="\s+")
    Y = pheno[pheno.columns[2:]].values.astype("float32")

    if binary and firth:
        pred_covars_arr = Parallel(n_jobs=num_threads)(
            delayed(firth_null_parallel)(
                offsetFileList[chr_no], Y, covar_effects, covars
            )
            for chr_no in range(len(np.unique(chr_map)))
        )
    prev = 0
    for chr_no, chr in enumerate(np.unique(chr_map)):
        ## read offset file and adjust phenotype file
        offset = pd.read_csv(offsetFileList[chr_no], sep="\s+")
        pheno = pd.read_csv(phenoFileList[chr_no], sep="\s+")
        offset = offset[offset.columns[2:]].values.astype("float32")
        Y = pheno[pheno.columns[2:]].values.astype("float32")
        # Y -= np.mean(Y, axis=0)
        # offset -= np.mean(offset, axis=0)
        num_snps_in_chr = int(np.sum(chr_map == int(chr)))
        for batch in tqdm(range(0, num_snps_in_chr, batch_size)):

            ## read genotype and calculate Alt allele freq
            X = 2 - (
                snp_on_disk[
                    :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
                ]
                .read(dtype="float32", num_threads=num_threads)
                .val
            )
            ## preprocess genotype and perform linear regression
            if binary:
                beta, chisq, afreq = MyLogRegr(X, Y, covars, offset)
            else:
                beta, chisq, afreq = MyLinRegr(X, Y, covars, offset)
            beta_arr[
                :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = beta
            chisq_arr[
                :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = chisq
            afreq_arr[
                prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = afreq
            if binary and firth:
                par_out = Parallel(n_jobs=num_threads)(
                    delayed(firth_parallel)(
                        chisq[:, snp],
                        beta[:, snp],
                        X[:, snp : snp + 1],
                        Y,
                        pred_covars_arr[chr_no],
                        covars,
                    )
                    for snp in range(0, chisq.shape[1])
                )
                par_out = np.array(par_out)
                chisq_arr[
                    :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
                ] = par_out[:, 0].T
                beta_arr[
                    :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
                ] = par_out[:, 1].T

        prev += num_snps_in_chr

    write_sumstats_file(
        bedFile,
        pd.read_csv(offsetFileList[chr_no], sep="\s+").columns[2:],
        snp_on_disk.shape[0],
        afreq_arr,
        beta_arr,
        chisq_arr,
        out,
    )


def get_unadjusted_test_statistics_bgen(
    bgenFile,
    sampleFile,
    phenoFileList,
    offsetFileList,
    covarFile,
    out,
    unique_chrs,
    snps_to_keep_filename=None,
    num_threads=-1,
    max_memory=7500,
    binary=False,
):
    raise NotImplementedError("This function is not implemented yet")
    if num_threads >= 1:
        numba.set_num_threads(num_threads)

    snp_on_disk = Bgen(bgenFile, sample=sampleFile)
    snp_on_disk = snp_on_disk.as_snp(max_weight=2)

    fid_iid = np.array(
        [
            [
                int(snp_on_disk.iid[i, 1].split(" ")[0]),
                int(snp_on_disk.iid[i, 1].split(" ")[1]),
            ]
            for i in range(snp_on_disk.shape[0])
        ]
    )

    ## --extract flag
    if snps_to_keep_filename is None:
        total_snps = snp_on_disk.sid_count
        snp_mask = np.ones(total_snps, dtype="bool")
    else:
        snps_to_keep = pd.read_csv(snps_to_keep_filename, sep="\s+", names=["SNP"])
        snps_to_keep = snps_to_keep[snps_to_keep.columns[0]].values
        snp_dict = {}
        total_snps = snp_on_disk.sid_count
        snp_mask = np.zeros(total_snps, dtype="bool")
        for snp_no, snp in enumerate(snp_on_disk.sid):
            snp_dict[snp] = snp_no
            snp_dict[snp.split(",")[1]] = snp_no
        for snp in snps_to_keep:
            try:
                snp_mask[snp_dict[snp]] = True
            except:
                pass

        del snp_dict
    ### Caution!! only removed temporarily

    offset = pd.read_csv(offsetFileList[0], sep="\s+")

    samples_dict = {}
    for i, fid in enumerate(fid_iid[:, 0]):
        samples_dict[int(fid)] = i
    sample_indices = []
    for fid in offset.FID:
        sample_indices.append(samples_dict[int(fid)])
    iid_fid_in_bgen = fid_iid[sample_indices]
    snp_on_disk = snp_on_disk[sample_indices, snp_mask]
    chr_map = np.array(snp_on_disk.pos[:, 0], dtype="int")  ## chr_no
    del samples_dict

    ## read covars and preprocess them
    covars = preprocess_covars(covarFile, iid_fid_in_bgen)
    assert len(covars) == len(iid_fid_in_bgen)

    ## calculate batch_size based on max_memory
    ## extra divide by 3 because bgen files give a distrution data as output (aa, Aa, AA)
    batch_size = int(max_memory * 1024 * 1024 / 8 / snp_on_disk.shape[0] / 3)
    beta_arr = np.zeros((offset.shape[1] - 2, len(chr_map)), dtype="float32")
    chisq_arr = np.zeros((offset.shape[1] - 2, len(chr_map)), dtype="float32")
    afreq_arr = np.zeros(len(chr_map), dtype="float32")

    print("Running linear/logistic regression to get association")

    prev = 0
    for chr_no, chr in enumerate(unique_chrs):
        ## read offset file and adjust phenotype file
        offset = pd.read_csv(offsetFileList[chr_no], sep="\s+")
        offset = np.array(offset[offset.columns[2:]].values, dtype="float32")
        pheno = pd.read_csv(phenoFileList[chr_no], sep="\s+")
        Y = pheno[pheno.columns[2:]].values.astype("float32")
        Y -= np.mean(Y, axis=0)
        offset -= np.mean(offset, axis=0)

        num_snps_in_chr = int(np.sum(chr_map == int(chr)))
        for batch in tqdm(range(0, num_snps_in_chr, batch_size)):

            ## read genotype and calculate Alt allele freq
            st = time.time()
            X = 2 - (
                snp_on_disk[
                    :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
                ]
                .read(dtype="float32", num_threads=num_threads)
                .val
            )
            print("1: " + str(time.time() - st))
            st = time.time()
            ## preprocess genotype and perform linear regression
            if binary:
                beta, chisq, afreq = MyLogRegr(X, Y, covars, offset)
            else:
                beta, chisq, afreq = MyLinRegr(X, Y, covars, offset)

            beta_arr[
                :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = beta
            chisq_arr[
                :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = chisq
            afreq_arr[
                prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = afreq
            print("2: " + str(time.time() - st))

        prev += num_snps_in_chr

    write_sumstats_file_bgen(
        snp_on_disk,
        pd.read_csv(offsetFileList[chr_no], sep="\s+").columns[2:],
        snp_on_disk.shape[0],
        afreq_arr,
        beta_arr,
        chisq_arr,
        out,
    )


if __name__ == "__main__":

    bedFile = "simulate/ukbb50k_unrel_gbp"
    residualFileList = []
    for chr in range(1, 23):
        residualFileList.append(
            "output/exact_loco_blr_unrel_gbploco_chr" + str(chr) + ".residuals"
        )
    # for chr in range(1, 23):
    #     residualFileList.append("simulate/ukbb50k_all.pheno")
    # residualFile = "simulate/ukbb50k_all.pheno"
    covarFile = "simulate/covariates.tab"
    out = "output/test_blr_all"
    # get_unadjusted_test_statistics(
    #     bedFile, residualFileList, covarFile, out, np.arange(1, 23), num_threads=8
    # )

    bgenFile = "ukbb_gbp/ukb_imp_chr18_v3.bgen"
    sampleFile = "/well/palamara/projects/UKBB_APPLICATION_43206/new_copy/data_download/ukb22828_c22_b0_v3_s487276.sample"
    get_unadjusted_test_statistics_bgen(
        bgenFile,
        sampleFile,
        residualFileList,
        covarFile,
        out,
        np.arange(1, 23),
        num_threads=8,
    )
