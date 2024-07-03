# This file is part of the Quickdraws GWAS software suite.
#
# Copyright (C) 2024 Quickdraws Developers
#
# Quickdraws is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Quickdraws is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Quickdraws. If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed
from pysnptools.distreader import Bgen
from pybgen import ParallelPyBGEN as PyBGEN
from scipy.stats import chi2
from tqdm import tqdm
import numba
import copy
import time
import pdb
from scipy.special import expit
from joblib import Parallel, delayed
import os

from .firth_logistic import firth_logit_covars, firth_logit_svt


def preprocess_covars(covarFile, iid_fid):
    if covarFile is None:
        covars = np.ones((len(iid_fid),1), dtype='float32')
        return covars
    covars = pd.read_csv(covarFile, sep=r'\s+')
    covars = pd.merge(
        pd.DataFrame(iid_fid.astype("int"), columns=["FID", "IID"]),
        covars,
        on=["FID", "IID"],
    )
    covars = covars.fillna(covars.median())
    covars = covars[covars.columns[2:]]
    covars = covars.loc[:, covars.std() > 0]
    covars["ALL_CONST"] = 1
    covars = np.array(covars.values, dtype="float32")
    return covars

def write_sumstats_file(bedFile, pheno_names, num_samples, afreq, beta, chisq, firth_arr, out):
    bim = pd.read_csv(
        bedFile + ".bim",
        sep=r'\s+',
        header=None,
        names=["CHR", "SNP", "GENPOS", "POS", "A1", "A2"],
    )
    bim["OBS_CT"] = num_samples
    bim["ALT_FREQS"] = afreq
    se = np.abs(beta) / np.sqrt(chisq)
    pval = chi2.sf(chisq, df=1)
    for pheno_no, pheno in enumerate(pheno_names):
        bim["BETA"] = beta[pheno_no]
        bim["SE"] = se[pheno_no]
        bim["CHISQ"] = chisq[pheno_no]
        bim["P"] = pval[pheno_no]
        if firth_arr is not None:
            bim["Firth"] = firth_arr[pheno_no]
        bim.to_hdf(
            "{0}.{1}.sumstats".format(out, pheno),
            key="sumstats",
            index=None,
            mode="w",
        )

def write_sumstats_file_bgen(
    snp_on_disk, bgenfile, pheno_names, num_samples, afreq, beta, chisq, firth_arr, out
):
    bim = pd.DataFrame()
    bim["CHR"] = np.array(snp_on_disk.pos[:, 0], dtype="int")
    bim["SNP"] = snp_on_disk.sid
    bim["GENPOS"] = snp_on_disk.pos[:, 1]
    bim["POS"] = snp_on_disk.pos[:, 2]
    bim["OBS_CT"] = num_samples
    bim["ALT_FREQS"] = afreq
    se = np.abs(beta) / np.sqrt(chisq)
    pval = chi2.sf(chisq, df=1)
    bim[['SNP_0', 'SNP_1']]= bim['SNP'].str.split(',', expand = True)
    try:
        bim[['base_pair', 'A1', 'A2']] = bim['SNP_0'].str.split('_', expand = True)
    except:
        bim[['base_pair', 'A1', 'A2']] = np.nan
    
    bim = bim.drop(columns=['SNP', 'GENPOS', 'base_pair'])
    bim = bim.rename(columns = {'SNP_1':'SNP'})

    ## Find Ref and Alt for SNPs not found by pysnptools
    if bim['A1'].isna().any():
        with PyBGEN(bgenfile) as f:
            iterator = f.iter_variants_by_names(bim.loc[bim['A1'].isna(), 'SNP'].values)
            name_a1_a2_dict = {}
            for var in iterator:
                name_a1_a2_dict[var[0].name] = [var[0].a1, var[0].a2]
            for index in bim.index[bim['A1'].isna()]:
                bim.loc[index, ['A1','A2']] = name_a1_a2_dict[bim.loc[index, 'SNP']]

    for pheno_no, pheno in enumerate(pheno_names):
        bim["BETA"] = beta[pheno_no]
        bim["SE"] = se[pheno_no]
        bim["CHISQ"] = chisq[pheno_no]
        bim["P"] = pval[pheno_no]
        if firth_arr is not None:
            bim["Firth"] = firth_arr[pheno_no]
        bim.to_hdf(
            "{0}.{1}.sumstats".format(out, pheno),
            key="sumstats",
            index=None,
            mode="w",
        )

def check_residuals_same_order(residualFileList):
    for chr_no in range(len(residualFileList)):
        df = pd.read_csv(residualFileList[chr_no], sep=r'\s+')
        if chr_no == 0:
            iid = df.IID.values
        else:
            assert (
                iid == df.IID.values
            ).all(), "Residuals file don't follow same order of FID, IID"

@numba.jit(nopython=True, parallel=True)
def MyLinRegr(X, Y, W, offset):
    ## Preprocess genotype first
    afreq = np.zeros(X.shape[1])
    num_samples = np.zeros(X.shape[1])
    for snp in numba.prange(X.shape[1]):
        isnan_at_snp = np.isnan(X[:, snp])
        num_samples[snp] = np.sum(~isnan_at_snp)
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
    if offset is not None:
        y_hat = Y - offset
    else:
        y_hat = Y - W.dot(K.dot(W.T.dot(Y)))
    numerators = X.T.dot(y_hat)
    for pheno in numba.prange(Y.shape[1]):
        var_y = Y[:, pheno].dot(y_hat[:, pheno])
        beta[pheno] = numerators[:, pheno] / var_X
        chisq[pheno] = (N - W.shape[1]) / (var_y * var_X / numerators[:, pheno] ** 2)
    return beta, chisq, afreq, num_samples


@numba.jit(nopython=True, parallel=True)
def MyLogRegr(X, Y, W, offset):
    ## Preprocess genotype first
    afreq = np.zeros(X.shape[1])
    num_samples = np.zeros(X.shape[1])
    for snp in numba.prange(X.shape[1]):
        isnan_at_snp = np.isnan(X[:, snp])
        num_samples[snp] = np.sum(~isnan_at_snp)
        freq = np.nansum(X[:, snp]) / np.sum(~isnan_at_snp)
        X[:, snp][isnan_at_snp] = 0
        X[:, snp][~isnan_at_snp] -= freq
        afreq[snp] = freq / 2
    N, M = X.shape
    beta, chisq = np.zeros((Y.shape[1], M)), np.zeros((Y.shape[1], M))
    y_hat = Y - offset
    Tau = offset * (1 - offset)
    for pheno in numba.prange(Y.shape[1]):
        numerators = X.T @ (y_hat[:, pheno])
        K = np.linalg.inv((W.T * Tau[:, pheno]) @ W)
        temp = W.T @ (X.T * Tau[:, pheno]).T
        denominator = np.array(
            [
                X[:, v].dot(X[:, v] * Tau[:, pheno]) + (temp[:, v]).T @ (K @ temp[:, v])
                for v in range(M)
            ]
        )
        chisq[pheno] = numerators**2 / denominator
        ##beta for logistic regression
        beta[pheno] = numerators / denominator
    return beta, chisq, afreq, num_samples

@numba.jit(nopython=True, parallel=True)
def MyWightedLinRegr(X, Y, C, offset, W):
    ## Preprocess genotype first
    afreq = np.zeros(X.shape[1])
    num_samples = np.zeros(X.shape[1])
    for snp in numba.prange(X.shape[1]):
        isnan_at_snp = np.isnan(X[:, snp])
        num_samples[snp] = np.sum(~isnan_at_snp)
        freq = np.nansum(X[:, snp]) / np.sum(~isnan_at_snp)
        X[:, snp][isnan_at_snp] = 0
        X[:, snp][~isnan_at_snp] -= freq
        afreq[snp] = freq / 2
    N, M = X.shape
    chisq = np.zeros((Y.shape[1], M))

    K = np.linalg.inv((C.T*W).dot(C))
    if offset is not None:
        ### Assumes offset doesn't have covariate effects
        y_hat = Y - offset
        y_hat = y_hat - C.dot(K.dot((C.T*W).dot(y_hat)))
    else:
        y_hat = Y - C.dot(K.dot((C.T*W).dot(Y)))

    temp = (C.T*W).dot(X)
    X_hat = X - C.dot(K.dot(temp))
    XWX_inv = 1 / np.array([X_hat[:, v].dot(W*X_hat[:, v]) for v in range(M)])
    XWY = (X_hat.T*W).dot(y_hat)
    beta = XWX_inv*XWY.T
    X_hat = X_hat.T
    X_hat_square_W_square = (W**2) * (X_hat ** 2)
    for pheno in numba.prange(y_hat.shape[1]):
        var_beta = np.array([np.sum(X_hat_square_W_square[v] * (y_hat[:, pheno] - beta[pheno, v]*X_hat[v])**2)  for v in range(M)])
        var_beta = XWX_inv**2 * var_beta
        chisq[pheno] = beta[pheno]**2/var_beta
    return beta, chisq, afreq, num_samples

def firth_parallel(
    chisq_snp, beta_snp, freq_snp, geno_snp, Y, pred_covars, covars, firth_pval_thresh, firth_maf_thresh, firth_prevalence_thresh
):
    chisq_out = chisq_snp.copy()
    beta_out = beta_snp.copy()
    firth_out = np.zeros_like(chisq_out, dtype='bool')
    is_rare_snp = (freq_snp < firth_maf_thresh) | (freq_snp > (1 - firth_maf_thresh)) ##caution
    is_rare_pheno = ((np.nanmean(Y, axis=0) < firth_prevalence_thresh) | (np.nanmean(Y, axis=0) > (1 - firth_prevalence_thresh)))
    pheno_mask = ((chi2.sf(chisq_snp, df=1) < firth_pval_thresh) & (is_rare_pheno | is_rare_snp))
    beta_firth, se, loglike_diff, iters = firth_logit_svt(
        geno_snp, Y[:, pheno_mask], pred_covars[:, pheno_mask], covars
    )
    chisq_out[pheno_mask] = 2 * loglike_diff
    beta_out[pheno_mask] = beta_firth
    firth_out[pheno_mask] = True
    return chisq_out, beta_out, firth_out

def firth_null_parallel(offset, Y, covar_effects, covars, out):
    offset_columns = offset.columns.tolist()
    iid_fid = offset[["FID", "IID"]]
    offset = offset[offset.columns[2:]].values.astype("float32")
    random_effects = offset - covar_effects
    pred_covars, _, _ = firth_logit_covars(covars, Y, random_effects)
    dfc = pd.concat([iid_fid, pd.DataFrame(pred_covars)], axis=1)
    dfc.columns = offset_columns
    dfc.to_csv(
        out + ".firth_null", sep="\t", index=None
    )
    # pred_covars = pd.read_csv(out + ".firth_null", sep="\t")
    # pred_covars = pred_covars[pred_covars.columns[2:]].values
    return pred_covars

def get_unadjusted_test_statistics(
    bedFile,
    pheno,
    covar_effects,
    offset_list,
    covarFile,
    out,
    unique_chrs,
    num_threads=-1,
    max_memory=1000,
    binary=False,
    firth=False,
    firth_pval_thresh=0.05,
    firth_maf_thresh=0.05,
    firth_prevalence_thresh=0.05,
    weights=None
):
    if num_threads >= 1:
        numba.set_num_threads(num_threads)

    snp_on_disk = Bed(bedFile, count_A1=True)
    chr_map = np.array(snp_on_disk.pos[:, 0], dtype="int")  ## chr_no

    samples_dict = {}
    for i, fid in enumerate(snp_on_disk.iid[:, 0]):
        samples_dict[int(fid)] = i
    sample_indices = []
    for fid in pheno.FID:
        sample_indices.append(samples_dict[int(fid)])
    iid_fid = snp_on_disk.iid[sample_indices]
    snp_on_disk = snp_on_disk[sample_indices, :]

    ## read covars and preprocess them
    covars = preprocess_covars(covarFile, iid_fid)
    assert len(covars) == len(iid_fid)

    ## calculate batch_size based on max_memory
    batch_size = int(max_memory * 1024 * 1024 / 8 / snp_on_disk.shape[0])
    beta_arr = np.zeros((pheno.shape[1] - 2, len(chr_map)), dtype="float32")
    chisq_arr = np.zeros((pheno.shape[1] - 2, len(chr_map)), dtype="float32")
    firth_arr = np.zeros((pheno.shape[1] - 2, len(chr_map)), dtype="bool")
    afreq_arr = np.zeros(len(chr_map), dtype="float32")
    num_samples_arr = np.zeros(len(chr_map), dtype="float32")
    Y = pheno[pheno.columns[2:]].values.astype("float32")

    if binary and firth:
        covar_effects = covar_effects[covar_effects.columns[2:]].values.astype("float32")
        pred_covars_arr = Parallel(n_jobs=num_threads)(
            delayed(firth_null_parallel)(
                offset_list[chr_no], Y, covar_effects, covars, out + str(chr)
            )
            for chr_no, chr in enumerate(np.unique(chr_map))
        )
    prev = 0
    for chr_no, chr in enumerate(np.unique(chr_map)):
        ## read offset file and adjust phenotype file
        try:
            offset = offset_list[chr_no]
            offset = offset[offset.columns[2:]].values.astype("float32")
        except:
            offset = None

        Y = pheno[pheno.columns[2:]].values.astype("float32")
        if weights is not None:
            weights_arr = weights[weights.columns[2]].values.astype("float32")
        # if not binary:
        #     Y -= np.mean(Y, axis=0)
        #     offset -= np.mean(offset, axis=0)
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
                beta, chisq, afreq, num_samples = MyLogRegr(X, Y, covars, expit(offset))
            else:
                if weights is None:
                    beta, chisq, afreq, num_samples = MyLinRegr(X, Y, covars, offset)
                else:
                    beta, chisq, afreq, num_samples = MyWightedLinRegr(X, Y, covars, offset, weights_arr)

            beta_arr[
                :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = beta
            chisq_arr[
                :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = chisq
            afreq_arr[
                prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = afreq
            num_samples_arr[
                prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = num_samples
            if binary and firth:
                par_out = Parallel(n_jobs=num_threads)(
                    delayed(firth_parallel)(
                        chisq[:, snp],
                        beta[:, snp],
                        afreq[snp],
                        X[:, snp : snp + 1],
                        Y,
                        pred_covars_arr[chr_no],
                        covars,
                        firth_pval_thresh,
                        firth_maf_thresh,
                        firth_prevalence_thresh
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
                firth_arr[
                    :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
                ] = par_out[:, 2].T

        prev += num_snps_in_chr

    write_sumstats_file(
        bedFile,
        pheno.columns[2:],
        num_samples_arr,
        afreq_arr,
        beta_arr,
        chisq_arr,
        firth_arr if binary else None,
        out,
    )

def get_unadjusted_test_statistics_bgen(
    bgenFile,
    sampleFile,
    pheno,
    covar_effects,
    offset_list,
    covarFile,
    out,
    unique_chrs,
    snps_to_keep_filename=None,
    num_threads=-1,
    max_memory=1000,
    binary=False,
    firth=False,
    firth_pval_thresh=0.05,
    firth_maf_thresh=0.05,
    firth_prevalence_thresh=0.05,
    firth_null=None,
    weights=None
):
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
        snps_to_keep = pd.read_csv(snps_to_keep_filename, sep=r'\s+', names=["SNP"])
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

    samples_dict = {}
    for i, fid in enumerate(fid_iid[:, 0]):
        samples_dict[int(fid)] = i
    sample_indices = []
    for fid in pheno.FID:
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
    batch_size = int(max_memory * 1024 * 1024 / 8 / snp_on_disk.shape[0])
    beta_arr = np.zeros((pheno.shape[1] - 2, len(chr_map)), dtype="float32")
    chisq_arr = np.zeros((pheno.shape[1] - 2, len(chr_map)), dtype="float32")
    firth_arr = np.zeros((pheno.shape[1] - 2, len(chr_map)), dtype="bool")
    afreq_arr = np.zeros(len(chr_map), dtype="float32")
    num_samples_arr = np.zeros(len(chr_map), dtype="float32")
    Y = pheno[pheno.columns[2:]].values.astype("float32")

    if binary and firth and firth_null is None:
        covar_effects = covar_effects[covar_effects.columns[2:]].values.astype("float32")
        pred_covars_arr = Parallel(n_jobs=num_threads)(
            delayed(firth_null_parallel)(
                offset_list[chr_no], Y, covar_effects, covars
            )
            for chr_no in range(len(np.unique(chr_map)))
        )
    elif binary and firth and firth_null is not None:
        pred_covars_arr = []
        for chr_no in range(len(np.unique(chr_map))):
            firth_null_chr = firth_null[chr_no]
            mdf = pd.merge(
                offset_list[chr_no][["IID", "FID"]],
                firth_null_chr,
                on=["FID", "IID"],
            )
            pred_covars_arr.append(mdf.values[:, 2:])

    prev = 0
    for chr_no, chr in enumerate(unique_chrs):
        ## read offset file and adjust phenotype file
        try:
            offset = offset_list[chr_no]
            offset = np.array(offset[offset.columns[2:]].values, dtype="float32")
        except:
            offset = None

        Y = pheno[pheno.columns[2:]].values.astype("float32")
        if weights is not None:
            weights_arr = weights[weights.columns[2]].values.astype("float32")
        # if not binary:
        #     Y -= np.mean(Y, axis=0)
        #     offset -= np.mean(offset, axis=0)

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
                beta, chisq, afreq, num_samples = MyLogRegr(X, Y, covars, expit(offset))
            else:
                if weights is None:
                    beta, chisq, afreq, num_samples = MyLinRegr(X, Y, covars, offset)
                else:
                    beta, chisq, afreq, num_samples = MyWightedLinRegr(X, Y, covars, offset, weights_arr)

            beta_arr[
                :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = beta
            chisq_arr[
                :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = chisq
            afreq_arr[
                prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = afreq
            num_samples_arr[
                prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
            ] = num_samples
            if binary and firth:
                par_out = Parallel(n_jobs=num_threads)(
                    delayed(firth_parallel)(
                        chisq[:, snp],
                        beta[:, snp],
                        afreq[snp],
                        X[:, snp : snp + 1],
                        Y,
                        pred_covars_arr[chr_no],
                        covars,
                        firth_pval_thresh,
                        firth_maf_thresh,
                        firth_prevalence_thresh
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
                firth_arr[
                    :, prev + batch : prev + min(batch + batch_size, num_snps_in_chr)
                ] = par_out[:, 2].T

        prev += num_snps_in_chr

    write_sumstats_file_bgen(
        snp_on_disk,
        bgenFile,
        pheno.columns[2:],
        num_samples_arr,
        afreq_arr,
        beta_arr,
        chisq_arr,
        firth_arr if binary else None,
        out,
    )


if __name__ == "__main__":
    X = np.random.randn(1000, 100)
    Y = np.random.randn(1000, 5)
    # covars = np.random.randn(1000, 20)
    covars = np.ones((1000, 1))
    offset = None
    weights_arr = 5*np.random.rand(1000,)
    beta1, chisq1, afreq1 = MyLinRegr(X, Y, covars, offset)
    beta2, chisq2, afreq2 = MyWightedLinRegr(X, Y, covars, offset, weights_arr)
    

    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression

    model = LinearRegression() # Create a linear regression model
    model.fit(sm.add_constant(X[:,0]), Y, sample_weight=weights_arr) # Fit the model with weights
    results = sm.WLS(Y[:,2], sm.add_constant(X[:,0]), weights=weights_arr).fit() # Fit an OLS model with weights
    robust_results = results.get_robustcov_results() # Get the robust results
    print(robust_results.summary()) # Print the summary

