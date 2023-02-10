"""
Code for getting the test-statistics from LOCO residuals
Author: Yiorgos + Hrushi
"""
import os
import numpy as np
from pysnptools.snpreader import Bed
import subprocess
from multiprocessing import get_context
from multiprocessing.pool import Pool as Pool
from scipy.stats import chi2
import pandas as pd
from functools import partial
import argparse
from distutils.util import strtobool
import h5py
import time
from joblib import Parallel, delayed
from scipy.special import expit

from ldscore_calibration import calc_ldscore_chip, ldscore_intercept, get_mask_dodgy
from custom_linear_regression import (
    get_unadjusted_test_statistics,
    get_unadjusted_test_statistics_bgen,
)
from preprocess_phenotypes import preprocess_phenotypes, PreparePhenoRHE
import pdb


def str_to_bool(s: str) -> bool:
    return bool(strtobool(s))


def preprocess_offsets(
    bedfile, offsets, sample_indices, pheno_columns, sample_file=None, adj_suffix=""
):
    snp_on_disk = Bed(bedfile, count_A1=True)
    iid_fid = snp_on_disk.iid[sample_indices]
    df_iid_fid = pd.DataFrame(np.array(iid_fid, dtype=int), columns=["FID", "IID"])
    df_offsets = pd.read_csv(offsets, sep="\s+")
    if "IID" not in df_offsets.columns:
        df_concat = pd.concat([df_iid_fid, df_offsets], axis=1)
    else:
        df_concat = df_offsets
    df_concat.columns = pheno_columns
    if sample_file is not None:
        sample_file = pd.read_csv(sample_file, sep="\s+")
        sample_file = sample_file.rename(columns={"ID_1": "FID", "ID_2": "IID"})
        sample_file[["FID", "IID"]] = sample_file[["FID", "IID"]].astype("int")
        df_concat = pd.merge(df_concat, sample_file, on=["FID", "IID"])
        df_concat = df_concat[pheno_columns]

    print(offsets + " : " + str(df_concat.shape))
    df_concat.to_csv(offsets + adj_suffix, sep="\t", index=None)


def multi_run(cmd):
    subprocess.run(cmd, shell=True)
    return


def get_mask_dodgy_parallel(ldscores, bedfile, out, pheno):
    sumstats = pd.read_hdf("{0}.{1}.sumstats".format(out, pheno), key="sumstats")
    mask_dodgy = get_mask_dodgy(ldscores[["SNP", "LDSCORE"]], sumstats, bedfile)
    return mask_dodgy


def adjust_test_stats(out, pheno, correction):
    sumstats_cur = pd.read_hdf("{0}.{1}.sumstats".format(out, pheno), key="sumstats")
    sumstats_cur["CHISQ"] *= correction
    sumstats_cur["P"] = chi2.sf(sumstats_cur.CHISQ, df=1)
    sumstats_cur["CHISQ"] = sumstats_cur["CHISQ"].map(lambda x: "{:.8f}".format(x))
    sumstats_cur["P"] = sumstats_cur["P"].map(lambda x: "{:.2E}".format(x))

    sumstats_cur.to_csv(
        "{0}.{1}.sumstats".format(out, pheno),
        sep="\t",
        index=None,
    )

    os.system("gzip -f " + "{0}.{1}.sumstats".format(out, pheno))


def calibrate_test_stats(ldscores, bedfile, unrel_sample_list, out, pheno):
    sumstats_ref = pd.read_hdf(
        "{0}.{1}.sumstats".format(out + "_lrunrel", pheno), key="sumstats"
    )
    mask_dodgy_ref = get_mask_dodgy(
        ldscores[["SNP", "LDSCORE"]], sumstats_ref, np.mean(sumstats_ref["OBS_CT"])
    )
    ldscore_chip_ref = calc_ldscore_chip(bedfile, mask_dodgy_ref, unrel_sample_list)
    intercept_ref, mean_sumstats_ref = ldscore_intercept(
        ldscores, sumstats_ref, ldscore_chip_ref, mask_dodgy_ref
    )
    atten_ratio_ref = (intercept_ref - 1) / (mean_sumstats_ref - 1)

    sumstats_cur = pd.read_hdf("{0}.{1}.sumstats".format(out, pheno), key="sumstats")
    mask_dodgy_cur = get_mask_dodgy(
        ldscores[["SNP", "LDSCORE"]], sumstats_cur, np.mean(sumstats_cur["OBS_CT"])
    )
    ldscore_chip_cur = calc_ldscore_chip(bedfile, mask_dodgy_cur)
    intercept_cur, mean_sumstats_cur = ldscore_intercept(
        ldscores, sumstats_cur, ldscore_chip_cur, mask_dodgy_cur
    )
    correction = (1 - atten_ratio_ref) / (
        intercept_cur - atten_ratio_ref * mean_sumstats_cur
    )
    # correction = intercept_ref / intercept_cur

    sumstats_cur["CHISQ"] *= correction
    sumstats_cur["P"] = chi2.sf(sumstats_cur.CHISQ, df=1)
    sumstats_cur["CHISQ"] = sumstats_cur["CHISQ"].map(lambda x: "{:.8f}".format(x))
    sumstats_cur["P"] = sumstats_cur["P"].map(lambda x: "{:.2E}".format(x))

    sumstats_cur.to_csv(
        "{0}.{1}.sumstats".format(out, pheno),
        sep="\t",
        index=None,
    )

    os.system("gzip -f " + "{0}.{1}.sumstats".format(out, pheno))
    return correction


def get_test_statistics(
    bedfile,
    phenofile,
    covareffectsfile,
    offset,
    sample_indices,
    unrel_homo_file,
    ldscores,
    covar,
    calibrate,
    out="out",
    binary=False,
    n_workers=-1,
):
    if n_workers == -1:
        n_workers = os.cpu_count() - 1

    snp_on_disk = Bed(bedfile, count_A1=True)
    unique_chrs = np.unique(np.array(snp_on_disk.pos[:, 0], dtype=int))
    traits = pd.read_csv(phenofile, sep="\s+")
    pheno_columns = traits.columns.tolist()
    print(pheno_columns)

    Parallel(n_jobs=n_workers)(
        delayed(preprocess_offsets)(
            bedfile, offset + str(C) + ".offsets", sample_indices, pheno_columns
        )
        for C in unique_chrs
    )
    preprocess_offsets(bedfile, phenofile, sample_indices, pheno_columns)
    preprocess_offsets(bedfile, covareffectsfile, sample_indices, pheno_columns)

    if calibrate:
        # Run LR-unRel using our numba implementation
        unrel_homo = pd.read_csv(unrel_homo_file, names=["FID", "IID"], sep="\s+")
        covareffects = pd.read_csv(covareffectsfile, sep="\s+")
        unrel_sample_covareffect = pd.merge(covareffects, unrel_homo, on=["FID", "IID"])
        if binary:
            unrel_sample_covareffect[unrel_sample_covareffect.columns[2:]] = expit(
                unrel_sample_covareffect[unrel_sample_covareffect.columns[2:]].values
            )
        unrel_sample_covareffect.to_csv(
            out + ".covar_effects.unrel", sep="\t", index=None
        )
        unrel_sample_traits = pd.merge(traits, unrel_homo, on=["FID", "IID"])
        unrel_sample_traits.to_csv(out + ".traits.unrel", sep="\t", index=None)
        samples_dict = {}
        for i, fid in enumerate(snp_on_disk.iid[:, 0]):
            samples_dict[int(fid)] = i
        unrel_sample_indices = []
        for fid in unrel_sample_covareffect.FID:
            unrel_sample_indices.append(samples_dict[int(fid)])

        get_unadjusted_test_statistics(
            bedfile,
            [out + ".traits.unrel"] * len(unique_chrs),
            [out + ".covar_effects.unrel"] * len(unique_chrs),
            covar,
            out + "_lrunrel",
            unique_chrs,
            num_threads=n_workers,
            binary=binary,
        )
    offsetFileList = [offset + str(chr) + ".offsets" for chr in unique_chrs]
    get_unadjusted_test_statistics(
        bedfile,
        [phenofile] * len(unique_chrs),
        offsetFileList,
        covar,
        out,
        unique_chrs,
        num_threads=n_workers,
        binary=binary,
    )
    if calibrate:
        if ldscores is None:
            ldscores = calc_ldscore_chip(
                bedfile,
                np.ones(snp_on_disk.shape[1], dtype="bool"),
                unrel_sample_indices,
            )
            ldscores = ldscores.rename(columns={"LDSCORE_CHIP": "LDSCORE"})
        else:
            ldscores = pd.read_csv(ldscores, sep="\s+")

        partial_calibrate_test_stats = partial(
            calibrate_test_stats, ldscores, bedfile, unrel_sample_indices, out
        )
        correction = Parallel(n_jobs=min(8, n_workers))(
            delayed(partial_calibrate_test_stats)(i) for i in pheno_columns[2:]
        )
        np.savetxt(out + ".calibration", correction)
        return correction
    else:
        return None


def get_test_statistics_bgen(
    bgenfile,
    samplefile,
    bedfile,
    phenofile,
    offset,
    sample_indices,
    covar,
    calibrationFile,
    extractFile,
    out="out",
    binary=False,
    n_workers=-1,
):
    if n_workers == -1:
        n_workers = os.cpu_count() - 1
    if calibrationFile is not None:
        calibration_factors = np.loadtxt(calibrationFile)
        print(calibration_factors)

    snp_on_disk = Bed(bedfile, count_A1=True)
    unique_chrs = np.unique(np.array(snp_on_disk.pos[:, 0], dtype=int))
    offsetFileList = [offset + str(chr) + ".offsets" for chr in unique_chrs]
    traits = pd.read_csv(phenofile, sep="\s+")
    pheno_columns = traits.columns.tolist()

    Parallel(n_jobs=n_workers)(
        delayed(preprocess_offsets)(
            bedfile,
            offset + str(C) + ".offsets",
            sample_indices,
            pheno_columns,
            samplefile,
        )
        for C in unique_chrs
    )
    preprocess_offsets(
        bedfile,
        phenofile,
        sample_indices,
        pheno_columns,
        samplefile,
        adj_suffix=".preprocessed",
    )
    get_unadjusted_test_statistics_bgen(
        bgenfile,
        samplefile,
        [phenofile + ".preprocessed"] * len(unique_chrs),
        offsetFileList,
        covar,
        out + "_bgen",
        unique_chrs,
        extractFile,
        num_threads=n_workers,
        binary=binary,
    )

    if calibrationFile is not None:
        if calibration_factors.shape == ():
            calibration_factors = [calibration_factors]
        Parallel(n_jobs=n_workers)(
            delayed(adjust_test_stats)(out + "_bgen", pheno, correction)
            for (pheno, correction) in zip(pheno_columns[2:], calibration_factors)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", "-g", help="prefix for bed/bim/fam files", type=str)
    parser.add_argument("--output_step1", help="Filename of the offsets file", type=str)
    parser.add_argument(
        "--covar",
        "-c",
        help='file with covariates; should be in "FID,IID,Var1,Var2,..." format and tsv',
        type=str,
    )
    parser.add_argument(
        "--unrel_sample_list",
        help="File with un-related homogenous sample list (FID, IID)",
        type=str,
    )
    parser.add_argument(
        "--calibrate",
        help="Do you wish to calibrate your test-statistics ?",
        type=str_to_bool,
        default="True",
    )
    parser.add_argument(
        "--calibrationFile",
        help="Do you wish to calibrate your test-statistics ?",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ldscores",
        help="Path to ld score directory in format accepted by ldsc",
        type=str,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="prefix for where to save any results or files",
        default="out",
    )
    parser.add_argument("--hdf5", type=str, help="File name of the hdf5 file to use")

    parser.add_argument("--bgen", help="Location to Bgen file", type=str)
    parser.add_argument("--sample", help="Location to samples file", type=str)
    parser.add_argument(
        "--extract",
        help="Path to list of SNPs to be considered in test stats calculation for bgen file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--binary",
        help="Is the phenotype binary ?",
        action="store_const",
        const=True,
        default=False,
    )

    args = parser.parse_args()
    offsets_file = args.output_step1 + "loco_chr"

    traits = args.output_step1 + ".traits"
    covareffects = args.output_step1 + ".covar_effects"
    sample_indices = np.array(h5py.File(args.hdf5, "r")["sample_indices"])

    ######      Calculating test statistics       ######
    st = time.time()
    print("Calculating test statistics..")
    if args.bgen is None:
        get_test_statistics(
            args.bed,
            traits,
            covareffects,
            offsets_file,
            sample_indices,
            args.unrel_sample_list,
            args.ldscores,
            args.covar,
            args.calibrate,
            args.output,
            args.binary,
        )
    else:
        get_test_statistics_bgen(
            args.bgen,
            args.sample,
            args.bed,
            traits,
            offsets_file,
            sample_indices,
            args.covar,
            args.calibrationFile,
            args.extract,
            args.output,
            args.binary,
        )
    print("Done in " + str(time.time() - st) + " secs")
