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

from ldscore_calibration import calc_ldscore_chip, ldscore_intercept, get_mask_dodgy
from custom_linear_regression import (
    get_unadjusted_test_statistics,
    get_unadjusted_test_statistics_bgen,
)
from preprocess_phenotypes import preprocess_phenotypes, PreparePhenoRHE


def str_to_bool(s: str) -> bool:
    return bool(strtobool(s))


def preprocess_residuals(
    bedfile, residuals, sample_indices, pheno_columns, sample_file=None
):
    snp_on_disk = Bed(bedfile, count_A1=True)
    iid_fid = snp_on_disk.iid[sample_indices]
    df_iid_fid = pd.DataFrame(np.array(iid_fid, dtype=int), columns=["FID", "IID"])
    df_residuals = pd.read_csv(residuals, sep="\s+")
    if "IID" not in df_residuals.columns:
        df_concat = pd.concat([df_iid_fid, df_residuals], axis=1)
    else:
        df_concat = df_residuals
    df_concat.columns = pheno_columns
    if sample_file is not None:
        sample_file = pd.read_csv(sample_file, sep="\s+")
        sample_file = sample_file.rename(columns={"ID_1": "FID", "ID_2": "IID"})
        sample_file[["FID", "IID"]] = sample_file[["FID", "IID"]].astype("int")
        df_concat = pd.merge(df_concat, sample_file, on=["FID", "IID"])
        df_concat = df_concat[pheno_columns]

    print(residuals + " : " + str(df_concat.shape))
    df_concat.to_csv(residuals, sep="\t", index=None)


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
    adjusted_traits_file,
    residual,
    sample_indices,
    unrel_homo_file,
    ldscores,
    covar,
    calibrate,
    out="out",
    n_workers=-1,
):
    if n_workers == -1:
        n_workers = os.cpu_count() - 1

    snp_on_disk = Bed(bedfile, count_A1=True)
    unique_chrs = np.unique(np.array(snp_on_disk.pos[:, 0], dtype=int))
    adjusted_traits = pd.read_csv(adjusted_traits_file, sep="\s+")
    pheno_columns = adjusted_traits.columns.tolist()
    print(pheno_columns)

    Parallel(n_jobs=n_workers)(
        delayed(preprocess_residuals)(
            bedfile, residual + str(C) + ".residuals", sample_indices, pheno_columns
        )
        for C in unique_chrs
    )

    if calibrate:
        # Run LR-unRel using our numba implementation
        unrel_homo = pd.read_csv(unrel_homo_file, names=["FID", "IID"], sep="\s+")
        unrel_sample_pheno = pd.merge(adjusted_traits, unrel_homo, on=["FID", "IID"])
        unrel_sample_pheno.to_csv(out + "adjusted.unrel", sep="\t", index=None)
        samples_dict = {}
        for i, fid in enumerate(snp_on_disk.iid[:, 0]):
            samples_dict[int(fid)] = i
        unrel_sample_indices = []
        for fid in unrel_sample_pheno.FID:
            unrel_sample_indices.append(samples_dict[int(fid)])

        get_unadjusted_test_statistics(
            bedfile,
            [out + "adjusted.unrel"] * len(unique_chrs),
            covar,
            out + "_lrunrel",
            unique_chrs,
            num_threads=n_workers,
        )
    residualFileList = [residual + str(chr) + ".residuals" for chr in unique_chrs]
    get_unadjusted_test_statistics(
        bedfile,
        residualFileList,
        covar,
        out,
        unique_chrs,
        num_threads=n_workers,
    )
    if calibrate:
        pheno_list = pheno_columns[2:]
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
            delayed(partial_calibrate_test_stats)(i) for i in pheno_list
        )
        np.savetxt(out + ".calibration", correction)
        return correction
    else:
        return None


def get_test_statistics_bgen(
    bgenfile,
    samplefile,
    bedfile,
    adjusted_traits_file,
    residual,
    sample_indices,
    covar,
    calibrationFile,
    extractFile,
    out="out",
    n_workers=-1,
):
    if n_workers == -1:
        n_workers = os.cpu_count() - 1
    if calibrationFile is not None:
        calibration_factors = np.loadtxt(calibrationFile)
        print(calibration_factors)

    snp_on_disk = Bed(bedfile, count_A1=True)
    unique_chrs = np.unique(np.array(snp_on_disk.pos[:, 0], dtype=int))
    pheno_columns = pd.read_csv(adjusted_traits_file, sep="\s+").columns.tolist()
    residualFileList = [residual + str(chr) + ".residuals" for chr in unique_chrs]

    Parallel(n_jobs=n_workers)(
        delayed(preprocess_residuals)(
            bedfile,
            residual + str(C) + ".residuals",
            sample_indices,
            pheno_columns,
            samplefile,
        )
        for C in unique_chrs
    )
    get_unadjusted_test_statistics_bgen(
        bgenfile,
        samplefile,
        residualFileList,
        covar,
        out + "_bgen",
        unique_chrs,
        extractFile,
        num_threads=n_workers,
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
    parser.add_argument(
        "--output_step1", help="Filename of the residuals file", type=str
    )
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

    args = parser.parse_args()
    residuals_file = args.output_step1 + "loco_chr"
    adjusted_traits_file = args.output_step1 + ".adjusted_traits.phen"
    sample_indices = np.array(h5py.File(args.hdf5, "r")["sample_indices"])

    ######      Calculating test statistics       ######
    st = time.time()
    print("Calculating test statistics..")
    if args.bgen is None:
        get_test_statistics(
            args.bed,
            adjusted_traits_file,
            residuals_file,
            sample_indices,
            args.unrel_sample_list,
            args.ldscores,
            args.covar,
            args.calibrate,
            args.output,
        )
    else:
        get_test_statistics_bgen(
            args.bgen,
            args.sample,
            args.bed,
            adjusted_traits_file,
            residuals_file,
            sample_indices,
            args.covar,
            args.calibrationFile,
            args.extract,
            args.output,
        )
    print("Done in " + str(time.time() - st) + " secs")
