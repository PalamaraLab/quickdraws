"""
Code for getting the test-statistics from LOCO residuals
Author: Yiorgos + Hrushi
"""
import os
import numpy as np
from pysnptools.snpreader import Bed
from pysnptools.distreader import Bgen
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
import warnings
import logging
from datetime import datetime
from art import text2art

from ldscore_calibration import calc_ldscore_chip, ldscore_intercept, get_mask_dodgy
from custom_linear_regression import (
    get_unadjusted_test_statistics,
    get_unadjusted_test_statistics_bgen,
    preprocess_covars
)
from preprocess_phenotypes import preprocess_phenotypes, PreparePhenoRHE
import pdb

def str_to_bool(s: str) -> bool:
    return bool(strtobool(s))

def preprocess_offsets(offsets, sample_file):
    df_concat = pd.read_csv(offsets, sep="\s+")
    pheno_columns = df_concat.columns.tolist().copy()
    if sample_file is not None:
        sample_file = pd.read_csv(sample_file, sep="\s+")
        sample_file = sample_file.rename(columns={"ID_1": "FID", "ID_2": "IID"})
        sample_file[["FID", "IID"]] = sample_file[["FID", "IID"]].astype("int")
        df_concat = pd.merge(df_concat, sample_file, on=["FID", "IID"])
        df_concat = df_concat[pheno_columns]
    return df_concat

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

def rege_to_qd_format():
    df = pd.read_csv('regenie_unrel_50000_0.5_inter_2.loco', sep=' ')
    df_transposed = df.transpose()
    df_transposed.columns = df_transposed.iloc[0]
    df_transposed = df_transposed[1:].reset_index()
    df_transposed[['FID', 'IID']] = df_transposed['index'].str.split('_', expand=True)
    df_transposed = df_transposed.drop(columns=['index'])
    df_transposed.columns.name = ''
    df_transposed = df_transposed.loc[0:len(df_transposed)-2]
    df_transposed[['FID', 'IID']] = df_transposed[['FID','IID']].astype('int')
    return df_transposed

def calibrate_test_stats(
    ldscores, bedfile, unrel_sample_list, out, match_yinter, pheno
):
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
    ## we do this two times because the sumstats_cur could be way off from the sumstats_ref
    overall_correction = 1
    prev_correction = 1
    for ldsc_iter in range(5):
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
        if match_yinter and ldsc_iter < 2:
            correction = intercept_ref / intercept_cur
        
        if ldsc_iter == 4 and np.abs(prev_correction - correction) > 0.01:
            logging.info(pheno + " entering here...")
            logging.info(prev_correction - correction)
            logging.info(sumstats_cur)
            sumstats_cur["CHISQ"] /= overall_correction
            sumstats_cur["P"] = chi2.sf(sumstats_cur.CHISQ, df=1)
            overall_correction = 1  
            logging.info(sumstats_cur)
        else:
            overall_correction *= correction
            sumstats_cur["CHISQ"] *= correction
            sumstats_cur["P"] = chi2.sf(sumstats_cur.CHISQ, df=1)

        prev_correction = correction.copy()

    sumstats_cur = sumstats_cur.drop('CHISQ_FLT', axis=1)
    sumstats_cur["CHISQ"] = sumstats_cur["CHISQ"].map(lambda x: "{:.8f}".format(x))
    sumstats_cur["P"] = sumstats_cur["P"].map(lambda x: "{:.2E}".format(x))
    sumstats_cur.to_csv(
        "{0}.{1}.sumstats".format(out, pheno),
        sep="\t",
        index=None,
    )

    os.system("gzip -f " + "{0}.{1}.sumstats".format(out, pheno))
    return overall_correction


def check_case_control_sep(
    traits,
    covar, 
    offset,
    unique_chrs
):
    pheno_columns = traits.columns.tolist()
    pheno_mask = np.zeros(len(pheno_columns[2:]))
    W = preprocess_covars(covar, traits[['FID','IID']])
    for chr in unique_chrs:
        offset_df = pd.read_csv(offset + str(chr) + ".offsets", sep="\s+")
        for p, p_name in enumerate(pheno_columns[2:]):
            if binary:
                try:
                    offset_p = offset_df[p_name].values         
                    np.linalg.inv((W.T * offset_p * (1 - offset_p))@W)
                    pheno_mask[p] += 1
                except:
                    continue
            else:
                pheno_mask[p] += 1
    
    pheno_mask = pheno_mask == len(unique_chrs)
    pheno_columns = ['FID','IID'] + (np.array(pheno_columns[2:])[pheno_mask].tolist())
    if not pheno_mask.all():
        logging.info("Removed traits with complete case-control seperation, updated pheno list = " + str(pheno_columns))
    return pheno_columns

def get_test_statistics(
    bedfile,
    phenofile,
    covareffectsfile,
    offset,
    unrel_homo_file,
    ldscores,
    covar,
    calibrate,
    out="out",
    binary=False,
    firth=False,
    firth_pval_thresh=0.05,
    n_workers=-1,
):
    if n_workers == -1:
        n_workers = len(os.sched_getaffinity(0))

    snp_on_disk = Bed(bedfile, count_A1=True)
    unique_chrs = np.unique(np.array(snp_on_disk.pos[:, 0], dtype=int))
    
    traits = pd.read_csv(phenofile, sep="\s+")
    covar_effects = pd.read_csv(covareffectsfile, sep="\s+")
    offset_list = []
    for chr in unique_chrs:
        offset_df = pd.read_csv(offset + str(chr) + ".offsets", sep="\s+")
        offset_list.append(offset_df)
    pheno_columns = traits.columns.tolist()

    if binary:
        pheno_columns = check_case_control_sep(traits, covar, offset, unique_chr)
        traits = traits[pheno_columns]
        covar_effects = covar_effects[pheno_columns]
        for offset_df in offset_list:
            offset_df = offset_df[pheno_columns]

    if calibrate:
        logging.info("Preprocessing traits for unrelated homogenous samples...")
        unrel_sample_traits, unrel_sample_covareffect, unrel_sample_indices = preprocess_phenotypes(phenofile, covar, bedfile, unrel_homo_file, binary)
        unrel_sample_indices = unrel_sample_indices.tolist()
        if binary:
            unrel_sample_covareffect[unrel_sample_covareffect.columns[2:]] = expit(
                unrel_sample_covareffect[unrel_sample_covareffect.columns[2:]].values
            )

        logging.info("Running linear/logistic regression on unrelated individuals...")
        get_unadjusted_test_statistics(
            bedfile,
            unrel_sample_traits,
            unrel_sample_covareffect if binary else None,
            [unrel_sample_covareffect] * len(unique_chrs) if binary else None,
            covar,
            out + "_lrunrel",
            unique_chrs,
            num_threads=n_workers,
            binary=binary, 
            firth=False,
            firth_pval_thresh=firth_pval_thresh,
        )

    logging.info("Running linear/logistic regression...")
    get_unadjusted_test_statistics(
        bedfile,
        traits,
        covar_effects,
        offset_list,
        covar,
        out,
        unique_chrs,
        num_threads=n_workers,
        binary=binary,
        firth=firth,
        firth_pval_thresh=firth_pval_thresh,
    )

    if calibrate:
        if ldscores is None:
            logging.warning(
                "No LD scores provided, using LD score chip.. this might lead to error-prone power"
            )
            ldscores = calc_ldscore_chip(
                bedfile,
                np.ones(snp_on_disk.shape[1], dtype="bool"),
                unrel_sample_indices,
            )
            ldscores = ldscores.rename(columns={"LDSCORE_CHIP": "LDSCORE"})
        else:
            ldscores = pd.read_csv(ldscores, sep="\s+")

        logging.info("Calculating the calibration factors to correct the sumstats")
        partial_calibrate_test_stats = partial(
            calibrate_test_stats, ldscores, bedfile, unrel_sample_indices, out, binary
        )
        correction = Parallel(n_jobs=min(8, n_workers))(
            delayed(partial_calibrate_test_stats)(i) for i in pheno_columns[2:]
        )
        logging.info("Caliration factors stored in: " +str(out) + ".calibration")
        np.savetxt(out + ".calibration", correction)
    else:
        correction = None

    logging.info("Summary stats stored as: " + str(out) + ".{pheno}.sumstats.gz")
    return correction


def get_test_statistics_bgen(
    bgenfile,
    samplefile,
    phenofile,
    covareffectsfile,
    offset,
    firthnullfile,
    covar,
    calibrationFile,
    extractFile,
    out="out",
    binary=False,
    firth=False,
    firth_pval_thresh=0.05,
    n_workers=-1,
):
    if n_workers == -1:
        n_workers = len(os.sched_getaffinity(0))

    snp_on_disk = Bgen(bgenfile, sample=samplefile)
    unique_chrs = np.unique(np.array(snp_on_disk.pos[:, 0], dtype=int))

    traits = preprocess_offsets(phenofile, samplefile)
    covar_effects = preprocess_offsets(covareffectsfile, samplefile)
    offset_list = Parallel(n_jobs=n_workers)(
        delayed(preprocess_offsets)(offset + str(C) + ".offsets", samplefile)
        for C in unique_chrs
    )
    if firth:
        firth_null_list = Parallel(n_jobs=n_workers)(
            delayed(preprocess_offsets)(firthnullfile + str(C) + ".firth_null", samplefile)
            for C in unique_chrs
        )
    else:
        firth_null_list = []
    pheno_columns = traits.columns.tolist()

    if binary:
        pheno_columns = check_case_control_sep(traits, covar, offset, unique_chr)
        traits = traits[pheno_columns]
        covar_effects = covar_effects[pheno_columns]
        for offset_df in offset_list:
            offset_df = offset_df[pheno_columns]
        if firth:
            for firth_null_df in firth_null_list:
                firth_null_df = firth_null_df[pheno_columns]

    if calibrationFile is not None:
        calibration_factors = np.loadtxt(calibrationFile)
        logging.info("Using calibration file specified in: " + str(calibrationFile))
    else:
        calibration_factors = np.ones(len(traits.columns.tolist()) - 2)

    
    logging.info("Running linear/logistic regression...")
    get_unadjusted_test_statistics_bgen(
        bgenfile,
        samplefile,
        traits,
        covar_effects,
        offset_list,
        covar,
        out,
        unique_chrs,
        extractFile,
        num_threads=n_workers,
        binary=binary,
        firth=firth,
        firth_pval_thresh=firth_pval_thresh,
        firth_null=firth_null_list,
    )

    if calibration_factors.shape == ():
        calibration_factors = [calibration_factors]
    Parallel(n_jobs=n_workers)(
        delayed(adjust_test_stats)(out, pheno, correction)
        for (pheno, correction) in zip(pheno_columns[2:], calibration_factors)
    )
    logging.info("Summary stats stored as: " + str(out) + ".{pheno}.sumstats.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", "-g", help="prefix for bed/bim/fam files", type=str)
    parser.add_argument("--out_step1", help="Filename of the offsets file", type=str)
    parser.add_argument(
        "--covarFile",
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
        "--out",
        "-o",
        help="prefix for where to save any results or files",
        default="out",
    )
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
    parser.add_argument(
        "--firth",
        help="Approximate firth logistic regression ?",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--firth_pval_thresh",
        help="P-value threshold below which firth logistic regression done",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    ######      Logging setup                    #######
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.out + ".log"),
            logging.StreamHandler()
        ]
    )
    logging.info(text2art("Quickdraws"))
    logging.info("Copyright (c) 2024 Hrushikesh Loya and Pier Palamara.")
    logging.info("Distributed under the GPLv3 License.")
    logging.info("")
    logging.info("Logs saved in: " + str(args.out + ".step2.log"))
    logging.info("")

    logging.info("Options in effect: ")
    for arg in vars(args):
        logging.info('     {}: {}'.format(arg, getattr(args, arg)))

    logging.info("")

    st = time.time()
    logging.info("#### Start Time: " + str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')) + " ####")
    logging.info("")

    offsets_file = args.out_step1 + "loco_chr"
    traits = args.out_step1 + ".traits"
    covareffects = args.out_step1 + ".covar_effects"
    firth_null_file = args.out_step1

    assert (args.calibrate and args.unrel_sample_list is not None) or (not args.calibrate)
    "Provide a list of unrelated homogenous sample if you wish to calibrate"

    ######      Calculating test statistics       ######
    st = time.time()
    logging.info("#### Step 2. Calculating test statistics ####")
    warnings.simplefilter("ignore")
    if args.bgen is None:
        get_test_statistics(
            args.bed,
            traits,
            covareffects,
            offsets_file,
            args.unrel_sample_list,
            args.ldscores,
            args.covarFile,
            args.calibrate,
            args.out,
            args.binary,
            args.firth,
            args.firth_pval_thresh,
        )
    else:
        get_test_statistics_bgen(
            args.bgen,
            args.sample,
            traits,
            covareffects,
            offsets_file,
            firth_null_file,
            args.covarFile,
            args.calibrationFile,
            args.extract,
            args.out,
            args.binary,
            args.firth,
            args.firth_pval_thresh,
        )
    logging.info("#### Step 2. Done in: " + str(time.time() - st) + " secs ####")
    logging.info("")
    logging.info("#### End Time: " + str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')) + " ####")
    logging.info("")
