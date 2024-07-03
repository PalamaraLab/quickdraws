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
import copy

from quickdraws import (
    get_unadjusted_test_statistics_bgen,
    get_unadjusted_test_statistics,
    preprocess_covars,
    preprocess_phenotypes,
)

from quickdraws.scripts import get_copyright_string


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

def str_to_bool(s: str) -> bool:
    return bool(strtobool(s))

def preprocess_offsets(offsets, sample_file=None, weights=None, is_loco_file=False):
    if is_loco_file:
        df_concat = offsets
    else:
        df_concat = pd.read_csv(offsets, sep=r'\s+')
    pheno_columns = df_concat.columns.tolist().copy()
    if sample_file is not None:
        sample_file = pd.read_csv(sample_file, sep=r'\s+')
        sample_file = sample_file.rename(columns={"ID_1": "FID", "ID_2": "IID"})
        sample_file[["FID", "IID"]] = sample_file[["FID", "IID"]].astype("int")
        df_concat = pd.merge(df_concat, sample_file, on=["FID", "IID"])
        df_concat = df_concat[pheno_columns]
    if weights is not None:
        weights_file = pd.read_csv(weights, sep=r'\s+')
        weights_file = weights_file.rename(columns={"ID_1": "FID", "ID_2": "IID"})
        weights_file[["FID", "IID"]] = weights_file[["FID", "IID"]].astype("int")
        df_concat = pd.merge(df_concat, weights_file, on=["FID", "IID"])
        df_concat = df_concat[pheno_columns]
    
    if is_loco_file:
        logging.info("Processing loco files, Shape: " + str(df_concat.shape))
    else:
        logging.info("Processing " + str(offsets) + " Shape: " + str(df_concat.shape))

    return df_concat

def adjust_test_stats(out, pheno, correction):
    sumstats_cur = pd.read_hdf("{0}.{1}.sumstats".format(out, pheno), key="sumstats")
    if 'Firth' in sumstats_cur.columns:
        for chr in np.unique(sumstats_cur['CHR']):
            for is_firth in [True, False]:
                sumstats_cur.loc[(sumstats_cur.CHR == chr) & (sumstats_cur.Firth == is_firth), "CHISQ"] *= float(correction[str((chr, is_firth))])
    else:
        for chr in np.unique(sumstats_cur['CHR']):
            sumstats_cur.loc[sumstats_cur.CHR == chr, "CHISQ"] *= float(correction[str(chr)])

    sumstats_cur["P"] = chi2.sf(sumstats_cur.CHISQ, df=1)
    sumstats_cur["CHISQ"] = sumstats_cur["CHISQ"].map(lambda x: "{:.8f}".format(x))
    sumstats_cur["P"] = sumstats_cur["P"].map(lambda x: "{:.2E}".format(x))

    sumstats_cur.to_csv(
        "{0}.{1}.sumstats".format(out, pheno),
        sep="\t",
        index=None,
        na_rep='NA'
    )

    os.system("gzip -f " + "{0}.{1}.sumstats".format(out, pheno))

def load_offsets(offset, pheno_columns, unique_chrs, covar_effect):
    ## assert the first row of all .loco files in same
    loco_files = []
    for dim_out in range(len(pheno_columns[2:])):
        df = rege_to_qd_format(offset + '_' + str(dim_out + 1) + '.loco')
        loco_files.append(df)
    for loco_file in loco_files:
        if not (loco_file[['FID','IID']] == covar_effect[['FID','IID']]).all().all():
            logging.exception('LOCO files have different ordering of individuals')
            raise ValueError
    offset_list = []
    for chr_no, chr in enumerate(unique_chrs):
        offset = pd.DataFrame(columns = pheno_columns)
        offset[['FID','IID']] = loco_files[0][['FID','IID']]
        for dim_out in range(len(pheno_columns[2:])):
            offset[pheno_columns[2:][dim_out]] = loco_files[dim_out][chr] + covar_effect[pheno_columns[2+dim_out]]
        offset_list.append(offset)
    return offset_list

def rege_to_qd_format(filename):
    df = pd.read_csv(filename, sep=' ', low_memory=False)
    df_transposed = df.transpose()
    df_transposed.columns = df_transposed.iloc[0]
    df_transposed = df_transposed[1:].reset_index()
    df_transposed[['FID', 'IID']] = df_transposed['index'].str.split('_', expand=True)
    df_transposed = df_transposed.drop(columns=['index'])
    df_transposed.columns.name = ''
    df_transposed[['FID', 'IID']] = df_transposed[['FID','IID']].astype('int')
    return df_transposed[['FID','IID'] + df_transposed.columns[:-2].tolist()]

def calibrate_test_stats(
    out, pheno, neff
):
    sumstats_ref = pd.read_hdf(
        "{0}.{1}.sumstats".format(out + "_lrunrel", pheno), key="sumstats"
    )
    sumstats_cur = pd.read_hdf("{0}.{1}.sumstats".format(out, pheno), key="sumstats")
    sumstats_cur_prev = copy.deepcopy(sumstats_cur)
    assert (sumstats_ref[['CHR','SNP','POS']].values == sumstats_cur[['CHR','SNP','POS']].values).all()

    overall_correction_dict = {}
    if 'Firth' in sumstats_cur.columns:
        for chr in np.unique(sumstats_ref['CHR']):
            ## Order True, False is important
            for is_firth in [True, False]:
                ess = sumstats_cur.loc[(sumstats_ref.CHR == chr) & (sumstats_cur.Firth == is_firth),"OBS_CT"].values*float(neff[str(chr)])/sumstats_ref.loc[(sumstats_ref.CHR == chr)  & (sumstats_cur.Firth == is_firth),"OBS_CT"].values
                overall_correction = (ess * (np.mean(sumstats_ref[(sumstats_ref.ALT_FREQS > 0.01) & (sumstats_ref.ALT_FREQS < 0.99) & (sumstats_ref.CHR == chr) & (sumstats_cur.Firth == is_firth)].CHISQ) - 1) + 1)/np.mean(sumstats_cur[(sumstats_ref.ALT_FREQS > 0.01) & (sumstats_ref.ALT_FREQS < 0.99) & (sumstats_ref.CHR == chr) & (sumstats_cur.Firth == is_firth)].CHISQ) 
                if is_firth:
                    overall_correction_all = (ess * (np.mean(sumstats_ref[(sumstats_ref.ALT_FREQS > 0.01) & (sumstats_ref.ALT_FREQS < 0.99) & (sumstats_ref.CHR == chr)].CHISQ) - 1) + 1)/np.mean(sumstats_cur[(sumstats_ref.ALT_FREQS > 0.01) & (sumstats_ref.ALT_FREQS < 0.99) & (sumstats_ref.CHR == chr)].CHISQ)
                    if np.mean(overall_correction) > np.mean(overall_correction_all):
                        ## Being conservative: sometimes Firth only correction has high variance and leads to higher overall_correction, we instead chose a lower variance estimator utilizing all common SNPs in the chromosome
                        overall_correction = overall_correction_all.copy()
                if pheno == 'fake10':
                    print(str(np.mean(overall_correction)) + " " + str(chr) + " " + str(is_firth) + " " + str(np.mean(ess)) + " " + str(len(ess)))
                overall_correction_dict[(chr, is_firth)] = np.mean(overall_correction)
                sumstats_cur.loc[(sumstats_ref.CHR == chr) & (sumstats_cur.Firth == is_firth), "CHISQ"] *= overall_correction

    else:
        for chr in np.unique(sumstats_ref['CHR']):
            ess = sumstats_cur.loc[sumstats_ref.CHR == chr,"OBS_CT"].values*float(neff[str(chr)])/sumstats_ref.loc[sumstats_ref.CHR == chr,"OBS_CT"].values
            overall_correction = (ess * (np.mean(sumstats_ref[(sumstats_ref.ALT_FREQS > 0.01) & (sumstats_ref.ALT_FREQS < 0.99) & (sumstats_ref.CHR == chr)].CHISQ) - 1) + 1)/np.mean(sumstats_cur[(sumstats_ref.ALT_FREQS > 0.01) & (sumstats_ref.ALT_FREQS < 0.99) & (sumstats_ref.CHR == chr)].CHISQ)
            overall_correction_dict[chr] = np.mean(overall_correction)
            sumstats_cur.loc[sumstats_ref.CHR == chr, "CHISQ"] *= overall_correction

    # ess = sumstats_cur["OBS_CT"].values*float(neff)/sumstats_ref["OBS_CT"].values
    # overall_correction = (ess * (np.mean(sumstats_ref[(sumstats_ref.ALT_FREQS > 0.01) & (sumstats_ref.ALT_FREQS < 0.99)].CHISQ) - 1) + 1)/np.mean(sumstats_cur[(sumstats_ref.ALT_FREQS > 0.01) & (sumstats_ref.ALT_FREQS < 0.99)].CHISQ)
    # sumstats_cur["CHISQ"] *= overall_correction
    
    sumstats_cur["P"] = chi2.sf(sumstats_cur.CHISQ, df=1)
    sumstats_cur["CHISQ"] = sumstats_cur["CHISQ"].map(lambda x: "{:.8f}".format(x))
    sumstats_cur["P"] = sumstats_cur["P"].map(lambda x: "{:.2E}".format(x))
    sumstats_cur.to_csv(
        "{0}.{1}.sumstats".format(out, pheno),
        sep="\t",
        index=None,
        na_rep='NA'
    )

    os.system("gzip -f " + "{0}.{1}.sumstats".format(out, pheno))
    
    ### caution: mean across all SNPs make sense? imputed data doesnt have missingness
    return overall_correction_dict

def check_case_control_sep(
    traits,
    covar, 
    offset_list,
    unique_chrs
):
    pheno_columns = traits.columns.tolist()
    pheno_mask = np.zeros(len(pheno_columns[2:]))
    W = preprocess_covars(covar, traits[['FID','IID']])
    for chr_no, chr in enumerate(unique_chrs):
        offset_df = offset_list[chr_no]
        for p, p_name in enumerate(pheno_columns[2:]):
            try:
                offset_p = expit(offset_df[p_name].values)      
                np.linalg.inv((W.T * offset_p * (1 - offset_p))@W)
                pheno_mask[p] += 1
            except:
                continue
    
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
    covar,
    calibrate,
    out="out",
    binary=False,
    firth=False,
    firth_pval_thresh=0.05,
    firth_maf_thresh=0.05,
    firth_prevalence_thresh=0.05,
    n_workers=-1,
    weights=None
):
    if n_workers == -1:
        n_workers = len(os.sched_getaffinity(0))

    if weights is not None and binary:
        logging.exception("Weighted linear regression only supported for quantitive traits..")

    snp_on_disk = Bed(bedfile, count_A1=True)
    unique_chrs = np.unique(np.array(snp_on_disk.pos[:, 0], dtype=int))

    traits = preprocess_offsets(phenofile, weights)
    pheno_columns = traits.columns.tolist()
    covar_effects = pd.read_csv(covareffectsfile, sep=r'\s+')
    # neff = np.loadtxt(offset + '.step1.neff')
    neff = pd.read_csv(offset + '.neff', sep=r'\s+')
    logging.info("Using estimated effective sample fize from file specified in: " + str(offset) + '.neff')
    
    if weights is None:
        offset_list_pre = load_offsets(offset, pheno_columns, unique_chrs, covar_effects)
    else:
        ### Dont substract covariate effect with weighted linear regression,
        ### It is instead done in MyWightedLinRegr function
        covar_effects_copy = copy.deepcopy(covar_effects)
        covar_effects_copy[covar_effects_copy.columns[2:]] = 0
        offset_list_pre = load_offsets(offset, pheno_columns, unique_chrs, covar_effects_copy)
    
    covar_effects = preprocess_offsets(covareffectsfile, weights)
    offset_list = Parallel(n_jobs=n_workers)(
        delayed(preprocess_offsets)(offset_list_pre[chr_no], weights, None, True)
        for chr_no in range(len(unique_chrs))
    )
    if weights is not None:
        weights_df = pd.read_csv(weights, '\t')
        weights = pd.merge(traits[['FID','IID']], weights_df, on=['FID','IID'])

    if binary:
        pheno_columns = check_case_control_sep(traits, covar, offset_list, unique_chrs)
        traits = traits[pheno_columns]
        covar_effects = covar_effects[pheno_columns]
        for i in range(len(offset_list)):
            offset_list[i] = offset_list[i][pheno_columns]

    if calibrate:
        logging.info("Preprocessing traits for unrelated homogenous samples...")
        unrel_sample_traits, unrel_sample_covareffect, unrel_sample_indices = preprocess_phenotypes(phenofile, covar, bedfile, unrel_homo_file, binary, log=False)
        unrel_sample_traits = unrel_sample_traits[pheno_columns]
        unrel_sample_covareffect = unrel_sample_covareffect[['FID','IID'] + ["covar_effect_" + str(col) for col in pheno_columns[2:]]]
        unrel_sample_indices = unrel_sample_indices.tolist()

        if weights is not None:
            unrel_sample_traits = pd.merge(unrel_sample_traits, weights[['FID','IID']], on=['FID','IID'])
            unrel_sample_covareffect = pd.merge(unrel_sample_covareffect, weights[['FID','IID']], on=['FID','IID'])
            weights_unrel = pd.merge(unrel_sample_traits[['FID','IID']], weights, on=['FID','IID'])
        else:
            weights_unrel = None

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
            weights=weights_unrel
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
        firth_maf_thresh=firth_maf_thresh,
        firth_prevalence_thresh=firth_prevalence_thresh,
        weights=weights
    )
    if binary and firth:
        logging.info("Firth logistic regression null model estimates saved as: " + str(out) + "{chr}.firth_null")

    if calibrate:
        logging.info("Calculating the calibration factors to correct the sumstats")
        partial_calibrate_test_stats = partial(
            calibrate_test_stats, out
        )
        correction = Parallel(n_jobs=min(8, n_workers))(
            delayed(partial_calibrate_test_stats)(phen, neff.iloc[i]) for i, phen in enumerate(pheno_columns[2:])
        )
        logging.info("Caliration factors stored in: " +str(out) + ".calibration")
        df = pd.DataFrame(correction)
        df['pheno'] = pheno_columns[2:]
        df.to_csv(out + '.calibration', sep='\t', index=None)
        # np.savetxt(out + ".calibration", correction)
    else:
        correction = None

    logging.info("Summary stats stored as: " + str(out) + ".{pheno}.sumstats{.gz}")
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
    firth_maf_thresh=0.05,
    firth_prevalence_thresh=0.05,
    n_workers=-1,
    weights=None
):
    if n_workers == -1:
        n_workers = len(os.sched_getaffinity(0))

    if weights is not None and binary:
        logging.exception("Weighted linear regression only supported for quantitive traits..")

    snp_on_disk = Bgen(bgenfile, sample=samplefile)
    unique_chrs = np.unique(np.array(snp_on_disk.pos[:, 0], dtype=int))
    traits = preprocess_offsets(phenofile, samplefile, weights)
    pheno_columns = traits.columns.tolist()

    covar_effects = pd.read_csv(covareffectsfile, sep=r'\s+')
    offset_list_pre = load_offsets(offset, pheno_columns, unique_chrs, covar_effects) 
    covar_effects = preprocess_offsets(covareffectsfile, samplefile, weights)

    offset_list = Parallel(n_jobs=n_workers)(
        delayed(preprocess_offsets)(offset_list_pre[chr_no], samplefile, weights, True)
        for chr_no in range(len(unique_chrs))
    )
    if firth and binary:
        firth_null_list = Parallel(n_jobs=n_workers)(
            delayed(preprocess_offsets)(firthnullfile + str(C) + ".firth_null", samplefile, weights)
            for C in unique_chrs
        )
    else:
        firth_null_list = []

    if weights is not None:
        weights_df = pd.read_csv(weights, '\t')
        mdf_weights_traits = pd.merge(traits, weights_df, on=['FID','IID'])
        weights = np.array(mdf_weights_traits[weights_df.columns[2]].values, dtype='float32')

    if binary:
        pheno_columns = check_case_control_sep(traits, covar, offset_list, unique_chrs)
        pheno_columns = ['FID','IID'] + np.intersect1d(pheno_columns[2:], firth_null_list[0].columns.tolist()).tolist()  ## this sorts the phenotypes...
        print(pheno_columns)
        traits = traits[pheno_columns]
        covar_effects = covar_effects[pheno_columns]
        for i in range(len(offset_list)):
            offset_list[i] = offset_list[i][pheno_columns]
        if firth:
            for i in range(len(firth_null_list)):
                firth_null_list[i] = firth_null_list[i][pheno_columns]

    if calibrationFile is not None:
        calibration_factors = pd.read_csv(calibrationFile, sep='\t')
        logging.info("Using calibration file specified in: " + str(calibrationFile))
    else:
        ## TODO: make it general for binary traits with firth where the columns double
        calibration_factors = pd.DataFrame(np.ones((len(traits.columns.tolist()) - 2, len(unique_chrs))), columns=unique_chrs)

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
        firth_maf_thresh=firth_maf_thresh,
        firth_prevalence_thresh=firth_prevalence_thresh,
        firth_null=firth_null_list,
    )

    Parallel(n_jobs=n_workers)(
        delayed(adjust_test_stats)(out, phen, calibration_factors[calibration_factors.pheno == phen].iloc[0])
        for i, phen in enumerate(pheno_columns[2:])
    )
    logging.info("Summary stats stored as: " + str(out) + ".{pheno}.sumstats{.gz}")


def main():
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
        help="	Path to ldscores file (should have MAF and LDSCORE columns and tab-seperated)",
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
    parser.add_argument(
        "--firth_maf_thresh",
        help="MAF threshold below which firth logistic regression done",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--firth_prevalence_thresh",
        help="Prevalence threshold below which firth logistic regression done",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--num_threads",
        help="Number of threads to run this code",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--sample_weights",
        help="Sampling weights as (FID, IID, weights) to perform a weighted linear regression",
        type=str,
        default=None
    )

    args = parser.parse_args()

    ######      Logging setup                    #######
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.out + ".log", "w", "utf-8"),
            logging.StreamHandler()
        ]
    )

    logging.info(get_copyright_string())
    logging.info("")
    logging.info("Logs saved in: " + str(args.out + ".log"))
    logging.info("")

    logging.info("Options in effect: ")
    for arg in vars(args):
        logging.info('     {}: {}'.format(arg, getattr(args, arg)))

    logging.info("")

    st = time.time()
    logging.info("#### Start Time: " + str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')) + " ####")
    logging.info("")

    make_sure_path_exists(args.out)

    offsets_file = args.out_step1
    traits = args.out_step1 + ".traits"
    covareffects = args.out_step1 + ".covar_effects"
    firth_null_file = args.out_step1

    assert (args.calibrate and args.unrel_sample_list is not None) or (not args.calibrate) or (args.calibrationFile is not None)
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
            args.covarFile,
            args.calibrate,
            args.out,
            args.binary,
            args.firth,
            args.firth_pval_thresh,
            args.firth_maf_thresh,
            args.firth_prevalence_thresh,
            args.num_threads,
            args.sample_weights
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
            args.firth_maf_thresh,
            args.firth_prevalence_thresh,
            args.num_threads,
            args.sample_weights
        )
    logging.info("#### Step 2. Done in: " + str(time.time() - st) + " secs ####")
    logging.info("")
    logging.info("#### End Time: " + str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')) + " ####")
    logging.info("")


if __name__ == "__main__":
    main()
