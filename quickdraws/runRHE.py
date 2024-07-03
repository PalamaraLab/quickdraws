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
Code for getting h2 estimates from RHE-MC
Author: Yiorgos
"""
import pandas as pd
import numpy as np
from pysnptools.snpreader import Bed
import subprocess, os
import argparse
import random
import gc
import numba

from rhe import run_genie_multi_pheno

from .blr import str_to_bool
from scipy.stats import norm
import logging
logger = logging.getLogger(__name__)

def adj_r2_(dotprod, n):
    r2 = dotprod * dotprod
    return r2 - (1 - r2) / (n - 2)

def calc_ldscore_chip(bed, num_samples=4000):
    outlier_window = 1e6

    snp_data = Bed(bed, count_A1=True)
    num_samples = min(num_samples,len(snp_data.iid))
    logging.info("Calculating chip LD score on {0} random samples".format(num_samples))
    chr_map = snp_data.pos[:, 0]  ## chr_no
    pos = snp_data.pos[:, 2]  ## bp pos

    geno = 2 - (
        snp_data[
            random.sample(range(len(snp_data.iid)), num_samples), np.argsort(pos)
        ]
        .read(dtype="float32")
        .val
    )
    maf = np.nanmean(geno, axis=0)/2
    maf = np.minimum(maf, 1-maf)
    mask_dodgy_snp = maf > 0.01
    geno = np.where(
        np.isnan(geno),
        np.nanpercentile(geno, 50, axis=0, interpolation="nearest"),
        geno,
    )
    ## normalize the genotype
    geno -= np.mean(geno, axis=0)
    geno /= np.std(geno, axis=0)
    geno = np.nan_to_num(geno, nan=0)
    sid = snp_data.sid
    geno = geno.T

    ## calculate unadjusted r2 among variants, then adjust it
    ld_score_chip = make_ldscore_inner(geno, pos, mask_dodgy_snp, chr_map, outlier_window, num_samples)
    # ld_score_chip = np.zeros(geno.shape[0])
    # for chr in np.unique(chr_map):
    #     nearby_snps_in_chr = np.arange(geno.shape[0])[chr_map == chr]
    #     pos_chr = pos[chr_map == chr]
    #     mask_dodgy_chr = mask_dodgy_snp[chr_map == chr]
    #     for m1 in nearby_snps_in_chr:
    #         if mask_dodgy_snp[m1]:
    #             nearby_snps = nearby_snps_in_chr[
    #                 (pos[m1] - pos_chr <= outlier_window)
    #                 & (pos[m1] - pos_chr > 0)
    #                 & mask_dodgy_chr
    #             ]
    #             if len(nearby_snps) > 0:
    #                 r2_arr = adj_r2_(
    #                     np.sum(geno[m1] * geno[nearby_snps], axis=1) / num_samples,
    #                     num_samples,
    #                 )
    #                 for m2, r2 in zip(nearby_snps, r2_arr):
    #                     ld_score_chip[m1] += r2
    #                     ld_score_chip[m2] += r2
    ldscore_chip = pd.DataFrame(
        np.vstack([chr_map, sid, ld_score_chip, maf]).T, columns=["CHR", "SNP", "LDSCORE", "MAF"]
    )
    ldscore_chip.LDSCORE = ldscore_chip.LDSCORE.astype("float")
    ldscore_chip.MAF = ldscore_chip.MAF.astype("float")
    ldscore_chip.CHR = ldscore_chip.CHR.astype("float").astype("int")
   
    del geno
    gc.collect()
    return ldscore_chip

@numba.jit(nopython=True)
def make_ldscore_inner(geno, pos, mask_dodgy_snp, chr_map, outlier_window, num_samples):
    ld_score_chip = np.zeros(geno.shape[0])
    for chr in np.unique(chr_map):
        nearby_snps_in_chr = np.arange(geno.shape[0])[chr_map == chr]
        pos_chr = pos[chr_map == chr]
        mask_dodgy_chr = mask_dodgy_snp[chr_map == chr]
        for m1 in nearby_snps_in_chr:
            if mask_dodgy_snp[m1]:
                nearby_snps = nearby_snps_in_chr[
                    (pos[m1] - pos_chr <= outlier_window)
                    & (pos[m1] - pos_chr > 0)
                    & mask_dodgy_chr
                ]
                if len(nearby_snps) > 0:
                    r2_arr_unadjusted = np.sum(geno[m1] * geno[nearby_snps], axis=1) / num_samples
                    r2_arr = r2_arr_unadjusted*r2_arr_unadjusted
                    r2_arr = r2_arr - (1-r2_arr)/(num_samples - 2)
                    for m2, r2 in zip(nearby_snps, r2_arr):
                        ld_score_chip[m1] += r2
                        ld_score_chip[m2] += r2  

    return ld_score_chip 

def MakeAnnotation(bed, maf_ldscores, snps_to_keep_filename, maf_bins, ld_score_percentiles, outfile=None):
    logging.info("Making annotation for accurate h2 estimation")
    if maf_ldscores is not None:
        df = pd.read_csv(maf_ldscores, sep=r'\s+')
    else:
        df = calc_ldscore_chip(bed)

    bim = pd.read_csv(bed + ".bim", header=None, sep=r'\s+')
    bim = bim.rename(columns={0:'CHR',1:'SNP'})
    dtype = dict(CHR=str, SNP=str)
    df = pd.merge(bim.astype(dtype), df.astype(dtype), on=['CHR','SNP'], how='left')

    logging.info("Number of SNPs with MAF/LD information = " + str(len(df) - df['MAF'].isna().sum()))
    is_missing = int(df['MAF'].isna().any())

    if 'MAF' in df.columns:
        mafs = df.MAF.values
    elif 'maf' in df.columns:
        mafs = df.maf.values 
    else:
        logging.exception("Didn't find a MAF/maf column in ldscores files")

    if 'LDSCORE' in df.columns:
        ld_scores = df.LDSCORE.values
    elif 'ldscore' in df.columns:
        ld_scores = df.ldscore.values
    else:
        logging.exception("Didn't find a LDSCORE/ldscore column in ldscores files")

    # subsetting to variants in the correct MAF bin
    n_maf_bins = len(maf_bins) - 1
    n_ld_bins = len(ld_score_percentiles) - 1
    tot_cats = np.zeros(shape=(mafs.size, n_maf_bins * n_ld_bins + is_missing), dtype=np.uint8)

    i = 0
    for j in range(n_maf_bins):
        maf_idx = (mafs > maf_bins[j]) & (mafs <= maf_bins[j+1])
        if np.sum(maf_idx) > 0:
            tru_ld_quantiles = [
                np.quantile(ld_scores[maf_idx], i) for i in ld_score_percentiles
            ]
            for k in range(n_ld_bins):
                ld_idx = (ld_scores > tru_ld_quantiles[k]) & (
                    ld_scores <= tru_ld_quantiles[k+1]
                )
                cat_idx = np.where(maf_idx & ld_idx)[0]
                # Set the category to one
                tot_cats[cat_idx, i] = 1
                i += 1

    if is_missing > 0:
        cat_idx = np.where(np.isnan(mafs))[0]
        tot_cats[cat_idx, -1] = 1

    # remove variants from the HLA region (as they are sensitive to RHE)
    hla = bim[bim['CHR'] == 6]
    hla = hla[hla[3] > 28.4e6]
    hla = hla[hla[3] < 33.5e6]
    tot_cats[hla.index] = 0

    # remove variants not in modelSnps
    snp_on_disk = Bed(bed, count_A1=True)
    if snps_to_keep_filename is None:
        total_snps = snp_on_disk.sid_count
        snp_mask = np.ones(total_snps, dtype="bool")
    else:
        snps_to_keep = pd.read_csv(snps_to_keep_filename, sep=r'\s+')
        snps_to_keep = snps_to_keep[snps_to_keep.columns[0]].values
        snp_dict = {}
        total_snps = snp_on_disk.sid_count
        snp_mask = np.zeros(total_snps, dtype="bool")
        for snp_no, snp in enumerate(snp_on_disk.sid):
            snp_dict[snp] = snp_no
        for snp in snps_to_keep:
            snp_mask[snp_dict[snp]] = True

    tot_cats[np.logical_not(snp_mask)] = 0

    #   Make sure there are SNPs in every category
    tot_cats = tot_cats[:, np.sum(tot_cats, axis=0)>0]
    np.savetxt(outfile, tot_cats, fmt="%d")
    return


def runRHE(
    bedfile,
    pheno,
    snps_to_keep_filename,
    annotation,
    savelog,
    rhemc,
    covariates=None,
    out="out",
    binary=False,
    random_vectors=50,
    jn=10
):
    """
    A wrapper that prepares a file with SNP annotations (if needed), runs RHE on the background,
    (always), and returns a list with the estimates. Can deal with any number of components.
    If `annotation` is an integer, we create a new file, otherwise we read from disk.
    We assume K genetic components +1 for the environment.
    """
    snp_on_disk = Bed(bedfile, count_A1=True)
    _, M = snp_on_disk.shape  # get the number of SNPs

    if snps_to_keep_filename is None:
        snp_mask = np.ones(M, dtype="bool")
    else:
        snps_to_keep = pd.read_csv(snps_to_keep_filename, sep=r'\s+')
        snps_to_keep = snps_to_keep[snps_to_keep.columns[0]].values
        snp_dict = {}
        snp_mask = np.zeros(M, dtype="bool")
        for snp_no, snp in enumerate(snp_on_disk.sid):
            snp_dict[snp] = snp_no
        for snp in snps_to_keep:
            snp_mask[snp_dict[snp]] = True

    if annotation is not None:
        # we assume the annotation exists already and infer K
        logging.info("Opening annotation file provided, doing MC GWAS")
        table = pd.read_csv(annotation, sep=r'\s+', header=None)
        K = table.shape[1]
        assert table.shape[0] == M
    else:
        # create a new annotation
        K = 8
        annot = np.random.randint(0, high=K, size=M)
        table = np.zeros((M, K), dtype=np.int8)
        for i in range(M):
            if snp_mask[i]:
                table[i, annot[i]] = 1
        table = pd.DataFrame(table)
        table.to_csv(out + ".random.annot", header=None, index=None, sep=" ")
        annotation = out + ".random.annot"

    N_phen = pd.read_csv(pheno, sep=r'\s+').shape[1] - 2

    if random_vectors < N_phen:
        logging.exception("Supply more random vectors than the phenotypes being analyzed")
        raise ValueError

    # now run RHE
    if True:
        cmd = " -g " + bedfile + " -p " + pheno + " -annot " + annotation
        if covariates is not None:
            pheno_df = pd.read_csv(pheno, sep=r'\s+')
            covariates_df = pd.read_csv(covariates, sep=r'\s+')
            covar_df_cols = covariates_df.columns.tolist()
            covariates_df = pd.merge(covariates_df, pheno_df)[covar_df_cols]
            covariates_df.to_csv(out + ".rhe.covars", sep="\t", index=None, na_rep="NA")
            cmd += " -c " + out + ".rhe.covars"
        cmd += f' -k {random_vectors} -jn {jn} -m G -o {savelog}'
        logging.info("Invoking RHE as: GENIE_multi_pheno " + str(cmd))
        _ = run_genie_multi_pheno(cmd)

    if os.path.isfile(savelog):
        VC = []
        VC_phen = []
        with open(savelog, "r") as f:
            for line in f.readlines():
                if "Sigma^2" in line:
                    VC_phen.append(float(line.split(":")[1].split("SE")[0].split(" ")[1]))
                if "Sigma^2_e" in line:
                    VC.append(VC_phen)
                    VC_phen = []
        VC = np.array(VC[0:N_phen])  # take first N_phen rows

        VC = np.sum(VC[:, :-1], axis=1) / np.sum(VC, axis=1)
        VC[VC <= 0] = 1e-2
        VC[VC >= 1] = (1 - 1e-2)

        if binary:
            ## transform heritability from observed scale to liability scale
            pheno_df = pd.read_csv(pheno, sep=r'\s+')
            for pheno in range(len(VC)):
                prev = pheno_df.values[:, 2 + pheno].mean()
                z = norm.pdf(norm.ppf(1 - prev))
                VC[pheno] *= prev * (1 - prev) / (z**2)

        np.savetxt(out + ".h2", VC)
        logging.info(
            "Variance components estimated as "
            + str(VC)
            + " saved in "
            + str(out + ".h2")
        )
        return VC
    else:
        logging.exception("ERROR: RHEmc did not complete.")
        raise ValueError
        return "nan"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", "-g", help="prefix for bed/bim/fam files", type=str)
    parser.add_argument(
        "--output", "-o", help="prefix for where to save any results or files"
    )
    parser.add_argument(
        "--annot",
        help="file with annotation; one column per component; no overlapping",
        type=str,
    )
    parser.add_argument(
        "--make_annot",
        help="Do you wish to make the annot file ?",
        type=str_to_bool,
        default="false",
    )
    parser.add_argument(
        "--covar",
        "-c",
        help='file with covariates; should be in "FID,IID,Var1,Var2,..." format and tsv',
        type=str,
    )
    parser.add_argument(
        "--pheno",
        "-p",
        help='phenotype file; should be in "FID,IID,Trait" format and tsv',
        type=str,
    )
    parser.add_argument(
        "--unrel_sample_list",
        help="File with un-related homogenous sample list (FID, IID)",
        type=str,
    )
    parser.add_argument(
        "--modelSnps",
        help="Path to list of SNPs to be considered in BLR",
        default=None,
        type=str,
    )
    parser.add_argument("--rhemc", type=str, help="path to RHE-MC binary file")
    args = parser.parse_args()