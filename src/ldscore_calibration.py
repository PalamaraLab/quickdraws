"""
Custom script to perform weighted LD-score regression as in BOLT-LMM
Note: we assume and use base pair location instead of genetic position 
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import subprocess
from pathlib import Path
import random
from pysnptools.snpreader import Bed
import pdb
import copy
import gc

outlier_window = 1e6
minMAF = 0.01
outlierVarFracThresh = 0.001
min_outlier_chisq_thresh = 20.0


def adj_r2_(dotprod, n):
    r2 = dotprod * dotprod
    return r2 - (1 - r2) / (n - 2)


def calc_ldscore_chip(bed, mask_dodgy_snp, unrel_sample_list=None, num_samples=400):
    """
    Calculates the LD scores on the masked SNPs
    1. Normalize and mask the genotypes
    2. Calculate sum over dot product with nearby varaints
    3. adjust the dot product to get ld scores
    mask_dodgy_snp: binary mask array for SNPs
    mask_dodgy_samples: binary mask array for samples
    """
    # print("Calculating chip LD score on {0} random samples".format(num_samples))
    snp_data = Bed(bed, count_A1=True)
    chr_map = snp_data.pos[:, 0]  ## chr_no
    pos = snp_data.pos[:, 2]  ## bp pos

    assert len(mask_dodgy_snp) == len(pos)

    if unrel_sample_list is not None:
        geno = 2 - (
            snp_data[
                random.sample(unrel_sample_list, num_samples),
                np.argsort(pos),
            ]
            .read(dtype="float32")
            .val
        )
    else:
        geno = 2 - (
            snp_data[
                random.sample(range(len(snp_data.iid)), num_samples), np.argsort(pos)
            ]
            .read(dtype="float32")
            .val
        )
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
                    r2_arr = adj_r2_(
                        np.sum(geno[m1] * geno[nearby_snps], axis=1) / num_samples,
                        num_samples,
                    )
                    for m2, r2 in zip(nearby_snps, r2_arr):
                        ld_score_chip[m1] += r2
                        ld_score_chip[m2] += r2
    ldscore_chip = pd.DataFrame(
        np.vstack([sid, ld_score_chip]).T, columns=["SNP", "LDSCORE_CHIP"]
    )
    ldscore_chip.LDSCORE_CHIP = ldscore_chip.LDSCORE_CHIP.astype("float")
    del geno
    gc.collect()
    return ldscore_chip


def get_mask_dodgy(ldscores, sumstats, N):
    """
    Returns mask on sumstats file based on which SNPs to keep
    """
    num_snps = len(sumstats)
    mask_dodgy = np.ones(num_snps, dtype=bool)
    pos_column = "POS"

    sumstats["CHISQ_FLT"] = sumstats["CHISQ"].astype("float")
    sumstats = sumstats.sort_values("POS")
    chisq_thresh = max(min_outlier_chisq_thresh, N * outlierVarFracThresh)
    # print("Removing SNPs above chisq " + str(chisq_thresh))
    snp_above_chisqth = np.where(sumstats["CHISQ_FLT"] >= chisq_thresh)[0]

    pos_arr = sumstats[pos_column].values
    maf_arr = sumstats["ALT_FREQS"].values
    chisq_arr = sumstats["CHISQ_FLT"].values
    if len(snp_above_chisqth) > 0:
        count = snp_above_chisqth[0]

    for snp in range(num_snps):
        if maf_arr[snp] < minMAF or maf_arr[snp] > 1 - minMAF:
            mask_dodgy[snp] = False
        if np.isnan(chisq_arr[snp]):
            mask_dodgy[snp] = False
        if len(snp_above_chisqth) > 0:
            if np.abs(pos_arr[snp] - pos_arr[count]) < outlier_window:
                mask_dodgy[snp] = False
            else:
                if snp > count:
                    count += 1

    ldscores["SNP"] = ldscores["SNP"].astype("str")
    sumstats["SNP"] = sumstats["SNP"].astype("str")
    df_all = sumstats.merge(
        ldscores.drop_duplicates(), on="SNP", how="left", indicator=True
    )

    mask_dodgy2 = (df_all["_merge"] == "both").values
    # print(
    #     "Number of SNPs remaining after filtering = "
    #     + str(np.sum(mask_dodgy * mask_dodgy2))
    # )
    return mask_dodgy * mask_dodgy2


def ldscore_intercept(ldscores, sumstats, ldscore_chip, mask_dodgy):
    """
    Masking SNPs based on:
    1. availability in LDscore file and sumstats file
    2. MAF in our dataset >= 0.01
    3. Remove nearby and top 0.1% (or SNPs with CHISQ >= 20) based on CHISQ
    """
    sumstats = copy.deepcopy(sumstats)
    if "SNP" not in sumstats.columns:
        sumstats.rename(columns={"ID": "SNP"}, inplace=True)
    sumstats = sumstats.sort_values("POS")
    sumstats["CHISQ_FLT"] = sumstats["CHISQ"].astype("float")

    chisq_arr = sumstats["CHISQ_FLT"].values[mask_dodgy]
    ldscores["SNP"] = ldscores["SNP"].astype("str")
    sumstats["SNP"] = sumstats["SNP"].astype("str")
    ldscore_arr = pd.merge(sumstats, ldscores.drop_duplicates(), on="SNP", how="left")[
        "LDSCORE"
    ].values[mask_dodgy]
    ldscore_chip_arr = np.array(
        ldscore_chip["LDSCORE_CHIP"].values[mask_dodgy], dtype="float"
    )
    # print("Number of SNPs available with LDscores = " + str(len(ldscore_arr)))

    """
    Perform a weighted linear regression of CHISQ with LDSCORES to get intercept and slope
    """
    new_mask = ~np.isnan(ldscore_chip_arr)
    slopeToCM = (np.mean(chisq_arr[new_mask]) - 1) / np.mean(ldscore_arr[new_mask])
    weight = (1 / np.maximum(1, 1 + slopeToCM * ldscore_arr[new_mask])) ** 2
    weight *= 1 / np.maximum(1, ldscore_chip_arr[new_mask])
    wls_model = sm.WLS(
        chisq_arr[new_mask], sm.add_constant(ldscore_arr[new_mask]), weights=weight
    )
    results = wls_model.fit()
    return results.params[0], np.mean(chisq_arr[new_mask])


def ldscore_genetic_covar(ldscores, sumstats_1, sumstats_2, mask_dodgy):
    sumstats_1 = copy.deepcopy(sumstats_1)
    sumstats_2 = copy.deepcopy(sumstats_2)
    if "SNP" not in sumstats_1.columns:
        sumstats_1.rename(columns={"ID": "SNP"}, inplace=True)
        sumstats_2.rename(columns={"ID": "SNP"}, inplace=True)
    sumstats_1 = sumstats_1.sort_values("POS")
    sumstats_2 = sumstats_2.sort_values("POS")
    sumstats_1["CHISQ_FLT"] = sumstats_1["CHISQ"].astype("float")
    sumstats_2["CHISQ_FLT"] = sumstats_2["CHISQ"].astype("float")

    z1z2_arr = np.sqrt(sumstats_1["CHISQ_FLT"].values[mask_dodgy]) * np.sqrt(
        sumstats_2["CHISQ_FLT"].values[mask_dodgy]
    )
    assert (sumstats_1.SNP == sumstats_2.SNP).all()
    ldscore_arr = pd.merge(
        sumstats_1, ldscores.drop_duplicates(), on="SNP", how="left"
    )["LDSCORE"].values[mask_dodgy]
    # print("Number of SNPs available with LDscores = " + str(len(ldscore_arr)))

    """
    Perform a weighted linear regression of CHISQ with LDSCORES to get intercept and slope
    """
    slopeToCM = (np.mean(z1z2_arr) - 1) / np.mean(ldscore_arr)
    weight = (1 / np.maximum(1, 1 + slopeToCM * ldscore_arr)) ** 2
    wls_model = sm.WLS(z1z2_arr, sm.add_constant(ldscore_arr), weights=weight)
    results = wls_model.fit()
    return results.params[1] * len(ldscore_arr) / np.mean(sumstats_1["OBS_CT"])


if __name__ == "__main__":
    import glob

    bed = "ukbb_gbp/genotype"
    ldscores = pd.read_csv("LDSCORE.1000G_EUR.tab.gz", sep="\s+")
    for name in glob.glob("ukbb_gbp/blr_sc.*sumstats.gz"):
        pheno = name.split("ukbb_gbp/blr_sc")[1].split(".sumstats.gz")[0]
        sumstats = pd.read_csv(
            name,
            sep="\s+",
        )
        mask_dodgy = get_mask_dodgy(
            ldscores[["SNP", "LDSCORE"]], sumstats, sumstats["OBS_CT"].mean()
        )
        mask_dodgy *= (np.random.rand(len(mask_dodgy)) > 0.5).astype(bool)
        ldscore_chip = calc_ldscore_chip(bed, mask_dodgy)
        y_i, m_chi = ldscore_intercept(ldscores, sumstats, ldscore_chip, mask_dodgy)
        print("Quickdraws: " + str((y_i - 1) / (m_chi - 1)))

        sumstats = pd.read_hdf(
            "ukbb_gbp/blr_sc" + pheno + ".sumstats",
        )
        y_i, m_chi = ldscore_intercept(ldscores, sumstats, ldscore_chip, mask_dodgy)
        print("LR-unRel: " + str((y_i - 1) / (m_chi - 1)))
