"""
Code for getting h2 estimates from RHE-MC
Author: Yiorgos
"""
import pandas as pd
import numpy as np
from pysnptools.snpreader import Bed
import subprocess, os
import argparse

from preprocess_phenotypes import preprocess_phenotypes, PreparePhenoRHE
from blr import str_to_bool
from scipy.stats import norm


def MakeAnnotation(bed, maf_bins, ld_score_percentiles, outfile=None):
    """Intermediate helper function to generate MAF / LD structured annotations. Credits: Arjun"""
    try:
        df = pd.read_csv("/well/palamara/projects/UKBB_APPLICATION_43206/new_copy/plink_missingness_regenielike_filters/ukb_app43206_500k.maf00001.score.ld", sep="\s+")
        print("MAF/LD info are loaded for {0} SNPs".format(len(df)))
    except:
        print("File with MAF-LD scores is wrong!")
        raise ValueError
    mafs = df.MAF.values
    ld_scores = df.ldscore.values
    assert mafs.size == ld_scores.size

    # subsetting to variants in the correct MAF bin
    n_maf_bins = len(maf_bins) - 1
    n_ld_bins = len(ld_score_percentiles) - 1
    tot_cats = np.zeros(shape=(mafs.size, n_maf_bins * n_ld_bins), dtype=np.uint8)
    i = 0
    for j in range(n_maf_bins):
        if j == 0:
            maf_idx = (mafs > maf_bins[j]) & (mafs <= maf_bins[j + 1])
        else:
            maf_idx = (mafs > maf_bins[j]) & (mafs <= maf_bins[j + 1])
        tru_ld_quantiles = [
            np.quantile(ld_scores[maf_idx], i) for i in ld_score_percentiles
        ]
        for k in range(n_ld_bins):
            if k == 0:
                ld_idx = (ld_scores >= tru_ld_quantiles[k]) & (
                    ld_scores <= tru_ld_quantiles[k + 1]
                )
            else:
                ld_idx = (ld_scores > tru_ld_quantiles[k]) & (
                    ld_scores <= tru_ld_quantiles[k + 1]
                )
            cat_idx = np.where(maf_idx & ld_idx)[0]
            # Set the category to one
            tot_cats[cat_idx, i] = 1
            i += 1

    # remove variants from the HLA region (as they are sensitive to RHE)
    hla = pd.read_csv(
        bed + ".bim",
        header=None,
        sep="\s+",
    )
    hla = hla[hla[0] == 6]
    hla = hla[hla[3] > 28.4e6]
    hla = hla[hla[3] < 33.5e6]
    print("Variants removed from HLA:", len(hla))
    tot_cats[hla.index] = 0

    #   Make sure there are SNPs in every category
    assert np.all(np.sum(tot_cats, axis=0) > 0)
    np.savetxt(outfile, tot_cats, fmt="%d")

    return


def runSCORE(bedfile, pheno, snps_to_keep_filename, score, out="out"):
    snp_on_disk = Bed(bedfile, count_A1=True)
    _, M = snp_on_disk.shape  # get the number of SNPs

    if snps_to_keep_filename is None:
        snp_mask = np.ones(M, dtype="bool")
    else:
        snps_to_keep = pd.read_csv(snps_to_keep_filename, sep="\s+")
        snps_to_keep = snps_to_keep[snps_to_keep.columns[0]].values
        snp_dict = {}
        snp_mask = np.zeros(M, dtype="bool")
        for snp_no, snp in enumerate(snp_on_disk.sid):
            snp_dict[snp] = snp_no
        for snp in snps_to_keep:
            snp_mask[snp_dict[snp]] = True

    sdata = snp_on_disk[:, snp_mask].read(dtype="float32")
    Bed.write(
        bedfile + ".common.bed",
        sdata,
        count_A1=False,
    )
    bim_file = pd.read_csv(bedfile + ".bim", sep="\s+", header=None)
    bim_file.loc[snp_mask].to_csv(
        bedfile + ".common.bim", sep="\t", index=None, header=None
    )

    if not os.path.isfile(out):
        cmd = (
            score
            + " -mpheno 3,4 -g "
            + bedfile
            + ".common -p "
            + pheno
            + " -o "
            + str(out)
        )
        print("Invoking SCORE as", cmd)
        _ = subprocess.run(cmd, shell=True)
    N_phen = pd.read_csv(pheno, sep="\s+").shape[1] - 2
    VC_full = np.zeros((N_phen, N_phen))
    with open(out, "r") as f:
        for line in f.readlines():
            if "Vg/Vp(" in line:
                phen_no = int(line.split("(")[1].split(")")[0])
                VC_full[phen_no, phen_no] = float(line.split("\t")[1])
            elif "rho_g(" in line:
                phen_no1 = int(line.split("(")[1].split(",")[0])
                phen_no2 = int(line.split("(")[1].split(")")[0].split(",")[1])
                VC_full[phen_no1, phen_no2] = float(line.split("\t")[1])
                VC_full[phen_no2, phen_no1] = float(line.split("\t")[1])
    print("The genetic covariance matrix inferred from SCORE: " + str(VC_full))
    np.savetxt(out + ".h2", VC_full)
    return VC_full


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
        snps_to_keep = pd.read_csv(snps_to_keep_filename, sep="\s+")
        snps_to_keep = snps_to_keep[snps_to_keep.columns[0]].values
        snp_dict = {}
        snp_mask = np.zeros(M, dtype="bool")
        for snp_no, snp in enumerate(snp_on_disk.sid):
            snp_dict[snp] = snp_no
        for snp in snps_to_keep:
            snp_mask[snp_dict[snp]] = True

    if annotation is not None:
        # we assume the annotation exists already and infer K
        print("Opening annotation file provided, doing MC GWAS")
        table = pd.read_csv(annotation, sep="\s+", header=None)
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

        # snp_on_disk = Bed(bedfile, count_A1=True)
        # chrs = snp_on_disk.pos[:, 0]
        # chrs = chrs - np.min(chrs)
        # table = np.zeros((len(chrs), len(np.unique(chrs))), dtype=np.int8)
        # for i in range(len(chrs)):
        #     table[i, int(chrs[i])] = 1
        # table = pd.DataFrame(table)
        # table.to_csv(out + ".random.annot", header=None, index=None, sep=" ")
        # annotation = out + ".random.annot"

    N_phen = pd.read_csv(pheno, sep="\s+").shape[1] - 2
    # now run RHE
    if True:
        cmd = "./" + rhemc + " -g " + bedfile + " -p " + pheno + " -annot " + annotation
        if covariates is not None:
            pheno_df = pd.read_csv(pheno, sep="\s+")
            covariates_df = pd.read_csv(covariates, sep="\s+")
            covar_df_cols = covariates_df.columns.tolist()
            covariates_df = pd.merge(covariates_df, pheno_df)[covar_df_cols]
            covariates_df.to_csv(covariates + ".rhe", sep="\t", index=None, na_rep="NA")
            cmd += " -c " + covariates + ".rhe"
        cmd += " -k " + str(random_vectors) + " -jn 10 > " + savelog
        print("Invoking RHE as", cmd)
        _ = subprocess.run(cmd, shell=True)

    if os.path.isfile(savelog):
        VC = []
        VC_phen = []
        with open(savelog, "r") as f:
            for line in f.readlines():
                if "Sigma^2" in line:
                    VC_phen.append(float(line.split(":")[1].split(" ")[1]))
                if "Sigma^2_e" in line:
                    VC.append(VC_phen)
                    VC_phen = []
        VC = np.array(VC[0:N_phen])  # take first N_phen rows

        VC = np.sum(VC[:, :-1], axis=1) / np.sum(VC, axis=1)
        VC[VC <= 0] = 1e-2
        VC[VC >= 1] = (1 - 1e-2)

        if binary:
            ## transform heritability from observed scale to liability scale
            pheno_df = pd.read_csv(pheno, sep="\s+")
            for pheno in range(len(VC)):
                prev = pheno_df.values[:, 2 + pheno].mean()
                z = norm.pdf(norm.ppf(1 - prev))
                VC[pheno] *= prev * (1 - prev) / (z**2)

        np.savetxt(out + ".h2", VC)
        print(
            "Variance components estimated as "
            + str(VC)
            + " saved in "
            + str(out + ".h2")
        )
        return VC
    else:
        print("ERROR: RHEmc did not complete.")
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
    adj_pheno_file = ".".join([args.output, "adjusted_traits", "phen"])
    Traits, sample_indices = preprocess_phenotypes(
        args.pheno, args.covar, args.bed, None
    )
    PreparePhenoRHE(Traits, args.bed, adj_pheno_file, None)
    if args.make_annot:
        print("No annotation was given so we'll make one now")
        args.annot = args.output + ".maf2_ld4.annot"
        MakeAnnotation(
            args.bed,
            [0.01, 0.05, 0.5],
            [0.0, 0.25, 0.5, 0.75, 1.0],
            args.annot,
        )

    # VC = runRHE(
    #     adj_pheno_file,
    #     adj_pheno_file + ".rhe",
    #     args.annot,
    #     args.output + ".rhe.log",
    #     args.rhemc,
    #     args.output,
    # )
    runSCORE(args.bed, adj_pheno_file + ".rhe", args.modelSnps, args.rhemc, args.output)
