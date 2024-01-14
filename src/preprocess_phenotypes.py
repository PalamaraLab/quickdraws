"""
Code for pre-processing the phenotypes
Author: Yiorgos
"""
import pandas as pd
import numpy as np
from pysnptools.snpreader import Bed
import argparse
from sklearn.linear_model import LogisticRegression
import os
import pdb
from scipy.special import expit
from sklearn.preprocessing import quantile_transform
import logging
logger = logging.getLogger(__name__)

def preprocess_phenotypes(pheno, covar, bed, keepfile, binary, phen_thres = 0.0):
    snp_on_disk = Bed(bed, count_A1=True)
    samples_geno = [int(x) for x in snp_on_disk.iid[:, 0]]

    if keepfile is not None:
        keep_samples = pd.read_csv(keepfile, sep="\s+")
        keep_samples = [int(x) for x in keep_samples[keep_samples.columns[0]]]
        samples_geno = list(np.intersect1d(samples_geno, keep_samples))

    # Phenotype loading and alignment
    Traits = pd.read_csv(pheno, sep="\s+", low_memory=False)
    Traits = Traits.sample(frac=1)  ## shuffle the pheno file
    Traits = Traits.set_index(Traits["FID"])
    Traits = Traits.dropna(subset=[Traits.columns[0]])
    N_phen = Traits.shape[1] - 2  # exclude FID, IID

    logging.info("Loading and preparing phenotypes...")
    logging.info("{0} phenotype(s) were loaded for {1} samples".format(N_phen, Traits.shape[0]))
    # remove those without genotypes
    Traits = Traits.drop(
        set(Traits["FID"]).difference(set(samples_geno).intersection(Traits.FID)),
        axis=0,
    )
    Traits.reindex(sorted(Traits.columns), axis=1)
    ## check if any trait is less than 50% phenotyped
    traits_with_missing = np.where(
        Traits.isna().sum(axis=0) / Traits.shape[0] > (1-phen_thres)
    )[0]
    logging.info(
        "{0} traits are less than {1}% phenotyped and will be excluded".format(
            len(traits_with_missing), 100 * phen_thres
        )
    )
    Traits.drop(Traits.columns[traits_with_missing], axis=1, inplace=True)

    ## check if any trait is unary
    traits_unary = np.where(Traits.nunique() == 1)[0]
    logging.info("{0} traits are unary and will be excluded".format(len(traits_unary)))
    Traits.drop(Traits.columns[traits_unary], axis=1, inplace=True)

    ### Mean impute the missing values, to keep things simple ahead
    for phen_col in Traits.columns[2:]:
        Traits[phen_col] = Traits[phen_col].fillna(Traits[phen_col].mean())
    assert sum(Traits.isna().sum()) == 0

    ## check if traits are binary and if nunique == 2 make them binary
    if binary:
        assert np.all(Traits.iloc[:, 2:].nunique() == 2)
        unique_vals = np.unique(Traits.iloc[:, 2:])
        for col in Traits.columns[2:]:
            Traits[col] = Traits[col] == unique_vals[1]
            Traits[col] = Traits[col].astype(int)
        logging.info("Identified {0} binary traits.".format(Traits.shape[1] - 2))

    ## standardize the traits
    else:
        for col in Traits.columns[2:]:
            Traits[col] = (Traits[col] - Traits[col].mean()) / Traits[col].std()
        logging.info("Identified {0} continuous traits.".format(Traits.shape[1] - 2))

    ################ CAUTION #######################

    ### covariate adjustment
    if covar is not None:
        logging.info("Loading and preparing covariates...")
        df_covar = pd.read_csv(covar, sep="\s+", low_memory=False)
        trait_columns = Traits.columns[2:]
        covar_columns = df_covar.columns[2:]
        merged_df = pd.merge(Traits.reset_index(drop=True), df_covar)

        ## Some covariates may have some NaN
        # if np.isnan(merged_df.values).any():
        #     logging.info("Oops! There are a few missing values in the covariates..")
        #     merged_df = merged_df.fillna(merged_df.median())
        merged_df = merged_df.dropna(axis=0)

        samples_to_keep = np.array(merged_df.FID, dtype=int)
        N_total = len(merged_df)
        logging.info(
            "Samples with available genotypes, phenotypes, and covariates to keep for analysis:" + str(N_total)
        )
        W = np.concatenate(
            [merged_df[covar_columns].to_numpy()[:, np.std(merged_df[covar_columns].to_numpy(), axis=0) > 0], np.ones((N_total, 1))], axis=1
        )
        for col in trait_columns:
            Trait = merged_df[col]
            if not binary:
                merged_df["covar_effect_" + str(col)] = W.dot(
                    np.linalg.inv(W.T.dot(W))
                ).dot(W.T.dot(Trait))
            else:
                clf = LogisticRegression(
                    random_state=0, max_iter=50000, fit_intercept=False
                ).fit(W, Trait)
                merged_df["covar_effect_" + str(col)] = (clf.coef_ @ (W.T)).flatten()
    else:
        logging.warning("No covariates will be used! This might lead to calibration issues..")
        samples_to_keep = set(samples_geno).intersection(Traits.FID)
        Traits = Traits.drop(set(Traits["FID"]).difference(samples_to_keep), axis=0)

        N_total = len(Traits)
        logging.info(
            "Samples with available genotypes and phenotypes to keep for analysis:" + str(N_total)
        )

        for col in trait_columns:
            Trait = merged_df[col]
            # Trait -= np.mean(Trait)
            if not binary:
                merged_df["covar_effect_" + str(col)] = np.mean(Trait)
            else:
                clf = LogisticRegression(random_state=0, max_iter=5000).fit(
                    np.ones((N_total, 1)), Trait
                )
                merged_df["covar_effect_" + str(col)] = (
                    clf.coef_ @ (np.ones((N_total, 1)).T)
                ).flatten()
            # Traits.iloc[:, T + 2] = Trait / np.std(Trait)

    Traits = merged_df[["FID", "IID"] + trait_columns.tolist()]
    covar_effects = merged_df[
        ["FID", "IID"] + ["covar_effect_" + str(col) for col in trait_columns.tolist()]
    ]
    sample_indices_to_keep_dict = {}
    for i in range(len(snp_on_disk.iid)):
        if int(snp_on_disk.iid[i, 0]) in samples_to_keep:
            sample_indices_to_keep_dict[int(snp_on_disk.iid[i, 0])] = i

    sample_indices_to_keep = []
    for i in Traits.FID:
        sample_indices_to_keep.append(sample_indices_to_keep_dict[int(i)])

    return (
        Traits,
        covar_effects,
        np.array(sample_indices_to_keep),
    )


def PreparePhenoRHE(Trait, covar_effect, bed, filename, unrel_homo_samples=None):
    """
    Create a new tsv file, with labels [FID, IID, original phenotype, covariate effects] that is aligned with the given fam file,
    Create another tsv file as is required for RHEmc. Trait is assumed to be a dataframe, as usually.
    """
    Trait = Trait.reset_index(drop=True)
    pheno_columns = Trait.columns[2:].tolist()
    covar_effect = covar_effect.reset_index(drop=True)
    covar_columns = covar_effect.columns[2:].tolist()
    Trait_covar = pd.merge(Trait, covar_effect, on=["FID", "IID"])
    Trait_covar[["FID", "IID"] + pheno_columns].to_csv(
        filename + ".traits", index=None, sep="\t", na_rep="NA"
    )
    logging.info("Saving the remaining phenotypes in " + str(filename + ".traits"))
    covar_effects = Trait_covar[["FID", "IID"] + covar_columns]
    covar_effects.columns = ["FID", "IID"] + pheno_columns
    covar_effects.to_csv(filename + ".covar_effects", index=None, sep="\t", na_rep="NA")
    logging.info("Saving the covariate effects on phenotypes in " + str(filename + ".covar_effects"))
    snp_on_disk = Bed(bed, count_A1=True)

    if unrel_homo_samples is not None:
        unrel_homo_samples = pd.read_csv(
            unrel_homo_samples, sep="\s+", names=["FID", "IID"]
        )
        unrel_homo_samples = pd.merge(Trait, unrel_homo_samples, on=["FID", "IID"])
        unrel_sample_list = unrel_homo_samples.FID.tolist()
        logging.info("Number of unrelated homogenous samples: " + str(len(unrel_sample_list)))
    else:
        unrel_sample_list = np.array(snp_on_disk.iid[:, 0].tolist(), dtype="int")

    unrel_pheno = []
    for sample in unrel_sample_list:
        try:
            unrel_pheno.append(Trait[Trait.FID == sample].values[0] - covar_effects[covar_effects.FID == sample].values[0])
        except:
            unrel_pheno.append([sample, sample] + [np.nan] * (Trait.shape[1] - 2))

    assert len(unrel_pheno) == len(unrel_sample_list)

    df_rhe = pd.DataFrame(unrel_pheno, columns=Trait.columns)
    df_rhe.FID = df_rhe.FID.astype("int")
    df_rhe.IID = df_rhe.IID.astype("int")
    df_rhe.to_csv(filename + ".rhe", index=None, sep="\t", na_rep="NA")
    logging.info("Saving the postprocessed phenotypes (to be used by RHE) in " + str(filename + ".rhe"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", "-g", help="prefix for bed/bim/fam files", type=str)
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
        "--keepfile",
        "-r",
        help='file with sample id to keep; should be in "FID,IID" format and tsv',
        type=str,
    )
    args = parser.parse_args()