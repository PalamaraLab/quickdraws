"""
Code for pre-processing the phenotypes
Author: Yiorgos
"""
import pandas as pd
import numpy as np
from pysnptools.snpreader import Bed
import argparse
import os
import pdb


def preprocess_phenotypes(pheno, covar, bed, removeFile):
    snp_on_disk = Bed(bed, count_A1=True)
    samples_geno = [int(x) for x in snp_on_disk.iid[:, 0]]

    if removeFile is not None:
        remove_samples = pd.read_csv(removeFile, sep="\s+")
        remove_samples = [int(x) for x in remove_samples[remove_samples.columns[0]]]
        samples_geno = list(set(samples_geno) - set(remove_samples))

    # Phenotype loading and alignment
    Traits = pd.read_csv(pheno, sep="\s+", low_memory=False)
    Traits = Traits.sample(frac=1)  ## shuffle the pheno file
    Traits = Traits.set_index(Traits["FID"])
    Traits = Traits.dropna(subset=[Traits.columns[0]])
    N_phen = Traits.shape[1] - 2  # exclude FID, IID

    if N_phen > 50:
        raise "RHE doesn't work more than 50 phenotypes at once, try supplying atmost 50 phenotypes at once"

    print("{0} phenotypes were loaded for {1} samples.".format(N_phen, Traits.shape[0]))
    # remove those without genotypes
    Traits = Traits.drop(
        set(Traits["FID"]).difference(set(samples_geno).intersection(Traits.FID)),
        axis=0,
    )
    Traits.reindex(sorted(Traits.columns), axis=1)
    # check if any individual is less than 50% phenotyped
    phen_thres = 0.50
    samples_with_missing = np.where(Traits.isna().sum(axis=1) / N_phen > phen_thres)[0]
    print(
        "{0} samples are less than {1}% phenotyped and will be excluded.".format(
            len(samples_with_missing), 100 * phen_thres
        )
    )
    Traits.drop(Traits.FID.iloc[samples_with_missing], axis=0, inplace=True)

    ### Mean impute the missing values, to keep things simple ahead
    for phen_col in Traits.columns[2:]:
        Traits[phen_col] = Traits[phen_col].fillna(Traits[phen_col].mean())
    assert sum(Traits.isna().sum()) == 0

    ### covariate adjustment
    if covar is not None:
        print("Loading and preparing covariates...")
        df_covar = pd.read_csv(covar, sep="\s+", low_memory=False)
        trait_columns = Traits.columns[2:]
        covar_columns = df_covar.columns[2:]
        merged_df = pd.merge(Traits.reset_index(drop=True), df_covar)

        ## Some covariates may have some NaN
        if np.isnan(merged_df.values).any():
            print("Oops! There are a few missing values in the covariates..")
            merged_df = merged_df.fillna(merged_df.median())

        samples_to_keep = np.array(merged_df.FID, dtype=int)
        N_total = len(merged_df)
        print(
            "Samples with available genotypes, phenotypes, and covariates to keep for analysis:",
            N_total,
        )
        W = np.concatenate(
            [merged_df[covar_columns].to_numpy(), np.ones((N_total, 1))], axis=1
        )
        for col in trait_columns:
            Trait = merged_df[col]
            Trait -= W.dot(np.linalg.inv(W.T.dot(W))).dot(W.T.dot(Trait))
            merged_df[col] = (Trait - np.mean(Trait)) / np.std(Trait)
        print(
            "The traits are now adjusted with respect to the covariates, mean-centered and of unit variance."
        )
        Traits = merged_df[["FID", "IID"] + trait_columns.tolist()]
    else:
        print("\nWARNING: No covariates will be used! Are the traits already adjusted?")
        samples_to_keep = set(samples_geno).intersection(Traits.FID)
        Traits = Traits.drop(set(Traits["FID"]).difference(samples_to_keep), axis=0)

        N_total = len(Traits)
        print(
            "Samples with available genotypes and phenotypes to keep for analysis:",
            N_total,
        )

        for T in range(N_phen):
            Trait = Traits.iloc[:, T + 2]
            Trait -= np.mean(Trait)
            Traits.iloc[:, T + 2] = Trait / np.std(Trait)
        print("The traits are now mean-centered and of unit variance.")

    sample_indices_to_keep_dict = {}
    for i in range(len(snp_on_disk.iid)):
        if int(snp_on_disk.iid[i, 0]) in samples_to_keep:
            sample_indices_to_keep_dict[int(snp_on_disk.iid[i, 0])] = i

    sample_indices_to_keep = []
    for i in Traits.FID:
        sample_indices_to_keep.append(sample_indices_to_keep_dict[int(i)])

    return Traits, np.array(sample_indices_to_keep)


def PreparePhenoRHE(Trait, bed, filename, unrel_homo_samples=None):
    """
    Create a new tsv file, with labels [FID, IID, Trait] that is aligned with the given fam file,
    as is required for RHEmc. Trait is assumed to be a dataframe, as usually.
    """
    Trait = Trait.reset_index(drop=True)
    Trait.to_csv(filename, index=None, sep="\t", na_rep="NA")

    snp_on_disk = Bed(bed, count_A1=True)

    # if snps_to_keep_filename is None:
    #     total_snps = snp_on_disk.sid_count
    #     snp_mask = np.ones(total_snps, dtype="bool")
    # else:
    #     snps_to_keep = pd.read_csv(snps_to_keep_filename, sep="\s+")
    #     snps_to_keep = snps_to_keep[snps_to_keep.columns[0]].values
    #     snp_dict = {}
    #     total_snps = snp_on_disk.sid_count
    #     snp_mask = np.zeros(total_snps, dtype="bool")
    #     for snp_no, snp in enumerate(snp_on_disk.sid):
    #         snp_dict[snp] = snp_no
    #     for snp in snps_to_keep:
    #         snp_mask[snp_dict[snp]] = True

    if unrel_homo_samples is not None:
        unrel_homo_samples = pd.read_csv(
            unrel_homo_samples, sep="\s+", names=["FID", "IID"]
        )
        unrel_homo_samples = pd.merge(Trait, unrel_homo_samples, on=["FID", "IID"])
        unrel_sample_list = unrel_homo_samples.FID.tolist()
        print("Number of unrelated homogenous samples: " + str(len(unrel_sample_list)))
    else:
        unrel_sample_list = np.array(snp_on_disk.iid[:, 0].tolist(), dtype="int")

    # sample_dict = {}
    # for i in range(len(snp_on_disk.iid)):
    #     if int(snp_on_disk.iid[i, 0]) in unrel_sample_list:
    #         sample_dict[int(snp_on_disk.iid[i, 0])] = i

    # unrel_sample_indices = []
    # for i in unrel_sample_list:
    #     unrel_sample_indices.append(sample_dict[i])

    # sdata = snp_on_disk[unrel_sample_indices, snp_mask].read(
    #     dtype="int8", _require_float32_64=False
    # )
    # Bed.write(
    #     filename + ".bed",
    #     sdata,
    #     count_A1=True,
    # )
    # bim_file = pd.read_csv(bed + ".bim", sep="\s+", header=None)
    # bim_file.loc[snp_mask].to_csv(
    #     str(filename) + ".bim", sep="\t", index=None, header=None
    # )

    unrel_pheno = []
    for sample in unrel_sample_list:
        try:
            unrel_pheno.append(Trait[Trait.FID == sample].values[0])
        except:
            unrel_pheno.append([sample, sample] + [np.nan] * (Trait.shape[1] - 2))

    assert len(unrel_pheno) == len(unrel_sample_list)

    df_rhe = pd.DataFrame(unrel_pheno, columns=Trait.columns)
    df_rhe.FID = df_rhe.FID.astype("int")
    df_rhe.IID = df_rhe.IID.astype("int")
    df_rhe.to_csv(filename + ".rhe", index=None, sep="\t", na_rep="NA")


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
        "--removeFile",
        "-r",
        help='file with sample id to remove; should be in "FID,IID" format and tsv',
        type=str,
    )
    args = parser.parse_args()
    preprocess_phenotypes(args.pheno, args.covar, args.bed, args.removeFile)


"""
Phenotype QC:
import pandas as pd 
import numpy as np
df = pd.read_csv('blood_count_biochemistry.csv', sep='\s+')
for Trait in df.columns[2:]:
    df[Trait] = (df[Trait] - np.mean(df[Trait])) / np.std(df[Trait])
df2 = df[df.columns[2:]].clip(-10,10)
df2[['FID','IID']] = df[['FID','IID']]
df2 = df2[df.columns]
for Trait in df.columns[2:]:
    df2[Trait] = (df2[Trait] - np.mean(df2[Trait])) / np.std(df2[Trait])
df2 = df2[df2.columns[0:52]] ## only 50 phenptypes
df2.to_csv('blood_count_biochemistry_outlier.csv', sep='\t', index=None, na_rep='NA')
"""
