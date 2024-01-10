## Convert .Bed to .HDF5 file  (saving mean, std genotype, phenotypes and chr_map seperately)
import pandas as pd
from pysnptools.snpreader import Bed
import h5py
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import argparse
from pysnptools.distreader import Bgen
import pdb

from preprocess_phenotypes import preprocess_phenotypes, PreparePhenoRHE

## get covariate effect on genotypes and std_genotype
def get_geno_covar_effect(bed, sample_indices, covars, snp_mask, chunk_size=4096):
    snp_on_disk = Bed(bed, count_A1=True)
    snp_on_disk = snp_on_disk[sample_indices, snp_mask]
    chunk_size = min(chunk_size, snp_on_disk.shape[1])
    df_covar = pd.read_csv(covars, sep="\s+")
    df_covar = pd.merge(
        pd.DataFrame(snp_on_disk.iid[:, 0].astype("int"), columns=["FID"]),
        df_covar,
        on=["FID"],
    )
    df_covar = df_covar.loc[:, df_covar.std() > 0]
    df_covar["ALL_CONST"] = 1
    df_covar = df_covar.fillna(df_covar.median())
    covars = df_covar[df_covar.columns[2:]].values
    K = np.linalg.inv(covars.T @ covars)
    geno_covar_effect = np.zeros((covars.shape[1], snp_on_disk.shape[1]))
    xtx = np.zeros(snp_on_disk.shape[1])
    for i in tqdm(range(0, snp_on_disk.shape[1], chunk_size)):
        x = 2 - (
            snp_on_disk[:, i : min(i + chunk_size, snp_on_disk.shape[1])]
            .read(dtype="int8", _require_float32_64=False)
            .val
        )
        x = np.array(x, dtype="float32")
        x[x < 0] = np.nan
        x = np.where(
            np.isnan(x),
            np.nanpercentile(x, 50, axis=0, interpolation="nearest"),
            x,
        )
        temp = covars.T.dot(x)
        geno_covar_effect[:, i : min(i + chunk_size, snp_on_disk.shape[1])] = K @ temp
        xtx[i : min(i + chunk_size, snp_on_disk.shape[1])] = np.array(
            [
                x[:, v].dot(x[:, v]) - temp[:, v].dot(K.dot(temp[:, v])) #### CAUTION!!!
                for v in range(x.shape[1])
            ]
        )
        if (xtx[i : min(i + chunk_size, snp_on_disk.shape[1])] < 0).any():
            print("Check if covariates are independent, the covariate linear regression might be unstable...")
            pdb.set_trace() 
    return covars, geno_covar_effect, np.sqrt(xtx / len(sample_indices))


def convert_to_hdf5(
    bed,
    covars,
    sample_indices,
    out="out",
    snps_to_keep_filename=None,
    chunk_size=4096,
    train_split = 0.8,
    binary=False
):
    ## pheno is the adjusted pheno
    ## sample_indices come from the preprocess_phenotypes
    ## snps_to_keep is a list of SNPs to be included in the analysis
    h1 = h5py.File(out + ".hdf5", 'w') ###caution

    ## handle phenotypes here
    pheno = pd.read_csv(out + ".traits", sep="\s+")
    covareffect = pd.read_csv(out + ".covar_effects", sep="\s+")
    snp_on_disk = Bed(bed, count_A1=True)

    chunk_size = min(chunk_size, snp_on_disk.shape[0])

    ##Count total SNPs
    if snps_to_keep_filename is None:
        total_snps = snp_on_disk.sid_count
        snp_mask = np.ones(total_snps, dtype="bool")
    else:
        snps_to_keep = pd.read_csv(snps_to_keep_filename, sep="\s+")
        snps_to_keep = snps_to_keep[snps_to_keep.columns[0]].values
        snp_dict = {}
        total_snps = snp_on_disk.sid_count
        snp_mask = np.zeros(total_snps, dtype="bool")
        for snp_no, snp in enumerate(snp_on_disk.sid):
            snp_dict[snp] = snp_no
        for snp in snps_to_keep:
            snp_mask[snp_dict[snp]] = True


    covars_arr, geno_covar_effect, std_genotype = get_geno_covar_effect(
        bed, sample_indices, covars, snp_mask, chunk_size=4096
    )
    # save the chromosome information
    _ = h1.create_dataset("chr", data=snp_on_disk.pos[:, 0][snp_mask], dtype=np.int8)

    total_snps = int(sum(snp_mask))
    total_samples = len(sample_indices)
    print("Total samples = " + str(total_samples))

    ## store the PRS / phenotype
    y = pheno[pheno.columns[2:]].values
    z = covareffect[covareffect.columns[2:]].values
    ## handle genotypes here

    ## caution: removed compression
    dset1 = h1.create_dataset(
        "hap1",
        (total_samples, int(np.ceil(total_snps / 8))),
        chunks=(chunk_size, int(np.ceil(total_snps / 8))),
        # compression="gzip",
        dtype=np.uint8,
    )
    dset2 = h1.create_dataset(
        "hap2",
        (total_samples, int(np.ceil(total_snps / 8))),
        chunks=(chunk_size, int(np.ceil(total_snps / 8))),
        # compression="gzip",
        dtype=np.uint8,
    )
    dset3 = h1.create_dataset("phenotype", data=y, dtype=float)
    dset35 = h1.create_dataset("covar_effect", data=z, dtype=float)
    dset4 = h1.create_dataset("geno_covar_effect", data=geno_covar_effect, dtype=float)
    dset5 = h1.create_dataset("std_genotype", data=std_genotype, dtype=float)
    dset55 = h1.create_dataset("covars", data=covars_arr, dtype=float)
    dset6 = h1.create_dataset("sample_indices", data=sample_indices, dtype=int)
    dset7 = h1.create_dataset(
        "iid", data=np.array(snp_on_disk.iid[sample_indices], dtype=int), dtype="int"
    )

    # sum_genotype = np.zeros(total_snps)
    # sum_square_genotype = np.zeros(total_snps)
    for i in tqdm(range(0, total_samples, chunk_size)):
        x = 2 - (
            snp_on_disk[
                sample_indices[i : min(i + chunk_size, total_samples)], snp_mask
            ]
            .read(dtype="int8", _require_float32_64=False)
            .val
        )
        x = np.array(x, dtype="float32")
        x[x < 0] = np.nan
        x = np.where(
            np.isnan(x),
            np.nanpercentile(x, 50, axis=0, interpolation="nearest"),
            x,
        )
        # sum_genotype += np.sum(x, axis=0)
        # sum_square_genotype += np.sum(x**2, axis=0)
        dset1[i : i + x.shape[0]] = np.packbits(np.array(x > 0, dtype=np.int8), axis=1)
        dset2[i : i + x.shape[0]] = np.packbits(np.array(x > 1, dtype=np.int8), axis=1)
        ## np.packbits() requires most time (~ 80%)

    # dset4[:] = sum_genotype / total_samples
    # dset5[:] = np.sqrt(
    #     sum_square_genotype / total_samples - (sum_genotype / total_samples) ** 2
    # )
    h1.close()
    return out + ".hdf5"


def load_bgen_tempfiles(args):
    snp_on_disk = Bgen(args.bgen, sample=args.sample)
    snp_on_disk = snp_on_disk.as_snp(max_weight=2)
    snp_on_disk.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bed", "-g", help="prefix for bed/bim/fam files", type=str, default=None
    )
    parser.add_argument(
        "--output",
        "-o",
        help="prefix for where to save any results or files",
        default=None,
    )
    parser.add_argument(
        "--pheno",
        "-p",
        help='phenotype file; should be in "FID,IID,Trait" format and tsv',
        type=str,
        default=None,
    )
    parser.add_argument(
        "--covar",
        "-c",
        help='file with covariates; should be in "FID,IID,Var1,Var2,..." format and tsv',
        type=str,
    )
    parser.add_argument(
        "--removeFile",
        "-r",
        help='file with sample id to remove; should be in "FID,IID" format and tsv',
        type=str,
    )
    parser.add_argument("--bgen", help="Location to Bgen file", type=str, default=None)
    parser.add_argument(
        "--sample", help="Location to samples file", type=str, default=None
    )
    parser.add_argument(
        "--modelSnps",
        help="Path to list of SNPs to be considered in BLR",
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

    if args.bed is not None and args.pheno is not None:
        Traits, covar_effects, sample_indices = preprocess_phenotypes(
            args.pheno, args.covar, args.bed, args.removeFile, args.binary
        )
        PreparePhenoRHE(Traits, covar_effects, args.bed, args.output, None)

        filename = convert_to_hdf5(
            args.bed, args.covar, sample_indices, args.output, args.modelSnps
        )
    if args.bgen is not None and args.sample is not None:
        load_bgen_tempfiles(args)
