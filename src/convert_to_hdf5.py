## Convert .Bed to .HDF5 file  (saving mean, std genotype, phenotypes and chr_map seperately)
import pandas as pd
from pysnptools.snpreader import Bed
import h5py
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import argparse
from pysnptools.distreader import Bgen

from preprocess_phenotypes import preprocess_phenotypes, PreparePhenoRHE


def convert_to_hdf5(
    bed,
    pheno,
    sample_indices,
    out="out",
    snps_to_keep_filename=None,
    chunk_size=4096,
):
    ## pheno is the adjusted pheno
    ## sample_indices come from the preprocess_phenotypes
    ## snps_to_keep is a list of SNPs to be included in the analysis
    h1 = h5py.File(out + ".hdf5", "w")

    ## handle phenotypes here
    phenotype = pd.read_csv(pheno, sep="\s+")
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

    # save the chromosome information
    _ = h1.create_dataset("chr", data=snp_on_disk.pos[:, 0][snp_mask], dtype=np.int8)

    total_snps = int(sum(snp_mask))
    total_samples = len(sample_indices)
    print("Total samples = " + str(total_samples))

    ## store the PRS / phenotype
    y = phenotype[phenotype.columns[2:]].values
    ## handle genotypes here
    dset1 = h1.create_dataset(
        "hap1",
        (total_samples, int(np.ceil(total_snps / 8))),
        chunks=(chunk_size, int(np.ceil(total_snps / 8))),
        compression="gzip",
        dtype=np.uint8,
    )
    dset2 = h1.create_dataset(
        "hap2",
        (total_samples, int(np.ceil(total_snps / 8))),
        chunks=(chunk_size, int(np.ceil(total_snps / 8))),
        compression="gzip",
        dtype=np.uint8,
    )
    dset3 = h1.create_dataset("phenotype", data=y, dtype=float)
    dset4 = h1.create_dataset("mean_genotype", (total_snps,), dtype=float)
    dset5 = h1.create_dataset("std_genotype", (total_snps,), dtype=float)
    dset6 = h1.create_dataset("sample_indices", data=sample_indices, dtype=int)

    sum_genotype = np.zeros(total_snps)
    sum_square_genotype = np.zeros(total_snps)
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
        sum_genotype += np.sum(x, axis=0)
        sum_square_genotype += np.sum(x**2, axis=0)
        dset1[i : i + x.shape[0]] = np.packbits(np.array(x > 0, dtype=np.int8), axis=1)
        dset2[i : i + x.shape[0]] = np.packbits(np.array(x > 1, dtype=np.int8), axis=1)
        ## np.packbits() requires most time (~ 80%)

    dset4[:] = sum_genotype / total_samples
    dset5[:] = np.sqrt(
        sum_square_genotype / total_samples - (sum_genotype / total_samples) ** 2
    )
    h1.close()
    return out + ".hdf5"


def load_bgen_tempfiles(args):
    snp_on_disk = Bgen(args.bgen, sample=args.sample)
    snp_on_disk = snp_on_disk.as_snp(max_weight=2)
    snp_on_disk.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", "-g", help="prefix for bed/bim/fam files", type=str)
    parser.add_argument(
        "--output", "-o", help="prefix for where to save any results or files"
    )
    parser.add_argument(
        "--pheno",
        "-p",
        help='phenotype file; should be in "FID,IID,Trait" format and tsv',
        type=str,
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
    parser.add_argument("--bgen", help="Location to Bgen file", type=str)
    parser.add_argument("--sample", help="Location to samples file", type=str)

    args = parser.parse_args()

    adj_pheno_file = ".".join([args.output, "adjusted_traits", "phen"])
    Traits, sample_indices = preprocess_phenotypes(
        args.pheno, args.covar, args.bed, args.removeFile
    )
    PreparePhenoRHE(Traits, args.bed, adj_pheno_file, None)

    filename = convert_to_hdf5(args.bed, adj_pheno_file, sample_indices, args.output)
    load_bgen_tempfiles(args)
