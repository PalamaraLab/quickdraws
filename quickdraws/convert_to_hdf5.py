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


# Convert .Bed to .HDF5 file  (saving mean, std genotype, phenotypes and chr_map seperately)
import pandas as pd
from pysnptools.snpreader import Bed
import h5py
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import argparse
from pysnptools.distreader import Bgen
import pdb
import numba
import os

from .preprocess_phenotypes import preprocess_phenotypes, PreparePhenoRHE

import logging
logger = logging.getLogger(__name__)

@numba.jit(nopython=True)
def get_xtx(x, covars, K):
    for snp in numba.prange(x.shape[1]):
        isnan_at_snp = np.isnan(x[:, snp])
        freq = np.median(x[:, snp][~isnan_at_snp])
        x[:, snp][isnan_at_snp] = freq

    temp = covars.T.dot(x)
    geno_covar_effect = K @ temp
    xtx = np.array(
        [
            x[:, v].dot(x[:, v]) - temp[:, v].dot(K.dot(temp[:, v]))
            for v in range(x.shape[1])
        ]
    )
    return geno_covar_effect, xtx

## get covariate effect on genotypes and std_genotype
def get_geno_covar_effect(bed, sample_indices, covars, snp_mask, chunk_size=512, num_threads=1):
    snp_on_disk = Bed(bed, count_A1=True)
    snp_on_disk = snp_on_disk[sample_indices, snp_mask]
    chunk_size = min(chunk_size, snp_on_disk.shape[1])
    if covars is None:
        covars = np.ones((len(sample_indices), 1), dtype='float32')
    else:
        df_covar = pd.read_csv(covars, sep=r'\s+')
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
    for i in range(0, snp_on_disk.shape[1], chunk_size):
        x = 2 - (
            snp_on_disk[:, i : min(i + chunk_size, snp_on_disk.shape[1])]
            .read(dtype="float64", num_threads=num_threads)
            .val
        )
        geno_covar_effect_numba, xtx_numba = get_xtx(x, covars, K)
        xtx_numba[np.std(x, axis=0) == 0] = 0  ## set fixed variants to have 0 std (o/w have small -ve values due to numerical issues in numba)
        xtx[i : min(i + chunk_size, snp_on_disk.shape[1])] = xtx_numba
        geno_covar_effect[:, i : min(i + chunk_size, snp_on_disk.shape[1])] = geno_covar_effect_numba
        if (xtx_numba < 0).any():
            logging.exception("Check if covariates are independent, the covariate linear regression might be unstable...")
            raise ValueError
    return covars, geno_covar_effect, np.sqrt(xtx / len(sample_indices))

def convert_to_hdf5(
    bed,
    covars,
    sample_indices,
    out="out",
    snps_to_keep_filename=None,
    master_hdf5=None,
    chunk_size=4096,
):
    num_threads = len(os.sched_getaffinity(0))
    h1 = h5py.File(out + ".hdf5", 'w') ###caution

    ## handle phenotypes here
    pheno = pd.read_csv(out + ".traits", sep=r'\s+')
    covareffect = pd.read_csv(out + ".covar_effects", sep=r'\s+')
    snp_on_disk = Bed(bed, count_A1=True)

    ##Count total SNPs
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

    if master_hdf5 is not None:
        master_hdf5 = h5py.File(master_hdf5, 'r')
        bed_fid_iid = pd.DataFrame(snp_on_disk.iid[sample_indices], columns=['FID','IID'])
        hdf5_fid_iid = pd.DataFrame(master_hdf5['iid'][:].astype(str), columns=['FID','IID'])
        hdf5_fid_iid['index'] = hdf5_fid_iid.index
        mdf = pd.merge(bed_fid_iid, hdf5_fid_iid, on=['FID','IID'])
        if len(mdf) == len(bed_fid_iid):
            if (master_hdf5['sid'][:].astype(str) == snp_on_disk[:, snp_mask].sid).all():
                sample_order = mdf['index'].values
                logging.info("Found all SNPs from bed file in prespecified HDF5 file, using HDF5 file")
            else:
                logging.info("Didn't Find all SNPs from bed file in prespecified HDF5 file, using Bed file")
                master_hdf5.close()
                master_hdf5 = None
        else:
            logging.info("Didn't Find all samples from bed file in prespecified HDF5 file, using Bed file")
            master_hdf5.close()
            master_hdf5 = None
        

    chunk_size = min(chunk_size, snp_on_disk.shape[0])

    logging.info("Estimating variance per allele...")

    covars_arr, geno_covar_effect, std_genotype = get_geno_covar_effect(
        bed, sample_indices, covars, snp_mask, chunk_size=512, num_threads=num_threads
    )
    _ = h1.create_dataset("chr", data=snp_on_disk.pos[:, 0][snp_mask], dtype=np.int8)
    total_snps = int(sum(snp_mask))

    total_samples = len(sample_indices)
    logging.info("Total number of samples in HDF5 file = " + str(total_samples))

    ## store the PRS / phenotype
    y = pheno[pheno.columns[2:]].values
    z = covareffect[covareffect.columns[2:]].values

    ## caution: removed compression
    dset1 = h1.create_dataset(
        "hap1",
        (total_samples, int(np.ceil(total_snps / 8))),
        chunks=(chunk_size, int(np.ceil(total_snps / 8))),
        dtype=np.uint8,
    )
    dset2 = h1.create_dataset(
        "hap2",
        (total_samples, int(np.ceil(total_snps / 8))),
        chunks=(chunk_size, int(np.ceil(total_snps / 8))),
        dtype=np.uint8,
    )
    _ = h1.create_dataset("pheno_names", data=pheno.columns[2:].tolist())
    dset3 = h1.create_dataset("phenotype", data=y, dtype=float)
    dset35 = h1.create_dataset("covar_effect", data=z, dtype=float)
    dset4 = h1.create_dataset("geno_covar_effect", data=geno_covar_effect, dtype=float)
    dset5 = h1.create_dataset("std_genotype", data=std_genotype, dtype=float)
    dset55 = h1.create_dataset("covars", data=covars_arr, dtype=float)
    dset6 = h1.create_dataset("sample_indices", data=sample_indices, dtype=int)
    dset7 = h1.create_dataset(
        "iid", data=np.array(snp_on_disk.iid[sample_indices], dtype=int), dtype="int"
    )

    logging.info("Saving the genotype to HDF5 file...")
    for i in range(0, total_samples, chunk_size):
        logging.info(str(i))
        if master_hdf5 is None:
            x = 2 - (
                snp_on_disk[
                    sample_indices[i : min(i + chunk_size, total_samples)], snp_mask
                ]
                .read(dtype="int8", _require_float32_64=False, num_threads=num_threads)
                .val
            )
            x = np.array(x, dtype="float32")
            x[x < 0] = np.nan
            x = np.where(
                np.isnan(x),
                np.nanpercentile(x, 50, axis=0, interpolation="nearest"),
                x,
            )
            dset1[i : i + x.shape[0]] = np.packbits(np.array(x > 0, dtype=np.int8), axis=1)
            dset2[i : i + x.shape[0]] = np.packbits(np.array(x > 1, dtype=np.int8), axis=1)
            ## np.packbits() requires most time (~ 80%)
        else:
            for index in np.argsort(sample_order[i : min(i + chunk_size, total_samples)]):
                pos = sample_order[i : min(i + chunk_size, total_samples)][index]
                dset1[i + index] = master_hdf5['hap1'][pos]
                dset2[i + index] = master_hdf5['hap2'][pos]

    h1.close()
    logging.info("Done saving the genotypes to hdf5 file " + str(out + '.hdf5'))
    
    if master_hdf5 is not None:
        master_hdf5.close()

    return out + ".hdf5"


def make_master_hdf5(
    bed,
    out="out",
    snps_to_keep_filename=None,
    chunk_size=4096,
):
    num_threads = len(os.sched_getaffinity(0))
    h1 = h5py.File(out + ".hdf5", 'w') ###caution
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

    snp_on_disk = snp_on_disk[:, snp_mask]
    chunk_size = min(chunk_size, snp_on_disk.shape[0])

    _ = h1.create_dataset("iid", data=np.array(snp_on_disk.iid, dtype='S'))
    _ = h1.create_dataset("sid", data=np.array(snp_on_disk.sid, dtype='S'))

    total_snps = int(sum(snp_mask))
    total_samples = snp_on_disk.shape[0]

    dset1 = h1.create_dataset(
        "hap1",
        (total_samples, int(np.ceil(total_snps / 8))),
        chunks=(chunk_size, int(np.ceil(total_snps / 8))),
        dtype=np.uint8,
    )
    dset2 = h1.create_dataset(
        "hap2",
        (total_samples, int(np.ceil(total_snps / 8))),
        chunks=(chunk_size, int(np.ceil(total_snps / 8))),
        dtype=np.uint8,
    )
    logging.info("Saving the genotype to HDF5 file...")
    for i in range(0, total_samples, chunk_size):
        logging.info(str(i))
        x = 2 - (
            snp_on_disk[
                i : min(i + chunk_size, total_samples), snp_mask
            ]
            .read(dtype="int8", _require_float32_64=False, num_threads=num_threads)
            .val
        )
        x = np.array(x, dtype="float32")
        x[x < 0] = np.nan
        x = np.where(
            np.isnan(x),
            np.nanpercentile(x, 50, axis=0, interpolation="nearest"),
            x,
        )
        dset1[i : i + x.shape[0]] = np.packbits(np.array(x > 0, dtype=np.int8), axis=1)
        dset2[i : i + x.shape[0]] = np.packbits(np.array(x > 1, dtype=np.int8), axis=1)

    h1.close()
    logging.info("Done saving the genotypes to hdf5 file " + str(out + '.hdf5'))
    return out + ".hdf5"    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bed", "-g", help="prefix for bed/bim/fam files", type=str, default=None
    )
    parser.add_argument(
        "--out",
        "-o",
        help="prefix for where to save any results or files",
        default=None,
    )
    parser.add_argument(
        "--phenoFile",
        "-p",
        help='phenotype file; should be in "FID,IID,Trait" format and tsv',
        type=str,
        default=None,
    )
    parser.add_argument(
        "--covarFile",
        "-c",
        help='file with covariates; should be in "FID,IID,Var1,Var2,..." format and tsv',
        type=str,
    )
    parser.add_argument(
        "--keepFile",
        "-r",
        help='file with sample id to keep; should be in "FID,IID" format and tsv',
        type=str,
    )
    parser.add_argument(
        "--modelSnps",
        help="Path to list of SNPs to be considered in model fitting",
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
    parser.add_argument("--hdf5", help="master hdf5 file which stores genotype matrix in binary format", type=str)
    args = parser.parse_args()

    if args.bed is not None and args.phenoFile is not None:
        Traits, covar_effects, sample_indices = preprocess_phenotypes(
            args.phenoFile, args.covarFile, args.bed, args.keepFile, args.binary
        )
        PreparePhenoRHE(Traits, covar_effects, args.bed, args.out, None)
        # np.arange(405088)
        filename = convert_to_hdf5(
            args.bed, args.covarFile, sample_indices, args.out, args.modelSnps, args.hdf5
        )
    elif args.bed is not None:
        make_master_hdf5(args.bed, args.out, args.modelSnps)