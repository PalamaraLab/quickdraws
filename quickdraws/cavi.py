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


import os
from pysnptools.snpreader import Bed
import pandas as pd
import numpy as np
import h5py
import argparse
from joblib import Parallel, delayed


def preprocess_bedfile(bedFile, sample_indices, snps_to_keep_filename, chr_map=None):
    snp_on_disk = Bed(bedFile, count_A1=True)
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

    snp_on_disk = snp_on_disk[sample_indices, snp_mask]
    if chr_map is not None:
        snp_on_disk = snp_on_disk[:, chr_map]

    return snp_on_disk


## CAVI updates for spike-and-slab prior
def update_step_sparse(
    y,
    snp_on_disk,
    sum_x_sq,
    y_resid,
    sigma_sq_e,
    sigma_sq_1,
    p_0,
    vi_mu,
    vi_psi,
):
    ## Assumes snp_on_disk is an N X P input data matrix
    ## Assumes y is PxN matrix containing the ground truth
    ## Assumes y_resid is PxN matrix containing the residuals after an iteration of CAVI

    if y_resid is None:
        y_resid = np.copy(y)
        for i in range(vi_mu.shape[1]):
            x_i = snp_on_disk[:, i].read(dtype="float32").val.flatten()
            y_resid -= np.outer(vi_mu[:, i] * (1 - vi_psi[:, i]), x_i)  ##matrix prod

    new_mu = np.copy(vi_mu)
    new_psi = np.copy(vi_psi)

    for i in range(vi_mu.shape[1]):
        x_i = snp_on_disk[:, i].read(dtype="float32").val.flatten()
        ## subtracting the current SNP's effect estimate to get y_resid_j
        y_resid_j = y_resid + np.outer(new_mu[:, i] * (1 - new_psi[:, i]), x_i)

        this_mu_num = np.sum(y_resid_j * x_i, axis=1)  ## add axis
        this_mu = this_mu_num / (sigma_sq_e / sigma_sq_1[:, i] + sum_x_sq[i])
        new_mu[:, i] = this_mu

        psi_num = (
            p_0
            / (1 - p_0)
            * np.sqrt(1 + sum_x_sq[i] * sigma_sq_1[:, i] / sigma_sq_e)
            * np.exp(
                -0.5
                * (this_mu_num) ** 2
                / (sigma_sq_e**2 / sigma_sq_1[:, i] + sigma_sq_e * sum_x_sq[i])
            )
        )
        new_psi[:, i] = psi_num / (1 + psi_num)

        ## updating y_resid with the current effect estimates for SNP i:
        y_resid = y_resid_j - np.outer(new_mu[:, i] * (1 - new_psi[:, i]), x_i)

    return new_mu, new_psi, y_resid


def cavi_spike_slab(
    args,
    h2,
    alpha,
    hdf5_filename,
    vi_mu_inp=None,
    vi_psi_inp=None,
    chr_map=None,
):
    ## load and preprocess bed file
    sample_indices = np.array(h5py.File(hdf5_filename, "r")["sample_indices"])
    snp_on_disk = preprocess_bedfile(args.bed, sample_indices, args.modelSnps, chr_map)
    pheno = np.array(h5py.File(hdf5_filename, "r")["phenotype"]).T  ## PxN

    ## initialize
    std_genotype = np.array(h5py.File(hdf5_filename, "r")["std_genotype"])
    mean_genotype = np.array(h5py.File(hdf5_filename, "r")["mean_genotype"])
    if chr_map is not None:
        std_genotype = std_genotype[chr_map]
        mean_genotype = mean_genotype[chr_map]

    sigma_sq_e = 1 - h2
    p_zero = 1 - alpha
    sigma_sq_1 = np.outer(
        (1 - sigma_sq_e) / snp_on_disk.shape[1] / (1 - p_zero),
        1 / std_genotype**2,
    )
    sum_x_sq = (std_genotype**2) + (mean_genotype**2)
    sum_x_sq *= snp_on_disk.shape[0]
    if vi_mu_inp is None and vi_psi_inp is None:
        vi_psi = np.ones((len(pheno), snp_on_disk.shape[1]))
        vi_mu = np.zeros((len(pheno), snp_on_disk.shape[1]))
    else:
        vi_psi = vi_psi_inp
        vi_mu = vi_mu_inp

    y_resid = None
    ## run cavi for iterations
    for iter in range(args.cavi_on_top):
        vi_mu, vi_psi, y_resid = update_step_sparse(
            pheno,
            snp_on_disk,
            sum_x_sq,
            y_resid,
            sigma_sq_e,
            sigma_sq_1,
            p_zero,
            vi_mu=vi_mu,
            vi_psi=vi_psi,
        )

    return y_resid


def cavi_on_svi_chr(args, h2, alpha, hdf5_filename, chr_map, c):
    mu_blr = pd.read_csv(
        args.output + "loco_chr" + str(int(c)) + ".mu", sep=r'\s+'
    ).values
    psi_blr = pd.read_csv(
        args.output + "loco_chr" + str(int(c)) + ".psi", sep=r'\s+'
    ).values
    y_resid = cavi_spike_slab(
        args, h2, alpha, hdf5_filename, mu_blr, psi_blr, chr_map != c
    )
    pd.DataFrame(y_resid.T).to_csv(
        args.output + "loco_chr" + str(int(c)) + ".residuals", sep="\t"
    )


def cavi_on_svi(args, hdf5_filename, num_threads=-1):
    if num_threads == -1:
        num_threads = os.cpu_count() - 1

    chr_map = np.array(h5py.File(hdf5_filename, "r")["chr"])
    alpha = np.loadtxt(args.output + ".alpha")
    h2 = np.loadtxt(args.output + ".h2")
    Parallel(n_jobs=num_threads)(
        delayed(cavi_on_svi_chr)(args, h2, alpha, hdf5_filename, chr_map, c)
        for c in np.unique(chr_map)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", "-g", help="prefix for bed/bim/fam files", type=str)
    parser.add_argument(
        "--modelSnps",
        help="Path to list of SNPs to be considered in BLR",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--cavi_on_top",
        type=int,
        help="Runs a few cavi steps on top for better results",
        default=5,
    )
    parser.add_argument("--output", help="Filename of the residuals file", type=str)
    args = parser.parse_args()

    hdf5_filename = "output/qd.hdf5"
    cavi_on_svi(args, hdf5_filename)

# python src/cavi.py --bed example/example --output output/qd
