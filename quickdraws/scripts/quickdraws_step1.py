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


import argparse
import os
import numpy as np
import torch
import time
import h5py
import logging
from datetime import datetime
import warnings
from pathlib import Path
import pandas as pd 
from pysnptools.snpreader import Bed

import quickdraws.scripts
from quickdraws import (
    preprocess_phenotypes,
    PreparePhenoRHE,
    runRHE,
    MakeAnnotation,
    convert_to_hdf5,
    blr_spike_slab,
    str_to_bool
)

from quickdraws.scripts import get_copyright_string


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def main():

    overall_st = time.time()
    ######      Setting the random seeds         ######
    np.random.seed(2)
    torch.manual_seed(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2)

    ######      Parsing the input arguments       ######
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", "-g", help="prefix for bed/bim/fam files", type=str)
    parser.add_argument(
        "--covarFile",
        "-c",
        help='file with covariates; should be in "FID,IID,Var1,Var2,..." format and tsv',
        type=str,
    )
    parser.add_argument(
        "--phenoFile",
        "-p",
        help='phenotype file; should be in "FID,IID,Trait" format and tsv',
        type=str,
    )
    parser.add_argument(
        "--keepFile",
        "-r",
        help='file with sample id to keep; should be in "FID,IID" format and tsv',
        type=str,
    )
    parser.add_argument(
        "--out",
        "-o",
        help="prefix for where to save any results or files",
        default="out",
    )
    parser.add_argument(
        "--annot",
        help="file with annotation; one column per component; no overlapping",
        type=str,
    )
    parser.add_argument(
        "--kinship",
        help="King table file which stores relative information of upto 3rd degree relatives (tab-seperated and has ID1 ID2 Kinship as columns)",
        type=str,
    )
    parser.add_argument(
        "--ldscores",
        help="Path to ldscores file (should have MAF and LDSCORE columns and tab-seperated)",
        type=str,
        default=None
    )
    parser.add_argument(
        "--modelSnps",
        help="Path to list of SNPs to be considered in model fitting",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--rhemc",
        type=str,
        default=str(Path(os.path.dirname(os.path.abspath(__file__)),"GENIE_multi_pheno")),
        help="path to RHE-MCMT binary file",
    )
    parser.add_argument("--out_step0", help="prefix of the output files from step 0", type=str) ## depreciate
    parser.add_argument("--hdf5", help="master hdf5 file which stores genotype matrix in binary format", type=str)
    parser.add_argument("--h2_file", type=str, help="File containing estimated h2")
    parser.add_argument(
        "--h2_grid",
        help="grid search for h2 instead",
        action="store_const",
        const=True,
        default=False,
    )
    ## hyperparameters arguments
    parser.add_argument(
        "--num_epochs", help="number of epochs to train loco run", type=int, default=40
    )
    parser.add_argument(
        "--alpha_search_epochs", help="number of epochs to train for alpha search", type=int, default=80
    )
    parser.add_argument(
        "--validate_every", help="How often do you wanna validate the whole genomre regression (default = -1, which means never)", type=int, default=-1
    )
    parser.add_argument(
        "--lr",
        help="Learning rate of the optimizer",
        type=float,
        nargs="+",
        default=[
            4e-4,
            2e-4,
            2e-4,
            1e-4,
            2e-5,
            5e-6,
        ],
    )
    parser.add_argument(
        "--alpha",
        help="Sparsity grid for Bayesian linear regression",
        type=float,
        nargs="+",
        default=[
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
        ],
    )
    parser.add_argument(
        "-scheduler",
        "--cosine_scheduler",
        help="Cosine scheduling the outer learning rate",
        type=str_to_bool,
        default="false",
    )
    parser.add_argument(
        "--batch_size", help="Batch size of the dataloader", type=int, default=128
    )
    parser.add_argument(
        "--forward_passes", help="Number of forward passes in blr", type=int, default=1
    )
    parser.add_argument(
        "--num_workers",
        help="torch.utils.data.DataLoader num_workers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--train_split",
        help="The training split proportion in (0,1)",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--binary",
        help="Is the phenotype binary ?",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--lowmem",
        help="Enable low memory version",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--rhe_random_vectors",
        help="Number of random vectors in RHE MC",
        type=int,
        default=50
    )
    parser.add_argument(
        "--rhe_jn",
        help="Number of jack-knife partitions in RHE MC",
        type=int,
        default=10
    )
    parser.add_argument(
        "--phen_thres",
        help="The phenotyping rate threshold below which the phenotype isn't used to perform GWAS",
        type=float,
        default=0
    )
    parser.add_argument(
        "--predBetasFlag",
        help="Indicate if you want to calculate and store the BLUP betas",
        action="store_const",
        const=True,
        default=False,
    )
    ## wandb arguments
    wandb_group = parser.add_argument_group("WandB")
    wandb_mode = wandb_group.add_mutually_exclusive_group()
    wandb_mode.add_argument(
        "--wandb_mode",
        default="disabled",
        help="mode for wandb logging, useful while debugging",
    )
    wandb_group.add_argument(
        "--wandb_entity_name",
        help="wandb entity name (usualy github ID)",
    )
    wandb_group.add_argument(
        "--wandb_project_name",
        help="wandb project name",
        default="blr_genetic_association",
    )
    wandb_group.add_argument(
        "--wandb_job_type",
        help="Wandb job type. This is useful for grouping runs together.",
        default=None,
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

    ######      Preprocessing the phenotypes      ######
    st = time.time()
    logging.info("#### Start Time: " + str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')) + " ####")
    logging.info("")

    warnings.simplefilter("ignore")

    make_sure_path_exists(args.out)

    if args.out_step0 is not None:
        logging.info("#### Step 1a. Using preprocessed phenotype and hdf5 files ####")
        rhe_out = args.out_step0
        hdf5_filename = args.out_step0 + ".hdf5"
        sample_indices = np.array(h5py.File(hdf5_filename, "r")["sample_indices"])
        logging.info("#### Step 1a. Done in " + str(time.time() - st) + " secs ####")
        logging.info("")
    else:
        logging.info("#### Step 1a. Preprocessing the phenotypes and converting bed to hdf5 ####")
        rhe_out = args.out
        Traits, covar_effects, sample_indices = preprocess_phenotypes(
            args.phenoFile, args.covarFile, args.bed, args.keepFile, args.binary, args.hdf5, args.phen_thres
        )
        PreparePhenoRHE(Traits, covar_effects, args.bed, rhe_out, None)
        hdf5_filename = convert_to_hdf5(
            args.bed,
            args.covarFile,
            sample_indices,
            args.out,
            args.modelSnps,
            args.hdf5
        )
        logging.info("#### Step 1a. Done in " + str(time.time() - st) + " secs ####")
        logging.info("")

    ######      Run RHE-MC for h2 estimation      ######
    if args.h2_file is None and not args.h2_grid:
        st = time.time()
        logging.info("#### Step 1b. Calculating heritability estimates using RHE ####")
        args.annot = args.out + ".maf2_ld4.annot"
        MakeAnnotation(
            args.bed,
            args.ldscores,
            args.modelSnps,
            [0.01, 0.05, 0.5],
            [0.0, 0.25, 0.5, 0.75, 1.0],
            args.annot,
        )
        VC = runRHE(
            args.bed,
            rhe_out + ".rhe",
            args.modelSnps,
            args.annot,
            args.out + ".rhe.log",
            args.rhemc,
            args.covarFile,
            args.out,
            args.binary,
            args.rhe_random_vectors,
            args.rhe_jn
        )
        logging.info("#### Step 1b. Done in " + str(time.time() - st) + " secs ####")
        logging.info("")
    elif args.h2_file is not None:
        st = time.time()
        logging.info("#### Step 1b. Loading heritability estimates from: " + str(args.h2_file) + " ####")
        logging.info("")
        VC = np.loadtxt(args.h2_file)
        logging.info("#### Step 1b. Done in " + str(time.time() - st) + " secs ####")
        logging.info("")
    else:
        st = time.time()
        logging.info("#### Step 1b. Using h2_grid and performing a grid search in BLR ####")
        logging.info("")
        VC = None
        logging.info("#### Step 1b. Done in " + str(time.time() - st) + " secs ####")
        logging.info("")


    ######      Running variational inference     ######
    st = time.time()
    logging.info("#### Step 1c. Running VI using spike and slab prior ####")
    if torch.cuda.is_available():
        logging.info("Using GPU to run variational inference!!")
        logging.info("")
        device = 'cuda'
    else:
        logging.info("Didn't find any GPU, using CPU to run variational inference... expect very slow multiplications")
        logging.info("")
        device = 'cpu'
    beta = blr_spike_slab(args, VC, hdf5_filename, device)
    logging.info("#### Step 1c. Done in " + str(time.time() - st) + " secs ####")
    logging.info("")

    logging.info("Saved LOCO predictions per phenotype as: ")
    with h5py.File(hdf5_filename,'r') as f:
        pheno_names = f['pheno_names'][:]
    with open(args.out + "_pred.list" , 'w') as f:
        for i, pheno_name in enumerate(pheno_names):
            f.write(pheno_name.decode() + " " + str(Path(args.out).resolve()) + "_" + str(i+1) + ".loco \n")
            logging.info(pheno_name.decode() + " : " + str(Path(args.out).resolve()) + "_" + str(i+1) + ".loco")
    logging.info("")
    logging.info("LOCO prediction locations per phenotype saved as: " + str(args.out + '_pred.list'))
    logging.info("")

    logging.info("Saved h2 estimates per phenotype as: " + str(args.out + '.h2'))
    logging.info("")
    logging.info("Saved sparsity estimates per phenotype as: " + str(args.out + '.alpha'))
    logging.info("")
    if args.predBetasFlag:
        snp_on_disk = Bed(args.bed, count_A1=True)
        if args.modelSnps is None:
            total_snps = snp_on_disk.sid_count
            snp_mask = np.ones(total_snps, dtype="bool")
        else:
            snps_to_keep = pd.read_csv(args.modelSnps, sep=r'\s+')
            snps_to_keep = snps_to_keep[snps_to_keep.columns[0]].values
            snp_dict = {}
            total_snps = snp_on_disk.sid_count
            snp_mask = np.zeros(total_snps, dtype="bool")
            for snp_no, snp in enumerate(snp_on_disk.sid):
                snp_dict[snp] = snp_no
            for snp in snps_to_keep:
                snp_mask[snp_dict[snp]] = True

        snp_on_disk = snp_on_disk[:, snp_mask]
        df = pd.DataFrame(columns = ['CHR','GENPOS','POS', 'SNP','BETA'])
        df['CHR'] = snp_on_disk.pos[:, 0]
        df['GENPOS'] = snp_on_disk.pos[:, 1]
        df['POS'] = snp_on_disk.pos[:, 2]
        df['SNP'] = snp_on_disk.sid
        bim = pd.read_csv(
            args.bed + ".bim",
            sep=r'\s+',
            header=None,
            names=["CHR", "SNP", "GENPOS", "POS", "A1", "A2"],
        )
        bim = bim[['CHR','SNP','A1','A2']]
        df = pd.merge(df, bim, on=['CHR','SNP'])
        print(df.shape)
        for d, pheno_name in enumerate(pheno_names):
            df['BETA'] = beta[d]
            df.to_csv(args.out + '_' + pheno_name.decode() + '.blup', sep='\t', index=None, na_rep='NA')
        logging.info("Saved BLUP betas per phenotype as: ")
        for i, pheno_name in enumerate(pheno_names):
            logging.info(pheno_name.decode() + " : " + str(Path(args.out).resolve()) + "_" + pheno_name.decode() + ".blup")
        logging.info("")

    logging.info("#### Step 1 total Time: " + str(time.time() - overall_st) + " secs ####")
    logging.info("")
    logging.info("#### End Time: " + str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')) + " ####")
    logging.info("")


if __name__ == "__main__":
    main()
