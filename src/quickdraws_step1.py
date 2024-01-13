import argparse
import numpy as np
import torch
import time
import h5py
import logging
from datetime import datetime
from art import text2art
import warnings

from preprocess_phenotypes import preprocess_phenotypes, PreparePhenoRHE
from runRHE import runRHE, MakeAnnotation, runSCORE
from convert_to_hdf5 import convert_to_hdf5
from blr import blr_spike_slab, str_to_bool

overall_st = time.time()
######      Setting the random seeds         ######
np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed_all(2)

######      Parsing the input arguments       ######
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
    "--keepFile",
    "-r",
    help='file with sample id to keep; should be in "FID,IID" format and tsv',
    type=str,
)
parser.add_argument(
    "--output",
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
    "--make_annot",
    help="Do you wish to make the annot file ?",
    type=str_to_bool,
    default="false",
)
parser.add_argument(
    "--modelSnps",
    help="Path to list of SNPs to be considered in BLR",
    default=None,
    type=str,
)
parser.add_argument(
    "--rhemc",
    type=str,
    default="src/RHEmcmt",
    help="path to RHE-MC / SCORE binary file",
)
parser.add_argument("--output_step0", help="prefix of the output files from step 0", type=str) ## depreciate
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
    "--num_epochs", help="number of epochs to train loco run", type=int, default=30
)
parser.add_argument(
    "--alpha_search_epochs", help="number of epochs to train for alpha search", type=int, default=90
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
    default=8,
)
parser.add_argument(
    "--train_split",
    help="The training split proportion in (0,1)",
    type=float,
    default=0.8,
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
    "--phen_thres",
    help="The phenotyping rate threshold below which the phenotype isn't used to perform GWAS",
    type=int,
    default=0
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
        logging.FileHandler(args.output + ".log"),
        logging.StreamHandler()
    ]
)

logging.info(text2art("Quickdraws"))
logging.info("Copyright (c) 2024 Hrushikesh Loya and Pier Palamara.")
logging.info("Distributed under the GPLv3 License.")
logging.info("")
logging.info("Logs saved in: " + str(args.output + ".step1.log"))
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

if args.output_step0 is not None:
    logging.info("#### Step 1a. Using preprocessed phenotype and hdf5 files ####")
    rhe_output = args.output_step0
    hdf5_filename = args.output_step0 + ".hdf5"
    sample_indices = np.array(h5py.File(hdf5_filename, "r")["sample_indices"])
    logging.info("#### Step 1a. done in " + str(time.time() - st) + " secs ####")
    logging.info("")
else:
    logging.info("#### Step 1a. Preprocessing the phenotypes and converting bed to hdf5 ####")
    rhe_output = args.output
    Traits, covar_effects, sample_indices = preprocess_phenotypes(
        args.pheno, args.covar, args.bed, args.keepFile, args.binary, args.phen_thres
    )
    PreparePhenoRHE(Traits, covar_effects, args.bed, rhe_output, None)
    hdf5_filename = convert_to_hdf5(
        args.bed,
        args.covar,
        sample_indices,
        args.output,
        args.modelSnps,
    )
    logging.info("#### Step 1a. done in " + str(time.time() - st) + " secs ####")
    logging.info("")

######      Run RHE-MC for h2 estimation      ######
if args.h2_file is None and not args.h2_grid:
    st = time.time()
    logging.info("#### Step 1b. Calculating heritability estimates using RHE ####")
    if args.make_annot:
        args.annot = args.output + ".maf2_ld4.annot"
        MakeAnnotation(
            args.bed,
            [0.01, 0.05, 0.5],
            [0.0, 0.25, 0.5, 0.75, 1.0],
            args.annot,
        )
    VC = runRHE(
        args.bed,
        rhe_output + ".rhe",
        args.modelSnps,
        args.annot,
        args.output + ".rhe.log",
        args.rhemc,
        args.covar,
        args.output,
        args.binary,
        args.rhe_random_vectors,
    )
    logging.info("#### Step 1b. done in " + str(time.time() - st) + " secs ####")
    logging.info("")
elif args.h2_file is not None:
    st = time.time()
    logging.info("#### Step 1b. Loading heritability estimates from: " + str(args.h2_file) + " ####")
    logging.info("")
    VC = np.loadtxt(args.h2_file)
    logging.info("#### Step 1b. done in " + str(time.time() - st) + " secs ####")
    logging.info("")
else:
    st = time.time()
    logging.info("#### Step 1b. Using h2_grid : {0} and performing a grid search in BLR ####".format(args.h2_grid))
    logging.info("")
    VC = None
    logging.info("#### Step 1b. done in " + str(time.time() - st) + " secs ####")
    logging.info("")


######      Running variational inference     ######
st = time.time()
logging.info("#### Step 1c. Running VI using spike and slab prior ####")
blr_spike_slab(args, VC, hdf5_filename)
logging.info("#### Step 1c. done in " + str(time.time() - st) + " secs ####")
logging.info("")
logging.info("#### Step 1 total Time: " + str(time.time() - overall_st) + " secs ####")
logging.info("")
logging.info("#### End Time: " + str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')) + " ####")
logging.info("")