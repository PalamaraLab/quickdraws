import argparse
import numpy as np
import torch
import time
import h5py

from preprocess_phenotypes import preprocess_phenotypes, PreparePhenoRHE
from runRHE import runRHE, MakeAnnotation, runSCORE
from convert_to_hdf5 import convert_to_hdf5
from blr import blr_spike_slab, str_to_bool
from cavi import cavi_on_svi

## main script for the method
# dependency: pysnptools, pytorch, argparse, Path, wandb, qmplot, plink2 and RHE-MCMT

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
    "--removeFile",
    "-r",
    help='file with sample id to remove; should be in "FID,IID" format and tsv',
    type=str,
)
parser.add_argument(
    "--unrel_sample_list",
    help="File with un-related homogenous sample list (FID, IID)",
    type=str,
    default=None,
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
    "--calibrate",
    help="Do you wish to calibrate your test-statistics ?",
    type=str_to_bool,
    default="True",
)
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
    default=0,
)
parser.add_argument(
    "--rhemc",
    type=str,
    default="src/RHEmcmt",
    help="path to RHE-MC / SCORE binary file",
)
parser.add_argument("--hdf5", type=str, help="File name of the hdf5 file to use")
parser.add_argument("--h2_file", type=str, help="File containing estimated h2")

## hyperparameters arguments
parser.add_argument(
    "--num_epochs", help="number of epochs to train for", type=int, default=10
)
parser.add_argument(
    "--lr",
    help="Learning rate of the optimizer",
    type=float,
    nargs="+",
    default=[
        4e-4,
        2e-4,
        5e-5,
        2e-5,
        1e-5,
        5e-6,
    ],  ##changed from [4e-4, 4e-4, 2e-4, 2e-4, 5e-5, 2e-5]
)
# parser.add_argument(
#     "-lr_min",
#     "--min_learning_rate",
#     help="Minimum learning rate for cosine scheduler",
#     type=float,
#     default=1e-5,
# )
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
    "--gpu", help="Which GPU card do you wish to use ?", type=str, default="0"
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
    "--loco",
    help="select the loco scheme",
    type=str,
    choices=["approx", "exact"],
    default="exact",
)
parser.add_argument(
    "--binary",
    help="Is the phenotype binary ?",
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

assert (args.calibrate and args.unrel_sample_list is not None) or (not args.calibrate)
"Provide a list of unrelated homogenous sample if you wish to calibrate"

######      Preprocessing the phenotypes      ######
st = time.time()
print("Preprocessing the phenotypes..")
pheno_covareffect = ".".join([args.output, "traits_covareffects"])
Traits, covar_effects, sample_indices = preprocess_phenotypes(
    args.pheno, args.covar, args.bed, args.removeFile
)
PreparePhenoRHE(Traits, covar_effects, args.bed, pheno_covareffect, None)
print("Done in " + str(time.time() - st) + " secs")

######      Run RHE-MC for h2 estimation      ######
if args.h2_file is None:
    st = time.time()
    print("Calculating heritability estimates using RHE..")
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
        pheno_covareffect + ".rhe",
        args.modelSnps,
        args.annot,
        args.output + ".rhe.log",
        args.rhemc,
        args.covar,
        args.output,
    )
    print("Done in " + str(time.time() - st) + " secs")
else:
    print("Loading heritability estimates from: " + str(args.h2_file))
    VC = np.loadtxt(args.h2_file)

######      Converting .bed to .hdf5          ######
if args.hdf5 is None:
    st = time.time()
    print("Converting Bed file to HDF5..")
    hdf5_filename = convert_to_hdf5(
        args.bed, pheno_covareffect, sample_indices, args.output, args.modelSnps
    )
    print("Done in " + str(time.time() - st) + " secs")
else:
    print("Loading HDF5 file from: " + str(args.hdf5))
    hdf5_filename = args.hdf5
    sample_indices = np.array(h5py.File(hdf5_filename, "r")["sample_indices"])

######      Running variational inference     ######
st = time.time()
print("Running VI using spike and slab prior..")
blr_spike_slab(args, VC, hdf5_filename)
if args.cavi_on_top > 0:
    print("Running a few CAVI steps on top..")
    cavi_on_svi(args, hdf5_filename)

print("Done in " + str(time.time() - st) + " secs")

print("Step 1+2 total Time: " + str(time.time() - overall_st) + " secs")
