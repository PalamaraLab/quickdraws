#!/bin/bash

bed="example/example"
pheno="example/phenotypes.txt"
covar="example/covariates.txt"
bgen="example/example.bgen"
sample="example/example.sample"

## step 0: create HDF5 file containing genotype and phenotype
python src/convert_to_hdf5.py \
    --output output/qd \
    --bed ${bed} \
    --pheno ${pheno} \
    --covar ${covar} \
    --bgen ${bgen} \
    --sample ${sample}

# step 1: run model fitting (step 1) on genotypes and phenotypes
python src/quickdraws_step1.py \
    --output output/qd \
    --bed ${bed} \
    --pheno ${pheno} \
    --hdf5 output/qd.hdf5 \
    --covar ${covar} \
    --unrel_sample_list example/unrelated_FID_IID.txt 

## step 2: get association stats for SNPs in bgen file
python src/quickdraws_step2.py \
    --output output/qd \
    --bed ${bed} \
    --output_step1 output/qd \
    --hdf5 output/qd.hdf5 \
    --covar ${covar} \
    --unrel_sample_list example/unrelated_FID_IID.txt \

python src/quickdraws_step2.py \
    --output output/qd_imputed \
    --bed ${bed} \
    --bgen ${bgen} \
    --sample ${sample} \
    --output_step1 output/qd \
    --calibrationFile output/qd.calibration \
    --hdf5 output/qd.hdf5  \
    --covar ${covar} \
    --unrel_sample_list example/unrelated_FID_IID.txt