#!/bin/bash

bed="example/example"
phenoFile="example/phenotypes.txt"
covarFile="example/covariates.txt"
bgen="example/example.bgen"
sample="example/example.sample"

outDir="example/output"
mkdir -p ${outDir}

## step 0: generating HDF5 file from data (optional step)
convert-to-hdf5 \
   --out ${outDir}/qd \
   --bed ${bed}  

## step 1: run model fitting (step 1) on genotypes and phenotypes
quickdraws-step-1 \
   --out ${outDir}/qd \
   --bed ${bed} \
   --phenoFile ${phenoFile} \
   --covarFile ${covarFile}

# step 2: get association stats for SNPs in bgen file
quickdraws-step-2 \
    --out ${outDir}/qd \
    --bed ${bed} \
    --out_step1 ${outDir}/qd \
    --covarFile ${covarFile} \
    --unrel_sample_list example/unrelated_FID_IID.txt

quickdraws-step-2 \
    --out ${outDir}/qd_imputed \
    --bgen ${bgen} \
    --sample ${sample} \
    --out_step1 ${outDir}/qd \
    --calibrationFile ${outDir}/qd.calibration \
    --covarFile ${covarFile} \
    --unrel_sample_list example/unrelated_FID_IID.txt
