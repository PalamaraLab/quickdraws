#!/bin/bash

bed="example/example"
phenoFile="example/phenotypes_binary.txt"
covarFile="example/covariates.txt"
bgen="example/example.bgen"
sample="example/example.sample"

outDir="example/output"
mkdir -p ${outDir}

## step 1: run model fitting (step 1) on genotypes and phenotypes
quickdraws-step-1 \
   --out ${outDir}/qd_bin \
   --bed ${bed} \
   --phenoFile ${phenoFile} \
   --covarFile ${covarFile} \
   --h2_grid \
   --binary

## step 2: get association stats for SNPs in bgen file
quickdraws-step-2 \
    --out ${outDir}/qd_bin \
    --bed ${bed} \
    --out_step1 ${outDir}/qd_bin \
    --covarFile ${covarFile} \
    --unrel_sample_list example/unrelated_FID_IID.txt \
    --binary \
    --firth

quickdraws-step-2 \
    --out ${outDir}/qd_bin_imputed \
    --bgen ${bgen} \
    --sample ${sample} \
    --out_step1 ${outDir}/qd_bin \
    --calibrationFile ${outDir}/qd_bin.calibration \
    --covarFile ${covarFile} \
    --unrel_sample_list example/unrelated_FID_IID.txt \
    --binary \
    --firth
