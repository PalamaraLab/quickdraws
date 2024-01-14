#!/bin/bash

bed="example/example"
phenoFile="example/phenotypes.txt"
covarFile="example/covariates.txt"
bgen="example/example.bgen"
sample="example/example.sample"

# step 1: run model fitting (step 1) on genotypes and phenotypes
python src/quickdraws_step1.py \
   --out output/qd \
   --bed ${bed} \
   --phenoFile ${phenoFile} \
   --covarFile ${covarFile} \
   --h2_grid \
   --predBetasFlag

## step 2: get association stats for SNPs in bgen file
python src/quickdraws_step2.py \
    --out output/qd \
    --bed ${bed} \
    --out_step1 output/qd \
    --covarFile ${covarFile} \
    --unrel_sample_list example/unrelated_FID_IID.txt \

python src/quickdraws_step2.py \
    --out output/qd_imputed \
    --bgen ${bgen} \
    --sample ${sample} \
    --out_step1 output/qd \
    --calibrationFile output/qd.calibration \
    --covarFile ${covarFile} \
    --unrel_sample_list example/unrelated_FID_IID.txt
