# Quickdraws #

## Installation ##
```
python3.9 -m venv venv
. venv/bin/activate
pip install -r  requirements.txt
chmod +x RHEmcmt
```

## How to run Quickdraws ##
### Step 0 ###
Step 0 involves generating HDF5 file corresponding to the bed file for fast sample-wise access:

```
python src/convert_to_hdf5.py \
  --output ${output} \
  --bed ${bed} \
  --pheno ${pheno} \
  --covar ${covar} \
  --modelSnps maf001.snps \
  --binary
```

### Step 1 (Needs GPUs!) ###
Step 1 involves estimating the genetic effect through the spike-and-slab Bayesian linear regression and is done as follows:

```
python src/quickdraws_step1.py \
    --output ${output} \
    --bed ${bed} \
    --pheno ${pheno} \
    --output_step0 ${output} \
    --covar ${covar} \
    --modelSnps maf001.snps \
    --binary \
    --h2_grid
```

### Step 2 ###
Step 2 involves calculating and calibrating the test statistics given the input and estimated genetic effects. In order to generate summary statistics for genotype array data (in bed format):

```
python src/quickdraws_step2.py \
   --output ${output} \
   --bed ${bed} \
   --output_step1 ${output} \
   --covar ${covar} \
   --unrel_sample_list unrelated_White_British.FID_IID.txt \
   --ldscores LDSCORE.1000G_EUR.tab.gz \
   --calibrate True \
   --binary \
   --firth \
```

and then to compute summary statistics for imputed data (in bgen format) do following:

```
python src/quickdraws_step2.py \
    --output ${output}_${chr} \
    --bgen ${bgen} \
    --sample ${sample} \
    --output_step1 ${output}  \
    --calibrationFile ${output}.calibration \
    --covar ${covar} \
    --extract snp_list.txt \
    --binary \
    --firth  
```

## Running example ##
```
mkdir output
bash run_example.sh
```
Note: Run the example on a computer with GPU access, as step 1 involves using GPUs

## Running on UK Biobank RAP ##
TODO

## List of options ##
You can use help to get a full list of options for step 1 and step 2 as follows:

```
python src/quickdraws_step1.py --help
python src/quickdraws_step2.py --help
```

A list of options for step 1:

| Option        | Required      | Data-type  |  Description   |
|:-------------:|:-------------:|:----------:|:--------------:|
| --bed    | &check; | STRING | Prefix for bed/bim/fam files  |
| --covar     | &check;     |   STRING | Covariate file; should be in "FID IID Var1 Var2 ..." format and tab-seperated  |
| --pheno | &check;      |   STRING | Phenotype file; should be in "FID IID Trait1 Trait2 ..." format and tab-seperated  |
| --output | &check; | STRING | output prefix |
| --modelSnps | &cross; | STRING | Path to list of SNPs to be considered in model fitting |
| --make_annot | &cross; | STRING |  |
| --binary | &cross; |  | Add if traits are binary |
| --h2_grid | &cross; | | Perform a grid-search for heritability instead of RHE-MCMT |
| --output_step0 | &cross; | STRING |  |
| --num_epochs | &cross; | STRING |  |
| --alpha_search_epochs | &cross; | STRING |  |
| --batch_size | &cross; | STRING |  |
| --lr | &cross; | STRING |  |
| --rhemc | &cross; | STRING |  |

A list of options for step 2:

| Option        | Required      | Data-type  |  Description   |
|:-------------:|:-------------:|:----------:|:--------------:|
| --bed    | &check; | STRING | Prefix for bed/bim/fam files  |
| --covar     | &check;     |   STRING | Covariate file; should be in "FID IID Var1 Var2 ..." format and tab-seperated  |
| --pheno | &check;      |   STRING | Phenotype file; should be in "FID IID Trait1 Trait2 ..." format and tab-seperated  |
| --output | &check; | STRING | output prefix |
| --modelSnps | &cross; | STRING | Path to list of SNPs to be considered in model fitting |
| --make_annot | &cross; | STRING |  |
| --binary | &cross; |  | Add if traits are binary |
| --h2_grid | &cross; | | Perform a grid-search for heritability instead of RHE-MCMT |
| --output_step0 | &cross; | STRING |  |
| --num_epochs | &cross; | STRING |  |

## Interpreting the output ##

## FAQs
### Recommendations to generate model snps file 

### Recommendations to generate list of unrelated homogenous samples 

### LDscores

### Using Regenie for step2 



## Citation ##

