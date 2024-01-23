# Quickdraws #

## Installation (uses conda) ##
```
conda create -n qd python=3.9 -y
conda activate qd
cd quickdraws/
conda install -c conda-forge cudatoolkit -y
python3.9 -m pip install -r requirements.txt 
chmod +x src/RHEmcmt 
```

## How to run Quickdraws ##
#### Step 0 ###
Step 0 involves generating HDF5 file corresponding to the bed file for fast sample-wise access:

```
python src/convert_to_hdf5.py \
  --out master \
  --bed ${bed} \
  --modelSnps maf001.snps
```
#### Step 1 ###
Step 1 involves estimating the genetic effect through the spike-and-slab Bayesian linear regression and is done as follows:

```
python src/quickdraws_step1.py \
    --out ${output} \
    --hdf5 master.hdf5 \
    --bed ${bed} \
    --phenoFile ${pheno} \
    --covarFile ${covar} \
    --modelSnps maf001.snps \
    --ldscores LDSCORE.1000G_EUR.tab.gz 
```

For binary traits, run with `--binary` option. Additionally, use `--h2_grid` option instead of `--ldscores` to perform a grid-search over heritability values instead of relying on RHE-MCMT
#### Step 2 ###
Step 2 involves calculating and calibrating the test statistics given the input and estimated genetic effects. In order to generate summary statistics for genotype array data (in bed format):

```
python src/quickdraws_step2.py \
   --out ${output} \
   --bed ${bed} \
   --out_step1 ${output} \
   --covarFile ${covar} \
   --unrel_sample_list unrelated_White_British.FID_IID.txt \
   --ldscores LDSCORE.1000G_EUR.tab.gz \
   --calibrate True
```

For binary traits, run with `--binary` option. Use `--firth` option to additionally perform firth logistic regression on rare variants or rare outcomes to correct the summary statistics
#### Step 2 (bgen files) ###
To compute summary statistics for imputed data (in bgen format) do following:

```
python src/quickdraws_step2.py \
    --output ${output}_${chr} \
    --bgen ${bgen} \
    --sample ${sample} \
    --output_step1 ${output}  \
    --calibrationFile ${output}.calibration \
    --covar ${covar} \
    --extract snp_list.txt
```

where `--calibrationFile` option records the attenuation ratio calibration recorded from previous step

## Running example ##
```
mkdir output
bash run_example.sh
```

## Running on UK Biobank RAP ##
TODO

## List of options ##
You can use help to get a full list of options for step 1 and step 2 as follows:

```
python src/quickdraws_step1.py --help
python src/quickdraws_step2.py --help
```

### Main options for step 1:

| Option                 | Required  | Data-type  |  Description   |
|:--------------------:|:---------:|:----------:|:-------------------------------------------|
| --bed                | &check; | STRING | Prefix for bed/bim/fam files  |
| --phenoFile          | &check; | STRING | Phenotype file; should be in "FID IID Trait1 Trait2 ..." format and tab-seperated  |
| --out                | &check; | STRING | Prefix for where to save any results or files |
| --covarFile          | &cross; | STRING | Covariate file; should be in "FID IID Var1 Var2 ..." format and tab-seperated  |
| --hdf5               | &cross; | STRING | Path to the master HDF5 file containing binarized genotypes |
| --modelSnps          | &cross; | STRING | Path to list of SNPs to be considered in model fitting |
| --ldscores           | &cross; | STRING | Path to ldscores file (should have MAF and LDSCORE columns and tab-seperated) |
| --binary             | &cross; |   -     | Add if traits being analyzed are binary |
| --h2_grid            | &cross; |   -     | Perform a grid-search for heritability instead of RHE-MCMT |
| --lowmem             | &cross; |    -    | Enables low memory operation |
| --predBetasFlag      | &cross; |   -     | Calculates and store the BLUP betas |
| --train_split        | &cross; | FLOAT  | Fraction of dataset used for training while searching for sparsity parameter |
| --phen_thres         | &cross; | FLOAT  | The phenotyping rate threshold below which the phenotype isn't used |
| --rhe_random_vectors | &cross; | INT    | Number of random vectors used in RHE-MCMT |

### Main options for step 2:

| Option               | Required  | Data-type  |  Description   |
|:--------------------:|:---------:|:----------:|:-------------------------------------------|
| --bed               | &check; | STRING | Prefix for bed/bim/fam files  |
| --bgen              | &check; | STRING | Prefix for bgen file  |
| --sample            | &check; | STRING | Prefix for sample file  |
| --out               | &check; | STRING | Prefix for where to save any results or files |
| --out_step1         | &check; | STRING | Prefix for output files generated in step 1 |
| --covarFile         | &cross; | STRING | Covariate file; should be in "FID IID Var1 Var2 ..." format and tab-seperated  |
| --ldscores          | &cross; | STRING | Path to ldscores file (should have MAF and LDSCORE columns and tab-seperated) |
| --unrel_sample_list | &cross; | STRING | List of unrelated homogenous samples ("FID IID" format and tab-seperated) |
| --binary            | &cross; |    -    | Add if traits being analyzed are binary |
| --firth             | &cross; |   -     | Indicate if firth logistic regression is desired |
| --calibrate         | &cross; |    -    | Calibrate the test statistics based on LDSC attn. ratio |
| --calibrationFile   | &cross; | STRING | File carrying calibration constants to be multiplied per phenotype |
| --extract           | &cross; | STRING | List of SNPs for which we calculate the test statistics |
| --sample_weights    | &cross; | STRING | Sampling weights ("FID IID weights" format and tab-seperated) |
| --firth_pval_thresh | &cross; | FLOAT  | P-value threshold below which firth logistic regression performed |
| --num_threads       | &cross; |  INT   | Number of threads used for test statistics calculation |

## Interpreting the output ##

We generate many output files, a list of all the files generated through the analysis:

| Filename            | Step  |   Description   |
|:--------------------:|:---------:|:-------------------------------------------|
| {out}.traits         | 1    | Stores the postprocessed traits used through the analysis |
| {out}.covar_effects         | 1    | Stores the covariate effects on phenotypes |
| {out}.hdf5         |  1    | HDF5 file which stores the genotype matrix for quick access |
| {out}.h2         |  1    | Heritability per phenotype |
| {out}.alpha         |  1    | Optimal sparsity parameter fit per phenotype |
| {out}.{pheno}.blup         | 1    | BLUP betas for phenotype ${pheno} |
| {out}.{pheno}.sumstats.gz         |  2    | QD summary statistics for phenotype ${pheno} (gzip-compressed) |
| {out}.{pheno}_lrunrel.sumstats    |  2    | linear regression on unrelated samples for phenotype ${pheno} (HDF accessible) |
| {out}.calibration         |  2    | Calibration factor caclulated through LDSC attn. ratio |


## FAQs
### Recommendations to generate model snps file 
```
plink --bfile genotype --freq --out all_snps
awk '$5 > 0.01' all_snps.frq > maf001.snps
```

### LDscores
TODO

### Using Regenie for step2 
TODO


## Citation ##
TODO
