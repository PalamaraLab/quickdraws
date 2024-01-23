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

## Running example ##
```
mkdir output
bash run_example.sh
```

## Contact information ##
For any technical issues please contact Hrushikesh Loya (loya@stats.ox.ac.uk)

## Citation ##
Loya et al., "A scalable variational inference approach for increased mixed-model association power" under review
