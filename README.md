# Quickdraws #

## Installation (uses conda) ##
```
conda create -n qd python=3.9 -y
conda activate qd
cd quickdraws/
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118
python3.9 -m pip install -r requirements.txt
chmod +x src/RHEmcmt 
```

## Running example ##
```
mkdir output
bash run_example.sh
```

## Documentation ##
See https://github.com/hrushikeshloya/quickdraws/wiki/Quickdraws 

## Contact information ##
For any technical issues please contact Hrushikesh Loya (loya@stats.ox.ac.uk)

## Citation ##
Loya et al., "A scalable variational inference approach for increased mixed-model association power" under review
