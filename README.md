# Quickdraws

Quickdraws relies on cuda-enabled pytorch for speed, and it is expected to work on most cuda-compatible Linux systems.


## Installation

It is strongly recommended to either set up a python virtual environment, or a conda environment:


### Python virtual environment

```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```


### Conda environment

```
conda create -n quickdraws python=3.11 -y
conda activate quickdraws
pip install --upgrade pip setuptools wheel
```


### Install pytorch and quickdraws

It is necessary for anything bigger than toy examples to use either:

1. on Linux, a cuda-enabled version of pytorch
2. on macOS, the latest nightly build of pytorch, which can leverage the MPS backend

Use the [pytorch configuration helper](https://pytorch.org/get-started/locally/) to find suitable installation instruction for your system.
The code snippets below will probably work for most systems, and should install quickdraws in approximately 10 minutes:

#### Linux

```
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install quickdraws
```

#### macOS

```
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install quickdraws
```


## Running example

Once you install `quickdraws`, three executables should be available in your path:

1. `convert-to-hdf5`
2. `quickdraws-step-1`
3.  `quickdraws-step-2`.

Clone the Git repository to access the example data and script demonstrating how these can be used:

```
git clone https://github.com/PalamaraLab/quickdraws.git
cd quickdraws
bash run_example.sh
```


## Local development

To make changes to the quickdraws sourcecode, obtain the repository and install it using [poetry](https://python-poetry.org/docs/).
Assuming you have poetry installed:

```
git clone https://github.com/PalamaraLab/quickdraws.git
cd quickdraws
poetry install
```


## Documentation
See https://github.com/PalamaraLab/quickdraws/wiki/Quickdraws-GWAS-Software-Documentation

## Summary Statistics for some UKB traits
See https://www.stats.ox.ac.uk/publication-data/sge/ppg/quickdraws/ 

## Contact information
For any technical issues please contact Hrushikesh Loya (loya@stats.ox.ac.uk)


## Citation
Loya et al., "A scalable variational inference approach for increased mixed-model association power" under review
