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

It is necessary for anything bigger than toy examples to use a cuda-enabled version of pytorch.
Use the [pytorch configuration helper](https://pytorch.org/get-started/locally/) to find suitable installation instruction for your system.
The code snippet below will probably work for most systems:

```
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install quickdraws
```


## Running example

Once you install `quickdraws`, two executables should be available in your path: `quickdraws-step-1` and `quickdraws-step-2`.
Clone the Git repository to access the example data and script demonstrating how these can be used:

```
git clone https://github.com/PalamaraLab/quickdraws.git
cd quickdraws
bash run_example.sh
```


## Documentation
See https://github.com/PalamaraLab/quickdraws/wiki/Quickdraws


## Contact information
For any technical issues please contact Hrushikesh Loya (loya@stats.ox.ac.uk)


## Citation
Loya et al., "A scalable variational inference approach for increased mixed-model association power" under review
