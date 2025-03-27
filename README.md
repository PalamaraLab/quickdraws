<img src="https://github.com/PalamaraLab/quickdraws/blob/384f5047aa3a278952a262c84bb1d4c3d14bc1f4/quickdraw.png" alt="Quickdraws Logo" width="100%"/>

<hr style="border: 3px solid #000; margin-top: 20px;"/>

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

1. on Linux, a cuda-enabled version of pytorch (now available by default from PyPI)
2. on macOS, the latest pytorch, which can leverage the MPS backend

Use the [pytorch configuration helper](https://pytorch.org/get-started/locally/) to find suitable installation instruction for your system, based on your preferred CUDA version.
If you want a specific CUDA version, you may need to start with something like `pip install torch --index-url https://download.pytorch.org/whl/cu118`.

The code snippet below will probably work for most systems, and should install quickdraws in approximately 10 minutes for cuda or much quicker on macOS:

#### Linux or macOS

```
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
See https://palamaralab.github.io/software/quickdraws/manual/

## Summary Statistics for some UKB traits
See https://www.stats.ox.ac.uk/publication-data/sge/ppg/quickdraws/ 

## Contact information
For any technical issues please contact Hrushikesh Loya (loya@stats.ox.ac.uk)


## Citation
Loya et al., "A scalable variational inference approach for increased mixed-model association power" under review


## Release Notes

### v0.1.4 (2025-03-27)

- Bug fix, introduced --chunksize argument in quickdraws-step-1 aswell

### v0.1.3 (2025-01-24)

- Better memory usage and speed for step 0, introduced --chunksize argument

### v0.1.2 (2025-01-07)

- Minor updates to documentation
- Remove reliance on specific CUDA torch for Linux
- Resolve numpy dependency conflict

### v0.1.1 (2024-10-24)

- Minor updates to documentation
- Remove reliance on pre-release Torch for macOS

### v0.1.0 (2024-10-15)

First public release to accompany the paper (see citation above).
