# This file is part of the Quickdraws GWAS software suite.
#
# Copyright (C) 2024 Quickdraws Developers
#
# Quickdraws is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Quickdraws is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Quickdraws. If not, see <http://www.gnu.org/licenses/>.


import pprint
import random
from pathlib import Path
import argparse
import os
from numpy.core.fromnumeric import mean
import wandb
from typing import Any, Callable, Optional
import h5py
from tqdm import tqdm
from distutils.util import strtobool
import time

import numpy as np
import scipy
from scipy import stats
from scipy.stats import norm
import math
import pandas as pd
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

import gc
import pdb
import logging

from .correct_for_relatives import get_correction_for_relatives

logger = logging.getLogger(__name__)
# torch._dynamo.config.cache_size_limit = 1024

if torch.cuda.is_available():
    import bitsandbytes as bnb

def str_to_bool(s: str) -> bool:
    return bool(strtobool(s))


## Custom layers:
class BBB_Linear_spike_slab(nn.Module):
    ''' This class is a custom PyTorch module implementing a Bayesian Linear layer with spike-and-slab priors. 
    This advanced concept is used in Bayesian neural networks to introduce sparsity in the model weights. 
    
    __init__ : Inherit from nn.Module: Inherits from PyTorch's base class for all neural network modules.
    Parameters: Accepts parameters like the number of samples to draw during inference (num_samples), the shape of the weights tensor (weights_shape, from the fc1 layer), 
    a sparsity-controlling hyperparameter (alpha), and the standard deviation of the prior distribution (prior_sig).

    Spike Parameter: Initializes the spike1 parameter as a learnable tensor with uniform values set to alpha. 
    This parameter represents the probability of a weight being in the "slab" (non-zero values) part of the spike-and-slab distribution.
    
    Model Attributes: Sets various attributes like the number of samples (num_samples), the sparsity parameter (alpha), 
    a scaling factor (c), and both the variance and log-variance of the prior distribution (prior_var and log_prior_var).
    
    Sigma Parameter: Initializes sigma1 as a learnable parameter representing the standard deviation 
    of the posterior distribution of the weights, starting with the values from prior_sig.
    '''
    def __init__(
        self,
        num_samples,
        weights_shape: torch.Size,
        alpha,
        prior_sig: torch.Tensor,
        mu1,
    ):
        '''
        Input Checks and Adjustments: Adjusts the scaling factor (c) to ensure stability and clamps the sigma (sig1) 
        and spike (spike) parameters to avoid extreme values.
        
        Local Reparameterization: Calls local_reparameterize to generate the output of the layer. 
        This method samples from the approximate posterior distribution of the weights and biases to obtain the layer's output.

        Output Only Mode: If only_output is True, returns only the layer's output, 
        skipping the calculation of the Kullback-Leibler (KL) divergence.
        
        KL Divergence: Computes the KL divergence between the posterior and prior distributions of the weights as a regularizer. 

        Evaluation Steps:
        1. Variance Adjustment: The sig1 parameter is clamped to a minimum value (eps) to prevent division by zero and other numerical issues.
        2. Log Variance: The log variance of the posterior distribution (sig1 squared) is computed to stabilize the computation and for ease of calculation.
        3. KL Computation: The KL divergence is then computed as the sum of two parts:
            The first part evaluates the Gaussian component, considering the log variance of the posterior, 
            the variance of the prior, and the squared difference between the posterior mean and zero.
            The second part accounts for the spike component, penalizing the probability of non-zero weights based on 
            the prior sparsity expectation (alpha).
        4. Scaling by Number of Samples: The final KL divergence is scaled by the number of samples (num_samples) to 
        average the divergence over all data points, making it more representative of the entire dataset.
        '''
        super(BBB_Linear_spike_slab, self).__init__()
        self.spike1 = nn.Parameter(torch.empty(weights_shape).uniform_(alpha, alpha))
        self.num_samples = num_samples
        self.alpha = alpha
        self.c = 2.0
        self.prior_var = prior_sig**2
        self.log_prior_var = torch.log(self.prior_var)
        self.sigma1 = nn.Parameter(prior_sig)

    def forward(self, x, mu1, only_output=False, test=False):
        eps = 1e-12
        self.c = min(self.c + 0.01, 10)
        sig1 = torch.clamp(F.relu(self.sigma1), min=eps)
        weight_mu = mu1.weight
        spike = torch.clamp(self.spike1, 1e-6, 1.0 - 1e-6)
        out = self.local_reparameterize(
            x, spike, weight_mu, sig1, weight_mu.device, test
        )
        if only_output:
            return out, 0

        logvar = torch.log(sig1.pow(2))

        KL_div = -0.5 * torch.sum(
            spike.mul(
                1
                + logvar
                - self.log_prior_var
                - mu1.weight.pow(2) / self.prior_var
                - logvar.exp() / self.prior_var
            )
        ) + torch.sum(
            (1 - spike).mul(torch.log((1 - spike) / (1 - self.alpha)))
            + spike.mul(torch.log(spike / self.alpha))
        )
        return out, KL_div / self.num_samples

    def reparameterize(self, mu1, sig1, device, test=False):
        if test:
            spike = torch.clamp(self.spike1, 1e-6, 1.0 - 1e-6)
            return spike.mul(mu1)
        if device.type == 'cuda':
            eps = torch.cuda.FloatTensor(mu1.shape)
        else:
            eps = torch.FloatTensor(mu1.shape)
        torch.randn(mu1.shape, out=eps)
        sig_eps = torch.mul(sig1, eps)
        gaussian1 = mu1 + sig_eps
        gaussian2 = mu1 - sig_eps

        spike = torch.clamp(self.spike1, 1e-6, 1.0 - 1e-6)
        log_spike = torch.log(spike / (1 - spike))
        if device.type == 'cuda':
            unif = torch.cuda.FloatTensor(spike.shape)
        else:
            unif = torch.FloatTensor(spike.shape)
        torch.rand(spike.shape, out=unif)
        log_unif = torch.log(unif / (1 - unif))
        eta1 = log_spike + log_unif
        eta2 = log_spike - log_unif

        selection = torch.stack(
            (
                F.sigmoid(self.c * eta1).mul(gaussian1),
                F.sigmoid(self.c * eta2).mul(gaussian2),
                F.sigmoid(self.c * eta1).mul(gaussian2),
                F.sigmoid(self.c * eta2).mul(gaussian1),
            ),
            dim=0,
        )  ##shape = 4 x O X I
        return selection

    def local_reparameterize(self, x, spike, mu1, sig1, device, test=False):
        assert x.ndim == 2
        assert mu1.shape == spike.shape
        assert mu1.shape == sig1.shape
        assert mu1.ndim == 2
        if test:
            return F.linear(x, spike.mul(mu1))
        mean_preactivations = spike.mul(mu1)
        var_preactivations = (
            spike.mul(sig1.pow(2)) + spike.mul(mu1.pow(2)) - mean_preactivations.pow(2)
        )
        if device.type == 'cuda':
            eps = torch.cuda.FloatTensor(x.shape[0], mu1.shape[0]).normal_()
        else:
            eps = torch.FloatTensor(x.shape[0], mu1.shape[0]).normal_()
        selection = torch.stack(
            (
                F.linear(x, mean_preactivations)
                + eps * torch.sqrt(F.linear(x.pow(2), var_preactivations)),
                F.linear(x, mean_preactivations)
                - eps * torch.sqrt(F.linear(x.pow(2), var_preactivations)),
            ),
            dim=0,
        )
        return selection


## Neural network model:
class Model(nn.Module):
    '''
    The Model class extends PyTorch's nn.Module and encapsulates a neural network model that utilizes 
    a Bayesian by Backpropagation Linear layer with spike-and-slab priors (BBB_Linear_spike_slab).

    __init__ : Inherits from nn.Module, PyTorch's base class for all neural network modules. Parameters:

    dim_in: Input dimensionality. 
    The input dimension of the model is set to the number of SNPs not on the chromosome currently being left out (LOCO analysis). 
    This is determined by counting the SNPs where chr_map != chr, ensuring that the model only considers SNPs from other chromosomes.

    dim_out: Output dimensionality.
    The output dimension is determined by the count of elements in dim_out that match the current alpha_no. 
    This effectively sets the number of outputs to match the number of traits or phenotypes 
    being analyzed under the current alpha configuration.

    num_samples: Number of samples to draw from the posterior during inference.
    Specifies the number of samples to draw from the posterior distribution during inference. 
    
    alpha: Hyperparameter for controlling the sparsity in the weights.

    prior_sig: Standard deviation of the prior distribution of the weights.

    posterior_sig: Optional; standard deviation of the posterior distribution of the weights, initialized during training.
    Specifies the standard deviation of the posterior distribution of the weights. Like prior_sig, it is adjusted for the current alpha_no and excludes SNPs from the current chromosome. 
    
    mu: Optional; initial mean values of the posterior distribution of the weights.
    Represents the mean values of the posterior distribution of the weights. If provided, mu is filtered based on the current alpha_no and chromosome exclusion, then converted to a tensor, set to the appropriate data type, and transferred to the specified device. 
    This operation initializes the model's weights with values potentially learned from previous analyses or a different context.

    spike: Optional; initial values of the spike variable, controlling the sparsity of the weights.
    Specifies the spike parameter values for the spike-and-slab prior. Like mu, if spike is provided, it is processed based on the current alpha_no and chromosome exclusion, then converted to a tensor, set to the appropriate data type, and transferred to the device. 
    The spike parameter controls the sparsity in the model's weights, complementing the role of alpha.

    Initializes a linear layer (fc1) without bias, followed by a BBB_Linear_spike_slab layer (sc1) with specified parameters.
    Optionally sets posterior_sig, mu, and spike if provided.
    '''
    def __init__(
        self,
        dim_in,
        dim_out,
        num_samples,
        alpha,
        prior_sig,
        posterior_sig=None,
        mu=None,
        spike=None,
    ):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_out, bias=False)
        self.sc1 = BBB_Linear_spike_slab(
            weights_shape=self.fc1.weight.shape,
            num_samples=num_samples,
            alpha=alpha,
            prior_sig=prior_sig,
            mu1=self.fc1,
        )
        if posterior_sig is not None:
            # logging.info("Setting posterior sigma to CAVI derived sigmas")
            self.sc1.sigma1 = nn.Parameter(posterior_sig, requires_grad=True)
        if mu is not None:
            # logging.info("Initializing posterior means through transfer learning")
            self.fc1.weight.data = mu.to(self.fc1.weight.device)
        if spike is not None:
            # logging.info("Initializing posterior spike through transfer learning")
            self.sc1.spike1.data = spike.to(self.sc1.spike1.device)

    '''
    x: Input tensor.
    offset: Offset to be added to the output, typically used in generalized linear models.
    only_output: If True, skips the computation of KL divergence, returning only the layer's output.
    test: Indicates inference mode, affecting the sampling behavior in the BBB_Linear_spike_slab layer.
    binary: If True, applies a sigmoid activation function to the output, useful for binary classification tasks.

    Propagates x through the BBB_Linear_spike_slab layer, obtaining the output and the KL divergence term.
    Adds the offset to the output. If binary is True, applies the sigmoid function to transform outputs into probabilities.
    Returns the final output and the scaled KL divergence term, multiplied by the number of batches.
    '''
    def forward(self, x, offset, only_output=False, test=False, binary=False):
        num_batches = x.shape[0]
        x, reg = self.sc1(x, self.fc1, only_output, test)
        if binary:
            x = torch.sigmoid(x + offset)
        else:
            x = x + offset
        return x, reg * num_batches
        ## caution


class HDF5Dataset:
    def __init__(
        self,
        split: str,
        filename: str,
        phen_col: str,
        batch_size: int,
        transform: Optional[Callable] = None,
        train_split=0.8,
        lowmem=False,
    ):
        if split not in ["train", "test", "both"]:
            raise ValueError("Invalid value for split argument")
        self.split = split
        self.filename = filename
        self.transform = transform
        self.h5py_file = h5py.File(self.filename, "r")
        assert self.h5py_file["hap1"].shape == self.h5py_file["hap2"].shape
        total_samples = len(self.h5py_file["hap1"])
        train_samples = (int(train_split * total_samples) // batch_size)*batch_size

        if split == "train":
            if not lowmem:
                self.hap1 = np.array(self.h5py_file["hap1"][0:train_samples])
                self.hap2 = np.array(self.h5py_file["hap2"][0:train_samples])
            else:
                self.hap1 = self.h5py_file["hap1"]
                self.hap2 = self.h5py_file["hap2"]
            self.output = torch.as_tensor(
                np.array(self.h5py_file[phen_col][0:train_samples], dtype=float)
            ).float()
            self.covar_effect = torch.as_tensor(
                np.array(self.h5py_file["covar_effect"][0:train_samples], dtype=float)
            ).float()
            self.covars = torch.as_tensor(np.array(self.h5py_file["covars"])[0:train_samples]).float()
        elif split == "test":
            ### No lowmem in testing, can just reduce train_split or test at the end
            self.hap1 = np.array(self.h5py_file["hap1"][train_samples:])
            self.hap2 = np.array(self.h5py_file["hap2"][train_samples:])
            self.output = torch.as_tensor(
                np.array(self.h5py_file[phen_col][train_samples:], dtype=float)
            ).float()
            self.covar_effect = torch.as_tensor(
                np.array(self.h5py_file["covar_effect"][train_samples:], dtype=float)
            ).float()
            self.covars = torch.as_tensor(np.array(self.h5py_file["covars"])[train_samples:]).float()
        else:
            if not lowmem:
                self.hap1, self.hap2 = self.h5py_file["hap1"][:], self.h5py_file["hap2"][:]
            else:
                self.hap1, self.hap2 = self.h5py_file["hap1"], self.h5py_file["hap2"]
            self.output = torch.as_tensor(
                np.array(self.h5py_file[phen_col][:], dtype=float)
            ).float()
            self.covar_effect = torch.as_tensor(
                np.array(self.h5py_file["covar_effect"][:], dtype=float)
            ).float()
            self.covars = torch.as_tensor(np.array(self.h5py_file["covars"])).float()

        self.std_genotype = np.maximum(np.array(self.h5py_file["std_genotype"]), 1e-6)
        self.geno_covar_effect = torch.as_tensor(
            np.array(self.h5py_file["geno_covar_effect"])
        ).float()
        self.num_snps = len(self.std_genotype)
        self.chr = torch.as_tensor(np.array(self.h5py_file["chr"])).float()
        self.iid = np.array(self.h5py_file["iid"], dtype=int)
        self.length = len(self.output)
        self.pheno_names = [p.decode() for p in self.h5py_file['pheno_names']]
        if not lowmem:
            self.h5py_file.close()

        if self.output.ndim == 1:
            self.output = self.output.reshape(-1, 1)

    def close_hdf5(self):
        self.h5py_file.close()
    
    def __getitem__(self, index: int) -> Any:
        input = torch.as_tensor(np.unpackbits(self.hap1[index]) + np.unpackbits(self.hap2[index])).float()
        input = input[0 : self.num_snps] - self.covars[index] @ self.geno_covar_effect
        output = self.output[index]
        covar_effect = self.covar_effect[index]
        if self.transform is not None:
            input = self.transform(input)
        return (input, covar_effect, output)

    def __len__(self):
        return self.length


## Training class - this is where training happens:
class Trainer:
    def __init__(
        self,
        args,
        alpha,
        h2,
        train_dataset,
        test_dataset,
        model_list,
        lr,
        device,
        adam_eps=1e-4,
        weight_decay=0,
        validate_every=1,
        chr_map=None,
        pheno_for_model=None,
        predBetasFlag=False
    ):
        self.args = args
        '''
        The dimensionality of h2 is checked. If h2 is two-dimensional, it is summed across one axis to reduce it to a 
        one-dimensional array. This simplification might be applied when h2 contains SNP-specific heritability estimates, 
        and a simpler per-trait estimate is needed for training.
        '''
        if h2.ndim == 2:
            self.h2 = torch.sum(h2, axis=1)
        else:
            self.h2 = h2
        self.device = device
        self.model_list = model_list
        self.num_samples = train_dataset.length
        '''
        df_iid_fid: Creates a DataFrame to map the internal individual IDs (IID) and family IDs (FID) from the train_dataset. 
        '''
        self.df_iid_fid = pd.DataFrame(
            train_dataset.iid, columns=["FID", "IID"]
        )
        self.df_iid_fid[['FID','IID']] = self.df_iid_fid[['FID','IID']].astype('str')
        self.df_iid_fid = self.df_iid_fid['FID'].str.cat(self.df_iid_fid['IID'].values,sep='_')
        '''
        Determines how often the validation should be performed during training. 
        '''
        self.validate_every = validate_every if validate_every > 0 else 1
        self.never_validate = validate_every < 0
        '''
        If provided, this indicates the chromosome mapping for each genetic variant in the dataset. 
        This information is crucial for LOCO analysis, where the influence of specific chromosomes is assessed 
        by excluding them from the model training.
        '''
        self.chr_map = chr_map
        self.alpha = alpha
        '''
        Calculates the variance of the covariate effects from the train_dataset. 
        '''
        self.var_covar_effect = torch.std(train_dataset.covar_effect, axis=0).float().to(device)**2
        if args.binary:
            self.var_covar_effect = torch.std(torch.sigmoid(train_dataset.covar_effect), axis=0).float().to(device)**2
        self.var_y = torch.std(train_dataset.output, axis=0).numpy()**2
        if self.chr_map is not None:
            self.unique_chr_map = torch.unique(self.chr_map).tolist()
            ## check if chr_map has chrs in chunks:
            chr_map_shifted = torch.as_tensor(
                self.chr_map.tolist()[1:] + [torch.inf]
            ).to(self.chr_map.device)
            assert torch.sum((self.chr_map - chr_map_shifted) != 0) == len(self.unique_chr_map)

            self.chr_loc = []
            for chr in self.unique_chr_map:
                self.chr_loc.append(int(min(torch.where(self.chr_map == chr)[0])))
            self.chr_loc.append(len(chr_map))

            if predBetasFlag: 
                self.unique_chr_map.append(max(self.unique_chr_map) + 1)
                self.chr_loc.append(len(self.chr_map))
            self.num_chr = len(self.unique_chr_map)


        if pheno_for_model is not None:
            self.pheno_for_model = pheno_for_model
        '''
        Initializes optimizers for each model, configuring them with the specified learning rates and other 
        optimization parameters (adam_eps, weight_decay).
        '''
        self.optimizer_list = []
        for model_no, model in enumerate(model_list):
            if device == 'cuda':
                ### caution !!!
                self.optimizer_list.append(
                    bnb.optim.Adam(
                        model.parameters(),
                        lr=lr[model_no],
                        eps=adam_eps,
                        weight_decay=weight_decay,
                        betas=(0.9, 0.995),
                        optim_bits=8,
                    )
                )
            else:
                self.optimizer_list.append(
                    torch.optim.Adam(
                        model.parameters(),
                        lr=lr[model_no],
                        eps=adam_eps,
                        weight_decay=weight_decay,
                        betas=(0.9, 0.995),
                    )
                )
        if self.args.cosine_scheduler:
            self.scheduler_list = []
            for model_no, optimizer in enumerate(self.optimizer_list):
                self.scheduler_list.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optimizer,
                        T_max=self.args.num_epochs,
                        eta_min=lr[model_no]/2,
                    )
                )
        '''
        sets up the loss functions (mse_loss, bce_loss) and selects the appropriate one based on whether the phenotype is binary.
        '''
        self.mse_loss = nn.MSELoss(reduction="none")
        self.bce_loss = nn.BCELoss(reduction="none")
        self.loss_func = (
            self.masked_mse_loss if not args.binary else self.masked_bce_loss
        )
        ## TODO: write a masked BCE loss function
        self.wandb_step = 1
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.num_workers > 0,
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.num_workers > 0,
        )
        self.pheno_names = train_dataset.pheno_names

    '''
    These functions compute the MSE and BCE losses, respectively, while ignoring NaN values in the labels and predictions. 
    '''
    def masked_bce_loss(self, preds, labels, h2=0.5):
        mask = (~torch.isnan(labels)) & (~torch.isnan(preds))
        loss = self.bce_loss(preds[mask], labels[mask])
        return torch.sum(loss)

    def masked_mse_loss(self, preds, labels, h2=0.5):
        mask = (~torch.isnan(labels)) & (~torch.isnan(preds))
        sq_error = (preds[mask] - labels[mask]) ** 2  ## B x O
        if type(h2) == torch.Tensor:
            sq_error = sq_error / (2 * (1 - h2[mask]))
        else:
            sq_error = sq_error
        return torch.sum(sq_error)  ## reduction = sun

    '''
    Calculates the coefficient of determination (R²) for model predictions versus actual labels, chromosome by chromosome. 
    This is crucial for getting the effective sample size for calibration in step2.
    '''
    def get_chr_r2(self, beta):
        with torch.no_grad():
            var = np.zeros((beta.shape[0], self.num_chr))
            '''
            Loops through each chromosome in self.unique_chr_map. 
            For each chromosome, the function prepares to collect model predictions (preds_arr_chr) and actual labels (labels_arr) across the entire test dataset.
            '''
            for chr_no, chr in enumerate(self.unique_chr_map):
                preds_arr_chr = []
                labels_arr = []
                '''
                Generates predictions (preds_c) for each genetic variant, excluding those from the current chromosome (chr). 
                This is achieved by selecting columns from input and beta corresponding to all chromosomes except chr, 
                effectively simulating a LOCO scenario.
                '''
                for input, covar_effect, label in self.test_dataloader:
                    input, covar_effect = (
                        input.to(self.device),
                        covar_effect.to(self.device),
                    )
                    preds_c = input[:, self.chr_map != chr]@(beta[:, self.chr_map != chr].T)
                    '''adjusts predictions by adding covariate effects (covar_effect), accounting for non-genetic factors.'''
                    preds_c += covar_effect
                    ''' Applies a sigmoid transformation to the predictions if the analysis is binary (self.args.binary), converting the linear output to probabilities.'''
                    if self.args.binary:
                        preds_c = torch.sigmoid(preds_c)
                    preds_arr_chr.extend(preds_c.cpu().numpy().tolist())
                    labels_arr.extend(label.numpy().tolist())
                    
                preds_arr_chr = np.array(preds_arr_chr)
                labels_arr = np.array(labels_arr)
                '''
                For each phenotype (prs_no), calculates the Pearson correlation coefficient between the aggregated predictions 
                and labels. The square of this coefficient gives the R² value, indicating how well the genetic variants 
                (excluding those on the current chromosome) predict the phenotype.
                '''
                for prs_no in range(beta.shape[0]):
                    if self.args.binary:
                        ### In low prevalence binary traits, the variance of the phenotype can be subs. diff in train and test split
                        ### We therefore only use the R calc. on test set, we calc. the residual variance on the train set which is closer to the full dataset
                        r = stats.pearsonr(preds_arr_chr[:, prs_no], labels_arr[:, prs_no])[0]
                        # var[prs_no, chr_no] = self.var_y[prs_no] + np.std(preds_arr_chr[:, prs_no])**2 - 2*r*np.std(preds_arr_chr[:, prs_no])*np.sqrt(self.var_y[prs_no])
                        var[prs_no, chr_no] = self.var_y[prs_no]*(1 - r**2)
                    else:
                        var[prs_no, chr_no] = np.std(labels_arr[:, prs_no] - preds_arr_chr[:, prs_no])**2
        
        return var

    def validation(self):
        '''
        Performs a validation run, using the models to predict on the test dataset and compute loss metrics. 
        '''
        labels_arr = [[] for _ in range(len(self.model_list))]
        preds_arr = [[] for _ in range(len(self.model_list))]
        # preds_arr_chr = [[[] for k in range(self.num_chr)] for _ in range(len(self.model_list))]
        with torch.no_grad():
            for input, covar_effect, label in self.test_dataloader:
                input, covar_effect, label = (
                    input.to(self.device),
                    covar_effect.to(self.device),
                    label.to(self.device),
                )
                for model_no, model in enumerate(self.model_list):
                    preds, _ = model(
                        input,
                        covar_effect,
                        only_output=True,
                        test=True,
                        binary=self.args.binary,
                    )
                    # spike = torch.clamp(model.sc1.spike1, 1e-6, 1.0 - 1e-6)
                    # mu = model.fc1.weight
                    # beta = spike.mul(mu)
                    # beta = beta.T
                    # for chr_no, chr in enumerate(self.unique_chr_map):
                    #     preds_c = input[:, self.chr_map != chr]@(beta[self.chr_map != chr])
                    #     preds_c += covar_effect
                    #     if self.args.binary:
                    #         preds_c = torch.sigmoid(preds_c)
                        
                    #     preds_arr_chr[model_no][chr_no].extend(preds_c.cpu().numpy().tolist())
                    
                    label[torch.isnan(label)] = preds[torch.isnan(label)]
                    preds_arr[model_no].extend(preds.cpu().numpy().tolist())
                    labels_arr[model_no].extend(label.cpu().numpy().tolist())

            test_loss = []
            for model_no in range(len(self.model_list)):
                test_loss.append(
                    self.loss_func(
                        torch.tensor(preds_arr[model_no]),
                        torch.tensor(labels_arr[model_no]),
                    )
                )
            preds_arr = np.array(preds_arr)  ## 6 x N x O
            labels_arr = np.array(labels_arr)  ## 6 x N x O
            # preds_arr_chr = np.array(preds_arr_chr) ## 6 x 22 x N x O

        return (
            test_loss,
            preds_arr,
            labels_arr,
            # preds_arr_chr,
        )

    def log_r2_loss(self, log_dict):
        ## Validation once a epoch
        (
            test_loss,
            preds_arr,
            labels_arr,
            # preds_arr_chr
        ) = self.validation()

        ## Average test loss
        for model_no, alpha in enumerate(self.alpha):
            log_dict["mean_test_loss_" + str(alpha)] = test_loss[model_no] / len(
                preds_arr[model_no]
            )
            ## Per phenotype R2 calculation
            for prs_no in range(preds_arr[model_no].shape[1]):
                test_loss_anc = self.loss_func(
                    torch.as_tensor(preds_arr[model_no, :, prs_no]),
                    torch.as_tensor(labels_arr[model_no, :, prs_no]),
                )
                test_r2_anc = stats.pearsonr(
                    preds_arr[model_no, :, prs_no],
                    labels_arr[model_no, :, prs_no],
                )
                log_dict[
                    "test_r2_pheno_" + str(prs_no) + "_alpha_" + str(alpha)
                ] = test_r2_anc[0]
                # for chr_no, chr in enumerate(self.unique_chr_map):
                    # test_r2_anc_c = stats.pearsonr(
                    #     preds_arr_chr[model_no, chr_no, :, prs_no],
                    #     labels_arr[model_no, :, prs_no],
                    # )
                    # log_dict[
                    #     "test_r2_pheno_" + str(prs_no) + "_alpha_" + str(alpha) + "_chr_" + str(chr)
                    # ] = test_r2_anc_c[0]

                log_dict[
                    "test_loss_pheno_" + str(prs_no) + "_alpha_" + str(alpha)
                ] = test_loss_anc.cpu().numpy() / len(preds_arr[model_no])
                log_dict[
                    "neff_pheno_" + str(prs_no) + "_alpha_" + str(alpha)
                ] = (self.var_y[prs_no] - self.var_covar_effect[prs_no].detach().cpu().numpy())/(self.var_y[prs_no] + np.std(preds_arr[model_no,:,prs_no])**2 - 2*test_r2_anc[0]*np.std(preds_arr[model_no,:,prs_no])*np.sqrt(self.var_y[prs_no]))
        return log_dict

    def save_exact_blup_estimates(self, best_alpha, out, test_r2_anc, correction_relatives):
        '''
        After training, this function computes and saves LOCO predictions for the entire dataset.
        '''
        ## Saves the exact LOCO estimates for the entire dataset in a text file
        ## best_alpha is a number_phenotypes x 1 vector indicating the best alpha index for each phenotype
        dim_out = len(self.h2)
        loco_estimates = np.zeros((self.num_chr, self.num_samples, dim_out))
        neff = np.zeros((self.num_chr, dim_out))
        covar_effect_arr = np.zeros((self.num_samples, dim_out))
        with torch.no_grad():
            prev = 0
            for input, covar_effect, label in self.test_dataloader:
                input = input.to(self.device)
                covar_effect_arr[prev: prev + len(input)] = covar_effect.numpy()
                for model_no, model in enumerate(self.model_list):
                    chr_no = model_no % self.num_chr
                    phen_no = self.pheno_for_model[model_no // self.num_chr]
                    spike = torch.clamp(model.sc1.spike1, 1e-6, 1.0 - 1e-6)
                    mu = model.fc1.weight
                    beta = spike.mul(mu)
                    preds = (input[:, self.chr_map != self.unique_chr_map[chr_no]]) @ (
                        beta.T
                    )
                    ## caution: remove sigmoid and covar_effect to save in regenie format
                    loco_estimates[chr_no][prev : prev + len(input)][:, phen_no] = preds.detach().cpu().numpy()
                prev += len(input)

            ## Saving it Regenie style...
            for d in range(dim_out):
                df = pd.DataFrame(loco_estimates[:,:, d])
                df.columns = self.df_iid_fid.values
                df_concat = pd.concat(
                    [pd.DataFrame(columns=['FID_IID'], data = self.unique_chr_map).astype('int'), df], axis=1
                )
                pd.DataFrame(df_concat).to_csv(
                    out + "_" + str(d+1) + ".loco", sep=" ", index=None
                )

            for chr_no in range(self.num_chr):
                loco_estimates[chr_no] += covar_effect_arr
                if self.args.binary:
                    loco_estimates[chr_no] = scipy.special.expit(loco_estimates[chr_no])

            for model_no, model in enumerate(self.model_list):
                chr_no = model_no % self.num_chr
                phen_no = self.pheno_for_model[model_no // self.num_chr]
                # num_snps_ratio = math.sqrt(sum(self.chr_map != self.unique_chr_map[chr_no])/len(self.chr_map))
                for prs_no in np.where(phen_no)[0]:
                    num = self.var_y[prs_no] - self.var_covar_effect[prs_no].detach().cpu().numpy()
                    #denom = self.var_y[prs_no] + np.std(loco_estimates[chr_no, :, prs_no])**2 - 2*test_r2_anc[prs_no, chr_no]*np.std(loco_estimates[chr_no, :, prs_no])*np.sqrt(self.var_y[prs_no])
                    # denom = self.var_y[prs_no]*(1 - test_r2_anc[prs_no, chr_no]**2) ## correction 1
                    denom = test_r2_anc[prs_no, chr_no]
                    neff[chr_no, prs_no] = num*correction_relatives[prs_no]/denom
            
            pd.DataFrame(data = neff.T, columns = np.array(self.unique_chr_map, dtype='int')).to_csv(out + '.neff', sep = ' ', index=None)

    def train_epoch(self, epoch):
        '''
        Main training function for whole-genome regression
        '''
        logging.info("Epoch: " + str(epoch+1) + "/" + str(self.args.alpha_search_epochs))
        for input, covar_effect, label in self.train_dataloader:
            log_dict = {}
            input, covar_effect, label = (
                input.to(self.device),
                covar_effect.to(self.device),
                label.to(self.device),
            )
            mask = ~(torch.isnan(label).all(axis=1))  ## remove samples with all nans
            input, covar_effect, label = input[mask], covar_effect[mask], label[mask]
            h2 = self.h2.unsqueeze(0).unsqueeze(0).repeat(2, label.shape[0], 1)
            var_covar_effect = self.var_covar_effect.unsqueeze(0).unsqueeze(0).repeat(2, label.shape[0], 1)
            label_4x = label.unsqueeze(0).repeat(2, 1, 1)
            covar_effect_4x = covar_effect.unsqueeze(0).repeat(2, 1, 1)
            mse_loss_arr = []
            reg_loss_arr = []
            total_loss_arr = []
            for model_no, model in enumerate(self.model_list):
                preds, reg_loss = model(input, covar_effect_4x, binary=self.args.binary)
                mse_loss = self.loss_func(preds, label_4x, h2*(1-var_covar_effect)+var_covar_effect)
                for k in range(self.args.forward_passes - 1):
                    preds, _ = model(
                        input,
                        covar_effect_4x,
                        only_output=True,
                        binary=self.args.binary,
                    )
                    mse1 = self.loss_func(preds, label_4x, h2*(1-var_covar_effect)+var_covar_effect)
                    mse_loss = mse_loss + mse1
                mse_loss = (
                    mse_loss / 2 / self.args.forward_passes
                )  ## adding extra 2 or 4 cause of antithetic variates
                loss = mse_loss + reg_loss
                ## store the mse loss and reg loss
                mse_loss_arr.append(mse_loss.item())
                reg_loss_arr.append(reg_loss.item())
                total_loss_arr.append(loss.item())
                ## Backprop ##
                self.optimizer_list[model_no].zero_grad()
                loss.backward()
                self.optimizer_list[model_no].step()

            ### Logging everything ###
            if (
                not self.never_validate
                and self.wandb_step
                % (self.validate_every * self.num_samples // self.args.batch_size)
                == 0
            ):
                log_dict = self.log_r2_loss(log_dict)

            if not self.never_validate and (
                self.wandb_step % 50 == 0
                or self.wandb_step
                % (self.validate_every * self.num_samples // self.args.batch_size)
                == 0
            ):
                for model_no, alpha in enumerate(self.alpha):
                    with torch.no_grad():
                        log_dict["train_mse_loss_" + str(alpha)] = mse_loss_arr[
                            model_no
                        ] / len(label)
                        log_dict["train_kl_loss_" + str(alpha)] = reg_loss_arr[
                            model_no
                        ] / len(label)
                        log_dict["train_total_loss_" + str(alpha)] = total_loss_arr[
                            model_no
                        ] / len(label)
                        log_dict["mean_sigma_" + str(alpha)] = torch.mean(
                            F.relu(self.model_list[model_no].sc1.sigma1)
                        ).cpu()
                        log_dict["mean_slab_" + str(alpha)] = torch.mean(
                            torch.clamp(
                                self.model_list[model_no].sc1.spike1, 1e-12, 1 - 1e-12
                            )
                        ).cpu()
                        grad_norm = 0
                        for p in list(
                            filter(
                                lambda p: p.grad is not None,
                                self.model_list[model_no].parameters(),
                            )
                        ):
                            grad_norm += p.grad.data.norm(2).item()
                        log_dict["grad_norm_" + str(alpha)] = grad_norm

                if not self.args.wandb_mode == "disabled":
                    wandb.log(log_dict, step=self.wandb_step)
            self.wandb_step += 1

        if hasattr(self, "scheduler_list"):
            for scheduler in self.scheduler_list:
                scheduler.step(epoch=epoch)
        return log_dict

    def train_epoch_loco(self, epoch):
        '''
        Main training function for the LOCO regression
        '''
        logging.info("Epoch: " + str(epoch+1) + "/" + str(self.args.num_epochs))
        for input, covar_effect, label in self.train_dataloader:
            # st = time.time()
            input, covar_effect, label = (
                input.to(self.device),
                covar_effect.to(self.device),
                label.to(self.device),
            )
            mask = ~(torch.isnan(label).all(axis=1))  ## remove samples with all nans
            input, covar_effect, label = input[mask], covar_effect[mask], label[mask]
            h2 = self.h2.unsqueeze(0).unsqueeze(0).repeat(2, label.shape[0], 1)
            var_covar_effect = self.var_covar_effect.unsqueeze(0).unsqueeze(0).repeat(2, label.shape[0], 1)
            label_4x = label.unsqueeze(0).repeat(2, 1, 1)
            covar_effect_4x = covar_effect.unsqueeze(0).repeat(2, 1, 1)
            for model_no, model in enumerate(self.model_list):
                input_loco_chr = torch.hstack(
                    (
                        input[:, 0 : self.chr_loc[model_no % self.num_chr]],
                        input[:, self.chr_loc[model_no % self.num_chr + 1] :],
                    )
                )
                preds, reg_loss = model(
                    input_loco_chr,
                    covar_effect_4x[
                        :, :, self.pheno_for_model[model_no // self.num_chr]
                    ],
                    binary=self.args.binary,
                )
                mse_loss = self.loss_func(
                    preds,
                    label_4x[:, :, self.pheno_for_model[model_no // self.num_chr]],
                    h2[:, :, self.pheno_for_model[model_no // self.num_chr]]*(1-var_covar_effect[:, :, self.pheno_for_model[model_no // self.num_chr]])+var_covar_effect[:, :, self.pheno_for_model[model_no // self.num_chr]],
                )
                loss = 0.5 * mse_loss + reg_loss

                ## Backprop ##
                self.optimizer_list[model_no].zero_grad()
                loss.backward()
                self.optimizer_list[model_no].step()

                del loss
                del input_loco_chr
                del preds
                del mse_loss
                del reg_loss
                model.zero_grad(set_to_none=True)
                # torch.cuda.empty_cache()
            
            # print(time.time() - st)

        if hasattr(self, "scheduler_list"):
            for scheduler in self.scheduler_list:
                scheduler.step(epoch=epoch)
        return None


def initialize_model(
    alpha_list,
    std_genotype,
    std_y,
    h2,
    dim_out,
    num_samples,
    device,
    loco=None,
    chr_map=None,
    mu=None,
    spike=None,
    predBetasFlag=False
):
    '''
     If chr_map is provided, the unique chromosomes are identified. 
     Additionally, if predBetasFlag is set to True, an extra chromosome is added to the list, 
     possibly for handling global effects.
    '''
    model_list = []
    num_snps = len(std_genotype)
    if chr_map is not None:
        unique_chr = torch.unique(chr_map).tolist()
        if predBetasFlag: 
            unique_chr.append(max(unique_chr) + 1)

    '''
    If h2 (heritability) has 2 dimensions, indicating different heritabilities for different SNPs, 
    both std_genotype and std_y are adjusted to add an extra dimension for compatibility.

    Multiple Phenotypes: In studies analyzing multiple traits or phenotypes simultaneously, 
    each phenotype might have a distinct heritability estimate. 
    A two-dimensional h2 array can store these varying estimates, 
    with one dimension indexing the phenotypes and the other representing the heritability estimates for each phenotype.
    '''

    if h2.ndim == 2:
        std_genotype = std_genotype.unsqueeze(0)
        std_y = std_y.unsqueeze(1)

    # std_y = 1 ## CAUTION!!!
    '''For each alpha value in alpha_list, which may control the sparsity of the model:'''
    for alpha_no, alpha in enumerate(alpha_list):
        '''
        Prior and Posterior Standard Deviations: For each alpha, the prior and posterior standard deviations (prior_sig and posterior_sig) 
        of the model weights are calculated based on std_genotype, std_y, and h2. 
        These calculations differ depending on the dimensionality of h2.
        '''
        if h2.ndim == 1:
            prior_sig = math.sqrt(1 / len(std_genotype) / alpha) / std_genotype
            prior_sig = torch.outer(std_y * torch.sqrt(h2), prior_sig)

            posterior_sig = math.sqrt(1 / len(std_genotype)) / std_genotype
            posterior_sig = torch.outer(
                std_y / (torch.sqrt(1 / (1 - h2) + alpha / h2)), posterior_sig
            )
        elif h2.ndim == 2:
            ## h2.shape = #phenotypes x #SNPs
            prior_sig = math.sqrt(1 / alpha) / std_genotype
            prior_sig = (std_y * torch.sqrt(h2)) * prior_sig

            posterior_sig = 1 / std_genotype
            posterior_sig = (
                std_y
                / (
                    torch.sqrt(
                        num_snps / (1 - torch.sum(h2, axis=1, keepdim=True))
                        + alpha / h2
                    )
                )
            ) * posterior_sig

        assert posterior_sig.shape == prior_sig.shape
        
        '''
        Model Instantiation:
        If LOCO analysis is specified (loco == "exact"), a model is created for each chromosome, excluding the current chromosome's 
        SNPs from the input dimension (dim_in). This is achieved by filtering chr_map != chr.
        
        If not in LOCO mode, a single model is created with input dimension equal to the total number of SNPs 
        and output dimension equal to the length of h2 (number of phenotypes or traits).
        '''
        if loco == "exact":
            assert len(dim_out)
            for chr_no, chr in enumerate(unique_chr):
                '''
                Optional parameters mu and spike are set if provided. 
                These parameters might represent the mean and sparsity (respectively) of the posterior distribution 
                of the model's weights, possibly derived from prior knowledge or a pre-training step.
                '''
                model = Model(
                    dim_in=len(chr_map[chr_map != chr]),
                    dim_out=int(sum(dim_out == alpha_no)),
                    num_samples=num_samples,
                    alpha=alpha,
                    prior_sig=prior_sig[dim_out == alpha_no][:, chr_map != chr].to(
                        device
                    ),
                    posterior_sig=posterior_sig[dim_out == alpha_no][
                        :, chr_map != chr
                    ].to(device),
                    mu=torch.as_tensor(mu[dim_out == alpha_no][:, chr_map != chr])
                    .float()
                    .to(device)
                    if mu is not None
                    else None,
                    spike=torch.as_tensor(spike[dim_out == alpha_no][:, chr_map != chr])
                    .float()
                    .to(device)
                    if spike is not None
                    else None,
                )
                model = model.to(device)
                # model = torch.compile(model)
                model_list.append(model)
        else:
            model = Model(
                dim_in=num_snps,
                dim_out=len(h2),
                num_samples=num_samples,
                alpha=alpha,
                prior_sig=prior_sig.to(device),
                posterior_sig=posterior_sig.to(device),
                mu=mu,
                spike=spike,
            )
            model = model.to(device)
            # model = torch.compile(model)
            model_list.append(model)
    '''The function returns the list of initialized models (model_list), ready for training or inference.'''
    return model_list

def hyperparam_search(args, alpha, h2, train_dataset, test_dataset, device="cuda"):
    '''
    The hyperparam_search function performs sparsity search
    '''
    # Define datasets and dataloaders
    # logging.info("Loading the data to RAM..")
    start_time = time.time()
    dim_out = train_dataset.output.shape[1]
    if np.ndim(h2) == 0:
        h2 = np.array([h2] * dim_out)
    assert len(h2) == dim_out
    if len(args.lr) != len(alpha) and len(args.lr) == 1:
        args.lr = args.lr * len(alpha)
    # logging.info("Done loading in: " + str(time.time() - start_time) + " secs")
    # Define model and enable wandb live policy
    std_genotype = torch.as_tensor(train_dataset.std_genotype).float()
    if args.binary:
        std_y = 1
    else:
        std_y = torch.sqrt(1 - torch.std(train_dataset.covar_effect, axis=0).float()**2)
    h2 = torch.as_tensor(h2).float()

    '''
    Initializes a list of models using the initialize_model function, with each model configured according to a specific alpha 
    value and other parameters like the standard deviations, heritability, and sample sizes.

    The hyperparameter search utilizes the entire dataset (all chromosomes) but splits into 90% train and 10% test data
    '''
    model_list = initialize_model(
        alpha,
        std_genotype,
        std_y,
        h2,
        dim_out,
        torch.sum(~(torch.isnan(train_dataset.output).all(axis=1))),
        device,
    )
    # Define trainer for training the model
    logging.info("Starting search for optimal alpha...")
    start_time = time.time()
    
    '''
    A Trainer instance is created to manage the model training process
    '''
    trainer = Trainer(
        args,
        alpha,
        h2.to(device),
        train_dataset,
        test_dataset,
        model_list,
        lr=args.lr,
        device=device,
        validate_every=args.validate_every,
        chr_map = train_dataset.chr,
        predBetasFlag=args.predBetasFlag
    )
    ##caution!!
    log_dict = {}
    
    '''
    Iterates over a specified number of epochs (args.alpha_search_epochs), 
    during which the models are trained using the training dataset.
    '''
    for epoch in tqdm(range(args.alpha_search_epochs)):
        log_dict = trainer.train_epoch(epoch)
    
    ## re-evaluate loss and r2 and the end of training
    '''
    After training, the function re-evaluates loss and R² (coefficient of determination) metrics to assess model performance. 
    These metrics are calculated for each phenotype and alpha value, providing insight into which alpha leads to the best 
    model performance in terms of prediction accuracy and error minimization.
    '''
    log_dict = trainer.log_r2_loss(log_dict)
    output_loss = np.zeros((dim_out, len(alpha)))
    output_r2 = np.zeros((dim_out, len(alpha)))
    # output_r2_chr = np.zeros((trainer.num_chr, dim_out, len(alpha)))
    output_neff = np.zeros((dim_out, len(alpha)))
    for prs_no in range(dim_out):
        for model_no, alpha_i in enumerate(alpha):
            output_loss[prs_no, model_no] = log_dict[
                "test_loss_pheno_" + str(prs_no) + "_alpha_" + str(alpha_i)
            ]
            output_r2[prs_no, model_no] = log_dict[
                "test_r2_pheno_" + str(prs_no) + "_alpha_" + str(alpha_i)
            ]
            output_neff[prs_no, model_no] = log_dict[
                "neff_pheno_" + str(prs_no) + "_alpha_" + str(alpha_i)
            ]
            # for chr_no, chr in enumerate(trainer.unique_chr_map):
            #     output_r2_chr[chr_no, prs_no, model_no] = log_dict[
            #         "test_r2_pheno_" + str(prs_no) + "_alpha_" + str(alpha_i) + "_chr_" + str(chr)
            #     ]

    logging.info("Best R2 across all alpha values: " + str(np.max(output_r2, axis=1)))
    logging.info("Best MSE across all alpha values: " + str(np.min(output_loss, axis=1)))
    logging.info("Best Neff across all alpha values: " + str(np.max(output_neff, axis=1)))
    
    '''
    Identifies the best alpha values by finding those that minimize the loss for each phenotype. 
    The effective number of samples (neff_best) corresponding to the best alpha values is saved for further use.
    '''
    best_alpha = np.argmin(output_loss, axis=1)
    neff_best = np.array([output_neff[i, best_alpha[i]] for i in range(dim_out)])
    np.savetxt(args.out + '.step1.neff', neff_best)

    # r2_best = np.array([output_r2_chr[:, i, best_alpha[i]] for i in range(dim_out)])
    
    '''
    Extracts the mean (mu) and spike parameters from the models corresponding to the best alpha values. 
    These parameters represent the learned weights and their sparsity, respectively.
    
    Calculates R² values for each chromosome using the best models, providing insight into the predictive performance 
    across different genomic regions.
    '''
    mu_list = torch.zeros((dim_out, len(std_genotype))).to(device)
    spike_list = torch.zeros((dim_out, len(std_genotype))).to(device)
    for prs_no in range(dim_out):
        mu = (
            trainer.model_list[best_alpha[prs_no]]
            .fc1.weight[prs_no]
            # .cpu()
            # .detach()
            # .numpy()
        )
        spike = (
            torch.clamp(
                trainer.model_list[best_alpha[prs_no]].sc1.spike1[prs_no],
                1e-6,
                1.0 - 1e-6,
            )
            # .cpu()
            # .detach()
            # .numpy()
        )
        mu_list[prs_no] = mu
        spike_list[prs_no] = spike

    r2_best = trainer.get_chr_r2(mu_list*spike_list)
    mu_list = mu_list.cpu().detach().numpy()
    spike_list = spike_list.cpu().detach().numpy()

    for i in range(len(model_list)):
        del model_list[0]
    del model_list
    
    if device == 'cuda':
    # torch._dynamo.reset()
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    logging.info("Done search for alpha in: " + str(time.time() - start_time) + " secs")
    return -output_loss, mu_list, spike_list, r2_best


def blr_spike_slab(args, h2, hdf5_filename, device="cuda"):
    '''
    The blr_spike_slab function encapsulates the entire process of performing Bayesian Linear Regression (BLR) with spike-and-slab priors, 
    specifically tailored for genetic data analysis. It includes steps for hyperparameter tuning, model training on the entire 
    dataset, and the computation of Bayesian estimates. 
    '''
    overall_start_time = time.time()
    if not args.wandb_mode == "disabled":
        logging.info("Initializing wandb to log the progress..")
        wandb.init(
            mode=args.wandb_mode,
            project=args.wandb_project_name,
            entity=args.wandb_entity_name,
            job_type=args.wandb_job_type,
            config=args,
            dir=args.out,
        )

    alpha = args.alpha #[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    assert len(alpha) == len(args.lr), "Length of sparsity parameters should be equal to learning rates provided"
    '''Creates a dictionary mapping each alpha value to its corresponding learning rate.'''
    lr_dict = {}
    for a, l in zip(alpha, args.lr):
        lr_dict[a] = l

    ''' 
    Initializes training and testing datasets using the HDF5Dataset class, specifying parameters like split type, file name, 
    batch size, and memory management options.
    '''
    train_dataset = HDF5Dataset(
        split="train",
        filename=hdf5_filename,
        phen_col="phenotype",
        batch_size=args.batch_size,
        transform=None,
        train_split=args.train_split,
        lowmem=args.lowmem
    )
    test_dataset = HDF5Dataset(
        split="test",
        filename=hdf5_filename,
        phen_col="phenotype",
        batch_size=args.batch_size,
        transform=None,
        train_split=args.train_split,
        lowmem=args.lowmem
    )

    '''
    Hyperparameter Search (Optional Grid Search for Heritability):
    If args.h2_grid is enabled, performs a grid search over specified heritability (h2) values to find the optimal setting.
    For each heritability value in the grid, it runs the hyperparam_search function to find the best alpha values and 
    captures the corresponding R² scores, mean (mu), and spike parameters.
    Determines the best heritability setting based on maximum R² scores or other criteria and updates the parameters accordingly.
    '''
    if args.h2_grid:
        h2_grid = np.array([0.01, 0.25, 0.5, 0.75])
        logging.info("Starting a grid search for heritability within BLR...")
        for i, h2_i in enumerate(h2_grid):
            logging.info("Bayesian linear regression ({0}/{1}) with h2 = {2}".format(i+1, len(h2_grid), h2_i))
            output_r2_i, mu_i, spike_i, r2_best_i = hyperparam_search(
                args, alpha, h2_i, train_dataset, test_dataset, device=device
            )
            if i == 0:
                output_r2 = copy.deepcopy(output_r2_i)
                mu = copy.deepcopy(mu_i)
                spike = copy.deepcopy(spike_i)
                h2 = [h2_i] * len(mu_i)
                r2_best = copy.deepcopy(r2_best_i)
            else:
                for phen in range(len(mu_i)):
                    if np.max(output_r2_i[phen]) > np.max(output_r2[phen]):
                        mu[phen] = mu_i[phen]
                        spike[phen] = spike_i[phen]
                        h2[phen] = h2_i
                        r2_best[phen] = r2_best_i[phen]
                output_r2 = np.maximum(output_r2, output_r2_i)
    else:
        output_r2, mu, spike, r2_best = hyperparam_search(
            args, alpha, h2, train_dataset, test_dataset, device=device
        )

    if args.lowmem:
        train_dataset.close_hdf5()
        test_dataset.close_hdf5()

    # np.savetxt(args.out + ".step1.output_r2", output_r2)
    # np.savetxt(args.out + ".step1.mu", mu)
    # np.savetxt(args.out + ".step1.spike", spike)
    # np.savetxt(args.out + ".step1.r2_best", r2_best)

    # output_r2 = np.loadtxt(args.out + ".step1.output_r2")
    # mu = np.loadtxt(args.out + ".step1.mu")
    # spike = np.loadtxt(args.out + ".step1.spike")
    # r2_best = np.loadtxt(args.out + ".step1.r2_best")

    np.savetxt(args.out + ".alpha", np.array(alpha)[np.argmax(output_r2, axis=1)])
    if args.h2_grid:
        np.savetxt(args.out + ".h2", np.array(h2))
    
    logging.info("")
    logging.info("Heritability inferred for traits = " + str(np.array(h2)))
    logging.info("")
    logging.info("Sparsity inferred for traits = " + str(np.array(alpha)[np.argmax(output_r2, axis=1)]))
    logging.info("")

    del train_dataset.hap1
    del train_dataset.hap2
    del train_dataset.output
    del train_dataset.covar_effect
    del train_dataset.geno_covar_effect
    del train_dataset.covars
    del test_dataset.hap1
    del test_dataset.hap2
    del test_dataset.output
    del test_dataset.covar_effect
    del test_dataset.geno_covar_effect
    del test_dataset.covars
    gc.collect()

    if device == 'cuda':
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()
    alpha = np.array(alpha)[np.unique(np.argmax(output_r2, axis=1))]

    output_r2_subset = output_r2[:, np.unique(np.argmax(output_r2, axis=1))]
    pheno_for_model = []
    for j in range(len(alpha)):
        pheno_for_model.append(np.argmax(output_r2_subset, axis=1) == j)

    ## output_loss_subset.shape = number of phenotypes x len(best_alpha)
    logging.info("Training on entire data with best alphas: " + str(alpha))

    '''
    Model Training on Entire Dataset:
    Re-initializes the full dataset including both training and test data.
    Prepares standard genotype and phenotype data, along with heritability estimates, for model initialization.
    Initializes models for the entire dataset based on the best alpha values identified earlier, taking into account specific configurations like LOCO analysis.
    Trains the models on the entire dataset using the Trainer class with configurations for LOCO analysis and the best learning rates.
    
    mu and spike are initialised from the hyperparam search
    '''

    # logging.info("Loading the data to RAM..")
    start_time = time.time()
    full_dataset = HDF5Dataset(
        split="both",
        filename=hdf5_filename,
        phen_col="phenotype",
        batch_size=args.batch_size,
        transform=None,
        train_split=args.train_split,
        lowmem=args.lowmem
    )
    # logging.info("Done loading in: " + str(time.time() - start_time) + " secs")
    std_genotype = torch.as_tensor(
        full_dataset.std_genotype, dtype=torch.float32
    )  # .to(device)
    if args.binary:
        std_y = 1
    else:
        std_y = torch.sqrt(1 - torch.std(full_dataset.covar_effect, axis=0).float()**2)
    h2 = torch.as_tensor(h2, dtype=torch.float32)  # .to(device)
    chr_map = full_dataset.chr  # .to(device)
    num_chr = len(torch.unique(chr_map))

    if args.predBetasFlag: num_chr += 1
    
    ### caution: turn off transfer learning for spike
    model_list = initialize_model(
        alpha,
        std_genotype,
        std_y,
        h2,
        np.argmax(output_r2_subset, axis=1),
        torch.sum(~(torch.isnan(full_dataset.output).all(axis=1))),
        device,
        'exact',
        chr_map,
        mu=mu,
        spike=spike,
        predBetasFlag=args.predBetasFlag
    )
    del mu
    del spike
    gc.collect()
    
    if device == 'cuda':
        with torch.no_grad():
            torch.cuda.empty_cache()

    logging.info("Calculating estimates using entire dataset...")
    start_time = time.time()
    lr_loco = []
    for a in alpha:
        for chr_no in range(num_chr):
            lr_loco.append(lr_dict[a])

    trainer = Trainer(
        args,
        alpha,
        h2.to(device),
        full_dataset,
        full_dataset,
        model_list,
        lr=lr_loco,
        device=device,
        validate_every=-1,
        chr_map=chr_map.to(device),
        pheno_for_model=pheno_for_model,
        predBetasFlag=args.predBetasFlag
    )
    
    for epoch in tqdm(range(args.num_epochs)):
        _ = trainer.train_epoch_loco(epoch)

    ## Calculate correction for relatives
    if args.kinship is not None:
        kinship = pd.read_csv(args.kinship, sep=r'\s+')
        h2 = h2.numpy()
        if args.binary:
            ## convert from liability to observed scale
            prev = np.mean(full_dataset.output.numpy(), axis=0)
            prev = np.minimum(prev, 1-prev)
            z = norm.pdf(norm.ppf(1 - prev))
            h2 = h2/(prev * (1 - prev) / (z**2))

        correction_relatives = get_correction_for_relatives(full_dataset.iid[:, 0], kinship, h2)
        logging.info('Sample size correction factor due to relatives = ' + str(correction_relatives))
    else:
        logging.info('--kinship wasnt provided so not correcting sample size estimate for relatives, this might lead to small inflation..')
        correction_relatives = np.ones_like(h2)


    ## Calculate estimates

    logging.info("Done fitting the model in: " + str(time.time() - start_time) + " secs")

    logging.info("Saving exact LOCO estimates...")
    start_time = time.time()
    trainer.save_exact_blup_estimates(
        np.argmax(output_r2_subset, axis=1), args.out, r2_best, correction_relatives=correction_relatives
    )
    logging.info(
        "Done writing exact LOCO estimates in: "
        + str(time.time() - start_time)
        + " secs"
    )

    '''
    If args.predBetasFlag is true, calculates the Best Linear Unbiased Prediction (BLUP) Betas for the entire dataset 
    using the trained models. This involves extracting the mean (mu) and spike parameters from the models corresponding 
    to the best alpha values and computing the product to obtain the Betas.

    Calculation of BLUP Betas: The BLUP Betas are computed as the product of mu and spike parameters.
    '''

    if args.predBetasFlag:
        logging.info("Calculating the BLUP Betas using the entire data...")
        dim_out = full_dataset.output.shape[1]
        best_alpha = np.argmax(output_r2_subset, axis=1)
        mu_list = np.zeros((dim_out, len(std_genotype)))
        spike_list = np.zeros((dim_out, len(std_genotype)))
        for prs_no in range(dim_out):
            output_no = np.where(np.where(best_alpha == best_alpha[prs_no])[0] == prs_no)[0][0]
            mu = (
                trainer.model_list[best_alpha[prs_no]*num_chr + num_chr - 1]
                .fc1.weight[output_no]
                .cpu()
                .detach()
                .numpy()
            )
            spike = (
                torch.clamp(
                    trainer.model_list[best_alpha[prs_no]*num_chr + num_chr - 1].sc1.spike1[output_no],
                    1e-6,
                    1.0 - 1e-6,
                )
                .cpu()
                .detach()
                .numpy()
            )
            mu_list[prs_no] = mu
            spike_list[prs_no] = spike
        
        beta = mu_list * spike_list
    else:
        beta = None 

    wandb.finish()
    if args.lowmem:
        full_dataset.close_hdf5()
    
    return beta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## wandb arguments
    wandb_group = parser.add_argument_group("WandB")
    wandb_mode = wandb_group.add_mutually_exclusive_group()
    wandb_mode.add_argument(
        "--wandb_offline",
        dest="wandb_mode",
        default=None,
        action="store_const",
        const="offline",
    )
    wandb_mode.add_argument(
        "--wandb_disabled",
        dest="wandb_mode",
        default=None,
        action="store_const",
        const="disabled",
    )
    wandb_group.add_argument(
        "--wandb_project_name",
        help="wandb project name",
        default="blr_genetic_association",
    )
    wandb_group.add_argument(
        "--wandb_job_type",
        help="Wandb job type. This is useful for grouping runs together.",
        default=None,
    )

    ## hyperparameters arguments
    parser.add_argument(
        "--num_epochs", help="number of epochs to train for", type=int, default=60
    )
    parser.add_argument(
        "--lr",
        help="Learning rate of the optimizer",
        type=float,
        nargs="+",
        default=[
            4e-4,
            2e-4,
            2e-4,
            1e-4,
            2e-5,
            5e-6,
        ],
    )
    parser.add_argument(
        "-lr_min",
        "--min_learning_rate",
        help="Minimum learning rate for cosine scheduler",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "-scheduler",
        "--cosine_scheduler",
        help="Cosine scheduling the outer learning rate",
        type=str_to_bool,
        default="false",
    )
    parser.add_argument(
        "--batch_size", help="Batch size of the dataloader", type=int, default=128
    )
    parser.add_argument(
        "--forward_passes", help="Number of forward passes in blr", type=int, default=1
    )
    parser.add_argument(
        "--gpu", help="Which GPU card do you wish to use ?", type=str, default="0"
    )
    parser.add_argument(
        "--num_workers",
        help="torch.utils.data.DataLoader num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--train_split",
        help="The training split proportion in (0,1)",
        type=float,
        default=0.8,
    )
    ## path arguments
    parser.add_argument(
        "--hdf5_filename",
        help="File name of the HDF5 dataset",
        default="demo_simulation.hdf5",
    )
    parser.add_argument("--h2_file", type=str, help="File containing estimated h2")
    parser.add_argument(
        "--output",
        "-o",
        help="prefix for where to save any results or files",
        default="out",
    )

    args = parser.parse_args()

# python blr.py --hdf5_filename  --h2_file