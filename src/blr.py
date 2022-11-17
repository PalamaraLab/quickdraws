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
from scipy import stats
import math
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed


def str_to_bool(s: str) -> bool:
    return bool(strtobool(s))


## Custom layers:
class BBB_Linear_spike_slab(nn.Module):
    def __init__(
        self,
        num_samples,
        weights_shape: torch.Size,
        alpha,
        prior_sig: torch.Tensor,
        mu1,
    ):
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
        sig1 = F.relu(self.sigma1)
        weight_mu = mu1.weight
        spike = torch.clamp(self.spike1, 1e-6, 1.0 - 1e-6)
        # weight = self.reparameterize(weight_mu, sig1, weight_mu.device, test)
        # if test:
        #     out = F.linear(x, weight)
        # else:
        #     weight = weight.permute(0, 2, 1)
        #     out = torch.matmul(x.unsqueeze(0).repeat(4, 1, 1), weight)  ## 2xBxO
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
        eps = torch.cuda.FloatTensor(mu1.shape)
        torch.randn(mu1.shape, out=eps)
        sig_eps = torch.mul(sig1, eps)
        gaussian1 = mu1 + sig_eps
        gaussian2 = mu1 - sig_eps

        spike = torch.clamp(self.spike1, 1e-6, 1.0 - 1e-6)
        log_spike = torch.log(spike / (1 - spike))
        unif = torch.cuda.FloatTensor(spike.shape)
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
        # eps = torch.cuda.FloatTensor((x.shape[0], mu1.shape[0]))
        # torch.randn((x.shape[0], mu1.shape[0]), out=eps)
        eps = torch.cuda.FloatTensor(x.shape[0], mu1.shape[0]).normal_()
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
    def __init__(
        self,
        dim_in,
        dim_out,
        num_samples,
        alpha,
        prior_sig,
        posterior_sig=None,
        mu_var=None,
        mu=None,
        spike=None,
    ):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.sc1 = BBB_Linear_spike_slab(
            weights_shape=self.fc1.weight.shape,
            num_samples=num_samples,
            alpha=alpha,
            prior_sig=prior_sig,
            mu1=self.fc1,
        )
        if posterior_sig is not None:
            print("Setting posterior sigma to CAVI derived sigmas")
            self.sc1.sigma1 = nn.Parameter(posterior_sig, requires_grad=False)
        if mu_var is not None:
            print("Initializing posterior means based on variance of means from CAVI")
            self.fc1.weight.data = mu_var.to(self.fc1.weight.device) * (
                torch.randn(self.fc1.weight.shape).to(self.fc1.weight.device)
            )
        if mu is not None:
            print("Initializing posterior means through transfer learning")
            self.fc1.weight.data = mu.to(self.fc1.weight.device)
        if spike is not None:
            print("Initializing posterior spike through transfer learning")
            self.sc1.spike1.data = spike.to(self.sc1.spike1.device)

    def forward(self, x, only_output=False, test=False):
        num_batches = x.shape[0]
        x, reg = self.sc1(x, self.fc1, only_output, test)
        return x, reg * num_batches


## Dataloader class - efficient access of the data:
# Idea: In case the HDF5 file can't be entirely opened in RAM, use multiple HDF5 files and load one after another
# Idea: PBWT the input haplotype matrix, in the inverse PBWT at __getitem__
class HDF5Dataset:
    def __init__(
        self,
        split: str,
        filename: str,
        phen_col: str,
        transform: Optional[Callable] = None,
        train_split=0.8,
    ):
        ### Assumes the HDF5 dataset is already randomly shuffled
        ### Assumes the HDF5 dataset has mean_genotype and std_genotype computed

        if split not in ["train", "test", "both"]:
            raise ValueError("Invalid value for split argument")
        self.split = split
        self.filename = filename
        self.transform = transform
        h5py_file = h5py.File(self.filename, "r")
        assert h5py_file["hap1"].shape == h5py_file["hap2"].shape
        total_samples = len(h5py_file["hap1"])
        train_samples = int(train_split * total_samples)

        if split == "train":
            self.hap1 = np.array(h5py_file["hap1"][0:train_samples])
            self.hap2 = np.array(h5py_file["hap2"][0:train_samples])
            self.output = torch.as_tensor(
                np.array(h5py_file[phen_col][0:train_samples], dtype=float)
            ).float()
        elif split == "test":
            self.hap1 = np.array(h5py_file["hap1"][train_samples:])
            self.hap2 = np.array(h5py_file["hap2"][train_samples:])
            self.output = torch.as_tensor(
                np.array(h5py_file[phen_col][train_samples:], dtype=float)
            ).float()
        else:
            self.hap1 = np.array(h5py_file["hap1"][:])
            self.hap2 = np.array(h5py_file["hap2"][:])
            self.output = torch.as_tensor(
                np.array(h5py_file[phen_col][:], dtype=float)
            ).float()

        # self.output = self.output[:, 1]  ## 1 phenotype
        # self.output = self.output[:, 1:11]  ## 10 phenotypes
        # self.output = torch.tile(self.output[:, 1:], (1, 2))[:, 0:50]  ## 50 phenotypes
        # self.output = torch.tile(self.output[:, 1:], (1, 4))[
        #     :, 0:100
        # ]  ## 100 phenotypes
        # self.output = torch.tile(self.output[:, 1:], (1, 8))[
        #     :, 0:200
        # ]  ## 200 phenotypes

        # self.output = self.output[:, 1:6]  ##fixing it to only 1st phenotype
        self.std_genotype = np.maximum(np.array(h5py_file["std_genotype"]), 1e-6)
        self.mean_genotype = torch.as_tensor(
            np.array(h5py_file["mean_genotype"])
        ).float()
        self.num_snps = len(self.mean_genotype)
        self.chr = torch.as_tensor(np.array(h5py_file["chr"])).float()
        ### Using the mean and std genotype calculated on entire dataset, doesn't change the results
        self.length = len(self.output)
        h5py_file.close()

        if self.output.ndim == 1:
            self.output = self.output.reshape(-1, 1)

    def __getitem__(self, index: int) -> Any:
        input = (
            torch.as_tensor(np.unpackbits(self.hap1[index])).float()
            + torch.as_tensor(np.unpackbits(self.hap2[index])).float()
        )
        input = input[0 : self.num_snps] - self.mean_genotype
        output = self.output[index]
        if self.transform is not None:
            input = self.transform(input)
        return (input, output)

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
        adam_eps=1e-8,
        weight_decay=0,
        validate_every=1,
        chr_map=None,
        pheno_for_model=None,
    ):
        self.args = args
        if h2.ndim == 2:
            self.h2 = torch.sum(h2, axis=1)
        else:
            self.h2 = h2
        self.device = device
        self.model_list = model_list
        self.num_samples = train_dataset.length
        self.validate_every = validate_every if validate_every > 0 else 1
        self.never_validate = validate_every < 0
        self.chr_map = chr_map
        self.alpha = alpha
        if self.chr_map is not None:
            self.unique_chr_map = torch.unique(self.chr_map)
            self.num_chr = len(self.unique_chr_map)

            ## check if chr_map has chrs in chunks:
            chr_map_shifted = torch.as_tensor(
                self.chr_map.tolist()[1:] + [torch.inf]
            ).to(self.chr_map.device)
            assert torch.sum((self.chr_map - chr_map_shifted) != 0) == self.num_chr

            self.chr_loc = []
            for chr in self.unique_chr_map:
                self.chr_loc.append(int(min(torch.where(self.chr_map == chr)[0])))
            self.chr_loc.append(len(chr_map))

        if pheno_for_model is not None:
            self.pheno_for_model = pheno_for_model
        self.optimizer_list = []
        for model_no, model in enumerate(model_list):
            self.optimizer_list.append(
                torch.optim.Adam(
                    model.parameters(),
                    lr=lr[model_no // self.num_chr]
                    if len(lr) != len(model_list)
                    else lr[model_no],
                    eps=adam_eps,
                    weight_decay=weight_decay,
                )
            )
        if self.args.cosine_scheduler:
            self.scheduler_list = []
            for optimizer in self.optimizer_list:
                self.scheduler_list.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optimizer,
                        T_max=self.args.num_epochs,
                        eta_min=self.args.min_learning_rate,
                    )
                )
        self.mse_loss = nn.MSELoss(reduction="none")
        self.loss_func = self.masked_mse_loss  # nn.MSELoss(reduction="sum")  #
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

    def masked_mse_loss(self, preds, labels, h2=0.5):
        mask = (~torch.isnan(labels)) & (~torch.isnan(preds))
        sq_error = (preds[mask] - labels[mask]) ** 2  ## B x O
        if type(h2) == torch.Tensor:
            sq_error = sq_error / (2 * (1 - h2[mask]))
        else:
            sq_error = sq_error / (2 * (1 - h2))
        return torch.sum(sq_error)  ## reduction = sun

    # def masked_mse_loss(self, preds, labels, h2=0.5):
    #     mask = (~torch.isnan(labels)) & (~torch.isnan(preds))
    #     loss = self.mse_loss(labels, preds) / (2 * (1 - h2))
    #     loss = (loss[mask]).sum()
    #     return loss

    def validation(self):
        labels_arr = [[] for _ in range(len(self.model_list))]
        preds_arr = [[] for _ in range(len(self.model_list))]
        with torch.no_grad():
            for input, label in self.test_dataloader:
                input, label = input.to(self.device), label.to(self.device)
                for model_no, model in enumerate(self.model_list):
                    preds, _ = model(input, only_output=True, test=True)
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

        return (
            test_loss,
            preds_arr,
            labels_arr,
        )

    def log_r2_loss(self, log_dict):
        ## Validation once a epoch
        (
            test_loss,
            preds_arr,
            labels_arr,
        ) = self.validation()

        ## Average test loss
        for model_no, alpha in enumerate(self.alpha):
            log_dict["mean_test_loss_" + str(alpha)] = test_loss[model_no] / len(
                preds_arr[model_no]
            )
            log_dict["preds_std_" + str(alpha)] = np.std(preds_arr[model_no])

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
                log_dict[
                    "test_loss_pheno_" + str(prs_no) + "_alpha_" + str(alpha)
                ] = test_loss_anc.cpu().numpy() / len(preds_arr[model_no])
        return log_dict

    def save_residuals(self, best_alpha, out=None):
        ## Saves the residuals (y - \beta x) for the entire dataset in a text file
        ## best_alpha is a number_phenotypes x 1 vector indicating the best alpha index for each phenotype
        test_loss, preds_arr, labels_arr = self.validation()
        residuals = np.zeros((preds_arr.shape[1], preds_arr.shape[2]))
        for prs_no in range(preds_arr.shape[2]):
            residuals[:, prs_no] = (
                labels_arr[best_alpha[prs_no], :, prs_no]
                - preds_arr[best_alpha[prs_no], :, prs_no]
            )
        if out is not None:
            pd.DataFrame(residuals).to_csv(out + ".residuals", sep="\t")
        else:
            return residuals

    def save_blup_estimates(self, best_alpha, out=None):
        ## Saves the BLUP estimates \beta trained in a text file
        ## Add chromosome, position, allele's etc. later
        dim_out, dim_in = self.model_list[0].fc1.weight.shape
        beta = np.zeros((dim_in, dim_out))
        for prs_no in range(dim_out):
            spike = torch.clamp(
                self.model_list[best_alpha[prs_no]].sc1.spike1[prs_no], 1e-6, 1.0 - 1e-6
            )
            mu = self.model_list[best_alpha[prs_no]].fc1.weight[prs_no]
            beta[:, prs_no] = spike.mul(mu).detach().cpu().numpy()
        if out is not None:
            pd.DataFrame(beta).to_csv(out + ".blup", sep="\t")
        else:
            return beta

    def save_approx_blup_residuals(self, best_alpha, out):
        ## Saves the approximate LOCO residuals for the entire dataset in a text file
        ## best_alpha is a number_phenotypes x 1 vector indicating the best alpha index for each phenotype
        dim_out, dim_in = self.model_list[0].fc1.weight.shape
        loco_residuals = np.zeros(
            (len(torch.unique(self.chr_map)), self.num_samples, dim_out)
        )
        all_chr_residuals = self.save_residuals(best_alpha)

        with torch.no_grad():
            beta = torch.zeros((dim_in, dim_out), device="cuda")
            for prs_no in range(dim_out):
                spike = torch.clamp(
                    self.model_list[best_alpha[prs_no]].sc1.spike1[prs_no],
                    1e-6,
                    1.0 - 1e-6,
                )
                mu = self.model_list[best_alpha[prs_no]].fc1.weight[prs_no]
                beta[:, prs_no] = spike.mul(mu)

            assert len(beta) == len(self.chr_map)
            prev = 0
            for input, label in self.test_dataloader:
                input = input.to(self.device)
                for chr_no, chr in enumerate(torch.unique(self.chr_map)):
                    ### zero the betas for other chromosome
                    beta_c = beta[self.chr_map == chr]
                    beta_c = beta_c.float()
                    preds = (input[:, self.chr_map == chr]) @ beta_c
                    loco_residuals[chr_no, prev : prev + len(input)] = (
                        all_chr_residuals[prev : prev + len(input)]
                        + preds.detach().cpu().numpy()
                    )
                prev += len(input)
            for chr_no, chr in enumerate(torch.unique(self.chr_map)):
                pd.DataFrame(loco_residuals[chr_no]).to_csv(
                    out + "loco_chr" + str(int(chr)) + ".residuals", sep="\t"
                )

    def save_exact_blup_residuals(self, best_alpha, out):
        ## Saves the exact LOCO residuals for the entire dataset in a text file
        ## best_alpha is a number_phenotypes x 1 vector indicating the best alpha index for each phenotype
        dim_out = len(self.h2)
        loco_residuals = np.zeros((self.num_chr, self.num_samples, dim_out))
        with torch.no_grad():
            prev = 0
            for input, label in self.test_dataloader:
                input, label = input.to(self.device), label.to(self.device)
                for model_no, model in enumerate(self.model_list):
                    chr_no = model_no % self.num_chr
                    phen_no = self.pheno_for_model[model_no // self.num_chr]
                    spike = torch.clamp(model.sc1.spike1, 1e-6, 1.0 - 1e-6)
                    mu = model.fc1.weight
                    beta = spike.mul(mu)
                    preds = (input[:, self.chr_map != self.unique_chr_map[chr_no]]) @ (
                        beta.T
                    )
                    # label[:, phen_no][torch.isnan(label[:, phen_no])] = preds[
                    #     torch.isnan(label[:, phen_no])
                    # ]
                    loco_residuals[chr_no][prev : prev + len(input)][:, phen_no] = (
                        (label[:, phen_no] - preds).detach().cpu().numpy()
                    )
                prev += len(input)
            for chr_no, chr in enumerate(torch.unique(self.chr_map)):
                pd.DataFrame(loco_residuals[chr_no]).to_csv(
                    out + "loco_chr" + str(int(chr)) + ".residuals", sep="\t"
                )

    def train_epoch(self, epoch):
        for input, label in self.train_dataloader:
            log_dict = {}
            input, label = input.to(self.device), label.to(self.device)
            mask = ~(torch.isnan(label).all(axis=1))  ## remove samples with all nans
            input, label = input[mask], label[mask]
            h2 = self.h2.unsqueeze(0).unsqueeze(0).repeat(2, label.shape[0], 1)
            label_4x = label.unsqueeze(0).repeat(2, 1, 1)
            mse_loss_arr = []
            reg_loss_arr = []
            total_loss_arr = []
            for model_no, model in enumerate(self.model_list):
                preds, reg_loss = model(input)
                mse_loss = self.loss_func(label_4x, preds, h2)
                for k in range(self.args.forward_passes - 1):
                    preds, _ = model(input, only_output=True)
                    mse1 = self.loss_func(label_4x, preds, h2)
                    mse_loss = mse_loss + mse1
                mse_loss = (
                    mse_loss / 2 / self.args.forward_passes
                )  ## adding extra 2 or 4 cause of antithetic variates
                loss = mse_loss + reg_loss
                ## store the mse loss and reg loss
                mse_loss_arr.append(mse_loss)
                reg_loss_arr.append(reg_loss)
                total_loss_arr.append(loss)
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
                        ].item() / len(label)
                        log_dict["train_kl_loss_" + str(alpha)] = reg_loss_arr[
                            model_no
                        ].item() / len(label)
                        log_dict["train_total_loss_" + str(alpha)] = total_loss_arr[
                            model_no
                        ].item() / len(label)
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

                wandb.log(log_dict, step=self.wandb_step)
            self.wandb_step += 1

        if hasattr(self, "scheduler_list"):
            for scheduler in self.scheduler_list:
                scheduler.step(epoch=epoch)
        return log_dict

    def train_epoch_loco(self, epoch):
        for input, label in self.train_dataloader:
            log_dict = {}
            mse_loss_total, reg_loss_total, total_loss_total = 0, 0, 0
            input, label = input.to(self.device), label.to(self.device)
            mask = ~(torch.isnan(label).all(axis=1))  ## remove samples with all nans
            input, label = input[mask], label[mask]
            h2 = self.h2.unsqueeze(0).unsqueeze(0).repeat(2, label.shape[0], 1)
            label_4x = label.unsqueeze(0).repeat(2, 1, 1)
            for model_no, model in enumerate(self.model_list):
                input_loco_chr = torch.hstack(
                    (
                        input[:, 0 : self.chr_loc[model_no % self.num_chr]],
                        input[:, self.chr_loc[model_no % self.num_chr + 1] :],
                    )
                )
                preds, reg_loss = model(input_loco_chr)
                mse_loss = self.loss_func(
                    label_4x[:, :, self.pheno_for_model[model_no // self.num_chr]],
                    preds,
                    h2[:, :, self.pheno_for_model[model_no // self.num_chr]],
                )
                loss = 0.5 * mse_loss + reg_loss

                ## log losses
                mse_loss_total += mse_loss.item()
                reg_loss_total += reg_loss.item()
                total_loss_total += loss.item()

                ## Backprop ##
                self.optimizer_list[model_no].zero_grad()
                loss.backward()
                self.optimizer_list[model_no].step()

            if not self.never_validate and (self.wandb_step % 50 == 0):
                with torch.no_grad():
                    log_dict["train_mse_loss"] = (
                        mse_loss_total / len(self.model_list) / len(label)
                    )
                    log_dict["train_kl_loss"] = (
                        reg_loss_total / len(self.model_list) / len(label)
                    )
                    log_dict["train_total_loss"] = (
                        total_loss_total / len(self.model_list) / len(label)
                    )
                wandb.log(log_dict, step=self.wandb_step)
            self.wandb_step += 1

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
):
    model_list = []
    num_snps = len(std_genotype)
    if h2.ndim == 2:
        std_genotype = std_genotype.unsqueeze(0)
        std_y = std_y.unsqueeze(1)

    for alpha_no, alpha in enumerate(alpha_list):
        if h2.ndim == 1:
            prior_sig = math.sqrt(1 / len(std_genotype) / alpha) / std_genotype
            prior_sig = torch.outer(std_y * torch.sqrt(h2), prior_sig)

            posterior_sig = math.sqrt(1 / len(std_genotype)) / std_genotype
            posterior_sig = torch.outer(
                std_y / (torch.sqrt(1 / (1 - h2) + alpha / h2)), posterior_sig
            )

            mu_var = math.sqrt(1 / len(std_genotype)) / std_genotype
            mu_var = torch.outer(std_y / (1 + (1 - h2) * alpha / h2), mu_var)
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

            mu_var = math.sqrt(num_snps) / std_genotype
            mu_var = (
                std_y
                / (num_snps + (1 - torch.sum(h2, axis=1, keepdim=True)) * alpha / h2)
            ) * mu_var

        assert posterior_sig.shape == prior_sig.shape
        assert mu_var.shape == posterior_sig.shape
        if loco == "exact":
            assert len(dim_out)
            for chr_no, chr in enumerate(torch.unique(chr_map)):
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
                    mu_var=mu_var[dim_out == alpha_no][:, chr_map != chr].to(device),
                    mu=mu[dim_out == alpha_no][:, chr_map != chr].to(device)
                    if mu is not None
                    else None,
                    spike=spike[dim_out == alpha_no][:, chr_map != chr].to(device)
                    if spike is not None
                    else None,
                )
                model = model.to(device)
                model_list.append(model)
        else:
            model = Model(
                dim_in=num_snps,
                dim_out=len(h2),
                num_samples=num_samples,
                alpha=alpha,
                prior_sig=prior_sig.to(device),
                posterior_sig=posterior_sig.to(device),
                mu_var=mu_var.to(device),
                mu=mu,
                spike=spike,
            )
            model = model.to(device)
            model_list.append(model)
    return model_list


def hyperparam_search(args, alpha, h2, hdf5_filename):
    # Define datasets and dataloaders
    print("Loading the data to RAM..")
    start_time = time.time()
    train_dataset = HDF5Dataset(
        split="train",
        filename=hdf5_filename,
        phen_col="phenotype",
        transform=None,
        train_split=args.train_split,
    )
    dim_out = train_dataset.output.shape[1]
    assert len(h2) == dim_out
    if len(args.lr) != len(alpha) and len(args.lr) == 1:
        args.lr = args.lr * len(alpha)
    test_dataset = HDF5Dataset(
        split="test",
        filename=hdf5_filename,
        phen_col="phenotype",
        transform=None,
        train_split=args.train_split,
    )
    print("Done loading in: " + str(time.time() - start_time) + " secs")
    # get freer GPU among all possible options
    device = "cuda"
    # Define model and enable wandb live policy
    std_genotype = torch.as_tensor(train_dataset.std_genotype).float()
    std_y = torch.std(train_dataset.output, axis=0).float()
    h2 = torch.as_tensor(h2).float()

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
    print("Starting search for optimal alpha..")
    start_time = time.time()
    trainer = Trainer(
        args,
        alpha,
        h2.to(device),
        train_dataset,
        test_dataset,
        model_list,
        lr=args.lr,
        device=device,
        validate_every=-1,
    )
    ##caution!!
    for epoch in tqdm(range(30)):
        log_dict = trainer.train_epoch(epoch)

    print("Done search for alpha in: " + str(time.time() - start_time) + " secs")

    ## re-evaluate loss and r2 and the end of training
    log_dict = trainer.log_r2_loss(log_dict)
    output_loss = np.zeros((dim_out, len(alpha)))
    output_r2 = np.zeros((dim_out, len(alpha)))
    for prs_no in range(dim_out):
        for model_no, alpha_i in enumerate(alpha):
            output_loss[prs_no, model_no] = log_dict[
                "test_loss_pheno_" + str(prs_no) + "_alpha_" + str(alpha_i)
            ]
            output_r2[prs_no, model_no] = log_dict[
                "test_r2_pheno_" + str(prs_no) + "_alpha_" + str(alpha_i)
            ]

    print("Best R2 across all alpha values: " + str(np.max(output_r2, axis=1)))
    print("Best MSE across all alpha values: " + str(np.max(output_loss, axis=1)))

    best_alpha = np.argmin(output_loss, axis=1)
    mu_list = torch.zeros((dim_out, len(std_genotype)))
    spike_list = torch.zeros((dim_out, len(std_genotype)))
    for prs_no in range(dim_out):
        mu = trainer.model_list[best_alpha[prs_no]].fc1.weight[prs_no]
        spike = torch.clamp(
            trainer.model_list[best_alpha[prs_no]].sc1.spike1[prs_no], 1e-6, 1.0 - 1e-6
        )
        mu_list[prs_no] = mu
        spike_list[prs_no] = spike

    del train_dataset.hap1
    del train_dataset.hap2
    del test_dataset.hap1
    del test_dataset.hap2
    del model_list
    return output_loss, device, mu_list, spike_list


def blr_spike_slab(args, h2, hdf5_filename):
    overall_start_time = time.time()
    if not args.wandb_mode == "disabled":
        print("Initializing wandb to log the progress..")
        wandb.init(
            mode=args.wandb_mode,
            project=args.wandb_project_name,
            entity="hrushikeshloya",
            job_type=args.wandb_job_type,
            config=args,
            dir=args.output,
        )
        # Save ENV variables
        with (Path(wandb.run.dir) / "env.txt").open("wt") as f:
            pprint.pprint(dict(os.environ), f)

    ## hard-coding alpha values as in BOLT
    alpha = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    output_loss, device, mu, spike = hyperparam_search(args, alpha, h2, hdf5_filename)
    ### 🌵
    # device = "cuda"
    # output_loss = np.zeros((len(h2), len(alpha)))
    torch.cuda.empty_cache()

    ## Only run for best alpha's
    alpha = np.array(alpha)[np.unique(np.argmin(output_loss, axis=1))]
    output_loss_subset = output_loss[:, np.unique(np.argmin(output_loss, axis=1))]
    pheno_for_model = []
    for j in range(len(alpha)):
        pheno_for_model.append(np.argmin(output_loss_subset, axis=1) == j)
    ## output_loss_subset.shape = number of phenotypes x len(best_alpha)
    print("Training on entire data with best alphas: " + str(alpha))

    print("Loading the data to RAM..")
    start_time = time.time()
    full_dataset = HDF5Dataset(
        split="both",
        filename=hdf5_filename,
        phen_col="phenotype",
        transform=None,
        train_split=args.train_split,
    )
    print("Done loading in: " + str(time.time() - start_time) + " secs")
    std_genotype = torch.as_tensor(full_dataset.std_genotype).float().to(device)
    std_y = torch.std(full_dataset.output, axis=0).float().to(device)
    h2 = torch.as_tensor(h2).float().to(device)
    chr_map = full_dataset.chr.to(device)

    model_list = initialize_model(
        alpha,
        std_genotype,
        std_y,
        h2,
        np.argmin(output_loss_subset, axis=1),
        torch.sum(~(torch.isnan(full_dataset.output).all(axis=1))),
        device,
        args.loco,
        chr_map,
        mu=mu.to(device),
        spike=spike.to(device),
    )
    torch.cuda.empty_cache()

    print("Calculating Residuals using entire dataset..")
    start_time = time.time()
    trainer = Trainer(
        args,
        alpha,
        h2.to(device),
        full_dataset,
        full_dataset,
        model_list,
        lr=args.lr,
        device=device,
        validate_every=1,
        chr_map=chr_map,
        pheno_for_model=pheno_for_model,
    )
    if args.loco == "approx":
        for epoch in tqdm(range(args.num_epochs)):
            _ = trainer.train_epoch(epoch)
    else:
        for epoch in tqdm(range(args.num_epochs)):
            _ = trainer.train_epoch_loco(epoch)

    ## Calculate residuals

    print("Done fitting the model in: " + str(time.time() - start_time) + " secs")

    if args.loco == "approx":
        print("Writing BLUP estimates..")
        start_time = time.time()
        trainer.save_blup_estimates(np.argmin(output_loss_subset, axis=1), args.output)
        print(
            "Done writing BLUP estimates in: " + str(time.time() - start_time) + " secs"
        )

        print("Saving approximate LOCO residuals..")
        start_time = time.time()
        trainer.save_approx_blup_residuals(
            np.argmin(output_loss_subset, axis=1), args.output
        )
        print(
            "Done writing approximate LOCO estimates in: "
            + str(time.time() - start_time)
            + " secs"
        )
    else:
        print("Saving exact LOCO residuals..")
        start_time = time.time()
        trainer.save_exact_blup_residuals(
            np.argmin(output_loss_subset, axis=1), args.output
        )
        print(
            "Done writing exact LOCO estimates in: "
            + str(time.time() - start_time)
            + " secs"
        )

    print("Saving the learned model..")
    start_time = time.time()
    for (alpha_i, model) in zip(alpha, model_list):
        model_path = args.output + str("model_" + str(alpha_i) + ".pt")
        torch.save(model.state_dict(), model_path)
    print("Done saving the model in: " + str(time.time() - start_time) + " secs")
    wandb.finish()
    print("Total time elapsed: " + str(time.time() - overall_start_time) + " secs")
    return args.output + "loco_chr"


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
        default=[4e-4, 4e-4, 2e-4, 2e-4, 5e-5, 2e-5],
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
    parser.add_argument(
        "--loco",
        help="select the loco scheme",
        type=str,
        choices=["approx", "exact"],
        default="approx",
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
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = args.gpu.split(",")
    print("Using " + str(len(device)) + " GPU(s)...")

    assert args.train_split < 1 and args.train_split > 0
    h2 = np.loadtxt(args.h2_file)
    residuals_filename = blr_spike_slab(args, h2, args.hdf5_filename)

# python blr.py --hdf5_filename  --h2_file
