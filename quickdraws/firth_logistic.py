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


import numpy as np
from scipy.stats import chi2
import time
import pdb
import copy


def _loglikelihood_se_svt(X, y, preds):
    # penalized log-likelihood
    W = preds * (1 - preds)
    I = np.sum(W * X[:, 0] * X[:, 0])

    penalty = 0.5 * np.log(I)
    se = np.sqrt(1 / I)
    return -1 * (np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds)) + penalty), se


def firth_logit_svt(
    X_orig, y_all, offset_all, covars=None, max_iter=250, max_stepsize=5, tol=1e-4
):
    """
    Firth logistic regression for single variant testing
    X_orig: Genotype vector shape N
    y_all: Binary outcome matrix shape NxP
    offset_all: Offset matrix shape NxP
    covars: Covariate matrix shape NxC
    output: Weights, log-likelihood, iters
    """
    weights_all = np.zeros(y_all.shape[1])
    loglike_null = np.zeros(y_all.shape[1])
    loglike_all = np.zeros(y_all.shape[1])
    iters_all = np.zeros(y_all.shape[1])
    se_all = np.zeros(y_all.shape[1])

    ## Regress out covars from X

    for p in range(y_all.shape[1]):
        y = y_all[:, p]
        offset = offset_all[:, p]

        if covars is not None:
            y_pred_null = 1 / (1 + np.exp(-offset))
            Tau = y_pred_null * (1 - y_pred_null)
            X = X_orig - (
                covars
                @ (
                    np.linalg.inv((covars.T * Tau) @ covars)
                    @ (covars.T @ (X_orig.T * Tau).T)
                )
            )
        # Initialize weights
        weights = np.zeros(X.shape[1])

        # Perform gradient descent
        for iter in range(max_iter):
            z = np.dot(X, weights)
            y_pred = 1 / (1 + np.exp(-z - offset))

            # Calculate Fisher information matrix
            W = y_pred * (1 - y_pred)
            I = np.sum(W * X[:, 0] * X[:, 0])
            I_inv = 1 / I

            hat_diag = W * (X[:, 0] ** 2) * I_inv

            # Calculate U_star
            U_star = (y - y_pred + hat_diag * (0.5 - y_pred)) @ X

            step_size = I_inv * U_star

            # step-halving
            mx = np.max(np.abs(step_size)) / max_stepsize
            if mx > 1:
                step_size = step_size / mx  # restrict to max_stepsize
            weights_new = weights + step_size

            z = np.dot(X, weights_new)
            preds_new = 1 / (1 + np.exp(-z - offset))

            loglike, _ = _loglikelihood_se_svt(X, y, y_pred)
            if iter == 0:
                loglike_null[p] = -loglike

            loglike_new, se = _loglikelihood_se_svt(X, y, preds_new)

            while loglike < loglike_new:
                step_size *= 0.5
                weights_new = weights + step_size
                z = np.dot(X, weights_new)
                preds_new = 1 / (1 + np.exp(-z - offset))
                loglike_new, se = _loglikelihood_se_svt(X, y, preds_new)

            if iter > 1 and np.linalg.norm(weights_new - weights) < tol:
                weights_all[p] = weights_new
                loglike_all[p] = -loglike_new
                iters_all[p] = iter
                se_all[p] = se
                break

            weights += step_size

        if iter == max_iter - 1:
            weights_all[p] = weights
            loglike_all[p] = -loglike
            iters_all[p] = max_iter
            se_all[p] = se
    return weights_all, se_all, loglike_all - loglike_null, iters_all


def _loglikelihood_covars(X, y, preds):
    # penalized log-likelihood
    W = preds * (1 - preds)
    I = np.zeros((X.shape[1], X.shape[1]))
    for i in range(I.shape[1]):
        for j in range(I.shape[0]):
            I[i, j] = np.sum(W * X[:, i] * X[:, j], axis=0)

    penalty = 0.5 * np.log(np.linalg.det(I))
    return -1 * (np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds)) + penalty)


def firth_logit_covars(X, y_all, offset_all, max_iter=1000, max_stepsize=25, tol=1e-5):
    """
    Firth logistic regression for 'only' covariates
    X: Covariate matrix shape NxC
    y_all: Binary outcome matrix shape NxP
    offset_all: Offset matrix shape NxP
    output: Weights, log-likelihood
    """
    preds_all = np.zeros(y_all.shape)
    loglike_all = np.zeros(y_all.shape[1])
    iters_all = np.zeros(y_all.shape[1])
    for p in range(y_all.shape[1]):
        y = y_all[:, p]
        offset = offset_all[:, p]

        # Initialize weights
        weights = np.zeros(X.shape[1])

        # Perform gradient descent
        for iter in range(max_iter):

            z = np.dot(X, weights)
            y_pred = 1 / (1 + np.exp(-z - offset))

            # Calculate Fisher information matrix
            W = y_pred * (1 - y_pred)
            I = np.zeros((X.shape[1], X.shape[1]))
            for i in range(I.shape[1]):
                for j in range(I.shape[0]):
                    I[i, j] = np.sum(W * X[:, i] * X[:, j], axis=0)

            # Find diagonal of Hat Matrix
            I_inv = np.linalg.inv(I)

            hat_diag = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                hat_diag[i] = np.sum(W[i] * X[i, :] @ I_inv @ X[i, :].T)

            # Calculate U_star
            U_star = np.matmul((y - y_pred + hat_diag * (0.5 - y_pred)), X)

            step_size = np.matmul(I_inv, U_star)

            # step-halving
            mx = np.max(np.abs(step_size)) / max_stepsize
            if mx > 1:
                step_size = step_size / mx  # restrict to max_stepsize
            weights_new = weights + step_size

            z = np.dot(X, weights_new)
            preds_new = 1 / (1 + np.exp(-z - offset))

            loglike = _loglikelihood_covars(X, y, y_pred)
            loglike_new = _loglikelihood_covars(X, y, preds_new)
            while loglike < loglike_new:
                step_size *= 0.5
                weights_new = weights + step_size
                z = np.dot(X, weights_new)
                preds_new = 1 / (1 + np.exp(-z - offset))
                loglike_new = _loglikelihood_covars(X, y, preds_new)

            if iter > 1 and np.linalg.norm(weights_new - weights) < tol:
                preds_all[:, p] = z + offset
                loglike_all[p] = -loglike_new
                iters_all[p] = iter
                break

            weights += step_size

        if iter == max_iter - 1:
            preds_all[:, p] = np.dot(X, weights) + offset
            loglike_all[p] = -loglike
            iters_all[p] = max_iter

    return preds_all, loglike_all, iters_all


if __name__ == "__main__":
    ## input = covars + loco_pred + SNP
    ## output = phenotype
    num_covars = 10
    num_pheno = 50
    num_samples = 50000

    covar = np.random.randn(num_samples, num_covars)
    geno = np.random.randn(num_samples, 1)
    y_all = np.zeros((num_samples, num_pheno))
    offset_all = np.zeros((num_samples, num_pheno))
    for p in range(y_all.shape[1]):
        alpha = np.random.randn(num_covars)
        offset = np.random.randn(num_samples)
        y = covar @ alpha + 0.1 * geno[:, 0] + offset
        y = (y > np.percentile(y, 90)).astype(int)
        y_all[:, p] = y
        offset_all[:, p] = 0.5 * offset

    st = time.time()
    pred_covars, loglike_null, iters_null = firth_logit_covars(covar, y_all, offset_all)
    print(time.time() - st)
    beta, se, loglike, iters = firth_logit_svt(geno, y_all, pred_covars, covar)
    lr_stat = 2 * loglike
    p_value = chi2.sf(lr_stat, df=1)
    print("X^2 = " + str(lr_stat) + " p-value = " + str(p_value))
    pdb.set_trace()
