import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import random
from nats_bench import create
from RBF import RadialBasisFunctionKernel
from HDK import HammingDistanceKernel

"""""""""""""""""""""""
 For Our Bayesian Optimization approach to solve the NATS Bench problem, we need to define the following:
    - A surrogate model: Gaussian Process Regression (GPR) model that approximates the performance of architectures based on their features.
    - An acquisition function: Thompson Sampling will be used to select the next architecture to evaluate based on the surrogate model's predictions.
    - An optimization loop that iteratively selects new architectures to evaluate based on the surrogate model and acquisition function.

    For our GPR Kernel, we will use the Radial Basis Function (RBF) kernel, which is commonly used in GPR due to its smoothness properties. The RBF kernel is defined as:
    k(x_i, x_j) = σ_f² * exp(-1/(2l²) * ||x_i - x_j||²)

    where σ_f² is the signal variance, l is the length scale, and ||x_i - x_j|| is the Euclidean distance between two input points x_i and x_j.
    
    """""""""""""""""""""""

class GaussianProcessRegressor:
    """Gaussian Process Regression (GPR) model."""
    def __init__(self, kernel: RadialBasisFunctionKernel | HammingDistanceKernel, noise_variance=1e-10):
        # Initialize the GPR model with a kernel and noise variance.
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.L = None
        self.alpha = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Fit the GPR model to the training data.
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).reshape(-1)
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must contain the same number of samples.")

        K = self._compute_kernel_matrix(X_train)
        K += self.noise_variance * np.eye(len(X_train))
        jitter = 1e-12
        for _ in range(6):
            try:
                L = np.linalg.cholesky(K + jitter * np.eye(len(K)))
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        else:
            raise np.linalg.LinAlgError(
                "Kernel matrix is not positive definite after adding jitter."
            )

        alpha = np.linalg.solve(
            L.T,
            np.linalg.solve(L, y_train),
        )
        self.X_train = X_train
        self.y_train = y_train
        self.L = L
        self.alpha = alpha

    def predict(self, X_test: np.ndarray):
        # Compute the mean and covariance of the predictions for the test data.
        if self.L is None or self.alpha is None:
            raise RuntimeError("The Gaussian process must be fitted before prediction.")

        X_test = np.asarray(X_test)
        K_s = self._compute_kernel_matrix(self.X_train, X_test)
        K_ss = self._compute_kernel_matrix(X_test)
        v = np.linalg.solve(self.L, K_s)
        mu_s = K_s.T @ self.alpha
        cov_s = K_ss - v.T @ v
        cov_s = (cov_s + cov_s.T) / 2
        return mu_s, cov_s

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        # Kernels accept batches and return the full pairwise matrix.
        if X2 is None:
            X2 = X1
        return self.kernel(X1, X2)