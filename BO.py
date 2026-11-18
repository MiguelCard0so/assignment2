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
        self.K_inv = None # Inverse of the kernel matrix for training data.

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Fit the GPR model to the training data.
        self.X_train = X_train
        self.y_train = y_train
        K = self._compute_kernel_matrix(X_train)
        K += self.noise_variance * np.eye(len(X_train))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test: np.ndarray):
        # Compute the mean and covariance of the predictions for the test data.
        try:
            K_s = self._compute_kernel_matrix(self.X_train, X_test)
            K_ss = self._compute_kernel_matrix(X_test)

        except (self.X_train is None, self.y_train is None) as e:
            print(f"Error occurred while predicting: {e}")
            raise
        mu_s = K_s.T @ self.K_inv @ self.y_train
        cov_s = K_ss - K_s.T @ self.K_inv @ K_s
        return mu_s, cov_s

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        # Compute the kernel matrix between two sets of input points
        try:
            if X2 is None:
                X2 = X1
            n1, n2 = len(X1), len(X2)
            K = np.zeros((n1, n2))
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.kernel(X1[i], X2[j])
        except (X1 is None, X2 is None) as e:
            print(f"Error occurred while computing kernel matrix: {e}")
            raise
        return K