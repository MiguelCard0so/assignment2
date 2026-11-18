import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import random
from nats_bench import create

class HammingDistanceKernel:
    def __init__(self, signal_variance=1.0, lenght_scale=1.0, normalize=True):
        # Initialize the Hamming Distance kernel with signal variance, length scale, and normalization.
        self.signal_variance = signal_variance
        self.lenght_scale = lenght_scale
        self.normalize = normalize

    def __call__(self, x_i: np.ndarray, x_j: np.ndarray) -> float:
        # Compute the Hamming distance between two input points and return the kernel value.
        X1 = np.atleast_2d(x_i)
        X2 = np.atleast_2d(x_j)

        # Summing over the last axis counts the mismatches -> shape (N, K)
        pairwise_mismatches = np.sum(X1[:, np.newaxis, :] != X2[np.newaxis, :, :], axis=-1)

        if self.normalize:
            vector_lenght = X1.shape[-1]
            pairwise_mismatches = pairwise_mismatches / vector_lenght

        return self.signal_variance * np.exp(-pairwise_mismatches / self.lenght_scale)