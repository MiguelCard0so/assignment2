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

    def __call__(self, x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray | float:
        # Compute all pairwise Hamming distances between the input rows.
        X1 = np.asarray(x_i)
        X2 = np.asarray(x_j)
        scalar_result = X1.ndim == 1 and X2.ndim == 1
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        pairwise_mismatches = np.count_nonzero(
            X1[:, np.newaxis, :] != X2[np.newaxis, :, :],
            axis=-1,
        )

        if self.normalize:
            vector_length = X1.shape[-1]
            pairwise_mismatches = pairwise_mismatches / vector_length

        result = self.signal_variance * np.exp(
            -pairwise_mismatches / self.lenght_scale
        )
        return float(result[0, 0]) if scalar_result else result