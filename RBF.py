import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import random
from nats_bench import create


class RadialBasisFunctionKernel:
    """Radial Basis Function (RBF) kernel for Gaussian Process Regression."""
    def __init__(self, length_scale=1.0, signal_variance=1.0):
        # Initialize the RBF kernel with length scale and signal variance.
        self.length_scale = length_scale
        self.signal_variance = signal_variance


    def __call__(self, x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray | float:
        X1 = np.asarray(x_i)
        X2 = np.asarray(x_j)
        scalar_result = X1.ndim == 1 and X2.ndim == 1
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        squared_distances = np.sum(
            (X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2,
            axis=-1,
        )
        result = self.signal_variance * np.exp(
            -0.5 * squared_distances / (self.length_scale ** 2)
        )
        return float(result[0, 0]) if scalar_result else result