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


    def __call__(self, x_i: np.ndarray, x_j: np.ndarray) -> float:
        distance = np.linalg.norm(x_i - x_j)
        return self.signal_variance * np.exp(-0.5 * (distance ** 2) / (self.length_scale ** 2))