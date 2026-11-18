import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import random
from BO import GaussianProcessRegressor
from nats_bench import create


class ThompsonSampling:
    def __init__(self, gpr_model: GaussianProcessRegressor):
            # Initialize the Thompson Sampling acquisition function with a GPR model.
            self.gpr_model = gpr_model
    
    def select_next_architecture(self, api, candidate_architectures: np.ndarray):
        # Select the next architecture to evaluate based on Thompson Sampling.
        mu, cov = self.gpr_model.predict(candidate_architectures)
        sampled_values = np.random.multivariate_normal(mu, cov)
        return int(np.argmax(sampled_values))