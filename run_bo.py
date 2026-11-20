"""Bayesian optimization (GP + Thompson sampling) on NATS-Bench (topology search space).

Validation accuracy drives every decision (GP targets, acquisition, incumbent).
Test accuracy is only recorded for plotting / reporting.
"""

import csv
import os
import pickle
import re
from time import time

import matplotlib.pyplot as plt
import numpy as np
from nats_bench import create

from BO import GaussianProcessRegressor
from HDK import HammingDistanceKernel
from hamming import maximin_hamming_design
from ThompsonSampling import ThompsonSampling
import argparse

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BENCH_PATH = os.environ.get('NATS_BENCH_PATH', 'NATS-tss-v1_0-3ffb9-simple')

SUPPORTED_DATASETS = ("cifar10-valid", "cifar100", "ImageNet16-120")

IS_RANDOM = False
DATASET = "cifar10-valid"
INITIAL_LENGTH = 10
CANDIDATE_POOL_SIZE = 256
RANDOM_SEED = 42
OUTPUT_DIR = "results/BO"

OPTIMIZATION_SPACE = {
    "none": 0,
    "skip_connect": 1,
    "nor_conv_1x1": 2,
    "nor_conv_3x3": 3,
    "avg_pool_3x3": 4,
}

# Key names differ between datasets and training schedules. Order = priority.
# On 200-epoch schedules, cifar100 / ImageNet16-120 report the test metric as
# "valtest-accuracy" instead of "test-accuracy".
VALID_KEYS = ("valid-accuracy", "x-valid-accuracy")
TEST_KEYS = ("test-accuracy", "x-test-accuracy", "valtest-accuracy", "ori-test-accuracy")


# --------------------------------------------------------------------------- #
# Dataset / metric handling
# --------------------------------------------------------------------------- #
def dataset_label(dataset):
    """Name used in file names / titles ('cifar10-valid' -> 'cifar10')."""
    return dataset.removesuffix("-valid")


def _first_available(info, keys):
    """Return the first non-None value found under any of `keys`, else None."""
    for key in keys:
        if info.get(key) is not None:
            return float(info[key])
    return None


def query_accuracies(api, index, dataset, hp, iepoch=None):
    """Return (validation_accuracy, test_accuracy) for one architecture.

    `iepoch=None` means the last epoch of the `hp` training schedule.
    """
    info = api.get_more_info(index, dataset, iepoch=iepoch, hp=hp, is_random=IS_RANDOM)

    valid_acc = _first_available(info, VALID_KEYS)
    if valid_acc is None:
        raise ValueError(
            f"No validation accuracy for {dataset!r} (hp={hp}). Available keys: {sorted(info)}"
        )

    test_acc = _first_available(info, TEST_KEYS)
    if test_acc is None and dataset == "cifar10-valid":
        # Some benchmark files omit the test split on 'cifar10-valid';
        # the same test set is available through plain 'cifar10'.
        fallback = api.get_more_info(index, "cifar10", iepoch=iepoch, hp=hp, is_random=IS_RANDOM)
        test_acc = _first_available(fallback, TEST_KEYS)
    if test_acc is None:
        raise ValueError(
            f"No test accuracy for {dataset!r} (hp={hp}). Available keys: {sorted(info)}"
        )

    return valid_acc, test_acc


# --------------------------------------------------------------------------- #
# Search space encoding
# --------------------------------------------------------------------------- #
def architecture_to_features(architecture):
    """Convert an architecture string to a feature vector."""
    operation_names = re.findall(r"([A-Za-z][A-Za-z0-9_]*)~\d+", architecture)
    if not operation_names:
        raise ValueError(f"No operations found in architecture: {architecture!r}")
    try:
        return [OPTIMIZATION_SPACE[name] for name in operation_names]
    except KeyError as error:
        raise ValueError(f"Unknown operation in architecture: {error.args[0]}") from error


def build_feature_matrix(api):
    return np.array([architecture_to_features(api.arch(i)) for i in range(len(api))])


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def fit_surrogate(X_train, Y_train):
    # Alternative kernel (import RadialBasisFunctionKernel from RBF):
    # kernel = RadialBasisFunctionKernel(length_scale=1.0, signal_variance=1.0)
    kernel = HammingDistanceKernel(signal_variance=1.0, lenght_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, noise_variance=1e-10)
    gpr.fit(X_train, Y_train)
    return gpr


# --------------------------------------------------------------------------- #
# Search loop
# --------------------------------------------------------------------------- #
def make_entry(index, iteration, phase, valid_acc, test_acc):
    return {
        "architecture_index": int(index),
        "iteration": iteration,
        "phase": phase,
        "performance": float(valid_acc),   # validation accuracy
        "test_accuracy": float(test_acc),
    }


def search(api, X_all, dataset, hp, budget, initial_length, rng):
    # ---- initial design -------------------------------------------------- #
    initial_idxs = [
        int(i) for i in maximin_hamming_design(X_all=X_all, n_samples=initial_length, rng=rng)
    ]
    history = []
    Y_train = []
    for idx in initial_idxs:
        valid_acc, test_acc = query_accuracies(api, idx, dataset, hp)
        Y_train.append(valid_acc)
        history.append(make_entry(idx, 0, "initial", valid_acc, test_acc))

    Y_train = np.array(Y_train)
    X_train = X_all[initial_idxs]
    observed = set(initial_idxs)

    best_pos = int(np.argmax(Y_train))
    best = {
        "index": initial_idxs[best_pos],
        "architecture": api.arch(initial_idxs[best_pos]),
        "validation_accuracy": float(Y_train[best_pos]),
        "test_accuracy": history[best_pos]["test_accuracy"],
    }

    # ---- BO loop --------------------------------------------------------- #
    gpr = None
    for iteration in range(1, budget + 1):
        gpr = fit_surrogate(X_train, Y_train)

        remaining = [i for i in range(len(api)) if i not in observed]
        if not remaining:
            raise ValueError("The evaluation budget exceeds the number of unobserved architectures.")

        candidate_indices = rng.choice(
            remaining, min(CANDIDATE_POOL_SIZE, len(remaining)), replace=False
        )
        X_candidates = X_all[candidate_indices]

        position = ThompsonSampling(gpr).select_next_architecture(api, X_candidates, rng)
        next_idx = int(candidate_indices[position])

        valid_acc, test_acc = query_accuracies(api, next_idx, dataset, hp)

        observed.add(next_idx)
        X_train = np.vstack([X_train, X_all[next_idx]])
        Y_train = np.append(Y_train, valid_acc)
        history.append(make_entry(next_idx, iteration, "bayesian_optimization", valid_acc, test_acc))

        if valid_acc > best["validation_accuracy"]:
            best = {
                "index": next_idx,
                "architecture": api.arch(next_idx),
                "validation_accuracy": valid_acc,
                "test_accuracy": test_acc,
            }

    return best, history, gpr


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def incumbent_test_curve(history):
    """Test accuracy of the incumbent (best *validation* so far), per iteration."""
    best_valid, best_test = -np.inf, None
    iterations, values = [], []
    for entry in history:
        if entry["performance"] > best_valid:
            best_valid, best_test = entry["performance"], entry["test_accuracy"]
        if entry["phase"] == "initial":
            continue
        iterations.append(entry["iteration"])
        values.append(best_test)
    # Point at iteration 0: incumbent after the initial design only.
    initial = [e for e in history if e["phase"] == "initial"]
    top = max(initial, key=lambda e: e["performance"])
    return [0] + iterations, [top["test_accuracy"]] + values


def plot_history(history, dataset, path):
    initial = [e for e in history if e["phase"] == "initial"]
    optimized = [e for e in history if e["phase"] == "bayesian_optimization"]

    figure, axis = plt.subplots(constrained_layout=True)
    axis.scatter(
        [e["iteration"] for e in initial], [e["test_accuracy"] for e in initial],
        color="0.55", alpha=0.7, label="Initial evaluations",
    )
    axis.scatter(
        [e["iteration"] for e in optimized], [e["test_accuracy"] for e in optimized],
        color="tab:blue", alpha=0.7, label="BO evaluations",
    )
    iterations, values = incumbent_test_curve(history)
    axis.step(iterations, values, where="post", color="tab:green", linewidth=2,
              label="Best observed (selected on validation)")
    axis.set(
        xlabel="Bayesian optimization iteration",
        ylabel="Test accuracy",
        title=f"Bayesian optimization on {dataset_label(dataset)}",
    )
    axis.legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_outputs(output_dir, dataset, history, gpr):
    os.makedirs(output_dir, exist_ok=True)
    label = dataset_label(dataset)

    if gpr is not None:
        with open(os.path.join(output_dir, "gpr_model.pkl"), "wb") as f:
            pickle.dump(gpr, f)

    with open(os.path.join(output_dir, f"bo_results_{label}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    plot_history(history, dataset, os.path.join(output_dir, f"bayesian_optimization_performance_on_{label}.png"))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(
    budget_epochs=10000,
    initial_length=INITIAL_LENGTH,
    dataset=DATASET,
    seed=RANDOM_SEED,
    output_dir=None,
    api=None,
    X_all=None,
):
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset must be one of {SUPPORTED_DATASETS}, got {dataset!r}.")
    if api is None:
        api = create(BENCH_PATH, "tss", fast_mode=False, verbose=False)
    if X_all is None:
        X_all = build_feature_matrix(api)

    n_evals = budget_epochs // 200                    # one evaluation = one full 200-epoch training
    initial_length = min(initial_length, max(1, n_evals // 2))
    budget = n_evals - initial_length                 # BO iterations after the initial design
    rng = np.random.default_rng(seed)

    started_at = time()
    best, history, gpr = search(api, X_all, dataset, "200", budget, initial_length, rng)
    elapsed = time() - started_at

    if output_dir is not None:
        save_outputs(output_dir, dataset, history, gpr)

    return {
        "best_index": best["index"],
        "best_architecture": best["architecture"],
        "test_acc": best["test_accuracy"],
        "epochs_used": n_evals * 200,
        "elapsed_seconds": elapsed,
        "history": history,
    }


if __name__ == "__main__":
    run()