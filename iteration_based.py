"""Iteration-based learning-curve method (successive halving + power-law extrapolation)
on NATS-Bench (topology search space).

Validation accuracy drives all decisions (curve fitting, pruning, final pick).
Test accuracy is only recorded for plotting / reporting.
"""

import csv
import math
import os
import pickle
import random
from time import time

import matplotlib.pyplot as plt
import numpy as np
from nats_bench import create
from scipy.optimize import least_squares

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BENCH_PATH = os.path.join(
    os.path.dirname(__file__),
    "NATS-tss-v1_0-3ffb9-simple",
    "NATS-tss-v1_0-3ffb9-simple",
)

SUPPORTED_DATASETS = ("cifar10-valid", "cifar100", "ImageNet16-120")

HP_CONSTANT = 1                                # hp = extrapolation target = HP_CONSTANT * MAX_EPOCHS
MAX_EPOCHS = 200                               # last checkpoint = full training
CHECKPOINTS = (1, 3, 9, 27, 81, MAX_EPOCHS)
CANDIDATE_COUNT = 81
PROMOTION_FACTOR = 3
PRUNING_START_RUNG = 2                         # first rung at which pruning happens
RANDOM_SEED = 42
DATASET = "cifar10-valid"
OUTPUT_DIR = "results/iteration_based"


# --------------------------------------------------------------------------- #
# Dataset / metric handling
# --------------------------------------------------------------------------- #
def dataset_label(dataset):
    """Name used in file names / titles ('cifar10-valid' -> 'cifar10')."""
    return dataset.removesuffix("-valid")


# Key names differ between datasets and training schedules. Order = priority.
# On 200-epoch schedules, cifar100 / ImageNet16-120 report the test metric as
# "valtest-accuracy" instead of "test-accuracy".
VALID_KEYS = ("valid-accuracy", "x-valid-accuracy", "valtest-accuracy")
TEST_KEYS = ("test-accuracy", "x-test-accuracy", "valtest-accuracy", "ori-test-accuracy")


def _first_available(info, keys):
    """Return the first non-None value found under any of `keys`, else None."""
    for key in keys:
        if info.get(key) is not None:
            return float(info[key])
    return None


def query_accuracies(api, index, dataset, hp, iepoch=None):
    """Return (validation_accuracy, test_accuracy) for one architecture.

    NATS-Bench datasets expose slightly different keys:
      - cifar10-valid : valid-accuracy (x-valid), test-accuracy (ori-test, may be missing)
      - cifar100 / ImageNet16-120 : valid-accuracy (x-valid), test-accuracy (x-test)
    `iepoch=None` means the last epoch of the `hp` training schedule.
    """
    info = api.get_more_info(index, dataset, iepoch=iepoch, hp=hp, is_random=False)

    valid_acc = _first_available(info, VALID_KEYS)
    if valid_acc is None:
        raise ValueError(
            f"No validation accuracy for {dataset!r} (hp={hp}). Available keys: {sorted(info)}"
        )

    test_acc = _first_available(info, TEST_KEYS)
    if test_acc is None and dataset == "cifar10-valid":
        # Some benchmark files omit the test split on 'cifar10-valid';
        # the same test set is available through plain 'cifar10'.
        fallback = api.get_more_info(index, "cifar10", iepoch=iepoch, hp=hp, is_random=False)
        test_acc = _first_available(fallback, TEST_KEYS)
    if test_acc is None:
        raise ValueError(
            f"No test accuracy for {dataset!r} (hp={hp}). Available keys: {sorted(info)}"
        )

    return valid_acc, test_acc


# --------------------------------------------------------------------------- #
# Learning-curve extrapolation
# --------------------------------------------------------------------------- #
def power_law_curve(epochs, asymptote, scale, exponent):
    return asymptote - scale * np.power(epochs, -exponent)


def predict_final_accuracy(epochs, accuracies, max_epochs=MAX_EPOCHS):
    epochs = np.asarray(epochs, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    target_epoch = HP_CONSTANT * max_epochs

    if len(epochs) < 3:
        return float(accuracies[-1])

    scale_guess = max(float(np.ptp(accuracies)), 1e-3)
    x0 = (float(accuracies[-1] + scale_guess), scale_guess, 0.5)
    try:
        fit = least_squares(
            lambda p: power_law_curve(epochs, *p) - accuracies,
            x0=x0,
            bounds=([-np.inf, 0.0, 0.01], [np.inf, np.inf, 2.0]),
            max_nfev=20000,
        )
        if not fit.success:
            return float(accuracies[-1])
        prediction = power_law_curve(target_epoch, *fit.x)
        return max(float(accuracies[-1]), float(prediction))
    except (RuntimeError, ValueError, FloatingPointError):
        return float(accuracies[-1])


# --------------------------------------------------------------------------- #
# Validation of arguments
# --------------------------------------------------------------------------- #
def validate_arguments(candidate_count, promotion_factor, checkpoints, dataset):
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset must be one of {SUPPORTED_DATASETS}, got {dataset!r}.")
    if len(checkpoints) < 4 or checkpoints[-1] != MAX_EPOCHS:
        raise ValueError(f"Use at least four checkpoints ending at {MAX_EPOCHS} epochs.")
    if any(e < 1 for e in checkpoints) or any(a >= b for a, b in zip(checkpoints, checkpoints[1:])):
        raise ValueError("Checkpoints must be positive and strictly increasing.")
    if promotion_factor < 2 or candidate_count < promotion_factor:
        raise ValueError("candidate_count must be >= promotion_factor, which must be >= 2.")


# --------------------------------------------------------------------------- #
# Search loop
# --------------------------------------------------------------------------- #
def sample_candidates(api, candidate_count, seed):
    rng = random.Random(seed)
    indices = rng.sample(range(len(api)), candidate_count)
    return [
        {
            "index": i,
            "architecture": api.arch(i),
            "epochs": [],
            "accuracies": [],        # validation (used by the algorithm)
            "test_accuracies": [],   # test (plotting / reporting only)
            "predicted_final": None,
        }
        for i in indices
    ]


def evaluate_at_checkpoint(api, candidate, epoch, dataset, hp):
    """Query one candidate at `epoch`, update its curve and return the epochs spent."""
    iepoch = None if epoch >= MAX_EPOCHS else epoch - 1
    valid_acc, test_acc = query_accuracies(api, candidate["index"], dataset, hp, iepoch=iepoch)

    previous_epoch = candidate["epochs"][-1] if candidate["epochs"] else 0
    candidate["epochs"].append(epoch)
    candidate["accuracies"].append(valid_acc)
    candidate["test_accuracies"].append(test_acc)
    candidate["predicted_final"] = predict_final_accuracy(
        candidate["epochs"], candidate["accuracies"]
    )
    return epoch - previous_epoch


def prune(candidates, promotion_factor):
    candidates = sorted(candidates, key=lambda c: c["predicted_final"], reverse=True)
    keep = max(1, math.ceil(len(candidates) / promotion_factor))
    return candidates[:keep]


def search(api, candidates, checkpoints, promotion_factor, dataset, hp):
    active = list(candidates)
    history = []
    resource_used = 0

    for rung, epoch in enumerate(checkpoints):
        for candidate in active:
            resource_used += evaluate_at_checkpoint(api, candidate, epoch, dataset, hp)
            history.append(
                {
                    "architecture_index": candidate["index"],
                    "epoch": epoch,
                    "validation_accuracy": candidate["accuracies"][-1],
                    "test_accuracy": candidate["test_accuracies"][-1],
                    "predicted_final_accuracy": candidate["predicted_final"],
                }
            )

        if PRUNING_START_RUNG <= rung < len(checkpoints) - 1:
            active = prune(active, promotion_factor)

    best = max(active, key=lambda c: c["accuracies"][-1])  # selected on validation
    return best, history, resource_used


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def save_outputs(output_dir, dataset, candidates, best, history):
    os.makedirs(output_dir, exist_ok=True)
    label = dataset_label(dataset)

    with open(os.path.join(output_dir, "iteration_based_model.pkl"), "wb") as f:
        pickle.dump(candidates, f)

    with open(
        os.path.join(output_dir, f"power_law_history_{label}.csv"), "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    figure, axis = plt.subplots()
    for candidate in candidates:
        axis.plot(candidate["epochs"], candidate["test_accuracies"], color="0.75", alpha=0.45)
    axis.plot(
        best["epochs"], best["test_accuracies"], color="tab:blue", marker="o", label="Best finalist"
    )
    axis.set(
        xlabel="Training epoch",
        ylabel="Test accuracy",
        title=f"Iteration Based model on {label}",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        os.path.join(output_dir, f"power_law_learning_curves_on_{label}.png"), dpi=150
    )
    plt.close(figure)

def schedule_cost(n, checkpoints, promotion_factor):
    """Epochs spent by the checkpoint schedule when starting with n candidates."""
    cost, previous, active = 0, 0, n
    for rung, epoch in enumerate(checkpoints):
        cost += active * (epoch - previous)
        previous = epoch
        if PRUNING_START_RUNG <= rung < len(checkpoints) - 1:
            active = max(1, math.ceil(active / promotion_factor))
    return cost
# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(
    budget_epochs=None,
    candidate_count=CANDIDATE_COUNT,
    promotion_factor=PROMOTION_FACTOR,
    checkpoints=CHECKPOINTS,
    dataset=DATASET,
    seed=RANDOM_SEED,
    output_dir=OUTPUT_DIR,
    budget=MAX_EPOCHS
):
    checkpoints = tuple(checkpoints)
    if budget_epochs is not None:    # largest candidate set whose schedule fits the budget
        candidate_count = promotion_factor
        while schedule_cost(candidate_count + 1, checkpoints, promotion_factor) <= budget_epochs:
            candidate_count += 1
    validate_arguments(candidate_count, promotion_factor, checkpoints, dataset)

    if api is None:
        api = create(BENCH_PATH, "tss", fast_mode=True, verbose=False)
    if candidate_count > len(api):
        raise ValueError(f"candidate_count exceeds the search space size ({len(api)}).")

    hp = str(HP_CONSTANT * MAX_EPOCHS)
    candidates = sample_candidates(api, candidate_count, seed)

    started_at = time()
    best, history, resource_used = search(api, candidates, checkpoints, promotion_factor, dataset, hp)
    elapsed = time() - started_at

    # Final reported test accuracy of the winner: end of the full `hp` schedule.
    _, final_test_acc = query_accuracies(api, best["index"], dataset, hp, iepoch=None)
    best["test_accuracies"][-1] = final_test_acc

    if output_dir is not None:
        save_outputs(output_dir, dataset, candidates, best, history)

    return {
        "best_index": best["index"],
        "seed": seed,
        "best_architecture": best["architecture"],
        "test_acc": final_test_acc,
        "epochs_used": resource_used,
        "elapsed_seconds": elapsed,
        "history": history,
    }


if __name__ == "__main__":
    run()