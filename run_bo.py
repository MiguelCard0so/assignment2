import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import re
from nats_bench import create
from BO import RadialBasisFunctionKernel, GaussianProcessRegressor
from ThompsonSampling import ThompsonSampling
from RBF import RadialBasisFunctionKernel
from HDK import HammingDistanceKernel
from time import time
import random
from dotenv import load_dotenv
import pickle

load_dotenv()  # Load the Global budget, to be used by each algorithm

BENCH_PATH = os.path.join(
    os.path.dirname(__file__),
    "NATS-tss-v1_0-3ffb9-simple",
    "NATS-tss-v1_0-3ffb9-simple",
) # Path to the NATS-Bench dataset
HP = 200
IS_RANDOM = False

BUDGET = int(os.getenv("BUDGET"))  # budget for the number of loops through the method 
             #(it should be the same through all methods used for this project)

# Search Space
OPTIMIZATION_SPACE = {
    'none': 0,
    'skip_connect': 1,
    'nor_conv_1x1': 2,
    'nor_conv_3x3': 3,
    'avg_pool_3x3': 4
}

def architecture_to_features(architecture):
    """Convert an architecture string to a feature vector based on the defined optimization space."""
    operation_names = re.findall(r"([A-Za-z][A-Za-z0-9_]*)~\d+", architecture)
    if not operation_names:
        raise ValueError(f"No operations found in architecture: {architecture!r}")

    try:
        return [OPTIMIZATION_SPACE[name] for name in operation_names]
    except KeyError as error:
        raise ValueError(f"Unknown operation in architecture: {error.args[0]}") from error

def run(budget=100, initial_length=10, val_metric='valid-accuracy', dataset='cifar10-valid', seed=42, output_dir="results/BO"):
    # Define the algorithm budget
    budget = BUDGET if BUDGET is not None else budget

    # Define the path to the NATS-Bench dataset
    api = create(BENCH_PATH, 'tss', fast_mode=True, verbose=False)
    begin = time()
    if not 0 < initial_length <= len(api):
        raise ValueError(f"initial_length must be between 1 and {len(api)}.")

    rng = random.Random(seed)
    initial_idxs = rng.sample(range(len(api)), initial_length)
    initial_samples = [api.arch(idx) for idx in initial_idxs]
    X_initial = np.array([architecture_to_features(arch) for arch in initial_samples])

    # Y_train is the performance metric (e.g., accuracy) of the initial samples
    Y_train = np.array([
        api.get_more_info(arch, hp=HP, is_random=IS_RANDOM, dataset=dataset)[val_metric]
        for arch in initial_samples
    ])

    observed_indices = set(initial_idxs)
    candidate_pool_size = 256

    initial_best_position = int(np.argmax(Y_train))
    best_performance = float(Y_train[initial_best_position])
    best_candidate = {
        "index": initial_idxs[initial_best_position],
        "architecture": initial_samples[initial_best_position],
    }
    history = [
        {
            "architecture_index": idx,
            "iteration": 0,
            "phase": "initial",
            "performance": float(performance),
        }
        for idx, performance in zip(initial_idxs, Y_train)
    ]

    for i in range(budget):
        # Fit the Gaussian Process model to the training data
        # kernel = RadialBasisFunctionKernel(length_scale=1.0, signal_variance=1.0)
        kernel = HammingDistanceKernel(signal_variance=1.0, lenght_scale=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, noise_variance=1e-10)
        gpr.fit(X_initial, Y_train)

        remaining_indices = [idx for idx in range(len(api)) if idx not in observed_indices]
        if not remaining_indices:
            raise ValueError("The evaluation budget exceeds the number of unobserved architectures.")

        candidate_indices = rng.sample(
            remaining_indices, min(candidate_pool_size, len(remaining_indices))
        )
        X_candidates = np.array([architecture_to_features(api.arch(idx)) for idx in candidate_indices])

        # Use Thompson Sampling to select the next architecture to evaluate
        thompson_sampling = ThompsonSampling(gpr)
        position = thompson_sampling.select_next_architecture(api, X_candidates)
        next_arch_idx = candidate_indices[position]
        next_arch = api.arch(next_arch_idx)

        # Evaluate the selected architecture
        performance = api.get_more_info(next_arch, hp=HP, is_random=IS_RANDOM, dataset=dataset)[val_metric]

        # Update the training data with the new sample
        observed_indices.add(next_arch_idx)
        X_initial = np.vstack([X_initial, architecture_to_features(next_arch)])
        Y_train = np.append(Y_train, performance)

        # Test accuracy, to be used for the report
        test_accuracy = api.get_more_info(next_arch, hp=HP, is_random=IS_RANDOM, dataset=dataset)['test-accuracy']

        iteration = i + 1
        history.append({
            "architecture_index": next_arch_idx,
            "iteration": iteration,
            "phase": "bayesian_optimization",
            "performance": float(performance),
            "test_accuracy": float(test_accuracy),
        })

        # Update the best architecture if necessary
        if performance > best_performance:
            best_performance = performance
            best_candidate = {"index": next_arch_idx, "architecture": next_arch, "test_accuracy": test_accuracy}

        # just to follow the evolution of the algorithm
        if best_candidate is not None:
            print(f"Best architecture so far: {best_candidate['architecture']} with performance: {best_performance:.4f}")

    time_taken = time() - begin

    # Save model
    with open(os.path.join(output_dir, f"{output_dir}/gpr_model.pkl"), "wb") as f:
        pickle.dump(gpr, f)

    metric_label = {
        "test-accuracy": "Test accuracy",
        "train-accuracy": "Training accuracy",
    }.get("test-accuracy", "test-accuracy".replace("-", " ").capitalize())

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"bo_results_{dataset}.csv")
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["architecture_index", "iteration", "phase", "performance", "test_accuracy"],
            )
            writer.writeheader()
            writer.writerows(history)

        figure, axis = plt.subplots(constrained_layout=True)
        initial_history = [entry for entry in history if entry["phase"] == "initial"]
        optimization_history = [
            entry for entry in history if entry["phase"] == "bayesian_optimization"
        ]
        axis.scatter(
            [entry["iteration"] for entry in initial_history],
            [entry["test_accuracy"] for entry in initial_history],
            color="0.55",
            alpha=0.7,
            label="Initial evaluations",
        )
        axis.scatter(
            [entry["iteration"] for entry in optimization_history],
            [entry["test_accuracy"] for entry in optimization_history],
            color="tab:blue",
            alpha=0.7,
            label="BO evaluations",
        )

        best_so_far = max(entry["test_accuracy"] for entry in initial_history)
        best_iterations = [0]
        best_performances = [best_so_far]
        for entry in optimization_history:
            best_so_far = max(best_so_far, entry["test_accuracy"])
            best_iterations.append(entry["iteration"])
            best_performances.append(best_so_far)
        axis.step(
            best_iterations,
            best_performances,
            where="post",
            color="tab:green",
            linewidth=2,
            label="Best observed",
        )

        axis.set(
            xlabel="Bayesian optimization iteration",
            ylabel=metric_label,
            title=f"Bayesian optimization on {dataset}",
        )
        axis.legend()
        figure.savefig(
            os.path.join(output_dir, "bayesian_optimization_performance.png"),
            dpi=150,
        )
        plt.close(figure)
        
        result = {
            "best_index": best_candidate["index"],
            "best_architecture": best_candidate["architecture"],
            "best_test_accuracy": best_candidate["test_accuracy"],
            "elapsed_seconds": time_taken,
            "history": history,
        }
        print(f"Best architecture index: {result['best_index']}")
        print(f"Best {dataset} {metric_label.lower()}: {result['best_test_accuracy']:.4f}")
        print(f"Elapsed seconds: {time_taken:.2f}")
        return result
        

def main():
    run(budget=BUDGET)

if __name__ == "__main__":
    main()