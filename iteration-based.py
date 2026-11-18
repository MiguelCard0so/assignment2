import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import os
import random
from time import time
from scipy.optimize import least_squares
from nats_bench import create
from dotenv import load_dotenv
import pickle

load_dotenv()  # Load the Global budget, to be used by each algorithm

BENCH_PATH = os.path.join(
    os.path.dirname(__file__),
    "NATS-tss-v1_0-3ffb9-simple",
    "NATS-tss-v1_0-3ffb9-simple",
) # Path to the NATS-Bench dataset

DATASET = "cifar10-valid"
TEST_DATASET = "cifar10"
HP = "200"
MAX_EPOCHS = int(os.getenv("BUDGET", 200))  # budget for the number of loops through the method
CHECKPOINTS = (1, 3, 9, 27, 81, MAX_EPOCHS)
CANDIDATE_COUNT = 81
PROMOTION_FACTOR = 3
RANDOM_SEED = 42


def power_law_curve(epochs, asymptote, scale, exponent):
	return asymptote - scale * np.power(epochs, -exponent)


def predict_final_accuracy(epochs, accuracies, max_epochs=MAX_EPOCHS):
	epochs = np.asarray(epochs, dtype=float)
	accuracies = np.asarray(accuracies, dtype=float)
	if len(epochs) < 3:
		return float(accuracies[-1])

	scale_guess = max(float(np.ptp(accuracies)), 1e-3)
	initial_parameters = (float(accuracies[-1] + scale_guess), scale_guess, 0.5)
	try:
		fit = least_squares(
			lambda parameters: power_law_curve(epochs, *parameters) - accuracies,
			x0=initial_parameters,
			bounds=([-np.inf, 0.0, 0.01], [np.inf, np.inf, 2.0]),
			max_nfev=20000,
		)
		if not fit.success:
			return float(accuracies[-1])
		prediction = power_law_curve(max_epochs, *fit.x)
		return max(float(accuracies[-1]), float(prediction))
	except (RuntimeError, ValueError, FloatingPointError):
		return float(accuracies[-1])


def run(
	candidate_count=CANDIDATE_COUNT,
	promotion_factor=PROMOTION_FACTOR,
	checkpoints=CHECKPOINTS,
	dataset=DATASET,
	test_dataset=TEST_DATASET,
	seed=RANDOM_SEED,
	output_dir="results/iteration_based",
):
	checkpoints = tuple(checkpoints)
	if len(checkpoints) < 4 or checkpoints[-1] != MAX_EPOCHS:
		raise ValueError(f"Use at least four checkpoints ending at {MAX_EPOCHS} epochs.")
	if any(epoch < 1 for epoch in checkpoints) or any(
		earlier >= later for earlier, later in zip(checkpoints, checkpoints[1:])
	):
		raise ValueError("Checkpoints must be positive and strictly increasing.")
	if candidate_count < promotion_factor or promotion_factor < 2:
		raise ValueError("candidate_count must be >= promotion_factor, which must be >= 2.")

	api = create(BENCH_PATH, "tss", fast_mode=True, verbose=False)
	if candidate_count > len(api):
		raise ValueError(f"candidate_count exceeds the search space size ({len(api)}).")

	rng = random.Random(seed)
	candidate_indices = rng.sample(range(len(api)), candidate_count)
	all_candidates = [
		{"index": index, "architecture": api.arch(index), "epochs": [], "accuracies": [], "test_accuracies": []}
		for index in candidate_indices
	]
	active_candidates = list(all_candidates)
	history = []
	resource_used = 0
	started_at = time()

	for rung, epoch in enumerate(checkpoints):
		for candidate in active_candidates:
			info = api.get_more_info(
				api.query_index_by_arch(candidate["architecture"]),
				dataset,
				iepoch=epoch - 1,
				hp=HP,
				is_random=False,
			)
			
			if "valid-accuracy" not in info or info["valid-accuracy"] is None:
				raise ValueError(
					f"No validation accuracy for dataset {dataset!r}; "
					"choose a dataset with a validation split."
				)

			accuracy = float(info["valid-accuracy"])
			test_accuracy = float(info["test-accuracy"])
			previous_epoch = candidate["epochs"][-1] if candidate["epochs"] else 0
			resource_used += epoch - previous_epoch
			candidate["epochs"].append(epoch)
			candidate["accuracies"].append(accuracy)
			candidate["test_accuracies"].append(test_accuracy)
			candidate["predicted_final"] = predict_final_accuracy(
				candidate["epochs"], candidate["accuracies"]
			)
			history.append(
				{
					"architecture_index": candidate["index"],
					"epoch": epoch,
					"validation_accuracy": accuracy,
					"test_accuracy": test_accuracy,
					"predicted_final_accuracy": candidate["predicted_final"],
				}
			)

		if 2 <= rung < len(checkpoints) - 1:
			active_candidates.sort(
				key=lambda candidate: candidate["predicted_final"], reverse=True
			)
			keep_count = max(1, math.ceil(len(active_candidates) / promotion_factor))
			active_candidates = active_candidates[:keep_count]

	best_candidate = max(active_candidates, key=lambda candidate: candidate["accuracies"][-1])
	elapsed = time() - started_at

	# get test accuracy of the best candidate
	test_info = api.get_more_info(
		api.query_index_by_arch(best_candidate["architecture"]),
		test_dataset,
		hp=HP,
		is_random=False
		)

	best_candidate["test_accuracies"][-1] = float(test_info["test-accuracy"])

	# Save model to be later used for testing phase
	with open(os.path.join(output_dir, "iteration_based_model.pkl"), "wb") as f:
		pickle.dump(all_candidates, f)

	if output_dir is not None:
		os.makedirs(output_dir, exist_ok=True)
		csv_path = os.path.join(output_dir, "power_law_history.csv")
		with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
			writer = csv.DictWriter(csv_file, fieldnames=history[0].keys())
			writer.writeheader()
			writer.writerows(history)

		figure, axis = plt.subplots()
		for candidate in all_candidates:
			axis.plot(candidate["epochs"], candidate["test_accuracies"], color="0.75", alpha=0.45)
		axis.plot(
			best_candidate["epochs"],
			best_candidate["test_accuracies"],
			color="tab:blue",
			marker="o",
			label="Best finalist",
		)
		axis.set(xlabel="Training epoch", ylabel="Test accuracy")
		axis.legend()
		figure.tight_layout()
		figure.savefig(os.path.join(output_dir, "power_law_learning_curves.png"), dpi=150)
		plt.close(figure)

	result = {
		"best_index": best_candidate["index"],
		"best_architecture": best_candidate["architecture"],
		"best_test_accuracy": best_candidate["test_accuracies"][-1],
		"resource_used_epochs": resource_used,
		"elapsed_seconds": elapsed,
		"history": history,
	}
	print(f"Best architecture index: {result['best_index']}")
	print(f"Best {dataset} test accuracy: {result['best_test_accuracy']:.4f}")
	print(f"Epoch-equivalent resource used: {resource_used}")
	print(f"Elapsed seconds: {elapsed:.2f}")
	return result


if __name__ == "__main__":
    run()