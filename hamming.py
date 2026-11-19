import numpy as np
from nats_bench import create

# Hamming Distance to define the initial samples which our model will act upon (DoE)
def maximin_hamming_design(X_all, n_samples, rng):

    n_architectures = len(X_all)

    if n_samples > n_architectures:
        raise ValueError("Too many initial samples.")

    # Select the first architecture randomly
    selected = [
        rng.integers(n_architectures)
    ]

    remaining = np.ones(n_architectures, dtype=bool)
    remaining[selected[0]] = False

    # Minimum distance from each architecture
    # to the selected architectures
    min_distances = np.count_nonzero(
        X_all != X_all[selected[0]],
        axis=1
    )

    while len(selected) < n_samples:

        candidates = np.flatnonzero(remaining)

        # Find architectures maximizing minimum distance
        best_distance = np.max(
            min_distances[candidates]
        )

        best_candidates = candidates[
            min_distances[candidates] == best_distance
        ]

        # Randomized tie-breaking
        next_idx = rng.choice(best_candidates)

        selected.append(next_idx)
        remaining[next_idx] = False

        # Update minimum distances
        new_distances = np.count_nonzero(
            X_all != X_all[next_idx],
            axis=1
        )

        min_distances = np.minimum(
            min_distances,
            new_distances
        )

    return np.array(selected)
