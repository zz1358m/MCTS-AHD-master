import numpy as np


def heuristics_v2(prize, weight):
    n = len(prize)
    m = weight.shape[1]

    # Calculate prize-to-weight ratios
    ratio = prize / (np.sum(weight, axis=1) + 1e-5)  # Avoid division by zero

    # Cumulative total prize for diminishing returns adjustment
    total_prize_collected = np.sum(prize)
    diminishing_returns = 1 + (total_prize_collected / np.max(prize + 1e-5))  # Adjust based on total collected

    # Score calculation with diminishing returns normalization
    score = (ratio / diminishing_returns) * (1 / (np.max(weight, axis=1) - np.min(weight, axis=1) + 1e-5))

    # Sort items based on score in descending order
    sorted_indices = np.argsort(-score)

    heuristics = np.zeros(n)
    selected_weights = np.zeros(m)

    # Greedily select items based on sorted order with partial inclusion
    for i in sorted_indices:
        if np.all(selected_weights + weight[i] <= 1):  # Check if the entire item can be added
            heuristics[i] = prize[i]
            selected_weights += weight[i]  # Update selected weights
        else:
            remaining_capacity = 1 - selected_weights
            if np.any(remaining_capacity < weight[i]):
                partial_selection = np.minimum(remaining_capacity, weight[i])
                heuristics[i] = prize[i] * np.sum(partial_selection) / np.sum(weight[i])
                selected_weights += partial_selection

    # Normalize heuristics
    heuristics_matrix = heuristics / np.sum(heuristics) if np.sum(heuristics) > 0 else heuristics

    return heuristics_matrix