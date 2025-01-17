import numpy as np


def select_next_item(remaining_capacity, weights, values):
    capacity_threshold = 0.3
    value_weight_ratios = values / weights
    sorted_indices = np.argsort(-value_weight_ratios)

    eligible_items = []
    for i in sorted_indices:
        if weights[i] <= remaining_capacity:
            score = values[i] + (values[i] / weights[i] * 5)
            if weights[i] > remaining_capacity * capacity_threshold:
                score -= (weights[i] - remaining_capacity * capacity_threshold) * 0.1
            eligible_items.append((i, score))

    if not eligible_items:
        return -1

    eligible_items.sort(key=lambda x: -x[1])
    best_combination_value = -float('inf')
    result = -1

    for i, score in eligible_items:
        current_value = values[i]
        current_weight = weights[i]

        for j in eligible_items:
            if j[0] != i and current_weight + weights[j[0]] <= remaining_capacity:
                current_value += values[j[0]]
                current_weight += weights[j[0]]

        if current_value > best_combination_value:
            best_combination_value = current_value
            result = i

    return result