import numpy as np
def select_next_item(remaining_capacity: int, weights: np.ndarray, values: np.ndarray) -> int:
    """Enhanced version of `select_next_item_v1`, prioritizing items by value-to-weight ratio 
    and enforcing a minimum threshold for value.
    """

    # Create a mask to identify which items can fit into the remaining capacity
    valid_mask = weights <= remaining_capacity

    # If no items can fit, return -1
    if not np.any(valid_mask):
        return -1

    # Extract valid weights and values
    valid_weights = weights[valid_mask]
    valid_values = values[valid_mask]

    # Calculate value-to-weight ratios
    ratios = np.divide(valid_values, valid_weights, out=np.zeros_like(valid_values), where=valid_weights != 0)

    # Set a minimum value threshold to avoid items with negligible value
    min_value_threshold = np.percentile(valid_values, 20)  # Keeping only the top 20% of items by value
    high_value_mask = valid_values >= min_value_threshold

    # Combine ratios with a focus on high-value items
    overall_scores = np.zeros_like(ratios)
    overall_scores[high_value_mask] = ratios[high_value_mask] + valid_values[
        high_value_mask] * 0.1  # Slightly favor high-value items

    # If no high-value items, fallback to ratios
    if np.sum(overall_scores) == 0:
        overall_scores = ratios

    # Return the index of the item with the maximum overall score
    best_index_in_mask = np.nanargmax(overall_scores)

    # Map back to the original index
    best_fit_index = np.where(valid_mask)[0][best_index_in_mask]

    return best_fit_index