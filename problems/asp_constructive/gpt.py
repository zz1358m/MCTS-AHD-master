def priority(el, n, w):
    # Count non-zero elements and their total weighted contribution
    non_zero_positions = [i for i in range(n) if el[i] > 0]
    non_zero_count = len(non_zero_positions)
    weighted_sum = sum(el[i] * (1.5 if el[i] == 1 else 2) for i in non_zero_positions)

    # Calculate positional balance based on distances between non-zero elements
    if non_zero_count > 1:
        distances = [non_zero_positions[i] - non_zero_positions[i - 1] for i in range(1, non_zero_count)]
        positional_balance = sum(distances) / (non_zero_count - 1)  # Average distance
    elif non_zero_count == 1:
        positional_balance = n  # Maximal distance if only one non-zero
    else:
        positional_balance = 0  # No non-zero elements

    # Calculate cluster penalty with an adjusted approach focusing on isolation
    zero_clusters = sum(1 for i in range(1, n) if el[i] == 0 and el[i - 1] == 0)
    cluster_penalty = (zero_clusters ** 1.5) * 2  # Enhanced penalty for cluster size

    # Calculate diversity based on unique pairs of non-zero positions
    unique_pairs = sum(1 for i in range(non_zero_count) for j in range(i + 1, non_zero_count)
                       if abs(non_zero_positions[i] - non_zero_positions[j]) > 1) if non_zero_count > 1 else 0

    # Final priority score combining different metrics
    priority = (weighted_sum / (non_zero_count if non_zero_count > 0 else 1)) + (unique_pairs * 0.4) - (
                positional_balance * 0.3) - cluster_penalty

    return priority