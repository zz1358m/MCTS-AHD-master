import numpy as np

def heuristics_v2(demand, capacity):
    n = len(demand)
    heuristics_matrix = np.zeros((n, n))
    indices = np.argsort(demand)[::-1]  # Indices of items sorted by size
    used = np.zeros(n, dtype=bool)  # Track used items

    for index in indices:
        if not used[index]:
            candidate = index
            current_size = demand[candidate]
            fitting_indices = []

            # Try to find best fitting pairs for the current candidate
            for j in indices:
                if not used[j] and j != candidate:
                    if current_size + demand[j] <= capacity:
                        heuristics_matrix[candidate][j] = 1  # Mark pair as compatible
                        heuristics_matrix[j][candidate] = 1  # Symmetric
                        current_size += demand[j]
                        fitting_indices.append(j)
                        used[j] = True  # Mark this item as used

            used[candidate] = True  # Mark the candidate as used

    return heuristics_matrix
