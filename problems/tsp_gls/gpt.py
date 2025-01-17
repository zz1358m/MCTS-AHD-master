import numpy as np


def heuristics(distance_matrix):
    num_nodes = distance_matrix.shape[0]
    heuristics_matrix = np.zeros((num_nodes, num_nodes))

    # Calculate the sum of distances and the maximum distance from each node to all others
    for i in range(num_nodes):
        total_distance = np.sum(distance_matrix[i, :])
        max_distance = np.max(distance_matrix[i, :])

        for j in range(num_nodes):
            # Avoid self-distance and compute heuristic for edge (i, j)
            if i != j:
                # Modified penalty factor based on maximum distance
                heuristics_matrix[i, j] = (distance_matrix[i, j] / total_distance) * (
                            1 + distance_matrix[i, j] / max_distance)

    return heuristics_matrix