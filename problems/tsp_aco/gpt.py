import numpy as np


def heuristics_v2(distance_matrix):
    distance_matrix = np.array(distance_matrix)
    n = distance_matrix.shape[0]
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                unvisited_mask = np.array([k for k in range(n) if k != i and k != j])
                if unvisited_mask.size > 0:
                    centroid = np.mean(distance_matrix[unvisited_mask], axis=0)
                    distance_to_centroid_i = np.linalg.norm(distance_matrix[i] - centroid)
                    distance_to_centroid_j = np.linalg.norm(distance_matrix[j] - centroid)
                    edge_distance = distance_matrix[i, j]
                    if edge_distance > 0:  # Only consider positive distances
                        heuristics_matrix[i, j] = ((1 / edge_distance) ** 3) * (
                                    (distance_to_centroid_i + distance_to_centroid_j) / edge_distance)

    return heuristics_matrix