import numpy as np


def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(distance_matrix)
    heuristics_matrix = np.zeros((n, n))
    visited = [False] * n
    current_capacity = capacity
    cumulative_demand = 0

    for i in range(n):
        if i == 0:  # Start from the depot
            for j in range(1, n):
                heuristics_matrix[i][j] = 1 / distance_matrix[i][j]
        else:
            visited[i] = True
            current_demand = demands[i]
            cumulative_demand += current_demand
            current_capacity -= current_demand

            penalty = (max(0, cumulative_demand - capacity) ** 3) / (capacity ** 3)

            for j in range(1, n):
                if not visited[j]:
                    distance_weight = (distance_matrix[i][j] ** 2 + 1e-3)
                    demand_weighted_ratio = demands[j] / (distance_weight + 1e-3)
                    proximity_factor = 1 / (np.linalg.norm(coordinates[i] - coordinates[j]) + 1e-3)
                    heuristics_matrix[i][j] = (
                                                          1 / distance_weight) * demand_weighted_ratio * proximity_factor - penalty * 0.5

                    heuristics_matrix[j][i] = heuristics_matrix[i][j]

            if current_capacity < 0:
                cumulative_demand = current_demand
                current_capacity = capacity

    return heuristics_matrix