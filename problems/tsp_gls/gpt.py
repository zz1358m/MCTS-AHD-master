import numpy as np
import random


def calculate_tour_cost(tour, distance_matrix):
    return sum(distance_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def update_edge_distance_v2(edge_distance: np.ndarray, local_opt_tour: np.ndarray,
                            edge_n_used: np.ndarray) -> np.ndarray:
    """ Update edge distances with a more dynamic approach for guided exploration. """
    for i in range(len(local_opt_tour)):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % len(local_opt_tour)]

        # Update edge usage counters
        edge_n_used[u][v] += 1
        edge_n_used[v][u] += 1

        # Dynamic penalty influenced by usage frequency
        penalty = min(edge_n_used[u][v] * 0.01, 0.5)
        edge_distance[u][v] *= (1 + penalty)
        edge_distance[v][u] = edge_distance[u][v]  # Ensure symmetry

        # Reduced penalty for rarely used edges for diverse exploration
        if edge_n_used[u][v] == 1:
            edge_distance[u][v] *= 0.95  # Encourage unvisited paths

    return edge_distance


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """ Adaptive heuristic for TSP integrating edge feedback and structured optimization. """
    np.random.seed(0)  # For reproducibility
    num_cities = distance_matrix.shape[0]
    tour = list(range(num_cities))
    random.shuffle(tour)  # Initial random tour
    best_tour = tour[:]
    min_cost = calculate_tour_cost(best_tour, distance_matrix)
    edge_usage = np.zeros_like(distance_matrix)  # Initialize edge usage matrix

    improved = True
    while improved:
        improved = False
        for i in range(1, num_cities - 1):
            for j in range(i + 1, num_cities):
                if j - i == 1: continue  # Skip adjacent nodes

                new_tour = best_tour[:]
                new_tour[i:j] = reversed(new_tour[i:j])  # Perform 2-opt swap

                new_cost = calculate_tour_cost(new_tour, distance_matrix)
                if new_cost < min_cost:
                    best_tour, min_cost = new_tour, new_cost
                    improved = True

        # Update edge distances based on the updated tour
        distance_matrix = update_edge_distance_v2(distance_matrix.copy(), best_tour, edge_usage)

    return distance_matrix