import numpy as np
import random

def heuristics_v2(distance_matrix):
    n = distance_matrix.shape[0]
    heuristics_matrix = np.zeros((n, n))
    pheromone_matrix = np.ones((n, n))  # Initial pheromone levels
    alpha = 1.0  # Influence of pheromone
    beta = 2.0   # Influence of heuristic visibility
    base_evaporation_rate = 0.1
    iterations = 100
    penalty_factor = 0.5  # Penalty for long paths

    def visibility(i, j):
        return 1 / (distance_matrix[i][j]**2) if distance_matrix[i][j] != 0 else 0

    best_global_length = float('inf')

    for iteration in range(iterations):
        paths = []
        best_length = float('inf')
        best_path = None

        for _ in range(n):
            path = []
            visited = set()
            current_city = random.randint(0, n - 1)
            path.append(current_city)
            visited.add(current_city)

            while len(path) < n:
                probabilities = []
                for j in range(n):
                    if j not in visited:
                        prob = (pheromone_matrix[current_city][j] ** alpha) * (visibility(current_city, j) ** beta)
                        probabilities.append(prob)
                    else:
                        probabilities.append(0)

                total = sum(probabilities)
                if total > 0:
                    probabilities = [p / total for p in probabilities]
                    next_city = np.random.choice(range(n), p=probabilities)
                else:
                    break

                path.append(next_city)
                visited.add(next_city)
                current_city = next_city

            paths.append(path)

            # Calculate path length and apply penalty for longer paths
            path_length = sum(distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1)) + distance_matrix[path[-1]][path[0]]
            adjusted_length = path_length + penalty_factor * (path_length / n)
            if adjusted_length < best_length:
                best_length = adjusted_length
                best_path = path

        # Update best global path
        if best_length < best_global_length:
            best_global_length = best_length

        # Pheromone updates based on adjusted path length
        if best_path is not None:
            reward = max(1 / best_length, 0.1)
            for i in range(len(best_path)):
                pheromone_matrix[best_path[i]][best_path[(i + 1) % n]] += reward

        # Evaporate pheromones with adjustment for exploration
        evaporation_rate = base_evaporation_rate + (best_global_length / n) * 0.01
        pheromone_matrix *= (1 - evaporation_rate)

    # Fill heuristics matrix based on pheromone levels and distances
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristics_matrix[i][j] = pheromone_matrix[i][j] * (1 / (distance_matrix[i][j] if distance_matrix[i][j] != 0 else float('inf')))

    return heuristics_matrix