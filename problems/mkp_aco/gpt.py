import numpy as np

def heuristics_v2(prize, weight):
    n = len(prize)
    m = weight.shape[1]
    heuristics_matrix = np.zeros(n)

    current_solution = np.random.randint(2, size=n)
    current_value = np.sum(prize[current_solution == 1])
    current_weight = np.sum(weight[current_solution == 1], axis=0)

    initial_temperature = 10.0
    cooling_rate = 0.95
    min_temperature = 0.1

    temperature = initial_temperature

    while temperature > min_temperature:
        new_solution = current_solution.copy()
        for _ in range(np.random.randint(1, 3)):  # Random number of flips for exploration
            mutate_index = np.random.randint(n)
            new_solution[mutate_index] = 1 - new_solution[mutate_index]  # Flip bit

        new_weight = np.sum(weight[new_solution == 1], axis=0)
        if np.all(new_weight <= 1):
            new_value = np.sum(prize[new_solution == 1])
            adjustment_noise = np.random.normal(0, 0.1 * (1 / (1 + np.mean(weight[new_solution == 1])))) if np.any(new_solution) else 0
            adjusted_value = new_value + adjustment_noise

            if adjusted_value > current_value or np.random.rand() < np.exp((adjusted_value - current_value) / temperature):
                current_solution = new_solution
                current_value = new_value
                current_weight = new_weight
                
        temperature *= cooling_rate

    for i in range(n):
        if np.all(weight[i] <= 1):
            weight_max = np.max(weight[i])
            if weight_max > 0:
                heuristics_matrix[i] = (prize[i] / weight_max) ** 2  # Quadratic scaling for fitness

    max_weights = np.max(weight, axis=1)
    penalties = np.clip(1 - max_weights, 0, None)
    heuristics_matrix *= penalties

    diversity_factor = np.std(prize[current_solution == 1]) if np.any(current_solution) else 0
    heuristics_matrix += diversity_factor * (1 - np.mean(weight, axis=1))

    return heuristics_matrix