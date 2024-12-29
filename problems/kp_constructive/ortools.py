import os
from ortools.algorithms.python import knapsack_solver
from os import path

import numpy as np

solver = knapsack_solver.KnapsackSolver(
    knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
    "KnapsackExample",
)
factor = 10000
dataset_path = path.join("dataset", f"test200_dataset.npy")
node_positions = np.load(dataset_path)
n_instances = node_positions.shape[0]
total_value = 0
for i in range(n_instances):
    capacities = [int(25*factor)]
    weight = [(factor * node_positions[i][:, 0]).astype(int).tolist()]
    value = (factor * node_positions[i][:, 1]).astype(int).tolist()
    solver.init(value, weight, capacities)
    computed_value = solver.solve()
    print("Total value =", computed_value)
    total_value += computed_value

print(total_value / n_instances / factor)
# Optimal: training 40.187953125