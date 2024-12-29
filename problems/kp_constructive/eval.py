import math
from os import path
import numpy as np
import sys
import argparse
from scipy.spatial import distance_matrix
import logging
import copy

try:
    from gpt import select_next_item_v2 as select_next_item
except:
    from gpt import select_next_item


def eval_heuristic(node_positions: np.ndarray, capacity) -> float:
    '''
    Generate solution for TSP problem using the GPT-generated heuristic algorithm.
    
    Parameters
    ----------
    node_positions : np.ndarray
        2D array of node positions of shape (problem_size, 2).
    
    Returns
    -------
    obj : float
        The length of the generated tour.
    '''
    problem_size = node_positions.shape[0]
    # calculate distance matrix
    # set the starting node
    solution_value = 0.0
    remaining_capacity = copy.deepcopy(capacity)
    # init unvisited nodes
    weight = node_positions[:, 0]
    # remove the starting node
    value = node_positions[:, 1]
    # run the heuristic
    while remaining_capacity + 1e-3 > min(weight):
        next_node = select_next_item(
            remaining_capacity=remaining_capacity,
            weights=weight,
            values=value
        )
        if next_node < len(value) and remaining_capacity + 1e-3 > weight[next_node]:
            solution_value += value[next_node]
            remaining_capacity -= weight[next_node]
            weight = np.delete(weight, [next_node])
            value = np.delete(value, [next_node])
        else:
            print(remaining_capacity,weight[next_node])
            raise KeyError(f"Node {next_node} is illegal.")

    # calculate the length of the tour
    return solution_value


if __name__ == '__main__':
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']

    basepath = path.join(path.dirname(__file__), "dataset")
    if not path.isfile(path.join(basepath, "train100_dataset.npy")):
        from gen_inst import generate_datasets

        generate_datasets()

    if mood == 'train':
        capacity = 25
        dataset_path = path.join(basepath, f"train{problem_size}_dataset.npy")
        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]
        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")

        objs = []
        for i in range(n_instances):
            obj = eval_heuristic(node_positions[i], capacity)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj)

        print("[*] Average:")
        print(np.mean(objs))

    else:
        for problem_size in [50, 100, 200]:
            if problem_size == 50:
                capacity = 12.5
            else:
                capacity = 25
            dataset_path = path.join(basepath, f"val{problem_size}_dataset.npy")
            logging.info(f"[*] Evaluating {dataset_path}")
            node_positions = np.load(dataset_path)
            n_instances = node_positions.shape[0]
            objs = []
            for i in range(n_instances):
                obj = eval_heuristic(node_positions[i], capacity)
                objs.append(obj)
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")
