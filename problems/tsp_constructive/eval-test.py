import math
from os import path
import numpy as np
import sys
import argparse
from scipy.spatial import distance_matrix
import logging
from copy import copy

from tqdm import tqdm
import time
try:
    from gpt import select_next_node_v2 as select_next_node
except:
    from gpt import select_next_node


def eval_heuristic(node_positions: np.ndarray) -> float:
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
    dist_mat = distance_matrix(node_positions, node_positions)
    # set the starting node
    start_node = 0
    solution = [start_node]
    # init unvisited nodes
    unvisited = set(range(problem_size))
    # remove the starting node
    unvisited.remove(start_node)
    # run the heuristic
    for _ in range(problem_size - 1):
        next_node = select_next_node(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=copy(unvisited),
            distance_matrix=dist_mat.copy(),
        )
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f"Node {next_node} is already visited.")
    
    # calculate the length of the tour
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj
    

if __name__ == '__main__':
    print("[*] Running ...")

    for problem_size in [50]:
        dataset_path = path.join('dataset', f"test{problem_size}_dataset.npy")
        logging.info(f"[*] Evaluating {dataset_path}")
        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]
        objs = []
        for i in tqdm(range(n_instances)):
            obj = eval_heuristic(node_positions[i])
            objs.append(obj)
            # print(f"[*] Average for {problem_size}: {obj}")
        print(f"[*] Average for {problem_size}: {np.mean(objs)}")