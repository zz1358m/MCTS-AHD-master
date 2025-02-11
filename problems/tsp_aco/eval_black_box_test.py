from os import path
from aco import ACO
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
from scipy.spatial import distance_matrix
import logging
import sys
sys.path.insert(0, "../../../")

import gpt
from utils.utils import get_heuristic_name


possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


N_ITERATIONS = 500
N_ANTS = 30


def solve(node_pos):
    dist_mat = distance_matrix(node_pos, node_pos)
    dist_mat[np.diag_indices_from(dist_mat)] = 1 # set diagonal to a large number
    # Reshape it to (n_edges, 1)
    heu = heuristics(dist_mat.reshape(-1, 1).copy())
    heu = np.where(heu < 1e-9, 1e-9, heu)
    # Reshape it back to (n_nodes, n_nodes)
    heu = heu.reshape(node_pos.shape[0], node_pos.shape[0])
    aco = ACO(dist_mat, heu, n_ants=N_ANTS)
    obj = aco.run(N_ITERATIONS)
    return obj

if __name__ == "__main__":
    print("[*] Running ...")
    for problem_size in [50, 100]:
        dataset_path = path.join(f"dataset/test{problem_size}_dataset.npy")
        node_positions = np.load(dataset_path)
        logging.info(f"[*] Evaluating {dataset_path}")
        n_instances = node_positions.shape[0]
        objs = []
        for i, node_pos in enumerate(node_positions):
            obj = solve(node_pos)
            objs.append(obj.item())
        print(f"[*] Average for {problem_size}: {np.mean(objs)}")