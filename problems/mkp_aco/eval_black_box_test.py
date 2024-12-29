from aco import ACO
import numpy as np
import torch
import logging
import sys
sys.path.insert(0, "../../../")

import gpt
from utils.utils import get_heuristic_name


possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


N_ITERATIONS = 100
N_ANTS = 10

def solve(prize: np.ndarray, weight: np.ndarray):
    n, m = weight.shape
    heu = heuristics(prize.copy(), weight.copy()) + 1e-9
    assert heu.shape == (n,)
    heu[heu < 1e-9] = 1e-9
    aco = ACO(torch.from_numpy(prize), torch.from_numpy(weight), torch.from_numpy(heu), N_ANTS)
    obj, _ = aco.run(N_ITERATIONS)
    return obj


if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")

    basepath = os.path.dirname(__file__)
    for problem_size in [100, 300]:
        dataset_path = os.path.join(basepath, f"dataset/test{problem_size}_dataset.npz")
        dataset = np.load(dataset_path)
        prizes, weights = dataset['prizes'], dataset['weights']
        n_instances = prizes.shape[0]
        logging.info(f"[*] Evaluating {dataset_path}")

        objs = []
        for i, (prize, weight) in enumerate(zip(prizes, weights)):
            obj = solve(prize, weight)
            objs.append(obj.item())

        print(f"[*] Average for {problem_size}: {np.mean(objs)}")