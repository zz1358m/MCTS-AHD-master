import numpy as np
import logging
from gen_inst import TSPInstance, load_dataset, dataset_conf
from gls import guided_local_search
from tqdm import tqdm
import os
os.environ['OMP_NUM_THREADS'] = '1'
try:
    from gpt import heuristics_v2 as heuristics
except:
    from gpt import heuristics

perturbation_moves = 30
iter_limit = 1200

def calculate_cost(inst: TSPInstance, path: np.ndarray) -> float:
    return inst.distmat[path, np.roll(path, 1)].sum().item()

def solve(inst: TSPInstance) -> float:
    heu = heuristics(inst.distmat.copy())
    assert tuple(heu.shape) == (inst.n, inst.n)
    result = guided_local_search(inst.distmat, heu, perturbation_moves, iter_limit)
    # print(result)
    return calculate_cost(inst, result)

if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")
    for problem_size in dataset_conf['test']:
        dataset_path = os.path.join(f"dataset/test{problem_size}_dataset.npy")
        dataset = load_dataset(dataset_path)
        logging.info(f"[*] Evaluating {dataset_path}")

        objs = []
        for i, instance in enumerate(tqdm(dataset)):
            obj = solve(instance)
            objs.append(obj)

        print(f"[*] Average for {problem_size}: {np.mean(objs)}")
