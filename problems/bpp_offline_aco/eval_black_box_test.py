from aco import ACO
import numpy as np
import logging
from gen_inst import BPPInstance, load_dataset, dataset_conf
import sys
sys.path.insert(0, "../../../")

import gpt
from utils.utils import get_heuristic_name


possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


N_ITERATIONS = 100
N_ANTS = 20
SAMPLE_COUNT = 200

def solve(inst: BPPInstance, mode = 'sample'):
    heu = heuristics(inst.demands.copy(), inst.capacity) # normalized in ACO
    assert tuple(heu.shape) == (inst.n, inst.n)
    assert 0 < heu.max() < np.inf
    aco = ACO(inst.demands, heu.astype(float), capacity = inst.capacity, n_ants=N_ANTS, greedy=False)
    if mode == 'sample':
        obj, _ = aco.sample_only(SAMPLE_COUNT)
    else:
        obj, _ = aco.run(N_ITERATIONS)
    return obj

if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")
    basepath = os.path.dirname(__file__)
    for problem_size in dataset_conf['test']:
        dataset_path = os.path.join(basepath, f"dataset/test{problem_size}_dataset.npz")
        dataset = load_dataset(dataset_path)
        n_instances = dataset[0].n
        logging.info(f"[*] Evaluating {dataset_path}")

        objs = []
        for i, instance in enumerate(dataset):
            obj = solve(instance, mode='aco')
            objs.append(obj)

        print(f"[*] Average for {problem_size}: {np.mean(objs)}")