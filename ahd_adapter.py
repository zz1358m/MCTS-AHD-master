from source.mcts_ahd import MCTS_AHD
from source.getParas import Paras
from source import prob_rank, pop_greedy
from problem_adapter import Problem

from utils.utils import init_client

class AHD:
    def __init__(self, cfg, root_dir, workdir, client) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.problem = Problem(cfg, root_dir)

        self.paras = Paras() 
        self.paras.set_paras(method = "eoh",
                             # problem = "Not used", # Not used
                             # llm_api_endpoint = "api.openai.com",
                             init_size = self.cfg.init_pop_size,
                             pop_size = self.cfg.pop_size,
                             llm_model = client,
                             ec_fe_max = self.cfg.max_fe,  # total evals = 2 * pop_size + n_pop * 4 * pop_size; for pop_size = 10, n_pop = 5, total evals = 2 * 10 + 4 * 5 * 10 = 220
                             exp_output_path = f"{workdir}/",
                             exp_debug_mode = False,
                             eva_timeout=cfg.timeout)
        init_client(self.cfg)
    
    def evolve(self):
        print("- Evolution Start -")

        method = MCTS_AHD(self.paras, self.problem, prob_rank, pop_greedy)

        results = method.run()

        print("> End of Evolution! ")
        print("-----------------------------------------")
        print("---  MCTS-AHD successfully finished!  ---")
        print("-----------------------------------------")

        return results


