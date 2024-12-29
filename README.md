## Implementation of Paper: Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design


### MCTS-AHD



MCTS-AHD employs a Monte Carlo Tree for the representation of heuristic evolution and uses LLM-assisted Monte Carlo Tree Search for heuristic evolution
![process.png](process.png)

To visually demonstrate the workflow of MCTS-AHD and its ability to escape from local optima and conduct a comprehensive exploration, we provide two examples of heuristic function evolution in two tasks, as shown in Figure G. For example, in designing heuristics with a step-bystep construction framework for TSP, MCTS-AHD can expand potential child nodes from nodes (e.g., MCTS node with ”Expansion: t=611”) that are not among the top 10 optimal ones (the performance range of the top 10 optimal heuristics is the yellow shade), and ultimately reach the best heuristic.

![example.png](example.png)
Thanks to the implementations of [EoH](https://github.com/FeiLiu36/EoH) and [ReEvo](https://github.com/ai4co/reevo).

### Available tasks (Totally 15 scenarios provided):
NP-hard CO Problems as Tasks
* Step-by-step construction framework:
  * Travelling Salesman Problem (TSP)
  * TSP-copy for a reference on simultaneous heuristic function evaluations
  * 0-1 Knapsack (KP)
  * Online Bin Packing Problem (Online BPP) **(Please set max_fe = 2000 for re-implementing the report results for Online BPP)**
  * Admissible Set Problem (ASP)
* Ant Colony Optimization (ACO) **(Please set init_pop_size = 10 in re-implementing the report results for Black-box settings)**:
  * TSP and Black-box settings
  * Capacitated Vehicle Routing Problem (CVRP) and Black-box settings
  * Multiple Knapsack Problem (MKP) and Black-box settings
  * Offline Bin Packing Problem (Offline BPP) and Black-box settings
* Guided Local Search:
  * (Large-scale) TSP
  
Other Complex Tasks
* Bayesian Optimization (BO):
  * Cost-aware Function Design in Active Learning **(Please set botorch according to the requirements.txt to ensure the results of the report are generated)**

MCTS do not need a high-performance seed function. By Implementing eval.py and evaluation prompts in cfgs, MCTS-AHD can be applied to more scenarios with effectiveness easily.


### Run
Change parameter $\lambda_0$ in <a href="source/mcts.py">source/mcts.py</a>

set cfg/config.yaml and run main.py for heuristic evaluations.

**If want to run several evaluations simultaneously, please create multiple environments, including copied problems and prompts, and a copied cfg/problems with changing the problem name** (See tsp_constructive_copy for reference)
