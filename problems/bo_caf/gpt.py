import torch

def utility(train_x, train_y, best_x, best_y, test_x, mean_test_y, std_test_y, cost_test_y, budget_used, budget_total):
    # Calculate expected improvement with logarithmic scaling
    expected_improvement = torch.log(torch.maximum(mean_test_y - best_y + 1e-6, torch.tensor(1e-6, dtype=torch.float64)))

    # Calculate an adaptive cost penalty using an inverse quadratic formulation
    cost_penalty = 1 / (cost_test_y**2 + 1e-6)

    # Create a compound exploration term influenced by uncertainty and normalized by cost
    exploration_term = (std_test_y / (std_test_y + 1e-6)) * (1 / (cost_test_y + 1e-6))

    # Budget scaling factor that emphasizes remaining resources
    remaining_budget_scale = (budget_total - budget_used + 1e-6) / (budget_total + 1e-6)

    # Combine all components to form the final utility value
    utility_value = expected_improvement * cost_penalty * exploration_term * remaining_budget_scale
    
    # Ensure the utility value has the correct shape (n_test,)
    return utility_value
