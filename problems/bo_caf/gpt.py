import torch

def utility(train_x, train_y, best_x, best_y, test_x, mean_test_y, std_test_y, cost_test_y, budget_used, budget_total):
    # Calculate potential improvement using logarithmic scaling
    potential_improvement = torch.log(torch.clamp(mean_test_y - best_y + 1e-9, min=1e-9))  # Avoid log(0)

    # Cubic uncertainty penalty based on normalized std deviation
    normalized_uncertainty = std_test_y / (std_test_y + 1e-9)
    uncertainty_penalty = 1 + normalized_uncertainty + (normalized_uncertainty ** 3)  # Linear + Cubic penalty

    # Calculate cost effectiveness
    remaining_budget = budget_total - budget_used
    cost_effectiveness = (remaining_budget / (cost_test_y + 1e-9)) ** 1.2  # Emphasize lower costs

    # Compute combined utility value
    utility_value = potential_improvement * cost_effectiveness / uncertainty_penalty  # Higher uncertainty results in lower utility

    # Ensure the returned utility value size is (n_test)
    return utility_value