import torch

def utility_v2(train_x, train_y, best_x, best_y, test_x, mean_test_y, std_test_y, cost_test_y, budget_used, budget_total):
    # {This algorithm calculates utility by balancing potential improvement and uncertainty while applying an adaptive normalization based on remaining budget and a focus on exploring less-costly regions.}

    # Calculate the potential gain compared to the best known value
    potential_gain = mean_test_y - best_y

    # Normalize the potential gain
    normalized_gain = potential_gain / (torch.abs(potential_gain) + 1e-6)

    # Calculate uncertainty based on standard deviation
    uncertainty = std_test_y / (std_test_y + 1e-6)

    # Compute the effective budget remaining
    budget_remaining = budget_total - budget_used + 1e-6

    # Combine the costs in a way that encourages exploration of cheaper options
    cost_factor = (1 / (cost_test_y + 1e-6)) * torch.exp(-cost_test_y / budget_remaining)

    # Implement an exploration factor that boosts utility for inputs with higher uncertainty
    exploration_factor = 1 + uncertainty

    # Final utility value calculated by combining normalized gain, cost factor, and exploration factor
    utility_value = normalized_gain * cost_factor * exploration_factor * budget_remaining

    # Ensure the output has the correct shape
    result = utility_value.squeeze()  # Shape (n_test)

    return result