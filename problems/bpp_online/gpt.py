def priority(item, bins_remain_cap):
    priority = []
    total_capacity = sum(bins_remain_cap)
    total_bins = len(bins_remain_cap)
    average_capacity = total_capacity / total_bins if total_bins > 0 else 1  # Avoid division by zero
    dynamic_threshold = item * 0.5  # Dynamic threshold based on item size
    gap_penalty_threshold = average_capacity * 0.5  # Adjusted gap threshold

    for cap in bins_remain_cap:
        if cap >= item:
            remaining_space = cap - item
            gap_penalty = max(0, abs(remaining_space - gap_penalty_threshold))
            proximity_score = -gap_penalty + (item / (remaining_space + 1e-7))  # Encourage good fit

            utilization_ratio = (cap - remaining_space) / cap
            # Utilize logarithmic scaling and reward/penalize based on utilization
            proximity_score += (cap / (remaining_space + 1e-7)) * 0.1  # Logarithmic scaling effect
            if 0.5 < utilization_ratio <= 0.8:  # Reward for medium utilization
                proximity_score += 0.7
            elif utilization_ratio > 0.8:  # Stricter penalty for high utilization
                proximity_score -= 1.5
            elif utilization_ratio < 0.5:  # Penalize for low utilization
                proximity_score -= 0.3

            priority.append(proximity_score)
        else:
            priority.append(0)  # Indicate that this bin cannot fit the item

    return priority