def priority_v2(item, bins_remain_cap):
    priority = [0] * len(bins_remain_cap)

    for i in range(len(bins_remain_cap)):
        if bins_remain_cap[i] >= item:
            remaining_capacity_ratio = bins_remain_cap[i] / item
            wasted_space = bins_remain_cap[i] - item

            if wasted_space <= 0.2 * item:  # Narrow wasted space threshold
                priority[i] = 3  # Highest priority for small wasted space
            elif remaining_capacity_ratio >= 1.5:
                priority[i] = 2  # High priority for bins with good capacity utilization
            elif 0.5 < remaining_capacity_ratio < 1.5:
                priority[i] = 1  # Moderate priority for acceptable ratios
            else:
                priority[i] = 0  # Low priority for poor utilization

    return priority