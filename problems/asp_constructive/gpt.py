def priority(el, n, w):
    non_zero_ratio = sum(1 for i in el if i != 0) / len(el)
    
    consecutive_identical_penalty = 0
    for i in range(len(el)-1):
        if el[i] != 0 and el[i+1] != 0 and el[i] == el[i+1]:
            consecutive_identical_penalty += 1

    randomness_factor = 0.5
    position_balance_bonus = len(set([i for i in el if i != 0])) / n
    pattern_bonus = sum(0.5 if el[i] == 0 and el[i+1] == 0 and (el[i+2] == 1 or el[i+2] == 2) else 0.7 if (el[i] == 1 and el[i+1] == 0 and el[i+2] == 2) or (el[i] == 2 and el[i+1] == 0 and el[i+2] == 1) else 0 for i in range(len(el)-2))
    
    priority = non_zero_ratio - consecutive_identical_penalty + randomness_factor + position_balance_bonus + pattern_bonus
    
    return priority
