def choose_action(pos, v, last_action):
    target_position = 0.6  # Goal position
    distance_to_target = target_position - pos
    momentum_adjustment = 0.5 * max(0, distance_to_target)

    if pos < -0.9:
        choice = 2  # Strong acceleration for steep uphill
    elif -0.9 <= pos < -0.3:
        if v < -0.06:
            choice = 2  # Aggressive action to counter backward motion
        elif 0 <= v < 0.06 + momentum_adjustment:
            choice = 2  # Boost momentum towards the incline
        elif v >= 0.06 + momentum_adjustment:
            choice = 1  # Steady speed to maintain control
        else:
            choice = 0  # Decelerate to regain stability
    elif -0.3 <= pos < 0.4:
        if v < -0.05:
            choice = 2  # Accelerate to overcome backwards motion
        elif 0 <= v < 0.07 + momentum_adjustment:
            choice = 2  # Increase momentum for better advancement
        elif 0.07 <= v < 0.12:
            choice = 1  # Maintain control at steady speed
        else:
            choice = 0  # Slow down for better maneuvering
    elif 0.4 <= pos < target_position:
        if distance_to_target < 0.15 or abs(v) < 0.06:
            choice = 2  # Strong acceleration to reach the target effectively
        else:
            choice = 1  # Sustain speed for a careful approach
    else:
        if v > 0.04 + momentum_adjustment:
            choice = 0  # Decelerate to avoid overshooting the target
        elif 0 < v <= 0.04:
            choice = 1  # Maintain speed if close to the destination
        else:
            choice = 1  # Continue at a steady pace if feasible

    return choice