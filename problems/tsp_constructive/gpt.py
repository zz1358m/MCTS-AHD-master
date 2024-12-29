def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if not unvisited_nodes:
        return None

    next_node = None
    min_score = float('inf')
    total_unvisited_distance = sum(distance_matrix[current_node][node] for node in unvisited_nodes)

    # Improved decay factor inspired by No.1 algorithm
    decay_factor = 0.5 - (0.1 / max(1, len(unvisited_nodes)))

    for node in unvisited_nodes:
        local_distance = distance_matrix[current_node][node]
        global_contribution = total_unvisited_distance / (1 + sum(distance_matrix[node][j] for j in unvisited_nodes))

        # Score calculation emphasizing local distance with the new decay factor
        score = 0.6 * local_distance + 0.4 * decay_factor * global_contribution

        # Selecting the node with the minimum score
        if score < min_score:
            min_score = score
            next_node = node

    return next_node