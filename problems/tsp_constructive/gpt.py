def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    next_node = None
    best_combined_score = float('inf')

    # Compute the central node of unvisited nodes
    center_node = min(unvisited_nodes, key=lambda x: sum(distance_matrix[node][x] for node in unvisited_nodes))

    # Dynamic weights based on the number of remaining nodes
    num_unvisited = len(unvisited_nodes)
    immediate_weight = 0.7 if num_unvisited > 3 else 0.5
    future_weight = 0.3 if num_unvisited > 3 else 0.5

    for node in unvisited_nodes:
        immediate_distance = distance_matrix[current_node][node]

        # Estimate future distance to remaining nodes including return trip
        future_distance = 0
        temp_node = node
        remaining_nodes = unvisited_nodes.copy()
        remaining_nodes.remove(temp_node)

        while remaining_nodes:
            next_temp_node = min(remaining_nodes, key=lambda x: distance_matrix[temp_node][x])
            future_distance += distance_matrix[temp_node][next_temp_node]
            temp_node = next_temp_node
            remaining_nodes.remove(next_temp_node)

        return_distance = distance_matrix[temp_node][destination_node]
        future_distance += return_distance

        # Central node proximity factor with penalty for distance from center
        distance_from_center_penalty = max(0, distance_matrix[node][center_node] - 1)

        # Combined score
        combined_score = (immediate_distance * immediate_weight) + (
                    future_distance * future_weight) + distance_from_center_penalty

        if combined_score < best_combined_score:
            best_combined_score = combined_score
            next_node = node

    return next_node