import numpy as np

def heuristics_v2(edge_attr, node_attr):
    n = edge_attr.shape[0]
    heuristics_matrix = np.zeros((n, n))
    weight_scale = 1.5  # Adjusted weight scale for node attributes
    edge_sensitivity = 4  # Quartic sensitivity for edge attributes

    for i in range(n):
        for j in range(n):
            if edge_attr[i, j] > 0 and i != j:  # Considering only positive edge attributes
                heuristics_matrix[i, j] = (weight_scale * np.sqrt(node_attr[i]) * np.sqrt(node_attr[j])) / (edge_attr[i, j] ** edge_sensitivity)

    return heuristics_matrix