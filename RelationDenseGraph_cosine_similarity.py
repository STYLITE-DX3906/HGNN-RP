import torch
import torch.nn.functional as F

def preprocess_graph(relational_features_list, ratio):

    num_target_nodes = relational_features_list[0].shape[0]
    adjacency_matrix_homo = torch.zeros((num_target_nodes, num_target_nodes))

    for target_idx in range(num_target_nodes):
        target_rel_feature = relational_features_list[0][target_idx]

        # Calculate cosine similarities for each type
        similarities = F.cosine_similarity(relational_features_list[0], target_rel_feature.unsqueeze(0), dim=1)

        # Select nodes for same type based on cosine similarities
        num_nodes_to_select = int(ratio * relational_features_list[0].shape[0])
        _, selected_indices = torch.topk(similarities, num_nodes_to_select, largest=True)

        # Update the adjacency matrix to indicate connections between target nodes and their isomorphic neighbors
        for neighbor_idx in selected_indices:
            adjacency_matrix_homo[target_idx, neighbor_idx] = 1

    # Convert the adjacency matrix from torch.Tensor to numpy.ndarray
    adjacency_matrix_homo = adjacency_matrix_homo.numpy()

    return adjacency_matrix_homo