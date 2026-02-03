import torch

def preprocess_graph(relational_features_list, ratio):

    num_target_nodes = relational_features_list[0].shape[0]
    adjacency_matrix_homo = torch.zeros((num_target_nodes, num_target_nodes))

    for target_idx in range(num_target_nodes):
        target_rel_feature = relational_features_list[0][target_idx]

        # Calculate Euclidean distances for each type
        distances = torch.norm(relational_features_list[0] - target_rel_feature, dim=1)

        # Select nodes for same type based on distances
        num_nodes_to_select = int(ratio * relational_features_list[0].shape[0])
        _, selected_indices = torch.topk(distances, num_nodes_to_select, largest=False)

        # Update the adjacency matrix to indicate connections between target nodes and their isomorphic neighbors
        for neighbor_idx in selected_indices:
            adjacency_matrix_homo[target_idx, neighbor_idx] = 1

    # Convert the adjacency matrix from torch.Tensor to numpy.ndarray
    adjacency_matrix_homo = adjacency_matrix_homo.numpy()

    return adjacency_matrix_homo

# # 示例
# relational_features_list = [
#     torch.tensor([
#         [1.0, 2.0, 3.0],
#         [2.0, 3.0, 4.0],
#         [3.0, 4.0, 5.0],
#         [4.0, 5.0, 6.0],
#         [5.0, 6.0, 7.0]
#     ])
# ]
# ratio = 0.5  # 选择50%的邻居节点
#
# adjacency_matrix = preprocess(relational_features_list, ratio)
#
# print("Adjacency Matrix:")
# print(adjacency_matrix)