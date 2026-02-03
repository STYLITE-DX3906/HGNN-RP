import torch
import torch.nn as nn
import torch.nn.functional as F



def preprocess(relational_features_list, ratio, NS):

    num_target_nodes = relational_features_list[0].shape[0]
    selected_indices_list = []

    for target_idx in range(num_target_nodes):
        target_rel_feature = relational_features_list[0][target_idx]

        # Calculate Euclidean distances for each type
        distances = torch.norm(relational_features_list[0] - target_rel_feature, dim=1)

        target_selected_indices = [] #########################此处可以使用字典，存储选取的节点索引和相应节点的关系特征。可以但没必要，只存储索引数值然后直接根据索引数值进行索引提取即可。#########################
        # Select nodes for same type based on distances
        num_nodes_to_select = int(ratio * relational_features_list[0].shape[0])
        _, selected_indices = torch.topk(distances, num_nodes_to_select, largest=False) #########################此处可以选取节点索引和相应节点的关系特征,而不是只选取节点索引。可以但没必要，只存储索引数值然后直接根据索引数值进行索引提取即可。#########################
        target_selected_indices.append(selected_indices)

        for i in range(len(NS)):
            type_idx = NS[i][target_idx]
            target_selected_indices.append(type_idx)

        selected_indices_list.append(target_selected_indices)

    return selected_indices_list

# # Example usage of preprocessing
# if __name__ == "__main__":
#     # Dummy data
#     in_dims_R = [128, 256, 512]
#     hidden_dim = 64
#     ratio = [0.05, 0.05, 0.05]
#
#     relational_features_list = [
#         torch.rand((4019, in_dims_R[0])),  # P type
#         torch.rand((7167, in_dims_R[1])),  # A type
#         torch.rand((60, in_dims_R[2]))     # S type
#     ]
#
#     selected_indices_list = preprocess(relational_features_list, ratio)
#     print("Preprocessed selected indices:", selected_indices_list[0])



class RelationDenseAggregation(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, device='cuda:0'):
        super(RelationDenseAggregation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.device = device

    def forward(self, relational_features_list2, transformed_features, selected_indices_list):
        num_target_nodes = len(selected_indices_list)

        # 聚合每个目标节点的邻居特征
        aggregated_features = []
        for target_idx in range(num_target_nodes):
            target_selected_indices = selected_indices_list[target_idx]

            # 单独聚合每个目标节点的每类邻居特征
            aggregated_type_features = []
            # Aggregate features for each type
            #########################此处即为问题所在，target_selected_indices和selected_indices代表有类内次序的节点索引，transformed_features的节点索引无类内次序，因此无法通过前者获取除第一类节点以外的对应的正确的节点特征向量。#########################
            #########################措施：1.将索引编号在此处按照类别进行重新设定（统一进行加法运算）以适应transformed_features的无类内次序的节点索引的特性 2.将transformed_features进行分割，以适应target_selected_indices和selected_indices的有类内次序的节点索引的特性#########################
            #########################分析：1.因为一般数据集中的nei_a和nei_s均代表有类内次序的节点索引，因此采用第二种措施更具便捷性和可迁移性。2.对transformed_features进行分割时，将其中元素的数量与大小和relational_features_list2进行绑定
            for type_idx in range(self.num_types):
                selected_indices = target_selected_indices[type_idx]
                selected_features = transformed_features[type_idx][selected_indices].to(self.device)

                # # 可优化随机参数的加权求和
                # # Initialize weights as learnable parameters for the selected nodes
                # weights = nn.Parameter(torch.Tensor(len(selected_indices)).to(self.device))
                # nn.init.xavier_uniform_(weights.unsqueeze(0))
                # # Calculate weights and normalize
                # weights = weights * torch.norm(relational_features_list2[type_idx][selected_indices] - relational_features_list2[0][target_idx], dim=1)
                # normalized_weights = F.softmax(weights, dim=0)
                # # Weighted sum of features
                # aggregated_feature = torch.sum(selected_features * normalized_weights.unsqueeze(-1), dim=0)

                # Average of features 平均
                aggregated_feature = torch.mean(selected_features, dim=0)

                aggregated_type_features.append(aggregated_feature)

            aggregated_features.append(aggregated_type_features)

        # Transpose to get the output shape as (num_target_nodes, num_types, out_dim)
        aggregated_features = list(map(list, zip(*aggregated_features)))

        h = [torch.stack(type_features) for type_features in aggregated_features]

        return h

# # Example usage of the modified module
# if __name__ == "__main__":
#     # Dummy data
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     num_nodes_p = 4019
#     num_nodes_a = 7167
#     num_nodes_s = 60
#     in_dim = 64
#     out_dim = 64
#     num_types = 3
#
#     relational_features_list2 = [
#         torch.rand((4019, in_dim)),  # P type
#         torch.rand((7167, in_dim)),  # A type
#         torch.rand((60, in_dim))     # S type
#     ]
#
#     transformed_features = torch.rand((num_nodes_p + num_nodes_a + num_nodes_s, in_dim))
#
#     # Preprocess to get selected indices
#     selected_indices_list = preprocess(relational_features_list2, [0.05, 0.05, 0.05])
#
#     # Initialize and run the modified module
#     module = RelationDenseAggregation(in_dim, out_dim, num_types, device=device).to(device)
#     aggregated_features = module(relational_features_list2, transformed_features, selected_indices_list)
#
#     for idx, features in enumerate(aggregated_features):
#         print(f"Type {idx} aggregated features shape:", features.shape)