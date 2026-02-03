import numpy as np
import torch
from torch import nn
from layers import GC_HAN, HGAT_AC_R, GAT_AC_R, SemanticAttention, GCN, GAT, transformer_att
import torch.nn.functional as F
from RelationDenseAggregation import RelationDenseAggregation
from InterAggregation import HAN_Semantic_Agg



class NRSP(nn.Module):
    def __init__(self,
                 in_dims,
                 in_dims_R,
                 hidden_dim,
                 num_types,
                 num_classes,
                 num_heads,
                 graph,
                 adjacency_matrix_homo_g,
                 device,
                 ac_drop,
                 ac_layers,
                 dropout_rate=0.5,
                 ):
        super(NRSP, self).__init__()
        self.hidden_dim = hidden_dim
        self.g = graph
        self.adjacency_matrix_homo_g = adjacency_matrix_homo_g
        self.device = device
        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(m, hidden_dim, bias=True) for m in in_dims])
        self.fc_list_R = nn.ModuleList([nn.Linear(m, hidden_dim, bias=True) for m in in_dims_R])
        self.fc_list_R2 = nn.ModuleList([nn.Linear(m, hidden_dim, bias=True) for m in in_dims_R])
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        for fc in self.fc_list_R:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        for fc in self.fc_list_R2:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # feature dropout after attribute completion
        if ac_drop > 0:
            self.feat_drop = nn.Dropout(ac_drop)
        else:
            self.feat_drop = lambda x: x

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = lambda x: x

########################## 上游模型 属性补全#########################
        # 原HOAE中ACM和IMDB的最好补全层数都是1，YELP是2。
        num_layers = ac_layers
        heads = [num_heads] * num_layers + [1]
        # 原HOAE中ACM、YELP的属性补全模型
        self.HGAT_AC_R = HGAT_AC_R(hidden_dim, hidden_dim, num_classes, dropout_rate, dropout_rate, 0.5, num_layers,
                         heads, self.device)
        # 原HOAE中IMDB的属性补全模型
        self.GAT_AC_R = GAT_AC_R(hidden_dim, hidden_dim, dropout_rate, dropout_rate, num_heads, device)

########################## 下游模型 表示学习 #########################
        # 原HOAE的GC_HAN网络层
        # self.layer1 = GC_HAN(num_metapaths, hidden_dim, out_dim, num_heads, graph, graph_weight, meta_graph, GCN_hidden_dim, agg, attn_drop=dropout_rate)

#########################节点级聚合
        # # 方案零：直接平均
        # self.layer1 = RelationDenseAggregation(hidden_dim, hidden_dim, num_types, self.device)

        # HOAE属性增强的思路：先聚合异构邻居节点 再聚合同构邻居节点
        # # 方案一：GCN聚合低阶异构（ACM中包括同构）邻居节点+GAT聚合高阶同构邻居节点
        # self.layer1 = GCN(hidden_dim, hidden_dim)
        # self.layer1_2 = transformer_att(hidden_dim, hidden_dim, num_heads, dropout_rate, dropout_rate, activation=F.elu) # 多注意力头
        # # self.layer1_2 = transformer_att(hidden_dim, hidden_dim, 1, dropout_rate, dropout_rate, activation=F.elu) # 单注意力头(与多注意力头相当 分类差 聚类好)
        # # 方案二：GCN聚合低阶异构（ACM中包括同构）邻居节点+GCN聚合高阶同构邻居节点
        # self.layer1 = GCN(hidden_dim, hidden_dim)
        # self.layer1_2 = GCN(hidden_dim, hidden_dim)
        # 方案三：GAT聚合低阶异构（ACM中包括同构）邻居节点+GAT聚合高阶同构邻居节点
        # self.layer1 = transformer_att(hidden_dim, hidden_dim, num_heads, dropout_rate, dropout_rate, activation=F.elu) # 多注意力头
        # self.layer1_2 = transformer_att(hidden_dim, hidden_dim, num_heads, dropout_rate, dropout_rate, activation=F.elu) # 多注意力头
        # 方案四：GAT聚合低阶异构（ACM中包括同构）邻居节点+GCN聚合高阶同构邻居节点
        # self.layer1 = transformer_att(hidden_dim, hidden_dim, num_heads, dropout_rate, dropout_rate, activation=F.elu) # 多注意力头
        # self.layer1_2 = GCN(hidden_dim, hidden_dim)

        # HOAE属性增强的思路：先聚合同构邻居节点 再聚合异构邻居节点
        # # 方案五：GCN聚合高阶同构邻居节点+GAT聚合低阶异构（ACM中包括同构）邻居节点
        # self.layer1 = GCN(hidden_dim, hidden_dim)
        # self.layer1_2 = transformer_att(hidden_dim, hidden_dim, num_heads, dropout_rate, dropout_rate, activation=F.elu) # 多注意力头
        # # self.layer1_2 = transformer_att(hidden_dim, hidden_dim, 1, dropout_rate, dropout_rate, activation=F.elu) # 单注意力头(与多注意力头相当 分类差 聚类好)
        # # 方案六：GCN聚合高阶同构邻居节点+GCN聚合低阶异构（ACM中包括同构）邻居节点
        # self.layer1 = GCN(hidden_dim, hidden_dim)
        # self.layer1_2 = GCN(hidden_dim, hidden_dim)
        # 方案七：GAT聚合高阶同构邻居节点+GAT聚合低阶异构（ACM中包括同构）邻居节点
        # self.layer1 = transformer_att(hidden_dim, hidden_dim, num_heads, dropout_rate, dropout_rate, activation=F.elu) # 多注意力头
        # self.layer1_2 = transformer_att(hidden_dim, hidden_dim, num_heads, dropout_rate, dropout_rate, activation=F.elu) # 多注意力头
        # 方案八：GAT聚合高阶同构邻居节点+GCN聚合低阶异构（ACM中包括同构）邻居节点
        # self.layer1 = transformer_att(hidden_dim, hidden_dim, num_heads, dropout_rate, dropout_rate, activation=F.elu)  # 多注意力头
        # self.layer1_2 = GCN(hidden_dim, hidden_dim)

        # 低阶异质和高阶同质特征融合的思路：
        # # 方案九：GCN聚合低阶异构（ACM中包括同构）邻居节点+GAT聚合高阶同构邻居节点
        # self.layer1 = GCN(hidden_dim, hidden_dim)
        # self.layer1_2 = transformer_att(hidden_dim, hidden_dim, num_heads, dropout_rate, dropout_rate, activation=F.elu) # 多注意力头
        # # 方案十：GCN聚合低阶异构（ACM中包括同构）邻居节点+GCN聚合高阶同构邻居节点
        self.layer1 = GCN(hidden_dim, hidden_dim)
        self.layer1_2 = GCN(hidden_dim, hidden_dim)
        # 方案十一：GAT聚合低阶异构（ACM中包括同构）邻居节点+GAT聚合高阶同构邻居节点
        # self.layer1 = transformer_att(hidden_dim, hidden_dim, num_heads, dropout_rate, dropout_rate, activation=F.elu) # 多注意力头
        # self.layer1_2 = transformer_att(hidden_dim, hidden_dim, num_heads, dropout_rate, dropout_rate, activation=F.elu) # 多注意力头
        # 方案十二：GAT聚合低阶异构（ACM中包括同构）邻居节点+GCN聚合高阶同构邻居节点
        # self.layer1 = transformer_att(hidden_dim, hidden_dim, num_heads, dropout_rate, dropout_rate, activation=F.elu) # 多注意力头
        # self.layer1_2 = GCN(hidden_dim, hidden_dim)



#########################类型级聚合

        # # 方案零：使用HeCo中的HAN语义级注意力机制：（差）
        # self.layer2 = HAN_Semantic_Agg(hidden_dim, dropout_rate)

        # # 方案一：直接对不同类型的特征求平均（详情见下）

        # 方案二：使用HOAE程序中的语义级注意力机制：
        self.layer2 = SemanticAttention(hidden_dim, hidden_dim)

        # 方案三：使用线性层拼接后降维：
        # self.layer2 = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)

        # 方案四：使用HOAE程序中的语义级注意力机制融合目标节点的原始特征：
        # self.layer3 = SemanticAttention(hidden_dim, hidden_dim)

        # 方案五：再次融合拼接目标节点的原始特征后使用线性层进行降维：
        # self.layer3 = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)

        self.Leaner_Classification = nn.Linear(hidden_dim, num_classes, bias=True)

        # self.Leaner_Classification = nn.Linear(256, num_classes, bias=Tru1e)

    def forward(self, inputs1):
        feat_list, Relational_features_list, type_mask = inputs1
# 公式（1）
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=feat_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i+3)[0]
            transformed_features[node_indices] = fc(feat_list[i])
            # transformed_features[node_indices] = self.dropout(fc(feat_list[i]))

        transformed_features_R = torch.zeros(type_mask.shape[0], self.hidden_dim, device=Relational_features_list[0].device)
        for i, fc in enumerate(self.fc_list_R):
            node_indices = np.where(type_mask == i)[0]
            transformed_features_R[node_indices] = fc(Relational_features_list[i])
            # transformed_features_R[node_indices] = self.dropout(fc(Relational_features_list[i]))


# 公式（2）（3）（4）（5）
        # 投影之后的特征
        feat_src = transformed_features
        transformed_features = F.tanh(transformed_features)
        # transformed_features = F.leaky_relu(transformed_features)


        feat_src_R = transformed_features_R
        transformed_features_R = F.tanh(transformed_features_R)
        # transformed_features_R = F.leaky_relu(transformed_features_R)

        # transformed_features = self.feat_drop(transformed_features)
        # transformed_features_R = self.feat_drop(transformed_features_R)

        # ACM的属性补全
        if transformed_features_R.shape[0] == 11246:
            # _, transformed_features = self.HGAT_AC_R(self.g, transformed_features, transformed_features_R)
            transformed_features = self.GAT_AC_R(self.g, transformed_features, transformed_features_R)
            transformed_features = torch.squeeze(transformed_features, 1)
            transformed_features[0: 4019] = feat_src[0: 4019]  # 补全之后的源节点属性特征是不相同的，因此需要使用原始属性特征进行替换。

        # YELP的属性补全
        elif transformed_features_R.shape[0] == 3913:
            # _, transformed_features = self.HGAT_AC_R(self.g, transformed_features, transformed_features_R)
            transformed_features = self.GAT_AC_R(self.g, transformed_features, transformed_features_R)
            transformed_features = torch.squeeze(transformed_features, 1)
            transformed_features[0: 2614] = feat_src[0: 2614]

        # DBLP的属性补全
        elif transformed_features_R.shape[0] == 26128:
            # _, transformed_features = self.HGAT_AC_R(self.g, transformed_features, transformed_features_R)
            transformed_features = self.GAT_AC_R(self.g, transformed_features, transformed_features_R)
            transformed_features = torch.squeeze(transformed_features, 1)
            # transformed_features[0: 4057] = feat_src[0: 4057]
            # transformed_features[4057: 4057 + 14328] = feat_src[4057: 4057 + 14328]
            # transformed_features[4057 + 14328:4057 + 14328 + 7723] = feat_src[4057 + 14328:4057 + 14328 + 7723]
            transformed_features[4057 + 14328 + 7723:4057 + 14328 + 7723 + 20] = feat_src[4057 + 14328 + 7723:4057 + 14328 + 7723 + 20]

        # IMDB的属性补全
        else:
            # _, transformed_features = self.HGAT_AC_R(self.g, transformed_features, transformed_features_R)
            transformed_features = self.GAT_AC_R(self.g, transformed_features, transformed_features_R)
            transformed_features = torch.squeeze(transformed_features, 1)
            transformed_features[0: 4278] = feat_src[0: 4278]  # 补全之后的源节点属性特征是不相同的，因此需要使用原始属性特征进行替换。

        transformed_features = F.tanh(transformed_features)
        # transformed_features = F.leaky_relu(transformed_features)
        transformed_features = self.feat_drop(transformed_features)

# 公式（6）（7）（8）（9）直到（17），其中（6）和（7）的一部分在数据处理中就已经实现了。
        # hidden layers

        # # 获取每种类型节点的数量
        # num_nodes_list = [features.shape[0] for features in Relational_features_list]
        # # 使用 torch.split 分割 transformed_features
        # transformed_features_list = torch.split(transformed_features, num_nodes_list, dim=0)
        # # 设定三种不同的转换后关系特征的输入
        # 第一种 重新进行关系特征转换
        # transformed_features_R_list = [linear(rel_features) for linear, rel_features in zip(self.fc_list_R2, Relational_features_list)]
        # 第二种 上述非线性转换后的关系特征
        # transformed_features_R_list = torch.split(transformed_features_R, num_nodes_list, dim=0)
        # 第三种 上述线性转换后的关系特征
        # transformed_features_R_list = torch.split(feat_src_R, num_nodes_list, dim=0)

#########################节点级聚合#########################
        # # 方案零：直接平均
        # h = self.layer1(transformed_features_R_list, transformed_features_list, selected_indices_list)

        # # 方案一至方案四：GCN GAT 交互使用
        # GAT
        # h = self.layer1(self.g, transformed_features).mean(1)
        # GCN
        # h = self.layer1(self.g, transformed_features, edge_weight=None)

        # h = self.dropout(h)
        # if transformed_features.shape[0] == 11246:
        #     h = h[0:4019].flatten(1)
        # elif transformed_features.shape[0] == 11616:
        #     h = h[0:4278].flatten(1)
        # elif transformed_features.shape[0] == 3913:
        #     h = h[0:2614]
        # elif transformed_features.shape[0] == 26128:
        #     h = h[0:4057]
        # else:
        #     raise Exception("dataset exception!")

        # # 方案一至方案四：GCN GAT 交互使用
        # GAT
        # h = self.layer1_2(self.adjacency_matrix_homo_g,h).mean(1) # 多注意力头
        # h = self.layer1_2(self.adjacency_matrix_homo_g,h).flatten(1) # 单注意力头
        # GCN
        # h = self.layer1_2(self.adjacency_matrix_homo_g, h, edge_weight=None)



        # # 方案五至方案八：GCN GAT 交互使用
        # if transformed_features.shape[0] == 11246:
        #     h = transformed_features[0:4019]
        # elif transformed_features.shape[0] == 11616:
        #     h = transformed_features[0:4278]
        # elif transformed_features.shape[0] == 3913:
        #     h = transformed_features[0:2614]
        # elif transformed_features.shape[0] == 26128:
        #     h = transformed_features[0:4057]
        # else:
        #     raise Exception("dataset exception!")

        # GAT
        # h = self.layer1(self.adjacency_matrix_homo_g, h).mean(1)
        # GCN
        # h = self.layer1(self.adjacency_matrix_homo_g, h, edge_weight=None)

        # h = self.dropout(h)
        # if transformed_features.shape[0] == 11246:
        #     transformed_features[0:4019] = h
        # elif transformed_features.shape[0] == 11616:
        #     transformed_features[0:4278] = h
        # elif transformed_features.shape[0] == 3913:
        #     transformed_features[0:2614] = h
        # elif transformed_features.shape[0] == 26128:
        #     transformed_features[0:4057] = h
        # else:
        #     raise Exception("dataset exception!")

        # # 方案五至方案八：GCN GAT 交互使用
        # GAT
        # h = self.layer1_2(self.g,transformed_features).mean(1) # 多注意力头
        # h = self.layer1_2(self.g,transformed_features).flatten(1) # 单注意力头
        # GCN
        # h = self.layer1_2(self.g, transformed_features, edge_weight=None)



        # # 方案九至方案十二：GCN GAT 交互使用
        # GAT
        # h1 = self.layer1(self.g, transformed_features).mean(1)
        # GCN
        h1 = self.layer1(self.g, transformed_features, edge_weight=None)
        h1 = self.dropout(h1)

        if transformed_features.shape[0] == 11246:
            h1 = h1[0:4019]
        elif transformed_features.shape[0] == 11616:
            h1 = h1[0:4278]
        elif transformed_features.shape[0] == 3913:
            h1 = h1[0:2614]
        elif transformed_features.shape[0] == 26128:
            h1 = h1[0:4057]
        else:
            raise Exception("dataset exception!")

        if transformed_features.shape[0] == 11246:
            transformed_features = transformed_features[0:4019]
        elif transformed_features.shape[0] == 11616:
            transformed_features = transformed_features[0:4278]
        elif transformed_features.shape[0] == 3913:
            transformed_features = transformed_features[0:2614]
        elif transformed_features.shape[0] == 26128:
            transformed_features = transformed_features[0:4057]
        else:
            raise Exception("dataset exception!")
        # # 方案九至方案十二：GCN GAT 交互使用
        # GAT
        # h2 = self.layer1_2(self.adjacency_matrix_homo_g,transformed_features).mean(1) # 多注意力头
        # GCN
        h2 = self.layer1_2(self.adjacency_matrix_homo_g, transformed_features, edge_weight=None)
        h2 = self.dropout(h2)


#########################类型级聚合#########################

        # # 方案零：使用HeCo中的HAN语义级注意力机制：（差）

        # # 方案一：直接对不同类型的特征求和：
        # h = torch.sum(torch.stack(h), dim=0)

        # # 方案二：直接对不同类型的特征求平均：
        # # 将列表中的所有张量堆叠起来
        # stacked_tensors = torch.stack(h)
        # # 计算堆叠后的张量的平均值
        # h = torch.mean(stacked_tensors, dim=0)

        # # 方案三：使用HOAE程序中的语义级注意力机制：
        # h = torch.stack(h, dim=1)
        # h = self.layer2(h)

        # # 方案九至方案十二：使用HOAE程序中的语义级注意力机制：
        # 使用 torch.stack 沿着新的维度堆叠 h1 和 h2
        h = torch.stack([h1, h2], dim=1)
        # # 方案十三：使用拼接后降维操作:
        # 使用 torch.cat 沿着最后一个维度拼接 h1 和 h2
        # h = torch.cat([h1, h2], dim=1)

        h = self.layer2(h)

        # h = self.dropout(h)


        # # 融合自身特征
        # # 方案十四：使用HOAE程序中的语义级注意力机制：
        # h = torch.stack([transformed_features, h], dim=1)
        # # 方案十五：使用拼接后降维：
        # h = torch.cat([h,transformed_features], dim=1)
        # h = self.layer3(h)

        h = F.tanh(h)
        # h = F.leaky_relu(h)


# 公式（18）
        # 线性层
        logits = self.Leaner_Classification(h)

        loss_ac = 0

        return logits, h, loss_ac
