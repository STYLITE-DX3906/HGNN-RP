import math
import numpy as np
import torch
from torch import nn
from dgl.nn.pytorch import GraphConv
# Transformer验证代码
from gat_transformer import  transformer_att
from gat_transformer_Relation import transformer_att_R

import torch.nn.functional as F

class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel, weights=None):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)  # equal weight

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        # Row normalization of all graphs generated
        adj_list = F.normalize(adj_list, dim=1, p=1)
        # Hadamard product + summation -> Conv
        print(F.softmax(self.weight, dim=0))
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)

class GC_HAN(nn.Module):
    def __init__(self,
                 num_metapaths,
                 in_dim,
                 out_dim,
                 num_heads,
                 graph,
                 graph_weight,
                 meta_graph,
                 GCN_hidden_dim,
                 agg,
                 attn_drop=0.5):

        super(GC_HAN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.agg = agg
        self.graph = graph
        self.graph_weight = graph_weight
        self.meta_graph = meta_graph
        self.GAT_feat_drop = attn_drop
        self.GCN = GCN(in_dim, GCN_hidden_dim)
        self.GAT = transformer_att(GCN_hidden_dim, GCN_hidden_dim, 1, self.GAT_feat_drop, attn_drop, activation=F.elu)
        self.num_metapaths = num_metapaths
        self.dropout = nn.Dropout(attn_drop)
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapaths):  # meta-path Layers; 两个meta-path的维度是一致的
            self.gat_layers.append(transformer_att(GCN_hidden_dim, out_dim, num_heads,
                                           self.GAT_feat_drop, attn_drop, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_dim * num_heads)
        self.graph_attention = GraphChannelAttLayer(3)

    def forward(self, features):
        if self.agg == "mean":
            h = self.GCN(features, self.graph, edge_weight=None)
        elif self.agg == "att":
            h = self.GAT(self.graph, features)
        else:
            raise IOError("Aggregator input error!")
        h = self.dropout(h)
        if features.shape[0] == 11246:
            h = h[0:4019].flatten(1)
        elif features.shape[0] == 11616:
            h = h[0:4278].flatten(1)
        elif features.shape[0] == 3913:
            h = h[0:2614]
        elif features.shape[0] == 26128:
            h = h[0:4057]
        else:
            raise Exception("dataset exception!")
        embeddings = []
        for i, g in enumerate(self.meta_graph):
            embeddings.append(self.gat_layers[i](g, h).flatten(1))
        embeddings = torch.stack(embeddings, dim=1)
        # 语义注意力层
        h = self.semantic_attention(embeddings)
        return h


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.activation = F.elu
        self.gcn_layers = GraphConv(input_dim, output_dim, activation=self.activation)

    def forward(self, graph, features, edge_weight):
        h = self.gcn_layers(graph, features, edge_weight=edge_weight)
        return h


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SemanticAttention, self).__init__()
        # input:[Node, metapath, in_size]; output:[Node, metapath, 1]; 所有节点在每个meta-path上的重要性值
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)    # 每个节点在metapath维度的均值; mean(0): 每个meta-path上的均值(/|V|); (MetaPath, 1)
        beta = torch.softmax(w, dim=0)       # 归一化          # (M, 1)
        # beta = [[0.5], [0.5]]
        # beta = torch.tensor(beta)  # 串联的方式来聚合元路径
        beta = beta.expand((z.shape[0],) + beta.shape)  # 拓展到N个节点上的metapath的值   (N, M, 1)
        return (beta * z).sum(1)     # (beta * z)=>所有节点，在metapath上的attention值;    (beta * z).sum(1)=>节点最终的值      (N, D * K)


class HGAT_AC_R(nn.Module):
    def __init__(self, l_hid, nhid, nclass, feat_drop, attn_drop, negative_slope, num_layers, nheads, device):
        """Dense version of GAT."""
        super(HGAT_AC_R, self).__init__()
        self.device = device
        self.l_hid = l_hid
        self.activation = F.elu

        self.gat_layers = nn.ModuleList()
        self.num_layers = num_layers

        # 输入层
        self.gat_layers.append(transformer_att_R(
            l_hid, nhid, nheads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
# 方案二：
        # 隐藏层
        for l in range(1, num_layers):
            self.gat_layers.append(transformer_att_R(
                nhid * nheads[l-1], nhid, nheads[l], feat_drop, attn_drop, negative_slope, self.activation
            ))
        # 输出层 输出层里面用不着激活函数 GAT分类器
        self.gat_layers.append(transformer_att_R(
            nhid * nheads[-2], nhid, nheads[-1], feat_drop, attn_drop, negative_slope, None))
# # 方案一：
#         # 隐藏层
#         for l in range(1, num_layers):
#             self.gat_layers.append(transformer_att(
#                 nhid * nheads[l-1], nhid, nheads[l], feat_drop, attn_drop, negative_slope, self.activation
#             ))
#         # 输出层 输出层里面用不着激活函数 GAT分类器
#         self.gat_layers.append(transformer_att(
#             nhid * nheads[-2], nhid, nheads[-1], feat_drop, attn_drop, negative_slope, None))


        # 线性分类器
        self.Linear_Classification = nn.Linear(nhid * nheads[-1], nclass, bias=True)

    def forward(self, g, h, R):
# 方案二：
        for l in range(self.num_layers):
            h, R = self.gat_layers[l](g, h, R)
            h = h.flatten(1)
            R = R.flatten(1)
        h, R = self.gat_layers[-1](g, h, R)
        h = h.mean(1)

# # 方案一：
#         # 输入层
#         h, R = self.gat_layers[0](g, h, R)
#         h = h.flatten(1)
#         # 隐藏层
#         for l in range(1, self.num_layers):
#             h = self.gat_layers[l](g, h).flatten(1)
#         # 输出层
#         h = self.gat_layers[-1](g, h).mean(1)


        # 线性层
        logits = self.Linear_Classification(h)
        return logits, h


class GAT_AC_R(nn.Module):
    def __init__(self, in_dims, nhid, feat_drop, attn_drop, nheads, device):
        """Dense version of GAT."""
        super(GAT_AC_R, self).__init__()
        self.device = device
        self.activation = F.elu
        self.gat_layers = transformer_att_R(in_dims, nhid, nheads, feat_drop, attn_drop, activation=self.activation)

    def forward(self, g, h, R):
        # 调用GAT
        # h = self.gat_layers(g, h, R)
        h, R = self.gat_layers(g, h, R)
        h = h.mean(1)
        # R = R.mean(1)
        # logits = self.Leaner_Classification(h)
        return h


class HGAT(nn.Module):
    def __init__(self, nhid, l_hid, nclass, feat_drop, attn_drop, negative_slope, num_layers, nheads, device):
        """Dense version of GAT."""
        super(HGAT, self).__init__()
        self.device = device
        self.activation = F.elu

        self.gat_layers = nn.ModuleList()
        self.num_layers = num_layers
        # 输入层
        self.gat_layers.append(transformer_att(
            l_hid, nhid, nheads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))

        # 隐藏层
        for l in range(1, num_layers):
            self.gat_layers.append(transformer_att(
                nhid * nheads[l-1], nhid, nheads[l], feat_drop, attn_drop, negative_slope, self.activation
            ))

        # 输出层 输出层里面用不着激活函数 GAT分类器
        self.gat_layers.append(transformer_att(
            nhid * nheads[-2], nhid, nheads[-1], feat_drop, attn_drop, negative_slope, None))
        # 线性分类器
        self.Leaner_Classification = nn.Linear(nhid * nheads[-1], nclass, bias=True)

    def forward(self, g, h):
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output layers 输出层用线性层
        h = self.gat_layers[-1](g, h).mean(1)
        logits = self.Leaner_Classification(h)
        return logits, h


class GAT(nn.Module):
    def __init__(self, in_dims, nhid, feat_drop, attn_drop, nheads, device):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.device = device
        self.activation = F.elu
        self.gat_layers = transformer_att(in_dims, nhid, nheads, feat_drop, attn_drop, activation=F.elu)

    def forward(self, g, h):
        # 调用GAT
        h = self.gat_layers(g, h)
        # logits = self.Leaner_Classification(h)
        return h