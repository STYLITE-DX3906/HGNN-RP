import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class HAN_Semantic_Agg(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(HAN_Semantic_Agg, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        print("每种类型的邻居在类间聚合过程中的注意力系数  ", beta.data.cpu().numpy())  # 输出属性特征和结构特征的各自的注意力系数
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc
