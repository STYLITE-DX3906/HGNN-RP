import numpy as np
import scipy.sparse as sp
import networkx as nx
import scipy.sparse
import pickle
import torch
import torch.nn.functional as F

# def is_symmetric(matrix):
#     """Check if a sparse matrix is symmetric"""
#     return (matrix != matrix.T).nnz == 0
#
#
# def main():
#     # 示例数据（需要用实际数据替换）
#     adjM = sp.csr_matrix(np.random.rand(5, 5))
#     adjM = sp.load_npz('data/preprocessed/IMDB_processed/adjM.npz')
#
#     # 检查邻接矩阵是否对称
#     if is_symmetric(adjM):
#         print("邻接矩阵是对称的，表示的图是无向图或双向图。")
#     else:
#         print("邻接矩阵不是对称的，表示的图是有向图。")
#
#
# if __name__ == "__main__":
#     main()


# prefix = 'E:\山东科技大学\闫页宇课程资料（总）\preprocessed\复现数据集——大师兄\ACM_processed'
#
# nei_a = np.load(prefix + '/nei_a.npy', allow_pickle=True)
# print(nei_a)
# print(nei_a.shape)
#
# nei_s = np.load(prefix + '/nei_s.npy', allow_pickle=True)
# print(nei_s)
# print(nei_s.shape)


import torch
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR
from models import NRSP

# 假设 model 是你的模型实例
model = NRSP().to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 创建数据加载器
train_loader = DataLoader(your_dataset, batch_size=batch_size, shuffle=True)

# 创建学习率查找器
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")

# 进行一次快速的学习率查找
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)

# 查找结果可视化
lr_finder.plot()

# 选择最佳学习率
best_lr = lr_finder.suggestion()
print(f"Suggested learning rate: {best_lr}")

# 探索选取比率的大小与目标节点数量、总节点数量、目标节点的边数量（度数）、节点的边数量、元路径数量、
# load_ACM_data
# ap.add_argument('--ratio', type=int, default=0.11, help='选取出的同质节点的比率')  443 ≈ 400、
# load_DBLP_data
# ap.add_argument('--ratio', type=int, default=0.005, help='选取出的同质节点的比率')  203 ≈ 200、
# load_IMDB_data
# ap.add_argument('--ratio', type=int, default=0.005, help='选取出的同质节点的比率')  213 ≈ 200、
# load_Yelp_data
# ap.add_argument('--ratio', type=int, default=0.15, help='选取出的同质节点的比率')  392 ≈ 400、

