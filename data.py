import networkx as nx
import numpy as np
import scipy.sparse
import pickle
import torch
import torch.nn.functional as F
import scipy.sparse as sp


# 源码设置的行归一化函数
# def normalize_features(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = scipy.sparse.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx

# 改写源码设置 避免0作为分母
# def normalize_features(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1)).flatten()
#     # Convert rowsum to float to avoid integer power issues
#     rowsum = rowsum.astype(np.float32)
#     # Avoid division by zero by setting zero values to a small number before taking the inverse
#     rowsum[rowsum == 0] = 1e-12
#     r_inv = np.power(rowsum, -1).flatten()
#     r_mat_inv = scipy.sparse.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx

# # 改写源码设置的列归一化函数
# def normalize_features_col(mx):
#     """Column-normalize sparse matrix"""
#     colsum = np.array(mx.sum(0)).flatten()
#     col_inv = np.power(colsum, -1, where=colsum!=0).flatten()
#     col_inv[np.isinf(col_inv)] = 0.
#     col_mat_inv = scipy.sparse.diags(col_inv)
#     mx = mx.dot(col_mat_inv)
#     return mx



def load_ACM_data(prefix='data/preprocessed/ACM_processed'):
    # p的表示
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    # features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    # features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()
    # features_1_onehot = np.eye(7167)
    # features_2_onehot = np.eye(60)
    # features_1 = features_1_onehot
    # features_2 = features_2_onehot

    adjM = scipy.sparse.load_npz(prefix + '/adj_norm.npz').toarray()
    # adjM = scipy.sparse.load_npz(prefix + '/normalized_and_aggregated_adj.npz') # 完美复现 效果相同 normalized_and_aggregated_adj=adj_norm
    # adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()
    # np.fill_diagonal(adjM, adjM.diagonal() + 2)

    # 提取 P 类型节点的行
    P_nodes_rows = adjM[:4019, :]
    # 提取 A 类型节点的行
    A_nodes_rows = adjM[4019:4019 + 7167, :]
    # 提取 S 类型节点的行
    S_nodes_rows = adjM[4019 + 7167:4019 + 7167 + 60, :]

    P_nodes_rows = torch.FloatTensor(P_nodes_rows)
    A_nodes_rows = torch.FloatTensor(A_nodes_rows)
    S_nodes_rows = torch.FloatTensor(S_nodes_rows)
    Relational_features_list = [P_nodes_rows, A_nodes_rows, S_nodes_rows]

    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [features_0], \
           Relational_features_list, \
           adjM, \
           type_mask, \
           labels, \
           train_val_test_idx

def load_IMDB_data(prefix='data/preprocessed/IMDB_processed'):
    # p的表示
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()


    adjM = scipy.sparse.load_npz(prefix + '/normalized_and_aggregated_adj.npz').toarray()

    # 提取 M 类型节点的行
    M_nodes_rows = adjM[:4278, :]
    # 提取 D 类型节点的行
    D_nodes_rows = adjM[4278:4278 + 2081, :]
    # 提取 A 类型节点的行
    A_nodes_rows = adjM[4278 + 2081:4278 + 2081 + 5257, :]

    M_nodes_rows = torch.FloatTensor(M_nodes_rows)
    D_nodes_rows = torch.FloatTensor(D_nodes_rows)
    A_nodes_rows = torch.FloatTensor(A_nodes_rows)
    Relational_features_list = [M_nodes_rows, D_nodes_rows, A_nodes_rows]

    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [features_0], \
           Relational_features_list, \
           adjM, \
           type_mask, \
           labels, \
           train_val_test_idx

def load_Yelp_data(prefix='data/preprocessed/4_Yelp'):
    # p的表示
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_b.npz').toarray()

    adjM = scipy.sparse.load_npz(prefix + '/normalized_and_aggregated_adj.npz').toarray()

    # 提取 B 类型节点的行
    B_nodes_rows = adjM[:2614, :]
    # 提取 U 类型节点的行
    U_nodes_rows = adjM[2614:2614 + 1286, :]
    # 提取 S 类型节点的行
    S_nodes_rows = adjM[2614 + 1286:2614 + 1286 + 4, :]
    # 提取 L 类型节点的行
    L_nodes_rows = adjM[2614 + 1286 + 4:2614 + 1286 + 4 + 9, :]

    B_nodes_rows = torch.FloatTensor(B_nodes_rows)
    U_nodes_rows = torch.FloatTensor(U_nodes_rows)
    S_nodes_rows = torch.FloatTensor(S_nodes_rows)
    L_nodes_rows = torch.FloatTensor(L_nodes_rows)
    Relational_features_list = [B_nodes_rows, U_nodes_rows, S_nodes_rows, L_nodes_rows]


    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npy', allow_pickle=True)
    train_val_test_idx = train_val_test_idx.item()

    return [features_0], \
           Relational_features_list, \
           adjM, \
           type_mask, \
           labels, \
           train_val_test_idx

def load_DBLP_data(prefix='data/preprocessed/DBLP_processed'):
    # p的表示
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3_onehot = np.eye(20)
    features_3 = features_3_onehot

    adjM = scipy.sparse.load_npz(prefix + '/normalized_and_aggregated_adj.npz').toarray()

    # 提取 A 类型节点的行
    A_nodes_rows = adjM[:4057, :]
    # 提取 P 类型节点的行
    P_nodes_rows = adjM[4057:4057 + 14328, :]
    # 提取 T 类型节点的行
    T_nodes_rows = adjM[4057 + 14328:4057 + 14328 + 7723, :]
    # 提取 C 类型节点的行
    C_nodes_rows = adjM[4057 + 14328 + 7723:4057 + 14328 + 7723 + 20, :]

    A_nodes_rows = torch.FloatTensor(A_nodes_rows)
    P_nodes_rows = torch.FloatTensor(P_nodes_rows)
    T_nodes_rows = torch.FloatTensor(T_nodes_rows)
    C_nodes_rows = torch.FloatTensor(C_nodes_rows)
    Relational_features_list = [A_nodes_rows, P_nodes_rows, T_nodes_rows, C_nodes_rows]

    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    # [features_0, features_1, features_2, features_3]
    return [features_3], \
           Relational_features_list, \
           adjM, \
           type_mask, \
           labels, \
           train_val_test_idx

def load_Yelp_data_LP(prefix='data/preprocessed/4_Yelp'):
    # p的表示
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_b.npz').toarray()

    adjM = scipy.sparse.load_npz(prefix + '/normalized_and_aggregated_adj.npz').toarray()

    # 提取 B 类型节点的行
    B_nodes_rows = adjM[:2614, :]
    # 提取 U 类型节点的行
    U_nodes_rows = adjM[2614:2614 + 1286, :]
    # 提取 S 类型节点的行
    S_nodes_rows = adjM[2614 + 1286:2614 + 1286 + 4, :]
    # 提取 L 类型节点的行
    L_nodes_rows = adjM[2614 + 1286 + 4:2614 + 1286 + 4 + 9, :]

    B_nodes_rows = torch.FloatTensor(B_nodes_rows)
    U_nodes_rows = torch.FloatTensor(U_nodes_rows)
    S_nodes_rows = torch.FloatTensor(S_nodes_rows)
    L_nodes_rows = torch.FloatTensor(L_nodes_rows)
    Relational_features_list = [B_nodes_rows, U_nodes_rows, S_nodes_rows, L_nodes_rows]

    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_pos_adj = sp.coo_matrix(
        np.load('E:\山东科技大学\闫页宇课程资料（总）\preprocessed\YELP_processed\LP' + '/train_pos_adj.npy'))
    train_neg_adj = sp.coo_matrix(
        np.load('E:\山东科技大学\闫页宇课程资料（总）\preprocessed\YELP_processed\LP' + '/train_neg_adj.npy'))
    val_pos_adj = sp.coo_matrix(
        np.load('E:\山东科技大学\闫页宇课程资料（总）\preprocessed\YELP_processed\LP' + '/val_pos_adj.npy'))
    val_neg_adj = sp.coo_matrix(
        np.load('E:\山东科技大学\闫页宇课程资料（总）\preprocessed\YELP_processed\LP' + '/val_neg_adj.npy'))
    test_pos_adj = sp.coo_matrix(
        np.load('E:\山东科技大学\闫页宇课程资料（总）\preprocessed\YELP_processed\LP' + '/test_pos_adj.npy'))
    test_neg_adj = sp.coo_matrix(
        np.load('E:\山东科技大学\闫页宇课程资料（总）\preprocessed\YELP_processed\LP' + '/test_neg_adj.npy'))

    return [features_0], \
        Relational_features_list, \
        adjM, \
        type_mask, \
        labels, \
        train_pos_adj, train_neg_adj, val_pos_adj, val_neg_adj, test_pos_adj, test_neg_adj
