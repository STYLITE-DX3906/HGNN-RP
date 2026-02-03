# 基于关系的邻接矩阵归一化



import numpy as np
import scipy.sparse as sp

def get_submatrix(adjM, row_indices, col_indices):
    """Get submatrix by zeroing out all other elements"""
    mask = np.zeros(adjM.shape, dtype=bool)
    mask[np.ix_(row_indices, col_indices)] = True
    submatrix = adjM.multiply(mask)
    return submatrix

def split_adjacency_matrix(adjM, node_counts):
    """Split the adjacency matrix into different relation submatrices"""
    node_ranges = np.cumsum([0] + node_counts)
    submatrices = []

    for i in range(len(node_counts)):
        for j in range(len(node_counts)):
            row_indices = np.arange(node_ranges[i], node_ranges[i + 1])
            col_indices = np.arange(node_ranges[j], node_ranges[j + 1])
            submatrix = get_submatrix(adjM, row_indices, col_indices)
            submatrices.append(submatrix)

    return submatrices

def normalize_adjacency_matrix(adj):
    """Normalize the adjacency matrix"""
    rowsum = np.array(adj.sum(1)).flatten()
    rowsum = rowsum.astype(np.float64)  # Convert rowsum to float to avoid integer power issues
    d_inv = np.power(rowsum, -1, where=rowsum != 0).flatten()
    d_inv[np.isinf(d_inv)] = 0.  # replace inf values with zero
    d_mat_inv = sp.diags(d_inv)
    adj_normalized = d_mat_inv.dot(adj)
    return adj_normalized

def aggregate_normalized_matrices(adj_list):
    """Aggregate normalized adjacency matrices"""
    normalized_adj_list = [normalize_adjacency_matrix(adj) for adj in adj_list]
    # Sum the normalized adjacency matrices
    aggregated_adj = sum(normalized_adj_list)
    return aggregated_adj

def process_dataset(adjM, node_counts):
    """Process a dataset by splitting and normalizing the adjacency matrix"""
    # Split the adjacency matrix
    adj_list = split_adjacency_matrix(adjM, node_counts)
    # Normalize and aggregate adjacency matrices
    normalized_and_aggregated_adj = aggregate_normalized_matrices(adj_list)
    return normalized_and_aggregated_adj

def save_sparse_matrix(filename, matrix):
    """Save a sparse matrix to a .npz file"""
    sp.save_npz(filename, matrix)

# # 数据 处理ACM
# node_counts = [4019, 7167, 60]  # Example for three types of nodes
# adjM = sp.load_npz('data/preprocessed/ACM_processed/adjM.npz')
# # 处理数据集
# normalized_and_aggregated_adj = process_dataset(adjM, node_counts)
# # 保存处理后的数据集
# save_path = 'data/preprocessed/ACM_processed/normalized_and_aggregated_adj.npz'
# save_sparse_matrix(save_path, normalized_and_aggregated_adj)
# print(f"Processed adjacency matrix saved to {save_path}")

# # 数据 处理IMDB
# node_counts = [4278, 2081, 5257]  # Example for three types of nodes
# adjM = sp.load_npz('data/preprocessed/IMDB_processed/adjM.npz')
# # 处理数据集
# normalized_and_aggregated_adj = process_dataset(adjM, node_counts)
# # 保存处理后的数据集
# save_path = 'data/preprocessed/IMDB_processed/normalized_and_aggregated_adj.npz'
# save_sparse_matrix(save_path, normalized_and_aggregated_adj)
# print(f"Processed adjacency matrix saved to {save_path}")

# # 数据 处理Yelp
# node_counts = [2614, 1286, 4, 9]  # Example for three types of nodes
# adjM = sp.load_npz('data/preprocessed/4_Yelp/adjM.npz')
# # 处理数据集
# normalized_and_aggregated_adj = process_dataset(adjM, node_counts)
# # 保存处理后的数据集
# save_path = 'data/preprocessed/4_Yelp/normalized_and_aggregated_adj.npz'
# save_sparse_matrix(save_path, normalized_and_aggregated_adj)
# print(f"Processed adjacency matrix saved to {save_path}")

# 数据 处理DBLP
node_counts = [4057, 14328, 7723, 20]  # Example for three types of nodes
adjM = sp.load_npz('data/preprocessed/DBLP_processed/adjM.npz')
# 处理数据集
normalized_and_aggregated_adj = process_dataset(adjM, node_counts)
# 保存处理后的数据集
save_path = 'data/preprocessed/DBLP_processed/normalized_and_aggregated_adj.npz'
save_sparse_matrix(save_path, normalized_and_aggregated_adj)
print(f"Processed adjacency matrix saved to {save_path}")