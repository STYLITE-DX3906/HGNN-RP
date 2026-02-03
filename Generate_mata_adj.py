# 生成基于元路径的邻接矩阵



import numpy as np
import scipy.sparse as sp


# Step 1: 读取 npz 文件
def read_npz(file_path):
    return sp.load_npz(file_path)


# Step 2: 分割原始邻接矩阵
def split_adjacency_matrix(adjM, node_counts):
    """
    Split adjacency matrix into submatrices based on node types.

    Parameters:
    - adjM: The original adjacency matrix.
    - node_counts: A list with the number of nodes for each type.

    Returns:
    - A dictionary with keys as tuples of node type pairs and values as the corresponding submatrices.
    """
    node_ranges = np.cumsum([0] + node_counts)
    submatrices = {}

    for i in range(len(node_counts)):
        for j in range(len(node_counts)):
            row_start, row_end = node_ranges[i], node_ranges[i + 1]
            col_start, col_end = node_ranges[j], node_ranges[j + 1]
            submatrices[(i, j)] = adjM[row_start:row_end, col_start:col_end]

    return submatrices


# Step 3: 生成元路径邻接矩阵
def generate_metapath_adj(submatrices, metapaths):
    """
    Generate metapath adjacency matrices.

    Parameters:
    - submatrices: A dictionary with keys as tuples of node type pairs and values as the corresponding submatrices.
    - metapaths: A list of metapaths, each represented as a list of node type pairs.

    Returns:
    - A dictionary with keys as metapaths and values as the corresponding metapath adjacency matrices.
    """
    metapath_adjs = {}

    for metapath in metapaths:
        result_adj = submatrices[metapath[0]]
        for step in metapath[1:]:
            result_adj = result_adj.dot(submatrices[step])
        metapath_adjs[tuple(metapath)] = result_adj

    return metapath_adjs


# Step 4: 保存为 .npy 文件
def save_npy(file_path, matrix):
    np.save(file_path, matrix.toarray())


# 主函数

# 试验ACM生成元路径pap、psp
# def main():
#     # 文件路径
#     npz_path = 'data/preprocessed/ACM_processed/adjM.npz'
#     output_dir = 'data/preprocessed/ACM_processed/0_generate/'
#
#     # 节点数量
#     node_counts = [4019, 7167, 60]  # Example for ACM dataset
#
#     # 定义元路径
#     metapaths = [
#         [(0, 1), (1, 0)],  # PAP: Paper -> Author -> Paper
#         [(0, 2), (2, 0)]  # PSP: Paper -> Subject -> Paper
#     ]
#
#     # 读取原始邻接矩阵
#     adjM = read_npz(npz_path)
#
#     # 分割原始邻接矩阵
#     submatrices = split_adjacency_matrix(adjM, node_counts)
#
#     # 生成元路径邻接矩阵
#     metapath_adjs = generate_metapath_adj(submatrices, metapaths)
#
#     # 保存为 .npy 文件
#     for metapath, adj in metapath_adjs.items():
#         metapath_name = ''.join([str(step[0]) + str(step[1]) for step in metapath])
#         output_path = f"{output_dir}/{metapath_name}.npy"
#         save_npy(output_path, adj)
#         print(f"基于元路径的邻接矩阵已保存至 {output_path}")

# DBLP生成apa
def main():
    # 文件路径
    npz_path = 'data/preprocessed/DBLP_processed/adjM.npz'
    output_dir = 'data/preprocessed/DBLP_processed/0'

    # 节点数量
    node_counts = [4057, 14328, 7723, 20]  # Example for ACM dataset

    # 定义元路径
    metapaths = [
        [(0, 1), (1, 0)]  # PAP: Paper -> Author -> Paper
    ]

    # 读取原始邻接矩阵
    adjM = read_npz(npz_path)

    # 分割原始邻接矩阵
    submatrices = split_adjacency_matrix(adjM, node_counts)

    # 生成元路径邻接矩阵
    metapath_adjs = generate_metapath_adj(submatrices, metapaths)

    # 保存为 .npy 文件
    for metapath, adj in metapath_adjs.items():
        metapath_name = ''.join([str(step[0]) + str(step[1]) for step in metapath])
        output_path = f"{output_dir}/{metapath_name}.npy"
        save_npy(output_path, adj)
        print(f"基于元路径的邻接矩阵已保存至 {output_path}")



# 执行主函数
main()


