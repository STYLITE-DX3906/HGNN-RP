import os
import time
import argparse
import random
import dgl
import psutil
import scipy.sparse
import torch
import numpy as np
from pytorchtools import EarlyStopping
from tools import evaluate_results_nc, index_generator, parse_minibatch, parse_mask
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
from models import NRSP
from RelationDenseAggregation import preprocess
from RelationDenseGraph_cosine_similarity import preprocess_graph
from data import load_ACM_data



ap = argparse.ArgumentParser(description='NRSP testing for the ACM dataset')
ap.add_argument('--save_postfix', default='ACM', help='Postfix for the saved model and result. Default is ACM.')
ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--cuda', action='store_true', default=True, help='Using GPU or not.')
ap.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
ap.add_argument('--epoch', type=int, default=300, help='Number of epochs. Default is 100.')
ap.add_argument('--patience', type=int, default=30, help='Patience. Default is 5.')
ap.add_argument('--aggregator1', type=str, default="mean", help='Heterogeneous information aggregate layer1_2, att or mean')
ap.add_argument('--aggregator2', type=str, default="SemanticAttention", help='Heterogeneous information aggregate layer2, SemanticAttention or Linear')
ap.add_argument('--feats_drop_rate', type=float, default=0.2, help='The dropout of attribute completion.')
ap.add_argument('--ac_layers', type=int, default=2, help='layers of attribute completion. Default is 1.')
ap.add_argument('--lr', type=float, default=0.01, help='学习率 lr 决定了每次迭代时参数更新的幅度。')
ap.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减 weight_decay 是另一种正则化技术，它通过对权重施加 L2 正则化来防止模型过拟合。')
ap.add_argument('--schedule_step', type=int, default=400, help='指定整个训练过程中学习率调整的总步数。')
ap.add_argument('--max_lr', type=int, default=1e-3, help='指定了学习率在一次循环策略中的最大值。')
ap.add_argument('--dropout_rate', type=float, default=0.5)
ap.add_argument('--ratio', type=int, default=0.005, help='选取出的同质节点的比率')
args = ap.parse_args()
print(args)
save_postfix = args.save_postfix
repeat = args.repeat
is_cuda = args.cuda
hidden_dim = args.hidden_dim
num_heads = args.num_heads
num_epochs = args.epoch
patience = args.patience
agg1 = args.aggregator1
agg2 = args.aggregator2
ac_drop = args.feats_drop_rate
ac_layers = args.ac_layers
lr = args.lr
weight_decay = args.weight_decay
max_lr = 10 * lr
dropout_rate = args.dropout_rate
ratio = args.ratio
device = torch.device('cuda:0' if is_cuda else 'cpu')


# random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if is_cuda:
    print('Using CUDA')
    torch.cuda.manual_seed(seed)


# # 数据加载
features_list, Relational_features_list, adjM, type_mask, labels, train_val_test_idx = load_ACM_data()

# 特征列表和特征维度相关数据加载
features_list = [torch.FloatTensor(features).to(device) for features in features_list]
in_dims = [features.shape[1] for features in features_list]
Relational_features_list = [features.to(device) for features in Relational_features_list]
in_dims_R = [features.shape[1] for features in Relational_features_list]

num_classes = int(labels.max()) + 1
num_types = len(Relational_features_list)

labels = torch.LongTensor(labels).to(device)

train_idx = train_val_test_idx['train_idx']
train_idx = np.sort(train_idx)
val_idx = train_val_test_idx['val_idx']
val_idx = np.sort(val_idx)
test_idx = train_val_test_idx['test_idx']
test_idx = np.sort(test_idx)

# DGL图结构相关数据加载
adjM = torch.FloatTensor(adjM).to(device)
adjMX = adjM.data.cpu().numpy()
adjMX = scipy.sparse.csr_matrix(adjMX)

g = dgl.DGLGraph(adjMX + adjMX.T)  # 增加双向边
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
g = g.to(device)

# # # 加载 包含根据疏密特性选取的同质节点以及通过加载邻居索引获得的异质节点 的索引列表
# selected_indices_list = preprocess(Relational_features_list, ratio, NS)

# # 获取根据疏密特性选取的同质节点的邻接矩阵
adjacency_matrix_homo = preprocess_graph(Relational_features_list, ratio)
remove_self_loop = False
if (remove_self_loop):
    num_nodes = labels.shape[0]
    adjacency_matrix_homoadjacency_matrix_homo = scipy.sparse.csr_matrix(adjacency_matrix_homo - np.eye(num_nodes))
    pass
else:
    adjacency_matrix_homo = scipy.sparse.csr_matrix(adjacency_matrix_homo)
adjacency_matrix_homo_g = dgl.DGLGraph(adjacency_matrix_homo).to(device)



print('data load finish')
svm_macro_avg = np.zeros((4, ), dtype=float)
svm_micro_avg = np.zeros((4, ), dtype=float)
nmi_avg = 0
ari_avg = 0
print('start train with repeat = {}\n'.format(repeat))
for cur_repeat in range(repeat):
    print('cur_repeat = {}   ==============================================================='.format(cur_repeat))
    net = NRSP(in_dims, in_dims_R, hidden_dim, num_types, num_classes, num_heads, g, adjacency_matrix_homo_g,
                    device, ac_drop, ac_layers, dropout_rate)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    print('model init finish\n')

    # training loop
    print('training...')
    net.train()
    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.schedule_step, max_lr=max_lr, pct_start=0.05) # 在 OneCycleLR 学习率调度器中，max_lr 指定了学习率在一次循环策略中的最大值。OneCycleLR 会在训练过程中动态调整学习率，使其先逐渐增加到 max_lr，然后再逐渐减少回基础学习率或接近零的值。
    train_step = 0
    loss_fcn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # training
        t = time.time()
        net.train()
        logits, _, loss_ac = net((features_list, Relational_features_list, type_mask))
        # # 负对数似然损失
        # logp = F.log_softmax(logits, 1)
        # loss_classification = F.nll_loss(logp[train_idx], labels[train_idx])
        loss_classification = loss_fcn(logits[train_idx], labels[train_idx])
        # train_loss = loss_classification + 2.5*loss_ac
        train_loss = loss_classification
        # auto grad
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_time = time.time() - t
        train_step += 1
        print(u'当前进程的内存使用:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        scheduler.step(train_step)
        # validation
        t = time.time()
        net.eval()
        val_loss = 0
        with torch.no_grad():
            logits, _, _ = net((features_list, Relational_features_list, type_mask))
            # logp = F.log_softmax(logits, 1)
            # val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
            val_loss = loss_fcn(logits[val_idx], labels[val_idx])
            val_time = time.time() - t

        print(
            'Epoch {:05d} | Train_Loss {:.4f} | Train_Time(s) {:.4f} | Val_Loss {:.4f} | Val_Time(s) {:.4f}'.format(
                epoch, train_loss, train_time, val_loss, val_time))

        # early stopping
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    # testing with evaluate_results_nc
    print('\ntesting...')
    net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
    net.eval()
    test_embeddings = []
    t_start = time.time()
    with torch.no_grad():
        _, embeddings, _ = net(
            (features_list, Relational_features_list, type_mask))
        t_end = time.time()
        print("Time: {:.4f}".format(t_end - t_start))
        test_embeddings.append(embeddings)
        test_embeddings = torch.cat(test_embeddings, 0)
        embeddings = test_embeddings.detach().cpu().numpy()
        svm_macro, svm_micro, nmi, ari = evaluate_results_nc(embeddings[test_idx], labels[test_idx].cpu().numpy(), num_classes)
        svm_macro_avg = svm_macro_avg + svm_macro
        svm_micro_avg = svm_micro_avg + svm_micro
        nmi_avg += nmi
        ari_avg += ari
svm_macro_avg = svm_macro_avg / repeat
svm_micro_avg = svm_micro_avg / repeat
nmi_avg /= repeat
ari_avg /= repeat
print('---\nThe average of {} results:'.format(repeat))
print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))
print('NMI: {:.6f}'.format(nmi_avg))
print('ARI: {:.6f}'.format(ari_avg))
print('all finished')

with open('log-statistics.txt', 'a+') as f:
    f.writelines('\n' + 'Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]) + '\n' +
                 'Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]) + '\n')
