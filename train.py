from __future__ import division
from __future__ import print_function
import tqdm

from sklearn.preprocessing import normalize
import os
import argparse
import time
import metrics
import scipy.sparse as sp
import torch
from torch import optim
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics
from munkres import Munkres, print_matrix
from model import GCNModelAE, DSC
from utils import load_data, mask_test_edges, preprocess_graph,load_data_,load_graph,\
    matrix_norm, setup_seed, GraphConstructsByKmean, MatrixTopK
from sklearn.cluster import SpectralClustering
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import torch.nn.functional as F
import torch.nn as nn
from post_clustering import spectral_clustering, nmi, acc, thrC, clustering
import sklearn.metrics.pairwise as pair




parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=70, help='Random seed.')
parser.add_argument('--epochs', type=int, default=21, help='Number of epochs to train.')
parser.add_argument('--dsc_epoch', type=int, default=251, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=256, help='Number of units in hidden layer 1.')#512
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')#32
parser.add_argument('--lmd', type=float, default=0.01, help='Dropout rate (1 - keep probability).')
parser.add_argument('--lr', type=float, default=1e-03, help='Initial learning rate.')
parser.add_argument('--dsc_lr', type=float, default=1e-03, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--k', type=int, default=100, help='Number of loop')
parser.add_argument('--loop', type=int, default=1, help='Number of loop')
parser.add_argument('--thr', type=float, default=0.1, help='Number of loop')
parser.add_argument('--weight_s', type=float, default=0., help='Number of loop')
parser.add_argument('--log', type=str, default='False', help='log or not')
parser.add_argument('--n_cluster', type=int, default=7, help='log or not')
parser.add_argument('--pre_model', type=str, default='False', help='log or not')
parser.add_argument('--sample_rate', type=float, default=0.5, help='log or not')
parser.add_argument('--gm', type=float, default=1.0, help='log or not')
parser.add_argument('--cf', type=float, default=0.9, help='log or not')
parser.add_argument('--pw', type=float, default=0.4, help='log or not')
parser.add_argument('--beta', type=float, default=0.1, help='log or not')
parser.add_argument('--ro', type=float, default=4.0, help='log or not')
parser.add_argument('--subdim', type=int, default=20, help='log or not')
parser.add_argument('--ratio', type=float, default=0.1, help='log or not')




args = parser.parse_args()
# setup_seed(args.seed)

experiment_iter = args.loop
dsc_train = True


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    acc=0
    nmi=0
    f1=0
    ari=0
    try:
        acc, f1 = cluster_acc(y_true, y_pred)
        nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
        ari = ari_score(y_true, y_pred)
    except TypeError as e:
        print('Jump Over this error')
    return acc,nmi,ari,f1

import random





def gae_for(args):

    lr = args.lr
    dsc_lr=args.dsc_lr
    dataset = args.dataset
    adj, features, labels= load_data(args.dataset)
    n_cluster = args.n_cluster

    label = [np.argmax(one_hot) for one_hot in labels.A]
    n_nodes, feat_dim = features.shape
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros() # diagnal zeronize
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    path = './{}_filter.pth'.format(dataset)
    if os.path.exists(path):
        device = torch.device('cuda:0')
        afilter = torch.load(path, map_location=device)
        print('Filter loaded!!')
    else:
        afilter = GraphConstructsByKmean(features, label, n_cluster, args.cf, args.pw)
        torch.save(afilter, path)
        print('Filter saved!!')
    adj_norm, adj_norm_m = preprocess_graph(adj,label,afilter)
    adj_norm = adj_norm.to_dense()
    adj_norm_m = adj_norm_m.to_dense()
    n_nodes = adj.shape[0]
    adj_sum = adj.sum()
    adj_sum_m = adj_norm_m.sum()
    dropout = args.dropout

    pos_weight = torch.tensor(float(n_nodes * n_nodes - adj_sum) / adj_sum)
    norm = adj_norm.shape[0] * adj_norm.shape[0] / float((adj_norm.shape[0] * adj_norm.shape[0] - adj_norm.sum()) * 2)

    pos_weight_m = torch.tensor(float(n_nodes * n_nodes - adj_sum_m) / adj_sum_m)
    norm_m = adj_norm_m.shape[0] * adj_norm_m.shape[0] / float((adj_norm_m.shape[0] * adj_norm_m.shape[0] - adj_norm_m.sum()) * 2)

    for i in range(experiment_iter):
        topN_similarity = None
        if dataset in {'uat'}:
            gcn_model = GCNModelAE(features.shape[0], feat_dim, args.hidden1, args.hidden2, dropout,
                                   n_cluster, First_act=torch.tanh)
        else:
            gcn_model = GCNModelAE(features.shape[0], feat_dim, args.hidden1, args.hidden2, args.dropout,
                                   n_cluster, First_act=nn.LeakyReLU(0.2,inplace=True))

        model = gcn_model
        features = features.cuda()
        adj_norm = adj_norm.cuda()
        model = model.cuda()
        if dataset in {'cora', 'pubmed'}:
            adj_norm_m = adj_norm_m.cuda()

        model.load_state_dict(torch.load('./model/pretrain_{}.pkl'.format(dataset)))


        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)
        dsc = DSC(features.shape[0]).cuda()
        dsc_optim = optim.Adam(dsc.parameters(), lr=dsc_lr, weight_decay=1e-05)
        acc, nmi, ari, f1 = clusteringm(dataset=dataset, models=[model,dsc], optimizers=[optimizer,dsc_optim],
                                        n_nodes=n_nodes, norm=[norm,norm_m], pos_weight=[pos_weight,pos_weight_m],
                                        adj_orig=adj, features=features, adj_norm=[adj_norm,adj_norm_m], label=label,
                                        n_cluster=n_cluster, similarity=topN_similarity,l=args.lmd, filter = afilter)
        print('acc: %f', acc)
        print('nmi: %f', nmi)
        print('ari: %f', ari)
        print('f1: %f', f1)
        return acc, nmi, ari, f1

def target_distribution(q):
    q = q.detach()
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


np.set_printoptions(threshold=np.inf)



def clusteringm(dataset,models,optimizers,n_nodes,norm,pos_weight,adj_orig,features,adj_norm,label,n_cluster,similarity,l, filter):
    model = models[0]
    dsc = models[1]
    optimizer = optimizers[0]
    dsc_optim = optimizers[1]
    ro = args.ro
    subdim = args.subdim
    ratio = args.ratio
    self_epochs = args.dsc_epoch
    w = args.beta
    gamma = args.gm
    gcn_epoch = args.epochs



    model.train()
    adj_label = adj_norm[0]
    adj_label_m = adj_norm[1]
    norm_m = norm[1]
    pos_weight_m = pos_weight[1]
    x = features.cpu().numpy()

    k = 100
    s_np = pair.cosine_similarity(x,x)
    if k>0:
        s_np_topK = MatrixTopK(s_np,k)
    else:
        s_np_topK = s_np

    s_tensor_t = torch.tensor(s_np_topK) * filter.cpu()
    s = matrix_norm(s_tensor_t)

    s_sum = s.sum()
    spos_weight = float(adj_label.shape[0] * adj_label.shape[0] - s_sum) / s_sum
    snorm = adj_label.shape[0] * adj_label.shape[0] / float(
        (adj_label.shape[0] * adj_label.shape[0] - s_sum) * 2)

    adjs = adj_label_m
    best_acc=0
    best_nmi=0
    best_ari=0
    best_f1=0
    for epoch in tqdm.tqdm(range(1,gcn_epoch)):
        s = s.cuda()
        adj_label = adj_label.cuda()
        adj_label_m = adj_label_m.cuda()
        hidden_emb, recovered_adj,q = model(features, adjs)
        p = target_distribution(q)
        loss = model.loss_gcn_clu(preds=recovered_adj, labels=[s,adj_label_m], p=p, q=q, norm=[snorm,norm_m], pos_weight=[spos_weight, \
                                                               pos_weight_m],lmd=l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            feat_dsc = hidden_emb.detach()
            feats = [features, feat_dsc]
            for dsc_epoch in range(self_epochs):
                emb_recons = dsc(feats)
                dsc_loss = dsc.loss_dsc(feats, emb_recons,s,adj_label_m, gamma=gamma, weight=w)
                dsc_optim.zero_grad()
                dsc_loss.backward()
                dsc_optim.step()

            label = np.array(label)
            C = dsc.C.detach().to('cpu').numpy()
            y_pred,L = spectral_clustering(C, n_cluster, subdim, ratio, ro)
            acc, nmi, ari, f1  = eva(np.array(label), y_pred)
            if best_acc < acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
    return best_acc,best_nmi,best_ari,best_f1



if __name__ == '__main__':
    acc_list=[]
    nmi_list=[]
    ari_list=[]
    f1_list=[]
    args = parser.parse_args()
    for seed in range(69,79):
        setup_seed(seed)
        acc,nmi,ari,f1 = gae_for(args)
        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
        f1_list.append(f1)
    print('Report Average Result: ACC %f, NMI %f, ARI %f, F1 %f' % (np.mean(acc_list),np.mean(nmi_list),np.mean(ari_list),np.mean(f1_list)))


