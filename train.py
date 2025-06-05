from __future__ import division
from __future__ import print_function

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
    matrix_norm, setup_seed, GraphConstructsByKmean, setdiv, MatrixTopK
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
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=256, help='Number of units in hidden layer 1.')#512
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')#32
parser.add_argument('--lmd', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--lr', type=float, default=1e-03, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--k', type=int, default=100, help='Number of loop')
parser.add_argument('--loop', type=int, default=1, help='Number of loop')
parser.add_argument('--thr', type=float, default=0.1, help='Number of loop')
parser.add_argument('--weight_s', type=float, default=0., help='Number of loop')
parser.add_argument('--log', type=str, default='False', help='log or not')
parser.add_argument('--cluster', type=str, default='kmean', help='log or not')
parser.add_argument('--m', type=str, default='gae', help='log or not')
parser.add_argument('--impute', type=str, default='False', help='log or not')
parser.add_argument('--pre_model', type=str, default='False', help='log or not')
parser.add_argument('--sample_rate', type=float, default=0.5, help='log or not')
parser.add_argument('--gm', type=float, default=1.0, help='log or not')
parser.add_argument('--test', type=str, help='log or not')
parser.add_argument('--cf', type=float, default=0.1, help='log or not')
parser.add_argument('--pw', type=float, default=0.1, help='log or not')
parser.add_argument('--beta', type=float, default=1.0, help='log or not')
parser.add_argument('--alpha', type=float, default=1.0, help='log or not')
parser.add_argument('--gamma', type=float, default=1.0, help='log or not')


args = parser.parse_args()
setup_seed(args.seed)

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
        print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
                ', f1 {:.4f}'.format(f1))
    except TypeError as e:
        print('Jump Over this error')
    return acc,nmi,ari,f1

import random





def gae_for(args):
    start = time.time()
    lr = 0.001
    cfs={'cora':0.9,'citeseer':0.3, 'wiki':0.4,'uat':0.5,'amap':0.9,'acm':0.1,'eat':0.3,'dblp':0.1,'cornell':0.3,'texas':0.9,'wisc':0.5}
    pws={'cora':0.4,'citeseer':0.05,'wiki':0.1,'uat':0.1,'amap':0.3,'acm':0.1,'eat':0.1,'dblp':0.1,'cornell':0.9,'texas':0.1,'wisc':0.3}#uat=1
    dsc_lr = {'cora': 0.001, 'citeseer': 0.01, 'dblp': 0.001, 'acm': 0.001, 'wiki': 0.0005, 'amap': 0.0001,
              'uat': 0.001, 'eat': 0.0001, 'bat': 0.0005, \
              'cornell': 0.001, 'texas': 0.01, 'wisc': 0.0005}
    dataset = args.dataset
    adj = None
    adj_label = None
    features = None
    labels = None
    n_nodes = None
    feat_dim = None
    adj_norm = None
    if dataset in {'cora','pubmed'}:
        print("Using {} dataset".format(args.dataset))
        adj, features, labels= load_data(args.dataset)
        if dataset == 'cora':
            n_cluster = 7
            lr = 0.001


        elif dataset == 'pubmed':
            n_cluster = 3
            lr = 0.005


        label = [np.argmax(one_hot) for one_hot in labels.A]
        n_nodes, feat_dim = features.shape
        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros() # diagnal zeronize
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        adj = adj_train


        path = './model/{}_filter_{}_{}.pth'.format(dataset,cfs[dataset],pws[dataset])
        if os.path.exists(path):
            device = torch.device('cuda:0')
            afilter = torch.load(path,map_location=device)
            print('Filter loaded!! %s' %  args.dataset)
        else:
            afilter = GraphConstructsByKmean(features, label, n_cluster,cfs[dataset],pws[dataset])
            torch.save(afilter, path)
            print('Filter saved!! %s' %  args.dataset)
        adj_norm, adj_norm_m = preprocess_graph(adj,label,afilter)
        print(adj_norm[0])
        adj_norm = adj_norm.to_dense()
        adj_norm_m = adj_norm_m.to_dense()
        print('ADJ is masked by top ============================= %f' % args.k)
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        n_nodes = adj.shape[0]
        adj_sum = adj.sum()
        adj_sum_m = adj_norm_m.sum()




    dropout = args.dropout
    if dataset in {'citeseer','wiki','amap','uat'}:
        if dataset in {'citeseer'}:
            n_cluster = 6
            lr = 0.001

        elif dataset == 'wiki':
            n_cluster = 17
            lr = 0.0005
            dropout=0.3


        elif dataset == 'amap':
            n_cluster = 8
            lr = 0.0001


        if dataset in {'uat'}:
            n_cluster = 4
            lr = 0.001
            dropout = 0.

        features, labels = load_data_(dataset)


        path = './model/{}_filter_{}_{}.pth'.format(dataset,str(cfs[dataset]),str(pws[dataset]))

        if os.path.exists(path):
            device = torch.device('cuda:0')
            afilter = torch.load(path,map_location=device)
            print('Filter loaded!! %s' %  dataset)
        else:
            afilter = GraphConstructsByKmean(features, labels, n_cluster,cfs[dataset],pws[dataset])
            torch.save(afilter, path)
            print('Filter saved!! %s' %  dataset)

        adj, raw_adj, adjm = load_graph(dataset,None,afilter)
        print("Using {} dataset".format(args.dataset))
        label = labels
        n_nodes, feat_dim = features.shape
        adj_norm = adj
        if type(adj_norm) == np.ndarray:
            adj_norm = torch.tensor(adj_norm, dtype=torch.float32)
        else:
            adj_norm = adj_norm.to_dense()
        adj_norm_m_np = adjm

        if args.thr > 0:
            print("Using the thresh===============>{}".format(args.thr))
            adj_norm_m = torch.tensor(adj_norm_m_np, dtype=torch.float32).cuda()
            adj_sum_m = adj_norm_m.sum()
            print('The adj_sum_m is %f' % adj_sum_m)
        else:
            adj_norm_m = adj_norm

        adj_sum = adj_norm.sum()
        n_nodes = adj_norm.shape[0]
        adj_sum_m = adj_norm_m.sum()
        print('adj_sum_m is %f' % adj_sum_m)

    pos_weight = torch.tensor(float(n_nodes * n_nodes - adj_sum) / adj_sum)
    norm = adj_norm.shape[0] * adj_norm.shape[0] / float((adj_norm.shape[0] * adj_norm.shape[0] - adj_norm.sum()) * 2)

    pos_weight_m = torch.tensor(float(n_nodes * n_nodes - adj_sum_m) / adj_sum_m)
    norm_m = adj_norm_m.shape[0] * adj_norm_m.shape[0] / float((adj_norm_m.shape[0] * adj_norm_m.shape[0] - adj_norm_m.sum()) * 2)



    m = args.m

    print('============================================================={}'.format(m))
    for i in range(experiment_iter):
        print('This is epoch %d\\%d on dataset %s' % (i + 1, experiment_iter,dataset))
        if dataset in {'citeseer','cora'}:
            lamd = 0.01
        else:
            lamd = 10


        print('Now the Lamda is %f' % lamd)
        topN_similarity = None
        if dataset in {'uat','texas','wisc','cornell'}:  # acm
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
        dsc_optim = optim.Adam(dsc.parameters(), lr=dsc_lr[dataset], weight_decay=1e-05)
        acc, nmi, ari, f1, epoch = clusteringm(dataset=dataset, models=[model,dsc], optimizers=[optimizer,dsc_optim],
                                        n_nodes=n_nodes, norm=[norm,norm_m], pos_weight=[pos_weight,pos_weight_m],
                                        adj_orig=adj, features=features, adj_norm=[adj_norm,adj_norm_m], label=label,
                                        n_cluster=n_cluster, similarity=topN_similarity,l=lamd, filter = afilter)
        print('acc: %f', acc)
        print('nmi: %f', acc)
        print('ari: %f', acc)
        print('f1: %f', acc)


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
    parameters = setdiv(dataset)
    ro = parameters['ro']
    subdim = parameters['subdim']
    ratio = parameters['ratio']
    self_epochs = parameters['epochs']
    w1 = parameters['b']
    w2 = parameters['a']
    div = parameters['d']
    gamma = parameters['gamma']

    if dataset == 'cora':
        gcn_epoch = 21

    if dataset == 'citeseer':
        gcn_epoch = 221

    if dataset == 'reut':
        gcn_epoch = 101

    if dataset == 'wiki':
        gcn_epoch = 31

    if dataset == 'amap':
        gcn_epoch = 151

    if dataset == 'uat':
        gcn_epoch = 191



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

    for epoch in range(1, gcn_epoch):  # hidden1:512  hidden2:256 epoch:101
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

        if epoch % div == 0:
            print('epoch is %d' % epoch)
            feat_dsc = hidden_emb.detach()
            feats = [features, feat_dsc]
            for dsc_epoch in range(self_epochs):
                emb_recons = dsc(feats)
                dsc_loss = dsc.loss_dsc(feats, emb_recons,s,adj_label_m, gamma=gamma, weight=[w1, w2])
                dsc_optim.zero_grad()
                dsc_loss.backward()
                dsc_optim.step()

            label = np.array(label)
            C = dsc.C.detach().to('cpu').numpy()
            y_pred,L = spectral_clustering(C, n_cluster, subdim, ratio, ro)
            acc, nmi, ari, f1  = eva(np.array(label), y_pred)
    return acc,nmi,ari,f1,epoch



if __name__ == '__main__':
    gae_for(args)



