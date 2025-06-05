import pickle as pkl
import torch.nn as nn
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
from scipy import sparse
import tqdm
from kmeans_gpu import kmeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.cluster import DBSCAN,KMeans

from scipy.sparse import coo_matrix

import torchvision
from torch.utils.data import DataLoader,Dataset
def SimMatrixFilter(sim_matrix,filter,k=20):
    # 计算每行最小值的索引
    if sp.issparse(filter):
        print('sparse matrix!')
        f = filter.A

    if k == 0:
        return f, sim_matrix * f
    m = sim_matrix.shape[0]
    n = sim_matrix.shape[1]
    rmatrix = sim_matrix * f  #先按照邻接矩阵对相似度矩阵进行过滤
    mat_flat = rmatrix.reshape(1,-1)  #将相似度矩阵排成列表
    mat_flat_sorted = np.sort(mat_flat)
    positive_mask = mat_flat_sorted > 0
    first_positive_index = np.argmax(positive_mask) if positive_mask.any() else -1
    sorted_flat_index = np.argsort(mat_flat, axis=1)[:,first_positive_index:first_positive_index+k]
    for z in sorted_flat_index[0]:
        mat_flat[0][z]=0
    rmatrix_new = mat_flat.reshape(m,n)
    rfilter = np.where(rmatrix_new>0.,1.,0.)
    radj = f * rfilter
    return radj


def load_data(dataset):
    # load the data: x, tx, allx, graph
    if dataset in {'cora'}:
        names = ['x', 'tx', 'ty','allx','ally', 'graph']
        objects = []
        for i in range(len(names)):
            '''
            fix Pickle incompatibility of numpy arrays between Python 2 and 3
            https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
            '''
            with open("./dataset/{}/ind.{}.{}".format(dataset,dataset, names[i]), 'rb') as rf:
                u = pkl._Unpickler(rf)
                u.encoding = 'latin1'
                cur_data = u.load()
                objects.append(cur_data)

        x, tx, ty,allx,ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            "../data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)
        features = sp.vstack((allx, tx)).tolil()
        labels = sp.vstack((ally, ty)).tolil()

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        features[test_idx_reorder, :] = features[test_idx_range, :]
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        features = torch.FloatTensor(np.array(features.todense()))


    return adj, features,labels



def load():
    with open("./data/mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
        idx_train = range(3000)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"],idx_train

def onehot_encode(labes):
    classes = set(labes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labes_onehot = np.array(list(map(classes_dict.get,labes)), dtype=np.int32)
    #print(labes_onehot.shape)
    return labes_onehot



def to_onehot(prelabel):
    k = len(np.unique(prelabel))
    print(k)
    label = np.zeros([prelabel.shape[0], k])
    print(label.shape)
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label


def square_dist(prelabel, feature):
    if sp.issparse(feature):
        feature = feature.todense()
    feature = np.array(feature)


    onehot = to_onehot(prelabel)

    m, n = onehot.shape
    count = onehot.sum(1).reshape(m, 1)
    count[count==0] = 1

    mean = onehot.dot(feature)/count
    a2 = (onehot.dot(feature*feature)/count).sum(1)
    pdist2 = np.array(a2 + a2.T - 2*mean.dot(mean.T))

    intra_dist = pdist2.trace()
    inter_dist = pdist2.sum() - intra_dist
    intra_dist /= m
    inter_dist /= m * (m - 1)
    return intra_dist


def normalize(adj,type='rw'):
    """Row-normalize sparse matrix"""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        adj_normalized = r_mat_inv.dot(adj)
        return adj_normalized


def matrix_norm(mx):
    m = torch.norm(mx,dim=1,keepdim=True)
    one = torch.ones_like(m)
    z = torch.where(m==0,one,m)

    norm_matrix = torch.div(mx,z)
    del(mx)
    del(z)
    return norm_matrix

def matrix_norm_np(mx):
    m = torch.norm(mx,dim=1,keepdim=True)
    one = torch.ones_like(m)
    z = torch.where(m==0,one,m)
    norm_matrix = torch.div(mx,z)
    del(z)
    del(mx)
    return norm_matrix


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]

    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj,label, filter):
    adjm = adj.toarray() * filter.cpu().numpy()
    adj = sp.coo_matrix(adj)
    adjm = sp.coo_matrix(adjm)
    adj_ = adj + sp.eye(adj.shape[0])
    adjm_ = adjm + sp.eye(adj.shape[0])
    row_sum_m = np.array(adjm_.sum(1))
    degree_mat_inv_sqrt_m = sp.diags(np.power(row_sum_m, -0.5).flatten())
    adj_normalized_m = adjm_.dot(degree_mat_inv_sqrt_m).transpose().dot(degree_mat_inv_sqrt_m).tocoo()
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized), sparse_mx_to_torch_sparse_tensor(adj_normalized_m)


def smooth_graph(adj, layer, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    reg = [2 / 3] * (layer)

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))
    return adjs



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score




from scipy.ndimage.interpolation import shift

def gen_cluster_adj(db_result,n):
    print('Enter dbscan')
    y = np.unique(db_result)
    clist=[]
    for label in y:
        if label!=-1:
            orig_idx = np.argwhere(db_result==label)
            idx=orig_idx
            steps=len(idx)-1
            for mov in range(steps):
                idx = shift(idx,(1,0),cval=idx[-1])
                relation = np.concatenate((orig_idx,idx),axis=1)
                clist.append(relation)
    print('Relation created!')
    rows=[]
    cols=[]
    for arr in clist:
        rows.extend(arr[:, 0])
        cols.extend(arr[:, 1])
    r = np.array([])
    c = np.array([])
    for row in rows:
        r = np.hstack((r,row))
    for col in cols:
        c = np.hstack((c,col))
    data = np.array([1]*len(rows))
    smatrix = sp.coo_matrix((data,(r,c)), shape=(n,n))
    cadj = smatrix.todense()
    # cadj = cadj + cadj.T.multiply(cadj.T > cadj) - cadj.multiply(cadj.T > cadj)  # the cluster matrix is symmetric
    cadj = cadj + np.eye(cadj.shape[0])   #add self-loop
    cadj = normalize(cadj)
    cadj = torch.tensor(cadj)
    return cadj


def load_graph(dataset, k, filter, loop=True):
    if dataset in {'amap','corafull','amazon_photo','eat','uat','bat','cornell','texas','wisc', 'film'}:
        adj = torch.tensor(np.load('graph/{}_graph.npy'.format(dataset)))
    else:
        path = './graph/{}_graph.txt'.format(dataset)

    if dataset in {'amap','corafull','amazon_photo','eat','uat','bat','cornell','texas','wisc', 'film'}:
        data = np.load('data/{}.npy'.format(dataset))
    else:
        data = np.loadtxt('../data/{}.txt'.format(dataset))


    if dataset in {'amap','corafull','amazon_photo','eat','uat','bat','cornell','texas','wisc', 'film'}:
        pass
    else:
        n, _ = data.shape

        idx = np.array([i for i in range(n)], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(path, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                             shape=(n, n), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if loop:
        if dataset in {'amap','uat'}:
            adj = adj + np.eye(adj.shape[0])
        else:
            adj = adj + sp.eye(adj.shape[0])
    raw_adj = adj
    if dataset in {'amap', 'uat'}:
        adjm = adj.detach().numpy() * filter.cpu().detach().numpy()
    else:
        adjm = adj.toarray() * filter.cpu().detach().numpy()
    adjm = normalize(adjm)
    adj = normalize(adj)

    if sp.issparse(adj):
        print('sparse matrix!')
        adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj,raw_adj,adjm


def load_data_(dataset):
    if dataset in {'amap','corafull','amazon_photo','eat','uat','bat','cornell','texas','wisc', 'film'}:
        feature = np.load('dataset/{}/{}.npy'.format(dataset,dataset))
        label = np.load('dataset/{}/{}_label.npy'.format(dataset,dataset))
    else:
        feature = np.loadtxt('./dataset/{}/{}.txt'.format(dataset,dataset), dtype=float)
        label = np.loadtxt('./dataset/{}/{}_label.txt'.format(dataset,dataset), dtype=int)
    feature = torch.Tensor(feature)
    return feature,label

from sklearn import metrics
from munkres import Munkres


def clustering(Cluster, feature, true_labels):
    f_adj = np.matmul(feature, np.transpose(feature))
    predict_labels = Cluster.fit_predict(f_adj)

    cm = clustering_metrics(true_labels, predict_labels)
    db = -metrics.davies_bouldin_score(f_adj, predict_labels)
    acc, nmi, adj = cm.evaluationClusterModelFromLabel(tqdm)

    return db, acc, nmi, adj


class clustering_metrics:
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)

        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self, tqdm):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        return acc, nmi, adjscore

    @staticmethod
    def plot(X, fig, col, size, true_labels):
        ax = fig.add_subplot(1, 1, 1)
        for i, point in enumerate(X):
            ax.scatter(point[0], point[1], lw=0, s=size, c=col[true_labels[i]])

    def plotClusters(self, tqdm, hidden_emb, true_labels):
        tqdm.write('Start plotting using TSNE...')
        # Doing dimensionality reduction for plotting
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(hidden_emb)
        # Plot figure
        fig = plt.figure()
        self.plot(X_tsne, fig, ['red', 'green', 'blue', 'brown', 'purple', 'yellow', 'pink', 'orange'], 40, true_labels)
        plt.axis("off")
        fig.savefig("plot.png", dpi=120)
        tqdm.write("Finished plotting")


import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocess_graph_smooth(adj, layer=3, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    reg = [2 / 3] * (layer)

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))
    return adjs



def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro

def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1

def clusteringl(feature, true_labels, cluster_num):
    predict_labels, dis, initial = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda")
    return predict_labels.numpy(),dis


from kmeans_gpu import *
def GraphConstructsByKmean(z, labels, n_cluster,cf=0.3, pw=0.):
    power = pw
    predict,dis = clusteringl(z,labels,n_cluster)
    high_confidence = torch.min(dis, dim=1).values
    threshold = torch.sort(high_confidence).values[int(len(high_confidence) * cf)]
    high_confidence_idx = np.argwhere(high_confidence < threshold)[0]
    index = torch.tensor(range(z.shape[0]), device='cuda')[high_confidence_idx]
    sort_labels = torch.tensor(predict, device='cuda')[high_confidence_idx]
    index = index[torch.argsort(sort_labels)]
    class_num = {}
    for label in torch.sort(sort_labels).values:
        label = label.item()
        if label in class_num.keys():
            class_num[label] += 1
        else:
            class_num[label] = 1
    key = sorted(class_num.keys())
    n = z.shape[0]
    cgraph = torch.ones(size=(n,n))*power
    start = 0
    for i in range(len(key[:-1])):
        class_num[key[i + 1]] = class_num[key[i]] + class_num[key[i + 1]]#计算某个label的序列号的区间段，参照1034行的index
        for r in range(start, class_num[key[i]]):
            for c in range(start, class_num[key[i]]):
                cgraph[index[r]][index[c]]=1
        start = class_num[key[i]]
    filter=cgraph
    return filter


def MatrixTopK(matrix,k):
    tmp = np.sort(matrix,axis=1)[:,-k]
    rmatrix = np.where(matrix>=tmp[:,None],matrix,0)
    return rmatrix






























