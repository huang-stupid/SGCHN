import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_data,matrix_norm
from layers import GraphConvolution
from torch.nn import Linear
from torch.utils.data import Dataset,DataLoader
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


class MyDataSet(Dataset):
    def __init__(self,dataset):
        self.adj, self.features, self.labels= load_data(dataset)

        self.size = self.features.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.features[idx]

from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans



class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj



class GCNModelAE(nn.Module):
    def __init__(self, n_sample,input_feat_dim, hidden_dim1, hidden_dim2,dropout,n_clusters,First_act=F.relu):
        super(GCNModelAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=First_act)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, hidden_dim1+hidden_dim2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = 1


        self.layers=[]
        self.dc = InnerProductDecoder(dropout)



    def encode(self, x, adj):
        z1 = self.gc1(x,adj)
        z2 = self.gc2(z1,adj)
        return z1, z2

    #cat z1 and z2
    def forward(self, x, adj):
        encode_data0,encode_data1 = self.encode(x, adj)
        encode_data = torch.cat((encode_data0,encode_data1),dim=1)
        decode_adj = self.dc(encode_data)
        tmp_encode = encode_data

        q = 1.0 / (1.0 + torch.sum(torch.pow(encode_data.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return tmp_encode, decode_adj, q




    def loss_gcn_clu(self,preds, labels,p,q,norm, pos_weight,lmd):
        loss_s = norm[0] * F.binary_cross_entropy_with_logits(preds, labels[0], pos_weight=pos_weight[0])
        loss_adj_m = norm[1] * F.binary_cross_entropy_with_logits(preds, labels[1], pos_weight=pos_weight[1])
        loss_kl = F.kl_div(q.log(), p, reduction='batchmean')
        loss = loss_kl + lmd*loss_s + loss_adj_m
        return loss


class DSC(nn.Module):
    def __init__(self, n_sample):
        super(DSC, self).__init__()
        self.self_expression = SelfExpression(n_sample)
        self.C = self.self_expression.Coefficient

    def forward(self,zs):
        gcn_encode_recon1 = self.self_expression(zs[0])
        gcn_encode_recon2 = self.self_expression(zs[1])
        gcn_encode_recons = [gcn_encode_recon1, gcn_encode_recon2]
        return gcn_encode_recons

    def loss_dsc(self, zs, z_recons,s,adj,gamma=1,weight=0.1):
        loss_coef = torch.norm(self.self_expression.Coefficient)
        loss_selfExp1 = torch.norm(zs[0] - z_recons[0], p='fro')
        loss_selfExp2 = torch.norm(zs[1] - z_recons[1], p='fro')
        loss_selfExp = weight*loss_selfExp1 + loss_selfExp2
        loss =  gamma*loss_coef + loss_selfExp
        return loss




class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.C = torch.sub(torch.ones(n, n, dtype=torch.float32),torch.eye(n)) #added by leon
        self.Coefficient = nn.Parameter(1.0e-8 * self.C, requires_grad=True)
        self.act = torch.sigmoid  #add by leon


    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, self.act(x))
        return y

