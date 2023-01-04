#coding: UTF-8
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch.utils import data
import random
import scipy.sparse as sp
from config import opt
# import data

laplace = 0
laplace_a =0
laplace_b = 0
features = 0
labels = 0

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def degree(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, 1).flatten()
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv


def degree_0_5(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv


def load_data(miss_1_or_2, NC, opt):
    global laplace, laplace_a, laplace_b, features, labels
    if NC == 0:
        print("Loading {} dataset..." .format(opt.network))

        dataset = Planetoid(root='./data', name=opt.network)
        features = sp.csr_matrix(np.array(dataset.data['x']))
        edges = np.array(dataset.data['edge_index']).T
        labels = encode_onehot(np.array(dataset.data['y']))

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32) #构建邻接矩阵
        adj = adj + sp.eye(adj.shape[0])

        # degree_2 = degree_0_5(adj)
        # adj = np.dot(degree_2, adj)
        # adj = np.dot(adj, degree_2)

        laplace = adj.toarray()
        eigen_value, eigen_vector = np.linalg.eig(laplace)

        nodes_count = labels.shape[0]
        creat_miss = np.random.randint(0, 100, nodes_count)
        miss_1 = np.int64(creat_miss > 50)

        eigen_value_1 = sp.diags(eigen_value * miss_1).A
        laplace_a = np.dot(eigen_vector, eigen_value_1)
        laplace_a = np.dot(laplace_a, np.linalg.inv(eigen_vector))
        laplace_a = laplace_a.real
        laplace_b = laplace - laplace_a

        threshold = 0.8

        laplace_a = np.int8(laplace_a>=threshold)
        laplace_a = laplace_a + np.eye(nodes_count)
        laplace_a = np.int8(laplace_a>=1)
        laplace_a = sp.coo_matrix(laplace_a)
        degree_2 = degree_0_5(laplace_a)
        laplace_a = np.dot(degree_2, laplace_a)
        laplace_a = np.dot(laplace_a, degree_2)
        laplace_a = laplace_a.toarray()


        laplace_b = np.int8(laplace_b>=threshold)
        laplace_b = laplace_b + np.eye(nodes_count)
        laplace_b = np.int8(laplace_b>=1)
        laplace_b = sp.coo_matrix(laplace_b)
        degree_2 = degree_0_5(laplace_b)
        laplace_b = np.dot(degree_2, laplace_b)
        laplace_b = np.dot(laplace_b, degree_2)
        laplace_b = laplace_b.toarray()

        
        degree_2 = degree_0_5(adj)
        adj = np.dot(degree_2, adj)
        adj = np.dot(adj, degree_2)
        laplace = adj.toarray()

        adj = torch.FloatTensor(np.array(adj.todense()))
        laplace = torch.FloatTensor(laplace)
        laplace_a = torch.FloatTensor(laplace_a)
        laplace_b = torch.FloatTensor(laplace_b)
        
        features = normalize(features)
        features = np.array(features.todense())
        the_h = features.max(0)
        epsilon = opt.privacy_budget
        the_lambda = the_h / epsilon
        noise = np.random.laplace(0, the_lambda, features.shape)
        features = features + noise
        features = torch.FloatTensor(features)
    else:
        labels_map = {i: [] for i in range(labels.shape[1])}
        labels_1 = np.where(labels)[1]
        for i in range(labels_1.shape[0]):
            labels_map[labels_1[i]].append(i)
        for ele in labels_map:
            random.shuffle(labels_map[ele])
        idx_train = list()
        idx_val = list()
        idx_test = list()
        for ele in labels_map:
            idx_train.extend(labels_map[ele][0:int(opt.train_rate * labels_1.shape[0])])
            idx_val.extend(labels_map[ele][int(opt.train_rate * labels_1[0]):int((opt.train_rate + opt.val_rate) * labels_1.shape[0])])
            idx_test.extend(labels_map[ele][int((opt.train_rate + opt.val_rate) * labels_1.shape[0]):])

        if miss_1_or_2 % 2 == 0:
            return laplace_a, features, labels_1, idx_train, idx_val, idx_test, laplace

        if miss_1_or_2 % 2 == 1:
            return laplace_b, features, labels_1, idx_train, idx_val, idx_test, laplace


class Dataload(data.Dataset):

    def __init__(self, labels, id):
        self.data = id
        self.labels = labels

    def __getitem__(self, index):
        return index, self.labels[index]

    def __len__(self):
        return self.data.__len__()