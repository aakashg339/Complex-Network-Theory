import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, input, adj):
        I = torch.eye(adj.size(0)).to(adj.device)
        A_hat = I + adj
        D_hat_diag = torch.sum(A_hat, dim=1).pow(-0.5)
        D_hat = torch.diag(D_hat_diag)
        A_hat_prime = torch.mm(torch.mm(D_hat, A_hat), D_hat)
        return self.linear(torch.mm(A_hat_prime, input))

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

# Loading and Preprocessing Functions
def load_cora_data(content_path, train_path, test_path):
    # Read content file
    df_content = pd.read_csv(content_path, header=None, sep='\t')
    features = torch.FloatTensor(np.array(df_content.iloc[:, 1:-1], dtype=np.float32))
    labels = LabelEncoder().fit_transform(df_content.iloc[:, -1])
    labels = torch.LongTensor(labels)

    # Build graph
    idx = np.array(df_content[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = pd.read_csv(train_path, header=None, sep='\t').values
    edges_unordered = np.append(edges_unordered, pd.read_csv(test_path, header=None, sep='\t').values, axis=0)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # Build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = preprocess_adj(adj)
    return features, adj, labels

def preprocess_adj(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_inv_sqrt = np.power(rowsum, -0.5).flatten()
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
    degree_mat_inv_sqrt = sp.diags(degree_inv_sqrt)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return torch.FloatTensor(np.array(adj_normalized.todense()))

# Model Training and Evaluation
def train_model(model, optimizer, features, adj, labels, idx_train, epochs=200):
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Training loss {loss.item()}')

def evaluate_model(model, features, adj, labels, idx_test):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss = F.nll_loss(output[idx_test], labels[idx_test])
        pred = output[idx_test].max(1)[1].type_as(labels[idx_test])
        accuracy = pred.eq(labels[idx_test]).double().mean()
        print(f'Test Set Results:\nLoss: {loss.item()}\nAccuracy: {accuracy.item()}')

# Main
content_path = 'dataset/cora.content'
train_path = 'dataset/cora_train.cites'
test_path = 'dataset/cora_test.cites'

features, adj, labels = load_cora_data(content_path, train_path, test_path)
idx = np.arange(len(labels))
idx_train, idx_test = train_test_split(idx, test_size=0.1, random_state=42)

model = GCN(nfeat=features.shape[1], nhid=16, nclass=len(torch.unique(labels)), dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_model(model, optimizer, features, adj, labels, torch.LongTensor(idx_train))
evaluate_model(model, features, adj, labels, torch.LongTensor(idx_test))
