# GCN implementation of 'SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS' by Thomas N. Kipf and Max Welling
# Implementation will be using Pytorch, Numpy, Pandas

# 1. Study Kipf's original GCN paper (https://arxiv.org/abs/1609.02907) to understand
# the architecture and operations involved in GCNs.
# 2. Implement the GCN architecture (get help from the paper) with two GCN layers
# with 16 units each, RelU activation function and dropout rate = 0.5 .
# 3. Train the GCN model on the CORA graph dataset for the node classification
# task(Optimizer=Adam and Learning Rate = 0.01).
# 4. Evaluate the performance of the GCN model using the same evaluation metrics
# as fo LR.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, input, adj):
        # Add self-connections to the adjacency matrix
        I = torch.eye(adj.size(0)).to(adj.device)
        A_hat = I + adj
        
        # Compute the degree matrix D_hat
        D_hat_diag = torch.sum(A_hat, dim=1).pow(-0.5)
        D_hat = torch.diag(D_hat_diag)
        
        # Compute the symmetrically normalized adjacency matrix
        A_hat_prime = torch.mm(torch.mm(D_hat, A_hat), D_hat)
        
        # Apply linear transformation and return
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
    

def train(epoch, model, features, adj, labels, idx_train, optimizer, loss_func):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = loss_func(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}: Training loss {loss.item()}')

def evaluate(idx, model, features, adj, labels, loss_func):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss = loss_func(output[idx], labels[idx])
        pred = output.max(1)[1]
        accuracy = (pred[idx] == labels[idx]).float().mean()
        return loss.item(), accuracy.item()
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def loadData():
    # Load the CORA dataset
    data = pd.read_csv('dataset/cora.content', sep='\t', header=None)
    features = torch.FloatTensor(data.iloc[:, 1:-1].values)
    # TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
    labels = torch.LongTensor(LabelEncoder().fit_transform(data.iloc[:, -1]))
    
    # Load the CORA graph. There are two files: cora_train.cites and cora_test.cites. Format is <destination_id>\t<source_id>
    # To create the graph we will use both files.
    edge_train = pd.read_csv('dataset/cora_train.cites', sep='\t', header=None)
    edge_test = pd.read_csv('dataset/cora_test.cites', sep='\t', header=None)
    edges = pd.concat([edge_train, edge_test], axis=0)

    # Create the graph
    graph = nx.from_pandas_edgelist(edges, source=1, target=0)
    # Convert scipy sparse matrix to torch sparse tensor
    adj = nx.adjacency_matrix(graph)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # Indices handling
    idx_train = torch.LongTensor(edge_train.iloc[:, 1].unique())
    idx_test = torch.LongTensor(edge_test.iloc[:, 1].unique())
    
    return features, adj, labels, idx_train, idx_test
    
def main():
    # Load the data
    features, adj, labels, idx_train, idx_test = loadData()
    
    # Split the data into training and testing sets
    idx_train, idx_val = train_test_split(idx_train, test_size=0.2)
    
    # Initialize the model
    model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    
    # Train the model
    for epoch in range(200):
        train(epoch, model, features, adj, labels, idx_train, optimizer, loss_func)
        
    # Evaluate the model
    loss_train, acc_train = evaluate(idx_train, model, features, adj, labels, loss_func)
    loss_val, acc_val = evaluate(idx_val, model, features, adj, labels, loss_func)
    loss_test, acc_test = evaluate(idx_test, model, features, adj, labels, loss_func)
    
    print(f'Training loss: {loss_train}, Training accuracy: {acc_train}')
    print(f'Validation loss: {loss_val}, Validation accuracy: {acc_val}')
    print(f'Test loss: {loss_test}, Test accuracy: {acc_test}')

if __name__ == '__main__':
    main()