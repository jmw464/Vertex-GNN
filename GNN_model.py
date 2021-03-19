import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(in_features, hidden_features)
        self.conv2 = SAGEConv(hidden_features, out_features)
        #self.conv3 = GCNConv(int(hidden_features/2), out_features)

    def forward(self, x, edge_index):
        # inputs are features of nodes
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        #h = self.conv3(h, edge_index)
        #h = h.tanh()
        return h


class MLPPredictor(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.lin2 = nn.Linear(hidden_features, out_features)
    def forward(self, x, edge_index, device):
        nedges = list(edge_index.shape)[1]
        he = th.empty([nedges, self.in_features]).to(device)
        for i in range(nedges):
            edge = edge_index[:,i]
            sender = edge[0]
            receiver = edge[1]
            he[i] = th.cat((x[sender],x[receiver]))

        he = he.double()
        he = self.lin1(he)
        he = he.tanh()
        he = self.lin2(he)
        he = he.tanh()
        return he


class DotProductPredictor(nn.Module):
    def forward(self, x, edge_index):
        nedges = list(edge_index.shape)[1]
        he = th.empty([nedges, 1])
        for i in range(nedges):
            edge = edge_index[:,i]
            sender = edge[0]
            receiver = edge[1]
            he[i] = th.dot(x[sender],x[receiver])
        return he


class EdgePredModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.gcn = GCN(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features*2, out_features, 1)
    def forward(self, x, edge_index, device):
        h = self.gcn(x, edge_index)
        h = self.pred(h, edge_index, device)
        h = h.sigmoid()
        return h
