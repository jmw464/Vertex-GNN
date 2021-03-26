import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GATConv(in_features, out_features, 2)

    def forward(self, g, x):
        # inputs are features of nodes
        h = self.conv1(g, x)
        h = th.mean(h,1)
        h = nn.functional.relu(h)
        return h


class MLPPredictor(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.lin1 = nn.Linear(in_features*2, hidden_features)
        self.lin2 = nn.Linear(hidden_features, out_features)
    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        h = self.lin1(th.cat([h_u, h_v], 1))
        h = self.lin2(h)
        return {'score': h}
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class EdgePredModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.gcn = GCN(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features, out_features, 1)
    def forward(self, g, x):
        h = self.gcn(g, x)
        h = self.pred(g, h)
        return h
