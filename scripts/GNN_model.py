import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn


class GCN(nn.Module):

    def __init__(self, in_features, out_features, attn_heads):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GATConv(in_features, out_features, attn_heads)
    
    def forward(self, g, x):
        # inputs are features of nodes
        h = self.conv1(g, x)
        h = th.mean(h,1)
        h = nn.functional.relu(h)
        return h


class NodeMLP(nn.Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin1 = nn.Linear(in_features, out_features)
    
    def forward(self, h):
        h = self.lin1(h)
        h = nn.functional.relu(h)
        return h


class EdgeMLP(nn.Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin1 = nn.Linear(in_features, int(in_features/2))
        self.lin2 = nn.Linear(int(in_features/2), int(in_features/4))
        self.lin3 = nn.Linear(int(in_features/4), out_features)
    
    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        h = th.cat([h_u, h_v], 1)
        h = self.lin1(h)
        h = th.sigmoid(h)
        h = self.lin2(h)
        h = th.sigmoid(h)
        h = self.lin3(h)
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
    
    def __init__(self, in_features, gnn_hidden_features, attn_heads):
        super().__init__()
        self.nodemlp = NodeMLP(in_features, in_features*2)
        self.gcn = GCN(in_features*2, gnn_hidden_features, attn_heads)
        self.edgemlp = EdgeMLP(gnn_hidden_features*2, 1)
    
    def forward(self, g, x):
        h = self.nodemlp(x)
        h = self.gcn(g, h)
        h = self.edgemlp(g, h)
        return h
