import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn


class GCN(nn.Module):

    def __init__(self, gnn_size, attn_heads):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GATConv(gnn_size[0], gnn_size[1], attn_heads)
        self.conv2 = dglnn.GATConv(gnn_size[1], gnn_size[2], attn_heads)

    def forward(self, g, x):
        # inputs are features of nodes
        h = self.conv1(g, x)
        h = th.mean(h,1)
        h = nn.functional.relu(h)
        h = self.conv2(g, h)
        h = th.mean(h,1)
        h = nn.functional.relu(h)
        return h


class NodeMLP(nn.Module):
    
    def __init__(self, nodemlp_size):
        super().__init__()
        self.lin1 = nn.Linear(nodemlp_size[0], nodemlp_size[1])
    
    def forward(self, h):
        h = self.lin1(h)
        h = nn.functional.relu(h)
        return h


class EdgeMLP(nn.Module):
    
    def __init__(self, edgemlp_size, out_features):
        super().__init__()
        self.lin1 = nn.Linear(edgemlp_size[0], edgemlp_size[1])
        self.lin2 = nn.Linear(edgemlp_size[1], edgemlp_size[2])
        self.lin3 = nn.Linear(edgemlp_size[2], out_features)
    
    def apply_edges(self, edges):
        h1_u = edges.src['h1']
        h1_v = edges.dst['h1']
        h2_u = edges.src['h2']
        h2_v = edges.src['h2']
        h = th.cat([h1_u, h1_v, h2_u, h2_v], 1)
        h = self.lin1(h)
        h = th.sigmoid(h)
        h = self.lin2(h)
        h = th.sigmoid(h)
        h = self.lin3(h)
        return {'score': h}
    
    def forward(self, g, h1, h2):
        with g.local_scope():
            g.ndata['h1'] = h1
            g.ndata['h2'] = h2
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class DotProductPredictor(nn.Module):
    
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class EdgePredModel(nn.Module):
    
    def __init__(self, nodemlp_size, gnn_size, edgemlp_size, out_features, attn_heads):
        super().__init__()
        self.nodemlp = NodeMLP(nodemlp_size)
        self.gcn = GCN(gnn_size, attn_heads)
        self.edgemlp = EdgeMLP(edgemlp_size, out_features)
    
    def forward(self, g, x):
        h1 = self.nodemlp(x)
        h2 = self.gcn(g, x)
        h = self.edgemlp(g, h1, h2)
        g.edata['pred'] = h #still needs to be passed through an activation function
        return h
