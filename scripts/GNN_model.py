#!/usr/bin/env python

######################################## GNN_model.py ########################################
# PURPOSE: contains class definitions of GNN model
# EDIT TO: change GNN model
# ------------------------------------------Summary-------------------------------------------
# This script contains all class definitions specifying the GNN model itself. As such, these
# objects are only used in GNN_main. Tweaks to the model structure can be made here, but layer
# sizes are defined in options.py
##############################################################################################


import math
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):

    def __init__(self, gnn_size, in_features, attn_heads, dropout):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GATv2Conv(in_features, gnn_size[0], attn_heads[0], feat_drop=dropout, attn_drop=dropout, activation=nn.functional.relu)
        self.conv2 = dglnn.GATv2Conv(gnn_size[0]*attn_heads[0], gnn_size[1], attn_heads[1], feat_drop=dropout, attn_drop=dropout, activation=nn.functional.relu)

    def forward(self, g, x):
        # inputs are features of nodes
        h, a1 = self.conv1(g, x, get_attention=True)
        g.edata['attn1'] = a1
        h = h.view(*h.shape[:-2], h.shape[-1]*h.shape[-2])
        h, a2 = self.conv2(g, h, get_attention=True)
        g.edata['attn2'] = a2
        h = h.view(*h.shape[:-2], h.shape[-1]*h.shape[-2])
        return h


class NodeMLP(nn.Module):
    
    def __init__(self, nodemlp_size, in_features, dropout):
        super().__init__()
        self.lin = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        layer_sizes = [in_features]
        layer_sizes.extend(nodemlp_size)
        for i in range(len(layer_sizes)-1):
            self.lin.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, h):
        for layer in self.lin:
            h = F.relu(self.dropout(layer(h)))
        return h


class EdgeMLP(nn.Module):
    
    def __init__(self, edgemlp_size, in_features, out_features, dropout):
        super().__init__()
        self.lin = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        layer_sizes = [in_features]
        layer_sizes.extend(edgemlp_size)
        layer_sizes.extend([out_features])
        for i in range(len(layer_sizes)-1):
            self.lin.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
    
    def apply_edges(self, edges):
        h1_u = edges.src['h1']
        h1_v = edges.dst['h1']
        h2_u = edges.src['h2']
        h2_v = edges.dst['h2']
        h = th.cat([h1_u, h1_v, h2_u, h2_v], 1)
        for i, layer in enumerate(self.lin):
            if i != 0: h = th.relu(self.dropout(h))
            h = layer(h)
        return {'score': h}
    
    def forward(self, g, h1, h2):
        with g.local_scope():
            g.ndata['h1'] = h1
            g.ndata['h2'] = h2
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class DotProductPredictor(nn.Module):
    
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score']


class EdgePredModel(nn.Module):

    def __init__(self, nodemlp_size, gnn_size, edgemlp_size, in_features, out_features, attn_heads, dropout):
        super().__init__()
        self.nodemlp = NodeMLP(nodemlp_size, in_features, dropout)
        self.gcn = GCN(gnn_size, in_features, attn_heads, dropout)
        self.edgemlp = EdgeMLP(edgemlp_size, 2*(gnn_size[-1]*attn_heads[-1]+nodemlp_size[-1]), out_features, dropout)
    
    def forward(self, g, x):
        h1 = self.nodemlp(x)
        h2 = self.gcn(g, x)
        h = self.edgemlp(g, h1, h2)
        g.edata['pred'] = h #still needs to be passed through an activation function
        return h
