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


def bin_loss(output, labels, batch_edges, weight, a, b, device):
    bce_term = nn.BCEWithLogitsLoss(pos_weight=weight, reduction='sum').to(device)
    activation = nn.Sigmoid()
    act_output = activation(output)

    a = th.Tensor([a]).to(device)
    one = th.Tensor([1]).to(device)
    epsilon = 10e-5
    
    penalty_term = edge_index = 0
    if b != 0:
        for nedges in batch_edges:
            label_sum = th.sum(labels[edge_index:edge_index+nedges])
            output_sum = th.sum(act_output[edge_index:edge_index+nedges])
            fake_penalty = th.square(label_sum-output_sum)/(nedges*(label_sum+output_sum+epsilon))
            penalty_term += (th.pow(a,fake_penalty)-one)/(a-one)
            edge_index += nedges

    return bce_term(output, labels) + 0.1*b*len(batch_edges)*penalty_term


class GCNConv(nn.Module):
    
    def __init__(self, gnn_size, in_features):
        super(GCNConv, self).__init__()
        self.conv1 = dglnn.GraphConv(in_features, gnn_size[0], activation=nn.functional.relu)
        #self.conv2 = dglnn.GraphConv(gnn_size[0], gnn_size[1], activation=nn.functional.relu)

    def forward(self, g, x):
        # inputs are features of nodes
        h = self.conv1(g, x)
        #h = self.conv2(g, h)
        return h


class GATConv(nn.Module):

    def __init__(self, gnn_size, in_features, attn_heads, dropout):
        super(GATConv, self).__init__()
        self.conv1 = dglnn.GATv2Conv(in_features, int(gnn_size[0]/attn_heads[0]), attn_heads[0], feat_drop=dropout, attn_drop=dropout, activation=nn.functional.relu)
        #self.conv2 = dglnn.GATv2Conv(gnn_size[0], int(gnn_size[1]/attn_heads[1]), attn_heads[1], feat_drop=dropout, attn_drop=dropout, activation=nn.functional.relu)

    def forward(self, g, x):
        # inputs are features of nodes
        h, a1 = self.conv1(g, x, get_attention=True)
        g.edata['attn1'] = a1
        h = h.view(*h.shape[:-2], h.shape[-1]*h.shape[-2])
        #h, a2 = self.conv2(g, h, get_attention=True)
        #g.edata['attn2'] = a2
        #h = h.view(*h.shape[:-2], h.shape[-1]*h.shape[-2])
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


class EdgeMLP_alt(nn.Module):

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
        h_u = edges.src['h']
        h_v = edges.dst['h']
        h = th.cat([h_u, h_v], 1)
        for i, layer in enumerate(self.lin):
            if i != 0: h = th.relu(self.dropout(h))
            h = layer(h)
        return {'score': h}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


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

    def __init__(self, model_type, gnn_type, nodemlp_size, gnn_size, edgemlp_size, in_features, out_features, attn_heads, dropout):
        super().__init__()
        self.model_type = model_type
        if self.model_type == 'mlp' or self.model_type == 'both':
            self.nodemlp = NodeMLP(nodemlp_size, in_features, dropout)
        if self.model_type == 'gnn' or self.model_type == 'both':
            if gnn_type == 'gcn': self.gcn = GCNConv(gnn_size, in_features)
            elif gnn_type == 'gat': self.gcn = GATConv(gnn_size, in_features, attn_heads, dropout)

        if self.model_type == 'both':
            self.edgemlp = EdgeMLP(edgemlp_size, 2*(gnn_size[-1]+nodemlp_size[-1]), out_features, dropout)
        elif self.model_type == 'mlp':
            self.edgemlp = EdgeMLP_alt(edgemlp_size, 2*nodemlp_size[-1], out_features, dropout)
        elif self.model_type == 'gnn':
            self.edgemlp = EdgeMLP_alt(edgemlp_size, 2*gnn_size[-1], out_features, dropout)

    def forward(self, g, x):
        if self.model_type == 'both':
            h1 = self.nodemlp(x)
            h2 = self.gcn(g, x)
            h = self.edgemlp(g, h1, h2)
        elif self.model_type == 'mlp':
            h = self.nodemlp(x)
            h = self.edgemlp(g, h)
        elif self.model_type == 'gnn':
            h = self.gcn(g, x)
            h = self.edgemlp(g, h)
        g.edata['pred'] = h #still needs to be passed through an activation function
        return h
