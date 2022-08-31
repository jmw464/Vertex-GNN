"""Torch modules for graph attention networks v2 (GATv2)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

class GATv4Conv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 share_weights=False,
                 share_attn_weights=False):
        super(GATv4Conv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc_src_mut = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=bias)
        if share_weights:
            self.fc_dst_mut = self.fc_src_mut
        else:
            self.fc_dst_mut = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias)
        if share_attn_weights:
            self.fc_src_self = self.fc_src_mut
        else:
            self.fc_src_self = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias)
        self.fc_src_lin = nn.Linear(
            self._in_src_feats, out_feats, bias=bias)
        self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.share_weights = share_weights
        self.share_attn_weights = share_attn_weights
        self.bias = bias
        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src_mut.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_src_lin.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_src_mut.bias, 0)
            nn.init.constant_(self.fc_src_self.bias, 0)
            nn.init.constant_(self.fc_src_lin.bias, 0)
        if not self.share_weights:
            nn.init.xavier_normal_(self.fc_dst_mut.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.fc_dst_mut.bias, 0)
        if not self.share_attn_weights:
            nn.init.xavier_normal_(self.fc_src_self.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.fc_src_self.bias, 0)
        nn.init.xavier_normal_(self.attn, gain=gain)


    def set_allow_zero_in_degree(self, set_value):

        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
       
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            h_src = h_dst = self.feat_drop(feat)
            feat_src_mut = self.fc_src_mut(h_src).view(
                -1, self._num_heads, self._out_feats)
            feat_src_lin = self.fc_src_lin(h_src)
            feat_src_lin = th.unsqueeze(feat_src_lin, dim=1)
            if self.share_weights:
                feat_dst_mut = feat_src_mut
            else:
                feat_dst_mut = self.fc_dst_mut(h_src).view(
                    -1, self._num_heads, self._out_feats)
            if self.share_attn_weights:
                feat_src_self = feat_src_mut
            else:
                feat_src_self = self.fc_src_self(h_src).view(
                    -1, self._num_heads, self._out_feats)
            if graph.is_block:
                feat_dst_mut = feat_src_mut[:graph.number_of_dst_nodes()]
                h_dst = h_dst[:graph.number_of_dst_nodes()]
            graph.srcdata.update({'el_mut': feat_src_mut})# (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er_mut': feat_dst_mut})
            graph.srcdata.update({'el_self': feat_src_self})
            graph.apply_edges(fn.u_add_v('el_mut', 'er_mut', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))# (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) # (num_edge, num_heads)
            # message passing
            graph.update_all(fn.u_mul_e('el_self', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = th.cat((feat_src_lin, graph.dstdata['ft']),-2)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
