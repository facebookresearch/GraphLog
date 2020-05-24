"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
# GAT with Edge features
import torch
import torch.nn.functional as F
import torch.nn._VF as VF
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, scatter_
from codes.utils.util import get_param, get_param_id


class EdgeGatConv(MessagePassing):
    """Modified Graph Attention Networks (GATConv) (https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv)
    which supports Edge features.
    Changes:
    - attention now computed along with edge features
    - weights have to be passed as `params`
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        heads=1,
        concat=False,
        negative_slope=0.2,
        dropout=0.0,
        bias=True,
    ):
        super(EdgeGatConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        # params
        self.att = None
        self.edge_update = None
        self.use_bias = bias

    def forward(self, x, edge_index, edge_attr, params, param_name_dict, size=None):
        self.att = get_param(params, param_name_dict, "att")
        self.edge_update = get_param(params, param_name_dict, "edge_update")
        self.bias = None
        if self.use_bias:
            self.bias = get_param(params, param_name_dict, "bias")
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(
            edge_index.device
        )
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)  # (500, 10)
        # Note: we need to add blank edge attributes for self loops
        weight = get_param(params, param_name_dict, "weight")
        if torch.is_tensor(x):
            x = torch.matmul(x, weight)
        else:
            x = (
                None if x[0] is None else torch.matmul(x[0], weight),
                None if x[1] is None else torch.matmul(x[1], weight),
            )
        # x = x.view(-1, self.heads, self.out_channels)
        # x = torch.mm(x, weight).view(-1, self.heads, self.out_channels)
        return self.propagate(
            edge_index, size=size, x=x, num_nodes=x.size(0), edge_attr=edge_attr
        )

    def message(self, edge_index_i, x_i, x_j, size_i, num_nodes, edge_attr):
        # Compute attention coefficients
        # N.B - only modification is the attention is now computed with the edge attributes
        x_j = x_j.view(-1, self.heads, self.out_channels)
        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)
        x_j = torch.cat([x_j, edge_attr], dim=-1)

        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels :]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)  # N x (out_channels + edge_dim)
        # TODO: gated update here
        aggr_out = torch.mm(aggr_out, self.edge_update)  # N x out_channels

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out  # N x out_channels
        # return aggr_out[:, :, :self.out_channels].squeeze(1)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class GatConv(MessagePassing):
    """Modified GatConv (https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv)

    Changes:
    - Weights have to be passed in forward in `params`
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        heads=1,
        concat=False,
        negative_slope=0.2,
        dropout=0.0,
        bias=True,
    ):
        super(GatConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        # params
        self.att = None
        self.edge_update = None
        self.use_bias = bias

    def forward(self, x, edge_index, edge_attr, params, param_name_dict, size=None):
        self.att = get_param(params, param_name_dict, "att")
        self.bias = None
        if self.use_bias:
            self.bias = get_param(params, param_name_dict, "bias")
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Note: we need to add blank edge attributes for self loops
        weight = get_param(params, param_name_dict, "weight")
        if torch.is_tensor(x):
            x = torch.matmul(x, weight)
        else:
            x = (
                None if x[0] is None else torch.matmul(x[0], weight),
                None if x[1] is None else torch.matmul(x[1], weight),
            )
        # x = x.view(-1, self.heads, self.out_channels)
        # x = torch.mm(x, weight).view(-1, self.heads, self.out_channels)
        return self.propagate(
            edge_index, size=size, x=x, num_nodes=x.size(0), edge_attr=edge_attr
        )

    def message(self, edge_index_i, x_i, x_j, size_i, num_nodes, edge_attr):
        # Compute attention coefficients
        # N.B - only modification is the attention is now computed with the edge attributes
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels :]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)  # N x (out_channels + edge_dim)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out  # N x out_channels

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class GatedEdgeGatConv(MessagePassing):
    """Derived from EdgeGatConv
    Implements gated (GRU) update 
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        heads=1,
        concat=False,
        negative_slope=0.2,
        dropout=0.0,
        bias=True,
    ):
        super(GatedEdgeGatConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        # params
        self.att = None
        self.edge_update = None
        self.use_bias = bias

    def forward(self, x, edge_index, edge_attr, params, param_name_dict, size=None):
        self.att = get_param(params, param_name_dict, "att")
        # self.edge_update = params[self.get_param_id(param_name_dict, "edge_update")]
        self.bias = None
        if self.use_bias:
            self.bias = get_param(params, param_name_dict, "bias")
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # get gru params
        self.gru_weight_ih = get_param(params, param_name_dict, "gru_w_ih")
        self.gru_weight_hh = get_param(params, param_name_dict, "gru_w_hh")
        self.gru_bias_ih = get_param(params, param_name_dict, "gru_b_ih")
        self.gru_bias_hh = get_param(params, param_name_dict, "gru_b_hh")
        self.gru_hx = x

        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(
            edge_index.device
        )
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)  # (500, 10)
        # Note: we need to add blank edge attributes for self loops
        weight = get_param(params, param_name_dict, "weight")
        if torch.is_tensor(x):
            x = torch.matmul(x, weight)
        else:
            x = (
                None if x[0] is None else torch.matmul(x[0], weight),
                None if x[1] is None else torch.matmul(x[1], weight),
            )
        return self.propagate(
            edge_index, size=size, x=x, num_nodes=x.size(0), edge_attr=edge_attr
        )

    def message(self, edge_index_i, x_i, x_j, size_i, num_nodes, edge_attr):
        # Compute attention coefficients
        # N.B - only modification is the attention is now computed with the edge attributes
        x_j = x_j.view(-1, self.heads, self.out_channels)
        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)
        x_j = torch.cat([x_j, edge_attr], dim=-1)

        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels :]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)  # N x (out_channels + edge_dim)
        # Gated update
        aggr_out = self.gru_cell(aggr_out, self.gru_hx)
        # aggr_out = torch.mm(aggr_out, self.edge_update)  # N x out_channels

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out  # N x out_channels

    def gru_cell(self, x, hx):
        """
        implementation of GRUCell which is compatible with functional elements
        :param x:
        :return:
        """
        x = x.view(-1, x.size(1))

        gate_x = F.linear(x, self.gru_weight_ih.t(), self.gru_bias_ih)
        gate_h = F.linear(hx, self.gru_weight_hh.t(), self.gru_bias_hh)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hx - newgate)
        return hy

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class GatedGatConv(MessagePassing):
    """Derived from GatConv
    Implements gated (GRU) update 
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        heads=1,
        concat=False,
        negative_slope=0.2,
        dropout=0.0,
        bias=True,
    ):
        super(GatedGatConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        # params
        self.att = None
        self.edge_update = None
        self.use_bias = bias

    def forward(self, x, edge_index, edge_attr, params, param_name_dict, size=None):
        self.att = get_param(params, param_name_dict, "att")
        # self.edge_update = params[self.get_param_id(param_name_dict, 'edge_update')]
        self.bias = None
        if self.use_bias:
            self.bias = get_param(params, param_name_dict, "bias")
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # get gru params
        self.gru_weight_ih = get_param(params, param_name_dict, "gru_w_ih")
        self.gru_weight_hh = get_param(params, param_name_dict, "gru_w_hh")
        self.gru_bias_ih = get_param(params, param_name_dict, "gru_b_ih")
        self.gru_bias_hh = get_param(params, param_name_dict, "gru_b_hh")
        self.gru_hx = x

        # Note: we need to add blank edge attributes for self loops
        weight = get_param(params, param_name_dict, "weight")
        if torch.is_tensor(x):
            x = torch.matmul(x, weight)
        else:
            x = (
                None if x[0] is None else torch.matmul(x[0], weight),
                None if x[1] is None else torch.matmul(x[1], weight),
            )
        return self.propagate(
            edge_index, size=size, x=x, num_nodes=x.size(0), edge_attr=edge_attr
        )

    def message(self, edge_index_i, x_i, x_j, size_i, num_nodes, edge_attr):
        # Compute attention coefficients
        # N.B - only modification is the attention is now computed with the edge attributes
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels :]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)  # N x (out_channels + edge_dim)
        aggr_out = self.gru_cell(aggr_out, self.gru_hx)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out  # N x out_channels

    def gru_cell(self, x, hx):
        """
        implementation of GRUCell which is compatible with functional elements
        :param x:
        :return:
        """
        x = x.view(-1, x.size(1))

        gate_x = F.linear(x, self.gru_weight_ih.t(), self.gru_bias_ih)
        gate_h = F.linear(hx, self.gru_weight_hh.t(), self.gru_bias_hh)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hx - newgate)
        return hy

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )
