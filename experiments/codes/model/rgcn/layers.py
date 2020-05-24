"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

from codes.model.inits import uniform
from codes.utils.util import get_param, get_param_id


class RGCNConv(MessagePassing):
    """Derived from RGCNConv https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/rgcn_conv.html#RGCNConv
    Changes:
        - Weights have to be passed in forward using `param`
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_relations,
        num_bases,
        root_weight=True,
        bias=True,
        **kwargs
    ):
        super(RGCNConv, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.root_weight = root_weight
        self.use_bias = bias

    def forward(
        self,
        x,
        edge_index,
        edge_types,
        relation_weights,
        params,
        param_name_dict,
        edge_norm=None,
        size=None,
    ):
        self.basis = get_param(params, param_name_dict, "basis")
        if self.root_weight:
            self.root = get_param(params, param_name_dict, "root")
        if self.use_bias:
            self.bias = get_param(params, param_name_dict, "bias")

        return self.propagate(
            edge_index,
            size=size,
            x=x,
            edge_types=edge_types,
            edge_norm=edge_norm,
            relation_weights=relation_weights,
        )

    def message(self, x_j, edge_index_j, edge_types, edge_norm, relation_weights):
        w = torch.matmul(relation_weights, self.basis.view(self.num_bases, -1))

        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        w = torch.index_select(w, 0, edge_types)
        out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root_weight:
            out = aggr_out + torch.matmul(x, self.root)

        if self.use_bias:
            out = out + self.bias
        return out

    def __repr__(self):
        return "{}({}, {}, num_relations={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_relations,
        )
