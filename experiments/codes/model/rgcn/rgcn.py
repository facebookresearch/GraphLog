"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
import torch
import torch.nn.functional as F
from codes.model.base_model import BaseModel as Net
from codes.model.inits import *
from codes.model.rgcn.layers import *
from codes.utils.util import prepare_param_idx


class CompositionRGCNEncoder(Net):
    """Composition function which uses RGCN
    Accepts the relation embedding matrix as parameter
    """

    def __init__(self, config):
        super(CompositionRGCNEncoder, self).__init__(config)
        self.name = "CompositionRGCNConv"
        self.rgcn_layers = []
        for l in range(config.model.rgcn.num_layers):
            in_channels = config.model.relation_embedding_dim
            out_channels = config.model.relation_embedding_dim
            num_bases = config.model.relation_embedding_dim
            uniform_size = num_bases * in_channels

            basis = torch.Tensor(size=(num_bases, in_channels, out_channels)).to(
                config.general.device
            )
            basis.requires_grad = True
            self.add_weight(
                basis,
                "{}.{}.basis".format(self.name, l),
                initializer=(uniform, uniform_size),
                weight_norm=config.model.weight_norm,
            )

            if config.model.rgcn.root_weight:
                root = torch.Tensor(size=(in_channels, out_channels)).to(
                    config.general.device
                )
                root.requires_grad = True
                self.add_weight(
                    root,
                    "{}.{}.root".format(self.name, l),
                    initializer=(uniform, uniform_size),
                    weight_norm=config.model.weight_norm,
                )

            if config.model.rgcn.bias:
                bias = torch.Tensor(size=(out_channels,)).to(config.general.device)
                bias.requires_grad = True
                self.add_weight(
                    bias,
                    "{}.{}.bias".format(self.name, l),
                    initializer=(uniform, uniform_size),
                    weight_norm=config.model.weight_norm,
                )

            self.rgcn_layers.append(
                RGCNConv(
                    in_channels,
                    out_channels,
                    config.model.num_classes,
                    num_bases,
                    root_weight=config.model.rgcn.root_weight,
                    bias=config.model.rgcn.bias,
                )
            )

        ## Add classify params
        in_class_dim = (
            config.model.relation_embedding_dim * 2
            + config.model.relation_embedding_dim
        )
        self.add_classify_weights(in_class_dim)

    def forward(self, batch, rel_emb):
        data = batch.graphs
        param_name_to_idx = {k: v for v, k in enumerate(self.weight_names)}
        # initialize nodes randomly
        node_emb = torch.Tensor(
            size=(self.config.model.num_nodes, self.config.model.relation_embedding_dim)
        ).to(self.config.general.device)
        torch.nn.init.xavier_uniform_(node_emb, gain=1.414)
        x = F.embedding(data.x, node_emb)
        x = x.squeeze(1)
        # get edge attributes
        edge_types = data.edge_attr - 1
        edge_attr = rel_emb
        for nr in range(self.config.model.rgcn.num_layers - 1):
            param_name_dict = prepare_param_idx(self.weight_names, nr)
            x = F.dropout(x, p=self.config.model.rgcn.dropout, training=self.training)
            x = self.rgcn_layers[nr](
                x, data.edge_index, edge_types, edge_attr, self.weights, param_name_dict
            )
            x = F.relu(x)
        param_name_dict = prepare_param_idx(
            self.weight_names, self.config.model.rgcn.num_layers - 1
        )
        x = self.rgcn_layers[self.config.model.rgcn.num_layers - 1](
            x, data.edge_index, edge_types, edge_attr, self.weights, param_name_dict
        )
        # restore x into B x num_node x dim
        chunks = torch.split(x, batch.num_nodes, dim=0)
        chunks = [p.unsqueeze(0) for p in chunks]
        # x = torch.cat(chunks, dim=0)
        return self.pyg_classify(chunks, batch.queries, self.weights, param_name_to_idx)
