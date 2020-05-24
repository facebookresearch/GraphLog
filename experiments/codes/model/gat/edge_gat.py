"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
import torch

from codes.model.base_model import BaseModel as Net
from codes.model.inits import *
from codes.model.gat.layers import *
from codes.utils.util import get_param


class GatEncoder(Net):
    """
    Encoder which uses EdgeGatConv
    """

    def __init__(self, config, shared_embeddings=None):
        super(GatEncoder, self).__init__(config)

        # flag to enable one-hot embedding if needed
        self.graph_mode = True
        self.one_hot = self.config.model.gat.emb_type == "one-hot"
        self.edgeConvs = []

        ## Add EdgeGATConv params
        for l in range(config.model.gat.num_layers):
            in_channels = config.model.relation_embedding_dim
            out_channels = config.model.relation_embedding_dim
            heads = config.model.gat.num_heads
            edge_dim = config.model.relation_embedding_dim

            weight = torch.Tensor(size=(in_channels, heads * out_channels)).to(
                config.general.device
            )
            weight.requires_grad = True
            self.add_weight(
                weight,
                "EdgeGATConv.{}.weight".format(l),
                initializer=glorot,
                weight_norm=config.model.weight_norm,
            )

            att = torch.Tensor(size=(1, heads, 2 * out_channels + edge_dim)).to(
                config.general.device
            )
            att.requires_grad = True
            self.add_weight(
                att,
                "EdgeGATConv.{}.att".format(l),
                initializer=glorot,
                weight_norm=config.model.weight_norm,
            )

            edge_update = torch.Tensor(size=(out_channels + edge_dim, out_channels)).to(
                config.general.device
            )
            edge_update.requires_grad = True
            self.add_weight(
                edge_update,
                "EdgeGATConv.{}.edge_update".format(l),
                initializer=glorot,
                weight_norm=config.model.weight_norm,
            )

            if config.model.gat.bias and config.model.gat.concat:
                bias = torch.Tensor(size=(heads * out_channels,)).to(
                    config.general.device
                )
                bias.requires_grad = True
                self.add_weight(
                    bias,
                    "EdgeGATConv.{}.bias".format(l),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )
            elif config.model.gat.bias and not config.model.gat.concat:
                bias = torch.Tensor(size=(out_channels,)).to(config.general.device)
                bias.requires_grad = True
                self.add_weight(
                    bias,
                    "EdgeGATConv.{}.bias".format(l),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )

            self.edgeConvs.append(
                EdgeGatConv(
                    in_channels,
                    out_channels,
                    edge_dim,
                    heads=heads,
                    concat=config.model.gat.concat,
                    negative_slope=config.model.gat.negative_slope,
                    dropout=config.model.gat.dropout,
                    bias=config.model.gat.bias,
                )
            )

        ## Add classify params
        in_class_dim = (
            config.model.relation_embedding_dim * 2
            + config.model.relation_embedding_dim
        )
        self.add_classify_weights(in_class_dim)

    def prepare_param_idx(self, layer_id=0):
        full_name_idx = {n: i for i, n in enumerate(self.weight_names)}
        gat_layer_param_indx = [
            i for i, k in enumerate(self.weight_names) if "{}".format(layer_id) in k
        ]
        param_names = [self.weight_names[gi] for gi in gat_layer_param_indx]
        param_name_to_idx = {k: full_name_idx[k] for v, k in enumerate(param_names)}
        return param_name_to_idx

    def forward(self, batch, rel_emb=None):
        # import ipdb; ipdb.set_trace()
        data = batch.graphs
        param_name_to_idx = {k: v for v, k in enumerate(self.weight_names)}
        # initialize random node embeddings

        node_emb = torch.Tensor(
            size=(self.config.model.num_nodes, self.config.model.relation_embedding_dim)
        ).to(self.config.general.device)
        torch.nn.init.xavier_uniform_(node_emb, gain=1.414)
        x = F.embedding(data.x, node_emb)
        # x = F.embedding(data.x, self.weights[self.get_param_id(param_name_to_idx,
        #                                                        'node_embedding')])
        x = x.squeeze(1)
        # x = self.embedding(data.x).squeeze(1) # N x node_dim
        if rel_emb is not None:
            edge_attr = F.embedding(data.edge_attr, rel_emb)
        else:
            edge_attr = F.embedding(
                data.edge_attr,
                get_param(self.weights, param_name_to_idx, "relation_embedding"),
            )
        edge_attr = edge_attr.squeeze(1)
        # edge_attr = self.edge_embedding(data.edge_attr).squeeze(1) # E x edge_dim
        for nr in range(self.config.model.gat.num_layers - 1):
            param_name_dict = self.prepare_param_idx(nr)
            x = F.dropout(x, p=self.config.model.gat.dropout, training=self.training)
            x = self.edgeConvs[nr](
                x, data.edge_index, edge_attr, self.weights, param_name_dict
            )
            x = F.elu(x)
        x = F.dropout(x, p=self.config.model.gat.dropout, training=self.training)
        param_name_dict = self.prepare_param_idx(self.config.model.gat.num_layers - 1)
        x = self.edgeConvs[self.config.model.gat.num_layers - 1](
            x, data.edge_index, edge_attr, self.weights, param_name_dict
        )
        # restore x into B x num_node x dim
        chunks = torch.split(x, batch.num_nodes, dim=0)
        chunks = [p.unsqueeze(0) for p in chunks]
        # x = torch.cat(chunks, dim=0)
        return self.pyg_classify(chunks, batch.queries, self.weights, param_name_to_idx)


class GatedGatEncoder(Net):
    """
    Encoder which uses GatedEdgeGatConv
    """

    def __init__(self, config, shared_embeddings=None):
        super(GatedGatEncoder, self).__init__(config)

        # flag to enable one-hot embedding if needed
        self.graph_mode = True
        self.one_hot = self.config.model.gat.emb_type == "one-hot"
        self.edgeConvs = []

        ## Add EdgeGATConv params
        for l in range(config.model.gat.num_layers):
            in_channels = config.model.relation_embedding_dim
            out_channels = config.model.relation_embedding_dim
            heads = config.model.gat.num_heads
            edge_dim = config.model.relation_embedding_dim

            weight = torch.Tensor(size=(in_channels, heads * out_channels)).to(
                config.general.device
            )
            weight.requires_grad = True
            self.add_weight(
                weight,
                "GatedEdgeGATConv.{}.weight".format(l),
                initializer=glorot,
                weight_norm=config.model.weight_norm,
            )

            att = torch.Tensor(size=(1, heads, 2 * out_channels + edge_dim)).to(
                config.general.device
            )
            att.requires_grad = True
            self.add_weight(
                att,
                "GatedEdgeGATConv.{}.att".format(l),
                initializer=glorot,
                weight_norm=config.model.weight_norm,
            )

            if l == 0:
                # only add the gru weights once
                gru_weight_ih = torch.Tensor(
                    size=(out_channels + edge_dim, 3 * out_channels)
                ).to(config.general.device)
                gru_weight_ih.requires_grad = True
                self.add_weight(
                    gru_weight_ih,
                    "GatedEdgeGATConv.{}.gru_w_ih".format("_all_"),
                    weight_norm=config.model.weight_norm,
                )

                gru_weight_hh = torch.Tensor(size=(out_channels, 3 * out_channels)).to(
                    config.general.device
                )
                gru_weight_hh.requires_grad = True
                self.add_weight(
                    gru_weight_hh,
                    "GatedEdgeGATConv.{}.gru_w_hh".format("_all_"),
                    weight_norm=config.model.weight_norm,
                )

                gru_bias_ih = torch.Tensor(size=(3 * out_channels,)).to(
                    config.general.device
                )
                gru_bias_ih.requires_grad = True
                self.add_weight(
                    gru_bias_ih,
                    "GatedEdgeGATConv.{}.gru_b_ih".format("_all_"),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )

                gru_bias_hh = torch.Tensor(size=(3 * out_channels,)).to(
                    config.general.device
                )
                gru_bias_hh.requires_grad = True
                self.add_weight(
                    gru_bias_hh,
                    "GatedEdgeGATConv.{}.gru_b_hh".format("_all_"),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )

            if config.model.gat.bias and config.model.gat.concat:
                bias = torch.Tensor(size=(heads * out_channels,)).to(
                    config.general.device
                )
                bias.requires_grad = True
                self.add_weight(
                    bias,
                    "GatedEdgeGATConv.{}.bias".format(l),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )
            elif config.model.gat.bias and not config.model.gat.concat:
                bias = torch.Tensor(size=(out_channels,)).to(config.general.device)
                bias.requires_grad = True
                self.add_weight(
                    bias,
                    "GatedEdgeGATConv.{}.bias".format(l),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )

            self.edgeConvs.append(
                GatedEdgeGatConv(
                    in_channels,
                    out_channels,
                    edge_dim,
                    heads=heads,
                    concat=config.model.gat.concat,
                    negative_slope=config.model.gat.negative_slope,
                    dropout=config.model.gat.dropout,
                    bias=config.model.gat.bias,
                )
            )

        ## Add classify params
        in_class_dim = (
            config.model.relation_embedding_dim * 2
            + config.model.relation_embedding_dim
        )
        self.add_classify_weights(in_class_dim)

    def prepare_param_idx(self, layer_id=0):
        full_name_idx = {n: i for i, n in enumerate(self.weight_names)}
        gat_layer_param_indx = [
            i
            for i, k in enumerate(self.weight_names)
            if "{}".format(layer_id) in k or "_all_" in k
        ]
        param_names = [self.weight_names[gi] for gi in gat_layer_param_indx]
        param_name_to_idx = {k: full_name_idx[k] for v, k in enumerate(param_names)}
        return param_name_to_idx

    def forward(self, batch, rel_emb=None):
        data = batch.graphs
        param_name_to_idx = {k: v for v, k in enumerate(self.weight_names)}
        # initialize random node embeddings

        node_emb = torch.Tensor(
            size=(self.config.model.num_nodes, self.config.model.relation_embedding_dim)
        ).to(self.config.general.device)
        torch.nn.init.xavier_uniform_(node_emb, gain=1.414)
        x = F.embedding(data.x, node_emb)
        # x = F.embedding(data.x, self.weights[self.get_param_id(param_name_to_idx,
        #                                                        'node_embedding')])
        x = x.squeeze(1)
        # x = self.embedding(data.x).squeeze(1) # N x node_dim
        if rel_emb is not None:
            edge_attr = F.embedding(data.edge_attr, rel_emb)
        else:
            edge_attr = F.embedding(
                data.edge_attr,
                get_param(self.weights, param_name_to_idx, "relation_embedding"),
            )
        edge_attr = edge_attr.squeeze(1)
        # edge_attr = self.edge_embedding(data.edge_attr).squeeze(1) # E x edge_dim
        for nr in range(self.config.model.gat.num_layers - 1):
            param_name_dict = self.prepare_param_idx(nr)
            x = F.dropout(x, p=self.config.model.gat.dropout, training=self.training)
            x = self.edgeConvs[nr](
                x, data.edge_index, edge_attr, self.weights, param_name_dict
            )
            x = F.elu(x)
        x = F.dropout(x, p=self.config.model.gat.dropout, training=self.training)
        param_name_dict = self.prepare_param_idx(self.config.model.gat.num_layers - 1)
        if self.config.model.gat.num_layers > 0:
            x = self.edgeConvs[self.config.model.gat.num_layers - 1](
                x, data.edge_index, edge_attr, self.weights, param_name_dict
            )
        # restore x into B x num_node x dim
        chunks = torch.split(x, batch.num_nodes, dim=0)
        chunks = [p.unsqueeze(0) for p in chunks]
        # x = torch.cat(chunks, dim=0)
        return self.pyg_classify(chunks, batch.queries, self.weights, param_name_to_idx)
