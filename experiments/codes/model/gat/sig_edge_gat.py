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
from codes.model.gat.layers import *
from codes.model.inits import *
from codes.utils.util import get_param


class NodeGatEncoder(Net):
    """
    Encoder which uses `GatConv` and works on graphs without considering the edge
    """

    def __init__(self, config, shared_embeddings=None):
        super(NodeGatEncoder, self).__init__(config)

        # flag to enable one-hot embedding if needed
        self.graph_mode = True
        self.one_hot = self.config.model.signature_gat.emb_type == "one-hot"
        self.edgeConvs = []

        # common node & relation embedding
        # we keep one node embedding for all nodes, and individual relation embedding for relation nodes
        emb = torch.Tensor(
            size=(config.model.num_classes + 1, config.model.relation_embedding_dim)
        ).to(config.general.device)
        # rel_emb = torch.Tensor(size=(1, config.model.relation_embedding_dim)).to(config.general.device)
        emb.requires_grad = config.model.signature_gat.learn_node_and_rel_emb
        torch.nn.init.xavier_normal_(emb)
        self.add_weight(emb, "common_emb", weight_norm=config.model.weight_norm)

        ## Add EdgeGATConv params
        for l in range(config.model.signature_gat.num_layers):
            in_channels = config.model.relation_embedding_dim
            out_channels = config.model.relation_embedding_dim
            heads = config.model.signature_gat.num_heads
            edge_dim = config.model.relation_embedding_dim

            weight = torch.Tensor(size=(in_channels, heads * out_channels)).to(
                config.general.device
            )
            weight.requires_grad = True
            self.add_weight(
                weight,
                "GATConv.{}.weight".format(l),
                initializer=glorot,
                weight_norm=config.model.weight_norm,
            )

            att = torch.Tensor(size=(1, heads, 2 * out_channels)).to(
                config.general.device
            )
            att.requires_grad = True
            self.add_weight(
                att,
                "GATConv.{}.att".format(l),
                initializer=glorot,
                weight_norm=config.model.weight_norm,
            )

            if config.model.signature_gat.bias and config.model.signature_gat.concat:
                bias = torch.Tensor(size=(heads * out_channels,)).to(
                    config.general.device
                )
                bias.requires_grad = True
                self.add_weight(
                    bias,
                    "GATConv.{}.bias".format(l),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )
            elif (
                config.model.signature_gat.bias
                and not config.model.signature_gat.concat
            ):
                bias = torch.Tensor(size=(out_channels,)).to(config.general.device)
                bias.requires_grad = True
                self.add_weight(
                    bias,
                    "GATConv.{}.bias".format(l),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )

            self.edgeConvs.append(
                GatConv(
                    in_channels,
                    out_channels,
                    edge_dim,
                    heads=heads,
                    concat=config.model.signature_gat.concat,
                    negative_slope=config.model.signature_gat.negative_slope,
                    dropout=config.model.signature_gat.dropout,
                    bias=config.model.signature_gat.bias,
                )
            )

    def prepare_param_idx(self, layer_id=0):
        full_name_idx = {n: i for i, n in enumerate(self.weight_names)}
        gat_layer_param_indx = [
            i for i, k in enumerate(self.weight_names) if "{}".format(layer_id) in k
        ]
        param_names = [self.weight_names[gi] for gi in gat_layer_param_indx]
        param_name_to_idx = {k: full_name_idx[k] for v, k in enumerate(param_names)}
        return param_name_to_idx

    def forward(self, batch):
        data = batch.world_graphs
        param_name_to_idx = {k: v for v, k in enumerate(self.weight_names)}
        assert data.x.size(0) == data.edge_indicator.size(0)
        # extract node embeddings
        # data.edge_indicator contains 0's for all nodes and value > 0 for each unique relations
        x = F.embedding(
            data.edge_indicator,
            get_param(self.weights, param_name_to_idx, "common_emb"),
        )
        # edge attribute is None because we are not learning edge types here
        edge_attr = None
        if data.edge_index.dim() != 2:
            import ipdb

            ipdb.set_trace()
        for nr in range(self.config.model.signature_gat.num_layers - 1):
            param_name_dict = self.prepare_param_idx(nr)
            x = F.dropout(
                x, p=self.config.model.signature_gat.dropout, training=self.training
            )
            x = self.edgeConvs[nr](
                x, data.edge_index, edge_attr, self.weights, param_name_dict
            )
            x = F.elu(x)
        x = F.dropout(
            x, p=self.config.model.signature_gat.dropout, training=self.training
        )
        param_name_dict = self.prepare_param_idx(
            self.config.model.signature_gat.num_layers - 1
        )
        x = self.edgeConvs[self.config.model.signature_gat.num_layers - 1](
            x, data.edge_index, edge_attr, self.weights, param_name_dict
        )
        # restore x into B x num_node x dim
        chunks = torch.split(x, batch.num_edge_nodes, dim=0)
        batches = [p.unsqueeze(0) for p in chunks]
        # we only have one batch for world graph
        batch = batches[0][0]
        # sum over edge type nodes
        num_class = self.config.model.num_classes
        edge_emb = torch.zeros((num_class, batch.size(-1)))
        edge_emb = edge_emb.to(self.config.general.device)
        for ei_t in data.edge_indicator.unique():
            ei = ei_t.item()
            if ei == 0:
                # node of type "node", skip
                continue
            # node of type "edge", take
            # we subtract 1 here to re-align the vectors (L399 of data.py)
            edge_emb[ei - 1] = batch[data.edge_indicator == ei].mean(dim=0)
        return edge_emb, batch


class GatedNodeGatEncoder(Net):
    """
    Encoder which uses `GatedGatConv` and works on graphs without considering the edge
    """

    def __init__(self, config, shared_embeddings=None):
        super(GatedNodeGatEncoder, self).__init__(config)

        # flag to enable one-hot embedding if needed
        self.graph_mode = True
        self.one_hot = self.config.model.signature_gat.emb_type == "one-hot"
        self.edgeConvs = []

        # common node & relation embedding
        # we keep one node embedding for all nodes, and individual relation embedding for relation nodes
        emb = torch.Tensor(
            size=(config.model.num_classes + 1, config.model.relation_embedding_dim)
        ).to(config.general.device)
        # rel_emb = torch.Tensor(size=(1, config.model.relation_embedding_dim)).to(config.general.device)
        emb.requires_grad = True  # config.model.signature_gat.learn_node_and_rel_emb
        torch.nn.init.xavier_normal_(emb)
        self.add_weight(emb, "common_emb", weight_norm=config.model.weight_norm)

        ## Add params
        for l in range(config.model.signature_gat.num_layers):
            in_channels = config.model.relation_embedding_dim
            out_channels = config.model.relation_embedding_dim
            heads = config.model.signature_gat.num_heads
            edge_dim = config.model.relation_embedding_dim

            weight = torch.Tensor(size=(in_channels, heads * out_channels)).to(
                config.general.device
            )
            weight.requires_grad = True
            self.add_weight(
                weight,
                "GatedGATConv.{}.weight".format(l),
                initializer=glorot,
                weight_norm=config.model.weight_norm,
            )

            att = torch.Tensor(size=(1, heads, 2 * out_channels)).to(
                config.general.device
            )
            att.requires_grad = True
            self.add_weight(
                att,
                "GatedGATConv.{}.att".format(l),
                initializer=glorot,
                weight_norm=config.model.weight_norm,
            )

            if l == 0:
                # only add the gru weights once
                gru_weight_ih = torch.Tensor(size=(out_channels, 3 * out_channels)).to(
                    config.general.device
                )
                gru_weight_ih.requires_grad = True
                self.add_weight(
                    gru_weight_ih,
                    "GatedGATConv.{}.gru_w_ih".format("_all_"),
                    weight_norm=config.model.weight_norm,
                )

                gru_weight_hh = torch.Tensor(size=(out_channels, 3 * out_channels)).to(
                    config.general.device
                )
                gru_weight_hh.requires_grad = True
                self.add_weight(
                    gru_weight_hh,
                    "GatedGATConv.{}.gru_w_hh".format("_all_"),
                    weight_norm=config.model.weight_norm,
                )

                gru_bias_ih = torch.Tensor(size=(3 * out_channels,)).to(
                    config.general.device
                )
                gru_bias_ih.requires_grad = True
                self.add_weight(
                    gru_bias_ih,
                    "GatedGATConv.{}.gru_b_ih".format("_all_"),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )

                gru_bias_hh = torch.Tensor(size=(3 * out_channels,)).to(
                    config.general.device
                )
                gru_bias_hh.requires_grad = True
                self.add_weight(
                    gru_bias_hh,
                    "GatedGATConv.{}.gru_b_hh".format("_all_"),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )

            if config.model.signature_gat.bias and config.model.signature_gat.concat:
                bias = torch.Tensor(size=(heads * out_channels,)).to(
                    config.general.device
                )
                bias.requires_grad = True
                self.add_weight(
                    bias,
                    "GatedGATConv.{}.bias".format(l),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )
            elif (
                config.model.signature_gat.bias
                and not config.model.signature_gat.concat
            ):
                bias = torch.Tensor(size=(out_channels,)).to(config.general.device)
                bias.requires_grad = True
                self.add_weight(
                    bias,
                    "GatedGATConv.{}.bias".format(l),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )

            self.edgeConvs.append(
                GatedGatConv(
                    in_channels,
                    out_channels,
                    edge_dim,
                    heads=heads,
                    concat=config.model.signature_gat.concat,
                    negative_slope=config.model.signature_gat.negative_slope,
                    dropout=config.model.signature_gat.dropout,
                    bias=config.model.signature_gat.bias,
                )
            )

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

    def forward(self, batch):
        data = batch.world_graphs
        param_name_to_idx = {k: v for v, k in enumerate(self.weight_names)}
        assert data.x.size(0) == data.edge_indicator.size(0)
        # extract node embeddings
        # data.edge_indicator contains 0's for all nodes and value > 0 for each unique relations
        x = F.embedding(
            data.edge_indicator,
            get_param(self.weights, param_name_to_idx, "common_emb"),
        )
        # edge attribute is None because we are not learning edge types here
        edge_attr = None
        if data.edge_index.dim() != 2:
            import ipdb

            ipdb.set_trace()
        for nr in range(self.config.model.signature_gat.num_layers - 1):
            param_name_dict = self.prepare_param_idx(nr)
            x = F.dropout(
                x, p=self.config.model.signature_gat.dropout, training=self.training
            )
            x = self.edgeConvs[nr](
                x, data.edge_index, edge_attr, self.weights, param_name_dict
            )
            x = F.elu(x)
        x = F.dropout(
            x, p=self.config.model.signature_gat.dropout, training=self.training
        )
        param_name_dict = self.prepare_param_idx(
            self.config.model.signature_gat.num_layers - 1
        )
        if self.config.model.signature_gat.num_layers > 0:
            x = self.edgeConvs[self.config.model.signature_gat.num_layers - 1](
                x, data.edge_index, edge_attr, self.weights, param_name_dict
            )
        # restore x into B x num_node x dim
        chunks = torch.split(x, batch.num_edge_nodes, dim=0)
        batches = [p.unsqueeze(0) for p in chunks]
        # we only have one batch for world graph
        batch = batches[0][0]
        # sum over edge type nodes
        num_class = self.config.model.num_classes
        edge_emb = torch.zeros((num_class, batch.size(-1)))
        edge_emb = edge_emb.to(self.config.general.device)
        for ei_t in data.edge_indicator.unique():
            ei = ei_t.item()
            if ei == 0:
                # node of type "node", skip
                continue
            # node of type "edge", take
            # we subtract 1 here to re-align the vectors (L399 of data.py)
            edge_emb[ei - 1] = batch[data.edge_indicator == ei].mean(dim=0)
        return edge_emb, batch
