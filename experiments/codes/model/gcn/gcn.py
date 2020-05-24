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
from codes.model.gcn.layers import *
from codes.utils.util import prepare_param_idx, get_param, get_param_id


class RepresentationGCNEncoder(Net):
    """Composition function which uses GCN
    Returns a relation embedding after running GCN on the dual world graph
    """

    def __init__(self, config):
        super(RepresentationGCNEncoder, self).__init__(config)
        self.name = "CompositionGCNConv"
        self.gcn_layers = []
        # common node & relation embedding
        # we keep one node embedding for all nodes, and individual relation embedding for relation nodes
        emb = torch.Tensor(
            size=(config.model.num_classes + 1, config.model.relation_embedding_dim)
        ).to(config.general.device)
        emb.requires_grad = True  # config.model.signature_gat.learn_node_and_rel_emb
        torch.nn.init.xavier_normal_(emb)
        self.add_weight(emb, "common_emb")

        for l in range(config.model.gcn.num_layers):
            in_channels = config.model.relation_embedding_dim
            out_channels = config.model.relation_embedding_dim
            num_bases = config.model.relation_embedding_dim

            weight = torch.Tensor(size=(in_channels, out_channels)).to(
                config.general.device
            )
            weight.requires_grad = True
            self.add_weight(
                weight,
                "{}.{}.weight".format(self.name, l),
                initializer=glorot,
                weight_norm=config.model.weight_norm,
            )

            if config.model.gcn.bias:
                bias = torch.Tensor(size=(out_channels,)).to(config.general.device)
                bias.requires_grad = True
                self.add_weight(
                    bias,
                    "{}.{}.bias".format(self.name, l),
                    initializer=(uniform, 1),
                    weight_norm=config.model.weight_norm,
                )

            self.gcn_layers.append(
                GCNConv(
                    in_channels,
                    out_channels,
                    config.model.gcn.improved,
                    config.model.gcn.cached,
                    config.model.gcn.bias,
                    config.model.gcn.normalize,
                )
            )

    def re_init_relation_weight(self):
        common_emb_pos = self.weight_names.index("common_emb")
        torch.nn.init.xavier_normal_(self.weights[common_emb_pos][1:])

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
        for nr in range(self.config.model.gcn.num_layers - 1):
            param_name_dict = prepare_param_idx(self.weight_names, nr)
            x = self.gcn_layers[nr](x, data.edge_index, self.weights, param_name_dict)
            x = F.relu(x)
            x = F.dropout(x, p=self.config.model.gcn.dropout, training=self.training)
        param_name_dict = prepare_param_idx(
            self.weight_names, self.config.model.gcn.num_layers - 1
        )
        x = self.gcn_layers[self.config.model.gcn.num_layers - 1](
            x, data.edge_index, self.weights, param_name_dict
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
