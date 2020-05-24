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
from codes.model.gat.edge_gat import GatEncoder, GatedGatEncoder
from codes.model.gat.sig_edge_gat import NodeGatEncoder, GatedNodeGatEncoder
from codes.model.base_model import BaseModel as Net
from codes.model.inits import *
from codes.utils.util import _import_module, get_param
import copy


class Param(Net):
    def __init__(self, config):
        super(Param, self).__init__(config)
        edge_e = torch.randn(
            (self.config.model.num_classes, self.config.model.relation_embedding_dim)
        )
        edge_e = edge_e.to(self.config.general.device)
        edge_e.requires_grad = True
        self.add_weight(edge_e, "learned_param", weight_norm=config.model.weight_norm)

    def forward(self, batch):
        param_name_to_idx = {k: v for v, k in enumerate(self.weight_names)}
        edge_e = get_param(self.weights, param_name_to_idx, "learned_param")
        return edge_e, None


class ParamLinear(Net):
    def __init__(self, config):
        super(ParamLinear, self).__init__(config)
        edge_p = torch.randn((self.config.model.num_classes, 500))
        edge_p = edge_p.to(self.config.general.device)
        edge_p.requires_grad = True
        self.add_weight(edge_p, "learned_param_1", weight_norm=config.model.weight_norm)
        edge_e = torch.randn((500, self.config.model.relation_embedding_dim))
        edge_e = edge_e.to(self.config.general.device)
        edge_e.requires_grad = True
        self.add_weight(edge_e, "learned_param_2", weight_norm=config.model.weight_norm)

    def forward(self, batch):
        param_name_to_idx = {k: v for v, k in enumerate(self.weight_names)}

        edge_e = F.linear(
            get_param(self.weights, param_name_to_idx, "learned_param_1"),
            weight=get_param(self.weights, param_name_to_idx, "learned_param_2").t(),
        )
        return edge_e, None


class ParamNonLinear(Net):
    def __init__(self, config):
        super(ParamNonLinear, self).__init__(config)
        edge_p = torch.randn((self.config.model.num_classes, 500))
        edge_p = edge_p.to(self.config.general.device)
        edge_p.requires_grad = True
        self.add_weight(edge_p, "learned_param_1", weight_norm=config.model.weight_norm)
        edge_e = torch.randn((500, 200))
        edge_e = edge_e.to(self.config.general.device)
        edge_e.requires_grad = True
        self.add_weight(edge_e, "learned_param_2", weight_norm=config.model.weight_norm)
        edge_q = torch.randn((200, self.config.model.relation_embedding_dim))
        edge_q = edge_q.to(self.config.general.device)
        edge_q.requires_grad = True
        self.add_weight(edge_q, "learned_param_3", weight_norm=config.model.weight_norm)

    def forward(self, batch):
        param_name_to_idx = {k: v for v, k in enumerate(self.weight_names)}

        edge_e = F.linear(
            F.relu(
                F.linear(
                    get_param(self.weights, param_name_to_idx, "learned_param_1"),
                    weight=get_param(
                        self.weights, param_name_to_idx, "learned_param_2"
                    ).t(),
                )
            ),
            weight=get_param(self.weights, param_name_to_idx, "learned_param_3").t(),
        )
        return edge_e, None


class RepresentationFn(Net):
    """RepresentationFn
    (previously known as signature function)
    """

    def __init__(self, config):
        super(RepresentationFn, self).__init__(config)
        self.model = _import_module(config.model.representation_fn_path)(config)
        if not config.model.representation_learn_param:
            self.model.set_requires_grad(False)
        self.weights = self.model.weights
        self.weight_names = self.model.weight_names

    def forward(self, batch):
        return self.model(batch)

    def set_weights(self, weights):
        """ set the correct pointer
        """
        self.model.weights = weights
        self.weights = self.model.weights

    def freeze_weights(self):
        self.model.freeze_weights()

    def to(self, device):
        self.model.weights = [
            w.to(device).detach().requires_grad_(True) for w in self.model.weights
        ]
        self.weights = self.model.weights

    def re_init_relation_weight(self):
        self.model.re_init_relation_weight()


class CompositionFn(Net):
    """CompositionFn
    """

    def __init__(self, config):
        super(CompositionFn, self).__init__(config)
        self.model = _import_module(config.model.composition_fn_path)(config)
        if not config.model.composition_learn_param:
            self.model.set_requires_grad(False)
        try:
            rel_emb_index = self.model.weight_names.index("relation_embedding")
            if rel_emb_index > -1:
                del self.model.weights[rel_emb_index]
                del self.model.weight_names[rel_emb_index]
        except:
            print("relation_embedding not in model")
        self.weights = self.model.weights
        self.weight_names = self.model.weight_names

    def forward(self, batch, rel_emb):
        if type(rel_emb) == tuple:
            rel_emb = rel_emb[0]
        return self.model(batch, rel_emb)

    def set_weights(self, weights):
        """ set the correct pointer
        """
        self.model.weights = weights
        self.weights = self.model.weights

    def freeze_weights(self):
        self.model.freeze_weights()

    def to(self, device):
        self.model.weights = [
            w.to(device).detach().requires_grad_(True) for w in self.model.weights
        ]
        self.weights = self.model.weights


class LearnedSignature(Net):
    """
    Edge GAT with Signature function
    """

    def __init__(self, config, shared_embeddings=None):
        super(LearnedSignature, self).__init__(config)
        # self.signature_fn = NodeGatEncoder(config)
        self.signature_fn = GatedNodeGatEncoder(config)
        self.graph_fn = GatedGatEncoder(config)
        # self.graph_fn = GatEncoder(config)
        # remove rel embedding from model
        rel_emb_index = self.graph_fn.weight_names.index("relation_embedding")
        del self.graph_fn.weights[rel_emb_index]
        del self.graph_fn.weight_names[rel_emb_index]
        self.weights = self.signature_fn.weights + self.graph_fn.weights
        self.weight_names = self.signature_fn.weight_names + self.graph_fn.weight_names

    def forward(self, batch):
        edge_e, _ = self.signature_fn(batch)
        # edge_e = None
        return self.graph_fn(batch, edge_e)


class FixedSignature(Net):
    """
    Edge GAT with Signature function
    """

    def __init__(self, config, shared_embeddings=None):
        super(FixedSignature, self).__init__(config)
        self.signature_fn = NodeGatEncoder(config)
        self.graph_fn = GatedGatEncoder(config)
        # remove rel embedding from model
        rel_emb_index = self.graph_fn.weight_names.index("relation_embedding")
        del self.graph_fn.weights[rel_emb_index]
        del self.graph_fn.weight_names[rel_emb_index]
        self.weights = self.graph_fn.weights
        self.weight_names = self.graph_fn.weight_names

    def forward(self, batch):
        with torch.no_grad():
            edge_e, _ = self.signature_fn(batch)
        # edge_e = None
        # import ipdb; ipdb.set_trace()
        return self.graph_fn(batch, edge_e)


class FixedParamSignature(Net):
    """
    Edge GAT with Signature function
    """

    def __init__(self, config, shared_embeddings=None):
        super(FixedParamSignature, self).__init__(config)
        self.graph_fn = GatedGatEncoder(config)
        # remove rel embedding from model
        rel_emb_index = self.graph_fn.weight_names.index("relation_embedding")
        del self.graph_fn.weights[rel_emb_index]
        del self.graph_fn.weight_names[rel_emb_index]
        self.weights = self.graph_fn.weights
        self.weight_names = self.graph_fn.weight_names
        self.edge_e = torch.randn(
            (self.config.model.num_classes, self.config.model.relation_embedding_dim)
        )
        self.edge_e = self.edge_e.to(self.config.general.device)
        self.edge_e.requires_grad = False

    def forward(self, batch):
        # edge_e = None
        # import ipdb; ipdb.set_trace()
        return self.graph_fn(batch, self.edge_e)


class LearnedParamSignature(Net):
    """
    Edge GAT with Signature function
    """

    def __init__(self, config, shared_embeddings=None):
        super(LearnedParamSignature, self).__init__(config)
        self.graph_fn = GatedGatEncoder(config)
        # remove rel embedding from model
        rel_emb_index = self.graph_fn.weight_names.index("relation_embedding")
        del self.graph_fn.weights[rel_emb_index]
        del self.graph_fn.weight_names[rel_emb_index]
        edge_e = torch.randn(
            (self.config.model.num_classes, self.config.model.relation_embedding_dim)
        )
        edge_e = edge_e.to(self.config.general.device)
        edge_e.requires_grad = True
        self.add_weight(edge_e, "learned_param")
        self.weights = self.graph_fn.weights + self.weights
        self.weight_names = self.graph_fn.weight_names + self.weight_names

    def forward(self, batch):
        # edge_e = None
        param_name_to_idx = {k: v for v, k in enumerate(self.weight_names)}
        edge_e = self.weights[self.get_param_id(param_name_to_idx, "learned_param")]
        return self.graph_fn(batch, edge_e)


class PreTrainRepresentation(Net):
    """
    Pretraining objective
    """

    def __init__(self, config, shared_embeddings=None):
        super(PreTrainRepresentation, self).__init__(config)
        # add a simple one layer mlp classifier
        in_class_dim = config.model.relation_embedding_dim
        out_class_dim = config.model.num_classes
        cw = torch.Tensor(size=(in_class_dim, out_class_dim)).to(config.general.device)
        cw.requires_grad = True
        self.add_weight(cw, "node_classify_0.weight")
        cwb = torch.Tensor(size=[out_class_dim]).to(config.general.device)
        cwb.requires_grad = True
        self.add_weight(cwb, "node_classify_0.bias", initializer=zeros)
        in_class_dim = config.model.relation_embedding_dim * 2
        out_class_dim = 1
        cw = torch.Tensor(size=(in_class_dim, out_class_dim)).to(config.general.device)
        cw.requires_grad = True
        self.add_weight(cw, "edge_classify_0.weight")
        cwb = torch.Tensor(size=[out_class_dim]).to(config.general.device)
        cwb.requires_grad = True
        self.add_weight(cwb, "edge_classify_0.bias", initializer=zeros)

    def classify(self, edges):
        edges = F.linear(
            edges,
            weight=self.weights[self.weight_names.index("classify_0.weight")].t(),
            bias=self.weights[self.weight_names.index("classify_0.bias")],
        )
        return F.log_softmax(edges, dim=-1)

    def forward(self, batch, all_edges):
        edge_indicator = batch.world_graphs.edge_indicator
        # mask out 0's - which are the node ids
        zero_idx = edge_indicator == 0
        edge_indicator = edge_indicator[~zero_idx]
        edge_indicator = edge_indicator - 1  # correct for the node id
        all_edges = all_edges[~zero_idx]
        num_edges = all_edges.size(0)
        # randomly choose batchsize elements
        edge_ids = torch.randperm(num_edges)[: self.config.general.batch_size].to(
            edge_indicator.device
        )
        # edge_e = None
        return self.classify(all_edges[edge_ids]), edge_indicator[edge_ids]
