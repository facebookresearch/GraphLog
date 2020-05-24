"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
"""Base class for all the models (with batteries)"""
import importlib
import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from codes.logbook.filesystem_logger import write_message_logs
from codes.utils.checkpointable import Checkpointable
from codes.utils.util import get_param_id, get_param
from codes.model.inits import *


class BaseModel(nn.Module, Checkpointable):
    """Base class for all models"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = "base_model"
        self.description = (
            "This is the base class for all the models. "
            "All the other models should extend this class. "
            "It is not to be used directly."
        )
        self.criteria = nn.CrossEntropyLoss()
        self.epsilon = 1e-6
        self.weights = []
        self.weight_names = []
        self.is_signature = False
        self.model_config_key = ""

    def loss(self, outputs, labels):
        """Method to compute the loss"""
        return self.criteria(outputs, labels)

    def track_loss(self, outputs, labels):
        """There are two different functions related to loss as we might be interested in
         tracking one loss and optimising another"""
        return self.loss(outputs, labels)

    def get_param_id(self, param_name_dict, partial_name):
        match = [k for k, v in param_name_dict.items() if partial_name in k]
        if len(match) != 1:
            print(match)
            raise AssertionError("more than one match")
        name = match[0]
        if name not in param_name_dict:
            raise AssertionError("arg not found")
        return param_name_dict[name]

    def add_weight(
        self,
        tensor,
        tensor_name,
        initializer=None,
        skip_init=False,
        weight_norm=False,
        eps=0.00001,
    ):
        """
        Add weight to the list of weights
        if weight_norm, then add a `g` and `v` for each weight.
        modify names such that "{layername}.{layer}.{param_name}_{g/v}"
        :param tensor:
        :param tensor_name:
        :return:
        """
        if not skip_init:
            if not initializer:
                torch.nn.init.xavier_uniform_(tensor, gain=1.414)
            else:
                if type(initializer) == tuple:
                    initializer[0](tensor, initializer[1])
                else:
                    initializer(tensor)
        if not weight_norm:
            self.weights.append(tensor)
            self.weight_names.append(tensor_name)
        else:
            g = torch.norm(tensor) + eps
            v = tensor / g.expand_as(tensor)
            g = torch.tensor(g)
            g.requires_grad = True
            v = torch.tensor(v)
            v.requires_grad = True
            g_name = tensor_name + "_g"
            v_name = tensor_name + "_v"
            self.weights.append(g)
            self.weight_names.append(g_name)
            self.weights.append(v)
            self.weight_names.append(v_name)

    def set_requires_grad(self, value):
        for weight in self.weights:
            weight.requires_grad = value

    def add_classify_weights(self, in_class_dim=1):
        """add classify weights
        """
        out_class_dim = self.config.model.classify_hidden
        for layer in range(self.config.model.classify_layers):
            if layer + 1 == self.config.model.classify_layers:
                out_class_dim = self.config.model.num_classes
            cw = torch.Tensor(size=(in_class_dim, out_class_dim)).to(
                self.config.general.device
            )
            cw.requires_grad = True
            self.add_weight(cw, "classify_{}.weight".format(layer))
            cwb = torch.Tensor(size=[out_class_dim]).to(self.config.general.device)
            cwb.requires_grad = True
            self.add_weight(
                cwb, "classify_{}.bias".format(layer), initializer=(uniform, 1)
            )
            in_class_dim = out_class_dim

    def pyg_classify(self, nodes, query_edge, params=None, param_name_dict=None):
        """
        Run classification using MLP
        :param nodes:
        :param query_edge:
        :param params:
        :param param_name_dict:
        :return:
        """
        query_emb = []
        for i in range(len(nodes)):
            query = (
                query_edge[i].unsqueeze(0).unsqueeze(2).repeat(1, 1, nodes[i].size(2))
            )  # B x num_q x dim
            query_emb.append(torch.gather(nodes[i], 1, query))
        query_emb = torch.cat(query_emb, dim=0)
        query = query_emb.view(query_emb.size(0), -1)  # B x (num_q x dim)
        # pool the nodes
        # mean pooling
        node_avg = torch.cat(
            [torch.mean(nodes[i], 1) for i in range(len(nodes))], dim=0
        )  # B x dim
        # concat the query
        edges = torch.cat((node_avg, query), -1)  # B x (dim + dim x num_q)
        for layer in range(self.config.model.classify_layers):
            edges = F.linear(
                edges,
                weight=get_param(
                    params,
                    param_name_dict,
                    "classify_{}.weight".format(layer),
                    ignore_classify=False,
                ).t(),
                bias=get_param(
                    params,
                    param_name_dict,
                    "classify_{}.bias".format(layer),
                    ignore_classify=False,
                ),
            )
            if layer < self.config.model.classify_layers - 1:
                edges = F.relu(edges)
        return edges

    def predict(self, outputs):
        """
        Given a logit, calculate the predictions
        :param outputs:
        :return:
        """
        topv, topi = outputs.topk(1)
        predictions = topi.squeeze(1)
        confidence = torch.exp(F.log_softmax(outputs, dim=1))
        return predictions, confidence

    def accuracy(self, predictions, labels):
        """
        Calculate the accuracy
        :param outputs:
        :param labels:
        :return:
        """
        return torch.sum(predictions == labels).float() / labels.size(0)

    def save(
        self,
        epoch: int,
        optimizers: Optional[List[torch.optim.Optimizer]],
        is_best_model: bool = False,
    ) -> None:
        """Method to persist the model.
        Note this method is not well tested"""

        model_config = self.config.model
        if len(self.model_config_key) == 0:
            model_name = model_config.name
        else:
            model_name = model_config[self.model_config_key]["name"]

        # Updating the information about the epoch
        ## Check if the epoch_state is already saved on the file system
        if not os.path.exists(model_config.save_dir):
            os.makedirs(model_config.save_dir)
        epoch_state_path = os.path.join(model_config.save_dir, "epoch_state.tar")

        if os.path.exists(epoch_state_path):
            epoch_state = torch.load(epoch_state_path)
        else:
            epoch_state = {"best": epoch}
        epoch_state["current"] = epoch
        if is_best_model:
            epoch_state["best"] = epoch
        torch.save(epoch_state, epoch_state_path)

        state = {
            "metadata": {"epoch": epoch, "is_best_model": False,},
            "model": {"weights": self.weights, "weight_names": self.weight_names},
            "optimizers": [
                {"state_dict": optimizer.state_dict()} for optimizer in optimizers
            ],
            "random_state": {
                "np": np.random.get_state(),
                "python": random.getstate(),
                "pytorch": torch.get_rng_state(),
            },
        }
        path = os.path.join(
            model_config.save_dir, "{}_epoch_{}.tar".format(model_name, epoch)
        )
        if is_best_model:
            state["metadata"]["is_best_model"] = True
        torch.save(state, path)
        write_message_logs("saved experiment to path = {}".format(path))

    def load(
        self,
        epoch: int,
        should_load_optimizers: bool = True,
        optimizers=Optional[List[optim.Optimizer]],
        schedulers=Optional[List[optim.lr_scheduler.ReduceLROnPlateau]],
    ) -> None:
        """Public method to load the model"""
        model_config = self.config.model
        model_config = self.config.model
        if len(self.model_config_key) == 0:
            model_name = model_config.name
        else:
            model_name = model_config[self.model_config_key]["name"]
        path = os.path.join(
            model_config.save_dir, "{}_epoch_{}.tar".format(model_name, epoch)
        )
        if not os.path.exists(path):
            raise FileNotFoundError("Loading path {} not found!".format(path))
        write_message_logs("Loading model from path {}".format(path))
        if str(self.config.general.device) == "cuda":
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        load_random_state(checkpoint["random_state"])
        self.weights = checkpoint["model"]["weights"]
        self.weight_names = checkpoint["model"]["weight_names"]

        if should_load_optimizers:
            if optimizers is None:
                optimizers = self.get_optimizers()
            for optim_index, optimizer in enumerate(optimizers):
                optimizer.load_state_dict(
                    checkpoint["optimizers"][optim_index]["state_dict"]
                )

            key = "schedulers"
            if key in checkpoint:
                for scheduler_index, scheduler in enumerate(schedulers):
                    scheduler.load_state_dict(
                        checkpoint[key][scheduler_index]["state_dict"]
                    )
        return optimizers, schedulers

    def _load_model_params(self, state_dict):
        """Method to load the model params"""
        self.load_state_dict(state_dict)

    def get_model_params(self):
        """Method to get the model params"""
        model_parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        if params == 0:
            # get params from weights
            model_parameters = self.weights
            params = sum([np.prod(p.size()) for p in model_parameters])
        write_message_logs("Total number of params = " + str(params))
        return model_parameters

    def get_optimizers_and_schedulers(self):
        """Method to return the list of optimizers and schedulers for the model"""
        optimizers = self.get_optimizers()
        if optimizers:
            optimizers, schedulers = BaseModel.register_optimizers_to_schedulers(
                self.config, optimizers, is_signature=self.is_signature
            )
            return optimizers, schedulers
        return None

    def get_optimizers(self):
        """Method to return the list of optimizers for the model"""
        optimizers = []
        model_params = self.get_model_params()
        if model_params:
            optimizers.append(
                BaseModel.register_params_to_optimizer(
                    self.config, model_params, is_signature=self.is_signature
                )
            )
            return optimizers
        return None

    @staticmethod
    def register_params_to_optimizer(config, model_params, is_signature=False):
        """Method to map params to an optimizer"""
        optim_config = config.model.optim
        if is_signature:
            optim_config = config.model.signature_optim
        optimizer_cls = getattr(
            importlib.import_module("torch.optim"), optim_config.name
        )
        optim_name = optim_config.name.lower()
        if optim_name == "adam":
            return optimizer_cls(
                model_params,
                lr=optim_config.learning_rate,
                weight_decay=optim_config.weight_decay,
                eps=optim_config.eps,
            )
        if optim_name == "sgd":
            return optimizer_cls(
                model_params,
                lr=optim_config.learning_rate,
                weight_decay=optim_config.weight_decay,
            )
        return optimizer_cls(
            model_params,
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
            eps=optim_config.eps,
        )

    @staticmethod
    def register_optimizers_to_schedulers(config, optimizers, is_signature=False):
        """Method to map optimzers to schedulers"""
        optimizer_config = config.model.optim
        if is_signature:
            optimizer_config = config.model.signature_optim
        if optimizer_config.scheduler_type == "exp":
            schedulers = list(
                map(
                    lambda optimizer: optim.lr_scheduler.ExponentialLR(
                        optimizer=optimizer,
                        gamma=config.model.optimizer.scheduler_gamma,
                    ),
                    optimizers,
                )
            )
        elif optimizer_config.scheduler_type == "plateau":
            schedulers = list(
                map(
                    lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=optimizer,
                        mode="min",
                        patience=config.model.optim.scheduler_patience,
                        factor=config.model.optim.scheduler_gamma,
                        verbose=True,
                    ),
                    optimizers,
                )
            )
        elif optimizer_config.scheduler_type == "1cycle":
            schedulers = list(
                map(
                    lambda optimizer: optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=config.model.optim.scheduler_max_lr,
                        steps_per_epoch=config.model.optim.scheduler_steps_per_epoch,
                        epochs=config.model.num_epochs,
                    ),
                    optimizers,
                )
            )

        return optimizers, schedulers

    def reset_optim_lr(self, optimizer):
        for g in optimizer.param_groups:
            g["lr"] = self.config.model.optim.learning_rate
        return optimizer

    def forward(self, data):  # pylint: disable=W0221,W0613
        """Forward pass of the network"""
        return None

    def get_param_count(self):
        """Count the number of params"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def freeze_weights(self):
        """Freeze the model"""
        for param in self.weights:
            param.requires_grad = False

    def re_init_relation_weight(self):
        """Models should implement this def to re-initialize the embedding weights on every epoch
        """
        pass

    def __str__(self):
        """Return string description of the model"""
        return self.description


def load_random_state(random_state):
    """Method to load the random state"""
    np.random.set_state(random_state["np"])
    random.setstate(random_state["python"])
    torch.set_rng_state(random_state["pytorch"])
