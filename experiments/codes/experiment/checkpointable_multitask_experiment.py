"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
"""Class to run the experiments"""
# from time import time

import numpy as np
import torch
import torch.nn as nn
import copy
from codes.model.models import RepresentationFn, CompositionFn
import random
import pickle as pkl
import os

from codes.utils.checkpointable import Checkpointable
from codes.model.base_model import BaseModel as Net
from typing import Iterable, Optional, Tuple
from graphlog import GraphLog


class MultitaskExperiment(Checkpointable):
    """Experiment Class for Supervised, Multitask and Continual Learning experiments in GraphLog"""

    def __init__(self, config, model, data, logbook):
        super().__init__()
        self.data = data
        self.config = config
        self.logbook = logbook
        self.support_modes = self.config.model.modes
        self.device = self.config.general.device
        self.is_signature = False
        if self.config.logger.watch_model:
            self.logbook.watch_model(model=self.model)
        self.gl = GraphLog()
        self.dataloaders = self.load_data()
        (self.composition_fn, self.representation_fn,) = self.bootstrap_model()
        self.register_optim_sched()
        self._mode = None
        self.train_step = 0
        self.epoch = 0

    def bootstrap_model(self) -> [nn.Module, nn.Module, torch.optim.Optimizer]:
        composition_fn = CompositionFn(self.config)
        representation_fn = RepresentationFn(self.config)
        return composition_fn, representation_fn

    def register_optim_sched(
        self, skip_composition_registry=False, skip_representation_registry=False
    ):
        # NOTE: be careful of assigning the correct weights to the optimizer.
        # either assign `composition_fn.weights` or `composition_fn.model.weights`
        if skip_composition_registry:
            optimizer = Net.register_params_to_optimizer(
                self.config,
                self.representation_fn.model.weights,
                is_signature=self.is_signature,
            )
        elif skip_representation_registry:
            optimizer = Net.register_params_to_optimizer(
                self.config,
                self.composition_fn.model.weights,
                is_signature=self.is_signature,
            )
        else:
            optimizer = Net.register_params_to_optimizer(
                self.config,
                self.composition_fn.model.weights
                + self.representation_fn.model.weights,
                is_signature=self.is_signature,
            )
        self.optimizers, self.schedulers = Net.register_optimizers_to_schedulers(
            self.config, [optimizer]
        )

    def load_data(self):
        dataloaders = {}
        modes = ["train", "valid", "test"]
        for mi, mode in enumerate(modes):
            dataloaders[mode] = {}
            for graph_world in self.data[mi]:
                rule_world = graph_world.world_id
                dataloaders[mode][rule_world] = {}
                for data_mode in modes:
                    dataloaders[mode][rule_world][
                        data_mode
                    ] = self.gl.get_dataloader_by_mode(graph_world, data_mode)
        return dataloaders

    def periodic_save(self, epoch: int):
        if self.config.model.persist_frequency > 0:
            if epoch % self.config.model.persist_frequency == 0:
                self.save_model(epoch=epoch)

    def save_model(self, epoch: Optional[int] = 0) -> None:
        self.save(
            self.config.model.save_dir,
            epoch,
            self.composition_fn,
            self.representation_fn,
            self.optimizers,
        )

    def load_model(self, epoch: Optional[int] = None) -> None:
        (
            self.composition_fn,
            self.representation_fn,
            self.optimizers,
            self.epoch,
        ) = self.load(
            self.config.model.save_dir,
            epoch,
            self.composition_fn,
            self.representation_fn,
            self.optimizers,
        )

    def load_only_composition(self, epoch: Optional[int] = None) -> None:
        """Load only composition model
            epoch {Optional[int]} -- [description] (default: {None})
        """
        self.composition_fn, _, _, _ = self.load(
            self.config.model.load_dir, epoch, self.composition_fn, None, None
        )

    def load_only_representation(self, epoch: Optional[int] = None) -> None:
        """Load only representation model
            epoch {Optional[int]} -- [description] (default: {None})
        """
        _, self.representation_fn, _, _ = self.load(
            self.config.model.load_dir, epoch, representation_fn=self.representation_fn
        )

    def run_sequential_multitask_training(self):
        """supervised case I: train one model on all the tasks
         """
        if self.config.model.should_load_model:
            self.load_model()
        if self.epoch is None:
            self.epoch = 0
        # the order is very important here. double check while training
        train_world_names = self.config.general.train_rule.split(",")
        full_train_world_names = self.gl.get_dataset_names_by_split()["train"]
        if self.config.model.should_train:
            for train_rule_world in train_world_names:
                task_idx = train_world_names.index(train_rule_world)
                train_rule_world = full_train_world_names[task_idx]
                for epoch in range(self.epoch, self.config.model.num_epochs):
                    self.logbook.write_message_logs(f"Training rule {train_rule_world}")
                    # ipdb.set_trace()
                    self.logbook.write_message_logs(
                        f"Choosing to train the model " f"on {train_rule_world}"
                    )
                    # Train, optimize and test on the same world
                    train_data = self.dataloaders["train"][train_rule_world]
                    self.train(train_data, train_rule_world, epoch, task_idx=task_idx)
                    metrics = self.eval(
                        {train_rule_world: self.dataloaders["train"][train_rule_world]},
                        epoch=epoch,
                        mode="valid",
                        data_mode="train",
                        task_idx=task_idx,
                    )
                    for sched in self.schedulers:
                        sched.step(metrics["loss"])

                # current task performance
                self.eval(
                    {train_rule_world: self.dataloaders["train"][train_rule_world]},
                    epoch=epoch,
                    mode="test",
                    data_mode="train",
                )
                if task_idx > 0:
                    # previous tasks performance
                    self.eval(
                        {
                            task: self.dataloaders["train"][
                                full_train_world_names[task_idx]
                            ]
                            for task in train_world_names[:task_idx]
                        },
                        epoch=epoch,
                        mode="test",
                        data_mode="train_prev",
                    )
                self.periodic_save(task_idx)

    def run_sequential_multitask_unique_composition(self):
        """ Running continual learning experiment when we have unique composition
        function for each world
        """
        if self.config.model.should_load_model:
            self.load_model()
        # the order is very important here. double check while training
        train_world_names = self.config.general.train_rule.split(",")
        full_train_world_names = self.gl.get_dataset_names_by_split()["train"]
        # make all optimizers
        representation_optimizer = Net.register_params_to_optimizer(
            self.config,
            self.representation_fn.model.weights,
            is_signature=self.is_signature,
        )
        self.optimizers = [representation_optimizer, None]
        if self.epoch is None:
            self.epoch = 0
        if self.config.model.should_train:
            for train_rule_world in train_world_names:
                task_idx = train_world_names.index(train_rule_world)
                train_rule_world = full_train_world_names[task_idx]
                # initiate a new composition function
                self.composition_fn = CompositionFn(self.config)
                composition_optimizer = Net.register_params_to_optimizer(
                    self.config,
                    self.composition_fn.model.weights,
                    is_signature=self.is_signature,
                )
                self.optimizers[-1] = composition_optimizer
                for epoch in range(self.epoch, self.config.model.num_epochs):
                    self.logbook.write_message_logs(f"Training rule {train_rule_world}")
                    # ipdb.set_trace()
                    self.logbook.write_message_logs(
                        f"Choosing to train the model " f"on {train_rule_world}"
                    )
                    # Train, optimize and test on the same world
                    train_data = self.dataloaders["train"][train_rule_world]
                    self.train(train_data, train_rule_world, epoch, task_idx=task_idx)
                    metrics = self.eval(
                        {train_rule_world: self.dataloaders["train"][train_rule_world]},
                        epoch=epoch,
                        mode="valid",
                        data_mode="train",
                        task_idx=task_idx,
                    )
                    for sched in self.schedulers:
                        sched.step(metrics["loss"])
                # current task performance
                self.eval(
                    {train_rule_world: self.dataloaders["train"][train_rule_world]},
                    epoch=epoch,
                    mode="test",
                    data_mode="train",
                )
                if task_idx > 0:
                    # previous tasks performance
                    self.eval(
                        {
                            task: self.dataloaders["train"][
                                full_train_world_names[task_idx]
                            ]
                            for task in train_world_names[:task_idx]
                        },
                        epoch=epoch,
                        mode="test",
                        data_mode="train_prev",
                    )
                self.periodic_save(task_idx)

    def run_sequential_multitask_unique_representation(self):
        """ Running continual learning experiment when we have unique representation
        function for each world
        """
        if self.config.model.should_load_model:
            self.load_model()
        # the order is very important here. double check while training
        train_world_names = self.config.general.train_rule.split(",")
        full_train_world_names = self.gl.get_dataset_names_by_split()["train"]
        # make all optimizers
        composition_optimizer = Net.register_params_to_optimizer(
            self.config,
            self.composition_fn.model.weights,
            is_signature=self.is_signature,
        )
        self.optimizers = [composition_optimizer, None]
        if self.epoch is None:
            self.epoch = 0
        if self.config.model.should_train:
            for train_rule_world in train_world_names:
                task_idx = train_world_names.index(train_rule_world)
                train_rule_world = full_train_world_names[task_idx]
                # initiate a new representation function
                self.representation_fn = RepresentationFn(self.config)
                representation_optimizer = Net.register_params_to_optimizer(
                    self.config,
                    self.representation_fn.model.weights,
                    is_signature=self.is_signature,
                )
                self.optimizers[-1] = representation_optimizer
                for epoch in range(self.epoch, self.config.model.num_epochs):
                    self.logbook.write_message_logs(f"Training rule {train_rule_world}")
                    # ipdb.set_trace()
                    self.logbook.write_message_logs(
                        f"Choosing to train the model " f"on {train_rule_world}"
                    )
                    # Train, optimize and test on the same world
                    train_data = self.dataloaders["train"][train_rule_world]
                    self.train(train_data, train_rule_world, epoch, task_idx=task_idx)
                    metrics = self.eval(
                        {train_rule_world: self.dataloaders["train"][train_rule_world]},
                        epoch=epoch,
                        mode="valid",
                        data_mode="train",
                        task_idx=task_idx,
                    )
                    for sched in self.schedulers:
                        sched.step(metrics["loss"])
                # current task performance
                self.eval(
                    {train_rule_world: self.dataloaders["train"][train_rule_world]},
                    epoch=epoch,
                    mode="test",
                    data_mode="train",
                )
                if task_idx > 0:
                    # previous tasks performance
                    self.eval(
                        {
                            task: self.dataloaders["train"][
                                full_train_world_names[task_idx]
                            ]
                            for task in train_world_names[:task_idx]
                        },
                        epoch=epoch,
                        mode="test",
                        data_mode="train_prev",
                    )
                self.periodic_save(task_idx)

    def run_single_task(self, world_mode="train"):
        """
        Only run one task - Supervised setup
        :return:
        """
        if self.epoch is None:
            self.epoch = 0
        train_world_names = list(self.dataloaders[world_mode].keys())
        wn = [w.split("/")[-1] for w in train_world_names]
        wn_i = wn.index(self.config.general.train_rule)
        train_rule_world = train_world_names[wn_i]
        task_idx = train_world_names.index(train_rule_world)
        if self.config.model.should_train:
            for epoch in range(self.epoch, self.config.model.num_epochs):
                self.logbook.write_message_logs(f"Training rule {train_rule_world}")

                # ipdb.set_trace()
                self.logbook.write_message_logs(
                    f"Choosing to train the model " f"on {train_rule_world}"
                )

                train_data = self.dataloaders[world_mode][train_rule_world]
                self.train(train_data, train_rule_world, epoch, task_idx=task_idx)
                self.epoch = epoch
                self.periodic_save(epoch=epoch)
                metrics = self.eval(
                    {train_rule_world: train_data},
                    epoch=epoch,
                    mode="valid",
                    data_mode=world_mode,
                    task_idx=task_idx,
                )
                for sched in self.schedulers:
                    sched.step(metrics["loss"])
                self.eval(
                    {train_rule_world: self.dataloaders[world_mode][train_rule_world]},
                    epoch=epoch,
                    mode="test",
                    data_mode=world_mode,
                    task_idx=task_idx,
                )
                if self.config.logger.watch_model:
                    norms = [w.norm().item() for w in self.model.weights]
                    norm_metric = {
                        wn: norms[wi] for wi, wn in enumerate(self.model.weight_names)
                    }
                    norm_metric["mode"] = "train"
                    norm_metric["minibatch"] = self.train_step
                    self.logbook.write_metric_logs(norm_metric)
                self.periodic_save(epoch)

    def run_multitask_training(self):
        """Multitask learning : Learn a single model on all tasks
         """
        if self.config.model.should_load_model:
            self.load_model()
        if self.epoch is None:
            self.epoch = 0
        train_world_names = list(self.dataloaders["train"].keys())
        if self.config.model.should_train:
            for epoch in range(self.epoch, self.config.model.num_epochs):
                train_rule_world = random.choice(train_world_names)
                task_idx = train_world_names.index(train_rule_world)
                self.logbook.write_message_logs(f"Training rule {train_rule_world}")

                self.logbook.write_message_logs(
                    f"Choosing to train the model " f"on {train_rule_world}"
                )
                # Train, optimize and test on the same world
                train_data = self.dataloaders["train"][train_rule_world]
                self.train(train_data, train_rule_world, epoch, task_idx=task_idx)
                metrics = self.eval(
                    {train_rule_world: self.dataloaders["train"][train_rule_world]},
                    epoch=epoch,
                    mode="valid",
                    data_mode="train",
                    task_idx=task_idx,
                )
                for sched in self.schedulers:
                    sched.step(metrics["loss"])
                self.periodic_save(epoch)

    def run_multitask_training_unique_composition(self):
        """Run multitask training with unique composition functions
        One Representation Function, and individual composition functions for all tasks
         """
        if self.config.model.should_load_model:
            self.load_model()
        # setup 100 composition functions in the main memory
        num_worlds = len(list(self.dataloaders["train"].keys()))
        composition_world_cache = {
            ni: copy.deepcopy(self.composition_fn) for ni in range(num_worlds)
        }
        for ci, cm in composition_world_cache.items():
            cm.to("cpu")
        torch.cuda.empty_cache()
        optim_store_location = os.path.join(self.config.model.save_dir, "opts")
        os.makedirs(optim_store_location)
        # make all optimizers
        representation_optimizer = Net.register_params_to_optimizer(
            self.config,
            self.representation_fn.model.weights,
            is_signature=self.is_signature,
        )
        if self.epoch is None:
            self.epoch = 0
        if self.config.model.should_train:
            for epoch in range(self.epoch, self.config.model.num_epochs):
                train_world_names = list(self.dataloaders["train"].keys())
                train_rule_world = random.choice(train_world_names)
                task_idx = train_world_names.index(train_rule_world)
                self.logbook.write_message_logs(f"Training rule {train_rule_world}")

                self.logbook.write_message_logs(
                    f"Choosing to train the model " f"on {train_rule_world}"
                )
                # Train, optimize and test on the same world
                train_data = self.dataloaders["train"][train_rule_world]
                # set the correct composition function
                composition_world_cache[task_idx].to(self.config.general.device)
                self.composition_fn = composition_world_cache[task_idx]
                composition_optimizer = Net.register_params_to_optimizer(
                    self.config,
                    self.composition_fn.model.weights,
                    is_signature=self.is_signature,
                )
                optim_store_file = os.path.join(
                    optim_store_location, "{}_opt.pt".format(task_idx)
                )
                if os.path.exists(optim_store_file):
                    composition_optimizer.load_state_dict(torch.load(optim_store_file))
                self.optimizers = [representation_optimizer, composition_optimizer]
                self.train(train_data, train_rule_world, epoch, task_idx=task_idx)
                metrics = self.eval(
                    {train_rule_world: self.dataloaders["train"][train_rule_world]},
                    epoch=epoch,
                    mode="valid",
                    data_mode="train",
                    task_idx=task_idx,
                )
                for sched in self.schedulers:
                    sched.step(metrics["loss"])

                composition_world_cache[task_idx].to("cpu")
                torch.save(self.optimizers[-1].state_dict(), optim_store_file)
                torch.cuda.empty_cache()
                self.periodic_save(epoch)

    def run_multitask_training_unique_representation(self):
        """Run multitask training with unique representation functions
        One composition function, and individual representation functions for each task
         """
        if self.config.model.should_load_model:
            self.load_model()
        # setup 100 representation functions in the main memory
        num_worlds = len(list(self.dataloaders["train"].keys()))
        representation_world_cache = {
            ni: copy.deepcopy(self.representation_fn) for ni in range(num_worlds)
        }
        for ri, rm in representation_world_cache.items():
            rm.to("cpu")
        torch.cuda.empty_cache()
        # make all optimizers
        representation_world_optimizer_states = {}
        composition_optimizer = Net.register_params_to_optimizer(
            self.config,
            self.composition_fn.model.weights,
            is_signature=self.is_signature,
        )

        if self.epoch is None:
            self.epoch = 0
        if self.config.model.should_train:
            for epoch in range(self.epoch, self.config.model.num_epochs):
                train_world_names = list(self.dataloaders["train"].keys())
                train_rule_world = random.choice(train_world_names)
                task_idx = train_world_names.index(train_rule_world)
                self.logbook.write_message_logs(f"Training rule {train_rule_world}")

                # ipdb.set_trace()
                self.logbook.write_message_logs(
                    f"Choosing to train the model " f"on {train_rule_world}"
                )
                # Train, optimize and test on the same world
                train_data = self.dataloaders["train"][train_rule_world]
                # set the correct representation function
                representation_world_cache[task_idx].to(self.config.general.device)
                self.representation_fn = representation_world_cache[task_idx]
                representation_optimizer = Net.register_params_to_optimizer(
                    self.config,
                    self.representation_fn.model.weights,
                    is_signature=self.is_signature,
                )
                if task_idx in representation_world_optimizer_states:
                    representation_optimizer.load_state_dict(
                        representation_world_optimizer_states[task_idx]
                    )
                self.optimizers = [representation_optimizer, composition_optimizer]
                self.train(train_data, train_rule_world, epoch, task_idx=task_idx)
                metrics = self.eval(
                    {train_rule_world: self.dataloaders["train"][train_rule_world]},
                    epoch=epoch,
                    mode="valid",
                    data_mode="train",
                    task_idx=task_idx,
                )
                for sched in self.schedulers:
                    sched.step(metrics["loss"])

                representation_world_cache[task_idx].to("cpu")
                representation_world_optimizer_states[task_idx] = self.optimizers[
                    0
                ].state_dict()
                self.periodic_save(epoch)

    def run_random_model(self):
        """Sanity test: run a model with random weights
        """
        if self.config.model.should_load_model:
            self.model.load_model()
        if self.config.model.should_train:
            for task_idx, train_rule_world in enumerate(self.dataloaders["test"]):
                self.eval(
                    {train_rule_world: self.dataloaders["test"][train_rule_world]},
                    epoch=1,
                    mode="test",
                    data_mode="test",
                    task_idx=task_idx,
                )

    def run_pretraining(self):
        """run pretraining on the signature function
         """
        if self.config.model.should_train:
            for epoch in range(self.config.model.num_epochs):
                train_world_names = list(self.dataloaders["train"].keys())
                train_rule_world = random.choice(train_world_names)
                task_idx = train_world_names.index(train_rule_world)
                self.logbook.write_message_logs(f"Training rule {train_rule_world}")

                # ipdb.set_trace()
                self.logbook.write_message_logs(
                    f"Choosing to train the model " f"on {train_rule_world}"
                )

                train_data = self.dataloaders["train"][train_rule_world]
                self.pretrain(train_data, train_rule_world, epoch, task_idx=task_idx)
                # self.save(epochs=epoch)
                metrics = self.pretrain_eval(
                    self.dataloaders["train"],
                    epoch=epoch,
                    mode="valid",
                    data_mode="train",
                    task_idx=task_idx,
                )
                for sched in self.schedulers:
                    sched.step(metrics["loss"])
                self.pretrain_eval(
                    self.dataloaders["valid"],
                    epoch=epoch,
                    mode="test",
                    data_mode="valid",
                    task_idx=task_idx,
                )
                self.pretrain_eval(
                    self.dataloaders["test"], epoch=epoch, mode="test", data_mode="test"
                )
                if self.config.model.persist_frequency > 0:
                    if epoch % self.config.model.persist_frequency == 0:
                        self.save(epoch)

    def eval(
        self,
        data,
        epoch,
        mode="valid",
        data_mode="train",
        task_idx=None,
        skip_world=None,
        report=True,
    ):

        all_metrics = {
            "loss": [],
            "accuracy": [],
        }
        for val_rule_world, valid_data in data.items():
            if skip_world:
                if skip_world == val_rule_world:
                    continue
            # train few shot
            # self.train(valid_data, val_rule_world, epoch=epoch, report=False)
            metrics = self.evaluate(
                valid_data, val_rule_world, epoch=epoch, mode=mode, report=False
            )
            all_metrics["loss"].append(metrics["loss"])
            all_metrics["accuracy"].append(metrics["accuracy"])
        all_metrics = {
            "mode": "{}_{}".format(data_mode, mode),
            "minibatch": self.train_step,
            "loss": np.mean(all_metrics["loss"]),
            "accuracy": np.mean(all_metrics["accuracy"]),
        }
        if task_idx:
            all_metrics["task_idx"] = task_idx
        # self.train_step += 1
        if report:
            self.logbook.write_metric_logs(all_metrics)
        return all_metrics

    def pretrain_eval(
        self,
        data,
        epoch,
        mode="valid",
        data_mode="train",
        task_idx=None,
        skip_world=None,
        report=True,
    ):

        all_metrics = {
            "loss": [],
            "accuracy": [],
        }
        rule_worlds = []
        for val_rule_world, valid_data in data.items():
            if skip_world:
                if skip_world == val_rule_world:
                    continue
            # train few shot
            # self.train(valid_data, val_rule_world, epoch=epoch, report=False)
            metrics = self.pretrain_evaluate(
                valid_data, val_rule_world, epoch=epoch, mode=mode, report=False
            )
            all_metrics["loss"].append(metrics["loss"])
            all_metrics["accuracy"].append(metrics["accuracy"])
            rule_worlds.append(metrics["rule_world"])
        all_metrics = {
            "mode": "{}_{}".format(data_mode, mode),
            "minibatch": self.train_step,
            "loss": np.mean(all_metrics["loss"]),
            "accuracy": np.mean(all_metrics["accuracy"]),
            "rule_worlds": ",".join(rule_worlds),
        }
        if task_idx:
            all_metrics["task_idx"] = task_idx
        # self.train_step += 1
        if report:
            self.logbook.write_metric_logs(all_metrics)
        return all_metrics

    def log_gradients(self, mode):
        # Logging gradients
        grad_metrics = {"mode": mode, "minibatch": self.train_step}
        params = (
            self.composition_fn.model.weights + self.representation_fn.model.weights
        )
        param_names = (
            self.composition_fn.weight_names + self.representation_fn.weight_names
        )
        for wi in range(len(params)):
            weight = params[wi]
            weight_name = param_names[wi]
            if weight.grad is not None:
                grad_metrics[weight_name] = weight.grad.norm().item()
        self.logbook.write_metric_logs(grad_metrics)

    def log_weights(self, mode):
        # Logging gradients
        grad_metrics = {"mode": mode, "minibatch": self.train_step}
        params = (
            self.composition_fn.model.weights + self.representation_fn.model.weights
        )
        param_names = (
            self.composition_fn.weight_names + self.representation_fn.weight_names
        )
        for wi in range(len(params)):
            weight = params[wi]
            weight_name = param_names[wi]
            if weight.grad is not None:
                grad_metrics[weight_name] = weight.norm().item()
        self.logbook.write_metric_logs(grad_metrics)

    def train(self, data, rule_world, epoch=0, report=True, task_idx=None):
        """
        Method to train
        :return:
        """
        mode = "train"
        train_nb = self.config.general.overfit
        epoch_loss = []
        epoch_acc = []
        self.composition_fn.train()
        self.representation_fn.train()
        num_batches = len(data[mode])
        num_batches_to_train = num_batches if train_nb == 0 else train_nb
        for batch_idx, batch in enumerate(data[mode]):
            if batch_idx >= num_batches_to_train:
                continue
            batch.to(self.config.general.device)
            rel_emb = self.representation_fn(batch)
            logits = self.composition_fn(batch, rel_emb)
            loss = self.composition_fn.loss(logits, batch.targets)
            for opt in self.optimizers:
                opt.zero_grad()
            loss.backward()
            for opt in self.optimizers:
                opt.step()
            epoch_loss.append(loss.cpu().detach().item())
            predictions, conf = self.composition_fn.predict(logits)
            epoch_acc.append(
                self.composition_fn.accuracy(predictions, batch.targets)
                .cpu()
                .detach()
                .item()
            )
        if report:
            rule_world_last = rule_world.split("/")[-1]
            metrics = {
                "mode": mode,
                "minibatch": self.train_step,
                "loss": np.mean(epoch_loss),
                "accuracy": np.mean(epoch_acc),
                "epoch": epoch,
                "rule_world": rule_world,
            }
            if task_idx:
                metrics["task_idx"] = task_idx
            self.logbook.write_metric_logs(metrics)
            epoch_loss = []
            epoch_acc = []
        self.train_step += 1

    def pretrain(
        self, data, rule_world, epoch=0, report=True, task_idx=None, num_updates=100
    ):
        """
        Method to pretrain the representation function
        :return:
        """
        mode = "train"
        train_nb = self.config.general.overfit
        epoch_loss = []
        epoch_acc = []
        self.representation_fn.train()
        self.composition_fn.train()
        num_batches = len(data[mode])
        num_batches_to_train = num_batches if train_nb == 0 else train_nb
        # we only sample the first batch, as all batches contain the same world graph
        batches = list(data[mode])
        batch = batches[0]
        batch.to(self.config.general.device)
        for up_i in range(num_updates):
            rel_emb, all_nodes = self.representation_fn(batch)
            logits = self.composition_fn(batch, rel_emb)
            loss = self.composition_fn.loss(logits, targets)
            for opt in self.optimizers:
                opt.zero_grad()
            loss.backward()
            for opt in self.optimizers:
                opt.step()
            epoch_loss.append(loss.cpu().detach().item())
            predictions, conf = self.composition_fn.predict(logits)
            epoch_acc.append(
                self.composition_fn.accuracy(predictions, targets).cpu().detach().item()
            )
            if report:
                is_last = up_i + 1 >= num_updates
                if up_i % self.config.logger.remote.frequency == 0 or is_last:
                    rule_world_last = rule_world.split("/")[-1]
                    metrics = {
                        "mode": mode,
                        "minibatch": self.train_step,
                        "loss": np.mean(epoch_loss),
                        "accuracy": np.mean(epoch_acc),
                        "epoch": epoch,
                        "rule_world": rule_world,
                    }
                    if task_idx:
                        metrics["task_idx"] = task_idx
                    self.logbook.write_metric_logs(metrics)
                    epoch_loss = []
                    epoch_acc = []
            self.train_step += 1

    @torch.no_grad()
    def evaluate(
        self, data, rule_world, epoch=0, mode="valid", top_mode="train", report=True
    ):
        """Method to run the evaluation"""
        assert mode != "train"
        self.composition_fn.eval()
        self.representation_fn.eval()
        eval_nb = self.config.general.overfit
        # self.signature_model.eval()
        epoch_loss = []
        epoch_acc = []
        num_batches = len(data[mode])
        num_batches_to_eval = num_batches if eval_nb == 0 else eval_nb
        for batch_idx, batch in enumerate(data[mode]):
            batch.to(self.config.general.device)
            if batch_idx >= num_batches_to_eval:
                continue
            rel_emb = self.representation_fn(batch)
            logits = self.composition_fn(batch, rel_emb)
            loss = self.composition_fn.loss(logits, batch.targets)
            predictions, conf = self.composition_fn.predict(logits)
            epoch_loss.append(loss.cpu().detach().item())
            epoch_acc.append(
                self.composition_fn.accuracy(predictions, batch.targets)
                .cpu()
                .detach()
                .item()
            )

        rule_world_last = rule_world.split("/")[-1]
        metrics = {
            "mode": mode,
            "minibatch": self.train_step,
            "epoch": epoch,
            "accuracy": np.mean(epoch_acc),
            "loss": np.mean(epoch_loss),
            "top_mode": top_mode,
            "rule_world": rule_world,
        }
        if report:
            # self.train_step += 1
            self.logbook.write_metric_logs(metrics)
        return metrics

    @torch.no_grad()
    def pretrain_evaluate(
        self,
        data,
        rule_world,
        epoch=0,
        mode="valid",
        top_mode="train",
        report=True,
        num_updates=100,
    ):
        """Method to run the evaluation"""
        # assert mode != "train"
        self.model.eval()
        eval_nb = self.config.general.overfit
        epoch_loss = []
        epoch_acc = []
        # we only sample the first batch, as all batches contain the same world graph
        batches = list(data[mode])
        batch = batches[0]
        batch.to(self.config.general.device)
        for up_i in range(num_updates):
            logits, targets = self.model(batch)
            loss = self.model.loss(logits, targets)
            predictions, conf = self.model.predict(logits)
            epoch_loss.append(loss.cpu().detach().item())
            epoch_acc.append(
                self.model.accuracy(predictions, targets).cpu().detach().item()
            )

        rule_world_last = rule_world.split("/")[-1] + " - {}".format(
            np.round(np.mean(epoch_acc), 3)
        )
        metrics = {
            "mode": mode,
            "minibatch": self.train_step,
            "epoch": epoch,
            "accuracy": np.mean(epoch_acc),
            "loss": np.mean(epoch_loss),
            "top_mode": top_mode,
            "rule_world": rule_world_last,
        }
        if report:
            # self.train_step += 1
            self.logbook.write_metric_logs(metrics)
        return metrics

    def run(self):
        """Method to run the experiment"""
        # experiment.run()
        if self.config.model.use_composition_fn:
            self.load_only_composition()
        if self.config.model.use_representation_fn:
            self.load_only_representation()
        if self.config.model.freeze_composition_fn:
            self.load_only_composition()
            self.composition_fn.freeze_weights()
        if self.config.model.freeze_representation_fn:
            self.representation_fn.freeze_weights()
        # re-register the params to the optimizer
        self.register_optim_sched(
            skip_composition_registry=self.config.model.freeze_composition_fn,
            skip_representation_registry=self.config.model.freeze_representation_fn,
        )
        if self.config.general.train_mode == "run_mult":
            self.run_multitask_training()
        elif self.config.general.train_mode == "run_mult_unique_comp":
            self.run_multitask_training_unique_composition()
        elif self.config.general.train_mode == "run_mult_unique_rep":
            self.run_multitask_training_unique_representation()
        elif self.config.general.train_mode == "supervised":
            self.run_single_task(world_mode="train")
        elif self.config.general.train_mode == "supervised_valid":
            self.run_single_task(world_mode="valid")
        elif self.config.general.train_mode == "supervised_test":
            self.run_single_task(world_mode="test")
        elif self.config.general.train_mode == "seq_mult":
            self.run_sequential_multitask_training()
        elif self.config.general.train_mode == "seq_mult_comp":
            self.run_sequential_multitask_unique_composition()
        elif self.config.general.train_mode == "seq_mult_rep":
            self.run_sequential_multitask_unique_representation()
        elif self.config.general.train_mode == "seq_zero":
            self.run_sequential_zeroshot_transfer()
        elif self.config.general.train_mode == "seq_full":
            self.run_sequential_fewshot_transfer(full_shot=True)
        elif self.config.general.train_mode == "seq_few":
            self.run_sequential_fewshot_transfer()
        elif self.config.general.train_mode == "pretrain":
            self.run_pretraining()
        else:
            raise NotImplementedError(
                "training mode not implemented. should be either one of \n supervised / seq_mult / seq_zero / seq_full / seq_few"
            )
