"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
""" Class to run inference with a initial loaded model params """
import json
import pickle as pkl
import time
from os import listdir, path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from codes.utils.data import DataUtility
from codes.utils.checkpointable import Checkpointable
from codes.model.models import RepresentationFn, CompositionFn
from codes.model.base_model import BaseModel as Net
import numpy as np
import tempfile
import shutil


class InferenceExperiment(Checkpointable):
    """ Class to run inference with multitask and maml params """

    def __init__(self, config, logbook, data):
        self.config = config
        self.logbook = logbook
        self.data = data
        self.dataloaders = self.load_data()
        self.test_world = config.general.test_rule
        self.is_signature = False
        if "," not in self.test_world:
            self.test_data = self.dataloaders["test"][self.test_world]
        else:
            self.test_data = self.dataloaders["test"]
        (self.composition_fn, self.representation_fn) = self.bootstrap_model()
        self.epoch = 0
        self.train_step = 0
        self.best_validation_save_dir = None

    def bootstrap_model(self) -> [nn.Module, nn.Module, torch.optim.Optimizer]:
        composition_fn = CompositionFn(self.config)
        representation_fn = RepresentationFn(self.config)
        return composition_fn, representation_fn

    def register_optim_sched(
        self, skip_composition_registry=False, skip_representation_registry=False
    ):
        # NOTE: be careful of assigning the correct weights to the optimizer.
        # either assign `composition_fn.weights` or `composition_fn.model.weights`
        if self.config.general.is_meta:
            self.config.model.optim.name = "SGD"
            self.config.model.optim.learning_rate = self.config.model.lr_inner
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
        print("Registered Params : {}".format(len(optimizer.param_groups[0]["params"])))

    def load_data(self):
        dataloaders = {}
        modes = ["test"]
        for mi, mode in enumerate(modes):
            dataloaders[mode] = {}
            for graph_world in self.data[mi]:
                rule_world = graph_world.rule_world
                rule_world = rule_world.split("/")[-1]
                dataloaders[mode][rule_world] = graph_world.get_dataloaders(
                    modes=["train", "valid", "test"]
                )
        return dataloaders

    def load_model(self, epoch=None, load_dir=None):
        if not load_dir:
            load_dir = self.config.model.save_dir
        self.composition_fn, self.representation_fn, _, self.epoch = self.load(
            load_dir, epoch, self.composition_fn, self.representation_fn,
        )

    def load_only_composition(
        self, epoch: Optional[int] = None, use_save_dir=False
    ) -> None:
        """Load only composition model
            epoch {Optional[int]} -- [description] (default: {None})
        """
        load_dir = self.config.model.load_dir
        if use_save_dir:
            load_dir = self.config.model.save_dir
        self.composition_fn, _, _, _ = self.load(
            load_dir, epoch, composition_fn=self.composition_fn
        )

    def load_only_representation(
        self, epoch: Optional[int] = None, use_save_dir=False
    ) -> None:
        """Load only composition model
            epoch {Optional[int]} -- [description] (default: {None})
        """
        load_dir = self.config.model.load_dir
        if use_save_dir:
            load_dir = self.config.model.save_dir
        _, self.representation_fn, _, _ = self.load(
            load_dir, epoch, representation_fn=self.representation_fn
        )

    def save_model(self, save_dir=None, epoch: Optional[int] = 0) -> None:
        """save model in tmp directory
        """
        if not save_dir:
            save_dir = tempfile.mkdtemp()
        self.save(
            save_dir,
            epoch,
            self.composition_fn,
            self.representation_fn,
            self.optimizers,
        )
        return save_dir

    def adapt(self, k=0, eps=0.05, report=True, patience=7, data_k=-1, num_epochs=-1):
        """ K-shot adaptation. Here k defines the number of minibatches (or updates) 
        the model is exposed to
        
        Keyword Arguments:
            k {int} -- k shot. if k = -1, train till convergence (default: {0})
        """
        self.composition_fn.train()
        self.representation_fn.train()
        mode = "train"
        break_while = False
        convergence_mode = data_k == -1 and k == -1 and num_epochs == -1
        if convergence_mode:
            print("converging till best validation")
        best_epoch_loss = 10000
        counter = 0
        epoch_id = -1
        num_worlds = list(self.test_data.keys())
        assert len(num_worlds) == 1
        self.test_data = self.test_data[num_worlds[0]]
        while True:
            epoch_id += 1
            epoch_loss = []
            epoch_acc = []
            for batch_idx, batch in enumerate(self.test_data[mode]):
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
                epoch_loss_mean = np.mean(epoch_loss)
                self.train_step += 1
                if k > 0:
                    if self.train_step >= k:
                        break_while = True
                        break
            if data_k > 0:
                data_k -= 1
            epoch_loss_mean = np.mean(epoch_loss)
            valid_eval = self.evaluate(mode="valid")
            test_eval = self.evaluate(mode="test")
            if convergence_mode:
                if best_epoch_loss < valid_eval["loss"]:
                    counter += 1
                else:
                    # save the model in tmp loc
                    self.best_validation_save_dir = self.save_model(
                        epoch=epoch_id, save_dir=self.best_validation_save_dir
                    )
                    print(
                        "saved best model in {}".format(self.best_validation_save_dir)
                    )
                    counter = 0
            if report:
                metrics = {
                    "mode": mode,
                    "minibatch": self.train_step,
                    "loss": epoch_loss_mean,
                    "valid_loss": valid_eval["loss"],
                    "best_valid_loss": best_epoch_loss,
                    "test_acc": test_eval["accuracy"],
                    "accuracy": np.mean(epoch_acc),
                    "epoch": self.epoch,
                    "rule_world": self.test_world,
                    "patience_counter": counter,
                }
                print(metrics)
                # self.logbook.write_metric_logs(metrics)
            # if np.mean(epoch_loss_mean) <= eps:
            #     break
            if data_k == 0:
                break
            # else:
            #     patience = original_patience
            if convergence_mode:
                best_epoch_loss = min(best_epoch_loss, valid_eval["loss"])
                if counter >= patience:
                    break
            if break_while:
                break
            if num_epochs >= 0:
                if epoch_id >= num_epochs:
                    break
        if convergence_mode:
            # reload model from tmp loc
            self.load_model(load_dir=self.best_validation_save_dir)
            shutil.rmtree(self.best_validation_save_dir)
            self.best_validation_save_dir = None

    @torch.no_grad()
    def evaluate(self, mode="test", k=0, ale_mode=False):
        # if ale_mode:
        #     test_world = self.test_world + "_ale"
        # else:
        #     test_world = self.test_world
        if "," in self.test_world:
            worlds = self.test_world.split(",")
        else:
            worlds = self.test_world
        task_loss = []
        task_acc = []
        for test_world in worlds:
            if test_world not in self.dataloaders["test"]:
                continue
            self.test_data = self.dataloaders["test"][test_world]
            self.composition_fn.eval()
            self.representation_fn.eval()
            epoch_loss = []
            epoch_acc = []
            for batch_idx, batch in enumerate(self.test_data[mode]):
                batch.to(self.config.general.device)
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
            task_loss.append(np.mean(epoch_loss))
            task_acc.append(np.mean(epoch_acc))

        metrics = {
            "mode": mode,
            "minibatch": self.train_step,
            "epoch": self.epoch,
            "accuracy": np.mean(task_acc),
            "acc_std": np.std(task_acc),
            "loss": np.mean(task_loss),
            "k": k,
            "top_mode": "test",
            "rule_world": self.test_world,
        }
        # self.logbook.write_metric_logs(metrics)
        return metrics

    @torch.no_grad()
    def evaluate_representations(self, mode="test", k=0, ale_mode=False):
        """Evaluate and store the representations
        """
        # if ale_mode:
        #     test_world = self.test_world + "_ale"
        # else:
        #     test_world = self.test_world
        if "," in self.test_world:
            worlds = self.test_world.split(",")
        else:
            worlds = self.test_world
        task_loss = []
        task_acc = []
        rep_emb_worlds = {}
        for test_world in worlds:
            if test_world not in self.dataloaders["test"]:
                continue
            rep_emb_worlds[test_world] = {}
            self.test_data = self.dataloaders["test"][test_world]
            self.composition_fn.eval()
            self.representation_fn.eval()
            epoch_loss = []
            epoch_acc = []
            for batch_idx, batch in enumerate(self.test_data[mode]):
                batch.to(self.config.general.device)
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
                rep_emb_worlds[test_world][batch_idx] = rel_emb[0].to("cpu").numpy()
            task_loss.append(np.mean(epoch_loss))
            task_acc.append(np.mean(epoch_acc))

        metrics = {
            "mode": mode,
            "minibatch": self.train_step,
            "epoch": self.epoch,
            "accuracy": np.mean(task_acc),
            "acc_std": np.std(task_acc),
            "loss": np.mean(task_loss),
            "k": k,
            "top_mode": "test",
            "rule_world": self.test_world,
        }
        # self.logbook.write_metric_logs(metrics)
        return metrics, rep_emb_worlds

    def reset(self, epoch=None, zero_init=False):
        self.epoch = 0
        self.train_step = 0
        if not zero_init:
            if self.config.model.use_composition_fn:
                self.load_only_composition(epoch=epoch, use_save_dir=True)
            elif self.config.model.use_representation_fn:
                self.load_only_representation(epoch=epoch, use_save_dir=True)
            else:
                self.load_model(epoch)
        if self.config.model.freeze_composition_fn:
            self.composition_fn.freeze_weights()
        if self.config.model.freeze_representation_fn:
            self.representation_fn.freeze_weights()
        self.register_optim_sched(
            skip_composition_registry=self.config.model.freeze_composition_fn,
            skip_representation_registry=self.config.model.freeze_representation_fn,
        )

    def run(self, ale_mode=False):
        ## Evaluate Zero Shot
        zero_metrics = self.evaluate(ale_mode=ale_mode)
        ## Train K-shot
        self.adapt(self.config.general.few_shot_per)
        ## Evaluate
        k_metrics = self.evaluate(k=self.config.general.few_shot_per)
        return zero_metrics, k_metrics
