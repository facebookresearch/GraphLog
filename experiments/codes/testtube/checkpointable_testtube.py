"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
"""TestTube is the orchestrator of the experiment"""

import json
import pickle as pkl
import time
import os
from os import listdir, path
from typing import Optional, List, Any

import torch
from tqdm import tqdm

from codes.experiment.checkpointable_multitask_experiment import MultitaskExperiment
from codes.experiment.inference import InferenceExperiment

from codes.logbook.filesystem_logger import write_config_log, write_message_logs
from codes.logbook.logbook import LogBook
from codes.utils.checkpointable import Checkpointable
from codes.utils.config import get_config
from codes.utils.data import DataUtility
from codes.utils.util import _import_module, set_seed
from graphlog import GraphLog


class CheckpointableTestTube(Checkpointable):
    """Checkpointable TestTube Class

    This class provides a mechanism to checkpoint the (otherwise stateless) TestTube
    """

    def __init__(self, config_id, load_checkpoint=True, seed=-1):
        self.config = bootstrap_config(config_id, seed)
        self.logbook = LogBook(self.config)
        self.num_experiments = self.config.general.num_experiments
        torch.set_num_threads(self.num_experiments)
        self.device = self.config.general.device
        self.label2id = {}
        self.model = None
        self.gl = GraphLog()

    def bootstrap_model(self):
        """Method to instantiate the models that will be common to all
        the experiments."""
        model = choose_model(self.config)
        model.to(self.device)
        return model

    def load_label2id(self):
        self.label2id = self.gl.get_label2id()
        print("Found : {} labels".format(len(self.label2id)))

    def initialize_data(self, mode="train", override_mode=None) -> List[Any]:
        """
        Load and initialize data here
        :return:
        """
        datasets = self.gl.get_dataset_names_by_split()
        graphworld_list = []
        for rule_world in datasets[mode]:
            graphworld_list.append(self.gl.get_dataset_by_name(rule_world))

        self.config.model.num_classes = len(self.gl.get_label2id())
        self.load_label2id()

        return graphworld_list

    def run(self):
        """Method to run the task"""

        write_message_logs(
            "Starting Experiment at {}".format(
                time.asctime(time.localtime(time.time()))
            )
        )
        write_config_log(self.config)
        write_message_logs("torch version = {}".format(torch.__version__))

        if not self.config.general.is_meta:
            self.train_data = self.initialize_data(mode="train")
            self.valid_data = self.initialize_data(mode="valid")
            self.test_data = self.initialize_data(mode="test")
            self.experiment = MultitaskExperiment(
                config=self.config,
                model=self.model,
                data=[self.train_data, self.valid_data, self.test_data],
                logbook=self.logbook,
            )
        else:
            raise NotImplementedError("NA")
        self.experiment.load_model()
        self.experiment.run()

    def prepare_evaluator(
        self,
        epoch: Optional[int] = None,
        test_data=None,
        zero_init=False,
        override_mode=None,
        label2id=None,
    ):
        self.load_label2id()
        if test_data:
            assert label2id is not None
            self.test_data = test_data
            self.num_graphworlds = len(test_data)
            self.config.model.num_classes = len(label2id)
        else:
            self.test_data = self.initialize_data(
                mode="test", override_mode=override_mode
            )
        self.evaluator = InferenceExperiment(
            self.config, self.logbook, [self.test_data]
        )
        self.evaluator.reset(epoch=epoch, zero_init=zero_init)

    def evaluate(self, epoch: Optional[int] = None, test_data=None, ale_mode=False):
        self.prepare_evaluator(epoch, test_data)
        return self.evaluator.run(ale_mode=ale_mode)


def bootstrap_config(config_id, seed=-1):
    """Method    to generate the config (using config id) and set seeds"""
    config = get_config(config_id, experiment_id=0)
    if seed > 0:
        set_seed(seed=seed)
    else:
        set_seed(seed=config.general.seed)
    return config


def choose_model(config):
    """
    Dynamically load model
    :param config:
    :return:
    """
    model_name = config.model.base_path + "." + config.model.name
    module = _import_module(model_name)
    return module(config)
