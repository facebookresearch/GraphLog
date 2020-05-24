"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
## Data iterator
import json
import os
from os import listdir, path

import numpy as np
import torch
from addict import Dict
from tqdm import tqdm

from codes.utils.config import get_config
from codes.utils.data import DataUtility


class TaskFamily:
    """
    Meta Dataset
    """

    def __init__(self, config, meta_mode: str):
        self.config = config
        self.meta_mode = meta_mode

        # data_path = path.join(
        #     os.getcwd().split("lgw")[0], "lgw", config.general.data_name, mode
        # )
        data_path = path.join(
            os.path.expanduser("~/checkpoint/lgw/data"), config.general.data_name
        )
        # data_path is a directory containing multiple graphworlds
        # load label2id
        # TODO: assuming label2id is already generated, as in CheckpointableTestTube
        train_path = path.join(data_path, "train")
        label2id = json.load(open(os.path.join(train_path, "label2id.json")))
        data_path = path.join(data_path, meta_mode)

        self.graphworld_list = []
        mode_folders = [
            folder
            for folder in listdir(data_path)
            if path.isdir(path.join(data_path, folder))
        ]
        pb = tqdm(total=len(mode_folders))
        for folder in mode_folders:
            if not path.exists(path.join(data_path, folder, "config.json")):
                continue
            dt = DataUtility(
                config=config,
                data_folder=path.join(data_path, folder),
                label2id=label2id,
            )
            self.graphworld_list.append(dt)
            pb.update(1)
        pb.close()
        self.graphworld_next_datapoint_to_read = [
            dict(train=0, test=0, valid=0) for _ in self.graphworld_list
        ]
        # This has the list of all the graph worlds (1 graph world per directory)

        self.current_graphworld_idx = -1
        self.num_graphworlds = len(self.graphworld_list)
        self.num_classes = len(label2id)
        # self.num_classes = max([dt.get_num_classes() for dt in self.graphworld_list])

    def sample_task(self, task_id=None):
        """ Sampling a task means sampling an image. """
        # choose image
        if task_id is None:
            self.current_graphworld_idx = (
                self.current_graphworld_idx + 1
            ) % self.num_graphworlds
        else:
            self.current_graphworld_idx = task_id
        return self.get_target_function()

    def get_target_function(self):
        def target_function(input_index, mode):
            # file_name = self.file_list[file_index]
            data = self.graphworld_list[self.current_graphworld_idx]
            return data.labels[mode][input_index]

        return target_function

    def sample_inputs(self, batch_size, mode):
        # mode = kwargs["mode"]
        data = self.graphworld_list[self.current_graphworld_idx]
        next_index_to_read = self.graphworld_next_datapoint_to_read[
            self.current_graphworld_idx
        ][mode]
        index_to_read_till = min(
            next_index_to_read + batch_size - 1, data.get_num_graphs(mode) - 1
        )

        assert index_to_read_till - next_index_to_read < batch_size + 10

        indices = torch.arange(start=next_index_to_read, end=index_to_read_till + 1)

        next_index_to_read = (index_to_read_till + 1) % data.get_num_graphs(mode)
        self.graphworld_next_datapoint_to_read[self.current_graphworld_idx][
            mode
        ] = next_index_to_read

        if type(data.graphs[mode]) == np.array:
            return (
                data.graphs[mode][indices],
                data.queries[mode][indices],
                indices,
                data.world_graph,
            )
        else:
            graphs = [data.graphs[mode][idx] for idx in indices]
            return graphs, data.queries[mode][indices], indices, data.world_graph

    def get_input_range(self, mode="train"):
        data = self.graphworld_list[self.current_graphworld_idx]
        indices = range(data.get_num_graphs(mode))
        if type(data.graphs[mode]) == np.array:
            return data.graphs[mode][indices], data.queries[mode][indices], indices
        else:
            graphs = [data.graphs[mode][idx] for idx in indices]
            return graphs, data.queries[mode][indices], indices


if __name__ == "__main__":
    config = get_config("sample_config")
    meta_mode = "train"
    mode = "train"
    metadata = TaskFamily(config, mode=meta_mode)
    target_fn = metadata.sample_task()
    graphs, queries, indices = metadata.sample_inputs(batch_size=32, mode=mode)
    for graph, query, idx in zip(graphs, queries, indices):
        print(graph, queries, target_fn(idx, mode))
