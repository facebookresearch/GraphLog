"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
from addict import Dict as ADict
from tqdm.auto import tqdm

from graphlog.dataset import GraphLogDataset as Dataset
from graphlog.dataset import get_dataloader
from graphlog.types import DataLoaderType, StatType
from graphlog.utils import (
    download_and_extract_archive,
    get_avg_resolution_length,
    get_class,
    get_descriptors,
    get_num_nodes_edges,
    get_rule_key,
    load_single_networkx_graph,
)


class GraphLog:

    meta_url = "https://raw.githubusercontent.com/facebookresearch/GraphLog/master/repository.json"  # noqa: E501

    def __init__(self, data_dir: str = "./data/", data_key: str = "graphlog_v1.0"):
        self.data_dir = os.path.abspath(data_dir)
        # make dir if not exists
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self.data_key = data_key
        self.url = ""
        self.md5hash = ""
        self.data_filename = ""
        self.supported_difficulty: List[str] = []
        self.datasets_grouped_by_difficulty: Dict[str, List[int]] = {}
        self.populate_metadata(key=self.data_key)
        self.download()
        self.datasets_by_split = self.get_dataset_names_by_split()
        # get proper label2id
        self.label2id = self.get_label2id()
        self.datasets: Dict[str, Dataset] = {}

    def populate_metadata(self, key: str = "") -> None:
        """Populate metatdata from meta_url
        Keyword Arguments:
            key {str} -- [description] (default: {""})
        Raises:
            AssertionError: [description]
        """
        meta_path = os.path.join(self.data_dir, "meta.json")
        try:
            with urllib.request.urlopen(self.meta_url) as url:
                metadata = json.loads(url.read().decode())
                json.dump(metadata, open(meta_path, "w"))
        except Exception as e:
            print(
                f"Exception : {e}, (OR cannot connect remote to retrieve metadata), looking for saved config..."
            )
            metadata = json.load(open(meta_path))
        if key in metadata:
            self.url = metadata[key]["url"]
            self.md5hash = metadata[key]["md5hash"]
            self.data_filename = metadata[key]["data_filename"]
            if "difficulty" in metadata[key]:
                self.supported_difficulty = list(metadata[key]["difficulty"].keys())
                self.datasets_grouped_by_difficulty = metadata[key]["difficulty"]
            else:
                self.supported_difficulty = []
        else:
            raise AssertionError(
                f"key {key} not found in config repository. Please refer the docs for available dataset configs"
            )

    def _get_datafile_path(self) -> str:
        return os.path.join(self.data_dir, self.data_filename)

    def _check_exists(self) -> bool:
        return os.path.exists(self._get_datafile_path())

    def download(self) -> None:

        if self._check_exists():
            return

        os.makedirs(self.data_dir, exist_ok=True)

        download_and_extract_archive(
            url=self.url,
            download_dir=self.data_dir,
            filename=self.data_filename,
            md5=self.md5hash,
        )

    def get_label2id(self) -> Dict[str, int]:
        label2id_loc = os.path.join(
            self.data_dir, self.data_filename, "train", "label2id.json"
        )
        label2id = json.load(open(label2id_loc))
        assert isinstance(label2id, dict)
        return label2id

    def load_datasets(self) -> None:
        """Load all datasets
        Returns:
            Dict[str, Dataset] -- [description]
        """
        data_names = self.get_dataset_names_by_split()
        all_datasets = {}
        for mode, names in data_names.items():
            print(f"Loading {len(names)} {mode} datasets ...")
            pb = tqdm(total=len(names))
            for name in names:
                all_datasets[name] = self._load_single_dataset(mode, name)
                pb.update(1)
            pb.close()
        self.datasets = all_datasets

    def _load_single_dataset(self, mode: str, name: str) -> Dataset:
        return Dataset(
            data_loc=os.path.join(self.data_dir, self.data_filename, mode, name),
            world_id=name,
            label2id=self.get_label2id(),
        )

    def get_dataset_ids(self) -> List[str]:
        return sorted(self.datasets_grouped_by_difficulty.keys())

    def get_dataset_split(self, name: str) -> str:
        # determine mode
        if name in self.datasets_by_split["train"]:
            mode = "train"
        elif name in self.datasets_by_split["valid"]:
            mode = "valid"
        elif name in self.datasets_by_split["test"]:
            mode = "test"
        else:
            raise FileNotFoundError(
                name
                + " not found in any splits of GraphLog. "
                + "To view all available datasets, use `get_dataset_names_by_split`"
            )
        return mode

    def get_dataset_by_name(self, name: str) -> Dataset:
        mode = self.get_dataset_split(name)
        if name not in self.datasets:
            self.datasets[name] = self._load_single_dataset(mode, name)
        return self.datasets[name]

    def _check_difficulty_support(self) -> bool:
        """check if the loaded dataset supports difficulty
        """
        return len(self.supported_difficulty) > 0

    def _get_dataset_by_difficulty(self, difficulty: str) -> List[Dataset]:
        if not self._check_difficulty_support():
            raise AssertionError("Current dataset does not support difficulty API's")
        assert difficulty in self.datasets_grouped_by_difficulty
        return [
            self.get_dataset_by_name("rule_{id}")
            for id in self.datasets_grouped_by_difficulty[difficulty]
        ]

    def get_easy_datasets(self) -> List[Dataset]:
        difficulty = "easy"
        return self._get_dataset_by_difficulty(difficulty=difficulty)

    def get_moderate_datasets(self) -> List[Dataset]:
        difficulty = "moderate"
        return self._get_dataset_by_difficulty(difficulty=difficulty)

    def get_hard_datasets(self) -> List[Dataset]:
        difficulty = "hard"
        return self._get_dataset_by_difficulty(difficulty=difficulty)

    def get_dataset_names_by_split(self) -> Dict[str, List[str]]:
        """ Return list of available datasets as provided in the paper
        Returns:
            List[str] -- [list of world ids]
        """
        datasets_grouped_by_split: Dict[str, List[str]] = {
            "train": [],
            "valid": [],
            "test": [],
        }
        data_loc = os.path.join(self.data_dir, self.data_filename)
        for mode in datasets_grouped_by_split:
            mode_loc = os.path.join(data_loc, mode)
            dirs = [
                folder
                for folder in os.listdir(mode_loc)
                if os.path.isdir(os.path.join(mode_loc, folder))
            ]
            datasets_grouped_by_split[mode].extend([d.split("/")[-1] for d in dirs])
        return datasets_grouped_by_split

    def get_dataloader_by_mode(
        self, dataset: Dataset, mode: str = "train", **kwargs: Any
    ) -> DataLoaderType:
        """Get relevant dataloader of the dataset object
        Arguments:
            dataset {Dataset} -- GraphLogDataset object
            batch_size {int} -- integer
        Keyword Arguments:
            mode {str} -- [description] (default: {"train"})
        Returns:
            DataLoader -- [description]
        """
        return get_dataloader(dataset, mode, **kwargs)

    def compute_stats_by_dataset(self, name: str) -> StatType:
        """Compute stats for the given world
        Arguments:
            name {str} -- [description]
        Returns:
            Dict[str, Any] -- [description]
        """
        dataset = self.get_dataset_by_name(name)
        stat = ADict()
        stat.num_class = len(get_class(dataset.json_graphs))
        stat.num_des = len(get_descriptors(dataset.json_graphs))
        stat.avg_resolution_length = get_avg_resolution_length(dataset.json_graphs)
        stat.num_nodes, stat.num_edges = get_num_nodes_edges(dataset.json_graphs)
        stat.split = self.get_dataset_split(name)
        print(
            f"Data Split : {stat.split},"
            f"Number of Classes : {stat.num_class},"
            f"Number of Descriptors : {stat.num_des},"
            f"Average Resolution Length : {stat.avg_resolution_length},"
            f"Average number of nodes : {stat.num_nodes} and"
            f"edges : {stat.num_edges}"  # noqa: E501
        )
        assert isinstance(stat, dict)
        return stat

    def load_config(self, name: str) -> Dict[str, Any]:
        """Load the config of a dataset
        Arguments:
            name {str} -- [description]
        Returns:
            Dict[str, Any] -- [description]
        """
        data_mode = self.get_dataset_split(name)
        config_file_path = os.path.join(
            self.data_dir, self.data_filename, data_mode, name, "config.json"
        )
        config: Dict[str, Any] = json.load(open(config_file_path))
        return config

    def load_rules(self, name: str) -> List[Any]:
        """Load the rules of a dataset
        Arguments:
            name {str} -- [description]
        Returns:
            Dict -- [description]
        """
        config: Dict[str, List[Any]] = self.load_config(name)
        return config["rules"]

    def compute_similarity(self, name_1: str, name_2: str) -> float:
        """Compute Similarity between the rules of two datasets
        """
        rules_1 = set(get_rule_key(self.load_rules(name_1)))
        rules_2 = set(get_rule_key(self.load_rules(name_2)))
        return len(rules_1.intersection(rules_2)) / len(rules_1)

    def get_most_similar_datasets(
        self, name: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """get top_k most similar datasets to name
        Arguments:
            name {str} -- [description]
        Keyword Arguments:
            top_k {int} -- [description] (default: {10})
        Returns:
            List[str] -- [description]
        """
        all_data = self.get_dataset_names_by_split()
        sim_list = []
        for mode in ["train", "valid", "test"]:
            for data_name in all_data[mode]:
                sim_list.append((data_name, self.compute_similarity(name, data_name)))
        sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)
        return sim_list[:top_k]

    def draw_single_graph(
        self, dataset: Dataset, mode: str = "train", graph_id: int = 0
    ) -> None:
        """ Draw a single graph
        """
        if mode not in ["train", "valid", "test"]:
            raise NotImplementedError(
                "mode not available, has to be one of train/valid/test"
            )
        mode_raw_data = dataset.json_graphs[mode]
        if graph_id > len(mode_raw_data):
            raise AssertionError(
                f"graph_id greater than number of available graphs ({len(mode_raw_data)})"
            )
        raw_graph = mode_raw_data[graph_id]
        G, _ = load_single_networkx_graph(raw_graph)
        resolution_path = raw_graph["resolution_path"]
        pos = nx.spring_layout(G)

        def decide_node_color(node: int) -> str:
            if "is_query" in G.nodes()[node]:
                return "red"
            elif node in resolution_path:
                return "blue"
            else:
                return "black"

        edge_colors = [
            "blue" if G[e[0]][e[1]]["data"]["in_resolution"] else "black"
            for e in G.edges()
        ]
        node_colors = [decide_node_color(n) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors)
        nx.draw_networkx_edges(G, pos=pos, edge_color=edge_colors)
