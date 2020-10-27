"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
import os

import torch
from networkx import DiGraph
from torch.utils.data import Dataset

from graphlog import GraphLog
from graphlog.utils import load_networkx_graphs
import pytest


@pytest.mark.parametrize(  # type: ignore
    "gl", [GraphLog(), GraphLog(data_key="graphlog_v1.1")],
)
def test_download(gl) -> None:
    data_loc = os.path.join(gl.data_dir, gl.data_filename)
    assert os.path.exists(data_loc) & os.path.isdir(data_loc)
    train_data_loc = os.path.join(data_loc, "train")
    assert os.path.exists(train_data_loc) & os.path.isdir(train_data_loc)
    valid_data_loc = os.path.join(data_loc, "valid")
    assert os.path.exists(valid_data_loc) & os.path.isdir(valid_data_loc)
    test_data_loc = os.path.join(data_loc, "test")
    assert os.path.exists(test_data_loc) & os.path.isdir(test_data_loc)


@pytest.mark.parametrize(  # type: ignore
    "gl", [GraphLog(), GraphLog(data_key="graphlog_v1.1")],
)
def test_label2id(gl) -> None:
    data_loc = os.path.join(gl.data_dir, gl.data_filename)
    label2id_loc = os.path.join(data_loc, "train", "label2id.json")
    assert os.path.exists(label2id_loc) & os.path.isfile(label2id_loc)
    len(gl.label2id.keys()) == 21
    assert gl.label2id["UNK_REL"] == 0


@pytest.mark.parametrize(  # type: ignore
    "gl", [GraphLog(), GraphLog(data_key="graphlog_v1.1")],
)
def test_paper_data_ids(gl) -> None:
    train_ids = [f"rule_{d}" for d in range(0, 51)]
    valid_ids = [f"rule_{d}" for d in range(51, 54)]
    test_ids = [f"rule_{d}" for d in range(54, 57)]
    data_by_split = gl.get_dataset_names_by_split()
    for world in train_ids:
        assert world in data_by_split["train"]
    for world in valid_ids:
        assert world in data_by_split["valid"]
    for world in test_ids:
        assert world in data_by_split["test"]


@pytest.mark.parametrize(  # type: ignore
    "gl", [GraphLog(), GraphLog(data_key="graphlog_v1.1")],
)
def test_single_dataset(gl) -> None:
    dataset = gl.get_dataset_by_name("rule_0")
    assert isinstance(dataset, Dataset)
    assert len(gl.datasets) == 1


@pytest.mark.parametrize(  # type: ignore
    "gl", [GraphLog(), GraphLog(data_key="graphlog_v1.1")],
)
def test_all_dataset_loading(gl) -> None:
    gl.load_datasets()
    assert len(gl.datasets) == 57


@pytest.mark.parametrize(  # type: ignore
    "gl", [GraphLog(), GraphLog(data_key="graphlog_v1.1")],
)
def test_single_dataloader(gl) -> None:
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    cpu_device = torch.device("cpu")
    dataset = gl.get_dataset_by_name(name="rule_0")
    batch_size = 32
    modes = ["train", "valid", "test"]
    for mode in modes:
        dataloader = gl.get_dataloader_by_mode(
            dataset=dataset, batch_size=batch_size, mode=mode
        )
        for batch in dataloader:
            assert len(batch.targets) == batch_size
            assert len(batch.queries) == len(batch.targets)
            assert batch.targets.device == cpu_device
            batch.to(device)
            assert batch.targets.device == device
            break


# v1.0 specific tests


def test_stats() -> None:
    gl = GraphLog()
    stats = gl.compute_stats_by_dataset(name="rule_0")
    expected_stats = {
        "num_class": 17,
        "num_des": 286,
        "avg_resolution_length": 4.485714285714286,
        "num_nodes": 15.487,
        "num_edges": 19.295,
        "split": "train",
    }
    assert len(stats) == len(expected_stats)
    for key in expected_stats:
        assert expected_stats[key] == stats[key]


def test_load_networkx_graphs() -> None:
    gl = GraphLog()
    dataset = gl.get_dataset_by_name("rule_0")
    nx_graphs, query_nodes = load_networkx_graphs(dataset.json_graphs["train"])
    assert len(nx_graphs) == 5000
    assert isinstance(nx_graphs[0], DiGraph)
    assert len(nx_graphs[0]) == 35
    assert len(query_nodes) == 5000
    for nodes in query_nodes:
        assert len(nodes) == 2


def test_compute_similarity() -> None:
    gl = GraphLog()
    assert gl.compute_similarity("rule_0", "rule_0") == 1.0
    assert gl.compute_similarity("rule_0", "rule_2") == 0.9
    assert gl.compute_similarity("rule_0", "rule_10") == 0.5
    assert gl.compute_similarity("rule_0", "rule_20") == 0.0
    assert gl.compute_similarity("rule_0", "rule_50") == 0.0


def test_get_most_similar_datasets() -> None:
    gl = GraphLog()
    sim = gl.get_most_similar_datasets("rule_0", 5)
    true_sim = [
        ("rule_0", 1.0),
        ("rule_1", 0.95),
        ("rule_2", 0.9),
        ("rule_3", 0.85),
        ("rule_4", 0.8),
    ]
    for si, s in enumerate(sim):
        assert s[0] == true_sim[si][0]
        assert s[1] == true_sim[si][1]
