import os

import torch
from torch.utils.data import Dataset

from graphlog import GraphLog


def test_download() -> None:
    gl = GraphLog()
    data_loc = os.path.join(gl.data_dir, gl.data_filename)
    assert os.path.exists(data_loc) & os.path.isdir(data_loc)
    train_data_loc = os.path.join(data_loc, "train")
    assert os.path.exists(train_data_loc) & os.path.isdir(train_data_loc)
    valid_data_loc = os.path.join(data_loc, "valid")
    assert os.path.exists(valid_data_loc) & os.path.isdir(valid_data_loc)
    test_data_loc = os.path.join(data_loc, "test")
    assert os.path.exists(test_data_loc) & os.path.isdir(test_data_loc)


def test_label2id() -> None:
    gl = GraphLog()
    data_loc = os.path.join(gl.data_dir, gl.data_filename)
    label2id_loc = os.path.join(data_loc, "train", "label2id.json")
    assert os.path.exists(label2id_loc) & os.path.isfile(label2id_loc)
    len(gl.label2id.keys()) == 21
    assert gl.label2id["UNK_REL"] == 0


def test_paper_data_ids() -> None:
    gl = GraphLog()
    train_ids = ["rule_{}".format(d) for d in range(0, 51)]
    valid_ids = ["rule_{}".format(d) for d in range(51, 54)]
    test_ids = ["rule_{}".format(d) for d in range(54, 57)]
    data_by_split = gl.get_dataset_names_by_split()
    for world in train_ids:
        assert world in data_by_split["train"]
    for world in valid_ids:
        assert world in data_by_split["valid"]
    for world in test_ids:
        assert world in data_by_split["test"]


def test_single_dataset() -> None:
    gl = GraphLog()
    dataset = gl.get_dataset_by_name("rule_0")
    assert isinstance(dataset, Dataset)
    assert len(gl.datasets) == 1


def test_all_dataset_loading() -> None:
    gl = GraphLog()
    gl.load_datasets()
    assert len(gl.datasets) == 57

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
