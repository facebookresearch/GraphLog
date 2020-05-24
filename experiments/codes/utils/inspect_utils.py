"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
# Utilities for inspection of worlds

import matplotlib.pyplot as plt
import networkx as nx
import json
import os
from os import path
from addict import Dict
import numpy as np

# load graphs
def load_graphs(full_data_path):
    graphs = []
    with open(full_data_path, "r") as fp:
        for line in fp:
            graphs.append(json.loads(line))
    return graphs


def get_rel_pattern(g):
    res_path = g["resolution_path"]
    dt = {}
    for e in g["edges"]:
        dt[(e[0], e[1])] = e[2]
    pattern = []
    for i in range(len(res_path) - 1):
        e = (res_path[i], res_path[i + 1])
        pattern.append(dt[e])
    return ",".join(pattern)


def get_path_dict(data_name="", data_loc=""):
    """get all paths
    Return: dictionary containing {mode:{rule_worlds}}
    """
    loc = os.path.join(data_loc, data_name)
    all_paths = {}
    modes = ["train", "valid", "test"]
    for mode in modes:
        data_path = os.path.join(loc, mode)
        all_paths[mode] = [
            folder
            for folder in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, folder))
        ]
    return all_paths, loc


## loading helper functions
def get_paths(mode, rule_folder, loc):
    paths = Dict()
    paths.mode = mode
    paths.rule_folder = rule_folder
    paths.config_path = path.join(loc, mode, rule_folder, "config.json")
    paths.train_path = path.join(loc, mode, rule_folder, "train.jsonl")
    paths.test_path = path.join(loc, mode, rule_folder, "test.jsonl")
    paths.valid_path = path.join(loc, mode, rule_folder, "valid.jsonl")
    paths.meta_graph_path = path.join(loc, mode, rule_folder, "meta_graph.jsonl")
    paths.graph_prop_path = path.join(loc, mode, rule_folder, "graph_prop.json")
    return paths


def load_world(paths):
    world = Dict()
    world.paths = paths
    world.config = json.load(open(paths.config_path))
    world.train = load_graphs(paths.train_path)
    world.test = load_graphs(paths.test_path)
    world.valid = load_graphs(paths.valid_path)
    world.meta_graph = load_graphs(paths.meta_graph_path)
    # world.graph_prop = json.load(graph_prop_path)
    return world


def get_all_worlds(data_name="", data_loc="", all_paths=None):
    """load and return the json of all worlds
    """
    if not all_paths:
        all_paths, loc = get_path_dict(data_name, data_loc)
    worlds = {}
    for mode, rule_worlds in all_paths.items():
        worlds[mode] = {}
        for rule_world in rule_worlds:
            worlds[mode][rule_world] = load_world(get_paths(mode, rule_world))
    return worlds


## more helper functions
def get_class(world):
    modes = ["train", "valid", "test"]
    cls = []
    for mode in modes:
        graphs = world[mode]
        for g in graphs:
            cls.append(g["query"][-1])
    return set(cls)


def get_descriptors(world):
    modes = ["train", "valid", "test"]
    des = []
    for mode in modes:
        graphs = world[mode]
        des.extend([get_rel_pattern(g) for g in graphs])
    des = set(des)
    return des


def get_avg_resolution_length(world):
    modes = ["train", "valid", "test"]
    res_length = []
    for mode in modes:
        graphs = world[mode]
        le = [len(get_rel_pattern(g).split(",")) for g in graphs]
        res_length.extend(le)
    return np.mean(res_length)


def load_networkx_graphs(graphs):
    nx_graphs = []
    query_nodes = []
    for graph in graphs:
        g = nx.DiGraph()
        nodes = {}
        for edge in graph["edges"]:
            if edge[0] not in nodes:
                nodes[edge[0]] = len(nodes)
            if edge[1] not in nodes:
                nodes[edge[1]] = len(nodes)
            in_resolution = (edge[0] in graph["resolution_path"]) & (
                edge[1] in graph["resolution_path"]
            )
            g.add_edge(
                nodes[edge[0]],
                nodes[edge[1]],
                data={"relation": edge[2], "in_resolution": in_resolution},
            )
        g.node[nodes[graph["query"][0]]]["is_query"] = True
        g.node[nodes[graph["query"][1]]]["is_query"] = True
        nx_graphs.append(g)
        query_nodes.append((nodes[graph["query"][0]], nodes[graph["query"][1]]))
    return nx_graphs, query_nodes
