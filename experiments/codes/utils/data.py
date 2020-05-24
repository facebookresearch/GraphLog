"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
## [DEPRECATED] Data iterator for using raw files in GraphLog.
## Now just use the GraphLog API
import copy
import glob
import json
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData

UNK_LABEL = "UNK_REL"


class GraphRow:
    def __init__(self, graph, query, label, world_graph=None, edge_graph=None):
        self.graph = graph
        self.edge_graph = edge_graph
        self.query = query
        self.label = label
        self.world_graph = world_graph


class GraphBatch:
    """
    Batch object for PyG GAT
    """

    def __init__(self, graphs, queries, targets, world_graphs=None):
        self.num_nodes = [g.num_nodes for g in graphs]
        self.num_edge_nodes = [g.num_nodes for g in world_graphs]
        self.graphs = GeometricBatch.from_data_list(graphs)
        self.queries = torch.LongTensor(queries)
        self.targets = torch.tensor(targets).long()
        self.edge_indicator = [g["edge_indicator"] for g in world_graphs]
        self.edge_mapping = [g["edge_mapping"] for g in world_graphs]
        filtered_list = []
        # removing from the attribute "edge_mapping" from world graphs
        for g in world_graphs:
            e_g = copy.deepcopy(g)
            e_g["edge_mapping"] = torch.zeros(1)
            filtered_list.append(e_g)
        self.edge_graphs = GeometricBatch.from_data_list(filtered_list)
        self.world_graphs = GeometricBatch.from_data_list(world_graphs)

    def to(self, device):
        self.graphs = self.graphs.to(device)
        self.queries = self.queries.to(device)
        self.targets = self.targets.to(device)
        self.edge_graphs = self.edge_graphs.to(device)
        self.edge_indicator = [ei.to(device) for ei in self.edge_indicator]
        self.edge_mapping = [ei.to(device) for ei in self.edge_mapping]
        self.world_graphs = self.world_graphs.to(device)


def graph_batch_iterator(
    graphs: List[GeometricData],
    queries,
    targets,
    world_graphs: List[GeometricData],
    num_nodes_per_batch: int = 2000,
):
    """Method to create mutliple graph batches, such that none of the batches have more than a threshold number of nodes"""
    start_idx = 0
    end_idx = -1
    num_nodes_in_current_batch = 0
    while num_nodes_in_current_batch <= num_nodes_per_batch and end_idx < len(graphs):
        nodes_in_next_graph = graphs[end_idx].num_nodes
        if num_nodes_in_current_batch + nodes_in_next_graph <= num_nodes_per_batch:
            end_idx += 1
            num_nodes_in_current_batch += nodes_in_next_graph
        else:
            if start_idx == end_idx:

                args = [
                    [arg[start_idx]] if arg is not None else arg
                    for arg in [graphs, queries, targets]
                ] + [world_graphs]
            else:
                args = [
                    arg[start_idx:end_idx] if arg is not None else arg
                    for arg in [graphs, queries, targets]
                ] + [world_graphs]

            yield GraphBatch(*args)
            start_idx = end_idx + 1
            num_nodes_in_current_batch = 0


class GraphDataLoader(Dataset):
    """
    Graph Data loader instance
    """

    def __init__(self, dataRows: List[GraphRow]):
        self.dataRows = dataRows

    def __getitem__(self, item):
        return self.dataRows[item]

    def __len__(self):
        return len(self.dataRows)


class DataUtility:
    """
    Basic data utility class
    """

    def __init__(
        self,
        config,
        data_folder=None,
        label2id=None,
        load_graph=True,
        populate_labels=False,
    ):
        self.config = config
        self.graphs = {"train": [], "valid": [], "test": []}
        # same as graphs, with nodes represented as edges
        self.edge_graphs = {"train": [], "valid": [], "test": []}
        self.queries = {"train": [], "test": [], "valid": []}
        self.labels = {"train": [], "test": [], "valid": []}
        self.path_len = {"train": [], "test": [], "valid": []}
        self.label_set = set()
        self.world_graph = None

        if data_folder is None:
            # data_folder = os.path.join(os.getcwd().split("lgw")[0], "lgw", "data")
            data_folder = "~/checkpoint/lgw/data"
            rule_world = os.path.join(data_folder, self.config.general.data_name)
        else:
            rule_world = data_folder
        self.rule_world = data_folder
        self.rule_world_config = json.load(
            open(os.path.join(self.rule_world, "config.json"), "r")
        )
        # label2id can be provided in a multittask setting
        self.label2id = label2id
        # add UNK token if not present
        _ = self.get_label2id(UNK_LABEL, mutable=True)

        # mapping between edge to edge_id
        self.unique_edge_dict = {}
        self.meta_info = {}
        if load_graph:
            self.load_data_pyg(rule_world=rule_world)
        else:
            if populate_labels:
                self.populate_labels(rule_world)

    # creates an issue with fsrl_{} as the relation ids are
    # not contiguous
    def _get_label2id(self, label):
        """
        [DEPRECATED]
        store and retrieve label ids
        Deterministically determine the ID of the relation
        :param label:
        :return:
        """
        if label not in self.label2id:
            ps = label.split("_")
            lb_num = int(ps[1])
            if ps[2] == "-":
                lb_num += self.rule_world_config["num_rel"]
            self.label2id[label] = lb_num
        return self.label2id[label]

    def get_label2id(self, label, mutable=False):
        """
        store and retrieve label ids
        If label not found, return UNK_LABEL
        :param label:
        :return:
        """
        if label not in self.label2id:
            if mutable:
                self.label2id[label] = len(self.label2id)
            else:
                return self.get_label2id(UNK_LABEL)
        return self.label2id[label]

    def load_data_dgl(self, rule_world):
        """
        load data from a particular rule world
        in DGL
        :param rule_world:
        :return:
        """
        print("Loading data")
        # rule_world = os.path.join(data_exp, rule_world)
        for mode in self.graphs:
            mode_folder = os.path.join(rule_world, mode)
            gs = glob.glob(mode_folder + "/*.txt")
            g_query = [f for f in gs if "_query" in f]
            gs = [f for f in gs if f not in g_query]
            for gi, gl in enumerate(gs):
                graph_id = gl.split("/")[-1].split(".txt")[0]
                g = dgl.DGLGraph()
                node2id = {}
                edges = []
                with open(gl, "r") as fp:
                    for line in fp:
                        elem = line.rstrip().split(" ")
                        if elem[0] not in node2id:
                            node2id[elem[0]] = len(node2id)
                        if elem[1] not in node2id:
                            node2id[elem[1]] = len(node2id)
                        edges.append([node2id[elem[0]], node2id[elem[1]], elem[2]])
                node_query_flags = torch.zeros(len(node2id))
                with open(
                    os.path.join(mode_folder, "{}_query.txt".format(graph_id)), "r"
                ) as fp:
                    lines = fp.readlines()
                    elem = lines[0].rstrip().split(" ")
                    self.queries[mode].append((node2id[elem[0]], node2id[elem[1]]))
                    node_query_flags[node2id[elem[0]]] = 1
                    node_query_flags[node2id[elem[1]]] = 2
                    self.labels[mode].append(int(elem[2]))
                    self.label_set.update(elem[2])
                for nf in node_query_flags:
                    qr = torch.zeros(
                        1, 1, requires_grad=False, device=self.config.general.device
                    )
                    qr[0][0] = nf
                    g.add_nodes(1, data={"q": qr})
                for edge in edges:
                    rel = torch.zeros(1, 1, device=self.config.general.device)
                    rel[0][0] = int(edge[2])
                    self.label_set.add(edge[2])
                    rel = rel.long()
                    g.add_edge(edge[0], edge[1], data={"rel": rel})
                self.graphs[mode].append(g)
            print("{} Data loaded : {} graphs".format(mode, len(gs)))

        for key in self.graphs:
            self.graphs[key] = np.asarray(self.graphs[key])

        for key in self.queries:
            self.queries[key] = np.asarray(self.queries[key])

        for key in self.labels:
            self.labels[key] = np.asarray(self.labels[key])

    def populate_labels(self, rule_world):
        """
        Given a rule world, first populate labels
        which will be used subsequently in all places
        """
        graph_file = os.path.join(rule_world, "train.jsonl")
        graphs = []
        with open(graph_file, "r") as fp:
            for line in fp:
                graphs.append(json.loads(line))
        for gi, gs in enumerate(graphs):
            for elem in gs["edges"]:
                _ = self.get_label2id(elem[-1], mutable=True)

    def load_data_pyg(self, rule_world):
        """
        Load data in pytorch geometric
        :return:
        """
        # print("Loading data")
        # rule_world = os.path.join(data_exp, rule_world)

        for mode in self.graphs:
            graph_file = os.path.join(rule_world, "{}.jsonl".format(mode))
            graphs = []
            self.meta_info[graph_file] = []
            with open(graph_file, "r") as fp:
                for line in fp:
                    graphs.append(json.loads(line))
            for gi, gs in enumerate(graphs):
                ## Graph with Edge attributes
                node2id = {}
                edges = []
                # edge_types = []
                edge_attr = []
                for (src, dst, rel) in gs["edges"]:
                    if src not in node2id:
                        node2id[src] = len(node2id)
                    if dst not in node2id:
                        node2id[dst] = len(node2id)
                    edges.append([node2id[src], node2id[dst]])
                    target = self.get_label2id(rel)
                    edge_attr.append(target)
                    # edge_types.append(rel)
                # node_query_flags = torch.zeros(len(node2id))
                (src, dst, rel) = gs["query"]
                self.queries[mode].append((node2id[src], node2id[dst]))
                # node_query_flags[node2id[src]] = 1
                # node_query_flags[node2id[dst]] = 2
                target = self.get_label2id(rel)
                self.labels[mode].append(target)
                self.label_set.add(target)
                self.path_len[mode].append(len(gs["rules"]))
                x = torch.arange(len(node2id)).unsqueeze(1)

                edge_index = list(zip(*edges))
                edge_index = torch.LongTensor(edge_index)  # 2 x num_edges
                assert edge_index.dim() == 2
                edge_attr = torch.tensor(edge_attr)

                # num_e = len(edges)
                # edge_attr = torch.zeros(num_e, 1).long()  # [num_edges, 1]
                # for i, e in enumerate(edge_types):
                # edge_attr[i][0] = self.get_label2id(e)
                # nodes = list(set([p for x in edges for p in x]))
                geo_data = GeometricData(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor([target]),
                )

                # edge_query = gs["query"]
                # elem_edges = gs["edges"]
                ## Deprecated: Disabling edge graphs
                ## Graph with Edge as new nodes
                ## add the edges as new node : edge_id + len(nodes)
                ## s.t. later we can just subtract the len(nodes) from the graph
                ## There will be n - 1 new nodes for n nodes
                # num_nodes = len(node2id)
                # edges = []
                # edge_ct = num_nodes
                # if self.config.data.with_answer:
                #     # Adding answer edge in train mode
                #     if mode == "train":
                #         elem_edges.append(edge_query)
                # edge_mapping = torch.zeros(
                #     (len(self.label2id), len(node2id) + len(elem_edges))
                # ).long()
                # for ei, elem in enumerate(elem_edges):
                #     edges.append([node2id[elem[0]], num_nodes + ei])
                #     edges.append([num_nodes + ei, node2id[elem[1]]])
                #     edge_mapping[self.get_label2id(elem[2])][num_nodes + ei] = 1
                #     edge_ct += 1
                # x = torch.arange(edge_ct).unsqueeze(1)
                # edge_index = list(zip(*edges))
                # edge_index = torch.LongTensor(edge_index)  # 2 x num_edges
                # assert edge_index.dim() == 2
                # num_e = len(edges)
                # edge_indicator = torch.zeros_like(x)
                # for node_id in range(edge_ct):
                #     if node_id not in node2id:
                #         edge_indicator[node_id][0] = 1
                # edge_mapping = edge_mapping.unsqueeze(0)  # 1 x num_unique_edges x dim
                # # TODO: check if we need edge_graph at all, if not delete it
                # geo_edge_data = GeometricData(
                #     x=x,
                #     edge_index=edge_index,
                #     edge_indicator=edge_indicator,
                #     edge_mapping=edge_mapping,
                # )

                # for nf in node_query_flags:
                #     qr = torch.zeros(1, 1, requires_grad=False, device=self.config.general.device)
                #     qr[0][0] = nf
                #     g.add_nodes(1, data={'q': qr})

                # for edge in edges:
                #     rel = torch.zeros(1, 1, device=self.config.general.device)
                #     rel[0][0] = int(edge[2])
                #     self.label_set.add(edge[2])
                #     rel = rel.long()
                #     g.add_edge(edge[0], edge[1], data={'rel': rel})
                self.graphs[mode].append(geo_data)
                # self.edge_graphs[mode].append(geo_edge_data)
            # print("{} Data loaded : {} graphs".format(mode, len(graphs)))

        # load the meta graph
        meta_graph_file = os.path.join(rule_world, "meta_graph.jsonl")
        if os.path.exists(meta_graph_file):
            with open(meta_graph_file, "r") as fp:
                meta_graph = json.loads(fp.read())
                edges = []
                edge_types = []
                elem_edges = meta_graph["edges"]
                # populate edge ids
                for elem in elem_edges:
                    if elem[0] not in node2id:
                        node2id[elem[0]] = len(node2id)
                    if elem[1] not in node2id:
                        node2id[elem[1]] = len(node2id)
                edge_mapping = torch.zeros(
                    (len(self.label2id), len(node2id) + len(elem_edges))
                ).long()
                num_nodes = len(node2id)
                edge_ct = num_nodes
                edge_indicator = [0 for ni in range(num_nodes)]
                for ei, elem in enumerate(elem_edges):
                    edges.append([node2id[elem[0]], num_nodes + ei])
                    edges.append([num_nodes + ei, node2id[elem[1]]])
                    edge_mapping[self.get_label2id(elem[2])][num_nodes + ei] = 1
                    edge_ct += 1
                    # we are adding 1 to the edge indicator to keep the first position common for nodes
                    edge_indicator.append(self.get_label2id(elem[2]) + 1)
                x = torch.arange(edge_ct).unsqueeze(1)
                # torch.nn.init.xavier_uniform_(x, gain=1.414)
                edge_index = list(zip(*edges))
                edge_index = torch.LongTensor(edge_index)  # 2 x num_edges
                if edge_index.dim() != 2:
                    import ipdb

                    ipdb.set_trace()
                num_e = len(edges)
                edge_indicator = torch.tensor(edge_indicator)
                # for node_id in range(edge_ct):
                #     if node_id not in node2id:
                #         edge_indicator[node_id][0] = 1
                edge_mapping = edge_mapping.unsqueeze(0)  # 1 x num_unique_edges x dim
                self.world_graph = GeometricData(
                    x=x,
                    edge_index=edge_index,
                    edge_indicator=edge_indicator,
                    edge_mapping=edge_mapping,
                )

        for key in self.queries:
            self.queries[key] = np.asarray(self.queries[key])

        for key in self.labels:
            self.labels[key] = np.asarray(self.labels[key])

    def get_dataloaders(self, modes):
        """
        Return dataloaders
        :param modes:
        :return:
        """
        dls = {}
        for mode in modes:
            graphRows = []
            for i in range(len(self.graphs[mode])):
                graphRows.append(
                    GraphRow(
                        self.graphs[mode][i],
                        self.queries[mode][i],
                        self.labels[mode][i],
                        self.world_graph,
                    )
                )
            dls[mode] = DataLoader(
                GraphDataLoader(graphRows),
                batch_size=self.config.general.batch_size,
                collate_fn=pyg_collate,
            )
        return dls

    def get_num_classes(self):
        """
        Return number of classes
        :return:
        """
        return self.rule_world_config["num_rel"] * 2
        # return len(self.label2id)

    def get_num_graphs(self, mode):
        """
        Return number of graphs for a particular mode
        :return:
        """
        return len(self.graphs[mode])


def dgl_collate(data):
    """
    Custom collate function
    :param data:
    :return:
    """
    graphs = []
    queries = torch.zeros(len(data), 2)
    labels = torch.zeros(len(data), 1)
    for id, d in enumerate(data):
        graphs.append(d.graph)
        queries[id][0] = d.query[0]
        queries[id][1] = d.query[1]
        labels[id] = d.label
    graphs = dgl.batch(graphs)
    return graphs, queries, labels


def pyg_collate(data):
    """
    Custom collate function
    :param data:
    :return:
    """
    graphs = []
    queries = torch.zeros(len(data), 2).long()
    labels = torch.zeros(len(data)).long()
    world_graphs = []
    # num_nodes = torch.zeros(len(data), 1)
    for id, d in enumerate(data):
        graphs.append(d.graph)
        queries[id][0] = torch.LongTensor([d.query[0]])
        queries[id][1] = torch.LongTensor([d.query[1]])
        labels[id] = torch.LongTensor([d.label])
        # num_nodes[id] = d.graph.num_nodes
        if id == 0:
            # only add world_graphs once
            world_graphs.append(d.world_graph)
    return GraphBatch(
        graphs=graphs, queries=queries, targets=labels, world_graphs=world_graphs
    )
    # return graphs, queries, labels


if __name__ == "__main__":
    config = get_config("sample_config")
    data = DataUtility(config)
    dls = data.get_dataloaders(["train", "valid", "test"])
    for batch_idx, batch in enumerate(dls["train"]):
        print(batch_idx)
