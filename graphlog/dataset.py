import copy
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData

from graphlog.types import DataLoaderType, GraphType

UNK_LABEL = "UNK_REL"


class GraphRow:
    """Single row of information
    """

    def __init__(
        self,
        graph: GeometricData,
        query: np.ndarray,
        label: np.int64,
        world_graph: GeometricData,
        edge_graph: Optional[GeometricData] = None,
        json_graph: GraphType = None,
    ):
        self.graph = graph
        self.edge_graph = edge_graph
        self.query = query
        self.label = label
        self.world_graph = world_graph
        self.json_graph = json_graph


class GraphLogDataset(Dataset):  # type: ignore
    """GraphLog dataset instance
    Contains all information about a single world
    We provide both the raw json view in order to perform quick
    analysis and visualization, and also the tensorized views based
    on Pytorch Geometric for easy training and evaluation.
    Returns:
        [type] -- [description]
    """

    def __init__(
        self,
        data_loc: str,
        world_id: str,
        label2id: Dict[str, int],
        data_mode: str = "train",
        load_graph: bool = True,
        populate_labels: bool = False,
    ):
        self.data_loc = data_loc
        self.world_id = world_id
        self.label2id = label2id
        # torch.Dataset variables
        self.data_rows: Dict[str, List[GraphRow]] = {
            "train": [],
            "valid": [],
            "test": [],
        }
        self.data_mode = data_mode

        # Graph Level Loading Utilities
        self.graphs: Dict[str, List[GeometricData]] = {
            "train": [],
            "valid": [],
            "test": [],
        }
        # same as graphs, with nodes represented as edges
        self.edge_graphs: Dict[str, List[GeometricData]] = {
            "train": [],
            "valid": [],
            "test": [],
        }
        self.queries: Dict[str, List[Tuple[int, int]]] = {
            "train": [],
            "test": [],
            "valid": [],
        }
        self.labels: Dict[str, List[int]] = {
            "train": [],
            "test": [],
            "valid": [],
        }
        self.path_len: Dict[str, List[int]] = {"train": [], "test": [], "valid": []}
        # contains the raw json graphs

        self.json_graphs: Dict[str, List[GraphType]] = {
            "train": [],
            "test": [],
            "valid": [],
        }
        self.json_meta_graph = None
        self.label_set: Set[int] = set()
        self.world_graph = None
        self.rule_world = self.data_loc
        self.rule_world_config = json.load(
            open(os.path.join(self.rule_world, "config.json"), "r")
        )
        # label2id can be provided in a multittask setting
        self.label2id = label2id
        # add UNK token if not present
        _ = self.get_label2id(UNK_LABEL, mutable=True)

        # mapping between edge to edge_id
        self.unique_edge_dict: Dict[str, int] = {}
        if load_graph:
            self.load_data(rule_world=self.rule_world)
            self.populate_data_rows()
        else:
            if populate_labels:
                self.populate_labels(self.rule_world)

    def get_label2id(self, label: str, mutable: bool = False) -> int:
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

    def populate_labels(self, rule_world: str) -> None:
        """
        Given a rule world, first populate labels
        which will be used subsequently in all places
        NOTE: only to be used for a new dataset
        """
        graph_file = os.path.join(rule_world, "train.jsonl")
        graphs = []
        with open(graph_file, "r") as fp:
            for line in fp:
                graphs.append(json.loads(line))
        for gi, gs in enumerate(graphs):
            for elem in gs["edges"]:
                _ = self.get_label2id(elem[-1], mutable=True)

    def load_data(self, rule_world: str) -> None:
        """
        Load graph data in Pytorch Geometric
        :return:
        """

        for mode in self.graphs:
            graph_file = os.path.join(rule_world, "{}.jsonl".format(mode))
            graphs = []
            with open(graph_file, "r") as fp:
                for line in fp:
                    graphs.append(json.loads(line))
            self.json_graphs[mode] = graphs
            for gi, gs in enumerate(graphs):
                # Graph with Edge attributes
                node2id: Dict[str, int] = {}
                edges = []
                edge_attr = []
                for (src, dst, rel) in gs["edges"]:
                    if src not in node2id:
                        node2id[src] = len(node2id)
                    if dst not in node2id:
                        node2id[dst] = len(node2id)
                    edges.append([node2id[src], node2id[dst]])
                    target = self.get_label2id(rel)
                    edge_attr.append(target)

                (src, dst, rel) = gs["query"]
                self.queries[mode].append((node2id[src], node2id[dst]))
                target = self.get_label2id(rel)
                self.labels[mode].append(target)
                self.label_set.add(target)
                self.path_len[mode].append(len(gs["rules"]))
                x = torch.arange(len(node2id)).unsqueeze(1)

                edge_index = list(zip(*edges))
                edge_index = torch.LongTensor(edge_index)  # type: ignore
                # 2 x num_edges
                assert edge_index.dim() == 2  # type: ignore
                geo_data = GeometricData(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=torch.tensor(edge_attr),
                    y=torch.tensor([target]),
                )
                self.graphs[mode].append(geo_data)

        # load the meta graph
        meta_graph_file = os.path.join(rule_world, "meta_graph.jsonl")
        if os.path.exists(meta_graph_file):
            with open(meta_graph_file, "r") as fp:
                meta_graph = json.loads(fp.read())
                self.json_meta_graph = meta_graph
                edges = []
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
                    # NOTE: We are adding 1 to the edge indicator to keep the first position common for nodes
                    edge_indicator.append(self.get_label2id(elem[2]) + 1)
                x = torch.arange(edge_ct).unsqueeze(1)
                edge_index = list(zip(*edges))
                edge_index = torch.LongTensor(edge_index)  # type: ignore
                # 2 x num_edges
                if edge_index.dim() != 2:  # type: ignore
                    raise AssertionError("edge index dimension should be 2")
                edge_mapping = edge_mapping.unsqueeze(0)  # 1 x num_unique_edges x dim
                self.world_graph = GeometricData(
                    x=x,
                    edge_index=edge_index,
                    edge_indicator=torch.tensor(edge_indicator),
                    edge_mapping=edge_mapping,
                )

        for key in self.queries:
            self.queries[key] = np.asarray(self.queries[key])

        for key in self.labels:
            self.labels[key] = np.asarray(self.labels[key])

    def populate_data_rows(self) -> None:
        for mode in self.graphs:
            graphRows: List[GraphRow] = []
            for i in range(len(self.graphs[mode])):
                graphRows.append(
                    GraphRow(
                        self.graphs[mode][i],
                        self.queries[mode][i],
                        self.labels[mode][i],
                        self.world_graph,
                    )
                )
            self.data_rows[mode] = graphRows

    def set_data_mode(self, mode: str = "train") -> None:
        self.data_mode = mode

    def get_num_classes(self) -> int:
        """
        Return number of classes
        :return:
        """
        return int(self.rule_world_config["num_rel"] * 2)
        # return len(self.label2id)

    def get_num_graphs(self, mode: str) -> int:
        """
        Return number of graphs for a particular mode
        :return:
        """
        return len(self.graphs[mode])

    def __getitem__(self, index: int) -> GraphRow:
        return self.data_rows[self.data_mode][index]

    def __len__(self) -> int:
        return len(self.data_rows[self.data_mode])


# Data utilities


class GraphBatch:
    """
    Batching class
    """

    def __init__(
        self,
        graphs: List[GeometricData],
        queries: Tensor,
        targets: Tensor,
        world_graphs: GeometricData = None,
    ):
        self.num_nodes = [g.num_nodes for g in graphs]
        self.num_edge_nodes = [g.num_nodes for g in world_graphs]
        self.graphs = GeometricBatch.from_data_list(graphs)
        self.queries = torch.LongTensor(queries)  # type: ignore
        self.targets = targets.long()
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

    def to(self, device: str) -> None:
        self.graphs = self.graphs.to(device)
        self.queries = self.queries.to(device)  # type: ignore
        self.targets = self.targets.to(device)
        self.edge_graphs = self.edge_graphs.to(device)
        self.edge_indicator = [ei.to(device) for ei in self.edge_indicator]
        self.edge_mapping = [ei.to(device) for ei in self.edge_mapping]
        self.world_graphs = self.world_graphs.to(device)


def pyg_collate(data: List[GraphRow]) -> GraphBatch:
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
        queries[id][0] = torch.LongTensor([d.query[0]])  # type: ignore
        queries[id][1] = torch.LongTensor([d.query[1]])  # type: ignore
        labels[id] = torch.LongTensor([d.label])  # type: ignore
        # num_nodes[id] = d.graph.num_nodes
        if id == 0:
            # only add world_graphs once
            world_graphs.append(d.world_graph)
    return GraphBatch(
        graphs=graphs, queries=queries, targets=labels, world_graphs=world_graphs
    )


def get_dataloader(
    dataset: GraphLogDataset, mode: str = "train", **kwargs: Any
) -> DataLoaderType:
    """Return dataloader for the specific mode (train/test/valid)
    :param modes:
    :return:
    Arguments:
        dataset {GraphLogDataset} -- [description]
        batch_size {int} -- batch size
    Keyword Arguments:
        mode {str} -- [description] (default: {"train"})
    Returns:
        [type] -- [description]
    """
    dataset.set_data_mode(mode)
    return DataLoader(dataset, collate_fn=pyg_collate, **kwargs)
