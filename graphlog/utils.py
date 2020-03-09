# Some of this code is taken from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py
import hashlib
import os
import urllib
import zipfile
from os import path
from typing import Any, Callable, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from addict import Dict as ADict
from tqdm.auto import tqdm

from graphlog.types import GraphType, WorldType


def check_md5(fpath: str, md5: str) -> bool:
    return md5 == calculate_md5(fpath)


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(fpath: str, md5: str) -> bool:
    if not os.path.isfile(fpath):
        return False
    return check_md5(fpath, md5)


def makedir_exist_ok(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def download_and_extract_archive(
    url: str, download_dir: str, filename: str, md5: str,
) -> None:

    archive_filename = "data.zip"
    download_url(url=url, download_dir=download_dir, filename=archive_filename, md5=md5)

    archive_path = os.path.join(download_dir, archive_filename)
    data_path = os.path.join(download_dir)
    print("Extracting {} to {}".format(archive_path, data_path))
    extract_archive(from_path=archive_path, to_path=data_path)


def extract_archive(from_path: str, to_path: str) -> None:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with zipfile.ZipFile(from_path, "r") as z:
        z.extractall(to_path)

    os.remove(from_path)


def gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm(total=None)

    def bar_update(count: int, block_size: int, total_size: int) -> None:
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url: str, download_dir: str, filename: str, md5: str) -> None:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        download_dir (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    makedir_exist_ok(download_dir)
    fpath = os.path.join(download_dir, filename)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:  # download the file
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())  # type: ignore
        except (urllib.error.URLError, IOError) as e:  # type: ignore
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())  # type: ignore
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")


# Dataset utils


def get_rel_pattern(g: Dict[str, GraphType]) -> str:
    res_path = g["resolution_path"]
    dt = {}
    for e in g["edges"]:
        dt[(e[0], e[1])] = e[2]
    pattern = []
    for i in range(len(res_path) - 1):
        e = (res_path[i], res_path[i + 1])
        pattern.append(dt[e])
    return ",".join(pattern)


# loading helper functions
def get_paths(mode: str, rule_folder: str, loc: str) -> ADict:
    paths = ADict()
    paths.mode = mode
    paths.rule_folder = rule_folder
    paths.config_path = path.join(loc, mode, rule_folder, "config.json")
    paths.train_path = path.join(loc, mode, rule_folder, "train.jsonl")
    paths.test_path = path.join(loc, mode, rule_folder, "test.jsonl")
    paths.valid_path = path.join(loc, mode, rule_folder, "valid.jsonl")
    paths.meta_graph_path = path.join(loc, mode, rule_folder, "meta_graph.jsonl")
    paths.graph_prop_path = path.join(loc, mode, rule_folder, "graph_prop.json")
    return paths


# more helper functions
def get_class(world: WorldType) -> Set[str]:
    modes = ["train", "valid", "test"]
    cls = []
    for mode in modes:
        graphs = world[mode]
        for g in graphs:
            cls.append(g["query"][-1])
    return set(cls)


def get_descriptors(world: WorldType) -> List[str]:
    modes = ["train", "valid", "test"]
    des = []
    for mode in modes:
        graphs = world[mode]
        des.extend([get_rel_pattern(g) for g in graphs])
    des = list(set(des))
    return des


def get_avg_resolution_length(world: WorldType) -> Any:
    modes = ["train", "valid", "test"]
    res_length = []
    for mode in modes:
        graphs = world[mode]
        le = [len(get_rel_pattern(g).split(",")) for g in graphs]
        res_length.extend(le)
    return np.mean(res_length)


def get_num_nodes_edges(world: WorldType) -> Any:
    num_nodes = []
    num_edges = []
    modes = ["train", "valid", "test"]
    for mode in modes:
        graphs = world[mode]
        for gr in graphs:
            num_edges.append(len(gr["edges"]))
            all_nodes = [e[:2] for e in gr["edges"]]
            all_nodes = [r for n in all_nodes for r in n]
            num_nodes.append(len(set(all_nodes)))
    return round(np.mean(num_nodes), 3), round(np.mean(num_edges), 3)


def load_networkx_graphs(
    graphs: List[Dict[str, Any]]
) -> Tuple[List[nx.DiGraph], List[Tuple[int, int]]]:
    nx_graphs = []
    query_nodes = []
    for graph in graphs:
        g = nx.DiGraph()
        nodes: Dict[int, int] = {}
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
        g.nodes[nodes[graph["query"][0]]]["is_query"] = True
        g.nodes[nodes[graph["query"][1]]]["is_query"] = True
        nx_graphs.append(g)
        query_nodes.append((nodes[graph["query"][0]], nodes[graph["query"][1]]))
    return nx_graphs, query_nodes
