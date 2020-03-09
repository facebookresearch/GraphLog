import json
import os
from typing import Any, Dict, List

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
)


class GraphLog:

    url = "https://www.cs.mcgill.ca/~ksinha4/data/graphlog_icml_data.zip"
    md5hash = "6aeda25d56251328dcdca0b23b4abe0a"
    data_filename = "comp_r10_n100_ov"

    supported_difficulty = ["easy", "moderate", "hard"]

    def __init__(self, data_dir: str = "./data/"):
        self.data_dir = data_dir
        self.download()
        self.datasets_grouped_by_difficulty: Dict[str, List[int]] = {
            "easy": [
                9,
                12,
                18,
                19,
                20,
                24,
                27,
                31,
                34,
                35,
                36,
                37,
                38,
                39,
                46,
                47,
                48,
                49,
                51,
            ],
            "moderate": [
                4,
                7,
                8,
                10,
                11,
                21,
                22,
                26,
                28,
                32,
                33,
                40,
                41,
                43,
                45,
                50,
                52,
                54,
                55,
                56,
            ],
            "hard": [0, 1, 2, 3, 5, 6, 13, 14, 15, 16, 17, 23, 25, 29, 30, 42, 44, 53],
        }
        self.datasets_by_split = self.get_dataset_names_by_split()
        # get proper label2id
        self.label2id = self.get_label2id()
        self.datasets: Dict[str, Dataset] = {}

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
            print("Loading {} {} datasets ...".format(len(names), mode))
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

    def _get_dataset_by_difficulty(self, difficulty: str) -> List[Dataset]:
        assert difficulty in self.datasets_grouped_by_difficulty.keys()
        return [
            self.get_dataset_by_name("rule_{}".format(id))
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
            "Data Split : {}, Number of Classes : {}, Number of Descriptors : {}, Average Resolution Length : {}, Average number of nodes : {}  and edges : {}".format(  # noqa: E501
                stat.split,
                stat.num_class,
                stat.num_des,
                stat.avg_resolution_length,
                stat.num_nodes,
                stat.num_edges,
            )
        )
        assert isinstance(stat, dict)
        return stat
