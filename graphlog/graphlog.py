import os
import json
from typing import Dict, List, Any

from graphlog.dataset import GraphLogDataset as Dataset
from graphlog.utils import download_and_extract_archive


class GraphLog:

    url = "https://www.cs.mcgill.ca/~ksinha4/data/graphlog_icml_data.zip"
    md5hash = "6aeda25d56251328dcdca0b23b4abe0a"
    data_filename = "comp_r10_n100_ov"

    supported_difficulty = ["easy", "moderate", "hard"]

    def __init__(self, data_dir: str = "./data/"):
        self.data_dir = data_dir
        self.download()
        self.datasets: List[Dataset] = self.load_datasets()
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
        # get proper label2id
        self.label2id = self.get_label2id()
        pass

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

    def get_label2id(self) -> Any:
        label2id_loc = os.path.join(
            self.data_dir, self.data_filename, "train", "label2id.json"
        )
        label2id = json.load(open(label2id_loc))
        return label2id

    def load_datasets(self) -> List[Dataset]:
        pass

    def get_dataset_ids(self) -> List[str]:
        return sorted(self.datasets_grouped_by_difficulty.keys())

    def get_dataset_by_id(self, id: int) -> Dataset:
        return self.datasets[id]

    def _get_dataset_by_difficulty(self, difficulty: str) -> List[Dataset]:
        assert difficulty in self.datasets_grouped_by_difficulty.keys()
        return [
            self.get_dataset_by_id(id)
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
