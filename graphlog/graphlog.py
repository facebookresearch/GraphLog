import os
from typing import Dict, List

from graphlog.dataset import GraphDataset as Dataset
from graphlog.utils import download_and_extract_archive


class GraphLog:

    url = ""
    md5hash = ""
    data_filename = "data.jsonl"

    supported_difficulty = ["easy", "moderate", "hard"]

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.download()
        self.datasets: List[Dataset] = self.load_datasets()
        self.datasets_grouped_by_difficulty: Dict[str, List[int]] = {
            "easy": [1, 2, 3],
            "moderate": [5, 6],
            "hard": [7, 8],
        }
        pass

    def _get_datafile_path(self) -> str:
        return os.path.join(self.data_dir, "data.jsonl")

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

    def load_datasets(self) -> List[Dataset]:
        pass

    def get_dataset_ids(self) -> List[str]:
        pass

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
