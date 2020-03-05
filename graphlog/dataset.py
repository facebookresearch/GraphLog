from typing import List

from torch import Tensor
from torch.utils.data import Dataset


class GraphDataset(Dataset):  # type: ignore
    def __init__(self, id: str):
        self.id = id
        self._data: List[Tensor]

    def __getitem__(self, index: int) -> Tensor:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)
