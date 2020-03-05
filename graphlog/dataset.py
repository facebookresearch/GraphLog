from typing import List

from torch import Tensor
from torch.utils.data import Dataset


class GraphLogDataset(Dataset):  # type: ignore
    """GraphLog dataset instance
    Contains all information about a single world
    We provide both the raw json view in order to perform quick
    analysis and visualization, and also the tensorized views based
    on Pytorch Geometric for easy training and evaluation.
    Returns:
        [type] -- [description]
    """

    def __init__(self, id: str):
        self.id = id
        self._data: List[Tensor]

    def __getitem__(self, index: int) -> Tensor:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)
