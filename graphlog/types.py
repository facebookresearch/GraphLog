from typing import TYPE_CHECKING, Any, Dict, List, Union

from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    DataLoaderType = DataLoader[Tensor]
else:
    DataLoaderType = DataLoader


NumType = Union[int, float]
ValueType = Union[str, int, float]
StatType = Dict[str, ValueType]
GraphType = Any
WorldType = Dict[str, List[GraphType]]
