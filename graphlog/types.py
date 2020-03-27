"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
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
