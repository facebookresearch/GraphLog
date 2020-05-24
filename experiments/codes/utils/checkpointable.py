"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import os
from typing import Iterable, Optional, Tuple
from codes.utils.util import make_dir


class Checkpointable:
    """This class provides two methods: (i) save=, (ii) load"""

    def save(
        self,
        save_dir,
        epoch: Optional[int] = None,
        composition_fn: Optional[nn.Module] = None,
        representation_fn: Optional[nn.Module] = None,
        optimizers: Iterable[torch.optim.Optimizer] = None,
    ) -> None:
        """Persist the model weights in disk
        
        Arguments:
            save_dir {str} -- save location
        
        Keyword Arguments:
            epoch {Optional[int]} -- [description] (default: {None})
            composition_fn {Optional[nn.Module]} -- Graph Function (default: {None})
            representation_fn {Optional[nn.Module]} -- previously signature function (default: {None})
            optimizers {Iterable[torch.optim.Optimizer]} -- [description] (default: {None})
        """
        if epoch is None:
            epoch = 0
        path_to_save_model_at = os.path.join(save_dir, str(epoch),)
        make_dir(path_to_save_model_at)
        data = {
            "composition_fn": {
                "weights": composition_fn.weights,
                "weight_names": composition_fn.weight_names,
            },
            "representation_fn": {
                "weights": representation_fn.weights,
                "weight_names": representation_fn.weight_names,
            },
            "optimizers": [opt.state_dict() for opt in optimizers],
        }
        torch.save(data, os.path.join(path_to_save_model_at, "model.pt"))
        data = {"epoch": epoch}
        path_to_save_metadata_at = os.path.join(save_dir,)
        make_dir(path_to_save_metadata_at)
        torch.save(data, os.path.join(path_to_save_metadata_at, "metadata.pt"))

    def load(
        self,
        save_dir,
        epoch=None,
        composition_fn: Optional[nn.Module] = None,
        representation_fn: Optional[nn.Module] = None,
        optimizers: Iterable[torch.optim.Optimizer] = None,
    ) -> Tuple[nn.Module, nn.Module, Iterable[torch.optim.Optimizer], int]:
        """Load the given object from the filesystem
        
        Arguments:
            save_dir {[type]} -- [description]
        
        Keyword Arguments:
            epoch {[type]} -- [description] (default: {None})
            composition_fn {Optional[nn.Module]} -- Graph Function (default: {None})
            representation_fn {Optional[nn.Module]} -- previously signature function (default: {None})
            optimizers {Iterable[torch.optim.Optimizer]} -- [description] (default: {None})
        
        Returns:
            Tuple[nn.Module, nn.Module, Iterable[torch.optim.Optimizer]] -- [description]
        """
        # Check if there is anything to load

        path_to_load_metadata_from = os.path.join(save_dir, "metadata.pt")
        if os.path.exists(path_to_load_metadata_from):
            print("loading metadata from {}".format(path_to_load_metadata_from))
            meta_data = torch.load(path_to_load_metadata_from)
            print("loading epoch : {}".format(epoch))
            if epoch is None:
                epoch = meta_data["epoch"]
                print("loading from last saved epoch {}".format(epoch))

            path_to_load_model_from = os.path.join(save_dir, str(epoch), "model.pt")
            print("loading model from {}".format(path_to_load_model_from))
            if os.path.exists(path_to_load_model_from):
                data = torch.load(path_to_load_model_from)
                if composition_fn:
                    composition_fn.set_weights(data["composition_fn"]["weights"])
                    composition_fn.weight_names = data["composition_fn"]["weight_names"]
                if representation_fn:
                    representation_fn.set_weights(data["representation_fn"]["weights"])
                    representation_fn.weight_names = data["representation_fn"][
                        "weight_names"
                    ]
                if optimizers and len(optimizers) > 0:
                    for i, opt in enumerate(data["optimizers"]):
                        optimizers[i].load_state_dict(opt)
        return composition_fn, representation_fn, optimizers, epoch
