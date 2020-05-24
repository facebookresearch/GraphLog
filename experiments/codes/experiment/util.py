"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
import torch
from box import Box

from codes.model.pyg.models import (
    FixedParamSignature,
    FixedSignature,
    GatEncoder,
    LearnedParamSignature,
    LearnedSignature,
    NodeGatEncoder,
)


def select_signature_function(config: Box) -> torch.nn.Module:
    """Method to select the signature function"""
    if config.model.signature_gat.sig_fn_policy == "learn":
        model = LearnedSignature(config)
    elif config.model.signature_gat.sig_fn_policy == "learn-param":
        model = LearnedParamSignature(config)
    elif config.model.signature_gat.sig_fn_policy == "fixed":
        model = FixedSignature(config)
    elif config.model.signature_gat.sig_fn_policy == "fixed-param":
        model = FixedParamSignature(config)
    else:
        raise NotImplementedError("sig_fn_policy not implemented")

    return model
