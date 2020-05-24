"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
"""Code to bootstrap the metrics"""
from copy import deepcopy

from codes.utils.util import merge_nested_dicts


def get_default_metric_dict(mode, level="individual"):
    """Method to return a defualt metric dict"""
    _, keys_to_add, keys_to_replace = _get_metric_keys()

    metric_dict = {
        "mode": mode,
        "level": level,
    }

    if level == "individual":
        default = -1.0
    else:
        default = 0.0
    for key in keys_to_add + keys_to_replace:
        metric_dict[key] = default

    return metric_dict


def merge_metric_dicts(current_metric_dict, new_metric_dict):
    """Method to merge multiple metric dicts into one dict"""
    return merge_nested_dicts(current_metric_dict, new_metric_dict)


def prepare_metric_dict_to_log(current_metric_dict):
    """Method to prepare the metric dict before writing to the logger"""
    metric_dict = deepcopy(current_metric_dict)
    _, keys_to_add, _ = _get_metric_keys()
    total_num_key = "total_num"
    total_time_key = "total_time"
    total_num = metric_dict.pop(total_num_key) * 1.0
    keys_to_normalize = set(keys_to_add) - set([total_num_key, total_time_key])
    for key in keys_to_normalize:
        new_key = ("_".join(key.split("_")[1:])).strip("_")
        metric_dict[new_key] = metric_dict.pop(key) / total_num
    return metric_dict


def _get_metric_keys():
    """Method to obtain the different metric keys."""
    keys_to_replace = ["minibatch_idx", "epoch_idx"]

    # All these keys should start with "total".
    keys_to_add = ["total_num", "total_time", "total_correct", "total_loss"]

    keys_to_remain_fixed = ["mode", "level"]

    return keys_to_remain_fixed, keys_to_add, keys_to_replace


def merge_individual_metrics_into_aggregate_metrics(
    individual_metrics, aggregated_metrics
):
    """Method to merge the individual metric dict into aggregate metric dict"""

    _, keys_to_add, keys_to_replace = _get_metric_keys()

    for key in keys_to_replace:
        aggregated_metrics[key] = individual_metrics[key]

    for key in keys_to_add:
        aggregated_metrics[key] += individual_metrics[key]

    return aggregated_metrics
