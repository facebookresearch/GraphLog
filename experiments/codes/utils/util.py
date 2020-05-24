"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
"""Collection of utility functions"""

# pylint: disable=C0103

import collections
import gc
import os
import pathlib
import random
import re
import subprocess
from functools import reduce, wraps
from operator import mul
from time import time
import importlib

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from box import Box


def flatten_dict(d, parent_key="", sep="#"):
    """Method to flatten a given dict using the given seperator.
    Taken from https://stackoverflow.com/a/6027615/1353861
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(dictionary, sep="#"):
    """Method to flatten a given dict using the given seperator.
        Taken from https://stackoverflow.com/questions/6037503/python-unflatten-dict
        """
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(sep)
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def grouped(iterable, n):
    """Modified from
    https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list/39038787"""
    return zip(*[iter(iterable)] * n)


def padarray(A, size, const=1):
    """Taken from
    https://stackoverflow.com/questions/38191855/zero-pad-numpy-array/38192105"""
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode="constant", constant_values=const)


def parse_file(file_name):
    """Method to read the given input file and return an iterable for the lines"""
    with open(file_name, encoding="utf-8") as f:
        for line in f:
            yield line


def shuffle_list(*ls):
    """Taken from
    https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order"""
    l = list(zip(*ls))
    random.shuffle(l)
    return zip(*l)


def chunks(l, n):
    """
    Taken from
    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def reverse_dict(_dict):
    """Reverse Dict"""
    return {v: k for k, v in _dict.items()}


def make_dir(path):
    """Make dir"""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def get_device_name(device_type):
    """Get device name"""
    if torch.cuda.is_available() and "cuda" in device_type:
        return device_type
    return "cpu"


def get_current_commit_id():
    """Get current commit id"""
    command = "git rev-parse HEAD"
    commit_id = subprocess.check_output(command.split()).strip().decode("utf-8")
    return commit_id


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def timing(f):
    """Timing function"""

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("function:{} took: {} sec".format(f.__name__, te - ts))
        return result

    return wrap


def show_tensor_as_image(_tensor):
    """Plot a tensor as image"""
    plt.imshow(_tensor.astype(np.uint8), origin="lower")
    plt.show()


def save_tensor_as_image(_tensor, file_path):
    """Save a tensor as image """
    plt.imsave(file_path, _tensor.astype(np.uint8), origin="lower")


def get_product_of_iterable(_iterable):
    """Method to get the product of all the enteries in an iterable"""
    return reduce(mul, _iterable, 1)


def log_pdf(x, mu, std):
    """Method to compute the log pdf for x under a gaussian
    distribution with mean = mu and standard deviation = std
    Taken from: https://chrisorm.github.io/VI-MC-PYT.html"""

    return -0.5 * torch.log(2 * np.pi * std ** 2) - (
        0.5 * (1 / (std ** 2)) * (x - mu) ** 2
    )  # pylint: disable=no-member


def running_mean(x, N):
    """Taken from
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean"""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def merge_first_two_dims(batch):
    """Merge first two dims in a batch"""
    shape = batch.shape
    return batch.view(-1, *shape[2:])


def unmerge_first_and_second_dim(batch, first_dim=-1, second_dim=-1):
    """Method to modify the shape of a batch by unmerging the first dimension.
    Given a tensor of shape (a*b, c, ...), return a tensor of shape (a, b, c, ...)"""
    shape = batch.shape
    return batch.view(first_dim, second_dim, *shape[1:])


def merge_second_and_third_dim(batch):
    """Merge the second and the third dims in a batch.
    Used when flattening messages from the primitives ot the master"""
    shape = batch.shape
    return batch.view(shape[0], int(shape[1] * shape[2]), *shape[3:])


def unmerge_second_and_third_dim(batch, second_dim=-1, third_dim=-1):
    """Method to modify the shape of a batch by unmerging the second and the third dimension.
    Given a tensor of shape (a, b*c, ...), return a tensor of shape (a, b, c, ...)"""
    shape = batch.shape
    return batch.view(second_dim, third_dim, *shape[1:])


def _get_box(_dict, frozen_box=False):
    """Wrapper to get a box"""
    return Box(
        _dict, default_box_attr=None, box_duplicates="ignore", frozen_box=frozen_box
    )


def get_box(_dict):
    """Wrapper to get a box"""
    return _get_box(_dict, frozen_box=False)


def get_forzen_box(_dict):
    """Wrapper to get a frozen box"""
    return _get_box(_dict, frozen_box=True)


def merge_nested_dicts(dict1, dict2):
    """Method to merge dict1 and dict2.
        dict1 is over written
        """
    sep = "#"
    flattened_dict1 = flatten_dict(dict1, sep=sep)
    flattened_dict2 = flatten_dict(dict2, sep=sep)
    flattened_merged_dict = {**flattened_dict1, **flattened_dict2}
    return unflatten_dict(flattened_merged_dict, sep=sep)


def map_observation_space_to_shape(obs):
    """Method to obtain the shape from observation space"""

    if obs.__class__.__name__ == "Discrete":
        return (obs.n,)
    if obs.__class__.__name__ == "Box":
        return obs.shape
    return obs.shape


def split_on_caps(input_str):
    """Method to split a given string at uppercase characters.
    Taken from: https://stackoverflow.com/questions/2277352/split-a-string-at-uppercase-letters
    """
    return re.findall("[A-Z][^A-Z]*", input_str)


def print_mem_report():
    """Method to print the memory usage by different tensors.
    Taken from the pytorch forums."""
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


def get_cpu_stats():
    """Method to return/print the CPU stats. Taken from pytorch forums """
    cpu_percent = psutil.cpu_percent()
    virtual_memory = psutil.virtual_memory()
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / (2.0 ** 30)  # memory use in GB...I think
    # print('memory GB:', memoryUse)
    return dict(
        cpu_percent=cpu_percent, virtual_memory=virtual_memory, memory_use=memory_use
    )


def _import_module(full_module_name):
    """
    Import className from python file
    https://stackoverflow.com/a/8790232
    :param full_module_name: full resolvable module name
    :return: module
    """
    path, name = full_module_name.rsplit(".", 1)
    base_module = importlib.import_module(path)
    module = getattr(base_module, name)
    return module


def get_main_working_dir(project_name="lgw"):
    """
    Get main project directory
    :param project_name:
    :return:
    """
    return os.path.join(os.getcwd().split(project_name)[0], project_name)


def get_param_id(param_name_dict, partial_name, ignore_classify=True):
    """Get param id
    
    Arguments:
        param_name_dict {[type]} -- [description]
        partial_name {[type]} -- [description]
    
    Raises:
        AssertionError: [description]
    
    Returns:
        [type] -- [description]
    """
    match = [k for k, v in param_name_dict.items() if partial_name in k]
    if ignore_classify:
        match = [k for k in match if "classify" not in k]
    assert len(match) == 1
    name = match[0]
    if name not in param_name_dict:
        raise AssertionError("arg not found")
    return param_name_dict[name]


def get_param(params, param_name_dict, partial_name, ignore_classify=True, eps=0.00001):
    """Get params from the param list
    if two matches are found, check if weight normalized variables
    combine and return the param
    
    Arguments:
        params {list} -- list of params
        param_name_dict {dict} -- dict of name:location_id
        partial_name {str} -- partial name to search
    
    Raises:
        AssertionError: ["arg not found"]
    
    Returns:
        [tensor] -- [parameter tensor]
    """
    match = [k for k, v in param_name_dict.items() if partial_name in k]
    if ignore_classify:
        match = [k for k in match if "classify" not in k]
    if len(match) == 1:
        name = match[0]
        return params[param_name_dict[name]]
    elif len(match) == 2:
        # check if weight norm version
        # names should be "{layername}.{layer}.{param_name}_{g/v}"
        p1_head = "_".join(match[0].split("_")[:-1])
        p2_head = "_".join(match[1].split("_")[:-1])
        assert p1_head == p2_head
        name_d = {m.split("_")[-1]: m for m in match}
        param_g = name_d["g"]
        param_v = name_d["v"]
        return params[param_name_dict[param_v]] * (
            params[param_name_dict[param_g]]
            / (torch.norm(params[param_name_dict[param_v]]) + eps)
        ).expand_as(params[param_name_dict[param_v]])
    else:
        raise AssertionError("arg not found")


def prepare_param_idx(weight_names, layer_id=0):
    """prepare param index dict
    
    Arguments:
        weight_names {list} -- list of weight names
    
    Keyword Arguments:
        layer_id {int} -- layer id (default: {0})
    
    Returns:
        [dict] -- dict of param_name:id
    """
    full_name_idx = {n: i for i, n in enumerate(weight_names)}
    gat_layer_param_indx = [
        i for i, k in enumerate(weight_names) if "{}".format(layer_id) in k
    ]
    param_names = [weight_names[gi] for gi in gat_layer_param_indx]
    param_name_to_idx = {k: full_name_idx[k] for v, k in enumerate(param_names)}
    return param_name_to_idx


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]
