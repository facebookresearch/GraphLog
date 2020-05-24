"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
import yaml
import os
import argparse
from codes.utils.config import read_config_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="r1", type=str)
    parser.add_argument("--copy", default="debug", type=str)
    parser.add_argument("--rule", default="rule_1", type=str)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--load_epoch", default=99, type=int)
    parser.add_argument("--load_id", default="r1", type=str)
    parser.add_argument("--model", default="RelationGATClassifier", type=str)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = read_config_file(args.copy)
    config["general"]["id"] = args.id
    config["general"]["device"] = args.device
    config["general"]["data_name"] = args.rule
    config["model"]["name"] = args.model
    config["model"]["should_train"] = args.train
    config["model"]["should_load_model"] = args.load
    assert args.load_id != args.id
    load_path = "{}/current_model_epoch_{}.tar".format(args.load_id, args.load_epoch)
    config["model"]["load_path"] = load_path
    config["model"]["num_epochs"] = args.num_epochs
    path = os.path.dirname(os.path.realpath(__file__)).split("/codes")[0]
    config_name = "{}.yaml".format(args.id)
    config_path = os.path.join(path, "config", config_name)
    yaml.dump(config, open(config_path, "w"), default_flow_style=False)
