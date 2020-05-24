"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
"""Function to get the config id"""
import argparse


def argument_parser():
    """Function to get the config id"""
    parser = argparse.ArgumentParser(
        description="Argument parser to obtain the name of the config file"
    )
    parser.add_argument("--config_id", default="sample_config", help="config id to use")
    args = parser.parse_args()
    return args.config_id
