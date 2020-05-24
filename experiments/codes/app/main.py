"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
"""This is the main entry point for the code"""

from codes.testtube.checkpointable_testtube import CheckpointableTestTube
from codes.utils.argument_parser import argument_parser
from codes.utils.util import timing


@timing
def run(config_id: str) -> None:
    """Run the code"""

    testtube = CheckpointableTestTube(config_id)
    testtube.run()


@timing
def evaluate(config_id: str, epoch=None) -> None:
    """ Evaluate on saved models
    """
    testtube = CheckpointableTestTube(config_id, load_checkpoint=False)
    testtube.evaluate(epoch=epoch)


if __name__ == "__main__":
    # config_id = "maml/maml_exp_icml_6"
    # config_id = "multitask/multitask_uc_icml_5"
    # run(config_id=config_id)
    # evaluate(config_id=config_id)
    run(config_id=argument_parser())
    # evaluate(config_id=argument_parser())
