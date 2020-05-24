"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
## Script to evaluate the supervised setups

from codes.testtube.checkpointable_testube import (
    CheckpointableTestTube,
    bootstrap_config,
)
from codes.utils.util import set_seed
import glob
import os
import pandas as pd
import json
import copy
from lgw.args import get_args
from tabulate import tabulate
import yaml
import pickle as pkl
import numpy as np

yaml.warnings({"YAMLLoadWarning": False})

# result row
row = {
    "rep_fn": "",
    "comp_fn": "",
    "mode": "",
    "test_rule": "",
    "updates": 0,
    "accuracy": 0,
}


def eval(
    big_config,
    k=0,
    modified_config=None,
    mode="hard",
    data_k=-1,
    seed=-1,
    load_epoch=None,
    eval_rules=None,
    data_folder="train",
    store_rep=False,
):
    results_table = []
    for exp in big_config["representation"][mode]:
        print("evaluating experiment {}".format(json.dumps(exp)))
        res_row = copy.deepcopy(row)
        res_row["rep_fn"] = exp[0]
        res_row["comp_fn"] = exp[1]
        ev_exp = CheckpointableTestTube(config_id=exp[-1], seed=seed)
        if "use_representation_fn" not in ev_exp.config.model:
            ev_exp.config.model.use_representation_fn = False
        if modified_config:
            for key, val in modified_config.items():
                print(
                    "modifying {} from {} to {}".format(
                        key, ev_exp.config.model[key], val
                    )
                )
                ev_exp.config.model[key] = val
        if ev_exp.config.general.train_mode == "run_mult_unique_comp":
            ev_exp.config.model.use_representation_fn = True
            ev_exp.config.model.freeze_representation_fn = True
            ev_exp.config.model.use_composition_fn = False
            ev_exp.config.model.freeze_composition_fn = False
        if ev_exp.config.general.train_mode == "run_mult_unique_rep":
            ev_exp.config.model.use_representation_fn = False
            ev_exp.config.model.freeze_representation_fn = False
            ev_exp.config.model.use_composition_fn = True
            ev_exp.config.model.freeze_composition_fn = True
        if "_conv_" in exp[-1]:
            # set the test to the train rule
            print(
                "Detected k=conv run, setting test rule to the train rule {}".format(
                    ev_exp.config.general.train_rule
                )
            )
            ev_exp.config.general.test_rule = ev_exp.config.general.train_rule + ","
        elif eval_rules and len(eval_rules) > 0:
            print("setting test rules to {}".format(eval_rules))
            ev_exp.config.general.test_rule = eval_rules
        ev_exp.prepare_evaluator(epoch=load_epoch, override_mode=data_folder)
        if k != 0:
            ev_exp.evaluator.adapt(k=k, report=True, eps=0.1, data_k=data_k)
        pr = ev_exp.evaluator.evaluate()
        if store_rep:
            _, reps = ev_exp.evaluator.evaluate_representations()
            pkl.dump(reps, open("rep_{}.pkl".format(exp[-1].replace("/", "-")), "wb"))
        # pr_ale = ev_exp.evaluator.evaluate(ale_mode=True)
        res_row["mode"] = mode
        res_row.update(pr)
        # res_row["accuracy_ale"] = pr_ale["accuracy"]
        # res_row["loss_ale"] = pr_ale["loss"]
        # res_row["rule_world_ale"] = pr_ale["rule_world"]
        results_table.append(res_row)
    return pd.DataFrame(results_table)


def eval_continual(big_config, seed=-1, mode="hard"):
    """Evaluation of continual learning experiments
    """
    results_table = []
    # load all test data
    ev_exp = CheckpointableTestTube(
        config_id=big_config["representation"][mode][0][-1], seed=seed
    )
    ev_exp.config.general.test_rule = "rule_0"
    all_test_data = ev_exp.initialize_data(mode="test", override_mode="train")
    label2id = ev_exp.label2id
    del ev_exp
    for exp in big_config["representation"][mode]:
        print("evaluating experiment {}".format(json.dumps(exp)))
        res_row = copy.deepcopy(row)
        res_row["rep_fn"] = exp[0]
        res_row["comp_fn"] = exp[1]
        ev_exp = CheckpointableTestTube(config_id=exp[-1], seed=seed)
        # perform the modification with the worlds here
        train_worlds = ev_exp.config.general.train_rule.split(",")
        print("evaluating on {} train worlds".format(len(train_worlds)))
        for wi, current_world in enumerate(train_worlds):
            ev_exp = CheckpointableTestTube(config_id=exp[-1], seed=seed)
            ev_exp.config.general.test_rule = current_world + ","
            ev_exp.prepare_evaluator(
                epoch=wi,
                override_mode="train",
                test_data=all_test_data,
                label2id=label2id,
            )
            pr_current = ev_exp.evaluator.evaluate()
            if wi > 0:
                ev_exp.config.general.test_rule = ",".join(train_worlds[:wi]) + ","
                print("loading {} test worlds".format(len(train_worlds[:wi])))
                ev_exp.prepare_evaluator(
                    epoch=wi,
                    override_mode="train",
                    test_data=all_test_data,
                    label2id=label2id,
                )
                pr_past = ev_exp.evaluator.evaluate()
            else:
                pr_past = {}
            res_row["current_world"] = current_world
            res_row["accuracy"] = pr_current["accuracy"]
            if len(pr_past) > 0:
                res_row["past_accuracy"] = pr_past["accuracy"]
                res_row["acc_std"] = pr_past["acc_std"]
            results_table.append(copy.deepcopy(res_row))
    return pd.DataFrame(results_table)


def eval_continual_comp(big_config, seed=-1, mode="hard"):
    """Evaluation of continual learning with unique composition functions
    """
    results_table = []
    # load all test data
    ev_exp = CheckpointableTestTube(
        config_id=big_config["representation"][mode][0][-1], seed=seed
    )
    ev_exp.config.general.test_rule = "rule_0"
    all_test_data = ev_exp.initialize_data(mode="test", override_mode="train")
    label2id = ev_exp.label2id
    del ev_exp
    for exp in big_config["representation"][mode]:
        print("evaluating experiment {}".format(json.dumps(exp)))
        res_row = copy.deepcopy(row)
        res_row["rep_fn"] = exp[0]
        res_row["comp_fn"] = exp[1]
        ev_exp = CheckpointableTestTube(config_id=exp[-1], seed=seed)
        # perform the modification with the worlds here
        train_worlds = ev_exp.config.general.train_rule.split(",")
        print("evaluating on {} train worlds".format(len(train_worlds)))
        for wi, current_world in enumerate(train_worlds):
            ev_exp = CheckpointableTestTube(config_id=exp[-1], seed=seed)
            ev_exp.config.general.test_rule = current_world + ","
            ev_exp.prepare_evaluator(
                epoch=wi,
                override_mode="train",
                test_data=all_test_data,
                label2id=label2id,
            )
            pr_current = ev_exp.evaluator.evaluate()
            if wi > 0:
                pr_past = {}
                pr_past_w = []
                print("loading {} test worlds".format(len(train_worlds[:wi])))
                for pi, past_world in enumerate(train_worlds[:wi]):
                    ev_exp = CheckpointableTestTube(config_id=exp[-1], seed=seed)
                    ev_exp.config.general.test_rule = past_world + ","
                    # load current rep function
                    ev_exp.prepare_evaluator(
                        epoch=wi,
                        override_mode="train",
                        test_data=all_test_data,
                        label2id=label2id,
                    )
                    # load past comp function
                    # ev_exp.config.model.use_composition_fn = True
                    ev_exp.config.model.use_representation_fn = True
                    ev_exp.evaluator.reset(epoch=pi)
                    # eval
                    pr_past_w.append(ev_exp.evaluator.evaluate())
                pr_past["accuracy"] = np.mean([pw["accuracy"] for pw in pr_past_w])
                pr_past["acc_std"] = np.mean([pw["acc_std"] for pw in pr_past_w])
            else:
                pr_past = {}
            res_row["current_world"] = current_world
            res_row["accuracy"] = pr_current["accuracy"]
            if len(pr_past) > 0:
                res_row["past_accuracy"] = pr_past["accuracy"]
                res_row["acc_std"] = pr_past["acc_std"]
            results_table.append(copy.deepcopy(res_row))
    return pd.DataFrame(results_table)


def clean_table(results):
    clean_model_names = {
        "GatedNodeGatEncoder": "GAT",
        "GatedGatEncoder": "E-GAT",
        "RepresentationGCNEncoder": "GCN",
        "CompositionRGCNEncoder": "RGCN",
        "Param": "Param",
    }
    results.rep_fn = results.rep_fn.apply(lambda x: clean_model_names[x])
    results.comp_fn = results.comp_fn.apply(lambda x: clean_model_names[x])
    results.accuracy = results.accuracy.apply(lambda x: round(x, 3))
    results.acc_std = results.acc_std.apply(lambda x: round(x, 3))
    # results.accuracy_ale = results.accuracy_ale.apply(lambda x: round(x, 3))
    if "loss" in results.columns:
        results.loss = results.loss.apply(lambda x: round(x, 3))
    if "past_accuracy" in results.columns:
        results.past_accuracy = results.past_accuracy.apply(lambda x: round(x, 3))
    # results.loss_ale = results.loss_ale.apply(lambda x: round(x, 3))
    return results


if __name__ == "__main__":
    args = get_args()
    config_path = "/private/home/koustuvs/mlp/lgw/config/"
    config_id = args.config_path + "*"
    # config_id = 'multitask/multitask_comp_seq*'
    files = glob.glob(os.path.join(config_path, config_id))
    config_ids = ["/".join(fl.split("/")[-2:]).split(".yaml")[0] for fl in files]
    print("Found {} configs".format(len(config_ids)))
    config_dicts = [bootstrap_config(ci) for ci in config_ids]

    ## consolidate ids in one place
    big_config = {
        "composition": {"easy": [], "hard": []},
        "representation": {"easy": [], "hard": []},
    }
    for ci, cd in enumerate(config_dicts):
        data_type = (
            "composition" if "composition" in cd.general.data_name else "representation"
        )
        mode = "easy" if "easy" in cd.general.data_name else "hard"
        comp_fn = cd.model.composition_fn_path.split(".")[-1]
        rep_fn = cd.model.representation_fn_path.split(".")[-1]
        big_config[data_type][mode].append((rep_fn, comp_fn, config_ids[ci]))
    if len(args.config_toggle_true) > 0:
        keys = args.config_toggle_true.split(",")
        modified_config = {k: True for k in keys}
    else:
        modified_config = None
    stat_file = (
        args.config_path.split("/")[-1] + "_k{}".format(args.eval_k_shot) + args.output
    )
    if os.path.exists(stat_file):
        result_old = pd.read_csv(stat_file)
    else:
        result_old = None
    # evaluate
    results = clean_table(
        eval(
            big_config,
            k=args.eval_k_shot,
            modified_config=modified_config,
            mode=args.eval_data_mode,
            data_k=args.eval_k_epoch,
            load_epoch=args.eval_load_epoch if args.eval_load_epoch >= 0 else None,
            eval_rules=args.eval_rules,
            data_folder=args.eval_data_folder,
            store_rep=args.eval_store_rep,
        )
    )
    # results = clean_table(eval_continual(big_config))
    # results = clean_table(eval_continual_comp(big_config))
    if result_old is not None:
        results = pd.concat([result_old, results])
        results.reset_index(drop=True, inplace=True)
    results.to_csv(stat_file)
    print(tabulate(results, tablefmt="github", headers="keys"))
