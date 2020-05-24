#!/bin/sh
## Evaluate Sim-Sim - Zero
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_4,rule_8,rule_1
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_15,rule_20,rule_25
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_40,rule_45,rule_50
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 0 --eval_load_epoch 1900 --eval_rules rule_5,rule_7,rule_1,rule_6,rule_9,rule_3,rule_2,rule_10

## K-shot
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 5 --eval_load_epoch 1500 --eval_rules rule_4,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 5 --eval_load_epoch 1500 --eval_rules rule_8,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 5 --eval_load_epoch 1500 --eval_rules rule_1,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 5 --eval_load_epoch 1500 --eval_rules rule_15,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 5 --eval_load_epoch 1500 --eval_rules rule_20,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 5 --eval_load_epoch 1500 --eval_rules rule_25,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 5 --eval_load_epoch 1500 --eval_rules rule_40,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 5 --eval_load_epoch 1500 --eval_rules rule_45,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 5 --eval_load_epoch 1500 --eval_rules rule_50,

# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 10 --eval_load_epoch 1500 --eval_rules rule_4,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 10 --eval_load_epoch 1500 --eval_rules rule_8,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 10 --eval_load_epoch 1500 --eval_rules rule_1,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 10 --eval_load_epoch 1500 --eval_rules rule_15,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 10 --eval_load_epoch 1500 --eval_rules rule_20,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 10 --eval_load_epoch 1500 --eval_rules rule_25,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 10 --eval_load_epoch 1500 --eval_rules rule_40,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 10 --eval_load_epoch 1500 --eval_rules rule_45,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 10 --eval_load_epoch 1500 --eval_rules rule_50,

# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 15 --eval_load_epoch 1500 --eval_rules rule_4,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 15 --eval_load_epoch 1500 --eval_rules rule_8,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 15 --eval_load_epoch 1500 --eval_rules rule_1,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 15 --eval_load_epoch 1500 --eval_rules rule_15,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 15 --eval_load_epoch 1500 --eval_rules rule_20,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 15 --eval_load_epoch 1500 --eval_rules rule_25,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 15 --eval_load_epoch 1500 --eval_rules rule_40,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 15 --eval_load_epoch 1500 --eval_rules rule_45,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 15 --eval_load_epoch 1500 --eval_rules rule_50,

# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 20 --eval_load_epoch 1500 --eval_rules rule_4,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 20 --eval_load_epoch 1500 --eval_rules rule_8,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 20 --eval_load_epoch 1500 --eval_rules rule_1,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 20 --eval_load_epoch 1500 --eval_rules rule_15,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 20 --eval_load_epoch 1500 --eval_rules rule_20,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 20 --eval_load_epoch 1500 --eval_rules rule_25,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 20 --eval_load_epoch 1500 --eval_rules rule_40,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 20 --eval_load_epoch 1500 --eval_rules rule_45,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_sim --eval_k_shot 20 --eval_load_epoch 1500 --eval_rules rule_50,

## Evaluate Sim-Diss
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_dis --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_4,rule_8,rule_1
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_dis --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_15,rule_20,rule_23
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_dis --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_30,rule_31,rule_32
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_dis --eval_k_shot 0 --eval_load_epoch 1900 --eval_rules rule_5,rule_7,rule_1,rule_6,rule_9,rule_53,rule_50,rule_56

# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_dis --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_11,rule_15,rule_20
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_dis --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_15,rule_20,rule_25
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_dis --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_20,rule_25,rule_30
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_dis --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_40,rule_45,rule_50

# sim-diff = which is actually combination experiment
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_diff --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_4,rule_8,rule_1
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_diff --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_15,rule_20,rule_23
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_diff --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_30,rule_31,rule_32
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_diff --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_11,rule_15,rule_20
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_diff --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_15,rule_20,rule_25
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_diff --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_20,rule_25,rule_30
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_sim_diff --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_40,rule_45,rule_50

# diff-diff
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_diff_diff --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_10,rule_20,rule_30 --eval_data_folder "train"
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_diff_diff --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_54,rule_55,rule_56 --eval_data_folder "test"
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_diff_diff --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_6,rule_12,rule_25 --eval_data_folder "train"
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_diff_diff --eval_k_shot 0 --eval_load_epoch 1900 --eval_rules rule_0,rule_1,rule_2,rule_16,rule_17,rule_18,rule_32,rule_33,rule_34 --eval_data_folder "train"

# Difficulty
# multitask/multitask_logic_easy_easy
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 20 --eval_load_epoch 1900 --eval_rules rule_39,rule_38,rule_47
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 20 --eval_load_epoch 1900 --eval_rules rule_33,rule_32,rule_41
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 20 --eval_load_epoch 1900 --eval_rules rule_40,rule_42,rule_23
KSHOT=-1
## conv
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_54, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_55, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_56, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_54, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_55, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_56, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_54, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_55, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_56, --eval_data_folder "test"
# indivdual
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_36,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_47,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_24,
# #CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_45,rule_28,rule_11
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_45,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_28,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_11,

# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_30,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_2,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_16,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_easy_easy --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_34,rule_46,rule_37,rule_20,rule_48,rule_31,rule_35,rule_19

# # # multitask/multitask_logic_medium_medium
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 20 --eval_load_epoch 1900 --eval_rules rule_34,rule_46,rule_37
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 20 --eval_load_epoch 1900 --eval_rules rule_41,rule_21,rule_22
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 20 --eval_load_epoch 1900 --eval_rules rule_40,rule_42,rule_6
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_54, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_55, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_56, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_54, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_55, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_56, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_54, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_55, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_56, --eval_data_folder "test"


# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_45,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_28,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_11,
# # CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_47,rule_36,rule_24
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_47,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_36,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_24,
# # CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_30,rule_2,rule_16
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_30,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_2,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_16,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_med_med --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_18,rule_33,rule_32,rule_50,rule_8,rule_10,rule_7,rule_4

# # # multitask/multitask_logic_hard_hard
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 20 --eval_load_epoch 1900 --eval_rules rule_34,rule_37,rule_20
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 20 --eval_load_epoch 1900 --eval_rules rule_18,rule_33,rule_32
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 20 --eval_load_epoch 1900 --eval_rules rule_1,rule_3,rule_25
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_54, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_55, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_56, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_54, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_55, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_56, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_54, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_55, --eval_data_folder "test"
CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot $KSHOT --eval_load_epoch 1900 --eval_rules rule_56, --eval_data_folder "test"

# # CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_45,rule_28,rule_11
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_45,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_11,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_28,
# # CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_47,rule_36,rule_24
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_47,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_36,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_24,
# # CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_30,rule_2,rule_16
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_30,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_2,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_16,
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_40,rule_42,rule_6,rule_5,rule_23,rule_0,rule_15,rule_13

# k=0 learning composition fn
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_uc_ur --eval_k_shot 0 --eval_rules rule_10,rule_20,rule_30 --eval_data_folder "train"
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_uc_ur --eval_k_shot 0 --eval_rules rule_54,rule_55,rule_56 --eval_data_folder "test"
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_logic_uc_ur --eval_k_shot -1 --eval_rules rule_6,rule_12,rule_25 --eval_data_folder "train"


## Multitask Ablations
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_task_10 --eval_k_shot 0 --eval_rules rule_6,rule_24,rule_32,rule_27,rule_36,rule_48,rule_10,rule_13,rule_16,rule_35
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_task_20 --eval_k_shot 0 --eval_rules rule_42,rule_33,rule_32,rule_28,rule_1,rule_18,rule_43,rule_34,rule_45,rule_19,rule_24,rule_20,rule_27,rule_50,rule_44,rule_8,rule_37,rule_35,rule_4,rule_15
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_task_30 --eval_k_shot 0 --eval_rules rule_3,rule_6,rule_48,rule_38,rule_21,rule_35,rule_5,rule_45,rule_40,rule_16,rule_27,rule_44,rule_31,rule_12,rule_36,rule_46,rule_49,rule_7,rule_20,rule_1,rule_47,rule_29,rule_4,rule_42,rule_28,rule_43,rule_34,rule_33,rule_37,rule_15
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_task_40 --eval_k_shot 0 --eval_rules rule_43,rule_16,rule_38,rule_45,rule_48,rule_30,rule_36,rule_26,rule_8,rule_33,rule_18,rule_42,rule_15,rule_1,rule_7,rule_24,rule_27,rule_10,rule_44,rule_34,rule_50,rule_31,rule_25,rule_17,rule_19,rule_46,rule_23,rule_32,rule_3,rule_13,rule_11,rule_21,rule_37,rule_12,rule_22,rule_49,rule_5,rule_9,rule_39,rule_2
# CUDA_VISIBLE_DEVICES=1 python eval_supervised.py --config_path multitask/multitask_task_50 --eval_k_shot 0 --eval_rules rule_30,rule_41,rule_10,rule_13,rule_48,rule_4,rule_44,rule_25,rule_6,rule_28,rule_42,rule_31,rule_3,rule_46,rule_29,rule_21,rule_45,rule_7,rule_24,rule_35,rule_50,rule_1,rule_47,rule_34,rule_20,rule_26,rule_40,rule_16,rule_23,rule_2,rule_11,rule_37,rule_5,rule_33,rule_36,rule_43,rule_9,rule_14,rule_49,rule_22,rule_15,rule_39,rule_32,rule_0,rule_27,rule_8,rule_18,rule_12,rule_17,rule_19