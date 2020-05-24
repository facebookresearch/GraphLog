#!/usr/bin/env sh

python prepare_data.py --num_rel_choices "5" --graphs_per_world 5000 --num_worlds 1 --per_inverse_choices "0.3" --corrupt_eps_choices "0" --expand_steps_choices "8"
python prepare_data.py --num_rel_choices "5" --graphs_per_world 5000 --num_worlds 1 --per_inverse_choices "0.5" --corrupt_eps_choices "0" --expand_steps_choices "8"
python prepare_data.py --num_rel_choices "5" --graphs_per_world 5000 --num_worlds 1 --per_inverse_choices "0.7" --corrupt_eps_choices "0" --expand_steps_choices "8"
python prepare_data.py --num_rel_choices "5" --graphs_per_world 5000 --num_worlds 1 --per_inverse_choices "0.3" --corrupt_eps_choices "0.2" --expand_steps_choices "8"
python prepare_data.py --num_rel_choices "5" --graphs_per_world 5000 --num_worlds 1 --per_inverse_choices "0.5" --corrupt_eps_choices "0.2" --expand_steps_choices "8"
python prepare_data.py --num_rel_choices "5" --graphs_per_world 5000 --num_worlds 1 --per_inverse_choices "0.7" --corrupt_eps_choices "0.2" --expand_steps_choices "8"
python prepare_data.py --num_rel_choices "5" --graphs_per_world 5000 --num_worlds 1 --per_inverse_choices "0.9" --corrupt_eps_choices "0" --expand_steps_choices "8"
#python prepare_data.py --num_rel_choices "15" --graphs_per_world 5000 --num_worlds 1 --per_inverse_choices "0.5" --corrupt_eps_choices "0" --expand_steps_choices "7"
#python prepare_data.py --num_rel_choices "15" --graphs_per_world 5000 --num_worlds 1 --per_inverse_choices "0.7" --corrupt_eps_choices "0" --expand_steps_choices "7"
#python prepare_data.py --num_rel_choices "15" --graphs_per_world 5000 --num_worlds 1 --per_inverse_choices "0.3" --corrupt_eps_choices "0.2" --expand_steps_choices "7"
#python prepare_data.py --num_rel_choices "15" --graphs_per_world 5000 --num_worlds 1 --per_inverse_choices "0.5" --corrupt_eps_choices "0.2" --expand_steps_choices "7"
#python prepare_data.py --num_rel_choices "15" --graphs_per_world 5000 --num_worlds 1 --per_inverse_choices "0.7" --corrupt_eps_choices "0.2" --expand_steps_choices "7"

