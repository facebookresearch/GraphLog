# Evaluating Logical Generalization in Graph Neural Networks

Code for reproducing the experiments in _"Evaluating Logical Generalization in Graph Neural Networks"_, [Arxiv](https://arxiv.org/abs/2003.06560), which releases a multi-task synthetic graph benchmark data [GraphLog](https://github.com/facebookresearch/graphlog) built on first-order logic.

## Repository organization

- Entrypoint: [codes/app/main.py](codes/app/main.py)
- Config files : [config/](config/) directory
- Experiment files : [codes/experiment](codes/experiment/)
- Models : [codes/model](codes/model)
  - _Representation Functions_ : Param, GCN, GAT
  - _Composition Functions_ : RGCN, E-GAT

## Dependencies

- [PyTorch](https://pytorch.org/) 1.3.1
- [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) 1.3.2
- [GraphLog](https://github.com/facebookresearch/graphlog) 1.0

Other requirements can be found in [requirements.txt](requirements.txt)

## Dataset

- We use [GraphLog v1.0](https://github.com/facebookresearch/graphlog) dataset for all experiments.
- Raw files of the Dataset [can also be obtained here](https://drive.google.com/file/d/1nsVr-CXYouzrdiQUgqSbQLcfQJduTJyg/view?usp=sharing)
- Generator files will be made available in [GraphLog](https://github.com/facebookresearch/graphlog) repository soon.

## Training

- Config files can be placed either directly in [config](config/) directory, or in any subfolder. For example, if `supervised.yaml` config file resides in [config/supervised](config/supervised) directory,
the way to invoke training is:

  ```
  python codes/app/main.py --config_id supervised/supervised
  ```

- Sample configs of Supervised, Multitask and Continual learning are provided in `configs` folder.
- In config, `general.train_mode` supports different training modes, as can be seen in `checkpointable_multitask_experiment.py`
- Note that the models used in the experiments are modified to accept model weights as parameters instead of traditional `nn.Module` object attributes.

## Evaluation

- [scripts/eval_supervised.py](scripts/eval_supervised.py) contains the file to run supervised / multitask / continual learning evaluations. 
- Check [scripts/eval.sh](scripts/eval.sh) file for usage to evaluate individual tasks.

### Example evaluation script

```
python eval_supervised.py --config_path multitask/multitask_logic_hard_hard --eval_k_shot 0 --eval_load_epoch 1500 --eval_rules rule_45
```


## Statistics and Plots

All analysis in the paper is organized in various Jupyter Notebooks in [notebooks/](notebooks/) folder.

- Statistics regarding all the worlds in **GraphLog** can be found in [notebooks/GraphLogStats.ipynb](notebooks/GraphLogStats.ipynb) and [notebooks/clean_data/graphlog_stats.csv](notebooks/clean_data/graphlog_stats.csv)
- Multitask evaluation / dataset splits / preprocessing can be found in [notebooks/Multitask Similarity and Overlap Results.ipynb](notebooks/Multitask%20Similarity%20and%20Overlap%20Results.ipynb)
- Plots used in the paper can be found in [notebooks/Plots.ipynb](notebooks/Plots.ipynb)
- Unprocessed and processed results can be found in [notebooks/raw_data/](notebooks/raw_data/) and [notebooks/clean_data/](notebooks/clean_data/) respectively.


## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), as found in LICENSE file.
