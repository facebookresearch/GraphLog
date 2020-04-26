## Example Training Scripts

In this folder we provide example training scripts in [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) along with [Basic](Basic%20Usage.ipynb) and [Advanced Usage](Advanced%20Usage.ipynb) notebooks.

To run the training script, install [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) first:

```
pip install pytorch-lightning
```

Then, run using the `train.py` file:

```
python train.py --gpus 0 --train_world rule_0
```

- Please see the above notebooks for the detailed description on how to get available worlds.
- Currently, we provide a supervised example with a basic RGCN model for demonstration purposes. This model is not fine-tuned.
- We provide an example of supervised training. Extending this to multitask / continual learning settings should be trivial. Contributions are welcome!