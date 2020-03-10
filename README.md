
# GraphLog
API to interface with the GraphLog Dataset

### Installation

* Supported Python Version: 3.6+
* Install PyTorch from https://pytorch.org/get-started/locally/
* Install pytorch-geometric (and other dependencies) from https://github.com/rusty1s/pytorch_geometric#installation. Make sure that cpu/cuda versions for pytorch and pytorch-geometric etc matches.
* `pip install graphlog==1.0.0rc3`

### Documentation


### Use


### Dev Setup

* `pip install -e ".[dev] -f https://download.pytorch.org/whl/torch_stable.html -f https://pytorch-geometric.com/whl/torch-1.4.0.html"`
* Install pre-commit hooks `pre-commit install`
* The code is linted using:
    * `black`
    * `flake8`
    * `mypy`
* All the tests can be run locally using `nox`
