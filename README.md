
# GraphLog
API to interface with the GraphLog Dataset

### Installation

* `git clone git@github.com:fairinternal/GraphLog`
* `cd GraphLog`
* `pip install . -f https://pytorch-geometric.com/whl/torch-1.4.0.html`

Alternatively, `pip install git+ssh://git@github.com/fairinternal/GraphLog.git@master#egg=graphlog -f https://pytorch-geometric.com/whl/torch-1.4.0.html`

### Documentation


### Use







### Dev Setup

* `pip install -e ".[dev]"`
* Install pre-commit hooks `pre-commit install`
* The code is linted using:
    * `black`
    * `flake8`
    * `mypy`
* All the tests can be run locally using `nox`