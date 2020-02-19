
# GraphLog
API to interface with the GraphLog Dataset

### Setup

* `pip install .`

### Documentation



### Dev Setup

* `pip install -e .[dev] -e .`
* Install pre-commit hooks `pre-commit install`
* The code is linted using:
    * `black`
    * `flake8`
    * `mypy`
* Lint tests can be run locally using `nox -s lint`

### Acknowledgements

* Config for `circleci`, `pre-commit`, `mypy` etc are borrowed/modified from [Hydra](https://github.com/facebookresearch/hydra)