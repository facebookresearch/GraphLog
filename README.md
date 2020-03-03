
# GraphLog
API to interface with the GraphLog Dataset

### Installation

* `git clone git@github.com:fairinternal/GraphLog`
* `cd GraphLog`
* `pip install .`

Alternatively, `pip install git+ssh://git@github.com/fairinternal/GraphLog.git@master#egg=graphlog`

### Use




### Documentation


### Dev Setup

* `pip install -e .[dev] -e .`
* Install pre-commit hooks `pre-commit install`
* The code is linted using:
    * `black`
    * `flake8`
    * `mypy`
* CI currently only checks for linting. 
* Lint tests can be run locally using `nox -s lint`
