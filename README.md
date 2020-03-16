[![CircleCI](https://circleci.com/gh/fairinternal/GraphLog.svg?style=svg&circle-token=3de77dcba6da65107d3946878697d810251e00d9)](https://circleci.com/gh/fairinternal/GraphLog)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/graphlog)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# GraphLog
API to interface with the GraphLog Dataset. GraphLog is a multi-purpose, multi-relational graph dataset built using rules grounded in first-order logic.

* Homepage: https://www.cs.mcgill.ca/~ksinha4/graphlog/

<img src="docs/images/graphlog_rule.png" width="400">

### Installation

* Supported Python Version: 3.6+
* Install PyTorch from https://pytorch.org/get-started/locally/
* Install pytorch-geometric (and other dependencies) from https://github.com/rusty1s/pytorch_geometric#installation. Make sure that cpu/cuda versions for pytorch and pytorch-geometric etc matches.
* `pip install graphlog`

### QuickStart

Check out the notebooks on [Basic Usage](examples/Basic%20Usage.ipynb) and [Advanced Usage](examples/Advanced%20Usage.ipynb) to quickly start playing with GraphLog.

### Dev Setup

* `pip install -e ".[dev]"`
* Install pre-commit hooks `pre-commit install`
* The code is linted using:
    * `black`
    * `flake8`
    * `mypy`
* All the tests can be run locally using `nox`

### Questions

Please open an Issue!

### Contributing

Please open a Pull Request (PR).

### Citation

If our work is useful for your research, consider citing it using the following bibtex:

```
@article{sinha2020graphlog,
  Author = {Koustuv Sinha and Shagun Sodhani and Joelle Pineau and William L. Hamilton},
  Title = {Evaluating Logical Generalization in Graph Neural Networks},
  Year = {2020},
  arxiv = {}
}
```

### Licence

CC-BY-NC 4.0 (Attr Non-Commercial Inter.)
