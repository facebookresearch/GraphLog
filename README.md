[![CircleCI](https://circleci.com/gh/facebookresearch/GraphLog.svg?style=svg&circle-token=3de77dcba6da65107d3946878697d810251e00d9)](https://circleci.com/gh/facebookresearch/GraphLog)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/graphlog)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# GraphLog
API to interface with the GraphLog Dataset. GraphLog is a multi-purpose, multi-relational graph dataset built using rules grounded in first-order logic.

[Homepage](https://www.cs.mcgill.ca/~ksinha4/graphlog/) | [Paper](https://arxiv.org/abs/2003.06560) | [API Docs](https://graphlog.readthedocs.io/en/latest/)

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

### Experiments

Code for experiments used in our paper are available in `experiments/` folder.

### Questions

- If you have questions, open an Issue
- Or, [join our Slack channel](https://join.slack.com/t/logicalml/shared_invite/zt-e7osm7j7-vfIRgJAbEHxYN5D70njvyw) and post your questions / comments!
- To contribute, open a Pull Request (PR)

### Contributing

Please open a Pull Request (PR).

### Citation

If our work is useful for your research, consider citing it using the following bibtex:

```
@article{sinha2020graphlog,
  Author = {Koustuv Sinha and Shagun Sodhani and Joelle Pineau and William L. Hamilton},
  Title = {Evaluating Logical Generalization in Graph Neural Networks},
  Year = {2020},
  arxiv = {https://arxiv.org/abs/2003.06560}
}
```

### License

CC-BY-NC 4.0 (Attr Non-Commercial Inter.)

### Terms of Use

https://opensource.facebook.com/legal/terms

### Privacy Policy

https://opensource.facebook.com/legal/privacy
