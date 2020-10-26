"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements/base.txt") as f:
    base_requirements = f.read().splitlines()

with open("requirements/dev.txt") as f:
    dev_requirements = f.read().splitlines()

setuptools.setup(
    name="graphlog",
    version="1.0.0",
    author="Koustuv Sinha and Shagun Sodhani",
    author_email="sshagunsodhani@gmail.com",
    description="API to interface with the GraphLog Dataset",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=base_requirements,
    url="",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "docs", "docsrc"]
    ),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    # Install development dependencies with
    # pip install -e .[dev]
    extras_require={"dev": dev_requirements + base_requirements},
)
