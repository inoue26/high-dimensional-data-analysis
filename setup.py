from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages


def _requires_from_file(filename:str|None=None):
    if filename is not None:
        return open(filename).read().splitlines()
    else:
        return []

setup(
    name="highdim",
    version="0.0.1",
    author="inoue26",
    url="git@github.com:inoue26/high-dimensional-data-analysis.git",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    package_data={"highdim": ["py.typed"]},
    install_requires=_requires_from_file(None),
)