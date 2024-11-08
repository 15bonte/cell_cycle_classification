"""Python setup.py for cell_cycle_classification package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("cell_cycle_classification", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="cell_cycle_classification",
    version=read("cell_cycle_classification", "VERSION"),
    description="Awesome cell_cycle_classification created by 15bonte",
    url="https://github.com/15bonte/cell_cycle_classification/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="15bonte",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["cell_cycle_classification = cell_cycle_classification.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
