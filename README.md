---

# Cell cycle classification

<!-- [![codecov](https://codecov.io/gh/15bonte/cell_cycle_classification/branch/main/graph/badge.svg?token=cell_cycle_classification_token_here)](https://codecov.io/gh/15bonte/cell_cycle_classification) -->

<!-- [![CI](https://github.com/15bonte/cell_cycle_classification/actions/workflows/main.yml/badge.svg)](https://github.com/15bonte/cell_cycle_classification/actions/workflows/main.yml) -->
[![License BSD-3](https://img.shields.io/pypi/l/cut-detector.svg?color=green)](https://github.com/15bonte/cell_cycle_classification/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cell_cycle_classification.svg?color=green)](https://pypi.org/project/cell_cycle_classification)

Code associated to the paper "A Deep Learning approach for time-consistent cell cycle phase prediction from microscopy data" available on [bioRxiv].

![figure](architecture.png)

## Installation

### Conda environment

It is highly recommended to create a dedicated conda environment, by following these few steps:

1. Install an [Anaconda] distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.

2. Open an Anaconda prompt as admin to create a new environment using [conda]. We advice to use python 3.10 and conda 23.10.0, to get conda-libmamba-solver as default solver.

```
conda create --name cell_cycle_classification python=3.10 conda=23.10.0
conda activate cell_cycle_classification
```

### Package installation

Once in a dedicated environment, our package can be installed via [pip]:

```
pip install cell_cycle_classification
```

Alternatively, you can clone the github repo to access to notebooks.

```
git clone https://github.com/15bonte/cell_cycle_classification.git
cd cell_cycle_classification
pip install -e .
```

### GPU

We highly recommend to use GPU to speed up segmentation. To use your NVIDIA GPU, the first step is to download the dedicated driver from [NVIDIA].

Next we need to remove the CPU version of torch:

```bash
pip uninstall torch
```

The GPU version of torch to be installed can be found [here](https://pytorch.org/get-started/locally/). You may choose the CUDA version supported by your GPU, and install it with conda. This package has been developed with the version 11.6, installed with this command:

```bash
conda install pytorch==1.12.1 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```

## Update

To update cell_cycle_classification to the latest version, open an Anaconda prompt and use the following commands:

```bash
conda activate cell_cycle_classification
pip install cell_cycle_classification --upgrade
```

## Notebooks, models and data

Notebooks are provided as examples to demonstrate how to train and test our model. You can experiment with them using a subset of our data, available from [Zenodo]. Complete dataset is available on the [BioImage Archive].   

- Train: this notebook shows the training of both VAE and classification models.
- Eval VAE: this notebook evaluates image reconstruction accuracy and time consistency from our pretrained VAE model, available from [HuggingFace].
- Eval classifier: similarly, this notebook evaluates cell cycle classification accuracy from our pretrained classification model, available from [HuggingFace.]

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Release

Release is performed by updating the version number in VERSION and pushing the corresponding tag.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

## Citation

If you found our work useful, please consider citing:

 ```
@article{bonte2025deep,
  title={A Deep Learning approach for time-consistent cell cycle phase prediction from microscopy data},
  author={Bonte, Thomas and Pourcelot, Oriane and Safieddine, Adham and Slimani, Floric and Mueller, Florian and Weil, Dominique and Bertrand, Edouard and Walter, Thomas},
  journal={bioRxiv},
  pages={2025--05},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
 ```

[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[file an issue]: https://github.com/15bonte/cell_cycle_classification/issues
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[Anaconda]: https://www.anaconda.com/products/distribution
[NVIDIA]: https://www.nvidia.com/Download/index.aspx?lang=en-us
[conda]: https://docs.conda.io/en/latest/
[HuggingFace]: https://huggingface.co/thomas-bonte/cell_cycle_classification/tree/main/20241031-112222-4998324
[HuggingFace.]: https://huggingface.co/thomas-bonte/cell_cycle_classification/tree/main/20241101-055937-4998324
[Zenodo]: https://zenodo.org/records/14614787
[BioImage Archive]: https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1659
[bioRxiv]: https://www.biorxiv.org/content/10.1101/2025.05.16.654306v2.full.pdf
