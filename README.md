# VISEPOCH

A package for visualising [EPOCH](https://github.com/Warwick-Plasma/epoch) laser wakefield acceleration simulation results.

## Installation
- Clone this repository.
- Set up conda environment using the `visepoch_conda_environment.yml` file (see, for example, the [conda environment documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)).
- Activate the environment.
- Install EPOCH SDF utilities (`make sdfutils` from an EPOCH directory).
- Locally install the package by running `pip install -e .` from the repository directory.

## Usage
The modules are available via Python. Documentation is accessed via the docstrings.

## Requirements
- Python 3.8
- Conda packages: numpy, scipy, matplotlib, tqdm, ipython
- Pip packages: palettable
- EPOCH SDF utilities.

## License
[MIT](https://choosealicense.com/licenses/mit/)