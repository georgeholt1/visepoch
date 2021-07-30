# VISEPOCH

A Python 3 package for visualising [EPOCH](https://github.com/Warwick-Plasma/epoch) laser wakefield acceleration simulation results.

## Installation
While this package is not yet available on pip, it can still be used by following these steps:

- Clone this repository or download a release file.
- Set up conda environment using the `visepoch_conda_environment.yml` file (see, for example, the [conda environment documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)). The dependencies are also listed in the `requirements.txt` file in the repository root directory, should you wish to configure your own environment.
- Activate the environment.
- Install EPOCH SDF utilities (`make sdfutils` from an EPOCH directory).
- Locally install the package by running `pip install -e .` from the repository directory.

## Usage
The modules are available via Python. Documentation is accessed via the docstrings.

## Requirements
- Python 3.8
- Requirements listed in `requirements.txt`.
- EPOCH SDF utilities.

## Limitations
- Currently only works with 2D EPOCH data.

## License
[MIT](https://choosealicense.com/licenses/mit/)