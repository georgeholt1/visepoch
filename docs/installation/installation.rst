Installation
============

VISEPOCH is not currently available via pip but can still be installed from
source.

#. Clone the repository or download and extract a release file.
#. Set up and activate the conda environment using the supplied
   ``visepoch_conda_environment.yml`` file. See the `conda environment
   documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`_.
#. Install the EPOCH SDF utilities by following the steps in the EPOCH manual.
   This is typically done with ``make sdfutils`` in one of the ``epoch``
   directories.
#. Locally install the VISEPOCH package by running ``pip install -e .`` from the
   repository directory.