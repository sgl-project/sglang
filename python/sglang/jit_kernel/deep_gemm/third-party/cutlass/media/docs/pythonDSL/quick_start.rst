.. _quick_start:

Quick Start Guide
=======================

The CUTLASS DSL 4.0 release currently supports **Linux** and **Python 3.12** only. To install CUTLASS DSLs (limited to CuTe DSL for now), use the following command

Installation
-----------------------

To install the CUTLASS DSL, run:

.. code-block:: bash

   pip install nvidia-cutlass-dsl

The ``nvidia-cutlass-dsl`` wheel includes everything needed to generate GPU kernels. It requires 
the same NVIDIA driver version as the 
`CUDA Toolkit 12.9 <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>`_.

To ensure compatibility with the examples and code on `GitHub <https://github.com/NVIDIA/cutlass/tree/main/python>`_,
use the ``requirements.txt`` file from the corresponding commit in the repository.

Recommended Dependencies
---------------------------------

To run examples and begin development, we recommend installing:

.. code-block:: bash

   pip install torch jupyter

Recommended Python environment variables for jupyter notebooks
--------------------------------------------------------------

We recommend setting the following environment variable when running jupyter notebooks.

.. code-block:: bash

   export PYTHONUNBUFFERED=1
