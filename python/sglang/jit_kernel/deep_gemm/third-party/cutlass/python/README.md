![ALT](/media/images/gemm-hierarchy-with-epilogue-no-labels.png "Complete CUDA GEMM decomposition")

# Python packages associated with CUTLASS

This directory contains Python packages that are associated with CUTLASS:

* `cutlass_cppgen`: the CUTLASS Python interface, which enables one to compile and run CUTLASS kernels from within Python. Note that this was previously named `cutlass`, but was renamed to disambiguate with the CuTe Python DSL.
* `cutlass_library`: utilities used for enumerating and emitting C++ code for CUTLASS kernels

## CUTLASS Python Interface

The CUTLASS Python interface enables one to compile and run CUTLASS operations from within Python.

```python
import cutlass
import numpy as np

plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor)
A, B, C, D = [np.ones((1024, 1024), dtype=np.float16) for i in range(4)]
plan.run(A, B, C, D)
```

### Overview

The CUTLASS Python interface prioritizes ease of use.
It has the following features that support this goal.

* It presents high-level interfaces for operators, that require only few parameters.
* It selects sensible default configurations for an operator given the parameters that have been specified.
* It enumerates configurations for users that are known to work in a given setting.
* It favors emitting descriptive Python run-time exceptions instead of C++ compile-time errors, where possible.
* It simplifies exporting CUTLASS kernels to framework extensions (e.g., PyTorch CUDA extensions).

#### Non-goals
The CUTLASS Python interface does not intend to:

1. select optimal kernel configurations,
2. act as a fast container for CUTLASS kernels, or
3. act as a Python-to-CUDA-kernel just-in-time (JIT) compilation engine.

Regarding selection of optimal kernel configurations,
the interface favors ease-of-use over maximum configurability.
Thus, its default selections for operator parameters may
not achieve the highest possible performance in all scenarios. Users wishing to achieve the highest performance possible should either

* select parameters by profiling different combinations of them, or
* use a library such as [cuBLAS](https://developer.nvidia.com/cublas)
  that contains heuristics for selecting kernels.

Regarding acting as a fast container for CUTLASS kernels:
the interface does not strive to minimize overhead in its Python functions surrounding the running of a kernel.
Those wishing to deploy a CUTLASS kernel should either

* use the C++ emitted by the Python interface directly, or
* use one of the CUTLASS emitters for automatically creating a framework extension for the kernel (e.g., a PyTorch CUDA extension).

Regarding acting as a Python-to-CUDA-kernel JIT compilation engine:
the interface enables use of CUTLASS in Python code.
It can be used by frameworks for JIT compiling
Python to CUDA kernels, but does not set out to be such a framework.

#### Comparison to PyCUTLASS

The CUTLASS Python interface builds atop CUTLASS's [PyCUTLASS](https://github.com/NVIDIA/cutlass/tree/v3.0.0/tools/library/scripts/pycutlass) library. PyCUTLASS enables
one to declare, compile, and run GEMMs, convolutions, and grouped GEMM operators with nearly the same configuration
space as CUTLASS's C++ interface. While this flexibility enables one to achieve the similar levels of functionality
as available in CUTLASS's C++ interface, it comes with the burden of needing to specify many configuration parameters
to operators -- similar to what one must do in specifying template parameters to operations in CUTLASS's C++ interface.

In contrast, the CUTLASS Python interface aims to provide a higher-level API for declaring, emitting, and compiling
kernels that does not require exhaustively defining template parameters.

### Current functionality
The CUTLASS Python interface currently supports the following operations:
* GEMMs
* GEMMs with fused elementwise epilogues (e.g., ReLU) (for pre-SM90 kernels)
* Stream K swizzling (for pre-SM90 kernels)
* Grouped GEMM (for pre-SM90 kernels)

### Getting started
We recommend using the CUTLASS Python interface via an [NGC PyTorch Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch):

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:23.08-py3 -p 8888:8888
```

The CUTLASS Python interface has been tested with CUDA 11.8, 12.0, and 12.1 on Python 3.8 and 3.9.

#### Optional environment variables

Prior to installing the CUTLASS Python interface, one may optionally set the following environment variables:

* `CUTLASS_PATH`: the path to the cloned CUTLASS repository
* `CUDA_INSTALL_PATH`: the path to the installation of CUDA

If these environment variables are not set, the installation process will infer them to be the following:

* `CUTLASS_PATH`: either one directory level above the current directory (i.e., `$(pwd)/..`) if installed locally or in the `source` directory of the location in which `cutlass_library` was installed
* `CUDA_INSTALL_PATH`: the directory holding `/bin/nvcc` for the first version of `nvcc` on `$PATH` (i.e., `which nvcc | awk -F'/bin/nvcc' '{print $1}'`)

**NOTE:** The version of `cuda-python` installed must match the CUDA version in `CUDA_INSTALL_PATH`.

#### Installation

Stable releases of the CUTLASS Python interface are available via the `nvidia-cutlass` PyPI package. Any other packages with the name `cutlass` are not affiliated with NVIDIA CUTLASS.
```bash
pip install nvidia-cutlass
```

The CUTLASS Python interface can also be installed from source by navigating to the root of the CUTLASS directory and performing
```bash
pip install .
```

If you would like to be able to make changes to the CUTLASS Python interface and have them reflected when using the interface, perform:
```bash
pip install -e .
```

To test that your installation was successful, you can run:
```python
import cutlass
import numpy as np

plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor)
A, B, C, D = [np.ones((128, 128), dtype=np.float16) for i in range(4)]
plan.run(A, B, C, D)
```

### Deep learning framework CUDA extensions
The CUTLASS Python interface provides utilities for exporting a CUTLASS kernel to a deep learning framework CUDA extensions. Currently, PyTorch CUDA extensions can be exported, but a similar pattern could be applied for other frameworks as well. An example of this is provided [here](/examples/python/02_pytorch_extension_grouped_gemm.ipynb).

Currently, the following operations can be exported to a PyTorch CUDA extension:
* GEMM
* Grouped GEMM
* Conv2d

### Examples

Jupyter notebook examples of using the CUTLASS Python interface are located in [examples/python](/examples/python).

To launch these notebooks from this directory, run:
```bash
jupyter-lab ../examples/python
```

### Building documentation

The CUTLASS Python interface uses [Sphinx](https://www.sphinx-doc.org/en/master/) for documentation.

Building the documentation requires additional packages.  The following commands will install them.
```bash
sudo apt-get install pandoc
pip install --upgrade Sphinx furo pandoc myst-parser sphinx-copybutton nbsphinx nbsphinx-link sphinx-inline-tabs
```

To build documentation, you must first have installed the CUTLASS Python interface via the
[installation instructions](#installation).

Documentation can then be built via the following commands.
```bash
sphinx-apidoc -o docs_src/source/ cutlass/ cutlass/backend*
cd docs_src
make html
mv _build/* ../docs
```

## CUTLASS library package

[cutlass_library](/python/cutlass_library) contains utilities for enumerating and emitting CUTLASS C++ kernels.
It is used by the CUTLASS CMake system to construct a library of kernels that can be profiled using the CUTLASS profiler.

To install the `cutlass_library` package, run
```bash
python setup_library.py develop --user
```

Alternatively, `cutlass_library` will automatically be installed if you install the CUTLASS Python interface package.

You can also use the [generator.py](/python/cutlass_library/generator.py) script directly without installing the module.

# Copyright

Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
