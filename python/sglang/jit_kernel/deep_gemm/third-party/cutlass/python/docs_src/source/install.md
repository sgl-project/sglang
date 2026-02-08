# Installation

## Installing a stable release

Stable releases of the CUTLASS Python interface are available via the `nvidia-cutlass` PyPI package. Any other packages with the name `cutlass` are not affiliated with NVIDIA CUTLASS.
```bash
pip install nvidia-cutlass
```

## Installing from source

Installing from source requires the latest CUDA Toolkit that matches the major.minor of CUDA Python installed.

Prior to installing the CUTLASS Python interface, one may optionally set the following environment variables:
* `CUTLASS_PATH`: the path to the cloned CUTLASS repository
* `CUDA_INSTALL_PATH`: the path to the installation of CUDA

If these environment variables are not set, the installation process will infer them to be the following:
* `CUTLASS_PATH`: either one directory level above the current directory (i.e., `$(pwd)/..`) if installed locally or in the `source` directory of the location in which `cutlass_library` was installed
* `CUDA_INSTALL_PATH`: the directory holding `/bin/nvcc` for the first version of `nvcc` on `$PATH` (i.e., `which nvcc | awk -F'/bin/nvcc' '{print $1}'`)

**NOTE:** The version of `cuda-python` installed must match the CUDA version in `CUDA_INSTALL_PATH`.

### Installing a developer-mode package
The CUTLASS Python interface can currently be installed by navigating to the root of the CUTLASS directory and performing
```bash
pip install .
```

If you would like to be able to make changes to CULASS Python interface and have them reflected when using the interface, perform:
```bash
pip install -e .
```

## Docker
We recommend using the CUTLASS Python interface via an [NGC PyTorch Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch):

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:23.08-py3
```

## Copyright

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
