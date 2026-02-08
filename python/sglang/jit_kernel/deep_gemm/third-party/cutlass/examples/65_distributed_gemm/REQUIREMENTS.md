# Distributed GEMM

## Requirements

### Build
Make sure to set up CUTLASS with
support for [Programmatic Dependent Launch (PDL)](../../media/docs/dependent_kernel_launch.md),
that is with the `CUTLASS_ENABLE_GDC_FOR_SM90` flag.

```bash
cmake $PATH -DCUTLASS_NVCC_ARCHS="90a" -DCUTLASS_ENABLE_GDC_FOR_SM90=1
```

### Minimum software

Like all other CUTLASS examples, the NVIDIA driver, runtime, and CUDA Toolkit are required.
This example specifically requires CUDA Toolkit 12.6 or newer, due to some of the necessary
CUDA graph APIs.

The minimum CUDA driver version for running this example is [560.28.03](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html#id5).

### Hardware / driver settings

This example requires Hopper GPUs with NVLink network.

If you're not sure, first run the following command and make sure your GPU
compute capability is 9.0:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Sample output:

```
name, compute_cap
NVIDIA H100 80GB HBM3, 9.0
NVIDIA H100 80GB HBM3, 9.0
NVIDIA H100 80GB HBM3, 9.0
NVIDIA H100 80GB HBM3, 9.0
NVIDIA H100 80GB HBM3, 9.0
NVIDIA H100 80GB HBM3, 9.0
NVIDIA H100 80GB HBM3, 9.0
NVIDIA H100 80GB HBM3, 9.0
```


Then you should make sure there is an NVLink network by checking the GPU network topology,
and making sure there's `NV*` links between every pair of GPUs:

```bash
nvidia-smi topo -m
```

Sample output:

```
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0     X      NV18    NV18    NV18    NV18    NV18    NV18    NV18
GPU1    NV18     X      NV18    NV18    NV18    NV18    NV18    NV18
GPU2    NV18    NV18     X      NV18    NV18    NV18    NV18    NV18
GPU3    NV18    NV18    NV18     X      NV18    NV18    NV18    NV18
GPU4    NV18    NV18    NV18    NV18     X      NV18    NV18    NV18
GPU5    NV18    NV18    NV18    NV18    NV18     X      NV18    NV18
GPU6    NV18    NV18    NV18    NV18    NV18    NV18     X      NV18
GPU7    NV18    NV18    NV18    NV18    NV18    NV18    NV18     X
```

Finally, check if the driver enables peer to peer access, which should usually be the case,
but it's good to check anyway:

```bash
nvidia-smi topo -p2p r
```

Sample output:

```
       GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0   X       OK      OK      OK      OK      OK      OK      OK
GPU1   OK      X       OK      OK      OK      OK      OK      OK
GPU2   OK      OK      X       OK      OK      OK      OK      OK
GPU3   OK      OK      OK      X       OK      OK      OK      OK
GPU4   OK      OK      OK      OK      X       OK      OK      OK
GPU5   OK      OK      OK      OK      OK      X       OK      OK
GPU6   OK      OK      OK      OK      OK      OK      X       OK
GPU7   OK      OK      OK      OK      OK      OK      OK      X
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
