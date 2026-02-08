# CUTLASS 3.0

_CUTLASS 3.0 - January 2023_

CUTLASS is a collection of CUDA C++ template abstractions for implementing
high-performance matrix-multiplication (GEMM) at all levels and scales within CUDA.
It incorporates strategies for hierarchical decomposition and data movement similar
to those used to implement cuBLAS.  CUTLASS decomposes these "moving parts" into
reusable, modular software components abstracted by C++ template classes.  These
components can be specialized
and tuned via custom tiling sizes, data types, and other algorithmic policies. The
resulting flexibility simplifies their use as building blocks within custom kernels
and applications.

To support a wide variety of applications, CUTLASS provides extensive support for
mixed-precision computations, providing specialized data-movement and
multiply-accumulate abstractions for 8-bit integer, half-precision floating
point (FP16), single-precision floating point (FP32), and double-precision floating
point (FP64) types.  Furthermore, CUTLASS exploits the _Tensor Cores_ and asynchronous
memory copy operations of the latest NVIDIA GPU architectures.

# What's New in CUTLASS 3.0

For an overview of CUTLASS 3.0's GEMM interface levels,
please refer to the
[CUTLASS 3.0 GEMM API document](./gemm_api_3x.md).
To learn how to migrate code using CUTLASS 2.x's interface
to CUTLASS 3.0, please refer to the
[backwards compatibility document](./cutlass_3x_backwards_compatibility.md).

# GEMM examples

For a code example showing how to define
a GEMM kernel using CUTLASS, please refer to
[the quickstart guide](./quickstart.md).
The [`examples` directory](https://github.com/NVIDIA/cutlass/tree/main/examples)
has a variety of examples.

# Copyright

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
