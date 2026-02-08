This example extends Example 55 to support Grouped GEMMs in CUTLASS.

## High level overview

This example shows how to perform Grouped GEMMs on Hopper when A and B have different types. In the Grouped GEMM, multiple GEMMs with potentially different problem shapes can be excetued in a batch. The interface is similar to the standard mixed-input GEMM presented in Example 55, with a few noteworthy differences:
- inside the collective builder, replace the layout types with layout pointer types.
- in the arguments, pass the group size, array of the problem sizes, and the array of strides for matrix A and B.
- if scales and zero-points are included, also pass the array of their strides in the arguments.

Note that in Example 55, the argument `--g` is used to determine the group size of scaling. To avoid confusion with the `--groups` argument in this example, which defines the number of GEMMs, `--c` is used here to represent the group size for scaling.

## Upcoming features

Currently, the Mixed-input Grouped GEMM only supports row-wise scaling, and group-wise scaling for identical problem shapes across all groups. Please contact us if zero-points or block-wise scaling are needed.

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
