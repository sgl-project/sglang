# Introduction

This example shows fusing two back-to-back GEMMs/Convolutions into one kernel.

<p align="center"><img src=/media/images/13_example_fusion.png></p>

When running two unfused GEMM/Conv operations, each operation loads one input
activation matrix, one weight matrix (or filter matrix) from the memory and then
stores the result activation matrix back to the memory.

When the two GEMM/Conv operations are fused together, the mainloops of the two
GEMMs/Convs run back to back in a single kernel. The output accumulator of the
1st GEMM/Conv will be stored in the register file and reused as the activation
input of the 2nd GEMM/Conv. This saves a round trip to memory for the activation
matrix.


This example computes the following:
- 1st GEMM/Conv: D0 = relu(alpha0 .\* A0 \*\* B0)
- 2nd GEMM/Conv: D1 = relu(alpha1 .\* D0 \*\* B1 + beta1 .\* C1)

In the above equation, operator \*\* can be matrix multiplication or convolution operation.

# Implementation Details

In order to run two GEMM/Convs in a single kernel, the example requires the same number of
threadblocks are used across 2 GEMMs/Convs. This also ensures the same threadblock tile M across
2 GEMMs/Convs.

In order to reuse the output accumulator (stored in register-file) of the 1st GEMM as the
input activation, the example enforces the following two constraints:

- thread_block_tile_N = problem_N

<p align="center"><img src=/media/images/13_example_block_resident_fusion.png></p>

This constraint ensures that each threadblock loads the entire weight/filter matrix in
addition to its own input activation tile. Therefore the input activation tile of the
2nd GEMM/Conv only depends on the output activation tile of the 1st GEMM/Conv, and the
operation can be fully block-resident.

- warp_tile_N = thread_block_tile_N

<p align="center"><img src=/media/images/13_example_rf_resident_fusion.png></p>

This constraint ensures that each warp loads the entire weight/filter kBlock in
addition to its own input activation tile. Therefore the input activation warp tile of the
2nd GEMM/Conv only depends on the output warp accumulator of the 1st GEMM/Conv in the
register file, and the operation can be fully register-file-resident.

On the other hand, this constraint can be relaxed if the output accumulator of the 1st GEMM/CONV
is staged in the shared memory and then used as input for the 2nd GEMM/CONV. In this case, the
input of each warp tile can be loaded from the shared memory so they do not need to be RF-resident,
therefore each warp does not need to store the entire input matrix of 2nd GEMM in its RF. This is
illustrated in the diagram below.

<p align="center"><img src=/media/images/13_example_shmem_resident_fusion.png></p>


When applying the above constraint to convolutions, it is required that the 2nd Convolution
kernel doesn't have halos such that data used by each threadblock doesn't depend on any other
threadblock. Typically this requires the 2nd Convolution uses 1x1 filter without any paddings.

# Build and run

- Run cmake at top-level CUTLASS
- `make 13_two_tensor_op_fusion`
- Run individual benchmarks
  - `./examples/13_two_tensor_op_fusion/13_fused_two_convs_f16_sm75_rf`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_convs_f16_sm75_shmem`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_convs_f16_sm80_rf`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_convs_f16_sm80_shmem`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_convs_s8_sm75_rf`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_convs_s8_sm75_shmem`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_convs_s8_sm80_rf`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_convs_s8_sm80_shmem`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_gemms_f16_sm75_rf`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_gemms_f16_sm75_shmem`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_gemms_f16_sm80_rf`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_gemms_f16_sm80_shmem`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm75_rf`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm75_shmem`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm80_rf`
  - `./examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm80_shmem`


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
