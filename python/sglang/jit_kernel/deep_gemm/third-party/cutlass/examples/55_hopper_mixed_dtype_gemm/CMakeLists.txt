
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Note that we set --iterations=0 for all tests below to disable the performance benchmarking.
# Only the correctness check will be run by these commands.

set(TEST_DIRECT_BATCHED --m=2048 --n=2048 --k=2048 --l=2 --mode=0 --iterations=0)        # Direct conversion

set(TEST_SCALE_PERCOL --m=4096 --n=5120 --k=8192 --g=8192 --mode=1 --iterations=0)       # Per Column scaling
set(TEST_SCALE_ZERO_PERCOL --m=4096 --n=5120 --k=8192 --g=8192 --mode=2 --iterations=0)  # Per Column scaling

set(TEST_SCALE_GROUP --m=2048 --n=5120 --k=8192 --g=512 --mode=1 --iterations=0)         # Group-wise scaling
set(TEST_SCALE_ZERO_GROUPED --m=2048 --n=5120 --k=8192 --g=256 --mode=2 --iterations=0)  # Group-wise scaling with zero-point

set(TEST_SCALE_RESIDUE --m=128 --n=128 --k=320 --g=128 --mode=1 --iterations=0)          # Final group has residue
set(TEST_SCALE_ZERO_RESIDUE --m=128 --n=128 --k=192 --g=128 --mode=2 --iterations=0)     # Final group has residue

set(TEST_ALPHA_BETA --alpha=0.5 --beta=0.7 --mode=2 --iterations=0)                      # Alpha and Beta with default shapes


cutlass_example_add_executable(
  55_hopper_mixed_dtype_gemm
  55_hopper_mixed_dtype_gemm.cu
  TEST_COMMAND_OPTIONS
  TEST_DIRECT_BATCHED
  TEST_SCALE_PERCOL
  TEST_SCALE_ZERO_PERCOL
  TEST_SCALE_GROUP
  TEST_SCALE_ZERO_GROUPED
  TEST_SCALE_RESIDUE
  TEST_SCALE_ZERO_RESIDUE
  # TEST_ALPHA_BETA
  )

cutlass_example_add_executable(
  55_hopper_int4_fp8_gemm
  55_hopper_int4_fp8_gemm.cu
  TEST_COMMAND_OPTIONS
  TEST_DIRECT_BATCHED
  TEST_SCALE_PERCOL
  TEST_SCALE_GROUP
  TEST_SCALE_RESIDUE
  # TEST_ALPHA_BETA
  )

  cutlass_example_add_executable(
    55_hopper_int4_bf16_gemm
    55_hopper_int4_bf16_gemm.cu
    TEST_COMMAND_OPTIONS
    TEST_DIRECT_BATCHED
    TEST_SCALE_PERCOL
    TEST_SCALE_GROUP
    TEST_SCALE_RESIDUE
    # TEST_ALPHA_BETA
    )
