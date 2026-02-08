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

set(TEST_RANDOM --iterations=0)                                                          # Random problem sizes
set(TEST_RANDOM_LARGE_GROUP --groups=100 --iterations=0)                                 # Random problem sizes
    
set(TEST_EPILOGUE --alpha=0.5 --beta=0.5 --iterations=0)                                 # Random problem sizes
set(TEST_EPILOGUE_LARGE_GROUP --alpha=2.0 --beta=2.0 --groups=100 --iterations=0)        # Random problem sizes
    
set(TEST_EPILOGUE_OP --beta=0.5 --iterations=1)                                          # Random problem sizes
set(TEST_EPILOGUE_OP_LARGE_GROUP --alpha=0.25 --iterations=1)                            # Random problem sizes
    
set(TEST_FIXED --m=2048 --n=5120 --k=8192 --groups=16 --iterations=0)                    # Fixed problem sizes
set(TEST_FIXED_LARGE_GROUP --m=2048 --n=512 --k=512 --groups=100 --iterations=0)         # Fixed problem sizes
    
set(TEST_SMALL --m=256 --n=128 --iterations=0)                                           # Small problem sizes
set(TEST_SMALL_LARGE_GROUP --m=128 --n=128 --groups=100 --iterations=0)                  # Small problem sizes
    
set(TEST_RANDOM_PERF --iterations=10)                                                    # Random problem sizes
set(TEST_RANDOM_PERF_LARGE_GROUP --groups=100 --iterations=10)                           # Random problem sizes
    
set(TEST_DIRECT_BATCHED --m=2048 --n=5120 --k=8192 --mode=0 --iterations=0)              # Direct conversion

set(TEST_SCALE_PERCOL --m=4096 --n=5120 --k=8192 --c=8192 --mode=1 --iterations=0)       # Per Column scaling
set(TEST_SCALE_GROUP --m=2048 --n=5120 --k=8192 --c=512 --mode=1 --iterations=0)         # Group-wise scaling

cutlass_example_add_executable(
  69_hopper_mixed_dtype_grouped_gemm
  69_hopper_mixed_dtype_grouped_gemm.cu
  TEST_COMMAND_OPTIONS
  TEST_RANDOM
  TEST_RANDOM_LARGE_GROUP
  TEST_EPILOGUE
  TEST_EPILOGUE_LARGE_GROUP
  TEST_EPILOGUE_OP
  TEST_EPILOGUE_OP_LARGE_GROUP
  TEST_FIXED
  TEST_FIXED_LARGE_GROUP
  TEST_SMALL
  TEST_SMALL_LARGE_GROUP
  TEST_RANDOM_PERF
  TEST_RANDOM_PERF_LARGE_GROUP
  TEST_DIRECT_BATCHED
  TEST_SCALE_PERCOL
  TEST_SCALE_GROUP
)

cutlass_example_add_executable(
  69_hopper_int4_fp8_grouped_gemm
  69_hopper_int4_fp8_grouped_gemm.cu
  TEST_COMMAND_OPTIONS
  TEST_RANDOM
  TEST_RANDOM_LARGE_GROUP
  TEST_EPILOGUE
  TEST_EPILOGUE_LARGE_GROUP
  TEST_EPILOGUE_OP
  TEST_EPILOGUE_OP_LARGE_GROUP
  TEST_FIXED
  TEST_FIXED_LARGE_GROUP
  TEST_SMALL
  TEST_SMALL_LARGE_GROUP
  TEST_RANDOM_PERF
  TEST_RANDOM_PERF_LARGE_GROUP
  TEST_DIRECT_BATCHED
  TEST_SCALE_PERCOL
  TEST_SCALE_GROUP
)

cutlass_example_add_executable(
  69_hopper_int4_bf16_grouped_gemm
  69_hopper_int4_bf16_grouped_gemm.cu
  TEST_COMMAND_OPTIONS
  TEST_RANDOM
  TEST_RANDOM_LARGE_GROUP
  TEST_EPILOGUE
  TEST_EPILOGUE_LARGE_GROUP
  TEST_EPILOGUE_OP
  TEST_EPILOGUE_OP_LARGE_GROUP
  TEST_FIXED
  TEST_FIXED_LARGE_GROUP
  TEST_SMALL
  TEST_SMALL_LARGE_GROUP
  TEST_RANDOM_PERF
  TEST_RANDOM_PERF_LARGE_GROUP
  TEST_DIRECT_BATCHED
  TEST_SCALE_PERCOL
  TEST_SCALE_GROUP
)
