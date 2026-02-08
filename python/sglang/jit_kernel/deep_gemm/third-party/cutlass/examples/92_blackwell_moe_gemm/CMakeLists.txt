# Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(TEST_MIXTRAL_A --m=28672 --n=4 --k=4096 --l=8)
set(TEST_MIXTRAL_B --m=4096 --n=4 --k=14336 --l=8)
set(TEST_DEEPSEEK_A --m=4096 --n=1 --k=7168 --l=256)
set(TEST_DEEPSEEK_B --m=7168 --n=1 --k=2048 --l=256)
set(TEST_IRREGULAR_MNK --m=4080 --n=9 --k=4112 --l=8)       # M,N,K not multiples of tile size

set(TEST_DEEPSEEK_A_FP4 --m=1024 --n=1 --k=7168 --l=256)     # TP=1 shape is too large for PackedVectorLayout
set(TEST_DEEPSEEK_B_FP4 --m=7168 --n=1 --k=512 --l=256)
set(TEST_IRREGULAR_MNK_FP4 --m=4080 --n=9 --k=4160 --l=8)

if (CUTLASS_NVCC_ARCHS MATCHES 100a)
cutlass_example_add_executable(
  92_blackwell_moe_gemm_regular
  92_blackwell_moe_gemm_regular.cu
  TEST_COMMAND_OPTIONS
  TEST_MIXTRAL_A
  TEST_MIXTRAL_B
  TEST_DEEPSEEK_A
  TEST_DEEPSEEK_B
  TEST_IRREGULAR_MNK
)

cutlass_example_add_executable(
  92_blackwell_moe_gemm_grouped
  92_blackwell_moe_gemm_grouped.cu
)

cutlass_example_add_executable(
  92_blackwell_moe_gemm_fp4_regular
  92_blackwell_moe_gemm_fp4_regular.cu
  TEST_COMMAND_OPTIONS
  TEST_MIXTRAL_A
  TEST_MIXTRAL_B
  TEST_DEEPSEEK_A_FP4
  TEST_DEEPSEEK_B_FP4
  TEST_IRREGULAR_MNK_FP4
)
cutlass_example_add_executable(
  92_blackwell_moe_gemm_fp4_grouped
  92_blackwell_moe_gemm_fp4_grouped.cu
)
endif()
