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

set(TEST_S_TILE_SHAPE --m=256 --n=128 --k=32 --verify --iterations=0)
set(TEST_S_TILE_SHAPE_MULTIPLE_KITER --m=256 --n=128 --k=128 --verify --iterations=0)
set(TEST_S_DIFFERENT_MN --m=16384 --n=4608 --k=4608 --verify --iterations=0)                      
set(TEST_S_ONE_WAVE --m=1536 --n=1536 --k=32 --verify --iterations=0)                 # Assuming 144 SMs
set(TEST_S_2048 --m=2048 --n=2048 --k=2048 --verify --iterations=0)                   # Multi-wave

if(NOT WIN32)
  cutlass_example_add_executable(
    86_blackwell_mixed_dtype_gemm
    86_blackwell_mixed_dtype.cu
    TEST_COMMAND_OPTIONS
    TEST_S_TILE_SHAPE
    TEST_S_TILE_SHAPE_MULTIPLE_KITER
    TEST_S_ONE_WAVE
    TEST_S_2048
  )
endif()
