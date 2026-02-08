
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

set(TEST_SQUARE --m=2048 --n=2048 --k=2048 -l=10 --iterations=1)                   # Square problem sizes
set(TEST_SQUARE_LARGE_BATCH --m=2048 --n=2048 --k=2048 -l=500 --iterations=1)      # Square problem sizes

set(TEST_EPILOGUE --alpha=0.5 --beta=0.5 --iterations=1)                           # Default problem sizes
set(TEST_EPILOGUE_LARGE_BATCH --alpha=1.5 --beta=2.0 -l=500 --iterations=1)        # Default problem sizes

set(TEST_EPILOGUE_OP --beta=0.5 --iterations=1)                                    # Default problem sizes w/ Epilogue Op test
set(TEST_EPILOGUE_OP_LARGE_BATCH --alpha=1.5 -l=500 --iterations=1)                # Default problem sizes w/ Epilogue Op test

set(TEST_SMALLK --m=2048 --n=5120 --k=128 --l=5 --iterations=1)                    # Small-k problem sizes
set(TEST_SMALLK_LARGE_BATCH --m=1024 --n=512 --k=64 --l=500 --iterations=1)        # Small-k problem sizes

cutlass_example_add_executable(
  56_hopper_ptr_array_batched_gemm
  56_hopper_ptr_array_batched_gemm.cu
  TEST_COMMAND_OPTIONS
  TEST_SQUARE
  TEST_SQUARE_LARGE_BATCH
  TEST_EPILOGUE
  TEST_EPILOGUE_LARGE_BATCH
  TEST_EPILOGUE_OP
  TEST_EPILOGUE_OP_LARGE_BATCH
  TEST_SMALLK
  TEST_SMALLK_LARGE_BATCH
  )
