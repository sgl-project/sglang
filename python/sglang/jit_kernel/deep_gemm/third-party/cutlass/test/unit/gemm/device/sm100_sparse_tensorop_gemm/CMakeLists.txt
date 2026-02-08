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

add_subdirectory(narrow_precision)

if (CUTLASS_NVCC_ARCHS MATCHES 100a)

add_custom_target(
  cutlass_test_unit_gemm_device_sm100_sparse
  DEPENDS
  cutlass_test_unit_gemm_device_sm100_sparse_general
  cutlass_test_unit_gemm_device_sm100_sparse_streamk
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_sparse_general

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_sp_gemm_s8_s8_s32_s8_s8_imma.cu
  sm100_sp_gemm_f8_f8_f32_f16_f8_qmma.cu
  sm100_sp_gemm_f8_f8_f32_f16_f16_qmma.cu
  sm100_sp_gemm_f8_f8_f32_f32_f32_qmma.cu
  sm100_sp_gemm_f32_f32_f32_f32_f32_tfmma.cu
  sm100_sp_gemm_f16_f16_f32_f16_f16_hmma.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_sparse_streamk

  # No batching of source to control compiler memory usage
  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_sp_gemm_f16_f16_f32_f32_f32_streamk.cu
)

endif()
