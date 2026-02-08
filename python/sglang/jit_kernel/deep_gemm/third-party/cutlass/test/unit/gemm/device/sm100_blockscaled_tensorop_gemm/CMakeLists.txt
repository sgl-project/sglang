# Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if (CUTLASS_NVCC_ARCHS MATCHES 100a)
add_custom_target(
  cutlass_test_unit_gemm_device_sm100_blockscaled
  DEPENDS
  cutlass_test_unit_gemm_device_bstensorop_sm100_nvf4xnvf4
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf4xmxf4
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf6xmxf6
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf8xmxf8
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf6xmxf8
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf8xmxf6
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf4xmxf8
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf8xmxf4
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf6xmxf4
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf4xmxf6
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_bstensorop_sm100_nvf4xnvf4

  BATCH_SOURCES ON
  BATCH_SIZE 1

  nvf4_nvf4_bf16_bf16.cu
  nvf4_nvf4_bf16_bf16_features.cu
  nvf4_nvf4_f16_nvfp4_epilogue.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf4xmxf4

  BATCH_SOURCES ON
  BATCH_SIZE 1

  mxf4_mxf4_void_f16_tn_layout.cu
  mxf4_mxf4_void_f16_nt_layout.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf6xmxf6

  BATCH_SOURCES ON
  BATCH_SIZE 1

  mxf6_mxf6_void_bf16_tn_layout.cu
  mxf6_mxf6_void_bf16_nt_layout.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf8xmxf8

  BATCH_SOURCES ON
  BATCH_SIZE 1

  mxf8_mxf8_void_f8_tn_layout.cu
  mxf8_mxf8_void_f8_nt_layout.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf6xmxf8

  BATCH_SOURCES ON
  BATCH_SIZE 1

  mxf6_mxf8_void_f32_tn_layout.cu
  mxf6_mxf8_void_f32_nt_layout.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf8xmxf6

  BATCH_SOURCES ON
  BATCH_SIZE 1

  mxf8_mxf6_f16_f8_tn_layout.cu
  mxf8_mxf6_f16_f8_nt_layout.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf4xmxf8

  BATCH_SOURCES ON
  BATCH_SIZE 1

  mxf4_mxf8_bf16_bf16_tn_layout.cu
  mxf4_mxf8_bf16_bf16_nt_layout.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf8xmxf4

  BATCH_SOURCES ON
  BATCH_SIZE 1

  mxf8_mxf4_f16_bf16_tn_layout.cu
  mxf8_mxf4_f16_bf16_nt_layout.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf6xmxf4

  BATCH_SOURCES ON
  BATCH_SIZE 1

  mxf6_mxf4_f16_f16_tn_layout.cu
  mxf6_mxf4_f16_f16_nt_layout.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_bstensorop_sm100_mxf4xmxf6

  BATCH_SOURCES ON
  BATCH_SIZE 1

  mxf4_mxf6_f32_f16_tn_layout.cu
  mxf4_mxf6_f32_f16_nt_layout.cu
)

endif()
