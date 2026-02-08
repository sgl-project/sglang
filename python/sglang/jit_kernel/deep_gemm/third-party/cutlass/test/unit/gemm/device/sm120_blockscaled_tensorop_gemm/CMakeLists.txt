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

#
#

if (CUTLASS_NVCC_ARCHS MATCHES 120a)

add_custom_target(
  cutlass_test_unit_gemm_device_sm120_bs
  DEPENDS
  cutlass_test_unit_bs_gemm_device_tensorop_epilogue_fusion_sm120
  cutlass_test_unit_bs_gemm_device_tensorop_sm120
  cutlass_test_unit_bs_gemm_device_tensorop_sm120_stream_k
  cutlass_test_unit_bs_grouped_gemm_device_tensorop_sm120
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_bs_gemm_device_tensorop_epilogue_fusion_sm120

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm120_bs_gemm_nvf4_nvf4_f32_f32_epilogue_fusion.cu
  sm120_bs_gemm_nvf4_nvf4_f32_nvf4_epilogue_fusion.cu
  sm120_bs_gemm_nvf4_nvf4_f32_bf16_epilogue_fusion.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_bs_gemm_device_tensorop_sm120

  sm120_bs_gemm_nvf4_nvf4_f32_bf16.cu
  sm120_bs_gemm_nvf4_nvf4_f32_f16.cu
  sm120_bs_gemm_nvf4_nvf4_f32_f32.cu
  sm120_bs_gemm_nvf4_nvf4_f32_f32_narrow_output.cu
  sm120_bs_gemm_nvf4_nvf4_f32_epilogue.cu
  sm120_bs_gemm_mxf4_mxf4_f32_f32.cu
  sm120_bs_gemm_mxf6_mxf8_f32_f32.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_bs_gemm_device_tensorop_sm120_stream_k
  sm120_bs_gemm_nvf4_nvf4_f32_f32_stream_k.cu
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_bs_grouped_gemm_device_tensorop_sm120
  sm120_bs_gemm_nvf4_nvf4_f32_nvf4_group_gemm_fusion.cu
  sm120_bs_gemm_mxf8_mxf4_f32_group_gemm_fusion.cu
)

endif()
