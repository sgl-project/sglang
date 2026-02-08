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

add_custom_target(
  cutlass_test_unit_conv_dgrad_device
  DEPENDS
  cutlass_test_unit_conv_dgrad_device_tensorop_sm90
  cutlass_test_unit_conv_dgrad_device_tensorop_sm100
  cutlass_test_unit_conv_dgrad_device_tensorop_sm100_fusion
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_conv_dgrad_device_tensorop_sm90

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm90_conv1d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f16.cu
  sm90_conv2d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f16.cu
  sm90_conv3d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f16.cu

  sm90_conv1d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f32.cu
  sm90_conv2d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f32.cu
  sm90_conv3d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f32.cu
)

if (CUTLASS_NVCC_ARCHS MATCHES 100a)

set(cutlass_test_unit_conv_dgrad_device_tensorop_sm100_kernels
  sm100_conv2d_dgrad_implicit_gemm_f8_f8_f8_tensorop_f32.cu
  sm100_conv2d_dgrad_implicit_gemm_f8_f8_bf16_tensorop_f32.cu
  sm100_conv2d_dgrad_implicit_gemm_f8_f8_f16_tensorop_f32.cu
  sm100_conv2d_dgrad_implicit_gemm_f8_f8_f32_tensorop_f32.cu

  sm100_conv3d_dgrad_implicit_gemm_f8_f8_f8_tensorop_f32.cu
  sm100_conv3d_dgrad_implicit_gemm_f8_f8_bf16_tensorop_f32.cu
  sm100_conv3d_dgrad_implicit_gemm_f8_f8_f16_tensorop_f32.cu
  sm100_conv3d_dgrad_implicit_gemm_f8_f8_f32_tensorop_f32.cu

  sm100_conv1d_dgrad_implicit_gemm_f16_f16_f16_tensorop_f16.cu
  sm100_conv2d_dgrad_implicit_gemm_f16_f16_f16_tensorop_f16.cu
  sm100_conv3d_dgrad_implicit_gemm_f16_f16_f16_tensorop_f16.cu

  sm100_conv1d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f32.cu
  sm100_conv2d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f32.cu
  sm100_conv3d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f32.cu
)

# Add the executable
cutlass_test_unit_add_executable(
  cutlass_test_unit_conv_dgrad_device_tensorop_sm100
  ${cutlass_test_unit_conv_dgrad_device_tensorop_sm100_kernels}
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_conv_dgrad_device_tensorop_sm100_fusion

  sm100_conv2d_dgrad_implicit_gemm_f8_f8_f16_tensorop_f32_with_fusion.cu
  sm100_conv3d_dgrad_implicit_gemm_f8_f8_f16_tensorop_f32_with_fusion.cu

  sm100_conv1d_dgrad_implicit_gemm_f16_f16_f16_tensorop_f16_with_fusion.cu
  sm100_conv2d_dgrad_implicit_gemm_f16_f16_f16_tensorop_f16_with_fusion.cu
  sm100_conv3d_dgrad_implicit_gemm_f16_f16_f16_tensorop_f16_with_fusion.cu
)

endif()
