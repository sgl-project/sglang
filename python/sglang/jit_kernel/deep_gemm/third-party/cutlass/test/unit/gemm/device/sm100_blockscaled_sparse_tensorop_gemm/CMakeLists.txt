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

if (CUTLASS_NVCC_ARCHS MATCHES 100a)

add_custom_target(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse
  DEPENDS
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_nvf4_nvf4_f32_f32_f32_o
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_nvf4_nvf4_f32_f16_f16_o
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_nvf4_nvf4_f32_f16_nvf4_o
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf8_mxf8_f32_f32_f32_q
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf8_mxf8_f32_f16_f16_q
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf8_mxf8_f32_f16_mxf8_q
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf8mxf4_mxf4mxf8_f32_q
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf8mxf6_mxf6mxf8_f32_q
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf4mxf6_mxf4mxf6_f32_q
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf6_mxf6_f32_q
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf4_mxf4_f32_q
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf4_mxf4_f32_o
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_streamk
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_nvf4_nvf4_f32_f32_f32_o

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_nvf4_nvf4_f32_f32_f32_o_tnt.cu
  sm100_bssp_gemm_nvf4_nvf4_f32_void_f32_o_tnt.cu

  sm100_bssp_gemm_nvf4_nvf4_f32_f32_f32_o_tnn.cu
  sm100_bssp_gemm_nvf4_nvf4_f32_void_f32_o_tnn.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_nvf4_nvf4_f32_f16_f16_o

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_nvf4_nvf4_f32_f16_f16_o_tnt.cu
  sm100_bssp_gemm_nvf4_nvf4_f32_void_f16_o_tnt.cu

  sm100_bssp_gemm_nvf4_nvf4_f32_f16_f16_o_tnn.cu
  sm100_bssp_gemm_nvf4_nvf4_f32_void_f16_o_tnn.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_nvf4_nvf4_f32_f16_nvf4_o

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_nvf4_nvf4_f32_f16_nvf4_o_tnt_sfd.cu
  sm100_bssp_gemm_nvf4_nvf4_f32_void_nvf4_o_tnt_sfd.cu

  sm100_bssp_gemm_nvf4_nvf4_f32_f16_nvf4_o_tnn_sfd.cu
  sm100_bssp_gemm_nvf4_nvf4_f32_void_nvf4_o_tnn_sfd.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf8_mxf8_f32_f32_f32_q

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_mxf8_mxf8_f32_f32_f32_q_tnt.cu
  sm100_bssp_gemm_mxf8_mxf8_f32_void_f32_q_tnt.cu

  sm100_bssp_gemm_mxf8_mxf8_f32_f32_f32_q_tnn.cu
  sm100_bssp_gemm_mxf8_mxf8_f32_void_f32_q_tnn.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf8_mxf8_f32_f16_f16_q

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_mxf8_mxf8_f32_f16_f16_q_tnt.cu
  sm100_bssp_gemm_mxf8_mxf8_f32_void_f16_q_tnt.cu

  sm100_bssp_gemm_mxf8_mxf8_f32_f16_f16_q_tnn.cu
  sm100_bssp_gemm_mxf8_mxf8_f32_void_f16_q_tnn.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf8_mxf8_f32_f16_mxf8_q

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_mxf8_mxf8_f32_f16_mxf8_q_tnt_sfd.cu
  sm100_bssp_gemm_mxf8_mxf8_f32_void_mxf8_q_tnt_sfd.cu

  sm100_bssp_gemm_mxf8_mxf8_f32_f16_mxf8_q_tnn_sfd.cu
  sm100_bssp_gemm_mxf8_mxf8_f32_void_mxf8_q_tnn_sfd.cu

  sm100_bssp_gemm_mxf8_mxf8_f32_f16_mxf8_q_ttt_sfd.cu
  sm100_bssp_gemm_mxf8_mxf8_f32_f16_mxf8_q_ttn_sfd.cu
  sm100_bssp_gemm_mxf8_mxf8_f32_f16_mxf8_q_nnt_sfd.cu
  sm100_bssp_gemm_mxf8_mxf8_f32_f16_mxf8_q_nnn_sfd.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf4_mxf4_f32_o

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_mxf4_mxf4_f32_f16_f16_o_tnn.cu
  sm100_bssp_gemm_mxf4_mxf4_f32_f16_f16_o_tnt.cu

  sm100_bssp_gemm_mxf4_mxf4_f32_f32_f32_o_tnt.cu
  sm100_bssp_gemm_mxf4_mxf4_f32_f32_f32_o_tnn.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf8mxf4_mxf4mxf8_f32_q

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_mxf8_mxf4_f32_f16_mxf8_q_tnt.cu
  sm100_bssp_gemm_mxf8_mxf4_f32_f16_f16_q_tnt.cu
  sm100_bssp_gemm_mxf8_mxf4_f32_f32_f32_q_tnt.cu

  sm100_bssp_gemm_mxf4_mxf8_f32_f16_f16_q_tnt.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf8mxf6_mxf6mxf8_f32_q

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_mxf6_mxf8_f32_f16_f16_q_tnt.cu
  sm100_bssp_gemm_mxf8_mxf6_f32_f16_f16_q_tnt.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf4mxf6_mxf4mxf6_f32_q

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_mxf4_mxf6_f32_f16_f16_q_tnt.cu
  sm100_bssp_gemm_mxf6_mxf4_f32_f16_f16_q_tnt.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf4_mxf4_f32_q

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_mxf4_mxf4_f32_f16_mxf8_q_tnt.cu
  sm100_bssp_gemm_mxf4_mxf4_f32_f16_f16_q_tnt.cu
  sm100_bssp_gemm_mxf4_mxf4_f32_f32_f32_q_tnt.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_mxf6_mxf6_f32_q

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_mxf6_mxf6_f32_f16_f16_q_tnt.cu
)

cutlass_test_unit_gemm_device_add_executable_split_file(
  cutlass_test_unit_gemm_device_sm100_blockscaled_sparse_streamk

  BATCH_SOURCES ON
  BATCH_SIZE 1

  sm100_bssp_gemm_nvf4_nvf4_f32_f16_nvf4_o_tnt_streamk.cu
  sm100_bssp_gemm_mxf8_mxf8_f32_f16_mxf8_q_tnt_streamk.cu
)

endif()
