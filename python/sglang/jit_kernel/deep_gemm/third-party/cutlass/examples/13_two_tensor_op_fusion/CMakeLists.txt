
# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


include_directories(
  .
)

add_custom_target(13_fused_two_gemms)

add_custom_target(13_fused_two_convs)

add_custom_target(13_two_tensor_op_fusion
DEPENDS 13_fused_two_gemms
        13_fused_two_convs
)

foreach(FUSION_CONV_EXAMPLE
  fused_two_convs_f16_sm75_rf
  fused_two_convs_f16_sm75_shmem
  fused_two_convs_f16_sm80_rf
  fused_two_convs_f16_sm80_shmem
  fused_two_convs_s8_sm75_rf
  fused_two_convs_s8_sm75_shmem
  fused_two_convs_s8_sm80_rf
  fused_two_convs_s8_sm80_shmem
)

  cutlass_example_add_executable(
    13_${FUSION_CONV_EXAMPLE}
    ${FUSION_CONV_EXAMPLE}.cu
  )

  add_dependencies(13_fused_two_convs 13_${FUSION_CONV_EXAMPLE})

endforeach()

foreach(FUSION_GEMM_EXAMPLE
  fused_two_gemms_f16_sm75_rf
  fused_two_gemms_f16_sm75_shmem
  fused_two_gemms_grouped_f16_sm80_rf
  fused_two_gemms_f16_sm80_rf
  fused_two_gemms_f16_sm80_shmem
  fused_two_gemms_s8_sm75_rf
  fused_two_gemms_s8_sm75_shmem
  fused_two_gemms_s8_sm80_rf
  fused_two_gemms_s8_sm80_shmem
)
  cutlass_example_add_executable(
    13_${FUSION_GEMM_EXAMPLE}
    ${FUSION_GEMM_EXAMPLE}.cu
  )

  add_dependencies(13_fused_two_gemms 13_${FUSION_GEMM_EXAMPLE})

endforeach()
