
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


set(CUTLASS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_custom_target(cutlass_examples)
add_custom_target(test_examples)

function(cutlass_example_add_executable NAME)

  set(options)
  set(oneValueArgs DISABLE_TESTS)
  set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED __DISABLE_TESTS)
    set(__DISABLE_TESTS OFF)
  endif()

  cutlass_add_executable(${NAME} ${__UNPARSED_ARGUMENTS} BATCH_SOURCES OFF)

  add_dependencies(cutlass_examples ${NAME})

  target_link_libraries(
    ${NAME}
    PRIVATE
    CUTLASS
    cutlass_tools_util_includes
    $<$<BOOL:${CUTLASS_ENABLE_CUBLAS}>:nvidia::cublas>
    cuda
    )

  target_include_directories(
    ${NAME}
    PRIVATE
    ${CUTLASS_EXAMPLES_COMMON_SOURCE_DIR}
    ${CUTLASS_EXAMPLES_UTILS_DIR}
    )

  install(
    TARGETS ${NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

  cutlass_add_executable_tests(
    test_examples_${NAME} ${NAME}
    DEPENDS ${__DEPENDS}
    DEPENDEES test_examples ${__DEPENDEES}
    TEST_COMMAND_OPTIONS ${__TEST_COMMAND_OPTIONS}
    DISABLE_EXECUTABLE_INSTALL_RULE
    DISABLE_TESTS ${__DISABLE_TESTS}
    )

endfunction()

foreach(EXAMPLE
  00_basic_gemm
  01_cutlass_utilities
  02_dump_reg_shmem
  03_visualize_layout
  04_tile_iterator
  05_batched_gemm
  06_splitK_gemm
  07_volta_tensorop_gemm
  08_turing_tensorop_gemm
  09_turing_tensorop_conv2dfprop
  10_planar_complex
  11_planar_complex_array
  12_gemm_bias_relu
  13_two_tensor_op_fusion
  14_ampere_tf32_tensorop_gemm
  15_ampere_sparse_tensorop_gemm
  16_ampere_tensorop_conv2dfprop
  17_fprop_per_channel_bias
  18_ampere_fp64_tensorop_affine2_gemm
  19_tensorop_canonical
  20_simt_canonical
  21_quaternion_gemm
  22_quaternion_conv
  23_ampere_gemm_operand_reduction_fusion
  24_gemm_grouped
  25_ampere_fprop_mainloop_fusion
  26_ampere_wgrad_mainloop_fusion
  27_ampere_3xtf32_fast_accurate_tensorop_gemm
  28_ampere_3xtf32_fast_accurate_tensorop_fprop
  29_ampere_3xtf32_fast_accurate_tensorop_complex_gemm
  30_wgrad_split_k
  31_basic_syrk
  32_basic_trmm
  33_ampere_3xtf32_tensorop_symm
  34_transposed_conv2d
  35_gemm_softmax
  36_gather_scatter_fusion
  37_gemm_layernorm_gemm_fusion
  38_syr2k_grouped
  cute
  39_gemm_permute
  41_fused_multi_head_attention
  42_ampere_tensorop_group_conv
  43_ell_block_sparse_gemm
  45_dual_gemm
  46_depthwise_simt_conv2dfprop
  47_ampere_gemm_universal_streamk
  48_hopper_warp_specialized_gemm
  49_hopper_gemm_with_collective_builder
  50_hopper_gemm_with_epilogue_swizzle
  51_hopper_gett
  52_hopper_gather_scatter_fusion
  53_hopper_gemm_permute
  54_hopper_fp8_warp_specialized_gemm
  55_hopper_mixed_dtype_gemm
  56_hopper_ptr_array_batched_gemm
  57_hopper_grouped_gemm
  58_ada_fp8_gemm
  59_ampere_gather_scatter_conv
  61_hopper_gemm_with_topk_and_softmax
  62_hopper_sparse_gemm
  63_hopper_gemm_with_weight_prefetch
  64_ada_fp8_gemm_grouped
  65_distributed_gemm
  67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling
  68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling
  69_hopper_mixed_dtype_grouped_gemm
  70_blackwell_gemm                             
  71_blackwell_gemm_with_collective_builder     
  72_blackwell_narrow_precision_gemm            
  73_blackwell_gemm_preferred_cluster           
  74_blackwell_gemm_streamk                     
  75_blackwell_grouped_gemm                     
  76_blackwell_conv                             
  77_blackwell_fmha                             
  78_blackwell_emulated_bf16x9_gemm             
  79_blackwell_geforce_gemm             
  80_blackwell_geforce_sparse_gemm
  81_blackwell_gemm_blockwise 
  82_blackwell_distributed_gemm
  83_blackwell_sparse_gemm
  84_blackwell_narrow_precision_sparse_gemm
  86_blackwell_mixed_dtype_gemm                 
  87_blackwell_geforce_gemm_blockwise
  88_hopper_fmha
  89_sm103_fp4_ultra_gemm
  90_sm103_fp4_ultra_grouped_gemm
  91_fp4_gemv
  92_blackwell_moe_gemm
  )

  add_subdirectory(${EXAMPLE})

endforeach()

