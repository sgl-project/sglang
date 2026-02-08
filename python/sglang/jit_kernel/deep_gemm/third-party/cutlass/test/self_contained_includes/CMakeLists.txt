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

# The purpose of this target is to check if the following header files are self-contained,
# i.e. they can be included in a source file without needing to include other headers before it.

set(header_files_to_check
    # cutlass

    # cutlass/gemm/kernel
    cutlass/gemm/kernel/default_gemm.h
    cutlass/gemm/kernel/default_gemm_complex.h
    cutlass/gemm/kernel/gemm_universal_decl.h
    # cutlass/gemm/kernel/sm90_gemm_warpspecialized.hpp

    # cute
    cute/config.hpp
    cute/int_tuple.hpp
    cute/layout.hpp
    cute/layout_composed.hpp
    cute/pointer.hpp
    cute/pointer_base.hpp
    cute/pointer_flagged.hpp
    cute/pointer_swizzle.hpp
    cute/stride.hpp
    cute/swizzle.hpp
    cute/swizzle_layout.hpp
    cute/tensor.hpp
    cute/tensor_impl.hpp
    cute/underscore.hpp
    # cute/algorithm
    cute/algorithm/axpby.hpp
    cute/algorithm/clear.hpp
    # cute/algorithm/cooperative_copy.hpp
    cute/algorithm/cooperative_gemm.hpp
    # cute/algorithm/copy.hpp
    cute/algorithm/fill.hpp
    cute/algorithm/functional.hpp
    # cute/algorithm/gemm.hpp
    cute/algorithm/prefer.hpp
    # cute/algorithm/prefetch.hpp
    cute/algorithm/tensor_algorithms.hpp
    cute/algorithm/tuple_algorithms.hpp

    # cute/container
    cute/container/alignment.hpp
    cute/container/array.hpp
    cute/container/array_aligned.hpp
    cute/container/array_subbyte.hpp
    cute/container/bit_field.hpp
    cute/container/cuda_types.hpp
    cute/container/tuple.hpp
    cute/container/type_list.hpp

    # cute/numeric
    cute/numeric/arithmetic_tuple.hpp
    cute/numeric/complex.hpp
    cute/numeric/int.hpp
    cute/numeric/integer_sequence.hpp
    cute/numeric/integral_ratio.hpp
    cute/numeric/math.hpp
    cute/numeric/numeric_types.hpp
    cute/numeric/real.hpp
    cute/numeric/integral_constant.hpp

    # cute/util
    cute/util/debug.hpp
    cute/util/print.hpp
    cute/util/type_traits.hpp
    # cute/arch
    cute/arch/cluster_sm90.hpp
    cute/arch/copy.hpp
    cute/arch/copy_sm50.hpp
    cute/arch/copy_sm75.hpp
    cute/arch/copy_sm80.hpp
    cute/arch/copy_sm90.hpp
    cute/arch/copy_sm90_desc.hpp
    cute/arch/copy_sm90_tma.hpp
    cute/arch/mma_sm61.hpp
    cute/arch/mma_sm70.hpp
    cute/arch/mma_sm75.hpp
    cute/arch/mma_sm80.hpp
    cute/arch/mma_sm90.hpp
    cute/arch/mma_sm90_desc.hpp
    cute/arch/mma_sm90_gmma.hpp
    cute/arch/mma.hpp
    cute/arch/util.hpp

    cute/arch/cluster_sm100.hpp
    cute/arch/copy_sm100.hpp
    cute/arch/copy_sm100_tma.hpp
    cute/arch/mma_sm100.hpp
    cute/arch/mma_sm100_desc.hpp
    cute/arch/mma_sm100_umma.hpp
    # cute/arch/tmem_allocator_sm100.hpp

    # cute/atom
    # cute/atom/copy_atom.hpp
    # cute/atom/copy_traits.hpp
    # cute/atom/copy_traits_sm50.hpp
    # cute/atom/copy_traits_sm75.hpp
    # cute/atom/copy_traits_sm80.hpp
    # cute/atom/copy_traits_sm90.hpp
    # cute/atom/copy_traits_sm90_im2col.hpp
    # cute/atom/copy_traits_sm90_tma.hpp
    # cute/atom/copy_traits_sm90_tma_swizzle.hpp
    cute/atom/mma_atom.hpp
    cute/atom/mma_traits.hpp
    cute/atom/mma_traits_sm61.hpp
    cute/atom/mma_traits_sm70.hpp
    cute/atom/mma_traits_sm75.hpp
    cute/atom/mma_traits_sm80.hpp
    cute/atom/mma_traits_sm90.hpp
    cute/atom/mma_traits_sm90_gmma.hpp

    cute/atom/mma_traits_sm100.hpp
    cute/atom/partitioner.hpp

    # cutlass
    cutlass/aligned_buffer.h
    cutlass/array.h
    cutlass/array_planar_complex.h
    cutlass/array_subbyte.h
    cutlass/barrier.h
    cutlass/bfloat16.h
    cutlass/blas3.h
    cutlass/blas3_types.h
    cutlass/block_striped.h
    cutlass/cluster_launch.hpp
    cutlass/complex.h
    cutlass/constants.h
    cutlass/coord.h
    cutlass/core_io.h
    cutlass/cuda_host_adapter.hpp
    cutlass/cutlass.h
    cutlass/device_kernel.h
    cutlass/fast_math.h
    cutlass/float8.h
    # cutlass/floating_point_nvrtc.h
    cutlass/functional.h
    cutlass/gemm_coord.h
    cutlass/gemm_coord.hpp
    cutlass/half.h
    cutlass/integer_subbyte.h
    cutlass/kernel_hardware_info.h
    cutlass/kernel_hardware_info.hpp
    cutlass/kernel_launch.h
    cutlass/matrix.h
    cutlass/matrix_coord.h
    cutlass/matrix_shape.h
    cutlass/numeric_conversion.h
    cutlass/numeric_size.h
    cutlass/numeric_types.h
    cutlass/pitch_linear_coord.h
    cutlass/predicate_vector.h
    cutlass/quaternion.h
    cutlass/real.h
    cutlass/relatively_equal.h
    cutlass/semaphore.h
    cutlass/subbyte_reference.h
    cutlass/tensor_coord.h
    cutlass/tensor_ref.h
    cutlass/tensor_ref_planar_complex.h
    cutlass/tensor_view.h
    cutlass/tensor_view_planar_complex.h
    cutlass/tfloat32.h
    cutlass/trace.h
    cutlass/uint128.h
    cutlass/version.h
    cutlass/wmma_array.h
    cutlass/workspace.h
    cutlass/exmy_base.h
    cutlass/float_subbyte.h

    # cutlass/platform
    cutlass/platform/platform.h

    # cutlass/pipeline
    cutlass/pipeline/pipeline.hpp
    cutlass/pipeline/sm90_pipeline.hpp

    cutlass/pipeline/sm100_pipeline.hpp


    # cutlass/detail
    cutlass/detail/cluster.hpp
    cutlass/detail/collective.hpp
    cutlass/detail/dependent_false.hpp
    cutlass/detail/helper_macros.hpp
    cutlass/detail/layout.hpp
    cutlass/detail/mma.hpp

    cutlass/detail/sm100_blockscaled_layout.hpp


    # cutlass/arch
    cutlass/arch/arch.h
    cutlass/arch/barrier.h
    cutlass/arch/cache_operation.h
    cutlass/arch/config.h
    cutlass/arch/grid_dependency_control.h
    cutlass/arch/memory.h
    # cutlass/arch/memory_sm75.h
    # cutlass/arch/memory_sm80.h
    cutlass/arch/mma.h
    # cutlass/arch/mma_sm50.h
    # cutlass/arch/mma_sm60.h
    # cutlass/arch/mma_sm61.h
    # cutlass/arch/mma_sm70.h
    # cutlass/arch/mma_sm75.h
    # cutlass/arch/mma_sm80.h
    # cutlass/arch/mma_sm89.h
    # cutlass/arch/mma_sm90.h
    cutlass/arch/mma_sparse_sm80.h
    cutlass/arch/mma_sparse_sm89.h
    # cutlass/arch/simd.h
    # cutlass/arch/simd_sm60.h
    # cutlass/arch/simd_sm61.h
    cutlass/arch/reg_reconfig.h
    cutlass/arch/wmma.h
    # cutlass/arch/wmma_sm70.h
    # cutlass/arch/wmma_sm72.h
    # cutlass/arch/wmma_sm75.h
    # cutlass/arch/wmma_sm80.h
     # cutlass/layout
    cutlass/layout/layout.h
    cutlass/layout/matrix.h
    cutlass/layout/permute.h
    cutlass/layout/pitch_linear.h
    cutlass/layout/tensor.h
    cutlass/layout/tensor_op_multiplicand_sm70.h
    cutlass/layout/tensor_op_multiplicand_sm75.h
    cutlass/layout/tensor_op_multiplicand_sm80.h
    cutlass/layout/vector.h
)

# for each header in _header_files:
#   create a .cu file with the same name as the header's path, except with / replaced with %
#   have the .cu file include that header
set(_gen_source_files "")
foreach(header_file ${header_files_to_check})
  string(REPLACE "/" "%" header_file_esc ${header_file})

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${header_file_esc}.cu"
       "#include <${header_file}>")

  list(APPEND _gen_source_files
       "${CMAKE_CURRENT_BINARY_DIR}/${header_file_esc}.cu")
endforeach()

# build all generated .cu files into a single library
cutlass_add_library(test_self_contained_includes MODULE ${_gen_source_files})

