/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Kernel performing a reduction over one or more ranks of an affine tensor
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/fast_math.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/device_kernel.h"

#include "cutlass/reduction/kernel/tensor_reduce_affine_contiguous.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reduction {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tensor reduction operator on layouts which are affine
template <
  int Rank,                                   ///< Rank of source tensor (e.g. NDHWC => 5)
  int ReducedRank,                            ///< Rank of reduced tensor (e.g. ND => 2)
  typename ElementOutput_,
  typename ElementSource_,
  typename ReductionOp_,
  int VectorLength  = 1,
  typename ElementCompute_ = ElementOutput_,
  int Threads = 256,                          ///< Number of participating threads
  int BatchSize = 4                           ///< Number of elements to load per batch
>
struct TensorReductionAffineContiguous {

  static int const kRank = Rank;
  static int const kReducedRank = ReducedRank;
  static int const kVectorLength = VectorLength;
  static int const kInnerRank = kRank - kReducedRank;
  static int const kThreads = Threads;
  static int const kBatchSize = BatchSize;

  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ReductionOp = ReductionOp_;
  using ElementCompute = ElementCompute_;

  //
  // Data members
  //

  /// Internal status field
  Status status;

  /// Extent of tensor in source layout
  Coord<kRank> extent;

  /// Number of points in the outer index space
  int64_t outer_count;

  /// Number of elements in the inner index space
  int64_t inner_count;

  /// Number of workspaces needed
  int workspace_count;

  /// CUDA Grid shape (.x => contiguous, .y => outer, .z => inner)
  dim3 grid_shape;

  /// CUDA Threadblock shape (.x => contiguous, .y => outer, .z => inner)
  dim3 threadblock_shape;

  /// CUDA grid shape for the final reduction step if needed
  dim3 grid_final;

  /// CUDA threadblock shape for the final reduction step if needed
  dim3 threadblock_final;

private:
  //
  // Methods
  //

  /// Helper to reshape 'count' such that it is less than 2 x 'ext'
  static int reshape_pow2(int ext, int count) {
    if (ext > count) {
      return 1;
    }
    int x = 1;
    for (; count >= ext * 2; ) {
      count >>= 1;
      x <<= 1;
    }
    return x;
  }

public:

  /// Default ctor
  TensorReductionAffineContiguous():
    status(Status::kErrorInvalidProblem),
    extent(),
    outer_count(0),
    inner_count(0),
    workspace_count(0),
    grid_shape(0, 0, 0),
    threadblock_shape(0, 0, 0) { }

  /// Constructor
  TensorReductionAffineContiguous(
    Coord<kRank> extent_,
    int target_threadblock_count = 128
  ):
    status(Status::kSuccess),
    extent(extent_), 
    outer_count(0),
    inner_count(0),
    workspace_count(0) {

    //
    // Plan the parallel mapping strategy.
    //

    outer_count = 1;
    inner_count = 1;

    // Compute number of elements in strided ranks
    for (int p = 0; p < kReducedRank; ++p) {
      outer_count *= extent[p];
    }

    for (int p = 0; p < kInnerRank; ++p) {
      inner_count *= extent[kReducedRank + p];
    }

    int cta_count_x = 1;
    int cta_count_y = 1;
    int cta_count_z = 1;

    int cta_threads_x = kThreads;
    int cta_threads_y = 1;
    int cta_threads_z = 1;

    // Determine CTA shape
    int64_t inner_vector_count = inner_count / kVectorLength;

    // Priority 1. Assign threadblocks to outer indices if possible
    if (outer_count > target_threadblock_count) {
      cta_count_x = 1;
      cta_count_y = target_threadblock_count;
      cta_count_z = 1;
    }
    else {

      cta_count_y = int(outer_count);
      int remaining_ctas = target_threadblock_count / cta_count_y;

      // Priority 2. Assign inner dimensions to one CTA
      if (inner_vector_count > cta_threads_x) {
        int64_t cta_z_bound = inner_vector_count / cta_threads_x;
        if (cta_z_bound > remaining_ctas) {
          cta_count_z = remaining_ctas;
        }
        else {
          cta_count_z = int(cta_z_bound);
        }
      }
      else {
        cta_threads_x = reshape_pow2(int(inner_vector_count), cta_threads_x);
        cta_count_z = 1;
      }
    }

    grid_shape = dim3(cta_count_x, cta_count_y, cta_count_z);
    threadblock_shape = dim3(cta_threads_x, cta_threads_y, cta_threads_z);

    workspace_count = (cta_count_z > 1 ? cta_count_z : 0);

    // Determine shape of final reduction kernel if needed
    if (workspace_count) {

      int final_threads = kThreads;
      int final_ctas = 1;

      if (outer_count > kThreads) {
        final_ctas = int(outer_count + kThreads - 1) / kThreads;
      }
      else {
        final_threads = int(outer_count);
      }

      grid_final = dim3(final_ctas, 1, 1);
      threadblock_final = dim3(final_threads, 1, 1); 
    }
    else {
      grid_final = dim3(0, 0, 0);
      threadblock_final = dim3(0, 0, 0);
    }
  }

  /// Simple check to verify the object is initialized correctly
  bool good() const {
    return status == Status::kSuccess;
  }

  /// Size (in bytes) of <outer_count> workspace elements which are densely packed together
  int64_t workspace_stride() const {
    
    // Error condition
    if (!good()) {
      return 0;
    }

    return outer_count * sizeof_bits<ElementCompute>::value / 8;
  }

  /// Returns the size (in bytes) of a temporary workspace needed for reduction across CTAs
  int64_t workspace_size() const {

    // Error condition
    if (!good()) {
      return 0;
    }

    // No reduction across CTAs
    if (grid_shape.z == 1) {
      return 0;
    }

    return workspace_stride() * grid_shape.z;
  }

  /// Performs a reduction
  Status reduce(
    ElementOutput *dst_ptr,                       ///< Pointer to destination tensor
    int64_t dst_stride[],                         ///< Stride vector (of length kReducedRank - 1)
    ElementSource const *src_ptr,                 ///< Pointer to source tensor
    int64_t src_stride[],                         ///< Stride vector (of length kRank - 1)
    void *device_workspace_ptr = nullptr,         ///< Device workspace
    ElementCompute reduction_identity = ElementCompute(), ///< Reduction identity element
    ReductionOp reduction_op = ReductionOp(),     ///< Reduction operator
    cudaStream_t stream = nullptr) {              ///< CUDA Stream into which all kernels are launched

    // Initial status check
    if (!good()) {
      return status;
    }

    // Guard against null workspace
    if (workspace_count > 1 && device_workspace_ptr == nullptr) {
      return Status::kErrorWorkspaceNull;
    }

    // Define reduction kernel
    using ReductionKernel = kernel::TensorReductionAffineContiguous<
      kRank,
      kReducedRank,
      ElementOutput, 
      ElementSource, 
      ReductionOp, 
      kVectorLength,
      ElementCompute,
      kThreads>;

    using FinalReductionKernel = kernel::TensorReductionAffineContiguousFinal<
      kRank,
      kReducedRank,
      ElementOutput, 
      ElementSource, 
      ReductionOp, 
      kVectorLength,
      ElementCompute,
      kThreads>;

    using Params = typename ReductionKernel::Params;

    // Construct the parameters
    Params params(
      extent, 
      dst_ptr,
      dst_stride, 
      src_ptr,
      src_stride,
      static_cast<ElementCompute *>(device_workspace_ptr),
      workspace_stride(),
      workspace_count,
      reduction_op,
      reduction_identity);

    // Shared memory size
    int shared_mem_bytes = sizeof(typename ReductionKernel::SharedStorage);

    // Launch the kernel
    cutlass::arch::synclog_setup();
    Kernel<ReductionKernel><<< grid_shape, threadblock_shape, shared_mem_bytes, stream >>>(params);

    // Check error condition
    if (cudaPeekAtLastError() == cudaSuccess) {
      status = Status::kSuccess;
    }
    else {
      status = Status::kErrorInternal;
    }

    // Final reduction kernel
    if (workspace_count) {
      Kernel<FinalReductionKernel><<< grid_final, threadblock_final, 0, stream >>>(params);
    }

    // Check error condition
    if (cudaPeekAtLastError() == cudaSuccess) {
      status = Status::kSuccess;
    }
    else {
      status = Status::kErrorInternal;
    }

    return status;
  }

  /// Helper to use overloaded function call operator
  Status operator()(
    ElementOutput *dst_ptr,                       ///< Pointer to destination tensor
    int64_t dst_stride[],                         ///< Stride vector (of length kReducedRank - 1)
    ElementSource const *src_ptr,                 ///< Pointer to source tensor
    int64_t src_stride[],                         ///< Stride vector (of length kRank - 1)
    void *device_workspace_ptr = nullptr,         ///< Pointer to device workspace
    ElementCompute reduction_identity = ElementCompute(), ///< Reduction identity element
    ReductionOp reduction_op = ReductionOp(),     ///< Reduction operator
    cudaStream_t stream = nullptr) {              ///< CUDA Stream into which all kernels are launched

    return reduce(dst_ptr, dst_stride, src_ptr, src_stride, device_workspace_ptr, reduction_identity, reduction_op, stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace reduction
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
