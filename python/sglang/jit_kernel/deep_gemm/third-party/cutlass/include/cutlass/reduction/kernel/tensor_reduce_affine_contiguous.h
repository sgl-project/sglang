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

#include "cutlass/reduction/thread/reduction_operators.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reduction {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure
template <
  int Rank,                                   ///< Rank of source tensor (e.g. NDHWC => 5)
  int ReducedRank,                            ///< Rank of reduced tensor (i.e. number of outer ranks)
  typename ElementOutput,                     ///< Data type of output tensor
  typename ElementSource,                     ///< Data type of source tensor
  typename ReductionOp,                       ///< Reduction operator
  int VectorLength  = 1,                      ///< Vector length for memory
  typename ElementCompute = ElementOutput,    ///< Internal compute type - input type of reduction operation
  int Threads = 256,                          ///< Number of participating threads
  int BatchSize = 4                           ///< Number of elements to load per batch
>
struct TensorReductionAffineContiguousParams {

  static int const kRank = Rank;
  static int const kReducedRank = ReducedRank;
  static int const kVectorLength = VectorLength;
  static int const kInnerRank = kRank - kReducedRank;
  static int const kThreads = Threads;
  static int const kBatchSize = BatchSize;

  Coord<kRank> extent;                          /// Extent of source tensor
  FastDivmodU64 divmod[kRank - 1];              /// FastDivmod by each strided rank
  int64_t dst_stride[kReducedRank];             /// stride (units of bytes) - I, J
  int64_t src_stride[kRank - 1];                /// stride (units of bytes) - I, J, K
  int64_t workspace_stride;                     /// stride (units of bytes) between workspace
  int workspace_count;                          /// number of workspaces
  
  uint64_t inner_count;                          /// Number of elements in reduced index space
  uint64_t outer_count;                          /// Number of elements in outer index space

  ElementOutput * destination;                  /// Pointer to output tensor of rank kReducedRank
  ElementSource const * source;                 /// Pointer to source pointer of rank kRank
  ReductionOp reduction_op;                     /// Reduction operator
  ElementCompute reduction_identity;            /// Identity element used by reduction operator
  ElementCompute *device_workspace;             /// Pointer to device workspace for inter-CTA reductions

  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorReductionAffineContiguousParams() {

  }

  /// Ctor
  TensorReductionAffineContiguousParams(
    Coord<kRank> extent_,                       ///< Extent of source tensor
    ElementOutput * dst_ptr_,                   ///< Output tensor data
    int64_t dst_stride_[],                      ///< Stride (units of elements)
    ElementSource const * src_ptr_,             ///< Source tensor data
    int64_t src_stride_[],                      ///< Stride (units of elements)
    ElementCompute *device_workspace_,          ///< Pointer to device workspace for inter-CTA reductions
    int64_t workspace_stride_,                  ///< Stride between workspaces
    int workspace_count_,                       ///< Number of workspaces
    ReductionOp reduction_op_,                  ///< Reduction operator
    ElementCompute reduction_identity_ = ElementCompute() ///< Identity element used by reduction operator
  ):
    extent(extent_),
    inner_count(1),
    outer_count(1),
    destination(dst_ptr_),
    source(src_ptr_),
    device_workspace(device_workspace_),
    workspace_stride(workspace_stride_),
    workspace_count(workspace_count_),
    reduction_op(reduction_op_),
    reduction_identity(reduction_identity_) {

    // Initialize divisors for fast div-mod
    for (int p = 1; p < kRank; ++p) {
      divmod[p - 1] = FastDivmodU64(uint64_t(extent[p]));
    }

    int input_size_bits = sizeof_bits<ElementSource>::value;
    int output_size_bits = sizeof_bits<ElementOutput>::value;

    // Compute strides in units of bytes
    for (int p = 0; p < kReducedRank; ++p) {
      dst_stride[p] = dst_stride_[p] * output_size_bits / 8;
    }  

    for (int p = 0; p < kRank - 1; ++p) {
      src_stride[p] = src_stride_[p] * input_size_bits / 8;
    }

    // Compute number of elements in strided ranks
    for (int p = 0; p < kReducedRank; ++p) {
      outer_count *= uint64_t(extent[p]);
    }

    for (int p = 0; p < kInnerRank; ++p) {
      inner_count *= uint64_t(extent[kRank - 1 - p]);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to reduce a tensor with affine layout over a set of ranks *INCLUDING* the contiguous
/// rank. This leads to favorable vectorized memory accesses over the contiguous rank.
template <
  int Rank,                                   ///< Rank of source tensor (e.g. NDHWC => 5)
  int ReducedRank,                            ///< Rank of reduced tensor (includes contiguous, e.g. NC => 2)
  typename ElementOutput,                     ///< Data type of output tensor
  typename ElementSource,                     ///< Data type of source tensor
  typename ReductionOp,                       ///< Reduction operator
  int VectorLength  = 1,                      ///< Vector length for memory
  typename ElementCompute = ElementOutput,    ///< Internal compute type - input type of reduction operation
  int Threads = 256,                          ///< Number of participating threads
  int BatchSize = 4                           ///< Number of elements to load per batch
>
class TensorReductionAffineContiguous {
public:

  static int const kRank = Rank;
  static int const kReducedRank = ReducedRank;
  static int const kVectorLength = VectorLength;
  static int const kInnerRank = kRank - kReducedRank;
  static int const kThreads = Threads;
  static int const kBatchSize = BatchSize;
  using ComputeFragment = Array<ElementCompute, VectorLength>;
  using SourceFragment = AlignedArray<ElementSource, VectorLength>;
  using OutputFragment = AlignedArray<ElementOutput, VectorLength>;

  /// Shared memory allocation used for reduction within the CTA
  struct SharedStorage {
    Array<ElementCompute, kThreads * kVectorLength> workspace;
  };

  /// Parameters structure
  using Params = TensorReductionAffineContiguousParams<
    Rank,
    ReducedRank,
    ElementOutput,
    ElementSource,
    ReductionOp,
    VectorLength,
    ElementCompute,
    Threads,
    BatchSize
  >;

private:

  /// Computes the coordinate and offset of a given linear index
  CUTLASS_DEVICE
  void compute_inner_coord_and_offset_(
    Params const &params, 
    Coord<kInnerRank> & coord, 
    int64_t &src_offset,
    uint64_t linear_idx) const {

    // Decompose into a coordinate of rank <kInnerRank>
    coord = CoordinateDecomposition<kInnerRank>(linear_idx, &params.divmod[kRank - kInnerRank]);

    // Compute an offset using the souce stride
    src_offset = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kInnerRank - 1; ++i) {
      src_offset += coord[i] * params.src_stride[kReducedRank + i];
    }
    src_offset += coord[kInnerRank - 1] * sizeof_bits<ElementSource>::value / 8;
  }

  /// Computes the coordinate and offset of a given linear index
  CUTLASS_DEVICE
  void compute_outer_coord_and_offset_(
    Params const &params, 
    Coord<kReducedRank> & coord, 
    int64_t &dst_offset,
    int64_t &src_offset,
    uint64_t linear_idx) const {

    // Decompose into coordinate of rank <kReducedRank>
    coord = CoordinateDecomposition<kReducedRank>(linear_idx, params.divmod);

    // Compute offsets using destination and source strides
    dst_offset = 0;
    src_offset = 0;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kReducedRank; ++i) {
      dst_offset += params.dst_stride[i] * coord[i];
      src_offset += params.src_stride[i] * coord[i];
    }
  }

  /// Reduces over the reduction indices yielding a single element
  CUTLASS_DEVICE
  ElementCompute reduce_indices_(
    Params const &params,
    ElementCompute *threadblock_workspace,
    char const *src_byte_ptr,
    int coord_c) {

    NumericArrayConverter<ElementCompute, ElementSource, VectorLength> convert_source;
    ReductionOp reduction_op(params.reduction_op);

    //
    // Early exit or initialize to identity element
    //
    if (!params.inner_count) {
      return params.reduction_identity;
    }

    ComputeFragment accumulator;
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(accumulator.size()); ++i) {
      accumulator[i] = params.reduction_identity;
    }
    
    // Compute the coordinate of the first access    
    int64_t src_byte_offset = 0;
    Coord<kInnerRank> coord; 

    uint64_t linear_idx = (threadIdx.x + blockDim.x * threadIdx.z + blockDim.x * blockIdx.z * blockDim.z) * kVectorLength;
    compute_inner_coord_and_offset_(params, coord, src_byte_offset, linear_idx);

    // Load the first vector
    SourceFragment source_fragment[kBatchSize];
    
    bool not_done = true;

    // Iterate over vectors in a linearized reduction index space
    while (not_done) {

      bool guards[kBatchSize];

      // Issue a batch of loads
      CUTLASS_PRAGMA_UNROLL
      for (int b = 0; b < kBatchSize; ++b) {

        if (linear_idx < params.inner_count) {
          source_fragment[b] = *reinterpret_cast<SourceFragment const *>(src_byte_ptr + src_byte_offset);
          guards[b] = true;
        }
        else {
          guards[b] = false;
          not_done = false;
        }

        linear_idx += (blockDim.z * gridDim.z * blockDim.x) * kVectorLength;
        compute_inner_coord_and_offset_(params, coord, src_byte_offset, linear_idx);
      }

      // Perform a batch of reduction operations
      CUTLASS_PRAGMA_UNROLL
      for (int b = 0; b < kBatchSize; ++b) {
        if (guards[b]) {
          auto cvt = convert_source(source_fragment[b]);

          accumulator = cutlass::reduction::thread::detail::ApplyArrayOperator(
            reduction_op, 
            accumulator, 
            cvt);
        }
      }
    };

    //
    // Reduction of vectors to scalar
    //

    ElementCompute reduced_accumulator = accumulator[0];

    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < kVectorLength; ++i) {
      reduced_accumulator = reduction_op(reduced_accumulator, accumulator[i]);
    }

    //
    // Reduction within CTA across threadIdx.xz => threadIdx{.x = 0, .z = 0}
    //
    // This re-arranges data so threadIdx.y is effectively a row index and threadIdx.xz is a column
    //

    int thread_count = blockDim.x * blockDim.z;
    int thread_j = threadIdx.x + blockDim.x * threadIdx.z;
    int thread_i = threadIdx.y;

    ElementCompute *frag_ptr = reinterpret_cast<ElementCompute *>(threadblock_workspace) + thread_i * thread_count;

    frag_ptr[thread_j] = reduced_accumulator;

    //
    // Reduce
    //
    CUTLASS_PRAGMA_NO_UNROLL
    while (thread_count > 1) {
      thread_count /= 2;

      __syncthreads();

      if (thread_j < thread_count) {
        ElementCompute other = frag_ptr[thread_j + thread_count];

        reduced_accumulator = reduction_op(reduced_accumulator, other);

        frag_ptr[thread_j] = reduced_accumulator;
      }

      __syncthreads();
    }


    return reduced_accumulator;
  }

public:

  /// Perform a reduction
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    int coord_c = (blockIdx.x * blockDim.x + threadIdx.x) * kVectorLength;

    char const * src_byte_ptr = reinterpret_cast<char const *>(params.source);
    char * dst_byte_ptr = nullptr;

    // If performing a reduction across CTAs, redirect output to device workspace
    if (gridDim.z == 1) {
      dst_byte_ptr = reinterpret_cast<char *>(params.destination);
    }
    else {
      dst_byte_ptr = reinterpret_cast<char *>(params.device_workspace);
    }

    uint64_t idx_linear = blockIdx.y * blockDim.y + threadIdx.y;

    // Use modulo division to compute location
    Coord<kReducedRank> outer_coord;
    int64_t dst_byte_offset;
    int64_t src_byte_offset;

    compute_outer_coord_and_offset_(
      params, 
      outer_coord, 
      dst_byte_offset, 
      src_byte_offset, 
      idx_linear);

    if (gridDim.z == 1) {

      /// Complete the reduction with no workspace
      while (idx_linear < params.outer_count) {

        ElementCompute result = reduce_indices_(
          params, 
          shared_storage.workspace.data(),
          src_byte_ptr + src_byte_offset,
          coord_c);

        // Store the result after possible final reduction within the CTA
        if (threadIdx.z == 0 && threadIdx.x == 0) {

          // Convert to output type and store
          NumericConverter<ElementOutput, ElementCompute> convert_output;
          ElementOutput cvt = convert_output(result);

          *reinterpret_cast<ElementOutput *>(dst_byte_ptr + dst_byte_offset) = cvt;
        }

        __syncthreads();

        // Update indices and pointers
        idx_linear += gridDim.y * blockDim.y;

        compute_outer_coord_and_offset_(
          params, 
          outer_coord, 
          dst_byte_offset, 
          src_byte_offset, 
          idx_linear);

      } // while 
    }
    else {

      /// Complete the reduction with workspace
      while (idx_linear < params.outer_count) {

        ElementCompute result = reduce_indices_(
          params, 
          shared_storage.workspace.data(),
          src_byte_ptr + src_byte_offset,
          coord_c);

        int64_t byte_offset = 
          blockIdx.z * params.workspace_stride + idx_linear * sizeof_bits<ElementCompute>::value / 8;

        // Store the result for final reduction
        if (threadIdx.z == 0 && threadIdx.x == 0) {
          *reinterpret_cast<ElementCompute *>(dst_byte_ptr + byte_offset) = result;
        }

        __syncthreads();

        // Update indices and pointers
        idx_linear += gridDim.y * blockDim.y;

        compute_outer_coord_and_offset_(
          params, 
          outer_coord, 
          dst_byte_offset, 
          src_byte_offset, 
          idx_linear);
      } // while
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to perform final reduction
template <
  int Rank,                                   ///< Rank of source tensor (e.g. NDHWC => 5)
  int ReducedRank,                            ///< Rank of reduced tensor (includes contiguous, e.g. NC => 2)
  typename ElementOutput,                     ///< Data type of output tensor
  typename ElementSource,                     ///< Data type of source tensor
  typename ReductionOp,                       ///< Reduction operator
  int VectorLength  = 1,                      ///< Vector length for memory
  typename ElementCompute = ElementOutput,    ///< Internal compute type - input type of reduction operation
  int Threads = 256,                          ///< Number of participating threads
  int BatchSize = 4                           ///< Number of elements to load per batch
>
class TensorReductionAffineContiguousFinal {
public:

  static int const kRank = Rank;
  static int const kReducedRank = ReducedRank;
  static int const kVectorLength = VectorLength;
  static int const kInnerRank = kRank - kReducedRank;
  static int const kThreads = Threads;
  static int const kBatchSize = BatchSize;

  /// Shared memory
  struct SharedStorage { };

  /// Parameters structure
  using Params = TensorReductionAffineContiguousParams<
    Rank,
    ReducedRank,
    ElementOutput,
    ElementSource,
    ReductionOp,
    VectorLength,
    ElementCompute,
    Threads,
    BatchSize
  >;

private:

  /// Computes the coordinate and offset of a given linear index
  CUTLASS_DEVICE
  void compute_outer_coord_and_offset_(
    Params const &params, 
    Coord<kReducedRank> & coord, 
    int64_t &dst_offset,
    uint64_t linear_idx) const {

    // Decompose into coordinate of rank <kReducedRank>
    coord = CoordinateDecomposition<kReducedRank>(linear_idx, params.divmod);

    // Compute offsets using destination and source strides
    dst_offset = 0;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kReducedRank; ++i) {
      dst_offset += params.dst_stride[i] * coord[i];
    }
  }

  /// Reduces over the reduction indices
  CUTLASS_DEVICE
  ElementCompute reduce_indices_(
    Params const &params,
    ElementCompute const *device_workspace) {

    ReductionOp reduction_op(params.reduction_op);
    char const *src_byte_ptr = reinterpret_cast<char const *>(device_workspace);

    // Accumulated output
    ElementCompute accumulator = params.reduction_identity;

    for (int iter = 0; iter < params.workspace_count; ++iter) {
      ElementCompute workspace_item = *reinterpret_cast<ElementCompute const *>(src_byte_ptr);
      
      accumulator = reduction_op(accumulator, workspace_item);

      src_byte_ptr += params.workspace_stride;
    }

    return accumulator;
  }

public:

  //
  // Methods
  //

  /// Perform a reduction
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    uint64_t idx_linear = blockIdx.x * blockDim.x + threadIdx.x;

    char * dst_byte_ptr = reinterpret_cast<char *>(params.destination);

    // Use modulo division to compute location
    Coord<kReducedRank> outer_coord;
    int64_t dst_byte_offset;

    compute_outer_coord_and_offset_(
      params, 
      outer_coord, 
      dst_byte_offset, 
      idx_linear);

    /// Complete the reduction
    while (idx_linear < params.outer_count) {

      ElementCompute result = reduce_indices_(params, params.device_workspace + idx_linear);

      // Convert to output type and store
      NumericConverter<ElementOutput, ElementCompute> convert_output;

      *reinterpret_cast<ElementOutput *>(dst_byte_ptr + dst_byte_offset) = convert_output(result);

      // Update indices and pointers
      idx_linear += gridDim.x * blockDim.x;

      compute_outer_coord_and_offset_(
        params, 
        outer_coord, 
        dst_byte_offset, 
        idx_linear);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace reduction
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
