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

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

/// Parameters structure
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
struct TensorReductionAffineStridedParams {

  static int const kRank = Rank;
  static int const kReducedRank = ReducedRank;
  static int const kVectorLength = VectorLength;
  static int const kInnerRank = kRank - kReducedRank;
  static int const kThreads = Threads;
  static int const kBatchSize = BatchSize;

  Coord<kRank> extent;                          /// Extent of source tensor
  FastDivmodU64 divmod[kRank - 1];              /// FastDivmod by each strided rank
  int64_t dst_stride[kReducedRank - 1];         /// stride (units of bytes) - I, J
  int64_t src_stride[kRank - 1];                /// stride (units of bytes) - I, J, K
  int64_t workspace_stride;                     /// stride (units of bytes) between workspace
  int64_t workspace_outer_stride;               /// stride (units of bytes) between 'rows' of the workspace
  int workspace_count;                          /// number of workspaces
  
  uint64_t inner_count;                          /// Number of elements in reduced index space
  uint64_t outer_count;                          /// Number of elements in outer index space

  ElementOutput * destination;                  /// Pointer to output tensor of rank kReducedRank
  ElementSource const * source;                 /// Pointer to source pointer of rank kRank
  ReductionOp reduction_op;                     /// Reduction operator
  ElementCompute reduction_identity;            /// Identity element for reduction operator
  ElementCompute *device_workspace;             /// Pointer to device workspace for inter-CTA reductions

  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorReductionAffineStridedParams() {

  }

  /// Ctor
  TensorReductionAffineStridedParams(
    Coord<kRank> extent_,                       ///< Extent of source tensor
    ElementOutput * dst_ptr_,                   ///< Output tensor data
    int64_t dst_stride_[],                      ///< Stride (units of elements)
    ElementSource const * src_ptr_,             ///< Source tensor data
    int64_t src_stride_[],                      ///< Stride (units of elements)
    ElementCompute *device_workspace_,          ///< Pointer to device workspace for inter-CTA reductions
    int64_t workspace_stride_,                  ///< Stride between workspaces
    int workspace_count_,                       ///< Number of workspaces
    ReductionOp reduction_op_,                  ///< Reduction operator
    ElementCompute reduction_identity_  = ElementCompute() ///< Identity element for reduction operator
  ):
    extent(extent_),
    inner_count(1),
    outer_count(1),
    destination(dst_ptr_),
    source(src_ptr_),
    device_workspace(device_workspace_),
    workspace_outer_stride(0),
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

    workspace_outer_stride = workspace_stride * workspace_count;

    // Compute strides in units of bytes
    for (int p = 0; p < kReducedRank - 1; ++p) {
      dst_stride[p] = dst_stride_[p] * output_size_bits / 8;
    }  

    for (int p = 0; p < kRank - 1; ++p) {
      src_stride[p] = src_stride_[p] * input_size_bits / 8;
    }

    // Compute number of elements in strided ranks
    for (int p = 0; p < kReducedRank - 1; ++p) {
      outer_count *= uint64_t(extent[p]);
    }

    for (int p = 0; p < kInnerRank; ++p) {
      inner_count *= uint64_t(extent[kReducedRank + p - 1]);
    }
  }
};

/// Kernel to reduce a tensor with affine layout over a set of ranks *EXCLUDING* the contiguous
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
class TensorReductionAffineStrided {
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
  using Params = TensorReductionAffineStridedParams<
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

    // Decompose into coordinate
    coord = CoordinateDecomposition<kInnerRank>(linear_idx, &params.divmod[kReducedRank - 1]);

    // Compute linear offset
    src_offset = 0;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kInnerRank; ++i) {
      src_offset += params.src_stride[kReducedRank + i - 1] * coord[i];
    }
  }

  /// Computes the coordinate and offset of a given linear index
  CUTLASS_DEVICE
  void compute_outer_coord_and_offset_(
    Params const &params, 
    Coord<kReducedRank - 1> & coord, 
    int64_t &dst_offset,
    int64_t &src_offset,
    uint64_t linear_idx) const {

    // Decompose linear coordinate
    coord = CoordinateDecomposition<kReducedRank - 1>(linear_idx, params.divmod);

    // Compute offset into tensors
    dst_offset = 0;
    src_offset = 0;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kReducedRank - 1; ++i) {
      dst_offset += params.dst_stride[i] * coord[i];
      src_offset += params.src_stride[i] * coord[i];
    }
  }

  /// Reduces over the reduction indices
  CUTLASS_DEVICE
  ComputeFragment reduce_indices_(
    Params const &params,
    ElementCompute *threadblock_workspace,
    char const *src_byte_ptr) {

    NumericArrayConverter<ElementCompute, ElementSource, VectorLength> convert_source;
    ReductionOp reduction_op(params.reduction_op);

    // Accumulated output
    ComputeFragment identity_frag;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(identity_frag.size()); ++i) {
      identity_frag[i] = params.reduction_identity;
    }

    if (!params.inner_count) {
      return identity_frag;
    }
    
    ComputeFragment accumulator = identity_frag;

    // Compute the coordinate of the first access    
    int64_t src_byte_offset = 0;
    Coord<kInnerRank> coord; 

    uint64_t linear_idx = threadIdx.z + blockIdx.z * blockDim.z;
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

        linear_idx += blockDim.z * gridDim.z;
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

    // Optional reduction within a CTA
    if (blockDim.z > 1) {

      // Linearized thread ID
      int thread_idx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

      // all threads store to workspace
      ComputeFragment *frag_ptr = reinterpret_cast<ComputeFragment *>(threadblock_workspace);

      frag_ptr[thread_idx] = accumulator;

      __syncthreads();

      if (threadIdx.z == 0) {
        // Load all additional block indices
        for (int z = 1; z < blockDim.z; ++z) {
          ComputeFragment frag = frag_ptr[thread_idx + z * blockDim.x * blockDim.y];

          accumulator = cutlass::reduction::thread::detail::ApplyArrayOperator(
            reduction_op, 
            accumulator, 
            frag);
        } 
      }

      __syncthreads();
    }

    return accumulator;
  }

public:

  /// Perform a reduction
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    int coord_c = (blockIdx.x * blockDim.x + threadIdx.x) * kVectorLength;

    char const * src_byte_ptr = reinterpret_cast<char const *>(params.source + coord_c);
    char * dst_byte_ptr = nullptr;

    // If performing a reduction across CTAs, redirect output to device workspace
    if (gridDim.z == 1) {
      dst_byte_ptr = reinterpret_cast<char *>(params.destination + coord_c);
    }
    else {
      dst_byte_ptr = reinterpret_cast<char *>(params.device_workspace + coord_c);
    }

    // If the C index is out of bounds, exit
    if (coord_c >= params.extent[kRank - 1]) {
      return;
    }

    int64_t idx_linear = blockIdx.y * blockDim.y + threadIdx.y;

    // Use modulo division to compute location
    Coord<kReducedRank - 1> outer_coord;
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

        ComputeFragment result;

        result = reduce_indices_(
          params, 
          shared_storage.workspace.data(),
          src_byte_ptr + src_byte_offset);

        // Store the result after possible final reduction within the CTA
        if (threadIdx.z == 0) {

          // Convert to output type and store
          NumericArrayConverter<ElementOutput, ElementCompute, VectorLength> convert_output;
          auto cvt = convert_output(result);

          *reinterpret_cast<OutputFragment *>(dst_byte_ptr + dst_byte_offset) = 
            reinterpret_cast<OutputFragment const &>(cvt);
        }

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

      /// Complete the reduction with a device workspace
      while (idx_linear < params.outer_count) {

        ComputeFragment result;

        result = reduce_indices_(
          params, 
          shared_storage.workspace.data(),
          src_byte_ptr + src_byte_offset);

        // Store the result after possible final reduction within the CTA
        if (threadIdx.z == 0) {

          int64_t byte_offset = 
            blockIdx.z * params.workspace_stride + idx_linear * params.workspace_outer_stride;

          // No conversion - store in compute type
          *reinterpret_cast<ComputeFragment *>(dst_byte_ptr + byte_offset) = 
            reinterpret_cast<ComputeFragment const &>(result);
        }

        // Update indices and pointers
        idx_linear += gridDim.y * blockDim.y;

        compute_outer_coord_and_offset_(
          params, 
          outer_coord, 
          dst_byte_offset, 
          src_byte_offset, 
          idx_linear);
        
      } // while (outer index)
    } // if ()
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
class TensorReductionAffineStridedFinal {
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

  /// Shared memory
  struct SharedStorage { };

  /// Parameters structure
  using Params = TensorReductionAffineStridedParams<
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
    Coord<kReducedRank - 1> & coord, 
    int64_t &dst_offset,
    uint64_t linear_idx) const {

    // Decompose linear index
    coord = CoordinateDecomposition<kReducedRank - 1>(linear_idx, params.divmod);

    // Compute tensor offset
    dst_offset = 0;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kReducedRank - 1; ++i) {
      dst_offset += params.dst_stride[i] * coord[i];
    }
  }

  /// Reduces over the reduction indices
  CUTLASS_DEVICE
  ComputeFragment reduce_indices_(
    Params const &params,
    char *src_byte_ptr) {

    ReductionOp reduction_op(params.reduction_op);

    // Accumulated output
    ComputeFragment identity_frag;
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(identity_frag.size()); ++i) {
      identity_frag[i] = params.reduction_identity;
    }

    ComputeFragment accumulator = identity_frag;
    ComputeFragment workspace_fragments[kBatchSize];

    // Partially unrolled loop
    for (int idx = 0; idx < params.workspace_count; idx += kBatchSize) {

      // Issue a batch of loads
      CUTLASS_PRAGMA_UNROLL
      for (int b = 0; b < kBatchSize; ++b) {
        if (idx + b < params.workspace_count) {
          workspace_fragments[b] = 
            *reinterpret_cast<ComputeFragment *>(src_byte_ptr);  
        }
        else {
          workspace_fragments[b] = identity_frag;
        }
        src_byte_ptr += + params.workspace_stride;
      }

      // Perform a reduction
      CUTLASS_PRAGMA_UNROLL
      for (int b = 0; b < kBatchSize; ++b) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kVectorLength; ++i) {
          accumulator[i] = reduction_op(accumulator[i], workspace_fragments[b][i]);
        }
      }
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

    int coord_c = (blockIdx.x * blockDim.x + threadIdx.x) * kVectorLength;

    char * src_byte_ptr = reinterpret_cast<char *>(params.device_workspace + coord_c);
    char * dst_byte_ptr = reinterpret_cast<char *>(params.destination + coord_c);

    // If the C index is out of bounds, exit
    if (coord_c >= params.extent[kRank - 1]) {
      return;
    }

    int64_t idx_linear = blockIdx.y * blockDim.y + threadIdx.y;

    // Use modulo division to compute location
    Coord<kReducedRank - 1> outer_coord;
    int64_t dst_byte_offset;

    compute_outer_coord_and_offset_(
      params, 
      outer_coord, 
      dst_byte_offset, 
      idx_linear);

    /// Complete the reduction
    while (idx_linear < params.outer_count) {

      int64_t src_byte_offset = idx_linear * params.workspace_outer_stride;

      ComputeFragment result = reduce_indices_(
        params, 
        src_byte_ptr + src_byte_offset);

      // Convert to output type and store
      NumericArrayConverter<ElementOutput, ElementCompute, VectorLength> convert_output;
      auto cvt = convert_output(result);

      *reinterpret_cast<OutputFragment *>(dst_byte_ptr + dst_byte_offset) = 
        reinterpret_cast<OutputFragment const &>(cvt);

      // Update indices and pointers
      idx_linear += gridDim.y * blockDim.y;

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
