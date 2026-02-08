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
  \brief Epilogue for threadblock scoped GEMMs.

  This does not attempt to target any particular output layout. Instead, each threadblock
  streams out its accumulator elements using 128b store operations. This assumes all threadblocks
  have unique output tiles.

  The target data layout is:
  - threadblock indices mapped to linear offsets as (m, n, k), where m is fastest-changing
  - threadblock output space partitioned into warps; each warp's region is contiguous
  - per-thread accumulators partitioned into 128b accesses
  - output memory striped across the threads of a warp

  This enables very fast streaming of data, completely limited by the memory system. No predication
  or data exchange is performed, and each threadblock is assumed to have a full region of memory
  to write to.

  This epilogue establishes an upper bound for epilogue performance and is suitable for
  reductions across the GEMM K dimension which require a separate workspace.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Shape_,      ///< shape of accumulator tile (concept: MatrixShape)
  int WarpCount,        ///< number of warps
  typename FragmentC_   ///< warp-level GEMM operator (concept: gemm::warp::Mma)
>
class EpilogueWorkspace {
public:

  using Shape = Shape_;
  using FragmentC = FragmentC_;
  using ElementC = typename FragmentC::value_type;

  static int const kWarpCount = WarpCount;

  /// Optimize for 128b accesses
  static int const kAccessSizeInBits = 128;

  /// Warp size from the perspective of memory operations
  static int const kWarpSize = 32;

  /// Vector length of accesses
  static int const kElementsPerAccess = 
    kAccessSizeInBits / sizeof_bits<ElementC>::value;

  /// Number of stores per thread
  static int const kIterations = FragmentC::kElements / kElementsPerAccess;

  static_assert(
    !(FragmentC::kElements % kElementsPerAccess), 
    "The number of accumulators must be divisible by the access size.");

  /// Total number of vectorized accesses in warp (in units of vector)
  static int const kWarpAccesses = kIterations * kWarpSize;

  /// Total number of vectorized accesses in threadblock tile (in units of vector)
  static int const kThreadblockAccesses = kWarpAccesses * kWarpCount;

  /// Parameters structure
  struct Params {

    /// Pointer to C matrix
    ElementC *ptr_C;

    /// Stride between tiles along the GEMM N dimension (in units of vectors)
    int stride_n;

    /// Stride between tiles along the GEMM K dimension (in units of vectors)
    int stride_k;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(
      ElementC *ptr_C,   ///< Pointer to C matrix
      int stride_n_,      ///< Stride between tiles along the GEMM N dimension (in units of ElementC)
      int stride_k_       ///< Stride between tiles along the GEMM K dimension (in units of ElementC)
    ):
      ptr_C(ptr_C), stride_n(stride_n_ / kElementsPerAccess), stride_k(stride_k_ / kElementsPerAccess) {

    }
  };

  /// Shared storage allocation needed by the epilogue
  struct SharedStorage {
    // Intentionally empty
  };

private:

  struct alignas((kAccessSizeInBits / 8)) AccessType {
    Array<ElementC, kElementsPerAccess> storage;
  };

  /// Constant reference to parameters object
  AccessType *pointer_;

  /// Stride between tiles along the n dimension (in vectors)
  int stride_n_;

  /// Stride between tiles along the k dimension (in vectors)
  int stride_k_;

public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueWorkspace(
    Params const &params,     ///< Host-constructable params object
    SharedStorage &,          ///< Shared storage object
    int warp_idx,             ///< ID of warp within threadblock
    int lane_idx              ///< Id of thread within warp

  ):
    pointer_(reinterpret_cast<AccessType *>(params.ptr_C)),
    stride_n_(params.stride_n), 
    stride_k_(params.stride_k) {

    // Add per-thread offset
    pointer_ += lane_idx + warp_idx * kWarpAccesses;
  }

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
    cutlass::gemm::GemmCoord problem_size,       ///< Problem size of GEMM (units of ElementC)
    cutlass::gemm::GemmCoord tb_tile_coord,      ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
    FragmentC const &accum) {     ///< Accumulator tile
    
    // Compute offset for entire threadblock (note, per-thread offset has been folded in already)
    AccessType *pointer = pointer_ + 
      tb_tile_coord.m() * kThreadblockAccesses + 
      tb_tile_coord.n() * stride_n_ +
      tb_tile_coord.k() * stride_k_;

    // Cast to vectorized view of accumulator fragments
    AccessType const * src_pointer = reinterpret_cast<AccessType const *>(&accum);

    // Write out accumulators at full speed
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kIterations; ++i) {
      pointer[i * kWarpSize] = src_pointer[i];
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
