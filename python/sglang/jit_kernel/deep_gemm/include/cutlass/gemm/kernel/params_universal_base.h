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
    \brief Base functionality for common types of universal GEMM kernel parameters
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/trace.h"
#include "cutlass/gemm/gemm.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace util {

template <class LayoutA, class LayoutB>
CUTLASS_HOST_DEVICE
static bool 
is_continous_k_aligned(GemmCoord problem_size, size_t alignmentA, size_t alignmentB) {
  return (platform::is_same<LayoutA, layout::RowMajor>::value && (problem_size.k() % alignmentA) == 0) ||
         (platform::is_same<LayoutB, layout::ColumnMajor>::value && (problem_size.k() % alignmentB) == 0);
}

}  // namespace util

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Argument structure
struct UniversalArgumentsBase
{
  //
  // Data members
  //

  GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;
  GemmCoord problem_size{};
  int batch_count{1};
  int64_t batch_stride_D{0};

  //
  // Methods
  //

  UniversalArgumentsBase() = default;

  /// constructs an arguments structure
  UniversalArgumentsBase(
    GemmUniversalMode mode,
    GemmCoord problem_size,
    int batch_count,
    int64_t batch_stride_D)
  :
    mode(mode),
    problem_size(problem_size),
    batch_count(batch_count),
    batch_stride_D(batch_stride_D)
  {
    CUTLASS_TRACE_HOST("GemmUniversal::Arguments::Arguments() - problem_size: " << problem_size);
  }
};


/// Parameters structure
template <
  typename ThreadblockSwizzle,
  typename ThreadblockShape,
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename LayoutA,
  typename LayoutB>
struct UniversalParamsBase
{
  //
  // Data members
  //

  GemmCoord problem_size{};
  GemmCoord grid_tiled_shape{};
  int swizzle_log_tile{0};
  GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;
  int batch_count {0};
  int gemm_k_size {0};
  int64_t batch_stride_D {0};
  int *semaphore = nullptr;


  //
  // Host dispatch API
  //

  /// Default constructor
  UniversalParamsBase() = default;

  /// Constructor
  UniversalParamsBase(
    UniversalArgumentsBase const &args, /// GEMM application arguments
    int device_sms,                     /// Number of SMs on the device
    int sm_occupancy)                   /// Kernel SM occupancy (in thread blocks)
  :
    problem_size(args.problem_size),
    mode(args.mode),
    batch_count(args.batch_count),
    batch_stride_D(args.batch_stride_D),
    semaphore(nullptr)
  {
    init_grid_tiled_shape();
  }

  /// Returns the workspace size (in bytes) needed for this problem geometry
  size_t get_workspace_size() const
  {
    size_t workspace_bytes = 0;
    if (mode == GemmUniversalMode::kGemmSplitKParallel)
    {
      // Split-K parallel always requires a temporary workspace
      workspace_bytes =
        sizeof(ElementC) *
        size_t(batch_stride_D) *
        size_t(grid_tiled_shape.k());
    }
    else if (mode == GemmUniversalMode::kGemm && grid_tiled_shape.k() > 1)
    {
      // Serial split-K only requires a temporary workspace if the number of partitions along the
      // GEMM K dimension is greater than one.
      workspace_bytes = sizeof(int) * size_t(grid_tiled_shape.m()) * size_t(grid_tiled_shape.n());
    }

    return workspace_bytes;
  }


  /// Assign and initialize the specified workspace buffer.  Assumes
  /// the memory allocated to workspace is at least as large as get_workspace_size().
  Status init_workspace(
    void *workspace,
    cudaStream_t stream = nullptr)
  {
    semaphore = static_cast<int *>(workspace);
    // Zero-initialize entire workspace
    if (semaphore)
    {
      size_t workspace_bytes = get_workspace_size();

      CUTLASS_TRACE_HOST("  Initialize " << workspace_bytes << " workspace bytes");

      cudaError_t result = cudaMemsetAsync(
        static_cast<int *>(workspace),
        0,
        workspace_bytes,
        stream);

      if (result != cudaSuccess) {
        CUTLASS_TRACE_HOST("  cudaMemsetAsync() returned error " << cudaGetErrorString(result));
        return Status::kErrorInternal;
      }
    }

    return Status::kSuccess;
  }


  /// Returns the GEMM volume in thread block tiles
  GemmCoord get_tiled_shape() const
  {
    return grid_tiled_shape;
  }


  /// Returns the total number of thread blocks to launch
  int get_grid_blocks() const
  {
    dim3 grid_dims = get_grid_dims();
    return grid_dims.x * grid_dims.y * grid_dims.z;
  }


  /// Returns the grid extents in thread blocks to launch
  dim3 get_grid_dims() const
  {
    return ThreadblockSwizzle().get_grid_shape(grid_tiled_shape);
  }

private:
  CUTLASS_HOST_DEVICE
  void init_grid_tiled_shape() {
    // Get GEMM volume in thread block tiles
    grid_tiled_shape = ThreadblockSwizzle::get_tiled_shape(
      problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      batch_count);

    swizzle_log_tile = ThreadblockSwizzle::get_log_tile(grid_tiled_shape);

    // Determine extent of K-dimension assigned to each block
    gemm_k_size = problem_size.k();

    if (mode == GemmUniversalMode::kGemm || mode == GemmUniversalMode::kGemmSplitKParallel)
    {
      static const uint32_t CACHELINE_BYTES = 128;
      static const size_t element_bytes_a = sizeof(ElementA);
      static const size_t element_bytes_b = sizeof(ElementB);
      static const size_t cacheline_elements_a = CACHELINE_BYTES / element_bytes_a;
      static const size_t cacheline_elements_b = CACHELINE_BYTES / element_bytes_b;

      const bool cacheline_alignment_needed =
          util::is_continous_k_aligned<LayoutA, LayoutB>(problem_size, cacheline_elements_a, cacheline_elements_b);

      int const kAlignK = const_max(
                                    const_max(128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value),
                                    cacheline_alignment_needed ? const_max(cacheline_elements_a, cacheline_elements_b) : 1);

      gemm_k_size = round_up(ceil_div(problem_size.k(), batch_count), kAlignK);
      if (gemm_k_size) {
        grid_tiled_shape.k() = ceil_div(problem_size.k(), gemm_k_size);
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
