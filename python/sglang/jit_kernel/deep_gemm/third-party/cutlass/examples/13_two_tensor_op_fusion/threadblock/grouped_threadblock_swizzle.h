/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Implements several threadblock-swizzling functions for grouped kernels
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/grouped_problem_visitor.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"
#include "kernel/b2b_gemm_grouped_problem_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

struct GroupedThreadblockSwizzleBase {};

/// Helper for determining if a swizzling function is specialized for grouped operation
template <typename ThreadblockSwizzle>
struct IsGroupedSwizzle {
  static bool const value = cutlass::platform::is_base_of<GroupedThreadblockSwizzleBase, ThreadblockSwizzle>::value;
};

} // namespace detail

/// Swizzling function for grouped kernels
template <typename ProblemVisitor_>
struct GroupedThreadblockSwizzle : detail::GroupedThreadblockSwizzleBase {

  using ProblemVisitor = ProblemVisitor_;
  ProblemVisitor problem_visitor;

  CUTLASS_HOST_DEVICE
  GroupedThreadblockSwizzle(typename ProblemVisitor::Params& params,
                            typename ProblemVisitor::SharedStorage& shared_storage,
                            int block_idx) : problem_visitor(params, shared_storage, block_idx) {}

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int /*log_tile*/) const {
    GemmCoord problem_size = problem_visitor.problem_size();
    int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());
    GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

    return GemmCoord(int(threadblock_idx / grid_shape.n()),
                     int(threadblock_idx % grid_shape.n()),
                     0);
  }

  /// Dummy method to satisfy API for threadblock swizzling functions
  CUTLASS_HOST_DEVICE
  static int get_log_tile(GemmCoord /*tiled_shape*/) {
    return 0;
  }
};

template <
  typename ThreadblockShape,
  typename LayoutC,
  cutlass::gemm::kernel::GroupScheduleMode GroupScheduleMode_ = cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
  int PrefetchTileCount = 128,
  int ThreadCount = PrefetchTileCount>
struct B2bGemmGroupedThreadblockSwizzle : GroupedThreadblockSwizzle<
                                            cutlass::gemm::kernel::B2bGemmGroupedProblemVisitor<
                                              ThreadblockShape,
                                              GroupScheduleMode_,
                                              PrefetchTileCount,
                                              ThreadCount,
                                              platform::is_same<LayoutC, cutlass::layout::ColumnMajor>::value
                                            >
                                          > {
  using Base = GroupedThreadblockSwizzle<cutlass::gemm::kernel::B2bGemmGroupedProblemVisitor<
                                          ThreadblockShape,
                                          GroupScheduleMode_,
                                          PrefetchTileCount,
                                          ThreadCount,
                                          platform::is_same<LayoutC, cutlass::layout::ColumnMajor>::value>>;

  CUTLASS_HOST_DEVICE
  B2bGemmGroupedThreadblockSwizzle(typename Base::ProblemVisitor::Params& params,
                                   typename Base::ProblemVisitor::SharedStorage& shared_storage,
                                   int block_idx) : Base(params, shared_storage, block_idx) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
