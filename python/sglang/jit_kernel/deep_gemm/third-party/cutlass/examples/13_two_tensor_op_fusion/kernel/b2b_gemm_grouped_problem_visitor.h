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
    \brief Scheduler for grouped B2b GEMMs
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/kernel/grouped_problem_visitor.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Visitor class to abstract away the algorithm for iterating over tiles
template <typename ThreadblockShape,
          GroupScheduleMode GroupScheduleMode_,
          int PrefetchTileCount,
          int ThreadCount,
          bool Transposed = false>
struct B2bGemmGroupedProblemVisitor : public GroupedProblemVisitor<
                                            detail::GemmGroupedProblemSizeHelper<ThreadblockShape, Transposed>,
                                            ThreadblockShape,
                                            GroupScheduleMode_,
                                            PrefetchTileCount,
                                            ThreadCount> {

  using ProblemSizeHelper = detail::GemmGroupedProblemSizeHelper<ThreadblockShape, Transposed>;
  using Base = GroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape, GroupScheduleMode_, PrefetchTileCount, ThreadCount>;
  using BaseParams = typename Base::Params;
  using SharedStorage = typename Base::SharedStorage;
  static bool const kTransposed = Transposed;

  cutlass::gemm::GemmCoord const *problem_sizes0;
  cutlass::gemm::GemmCoord const *problem_sizes1;

  struct Params {
    cutlass::gemm::GemmCoord const *problem_sizes0;
    cutlass::gemm::GemmCoord const *problem_sizes1;
    int32_t                         problem_count;
    void const                     *workspace;
    int32_t                         tile_count;

    //
    // Methods
    //

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(): problem_sizes0(nullptr), problem_sizes1(nullptr),
              problem_count(0), workspace(nullptr), tile_count(0) { }

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const *problem_sizes0,
      cutlass::gemm::GemmCoord const *problem_sizes1,
      int32_t                         problem_count,
      void const                     *workspace = nullptr,
      int32_t                         tile_count = 0
    ):
      problem_sizes0(problem_sizes0),
      problem_sizes1(problem_sizes1),
      problem_count(problem_count),
      workspace(workspace),
      tile_count(tile_count)
    {}

    /// Convert the B2b-GEMM-specific parameters to those used by the base class
    CUTLASS_HOST_DEVICE
    BaseParams to_base() const {
        return BaseParams(// Set problem_sizes as problem_sizes0 because these determine
                          // shape of the grid used in the non-grouped B2b GEMM
                          problem_sizes0,
                          problem_count,
                          workspace,
                          tile_count);
    }

  };

  //
  // Methods
  //
  CUTLASS_DEVICE
  B2bGemmGroupedProblemVisitor(
    Params const &params_,
    SharedStorage &shared_storage_, 
    int32_t block_idx
  ): Base (
        params_.to_base(),
        shared_storage_, block_idx),
     problem_sizes0(params_.problem_sizes0),
     problem_sizes1(params_.problem_sizes1)
  {}

  /// Returns the problem size 0 for the current problem
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size0() const {
    GemmCoord problem = problem_sizes0[this->problem_idx];
    ProblemSizeHelper::possibly_transpose_problem(problem);
    return problem;
  }

  /// Returns the problem size 1 for the current problem
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size1() const {
    GemmCoord problem = problem_sizes1[this->problem_idx];
    ProblemSizeHelper::possibly_transpose_problem(problem);
    return problem;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
