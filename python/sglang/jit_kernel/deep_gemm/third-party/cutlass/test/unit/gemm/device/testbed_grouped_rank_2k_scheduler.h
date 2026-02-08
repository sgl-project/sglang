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
    \brief Tests for grouped Rank2K problem visitors
*/

#pragma once

#include <iostream>
#include <numeric>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/rank_2k_grouped_problem_visitor.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/device_kernel.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////
// Use simple problem visitor as a baseline
template <typename ProblemSizeHelper,
          typename ThreadblockShape,
          int PrefetchTileCount,
          int ThreadCount,
          cutlass::FillMode FillModeC>
struct BaselineProblemVisitor : public cutlass::gemm::kernel::BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape> {
  using Base = cutlass::gemm::kernel::BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape>;
  using Params = typename Base::Params;
  static int const kThreadCount = ThreadCount;
  static cutlass::FillMode const kFillModeC = FillModeC;

  struct SharedStorage {};

  int32_t tile_count_sum;
  SharedStorage &shared_storage;

  //
  // Methods
  //
  CUTLASS_DEVICE
  BaselineProblemVisitor(
    Params const &params_,
    SharedStorage &shared_storage_,
    int32_t block_idx
  ): Base(params_, block_idx),
  shared_storage(shared_storage_)
  {
    cutlass::gemm::GemmCoord problem = this->problem_size();
    cutlass::gemm::GemmCoord  grid = this->grid_shape(problem);
    tile_count_sum = this->tile_count(grid);
  }

  CUTLASS_DEVICE
  bool next_tile() {
    if (this->tile_idx < tile_count_sum) {
      return true;
    }

    do {
      ++this->problem_idx;

      if (this->problem_idx >= this->params.problem_count) {
        return false;
      }

      cutlass::gemm::GemmCoord problem = this->problem_size();
      cutlass::gemm::GemmCoord  grid = this->grid_shape(problem);

      this->problem_tile_start = tile_count_sum;
      tile_count_sum += this->tile_count(grid);

    } while (tile_count_sum <= this->tile_idx);

    return true;
  }

  static size_t get_workspace_size(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count) {
    return 0;
  }

  static void host_precompute(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              void* host_workspace_ptr) {}

  CUTLASS_DEVICE
  cutlass::gemm::GemmCoord threadblock_offset(int32_t threadblock_id) const {
    int32_t macro_id = threadblock_id / ProblemSizeHelper::OffsetHelper::kThreadblockSkewRatio;
    int32_t macro_row = ceil(cutlass::fast_sqrt((2*macro_id) + 2.25) - 0.5) - 1;
    int32_t macro_col = macro_id - (((macro_row+1) * macro_row)/2);

    if (FillModeC == cutlass::FillMode::kUpper) {
      cutlass::swap(macro_row, macro_col);
    }

    int32_t row = ProblemSizeHelper::OffsetHelper::macro_row_to_row(macro_row, threadblock_id);
    int32_t col = ProblemSizeHelper::OffsetHelper::macro_col_to_col(macro_col, threadblock_id);

    return cutlass::gemm::GemmCoord(row, col, 0);
  }
};

template <typename ProblemVisitor>
struct ProblemVisitorKernel {
  struct SharedStorage {
    typename ProblemVisitor::SharedStorage problem_visitor;
  };

  struct Params {
    typename ProblemVisitor::Params problem_visitor_params;
    int32_t* visited_problems_ptr;
    int32_t* visited_tiles_ptr;
    int32_t visits_per_block;

    Params():
      visited_problems_ptr(nullptr),
      visited_tiles_ptr(nullptr),
      visits_per_block(0) {}

    Params(typename ProblemVisitor::Params problem_visitor_params_,
           int32_t* visited_problems_ptr_,
           int32_t* visited_tiles_ptr_,
           int32_t visits_per_block_):
      problem_visitor_params(problem_visitor_params_),
      visited_problems_ptr(visited_problems_ptr_),
      visited_tiles_ptr(visited_tiles_ptr_),
      visits_per_block(visits_per_block_) {}
  };

  CUTLASS_DEVICE
  void operator()(const Params& params, SharedStorage &shared_storage) {
    int32_t store_offset = params.visits_per_block * blockIdx.x;
    ProblemVisitor problem_visitor(params.problem_visitor_params,
                                   shared_storage.problem_visitor,
                                   blockIdx.x);

    while (problem_visitor.next_tile()) {
      cutlass::gemm::GemmCoord problem_size = problem_visitor.problem_size();
      int32_t problem_idx = problem_visitor.problem_index();
      int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());

      cutlass::gemm::GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);
      cutlass::gemm::GemmCoord tile_offset = problem_visitor.threadblock_offset(threadblock_idx);

      problem_visitor.advance(gridDim.x);

      //
      // Early exit conditions
      //   1) Out of range
      //   2) Upper-triangular block in lower-triangular problem
      //   3) Lower-triangular block in upper-triangular problem
      //

      if (grid_shape.m() <= tile_offset.m() ||
          grid_shape.n() <= tile_offset.n()) {
        continue;
      }

      if (ProblemVisitor::kFillModeC == cutlass::FillMode::kLower &&
          (tile_offset.m() + 1) * ProblemVisitor::ThreadblockShape::kM <= tile_offset.n() * ProblemVisitor::ThreadblockShape::kN) {
        continue;
      }

      if (ProblemVisitor::kFillModeC == cutlass::FillMode::kUpper &&
          tile_offset.m() * ProblemVisitor::ThreadblockShape::kM >= (tile_offset.n() + 1) * ProblemVisitor::ThreadblockShape::kN) {
        continue;
      }

      if (threadIdx.x == 0) {
        params.visited_problems_ptr[store_offset] = problem_idx;
        params.visited_tiles_ptr[store_offset] = threadblock_idx;
        ++store_offset;
      }
    }
  }
};

template <typename ProblemVisitor>
struct ProblemVisitorRunner {
  using BaseKernel = ProblemVisitorKernel<ProblemVisitor>;
  using Params = typename BaseKernel::Params;

  Params params;
  std::vector<cutlass::gemm::GemmCoord> host_problem_sizes;
  int32_t problem_count;
  int32_t threadblock_count;
  int32_t visits_per_block;
  cutlass::DeviceAllocation<int32_t> visited_problems;
  cutlass::DeviceAllocation<int32_t> visited_tiles;
  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> device_problem_sizes;
  cutlass::DeviceAllocation<uint8_t> workspace;
  std::vector<int32_t> host_visited_problems;
  std::vector<int32_t> host_visited_tiles;

  ProblemVisitorRunner(const std::vector<cutlass::gemm::GemmCoord>& host_problem_sizes_,
                       int32_t threadblock_count_):
      host_problem_sizes(host_problem_sizes_),
      problem_count(int32_t(host_problem_sizes_.size())),
      threadblock_count(threadblock_count_) {}

  /// Initializes GEMM state from arguments.
  cutlass::Status initialize() {
    size_t workspace_bytes = ProblemVisitor::get_workspace_size(
                                host_problem_sizes.data(),
                                problem_count,
                                threadblock_count);

    workspace.reset(workspace_bytes);
    std::vector<uint8_t> host_workspace(workspace_bytes);

    int32_t tile_count = ProblemVisitor::group_tile_count(host_problem_sizes.data(), problem_count);

    ProblemVisitor::host_precompute(host_problem_sizes.data(), problem_count,
                                    threadblock_count, host_workspace.data());

    workspace.copy_from_host(host_workspace.data(), workspace_bytes);

    device_problem_sizes.reset(problem_count);
    device_problem_sizes.copy_from_host(host_problem_sizes.data(), problem_count);

    visits_per_block = (tile_count - 1 + threadblock_count) / threadblock_count;
    int32_t total_visits = visits_per_block * threadblock_count;

    visited_problems.reset(total_visits);
    visited_tiles.reset(total_visits);
    host_visited_problems.resize(total_visits);
    host_visited_tiles.resize(total_visits);

    cudaError_t result = cudaMemset(visited_problems.get(), -1, sizeof(int32_t) * total_visits);
    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    result = cudaMemset(visited_tiles.get(), -1, sizeof(int32_t) * total_visits);
    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    typename ProblemVisitor::Params pv_params(device_problem_sizes.get(), problem_count, workspace.get(), tile_count);
    params = Params(pv_params, visited_problems.get(), visited_tiles.get(), visits_per_block);

    return cutlass::Status::kSuccess;
  }

  bool verify() {
    // Sort by problem size and then by threadblock_idx
    std::vector<int32_t> indices(host_visited_problems.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(indices.begin(), indices.end(),
      [&](int32_t i1, int32_t i2) {
        if (host_visited_problems[i1] == host_visited_problems[i2]) {
          return host_visited_tiles[i1] < host_visited_tiles[i2];
        }
        return host_visited_problems[i1] < host_visited_problems[i2];
      });

    int32_t idx = 0;

    // Skip any entries that were not visited
    while (host_visited_problems[indices[idx]] == -1) {
      ++idx;
    }

    // Check that each problem visited has the tiles we expect
    for (int32_t problem_idx = 0; problem_idx < problem_count; ++problem_idx) {
      auto problem = host_problem_sizes[problem_idx];
      ProblemVisitor::possibly_transpose_problem(problem);
      int32_t problem_tiles = ProblemVisitor::tile_count(ProblemVisitor::grid_shape(problem));
      for (int i = 0; i < problem_tiles; ++i) {
        EXPECT_EQ(problem_idx, host_visited_problems[indices[idx]]);
        EXPECT_EQ(i, host_visited_tiles[indices[idx]]);
        ++idx;
      }
    }

    return true;
  }

  bool run(bool skip_tile_check=false, cudaStream_t stream = nullptr) {
    cutlass::Status status = initialize();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Initialization failed" << std::endl;
      return false;
    }

    dim3 grid(threadblock_count, 1, 1);
    dim3 block(ProblemVisitor::kThreadCount, 1, 1);
    int smem_size = int(sizeof(typename BaseKernel::SharedStorage));

    cutlass::Kernel<BaseKernel><<<grid, block, smem_size, stream>>>(params);

    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
      std::cerr << "grid launch failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "cudaDeviceSynchronize failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    visited_problems.copy_to_host(host_visited_problems.data());
    visited_tiles.copy_to_host(host_visited_tiles.data());

    if (skip_tile_check) {
      return true;
    }

    return verify();
  }
};

template <typename ThreadblockShape,
          int PrefetchTileCount,
          int ThreadCount,
          cutlass::FillMode FillModeC,
          cutlass::gemm::kernel::GroupScheduleMode GroupScheduleMode0,
          cutlass::gemm::kernel::GroupScheduleMode... Args>
struct TestbedGroupedRank2KScheduler {

  using BaselinePV = BaselineProblemVisitor<cutlass::gemm::kernel::detail::Rank2KGroupedProblemSizeHelper<ThreadblockShape>,
                                            ThreadblockShape,
                                            PrefetchTileCount,
                                            ThreadCount,
                                            FillModeC>;

  //
  // Data members
  //

  // Whether to skip checking that the tiles are visited as expected. This is useful
  // in cases where ThreadblockShape::kM != ThreadblockShape::kN, for which the grouped
  // Rank2K scheduler may assign out-of-bounds tiles that will cause a threadblock to
  // exit early, but which are difficult to detect in tests without reimplementing
  // this functionality.
  bool skip_tile_check;
  uint32_t seed;
  int problem_count;
  int threadblock_count;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes_host;

  //
  // Methods
  //

  TestbedGroupedRank2KScheduler(bool skip_tile_check_=false, uint32_t seed_ = 3080):
    skip_tile_check(skip_tile_check_), seed(seed_) { srand(seed); }

  /// Initializes data structures
  void initialize(int32_t scale_factor) {

    //
    // Choose random problem sizes
    //

    problem_sizes_host.clear();
    problem_sizes_host.resize(problem_count);

    for (int32_t i = 0; i < problem_count; ++i) {
      int n = scale_factor * (rand() % 64) + 24;

      cutlass::gemm::GemmCoord problem(
        n,
        n,
        scale_factor * (rand() % 64) + 24);

      problem_sizes_host.at(i) = problem;
    }
  }

  template <cutlass::gemm::kernel::GroupScheduleMode GroupScheduleMode_>
  void compare_visitors(const ProblemVisitorRunner<BaselinePV>& baseline_runner) {
    using PV = cutlass::gemm::kernel::Rank2KGroupedProblemVisitor<
                                         ThreadblockShape,
                                         GroupScheduleMode_,
                                         PrefetchTileCount,
                                         ThreadCount,
                                         FillModeC>;
    ProblemVisitorRunner<PV> runner(problem_sizes_host, threadblock_count);
    EXPECT_TRUE(runner.run(skip_tile_check));

    // Check that this problem visitor visits the same problems and tiles as the baseline
    EXPECT_EQ(baseline_runner.host_visited_problems, runner.host_visited_problems);
    EXPECT_EQ(baseline_runner.host_visited_tiles, runner.host_visited_tiles);
  }

  template <cutlass::gemm::kernel::GroupScheduleMode GroupScheduleMode1_,
            cutlass::gemm::kernel::GroupScheduleMode GroupScheduleMode2_,
            cutlass::gemm::kernel::GroupScheduleMode... Rest>
  void compare_visitors(const ProblemVisitorRunner<BaselinePV>& baseline_runner) {
    // Compare the next visitor with the baseline visitor
    compare_visitors<GroupScheduleMode1_>(baseline_runner);

    // Recurse to compare the next visitors
    compare_visitors<GroupScheduleMode2_, Rest...>(baseline_runner);
  }

  /// Executes the test on all scheduler modes
  void run(int problem_count, int threadblock_count, int scale_factor=8) {

    this->problem_count = problem_count;
    this->threadblock_count = threadblock_count;

    // Initialize the problem
    initialize(scale_factor);

    // Run the baseline visitor to which we will compare all other visitors
    ProblemVisitorRunner<BaselinePV> baseline_runner(problem_sizes_host, threadblock_count);
    EXPECT_TRUE(baseline_runner.run(skip_tile_check));

    compare_visitors<Args...>(baseline_runner);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // device
} // gemm
} // test

/////////////////////////////////////////////////////////////////////////////////////////////////
