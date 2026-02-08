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
    \brief Tests that the stream-K scheduler covers the entire problem space.
*/

#include "cutlass/cluster_launch.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "../../common/cutlass_unit_test.h"

// Grids are launched with clusters enabled in these tests,
// so the CTK version must support cluster launching.
#if defined(CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED)

using namespace cute;
using ProblemShape_MNKL = Shape<int, int, int, int>;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel for getting each piece of work for a given block from the scheduler and logging
/// the K iterations visited by the block.
template <
  class Scheduler,
  class TileShape,
  class ClusterShape
>
__global__
void
run_scheduler(int* visit_counters, typename Scheduler::Params params, TileShape tile_shape, ClusterShape cluster_shape, ProblemShape_MNKL problem_shape_mnkl) {
  Scheduler scheduler{params};
  auto work_tile_info = scheduler.get_current_work();

  while (work_tile_info.is_valid()) {
    // Increment counters to indicate coverage
    auto tile_idx = Scheduler::output_tile_index(params, work_tile_info);
    auto offset = tile_idx * params.divmod_tiles_per_output_tile_.divisor + work_tile_info.K_idx;
    for (auto i = 0; i < work_tile_info.k_tile_count; ++i) {
      // Use atomicAdd because the visit counters are shared by multiple thread blocks.
      // While having more than one block increment the same counter indicates failure,
      // we need to ensure that this behavior is captured (by having both increments reflected).
      atomicAdd(visit_counters + offset + i, 1);
    }

    bool continue_current = scheduler.continue_current_work(work_tile_info);
    if (!continue_current) {
      scheduler.advance_to_next_work();
      work_tile_info = scheduler.get_current_work();
    }
  }
}

/// Host-side wrapper for launching the kernel to test the scheduler.
template <
  class TileShape,
  class ClusterShape,
  uint32_t NumMmaWarpGroups = 2
>
bool
test_scheduler(
  ProblemShape_MNKL problem_shape_mnkl,
  TileShape tile_shape,
  ClusterShape cluster_shape,
  int sm_count,
  int splits=1,
  bool expect_data_parallel=false) {

  using Scheduler = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>;

  cutlass::KernelHardwareInfo hw_info{0, sm_count};
  auto params = Scheduler::to_underlying_arguments(problem_shape_mnkl, tile_shape, cluster_shape, hw_info, {splits}, nullptr);

  typename Scheduler::Arguments args{};

  // Set up the grid for the problem
  dim3 grid = Scheduler::get_grid_shape(params, problem_shape_mnkl, tile_shape, cluster_shape, hw_info, args);

  auto print_info = [&]() {
    std::cout << "Failed with problem size "
      << size<0>(problem_shape_mnkl) << "x"
      << size<1>(problem_shape_mnkl) << "x"
      << size<2>(problem_shape_mnkl) << "x"
      << size<3>(problem_shape_mnkl)
      << " and grid size " << grid.x << "x"
      << grid.y << "x" << grid.z
      << " splits=" << params.divmod_splits_.divisor
      << " k_iter=" << params.divmod_tiles_per_output_tile_.divisor
      << " big_units_=" << params.big_units_
      << " big_groups_=" << params.big_groups_
      << " sk_tiles=" << params.sk_tiles_
      << " sk_units=" << params.sk_units_
      << " k_tiles_per_sk_unit=" << params.divmod_k_tiles_per_sk_unit_.divisor
      << " k_tiles_per_sk_big_unit=" << params.divmod_k_tiles_per_sk_big_unit_.divisor
      << " units_per_problem=" << params.units_per_problem_
      << " groups=" << params.divmod_sk_groups_.divisor << std::endl;
  };

  // If we expect the schedule to be data-parallel only, ensure that no stream-K tiles are launched.
  if (expect_data_parallel && params.sk_tiles_ != 0) {
    print_info();
    std::cout << "Expected stream-K to select a data-parallel decomposition." << std::endl;
    return false;
  }

  // Allocate counters indicating the number of times each k iteration of each output tile has been visited
  auto [blk_m, blk_n, blk_l] = Scheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
  auto total_counters = blk_m * blk_n * blk_l * params.divmod_tiles_per_output_tile_.divisor;
  cutlass::DeviceAllocation<int> visit_counters(total_counters);

  // Initialize counters to zero
  cudaError_t err = cudaMemset((void*)visit_counters.get(), 0, sizeof(int) * total_counters);
  if (err != cudaSuccess) {
    print_info();
    std::cout << __FILE__ << ":" << __LINE__ << " cudaMemset failed with error: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // Set up cluster and cluster launch. This is needed even for this simple kernel because
  // the SM90 scheduler needs to be able to query the CTA id within a cluster, which requires
  // explicitly launching with clusters.
  dim3 cluster{
    static_cast<uint32_t>(cute::get<0>(ClusterShape{})),
    static_cast<uint32_t>(cute::get<1>(ClusterShape{})),
    static_cast<uint32_t>(cute::get<2>(ClusterShape{}))
  };

  cudaLaunchConfig_t launch_config;
  launch_config.gridDim = grid;
  launch_config.blockDim = {1, 1, 1};
  launch_config.dynamicSmemBytes = 0;
  launch_config.stream = NULL;

  cudaLaunchAttribute launch_attribute[1];
  launch_attribute[0].id = cudaLaunchAttributeClusterDimension;
  launch_attribute[0].val.clusterDim.x = cluster.x;
  launch_attribute[0].val.clusterDim.y = cluster.y;
  launch_attribute[0].val.clusterDim.z = cluster.z;

  launch_config.attrs = launch_attribute;
  launch_config.numAttrs = 1;

  void const* kernel = (void const*) run_scheduler<Scheduler, TileShape, ClusterShape>;
  int* counters_ptr = visit_counters.get();
  void* kernel_params[] = {
    &counters_ptr,
    &params,
    &tile_shape,
    &cluster_shape,
    &problem_shape_mnkl
  };

  // Run the scheduler to completion and log visits to each k iteration
  err = cudaLaunchKernelExC(&launch_config, kernel, kernel_params);

  if (err != cudaSuccess) {
    print_info();
    std::cout << __FILE__ << ":" << __LINE__
              << " cudaLaunchKernelExC failed with error: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    print_info();
    std::cout << __FILE__ << ":" << __LINE__
              << " scheduler kernel failed with error: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // Copy visit counts back to host and ensure that all entries are ones
  std::vector<int> host_visit_counts(total_counters);
  visit_counters.copy_to_host(host_visit_counts.data());

  for (size_t i = 0; i < host_visit_counts.size(); ++i) {
    if (host_visit_counts[i] != 1) {
      print_info();
      std::cout << "Error at idx: " << i << ". Got count " << host_visit_counts[i] << std::endl;
      return false;
    }
  }

  return true;
}

/// Executes tests of the scheduler with a sweep across problem size K
template <
  class TileShape,
  class ClusterShape
>
bool sweep_k(
  ProblemShape_MNKL problem_shape_mnkl,
  TileShape tile_shape,
  ClusterShape cluster_shape,
  int sm_count,
  int splits=1,
  bool expect_data_parallel=false,
  int k_start=128,
  int k_stop=16384,
  int k_step=0) {

  if (k_step == 0) {
    k_step = 4 * cute::size<2>(tile_shape);
  }

  for (int k = k_start; k <= k_stop; k += k_step) {
    ProblemShape_MNKL problem{get<0>(problem_shape_mnkl), get<1>(problem_shape_mnkl), k, get<3>(problem_shape_mnkl)};
    bool passed = test_scheduler(problem, tile_shape, cluster_shape, sm_count, splits, expect_data_parallel);
    if (!passed) {
      return false;
    }
  }

  return true;
}

/// Executes tests of the scheduler that are expected to result in a data-parallel schedule.
/// This function assumes that the problem, tile, and cluster shape, alongside the SM count,
/// are such that the problem executes only full waves on the device.
template <
  class TileShape,
  class ClusterShape
>
bool test_data_parallel(
  int blocks_m,
  int blocks_n,
  TileShape tile_shape,
  ClusterShape cluster_shape,
  int sm_count) {

  // Since the configuration passed in executes only full waves, increasing
  // the batch dimension simply results in running more full waves.
  for (int l = 1; l < 4; ++l) {
    ProblemShape_MNKL problem_shape{
      size<0>(tile_shape) * blocks_m, size<1>(tile_shape) * blocks_n, 1, l};
    bool passed = sweep_k(problem_shape, tile_shape, cluster_shape, sm_count, /*splits=*/1, /*expect_data_parallel=*/true);

    if (!passed) {
      return false;
    }
  }
  return true;
}

/// Executes tests of the scheduler on the generic stream-K decomposition.
template <
  class TileShape,
  class ClusterShape
>
bool test_stream_k(
  TileShape tile_shape,
  ClusterShape cluster_shape,
  int sm_count) {

  int tile_m = size<0>(tile_shape);
  int tile_n = size<1>(tile_shape);

  for (int m_blocks = 1; m_blocks <= 24; ++m_blocks) {
    for (int n_blocks = 1; n_blocks <= 24; ++n_blocks) {
      for (int l = 1; l < 4; ++l) {
        ProblemShape_MNKL problem{m_blocks * tile_m, n_blocks * tile_n, 1, l};
        if (!sweep_k(problem, tile_shape, cluster_shape, sm_count)) {
          return false;
        }
      }
    }
  }

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_stream_k_scheduler, 256x128x64_2x1x1) {
  using TileShape_MNK = Shape<_256,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_1,_1>;

  TileShape_MNK tile_shape;
  ClusterShape_MNK cluster_shape;

  // Test various data-parallel cases
  EXPECT_TRUE(test_data_parallel(/*blocks_m=*/ 4, /*blocks_n=*/ 4, tile_shape, cluster_shape, /*sm_count=*/ 16));
  EXPECT_TRUE(test_data_parallel(/*blocks_m=*/16, /*blocks_n=*/ 4, tile_shape, cluster_shape, /*sm_count=*/ 64));
  EXPECT_TRUE(test_data_parallel(/*blocks_m=*/ 8, /*blocks_n=*/27, tile_shape, cluster_shape, /*sm_count=*/108));

  // Test various stream-K cases
  EXPECT_TRUE(test_stream_k(tile_shape, cluster_shape, /*sm_count=*/ 16));
  EXPECT_TRUE(test_stream_k(tile_shape, cluster_shape, /*sm_count=*/ 64));
  EXPECT_TRUE(test_stream_k(tile_shape, cluster_shape, /*sm_count=*/108));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_stream_k_scheduler, 128x128x64_2x1x1) {
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_1,_1>;

  TileShape_MNK tile_shape;
  ClusterShape_MNK cluster_shape;

  EXPECT_TRUE(test_scheduler({128, 512, 2048, 1}, tile_shape, cluster_shape, 114));
}

#endif // defined(CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED)

/////////////////////////////////////////////////////////////////////////////////////////////////
