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
    \brief Unit test for the PipelineCLCFetchAsync class
*/

#define KERNEL_DBG_TRACE false

#include <cuda/atomic>
#include "../common/cutlass_unit_test.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>

#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/cluster_launch.hpp>

#include "cutlass/core_io.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include "testbed_cluster_launch_control.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/barrier.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"


using namespace cute;
using namespace cutlass;
using namespace cutlass::gemm::kernel::detail;

//////////////////// Shared Memory  /////////////////////////

template <uint32_t Stages, typename ClusterShape>
struct SharedStorage
{
  alignas(16) typename PersistentTileSchedulerSm100<ClusterShape, Stages>::CLCResponse clc_response[Stages];
  alignas(16) typename PipelineCLCFetchAsync<Stages, ClusterShape>::SharedStorage storage;
};

//////////////////// Kernel /////////////////////////
template <typename ClusterShape, uint32_t Stages>
__launch_bounds__(256, 1)
__global__ static
void pipeline_device(int *d_workerCount)
{
  extern __shared__ char shared_memory[];

  // single producer, multiple consumers
  // producer: WG0
  // consumer: WG1

  using SharedStorage = SharedStorage<Stages, ClusterShape>;
  using Scheduler = PersistentTileSchedulerSm100<ClusterShape, Stages>;
  using TileSchedulingPipeline = PipelineCLCFetchAsync<Stages, ClusterShape>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  // Logistics
  int warp_idx = canonical_warp_idx();
  auto cluster_shape = ClusterShape{};

  typename TileSchedulingPipeline::Params params;
  params.transaction_bytes = 16;

  constexpr int NUM_PRODUCER = 32;
  constexpr int NUM_CONSUMERS_PER_CTA = 32;
  params.consumer_arv_count = NUM_PRODUCER + NUM_CONSUMERS_PER_CTA * cute::size<0>(cluster_shape) * cute::size<1>(cluster_shape);
  params.producer_arv_count = 1;
  // Only the first CTA in the Cluster is producing.
  params.producer_blockid = 0;

  dim3 block_id_in_cluster = cute::block_id_in_cluster();
  // mbarrier.init
  TileSchedulingPipeline scheduler_pipeline(shared_storage.storage, params );

  // Ensure All CTAs in Cluster have completed init before issuing commits
  cute::cluster_arrive_relaxed();
  cute::cluster_wait();

  uint32_t is_first_block_in_cluster = block_id_in_cluster.x == 0 && block_id_in_cluster.y == 0;
  int lane_predicate = cute::elect_one_sync();

  uint32_t is_producer = (is_first_block_in_cluster && warp_idx == 0);
  uint32_t is_consumer = (warp_idx == 4);

  PipelineState<Stages> scheduler_pipe_state;
  PipelineState<Stages> scheduler_pipe_state_write = cutlass::make_producer_start_state<TileSchedulingPipeline>();
  typename Scheduler::WorkTileInfo work_tile_info = {
    static_cast<int32_t>(blockIdx.x),
    static_cast<int32_t>(blockIdx.y),
    static_cast<int32_t>(blockIdx.z),
    false
  };

  // Persistent loop
  do {
    // Producer
    if (is_producer) {
      // Only 1 thread of the entire cluster issues the query.
      uint32_t mbarrier_addr = scheduler_pipeline.producer_get_barrier(scheduler_pipe_state_write);

      // Wait for clcID buffer to become empty with a flipped phase
      scheduler_pipeline.producer_acquire(scheduler_pipe_state_write);

      if (cute::elect_one_sync()) {
        Scheduler::issue_clc_query(scheduler_pipe_state_write, mbarrier_addr, shared_storage.clc_response);
      }

      ++scheduler_pipe_state_write;
    }

    // Consumers
    if (is_consumer) {
      int linearCLC = work_tile_info.N_idx * gridDim.x + work_tile_info.M_idx;
      // Atomically increment the worker count for the linearCLC by 1.
      if (lane_predicate) {
        atomicAdd(&d_workerCount[linearCLC], 1);
      }
    }

    // Union of all consumers. Note that the producer here is its own consumer.
    if (is_producer || is_consumer) {
      scheduler_pipeline.consumer_wait(scheduler_pipe_state);
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&shared_storage.clc_response[scheduler_pipe_state.index()]);
      work_tile_info = Scheduler::work_tile_info_from_clc_response(smem_addr);
      scheduler_pipeline.consumer_release(scheduler_pipe_state);
      ++scheduler_pipe_state;

      // Add block offset since the scheduler works at cluster level. 
      dim3 block_id_in_cluster = cute::block_id_in_cluster();
      work_tile_info.M_idx += block_id_in_cluster.x;
      work_tile_info.N_idx += block_id_in_cluster.y;
      work_tile_info.L_idx += block_id_in_cluster.z;

    }
  } while (work_tile_info.is_valid_tile);

  // End of kernel
  cute::cluster_sync();
}
/////////////////////////////////////////////////////

template<uint32_t Stages_, typename ClusterShape_>
struct PipelineTest {

  //
  // Data members
  //
  static constexpr uint32_t Stages = Stages_;
  static constexpr uint32_t BlockSize = 128 * 2;
  using ClusterShape = ClusterShape_;

  //
  // Methods
  //

  bool check_results(int *h_workerCount, int size ) {
    for (int i = 0 ; i< size; i++ ){
      if ( h_workerCount[i] != 1 )
      {
        std::cout << "linearCLC " << i << " has worker count " << h_workerCount[i] << "\n";
        return false;
      }
    }
    return true;
  }

  // Run CuTe GEMM kernel
  cudaError_t run(bool &success, dim3 grid_dim,
                  cudaStream_t stream = 0 ) {

    //
    // Configure and launch
    //
    cudaError_t result;

    int smem_size = 192 * 1024;  // 192kB to force 1CTA/SM
    auto cluster_shape = Shape<Int<ClusterShape::kM>, Int<ClusterShape::kN>, _1>{};
    // Launch a single Cluster, with BlockSize threads per CTA
    dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), 1);
    dim3 dimGrid = grid_dim;
    dim3 dimBlock(BlockSize,1,1);

    result = cudaFuncSetAttribute(
                  pipeline_device<
                    decltype(cluster_shape),
                    Stages>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_size
                  );

    if (result != cudaSuccess) {
      std::cerr << "Error: Failed to set Shared Memory size." << std::endl;
      return result;
    }

    int array_size = dimGrid.x * dimGrid.y;
    int *d_workerCount, *h_workerCount;

    /* Allocate memory. workerCount[i] counts the number of worker(s) which work
       on linear t i.  The expectation is that workerCount[i] == 1 for all i.
    */
    h_workerCount = (int*)malloc(array_size * sizeof(int));

    result = cudaMalloc(&d_workerCount, array_size * sizeof(int));
    if (result != cudaSuccess) {
      std::cerr << "Failed to do cudaMalloc." << result << "\n";
      return result;
    }

    for(int i = 0 ; i < array_size; i++)
    {
      h_workerCount[i] = 0;  // Initialize workerCount[i] to 0 for all i.
    }

    result = cudaMemcpy(d_workerCount, h_workerCount, array_size * sizeof(int), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
      std::cerr << "Failed to do cudaMemcpy." << result << "\n";
      return result;
    }

    //  Extended launch API
    const void* kernel = (const void*)pipeline_device<decltype(cluster_shape), Stages>;
    void* kernel_params[] = {&d_workerCount};
    cutlass::ClusterLauncher::launch(dimGrid, dimCluster, dimBlock, smem_size, stream, kernel, kernel_params);

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error: cudaDeviceSynchronize() failed" << std::endl;
      return result;
    }

    result = cudaMemcpy(h_workerCount, d_workerCount, array_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
      std::cerr << "Failed to do cudaMemcpy." << result << "\n";
      return result;
    }

    success = check_results(h_workerCount, array_size);

    free(h_workerCount);

    result = cudaFree(d_workerCount);
    if (result != cudaSuccess) {
      std::cerr << "Failed to do cudaFree." << result << "\n";
      return result;
    }

    return cudaSuccess;
  }
};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
//Cluster1x2 Stage4
TEST(SM100_Verify_PipelineClusterLaunchControlAsync_WS, Cluster1x2_Stage4) {
  OptionsClusterLaunch options;
  options.grid_dim = {32,32,1};
  using ClusterShape = cutlass::gemm::GemmShape<1, 2, 1>;
  static constexpr uint32_t Stages = 4;
  using Test = PipelineTest<Stages, ClusterShape>;
  TestbedClusterLaunch<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

//Cluster2x1 Stage4
TEST(SM100_Verify_PipelineClusterLaunchControlAsync_WS, Cluster2x1_Stage4) {
  OptionsClusterLaunch options;
  options.grid_dim = {32,32,1};
  using ClusterShape = cutlass::gemm::GemmShape<2, 1, 1>;
  static constexpr uint32_t Stages = 4;
  using Test = PipelineTest<Stages, ClusterShape>;
  TestbedClusterLaunch<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

//Cluster2x2 Stage4
TEST(SM100_Verify_PipelineClusterLaunchControlAsync_WS, Cluster2x2_Stage4) {
  OptionsClusterLaunch options;
  options.grid_dim = {32,32,1};
  using ClusterShape = cutlass::gemm::GemmShape<2, 2, 1>;
  static constexpr uint32_t Stages = 4;
  using Test = PipelineTest<Stages, ClusterShape>;
  TestbedClusterLaunch<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

//Cluster1x1 Stage3
TEST(SM100_Verify_PipelineClusterLaunchControlAsync_WS, Cluster1x1_Stage3) {
  OptionsClusterLaunch options;
  options.grid_dim = {32,32,1};
  using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr uint32_t Stages = 3;
  using Test = PipelineTest<Stages, ClusterShape>;
  TestbedClusterLaunch<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

//Cluster1x4 Stage4
TEST(SM100_Verify_PipelineClusterLaunchControlAsync_WS, Cluster1x4_Stage4) {
  OptionsClusterLaunch options;
  options.grid_dim = {32,32,1};
  using ClusterShape = cutlass::gemm::GemmShape<1, 4, 1>;
  static constexpr uint32_t Stages = 4;
  using Test = PipelineTest<Stages, ClusterShape>;
  TestbedClusterLaunch<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

//Cluster4x1 Stage4
TEST(SM100_Verify_PipelineClusterLaunchControlAsync_WS, Cluster4x1_Stage4) {
  OptionsClusterLaunch options;
  options.grid_dim = {32,32,1};
  using ClusterShape = cutlass::gemm::GemmShape<4, 1, 1>;
  static constexpr uint32_t Stages = 4;
  using Test = PipelineTest<Stages, ClusterShape>;
  TestbedClusterLaunch<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

//Cluster2x4 Stage4
TEST(SM100_Verify_PipelineClusterLaunchControlAsync_WS, Cluster2x4_Stage4) {
  OptionsClusterLaunch options;
  options.grid_dim = {32,32,1};
  using ClusterShape = cutlass::gemm::GemmShape<2, 4, 1>;
  static constexpr uint32_t Stages = 4;
  using Test = PipelineTest<Stages, ClusterShape>;
  TestbedClusterLaunch<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

//Cluster4x2 Stage4
TEST(SM100_Verify_PipelineClusterLaunchControlAsync_WS, Cluster4x2_Stage4) {
  OptionsClusterLaunch options;
  options.grid_dim = {32,32,1};
  using ClusterShape = cutlass::gemm::GemmShape<4, 2, 1>;
  static constexpr uint32_t Stages = 4;
  using Test = PipelineTest<Stages, ClusterShape>;
  TestbedClusterLaunch<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

//Cluster4x4 Stage4
TEST(SM100_Verify_PipelineClusterLaunchControlAsync_WS, Cluster4x4_Stage4) {
  OptionsClusterLaunch options;
  options.grid_dim = {32,32,1};
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 4;
  using Test = PipelineTest<Stages, ClusterShape>;
  TestbedClusterLaunch<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}
#endif
