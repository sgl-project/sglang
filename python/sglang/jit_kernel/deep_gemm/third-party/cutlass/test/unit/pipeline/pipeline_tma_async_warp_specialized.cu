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
    \brief Unit test for the PipelineTmaAsync class as it would be used in a Warp specialized loop
*/

#define KERNEL_DBG_TRACE false

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

#include "testbed.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/barrier.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"


using namespace cute;
using namespace cutlass;

//////////////////// KERNEL /////////////////////////

template <uint32_t Stages>
struct SharedStorage
{
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage storage ;
};

struct KernelParams
{
  uint32_t num_iterations;
  int* data_ptr;
};

// Goal of this kernel is to complete deadlock-free
template <typename ClusterShape, uint32_t Stages>
__launch_bounds__(384, 1)
__global__ static
void pipeline_device(KernelParams const kernel_params)
{
  extern __shared__ char shared_memory[];
  using MainloopPipeline = typename cutlass::PipelineTmaAsync<Stages>;  
  using PipelineState = typename cutlass::PipelineState<Stages>;  

  using SharedStorage = SharedStorage<Stages>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  [[maybe_unused]] auto cta_layout = Layout<ClusterShape>{};            // (m,n) -> cta_id
  int warp_group_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
  int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
  int warp_group_thread_idx = threadIdx.x % 128;
  dim3 block_id_in_cluster = cute::block_id_in_cluster();

  auto cluster_shape = ClusterShape{};

  // #Producers = #RowsInCluster + #ColsInCluster - 1 
  uint32_t const NumProducers = cute::size<0>(cluster_shape) + cute::size<1>(cluster_shape) - 1;
  uint32_t const TmaTransactionBytes = static_cast<uint32_t>(sizeof(uint32_t) * NumProducers);
  uint32_t const per_cta_bytes = sizeof(uint32_t);

  // mbarrier.init
  typename MainloopPipeline::Params params;
  params.transaction_bytes = TmaTransactionBytes;
  if (warp_group_idx == 0) {
    params.role = MainloopPipeline::ThreadCategory::Producer;
  }
  else {
    params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  params.is_leader = warp_group_thread_idx == 0;
  params.num_consumers = 128;

  MainloopPipeline pipeline(shared_storage.storage, params, cluster_shape);

  __syncthreads();

  // Ensure All CTAs in Cluster have completed init before issuing commits
  cute::cluster_arrive_relaxed();  
  cute::cluster_wait();


  // Producer WarpGroup
  if (warp_group_idx == 0) {
    cutlass::arch::warpgroup_reg_alloc<232>();

    int lane_predicate = cute::elect_one_sync();
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {

      int tma_k_prologue = min(Stages, kernel_params.num_iterations);

      // Simulating Prologue TMA Loads
      // For the DMA (prologue) - we start with an opposite phase - since we skip all waits
      // i.e., we know that the buffer is indeed empty
      PipelineState smem_pipe_write = make_producer_start_state<MainloopPipeline>();
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < tma_k_prologue; ++i) {
        pipeline.producer_acquire(smem_pipe_write);
        // Simulating cp.async.bulk.tensor behavior
        pipeline.producer_commit(smem_pipe_write, per_cta_bytes);
        ++smem_pipe_write;
      }
      int tma_k_iter = kernel_params.num_iterations - tma_k_prologue;

      // Simulating Mainloop TMA Loads
      CUTE_NO_UNROLL
      for ( ; tma_k_iter > 0; --tma_k_iter) {

        pipeline.producer_acquire(smem_pipe_write);

        // Simulating cp.async.bulk.tensor behavior
        pipeline.producer_commit(smem_pipe_write, per_cta_bytes);

        // Advance write stage
        ++smem_pipe_write;
      }

      // Tail Loop
      // Handles the case where we never enter the mainloop
      PipelineState tail = tma_k_prologue == Stages ? smem_pipe_write : PipelineState{};
      for ( int i = 0; i < tma_k_prologue; ++i) {
        pipeline.producer_acquire(tail);
        ++tail;
      }
    }
  // Consumer WarpGroup
  } else if(warp_group_idx == 1) {
    cutlass::arch::warpgroup_reg_alloc<232>();

    PipelineState smem_pipe_read;
    PipelineState smem_pipe_release;

    // simulates accumulators + extra reg. pressure
    int arr[168];

    // Init Shared Memory read stages & PhaseBit
    static constexpr uint32_t K_PIPE_MMAS = 1;
    static_assert( K_PIPE_MMAS < Stages, "ERROR : Too many MMAs in flight");

    // Total number of gemm iterations
    auto gemm_k_iterations  = kernel_params.num_iterations;

    // Simulating Prologue MMAs
    int mma_k_prologue = min(K_PIPE_MMAS, gemm_k_iterations);
    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < mma_k_prologue; ++iter) {
      pipeline.consumer_wait(smem_pipe_read);

      warpgroup_arrive();
      // GMMA would typically happen here

      ++smem_pipe_read;
    }
    gemm_k_iterations -= mma_k_prologue;

    // Simulating Mainloop MMAs
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; gemm_k_iterations > 0; --gemm_k_iterations) {

      /// Wait on the smem_pipe_read stage / phase
      pipeline.consumer_wait(smem_pipe_read);

      warpgroup_arrive();
      // GMMA would typically happen here

      // Dummy op - which will never happen
      // But simulates high register usage.
      CUTE_UNROLL
      for(int i = 0; i < 168; ++i){
        if (threadIdx.x > 256){
          arr[i] += kernel_params.data_ptr[i];
        }
      }

      pipeline.consumer_release(smem_pipe_release);

      // Advance stages
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    // Dummy op - which will never happen
    CUTE_UNROLL
    for(int i = 0; i < 168; ++i){
      if (threadIdx.x > 256){
        kernel_params.data_ptr[i] = arr[i];
      }
    }

    // Tail Loop
    for (int i = 0; i < K_PIPE_MMAS; ++i){
      pipeline.consumer_release(smem_pipe_release);
      ++smem_pipe_release;
    }

  // Warp-Group #2
  } else {
    cutlass::arch::warpgroup_reg_dealloc<40>();
  }
}
/////////////////////////////////////////////////////

/// Device NT GMMA + TMA specialized
template<uint32_t Stages_, typename ClusterShape_>
struct PipelineTest {

  //
  // Data members
  //
  static constexpr uint32_t Stages = Stages_;
  static constexpr uint32_t kBlockSize = 128 * 3;
  using ClusterShape = ClusterShape_;

  //
  // Methods
  //

  // Ctor
  PipelineTest(){};

  // Run CuTe GEMM kernel
  cudaError_t run(uint32_t const kNumIters,
                  cudaStream_t stream = 0) {

    float elapsed_ms = 0.0f;
    // Pipeline (multistage pipeline)
    [[maybe_unused]] auto num_stages = Int<Stages>{};
    auto cluster_shape = Shape<Int<ClusterShape::kM>, Int<ClusterShape::kN>, _1>{};

    //
    // Configure and launch
    //
    int iterations = 1;
    cudaEvent_t events[2];
    cudaError_t result;

    for (cudaEvent_t & event : events) {
      result = cudaEventCreate(&event);
      if (result != cudaSuccess) {
        std::cerr << "Error: Failed to create event.";
        return result;
      }
    }

    result = cudaEventRecord(events[0]);

    if (result != cudaSuccess) {
      std::cerr << "Error: Failed to record start event.";
      return result;
    }

    for (int iter = 0; iter < iterations; ++iter) {
    
      using MainloopPipeline = typename cutlass::PipelineTmaAsync<Stages>;

      int smem_size = int(sizeof(SharedStorage<Stages>));

      result = cudaFuncSetAttribute(
        pipeline_device<decltype(cluster_shape), Stages>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

      // Launch a single Cluster, with kBlockSize threads per CTA
      dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), 1);    
      dim3 dimGrid(size<0>(cluster_shape), size<1>(cluster_shape), 1);    
      dim3 dimBlock(kBlockSize,1,1);

      const void* kernel = (const void*)pipeline_device<decltype(cluster_shape), Stages>;
      KernelParams params{kNumIters, nullptr};
      void* kernel_params[] = {reinterpret_cast<void*>(&params)};
      cutlass::ClusterLauncher::launch(dimGrid, dimCluster, dimBlock, smem_size, stream, kernel, kernel_params);
  
    }

    result = cudaEventRecord(events[1]);

    if (result != cudaSuccess) {
      std::cerr << "Error: Failed to record stop event.";
      return result;
    }

    result = cudaDeviceSynchronize();

    if (result != cudaSuccess) {
      std::cerr << "Error: cudaDeviceSynchronize() failed" << std::endl;
      return result;
    }

    result = cudaEventElapsedTime(&elapsed_ms, events[0], events[1]);

    if (result != cudaSuccess) {
      std::cerr << "Failed to create event.";
      return result;
    }

    for (cudaEvent_t & event : events) {
      (void)cudaEventDestroy(event);
    }

    return cudaSuccess;
  }
};

#if CUDA_12_0_SM90_FEATURES_SUPPORTED
TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster1x1_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster1x1_Stage5) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr uint32_t Stages = 5;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster1x1_Stage10) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr uint32_t Stages = 10;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster2x2_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 2, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster2x2_Stage5) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 2, 1>;
  static constexpr uint32_t Stages = 5;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster2x2_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 2, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster4x4_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster4x4_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster2x1_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 1, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster2x1_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 1, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster1x2_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 2, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster1x2_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 2, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster4x1_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 1, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster4x1_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 1, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster1x4_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 4, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster1x4_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 4, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster2x4_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 4, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster2x4_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 4, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster4x2_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 2, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS, Cluster4x2_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 2, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}
#endif
