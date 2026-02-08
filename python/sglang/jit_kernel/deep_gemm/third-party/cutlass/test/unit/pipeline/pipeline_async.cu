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
    \brief Unit test for the PipelineAsync class
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

using namespace cute;

//////////////////// KERNEL /////////////////////////

template <uint32_t Stages>
struct SharedStorage
{
  typename cutlass::PipelineAsync<Stages>::SharedStorage storage;
};

// Goal of this kernel is to complete deadlock-free
// Simple 1 producer warp, one consumer warp scenario
template <class ClusterShape, uint32_t NumStages>
__global__ static 
void pipeline_async_basic_device(uint32_t const num_iterations)
{

  extern __shared__ char shared_memory[];
  using MainloopPipeline = typename cutlass::PipelineAsync<NumStages>;
  using PipelineState = typename cutlass::PipelineState<NumStages>;

  using SharedStorage = SharedStorage<NumStages>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  int lane_predicate = cute::elect_one_sync();
  dim3 block_id_in_cluster = cute::block_id_in_cluster();
  
  // This example showcases 2 producer 1 consumer example 
  typename MainloopPipeline::Params params;
  params.producer_arv_count = 2;
  params.consumer_arv_count = 1;
  MainloopPipeline pipeline(shared_storage.storage, params);

  // Ensure All CTAs in Cluster have completed init before issuing commits
  cute::cluster_arrive_relaxed();  
  cute::cluster_wait();
  __syncthreads();


  if (lane_predicate) {
    // Producer Warps
    if (warp_idx==0 || warp_idx==1) {

      PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
      int prologue_iterations = min(NumStages, num_iterations);
      for ( int i = 0; i < prologue_iterations; ++i) {
        // Can also specify stage to commit directly
        pipeline.producer_commit(smem_pipe_write);
        ++smem_pipe_write;
      }

      int mainloop_iterations = num_iterations - prologue_iterations;

      for ( ; mainloop_iterations > 0; --mainloop_iterations) {
        pipeline.producer_acquire(smem_pipe_write);
        pipeline.producer_commit(smem_pipe_write);
        ++smem_pipe_write;
      }
    }
    else {
      PipelineState smem_pipe_read;
      for (int iter=0 ; iter < num_iterations; ++iter) {
        pipeline.consumer_wait(smem_pipe_read);
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }
    }
  }

  // To make sure remote SMEM doesn't get destroyed
  cute::cluster_arrive();  
  cute::cluster_wait();  
}
/////////////////////////////////////////////////////

template<uint32_t Stages_, typename ClusterShape_>
struct PipelineTest {

  //
  // Data members
  //
  static constexpr uint32_t Stages = Stages_;
  static constexpr uint32_t kBlockSize = 96;
  using ClusterShape = ClusterShape_;

  //
  // Methods
  //

  // Ctor
  PipelineTest() = default;


  // Run CuTe GEMM kernel
  cudaError_t run(uint32_t const kNumIters,
                  cudaStream_t stream = nullptr) {

    // Pipeline (multistage pipeline)
    auto cluster_shape = Shape<Int<ClusterShape::kM>, Int<ClusterShape::kN>, _1>{};

    //
    // Configure and launch
    //
    int iterations = 2;
    cudaError_t result;

    for (int iter = 0; iter < iterations; ++iter) {

      // Define the tiled MMA layout (static, 4warps)
      using MainloopPipeline = typename cutlass::PipelineAsync<Stages>;

      int smem_size = int(sizeof(SharedStorage<Stages>));

      result = cudaFuncSetAttribute(
        pipeline_async_basic_device<decltype(cluster_shape), Stages>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

      // Launch a single Cluster, with 128 thread per CTA
      dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), 1);    
      dim3 dimGrid(size<0>(cluster_shape), size<1>(cluster_shape), 1);    
      dim3 dimBlock(kBlockSize,1,1);

      const void* kernel = (const void*)pipeline_async_basic_device<decltype(cluster_shape), Stages>;
      int iters = kNumIters;
      void* kernel_params[] = {reinterpret_cast<void*>(&iters)};
      cutlass::ClusterLauncher::launch(dimGrid, dimCluster, dimBlock, smem_size, stream, kernel, kernel_params);
  
    } // profiling loop ends

    result = cudaDeviceSynchronize();

    if (result != cudaSuccess) {
      std::cerr << "Error: cudaDeviceSynchronize() failed" << std::endl;
      return result;
    }

    return cudaSuccess;
  }

};

#if CUDA_12_0_SM90_FEATURES_SUPPORTED
TEST(SM90_Verify_PipelineAsync, Cluster1x1_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster1x1_Stage5) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr uint32_t Stages = 5;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster1x1_Stage10) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr uint32_t Stages = 10;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster2x2_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 2, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster2x2_Stage5) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 2, 1>;
  static constexpr uint32_t Stages = 5;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster2x2_Stage10) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 2, 1>;
  static constexpr uint32_t Stages = 10;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster1x2_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 2, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster1x2_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 2, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster1x2_Stage10) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 2, 1>;
  static constexpr uint32_t Stages = 10;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster2x1_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 1, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster2x1_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 1, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x1_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 1, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x1_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 1, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster1x4_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 4, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster1x4_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 4, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster2x4_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 4, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster2x4_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 4, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x2_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 2, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x2_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 2, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x4_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x4_Stage3) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 3;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x4_Stage4) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 4;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x4_Stage5) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 5;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x4_Stage6) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 6;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x4_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x4_Stage8) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 8;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x4_Stage9) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 9;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x4_Stage10) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 10;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineAsync, Cluster4x4_Stage11) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 11;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}
#endif
