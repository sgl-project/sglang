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
    \brief Unit test for the OrderedSequenceBarrier class
*/

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

template<typename OrderedSequencer>
struct SharedStorage
{
  typename OrderedSequencer::SharedStorage storage;
};

// Goal of this kernel is to complete deadlock-free
template<int Stages, int GroupCount, int ThreadsPerGroup>
__global__ static
void ordered_sequence_device(uint32_t const num_iterations)
{

  extern __shared__ char shared_memory[];
  using SequenceBarrier = typename cutlass::OrderedSequenceBarrier<Stages, GroupCount>;
  using SmemStorage = SharedStorage<SequenceBarrier>;

  SmemStorage& shared_storage = *reinterpret_cast<SmemStorage*>(shared_memory);

  int group_idx = threadIdx.x / ThreadsPerGroup;

  typename SequenceBarrier::Params params;
  params.group_id = group_idx;              // sequence ID
  params.group_size = ThreadsPerGroup;      // Number of threads / participants in a group

  SequenceBarrier barrier(shared_storage.storage, params);

  // Ensure All CTAs in Cluster have completed init before issuing commits
  __syncthreads();
  cute::cluster_arrive_relaxed();  
  cute::cluster_wait();

  CUTLASS_PRAGMA_NO_UNROLL
  for (int i = 0; i < num_iterations; ++i){

    barrier.wait();
    // STAGE 1 CODE...
    #ifndef NDEBUG
    int thread_idx_in_group = threadIdx.x % ThreadsPerGroup;
    if (thread_idx_in_group == 0) {
      printf("STAGE 0 : Group_IDX : %d, id = %d, iter = %d, tidx = %d\n", group_idx, params.group_id, i, threadIdx.x);
    }
    #endif
    // Simulates long running stage
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    __nanosleep(100000);
    #endif
    barrier.arrive();

    barrier.wait();
    // STAGE 2 CODE...
    #ifndef NDEBUG
    if (thread_idx_in_group == 0) {
      printf("STAGE 1 : Group_IDX : %d, id = %d, iter = %d, tidx = %d\n", group_idx, params.group_id, i, threadIdx.x);
    }
    #endif
    // Simulates long running stage
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    __nanosleep(100000);
    #endif
    barrier.arrive();
  }

  // To make sure remote SMEM doesn't get destroyed
  cute::cluster_arrive();  
  cute::cluster_wait();  
}
/////////////////////////////////////////////////////

template<uint32_t Stages_, uint32_t GroupCount_>
struct PipelineTest {

  //
  // Data members
  //
  static constexpr uint32_t ThreadsPerGroup = 128;
  static constexpr uint32_t BlockSize = GroupCount_ * ThreadsPerGroup;
  static constexpr uint32_t Stages = Stages_;
  static constexpr uint32_t GroupCount = GroupCount_;
  using SequenceBarrier = typename cutlass::OrderedSequenceBarrier<Stages, GroupCount>;
  using SmemStorage = SharedStorage<SequenceBarrier>;

  //
  // Methods
  //

  // Run CuTe GEMM kernel
  cudaError_t run(uint32_t const kNumIters,
                  cudaStream_t stream = nullptr) {

    // Pipeline (multistage pipeline)
    auto cluster_shape = Shape<_1, _1, _1>{};

    //
    // Configure and launch
    //
    int iterations = 1;
    cudaError_t result;

    for (int iter = 0; iter < iterations; ++iter) {

      int smem_size = int(sizeof(SmemStorage));

      result = cudaFuncSetAttribute(
        ordered_sequence_device<Stages, GroupCount, ThreadsPerGroup>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

      // Launch a single Cluster, with 128 thread per CTA
      dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));    
      dim3 dimGrid(size<0>(cluster_shape), size<1>(cluster_shape), 1);    
      dim3 dimBlock(BlockSize,1,1);

      const void* kernel = (const void*)ordered_sequence_device<Stages, GroupCount, ThreadsPerGroup>;
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
TEST(SM90_Verify_OrderedSequence, Depth_2_Length_2) {
  Options options;
  static constexpr uint32_t GroupCount = 2;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, GroupCount>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_OrderedSequence, Depth_2_Length_3) {
  Options options;
  static constexpr uint32_t GroupCount = 3;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, GroupCount>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_OrderedSequence, Depth_2_Length_4) {
  Options options;
  static constexpr uint32_t GroupCount = 4;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, GroupCount>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_OrderedSequence, Depth_2_Length_5) {
  Options options;
  static constexpr uint32_t GroupCount = 5;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, GroupCount>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}
#endif
