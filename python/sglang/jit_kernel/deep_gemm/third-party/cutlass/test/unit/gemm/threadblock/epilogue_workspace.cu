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
    \brief Unit tests for thread-level GEMM
*/

#include "../../common/cutlass_unit_test.h"

#include "cutlass/epilogue/epilogue_workspace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel computes accumulator data and stores it out
template <typename Epilogue>
__global__ void kernel_epilogue_workspace(typename Epilogue::Params params) {

  __shared__ typename Epilogue::SharedStorage shared_storage;

  int warp_id = threadIdx.y;
  int lane_id = threadIdx.x;

  Epilogue epilogue(params, shared_storage, warp_id, lane_id);

  //
  // Initialize accumulator tile
  //
  typename Epilogue::FragmentC accum;

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Epilogue::FragmentC::kElements; ++i) {
    accum[i] = Element(warp_id * blockDim.x + lane_id);
  }

  //
  // Efficient epilogue
  //

  cutlass::GemmCoord tb_tile_coord{blockIdx.x, blockIdx.y, 0};
  
  cutlass::GemmCoord problem_size = 
    tb_tile_coord * 
    cutlass::GemmCoord{Epilogue::Shape::kM, Epilogue::Shape::kN, 1};

  // Store accumulators
  epilogue(
    problem_size, 
    tb_tile_coord, 
    accum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_gemm_threadblock_epilogue_workspace, tensor_op_128x128_64x64) {

  //
  // Define an instance of the epilogue and see if it works
  //
  static int const kWarpCount = 4;
  static int const kWarpSize = 32;

  using Shape = cutlass::MatrixShape<128, 128>;
  using FragmentC = cutlass::Array<int, Shape::kCount / (kWarpCount * kWarpSize)>;

  using Epilogue = cutlass::gemm::threadblock::EpilogueWorkspace<
    Shape,
    kWarpCount,
    FragmentC
  >;

  typename Epilogue::Params params(
    
  );

  // Launch the kernel
  dim3 grid(1,1);
  dim3 block(kWarpSize, kWarpCount);

  test::gemm::threadblock::kernel_epilogue_workspace<Epilogue><<< grid, block >>>(
    params
  );

  cudaError_t result = cudaDeviceSynchronize();
  EXPECT_EQ(result, cudaSuccess) << "Kernel launch error - " << cudaGetErrorString(result);

  //
  // 
  //
}

/////////////////////////////////////////////////////////////////////////////////////////////////
