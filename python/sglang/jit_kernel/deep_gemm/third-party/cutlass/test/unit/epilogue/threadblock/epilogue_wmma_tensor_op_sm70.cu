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
#include "cutlass/arch/wmma.h"

#ifdef CUTLASS_ARCH_WMMA_SM70_ENABLED

#include "../../common/cutlass_unit_test.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/half.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/warp/default_mma_wmma_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_wmma_tensor_op.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "testbed.h"


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// F16 acumulation
//
/////////////////////////////////////////////////////////////////////////////////////////////////
TEST(SM70_Epilogue_threadblock_epilogue, f16_wmma_tensor_op_64x64_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
      WarpShape, 
      InstructionShape, 
      ElementA, 
      LayoutA, 
      ElementB, 
      LayoutB, 
      ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWmmaTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);

}

TEST(SM70_Epilogue_threadblock_epilogue, f16_wmma_tensor_op_64x128_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
      WarpShape, 
      InstructionShape, 
      ElementA, 
      LayoutA, 
      ElementB, 
      LayoutB, 
      ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWmmaTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);

}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// F32 acumulation and F32 output
//
/////////////////////////////////////////////////////////////////////////////////////////////////
TEST(SM70_Epilogue_threadblock_epilogue, f32_wmma_tensor_op_64x64_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //
  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = cutlass::half_t;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
      WarpShape, 
      InstructionShape, 
      ElementA, 
      LayoutA, 
      ElementB, 
      LayoutB, 
      ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWmmaTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);

}
/////////////////////////////////////////////////////////////////////////////////////////////////


#endif //CUTLASS_ARCH_WMMA_ENABLED
