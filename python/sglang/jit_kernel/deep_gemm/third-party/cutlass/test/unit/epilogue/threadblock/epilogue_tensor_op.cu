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

#include "cutlass/aligned_buffer.h"
#include "cutlass/half.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_threadblock_epilogue, s4_tensor_op_64x64_64x64x32) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::int4b_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 32 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s4_tensor_op_64x64_32x32x32) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::int4b_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 32 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s4_tensor_op_128x128_64x64x32) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::int4b_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 64 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s4_tensor_op_128x64_64x32x32) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::int4b_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 32 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s4_tensor_op_64x128_32x64x32) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::int4b_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 64 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s4_tensor_op_32x128_32x64x32) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::int4b_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 64 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<32, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s4_tensor_op_128x32_64x32x32) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::int4b_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 32 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 32, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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


TEST(SM75_Epilogue_threadblock_epilogue, s4_tensor_op_256x128_64x64x32) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::int4b_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 64 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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


TEST(SM75_Epilogue_threadblock_epilogue, s4_tensor_op_128x256_64x64x32) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::int4b_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 32 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 256, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s8_tensor_op_64x64_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = int8_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s8_tensor_op_64x64_32x3216) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = int8_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 64 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s8_tensor_op_128x128_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = int8_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s8_tensor_op_64x128_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = int8_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s8_tensor_op_128x64_64x32x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = int8_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 64 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s8_tensor_op_64x128_32x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = int8_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s8_tensor_op_32x128_32x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = int8_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<32, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, s8_tensor_op_128x32_64x32x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = int8_t;
  using ElementAccumulator = int;
  using ElementCompute = float;
  int const kElementsPerAccess = 64 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
  using Element = ElementOutput;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementAccumulator,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, tensor_op_64x64_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, tensor_op_128x128_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, tensor_op_128x256_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, tensor_op_256x128_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<256, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, tensor_op_32x32_32x32x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, tensor_op_64x64_32x32x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, tensor_op_64x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, tensor_op_128x64_64x32x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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
// Mixed precision tests
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_threadblock_epilogue, mixed_f16_f32_tensor_op_64x64_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, mixed_f16_f32_tensor_op_128x128_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, mixed_f16_f32_tensor_op_128x256_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, mixed_f16_f32_tensor_op_256x128_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<256, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, mixed_f16_f32_tensor_op_32x32_32x32x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, mixed_f16_f32_tensor_op_64x64_32x32x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, mixed_f16_f32_tensor_op_64x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, mixed_f16_f32_tensor_op_128x64_64x32x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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
// F16 acumulation
//
/////////////////////////////////////////////////////////////////////////////////////////////////


TEST(SM75_Epilogue_threadblock_epilogue, f16_tensor_op_64x64_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, f16_tensor_op_128x128_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, f16_tensor_op_128x256_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, f16_tensor_op_256x128_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<256, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, f16_tensor_op_32x32_32x32x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<32, 32, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, f16_tensor_op_64x64_32x32x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, f16_tensor_op_64x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, f16_tensor_op_128x64_64x32x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM80_Epilogue_threadblock_epilogue, f64_tensor_op_64x64_32x32x4) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = double;
  using ElementAccumulator = double;
  using ElementCompute = double;
  int const kElementsPerAccess = 1;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Element = double;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM80_Epilogue_threadblock_epilogue, f64_tensor_op_128x64_64x32x4) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = double;
  using ElementAccumulator = double;
  using ElementCompute = double;
  int const kElementsPerAccess = 1;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Element = double;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM80_Epilogue_threadblock_epilogue, f64_tensor_op_64x128_32x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = double;
  using ElementAccumulator = double;
  using ElementCompute = double;
  int const kElementsPerAccess = 1;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Element = double;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM80_Epilogue_threadblock_epilogue, f64_tensor_op_128x128_32x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = double;
  using ElementAccumulator = double;
  using ElementCompute = double;
  int const kElementsPerAccess = 1;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Element = double;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, vec1_mixed_f16_f32_tensor_op_128x128_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 1;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, vec1_mixed_f16_f32_tensor_op_128x256_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 1;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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


TEST(SM75_Epilogue_threadblock_epilogue, vec1_tensor_op_128x128_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 1;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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

TEST(SM75_Epilogue_threadblock_epilogue, vec1_tensor_op_128x256_64x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 1;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<128, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
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
