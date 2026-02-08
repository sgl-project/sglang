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

#include "cutlass/epilogue/thread/linear_combination_drelu.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_with_reduction.h"
#include "cutlass/epilogue/threadblock/epilogue_with_reduction.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "epilogue_with_reduction_testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Disable selected tests on CUDA 11.1
//
//
#define ENABLE_BLOCKED_TESTS (!(__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 1))

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_with_reduction_threadblock, f16_tensor_op_64x64_64x64x8) {

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

  using OutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = cutlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_with_reduction_threadblock, f32_tensor_op_64x64_64x64x8) {

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

  using OutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = cutlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_with_reduction_threadblock, f32_tensor_op_128x128_64x64x8) {

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

  using OutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = cutlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_with_reduction_threadblock, f16_tensor_op_128x128_64x64x8) {

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

  using OutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = cutlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_with_reduction_threadblock, f32_tensor_op_128x64_64x32x8) {

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

  using OutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = cutlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#if ENABLE_BLOCKED_TESTS

TEST(SM75_Epilogue_with_reduction_threadblock, f16_tensor_op_128x64_64x32x8) {

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

  using OutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = cutlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_with_reduction_threadblock, f32_tensor_op_64x128_32x64x8) {

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

  using OutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = cutlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_with_reduction_threadblock, f16_tensor_op_64x128_32x64x8) {

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

  using OutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = cutlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_with_reduction_threadblock, f32_tensor_op_128x256_64x64x8) {

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

  using OutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = cutlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_with_reduction_threadblock, f16_tensor_op_128x256_64x64x8) {

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

  using OutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = cutlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_with_reduction_threadblock, f32_tensor_op_256x128_64x64x8) {

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

  using OutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = cutlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_with_reduction_threadblock, f16_tensor_op_256x128_64x64x8) {

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

  using OutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = cutlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
