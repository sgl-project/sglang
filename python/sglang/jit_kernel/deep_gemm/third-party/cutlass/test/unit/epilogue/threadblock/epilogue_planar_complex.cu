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

#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"

// Tensor Op
#include "cutlass/gemm/warp/default_mma_tensor_op.h"

// Volta Tensor Op
#include "cutlass/gemm/warp/mma_tensor_op_sm70.h"
#include "cutlass/epilogue/warp/fragment_iterator_volta_tensor_op.h"

// Simt
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"

// Epilogue components

#include "cutlass/epilogue/threadblock/default_epilogue_planar_complex.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "testbed_planar_complex.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_threadblock_epilogue, planar_complex_f32_f32_tensor_op_64x64_32x32x8) {

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

  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    WarpShape, 
    InstructionShape, 
    Element, LayoutA, 
    Element, LayoutB, 
    ElementAccumulator, cutlass::layout::RowMajor
  >::Type;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpiloguePlanarComplex<
    Shape,
    WarpMmaTensorOp,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    kPartitionsK,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpiloguePlanarComplexTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_threadblock_epilogue, planar_complex_f16_f32_tensor_op_64x64_32x32x8) {

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

  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    WarpShape, 
    InstructionShape, 
    Element, LayoutA, 
    Element, LayoutB, 
    ElementAccumulator, cutlass::layout::RowMajor
  >::Type;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpiloguePlanarComplex<
    Shape,
    WarpMmaTensorOp,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    kPartitionsK,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpiloguePlanarComplexTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_threadblock_epilogue, planar_complex_f16_f16_tensor_op_64x64_32x32x8) {

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

  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    WarpShape, 
    InstructionShape, 
    Element, LayoutA, 
    Element, LayoutB, 
    ElementAccumulator, cutlass::layout::RowMajor
  >::Type;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpiloguePlanarComplex<
    Shape,
    WarpMmaTensorOp,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    kPartitionsK,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpiloguePlanarComplexTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_threadblock_epilogue, planar_complex_f32_f32_volta_tensor_op_64x64_32x32x4) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;

  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<32, 32, 4>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 4>;
  using Element = cutlass::half_t;

  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<Element>::value>;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      Element,
      cutlass::layout::ColumnMajor,
      Element,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    ElementAccumulator,
    cutlass::layout::RowMajor,
    Policy
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpiloguePlanarComplex<
    Shape,
    WarpMmaTensorOp,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    kPartitionsK,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpiloguePlanarComplexTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}
  
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_threadblock_epilogue, planar_complex_simt_f32_64x64_32x32x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 1;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
  using Element = float;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<4, 4, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpiloguePlanarComplex<
    Shape,
    WarpMmaSimt,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    kPartitionsK,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpiloguePlanarComplexTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_threadblock_epilogue, planar_complex_simt_f64_64x64_16x32x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = double;
  using ElementAccumulator = double;
  using ElementCompute = double;
  int const kElementsPerAccess = 1;
  int const kPartitionsK = 1;
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;
  using Element = double;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<4, 4, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpiloguePlanarComplex<
    Shape,
    WarpMmaSimt,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    kPartitionsK,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpiloguePlanarComplexTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
