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

#include "cutlass/gemm/warp/mma_tensor_op_sm70.h"
#include "cutlass/epilogue/warp/fragment_iterator_volta_tensor_op.h"

#include "cutlass/epilogue/threadblock/default_thread_map_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM70_Epilogue_threadblock_epilogue, f16_volta_tensor_op_64x64_32x32x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_volta_tensor_op_128x64_64x32x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_volta_tensor_op_64x128_32x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_volta_tensor_op_64x64_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_volta_tensor_op_64x128_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_volta_tensor_op_128x64_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_volta_tensor_op_128x128_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_volta_tensor_op_128x256_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 256, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_volta_tensor_op_256x128_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<256, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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
// Mixed: F32 accumulation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM70_Epilogue_threadblock_epilogue, f16_f32_volta_tensor_op_64x64_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;


  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_f32_volta_tensor_op_128x256_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 256, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;


  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_f32_volta_tensor_op_256x128_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<256, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_f32_volta_tensor_op_128x128_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_f32_volta_tensor_op_64x64_32x32x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_f32_volta_tensor_op_64x128_32x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f16_f32_volta_tensor_op_128x64_64x32x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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
// F32 accumulation, F32 output
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM70_Epilogue_threadblock_epilogue, f32_volta_tensor_op_64x64_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = float;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f32_volta_tensor_op_64x128_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = float;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f32_volta_tensor_op_128x64_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = float;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f32_volta_tensor_op_128x128_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = float;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f32_volta_tensor_op_128x256_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 256, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = float;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f32_volta_tensor_op_256x128_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<256, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = float;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f32_volta_tensor_op_64x64_32x32x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = float;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f32_volta_tensor_op_128x64_64x32x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = float;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, f32_volta_tensor_op_64x128_32x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = float;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;


  int const kPartitionsK = 1;
  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

// This works
TEST(SM70_Epilogue_threadblock_epilogue, vec8_f16_f32_volta_tensor_op_64x64_32x32x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 8;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

// This works
TEST(SM70_Epilogue_threadblock_epilogue, vec2_f16_f32_volta_tensor_op_64x64_32x32x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 2;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;


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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

// This fails
TEST(SM70_Epilogue_threadblock_epilogue, vec1_f16_f32_volta_tensor_op_64x64_32x32x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 1;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, vec1_f32_volta_tensor_op_128x128_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = float;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 1;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, vec1_f16_f32_volta_tensor_op_128x128_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 128, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  int const kPartitionsK = 1;
  int const kElementsPerAccess = 1;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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

TEST(SM70_Epilogue_threadblock_epilogue, vec1_f16_f32_volta_tensor_op_128x256_64x64x4) {

  //
  // Define the warp-level matrix multiply
  //

  using Shape = cutlass::gemm::GemmShape<128, 256, 4>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using WarpMmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;


  int const kPartitionsK = 1;
  int const kElementsPerAccess = 1;

  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape, 
    WarpShape, 
    kPartitionsK, 
    ElementC, 
    kElementsPerAccess,
    ElementAccumulator>::Type;

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

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
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
