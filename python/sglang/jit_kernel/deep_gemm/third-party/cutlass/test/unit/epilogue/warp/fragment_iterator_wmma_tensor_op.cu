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
#include "cutlass/layout/matrix.h"

#include "cutlass/gemm/warp/mma_tensor_op_wmma.h"

#include "cutlass/epilogue/warp/fragment_iterator_wmma_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_wmma_tensor_op.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"


/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM70_Epilogue_warp_FragmentIterator, wmma_f16_64x64x16) {

  using Shape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Wmma<
      InstructionShape,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaTensorOp = cutlass::gemm::warp::MmaTensorOpWmma<
    Shape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    Policy
  >;

  using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorWmmaTensorOp<
    Shape,
    typename MmaTensorOp::Policy::Operator::Shape,
    typename MmaTensorOp::Policy::Operator::ElementC,
    typename MmaTensorOp::Policy::Operator::FragmentC,
    cutlass::layout::RowMajor
  >;

  #if 0
  //
  // Enable this code block to print comments for debugging.
  //

  std::cout << "FragmentIterator::Policy = { \n"
    << "  OperatorCount:  (" << FragmentIterator::Policy::OperatorCount::kRow <<", "<<FragmentIterator::Policy::OperatorCount::kColumn << ")\n"
    << "  kRowPerIterations: " << FragmentIterator::Policy::kRowsPerIteration << "\n"
    << "  kWmmaFragmentsPerAccess: " << FragmentIterator::Policy::kWmmaFragmentsPerAccess << "\n"
    << "  kIterations: " << FragmentIterator::Policy::kIterations << "\n"
    << " }" << std::endl;

  typename MmaTensorOp::FragmentC accum;

  std::cout<<"MmaTensorOp::FragmentC::kElements " <<MmaTensorOp::FragmentC::kElements<<"\n";
  #endif

}

TEST(SM70_Epilogue_warp_FragmentIterator, wmma_f32_64x64x16) {

  using Shape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Wmma<
      InstructionShape,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaTensorOp = cutlass::gemm::warp::MmaTensorOpWmma<
    Shape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    Policy
  >;

  using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorWmmaTensorOp<
    Shape,
    typename MmaTensorOp::Policy::Operator::Shape,
    typename MmaTensorOp::Policy::Operator::ElementC,
    typename MmaTensorOp::Policy::Operator::FragmentC,
    cutlass::layout::RowMajor
  >;

  #if 0
  //
  // Enable this code block to print comments for debugging.
  //
  std::cout << "FragmentIterator::Policy = { \n"
    << "  OperatorCount:  (" << FragmentIterator::Policy::OperatorCount::kRow <<", "<<FragmentIterator::Policy::OperatorCount::kColumn << ")\n"
    << "  kRowPerIterations: " << FragmentIterator::Policy::kRowsPerIteration << "\n"
    << "  kWmmaFragmentsPerAccess: " << FragmentIterator::Policy::kWmmaFragmentsPerAccess << "\n"
    << "  kIterations: " << FragmentIterator::Policy::kIterations << "\n"
    << " }" << std::endl;

  typename MmaTensorOp::FragmentC accum;

  std::cout<<"MmaTensorOp::FragmentC::kElements " <<MmaTensorOp::FragmentC::kElements<<"\n";

  #endif
}
/////////////////////////////////////////////////////////////////////////////////////////////////
#endif //CUTLASS_ARCH_WMMA_SM70_ENABLED
