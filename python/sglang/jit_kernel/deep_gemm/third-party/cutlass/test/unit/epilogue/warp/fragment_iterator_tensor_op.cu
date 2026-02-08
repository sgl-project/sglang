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
#include "cutlass/gemm/warp/default_mma_tensor_op.h"

#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Epilogue_warp_FragmentIterator, mma_f32_64x64x8) {

  using Shape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    Shape,
    InstructionShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor
  >::Type;

  using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    Shape,
    typename MmaTensorOp::Policy::Operator::Shape,
    typename MmaTensorOp::Policy::Operator::ElementC,
    typename MmaTensorOp::Policy::Operator::FragmentC,
    cutlass::layout::RowMajor
  >;

  // This test just prints things.
  #if 0
  typename MmaTensorOp::FragmentC accum;

  std::cout << "Native accumulators:\n";

  for (size_t i = 0; i < MmaTensorOp::FragmentC::kElements; ++i) {
    accum[i] = ElementC(i);

    std::cout << accum[i] << " ";
    if (i && !((i + 1) % 4)) { 
      std::cout << "\n";
    }
  }

  std::cout << std::endl;

  std::cout << "FragmentIterator::Policy = { \n"
    << "  kAccessesPerInstruction:  " << FragmentIterator::Policy::kIterationsPerInstruction << "\n"
    << "  kAccumulatorRowStride:    " << FragmentIterator::Policy::kAccumulatorRowStride << "\n"
    << "  kAccumulatorColumnStride: " << FragmentIterator::Policy::kAccumulatorColumnStride << "\n"
    << "  kIterations:              " << FragmentIterator::Policy::kIterations << "\n"
    << " }" << std::endl;

  FragmentIterator fragment_iterator(accum);

  for (int iter = 0; iter < FragmentIterator::kIterations; ++iter) {

    typename FragmentIterator::Fragment frag;

    fragment_iterator.load(frag);

    std::cout << "Iteration " << iter << ":\n";

    for (size_t i = 0; i < FragmentIterator::Fragment::kElements; ++i) {
      std::cout << frag[i] << " ";
    }

    std::cout << std::endl;

    ++fragment_iterator;
  }
  #endif
}

TEST(SM75_Epilogue_warp_FragmentIterator, mma_f16_64x64x8) {

  using Shape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    Shape,
    InstructionShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor
  >::Type;

  using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    Shape,
    typename MmaTensorOp::Policy::Operator::Shape,
    typename MmaTensorOp::Policy::Operator::ElementC,
    typename MmaTensorOp::Policy::Operator::FragmentC,
    cutlass::layout::RowMajor
  >;

  // This test just prints things.
  #if 0
  typename MmaTensorOp::FragmentC accum;

  std::cout << "Native accumulators:\n";

  for (size_t i = 0; i < MmaTensorOp::FragmentC::kElements; ++i) {
    accum[i] = ElementC((int)i);

    std::cout << (float)accum[i] << " ";
    if (i && !((i + 1) % 4)) { 
      std::cout << "\n";
    }
  }

  std::cout << std::endl;

  std::cout << "FragmentIterator::Policy = { \n"
    << "  kAccessesPerInstruction:  " << FragmentIterator::Policy::kIterationsPerInstruction << "\n"
    << "  kAccumulatorRowStride:    " << FragmentIterator::Policy::kAccumulatorRowStride << "\n"
    << "  kAccumulatorColumnStride: " << FragmentIterator::Policy::kAccumulatorColumnStride << "\n"
    << "  kIterations:              " << FragmentIterator::Policy::kIterations << "\n"
    << " }" << std::endl;

  FragmentIterator fragment_iterator(accum);

  for (int iter = 0; iter < FragmentIterator::kIterations; ++iter) {

    typename FragmentIterator::Fragment frag;

    fragment_iterator.load(frag);

    std::cout << "Iteration " << iter << ":\n";

    for (size_t i = 0; i < FragmentIterator::Fragment::kElements; ++i) {
      std::cout << (float)frag[i] << " ";
    }

    std::cout << std::endl;

    ++fragment_iterator;
  }
  #endif
}

TEST(SM75_Epilogue_warp_FragmentIterator_column, mma_f32_64x64x8) {

  using Shape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    Shape,
    InstructionShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor
  >::Type;

  using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    Shape,
    typename MmaTensorOp::Policy::Operator::Shape,
    typename MmaTensorOp::Policy::Operator::ElementC,
    typename MmaTensorOp::Policy::Operator::FragmentC,
    cutlass::layout::ColumnMajor
  >;

  // This test just prints things.
  #if 0
  typename MmaTensorOp::FragmentC accum;

  std::cout << "Native accumulators:\n";

  for (size_t i = 0; i < MmaTensorOp::FragmentC::kElements; ++i) {
    accum[i] = ElementC((int)i);

    std::cout << (float)accum[i] << " ";
    if (i && !((i + 1) % 4)) { 
      std::cout << "\n";
    }
  }

  std::cout << std::endl;

  std::cout << "FragmentIterator::Policy = { \n"
    << "  kAccessesPerInstruction:  " << FragmentIterator::Policy::kIterationsPerInstruction << "\n"
    << "  kAccumulatorRowStride:    " << FragmentIterator::Policy::kAccumulatorRowStride << "\n"
    << "  kAccumulatorColumnStride: " << FragmentIterator::Policy::kAccumulatorColumnStride << "\n"
    << "  kIterations:              " << FragmentIterator::Policy::kIterations << "\n"
    << " }" << std::endl;

  FragmentIterator fragment_iterator(accum);

  for (int iter = 0; iter < FragmentIterator::kIterations; ++iter) {

    typename FragmentIterator::Fragment frag;

    fragment_iterator.load(frag);

    std::cout << "Iteration " << iter << ":\n";

    for (size_t i = 0; i < FragmentIterator::Fragment::kElements; ++i) {
      std::cout << (float)frag[i] << " ";
    }

    std::cout << std::endl;

    ++fragment_iterator;
  }
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
