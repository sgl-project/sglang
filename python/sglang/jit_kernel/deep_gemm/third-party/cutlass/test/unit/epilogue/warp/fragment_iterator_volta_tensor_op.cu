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
#include "cutlass/gemm/warp/mma_tensor_op_sm70.h"
#include "cutlass/epilogue/warp/fragment_iterator_volta_tensor_op.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM70_Epilogue_warp_FragmentIterator, mma_f16_64x64x4) {

  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

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

  using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    Shape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> accumulator_tensor({Shape::kM, Shape::kN});

  cutlass::reference::host::TensorFill(accumulator_tensor.host_view(), ElementC(-1));

  for (int tid = 0; tid < 1; ++tid) {
    typename MmaTensorOp::IteratorC::Fragment accumulator_tile;

    CUTLASS_PRAGMA_UNROLL
    for (size_t i = 0; i < accumulator_tile.size(); ++i) {
      accumulator_tile[i] = static_cast<ElementC>(int(i));
    }

    using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorVoltaTensorOp<
      cutlass::gemm::GemmShape<64, 64, 4>,
      cutlass::gemm::GemmShape<32, 32, 4>,
      cutlass::half_t,
      cutlass::layout::RowMajor
    >; 

    FragmentIterator frag_iterator(accumulator_tile);

    typename FragmentIterator::Fragment frag;

    for (int iter = 0; iter < FragmentIterator::kIterations; ++iter) {
      frag_iterator.load(frag);
      ++frag_iterator;

    #if 0
      std::cout << "T" << tid << ": ";
      for (size_t i = 0; i < frag.size(); ++i) {
        std::cout << "  " << frag[i];
      }
      std::cout << std::endl;
      #endif
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM70_Epilogue_warp_FragmentIterator, mma_f32_64x64x4) {

  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;
  using LayoutC = cutlass::layout::RowMajor;

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

  using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    Shape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  cutlass::HostTensor<ElementC, LayoutC> accumulator_tensor({Shape::kM, Shape::kN});

  cutlass::reference::host::TensorFill(accumulator_tensor.host_view(), ElementC(-1));

  for (int tid = 0; tid < 1; ++tid) {
    typename MmaTensorOp::IteratorC::Fragment accumulator_tile;

    CUTLASS_PRAGMA_UNROLL
    for (size_t i = 0; i < accumulator_tile.size(); ++i) {
      accumulator_tile[i] = static_cast<ElementC>(i);
    }

    typename MmaTensorOp::IteratorC iterator_C(accumulator_tensor.host_ref(), tid);  
    iterator_C.store(accumulator_tile);
  }

  /*
  std::ofstream output("volta_mma_f32_64x64x4.csv");
  output << accumulator_tensor.host_view() << std::endl;
  */

  for (int tid = 0; tid < 1; ++tid) {
    typename MmaTensorOp::IteratorC::Fragment accumulator_tile;

    using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorVoltaTensorOp<
      cutlass::gemm::GemmShape<64, 64, 4>,
      cutlass::gemm::GemmShape<32, 32, 4>,
      ElementC,
      LayoutC
    >; 
    
    FragmentIterator frag_iterator(accumulator_tile);

    for (int iter = 0; iter < FragmentIterator::kIterations; ++iter) {

      typename FragmentIterator::Fragment frag;
      frag_iterator.load(frag);
      ++frag_iterator;

      #if 0
      std::cout << "Iteration: " << iter << " - T" << tid << ": ";
      
      for (int i = 0; i < frag.size(); ++i) {
        std::cout << "  " << frag[i];
      }

      std::cout << std::endl;
      #endif
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
