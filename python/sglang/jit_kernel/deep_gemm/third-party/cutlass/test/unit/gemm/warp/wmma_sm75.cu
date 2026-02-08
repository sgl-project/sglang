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

#if defined(CUTLASS_ARCH_WMMA_SM75_ENABLED)

#include "../../common/cutlass_unit_test.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/half.h"

#include "cutlass/gemm/warp/default_mma_wmma_tensor_op.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed.h"
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// SUBBYTE wmma.mma ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(SM75_warp_wmma_row_col_s4, 64x64x32_8x8x32_8x8x32) {

  using WarpShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using ElementA = cutlass::int4b_t;
  using ElementB = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  
  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, LayoutA, 
    ElementB, LayoutB, 
    ElementC, LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<64, 64, 32> >().run();
  
}

TEST(SM75_warp_wmma_row_col_s4, 64x64x32_64x64x32_8x8x32) {

  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using ElementA = cutlass::int4b_t;
  using ElementB = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  
  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, LayoutA, 
    ElementB, LayoutB, 
    ElementC, LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<64, 64, 32> >().run();
  
}

TEST(SM75_warp_wmma_row_col_s4, 64x64x64_8x8x64_8x8x32) {

  using WarpShape = cutlass::gemm::GemmShape<8, 8, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using ElementA = cutlass::int4b_t;
  using ElementB = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  
  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, LayoutA, 
    ElementB, LayoutB, 
    ElementC, LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<64, 64, 64> >().run();
  
}

TEST(SM75_warp_wmma_row_col_b1, 64x64x128_8x8x128_8x8x128) {

  using WarpShape = cutlass::gemm::GemmShape<8, 8, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 128>;
  using ElementA = cutlass::uint1b_t;
  using ElementB = cutlass::uint1b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  
  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, LayoutA, 
    ElementB, LayoutB, 
    ElementC, LayoutC,
    cutlass::arch::OpXorPopc>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::arch::OpXorPopc>().run();
  
}

TEST(SM75_warp_wmma_row_col_b1, 64x64x128_64x64x128_8x8x128) {

  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 128>;
  using ElementA = cutlass::uint1b_t;
  using ElementB = cutlass::uint1b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  
  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, LayoutA, 
    ElementB, LayoutB, 
    ElementC, LayoutC,
    cutlass::arch::OpXorPopc>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::arch::OpXorPopc>().run();
  
}
#endif //CUTLASS_ARCH_WMMA_SM75_ENABLED
