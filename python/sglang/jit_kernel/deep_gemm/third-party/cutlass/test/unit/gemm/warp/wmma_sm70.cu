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

    \brief Unit tests for warp-level wmma gemm
*/
#include "cutlass/arch/wmma.h"

#if defined(CUTLASS_ARCH_WMMA_SM70_ENABLED)

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

/// Test name format: SM[arch]_warp_wmma_[alayout]_[blayout]_[clayout]_[dtype].[threadblock_shape]_[warp_shape]

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// f16 accumulation point wmma.mma //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////// [START] Verifying all layouts {N,T}x{N,T}=>{N,T} for WMMA 16x16x16 [START] //////////////////////

////////////////////////////////////////////////////////////  
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16
////////////////////////////////////////////////////////////
    
// 4 tests for {N,T}x{N,T}=>{T}
TEST(SM70_warp_wmma_row_col_row_f16, 16x16x16_16x16x16_16x16x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<16, 16, 16> >().run();
}

////////////////////////////////////////////////////////////  
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.col.row.m16n16k16.f16.f16
////////////////////////////////////////////////////////////
TEST(SM70_warp_wmma_col_row_row_f16, 16x16x16_16x16x16_16x16x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<16, 16, 16> >().run();
}

////////////////////////////////////////////////////////////  
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.row.m16n16k16.f16.f16
////////////////////////////////////////////////////////////
TEST(SM70_warp_wmma_row_row_row_f16, 16x16x16_16x16x16_16x16x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<16, 16, 16> >().run();
}


////////////////////////////////////////////////////////////  
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.col.row.m16n16k16.f16.f16
////////////////////////////////////////////////////////////
TEST(SM70_warp_wmma_col_col_row_f16, 16x16x16_16x16x16_16x16x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<16, 16, 16> >().run();
}

// 4 tests for {N,T}x{N,T}=>{N}
TEST(SM70_warp_wmma_row_col_col_f16, 16x16x16_16x16x16_16x16x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<16, 16, 16> >().run();
}

////////////////////////////////////////////////////////////  
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.col.row.m16n16k16.f16.f16
////////////////////////////////////////////////////////////
TEST(SM70_warp_wmma_col_row_col_f16, 16x16x16_16x16x16_16x16x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<16, 16, 16> >().run();
}

////////////////////////////////////////////////////////////  
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.row.m16n16k16.f16.f16
////////////////////////////////////////////////////////////
TEST(SM70_warp_wmma_row_row_col_f16, 16x16x16_16x16x16_16x16x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<16, 16, 16> >().run();
}


////////////////////////////////////////////////////////////  
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.col.row.m16n16k16.f16.f16
////////////////////////////////////////////////////////////
TEST(SM70_warp_wmma_col_col_col_f16, 16x16x16_16x16x16_16x16x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<16, 16, 16> >().run();
}
/////////// [END] Verifying all layouts {N,T}x{N,T}=>{N,T} for WMMA 16x16x16 [END] ///////////////////////////



TEST(SM70_warp_wmma_row_col_row_f16, 64x64x16_64x64x16_16x16x16) {

  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<64, 64, 16> >().run();
}


TEST(SM70_warp_wmma_row_col_row_f16, 64x64x32_64x64x32_16x16x16) {

  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;


  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<64, 64, 32> >().run();
}


TEST(SM70_warp_wmma_row_col_row_f16, 64x64x32_64x32x32_16x16x16) {

  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<64, 64, 32> >().run();
}

TEST(SM70_warp_wmma_row_col_row_f16, 64x64x32_32x64x32_16x16x16) {

  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<64, 64, 32> >().run();
}

TEST(SM70_warp_wmma_row_col_row_f16, 64x64x32_32x32x32_16x16x16) {

  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<64, 64, 32> >().run();
}

TEST(SM70_warp_wmma_row_col_row_f16, 128x128x16_64x64x16_16x16x16) {
  // Even though the test launches 128x128x16 CTA tile this test only verfies one warp
  // , i.e., warp_0 of size 64x64x16 out of the four warps required to cover the CTA
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<128, 128, 16> >().run();
}


////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m32n8k16.f16.f16
////////////////////////////////////////////////////////////
TEST(SM70_warp_wmma_row_col_row_f16, 32x8x16_32x8x16_32x8x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<32, 8, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<32, 8, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<32, 8, 16> >().run();
}

////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m8n32k16.f16.f16
////////////////////////////////////////////////////////////
TEST(SM70_warp_wmma_row_col_row_f16, 8x32x16_8x32x16_32x8x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<8, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 32, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<8, 32, 16> >().run();
}


////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.col.row.m8n32k16.f16.f16
////////////////////////////////////////////////////////////
TEST(SM70_warp_wmma_col_row_row_f16, 8x32x16_8x32x16_8x32x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<8, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 32, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<8, 32, 16> >().run();
}

TEST(SM70_warp_wmma_col_row_row_f16, 32x8x16_32x8x16_32x8x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<32, 8, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<32, 8, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<32, 8, 16> >().run();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// f32 accumulation point wmma.mma //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32
////////////////////////////////////////////////////////////
TEST(SM70_warp_wmma_row_col_row_f32, 16x16x16_16x16x16_16x16x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<16, 16, 16> >().run();
}

TEST(SM70_warp_wmma_row_col_row_f32, 64x64x16_64x64x16_16x16x16) {

  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<64, 64, 16> >().run();
}

TEST(SM70_warp_wmma_row_col_row_f32, 64x64x32_64x64x32_16x16x16) {

  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<64, 64, 32> >().run();
}


TEST(SM70_warp_wmma_row_col_row_f32, 128x128x16_64x64x16_16x16x16) {
  // Even though the test launches 128x128x16 CTA tile this test only verfies one warp
  // , i.e., warp_0 of size 64x64x16 out of the four warps required to cover the CTA
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<128, 128, 16> >().run();
}


/////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m32n8k16.f32.f32
////////////////////////////////////////////////////////////
TEST(SM70_warp_wmma_row_col_row_f32, 32x8x16_32x8x16_32x8x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<32, 8, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<32, 8, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<32, 8, 16> >().run();
}


/////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m8n32k16.f32.f32
////////////////////////////////////////////////////////////
TEST(SM70_warp_wmma_row_col_row_f32, 8x32x16_8x32x16_8x32x16) {
  // Threadblock and warp with just one native WMMA operation (most basic unit test)
  using WarpShape = cutlass::gemm::GemmShape<8, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 32, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using WmmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    WarpShape, 
    InstructionShape, 
    ElementA, 
    LayoutA, 
    ElementB, LayoutB, 
    ElementC,
    LayoutC>::Type;

  test::gemm::warp::Testbed<WmmaTensorOp, cutlass::gemm::GemmShape<8, 32, 16> >().run();
}

#endif //CUTLASS_ARCH_WMMA_SM70_ENABLED
