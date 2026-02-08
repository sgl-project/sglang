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

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma_simt.h"

#include "testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
TEST(SM61_warp_gemm_int8_col_row, col_row_8x4x8_1x1x4) {

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<8, 4>,
    cutlass::layout::ColumnMajorInterleaved<2>,
    cutlass::gemm::GemmShape<1, 1, 4>
  >;

  using Mma = cutlass::gemm::warp::MmaSimt<
    cutlass::gemm::GemmShape<8, 4, 8>,
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<4>,
    int,
    cutlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, cutlass::gemm::GemmShape<8, 4, 8> >().run();
}

TEST(SM61_warp_gemm_int8_col_row, col_row_8x4x4_1x1x4) {

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<8, 4>,
    cutlass::layout::ColumnMajorInterleaved<2>,
    cutlass::gemm::GemmShape<1, 1, 4>
  >;

  using Mma = cutlass::gemm::warp::MmaSimt<
    cutlass::gemm::GemmShape<8, 4, 8>,
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<4>,
    int,
    cutlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, cutlass::gemm::GemmShape<128, 64, 8> >().run();
}

TEST(SM61_warp_gemm_int8_col_row, col_row_16x4x4_2x1x4) {

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<8, 4>,
    cutlass::layout::ColumnMajorInterleaved<2>,
    cutlass::gemm::GemmShape<2, 1, 4>
  >;

  using Mma = cutlass::gemm::warp::MmaSimt<
    cutlass::gemm::GemmShape<16, 4, 4>,
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<4>,
    int,
    cutlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, cutlass::gemm::GemmShape<16, 4, 4> >().run();
}

TEST(SM61_warp_gemm_int8_col_row, col_row_16x4x4_2x2x4) {

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<8, 4>,
    cutlass::layout::ColumnMajorInterleaved<2>,
    cutlass::gemm::GemmShape<2, 2, 4>
  >;

  using Mma = cutlass::gemm::warp::MmaSimt<
    cutlass::gemm::GemmShape<16, 8, 4>,
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<4>,
    int,
    cutlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, cutlass::gemm::GemmShape<16, 8, 4> >().run();
}

TEST(SM61_warp_gemm_int8_col_row, col_row_32x16x4_4x4x4) {

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<8, 4>,
    cutlass::layout::ColumnMajorInterleaved<2>,
    cutlass::gemm::GemmShape<4, 4, 4>
  >;

  using Mma = cutlass::gemm::warp::MmaSimt<
    cutlass::gemm::GemmShape<32, 16, 16>,
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<4>,
    int,
    cutlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, cutlass::gemm::GemmShape<128, 64, 16> >().run();
}


TEST(SM61_warp_gemm_int8_col_row, col_row_128x64x4_16x16x4) {

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<8, 4>,
    cutlass::layout::ColumnMajorInterleaved<2>,
    cutlass::gemm::GemmShape<16, 16, 4>
  >;

  using Mma = cutlass::gemm::warp::MmaSimt<
    cutlass::gemm::GemmShape<128, 64, 4>,
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<4>,
    int,
    cutlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, cutlass::gemm::GemmShape<128, 64, 4> >().run();
}

TEST(SM61_warp_gemm_int8_col_row, col_row_64x64x4_4x4x4) {

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<8, 4>,
    cutlass::layout::ColumnMajorInterleaved<2>,
    cutlass::gemm::GemmShape<4, 4, 4>
  >;

  using Mma = cutlass::gemm::warp::MmaSimt<
    cutlass::gemm::GemmShape<64, 64, 8>,
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<4>,
    int,
    cutlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, cutlass::gemm::GemmShape<64, 64, 8> >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
