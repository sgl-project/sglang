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

TEST(SM60_warp_gemm_f16_col_row, 8x4x1_1x1x1) {

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<8, 4>,
    cutlass::layout::ColumnMajorInterleaved<2>,
    cutlass::gemm::GemmShape<1, 1, 1>
  >;

  using Mma = cutlass::gemm::warp::MmaSimt<
    cutlass::gemm::GemmShape<8, 4, 8>,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, cutlass::gemm::GemmShape<128, 128, 8> >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM60_warp_gemm_f16_col_row, 16x8x1_2x2x1) {

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<8, 4>,
    cutlass::layout::ColumnMajorInterleaved<2>,
    cutlass::gemm::GemmShape<2, 2, 1>
  >;

  using Mma = cutlass::gemm::warp::MmaSimt<
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, cutlass::gemm::GemmShape<128, 128, 8> >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM60_warp_gemm_f16_col_row, 32x16x1_4x4x1) {

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<8, 4>,
    cutlass::layout::ColumnMajorInterleaved<2>,
    cutlass::gemm::GemmShape<4, 4, 1>
  >;

  using Mma = cutlass::gemm::warp::MmaSimt<
    cutlass::gemm::GemmShape<32, 16, 8>,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, cutlass::gemm::GemmShape<128, 128, 8> >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM60_warp_gemm_f16_col_row, 64x16x1_8x4x1) {

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<8, 4>,
    cutlass::layout::ColumnMajorInterleaved<2>,
    cutlass::gemm::GemmShape<8, 8, 1>
  >;

  using Mma = cutlass::gemm::warp::MmaSimt<
    cutlass::gemm::GemmShape<64, 32, 8>,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, cutlass::gemm::GemmShape<128, 128, 8> >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
