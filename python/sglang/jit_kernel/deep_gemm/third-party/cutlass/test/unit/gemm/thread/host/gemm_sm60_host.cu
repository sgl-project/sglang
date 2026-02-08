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

#include "../../../common/cutlass_unit_test.h"

#include "cutlass/gemm/thread/mma.h"

#include "testbed_host.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Compute capability SM60
//

TEST(SM60_host_Hgemm_thread, col_row_col_1x1x16) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<1, 1, 16>,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor
  >().run();
}

TEST(SM60_host_Hgemm_thread, row_col_row_1x1x16) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<1, 1, 16>,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor
  >().run();
}

TEST(SM60_host_Hgemm_thread, row_row_row_2x2x2) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<2, 2, 2>,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor
  >().run();
}

TEST(SM60_host_Hgemm_thread, row_row_col_2x2x2) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<2, 2, 2>,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor
  >().run();
}

TEST(SM60_host_Hgemm_thread, row_col_row_2x2x2) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<2, 2, 2>,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor
  >().run();
}

TEST(SM60_host_Hgemm_thread, row_col_col_2x2x2) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<2, 2, 2>,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor
  >().run();
}

TEST(SM60_host_Hgemm_thread, col_row_row_2x2x2) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<2, 2, 2>,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor
  >().run();
}

TEST(SM60_host_Hgemm_thread, col_row_col_2x2x2) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<2, 2, 2>,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor
  >().run();
}

TEST(SM60_host_Hgemm_thread, col_col_row_2x2x2) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<2, 2, 2>,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor
  >().run();
}

TEST(SM60_host_Hgemm_thread, col_col_col_2x2x2) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<2, 2, 2>,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor
  >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
