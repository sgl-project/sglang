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

#include "cutlass/gemm/thread/mma.h"

#include "testbed.h"

#if 0
int main() {
  nvrtc::thread::Testbed<
    cutlass::gemm::GemmShape<3, 4, 2>,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor
  >().run("cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<3, 4, 2>, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor >");
  return 0;
}
#endif

TEST(SM50_Sgemm_thread_nvrtc, DISABLED_col_row_3x4x2) {

  test::nvrtc::thread::Testbed<
    cutlass::gemm::GemmShape<3, 4, 2>,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor
  >().run("cutlass::gemm::thread::Mma<cutlass::gemm::GemmShape<3, 4, 2>, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor >");
}

/////////////////////////////////////////////////////////////////////////////////////////////////
#if 0
TEST(SM50_Sgemm_thread, col_row_3x4x2) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<3, 4, 2>,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor
  >().run();
}

TEST(SM50_Sgemm_thread, col_row_4x4x2) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<4, 4, 2>,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor
  >().run();
}

TEST(SM50_Sgemm_thread, row_col_4x4x2) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<4, 4, 2>,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::ColumnMajor
  >().run();
}

TEST(SM50_Sgemm_thread, col_row_4x5x3) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<4, 5, 3>,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor
  >().run();
}

TEST(SM50_Sgemm_thread, col_row) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<8, 8, 1>,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor
  >().run();
}

TEST(SM50_Sgemm_thread, row_col) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<8, 8, 1>,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::ColumnMajor
  >().run();
}

TEST(SM50_Sgemm_thread, col_col) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<8, 8, 1>,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::ColumnMajor
  >().run();
}

TEST(SM50_Sgemm_thread, row_row) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<8, 8, 1>,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor
  >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Dgemm_thread, col_row) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<8, 8, 1>,
    double,
    cutlass::layout::ColumnMajor,
    double,
    cutlass::layout::RowMajor,
    double,
    cutlass::layout::ColumnMajor
  >().run();
}

TEST(SM50_Dgemm_thread, row_col) {

  test::gemm::thread::Testbed<
    cutlass::gemm::GemmShape<8, 8, 1>,
    double,
    cutlass::layout::RowMajor,
    double,
    cutlass::layout::ColumnMajor,
    double,
    cutlass::layout::ColumnMajor
  >().run();
}
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////
