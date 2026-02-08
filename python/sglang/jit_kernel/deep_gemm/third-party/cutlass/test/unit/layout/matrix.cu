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
\brief unit tests for matrix layout
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace test {
namespace layout {
  void test_row_major_layout(int row_size, int column_size, int ldm) {
    cutlass::layout::RowMajor row_major(ldm);

    // test pointer offset
    for (int row_idx = 0; row_idx < row_size; row_idx++) {
      for (int column_idx = 0; column_idx < column_size; column_idx++) {
        cutlass::MatrixCoord matrix_coord(row_idx, column_idx);
        auto ptr_offset = row_major(matrix_coord);
        decltype(ptr_offset) reference_offset = row_idx * ldm + column_idx;
        EXPECT_EQ(ptr_offset, reference_offset);
      }
    }

    // test stride
    EXPECT_EQ(row_major.stride()[0], ldm);

    // test capacity
    auto capacity = row_major.capacity(cutlass::MatrixCoord(row_size, column_size));
    decltype(capacity) reference_capacity = row_size * ldm;
    EXPECT_EQ(capacity, reference_capacity);

    // test packed
    auto packed = row_major.packed(cutlass::MatrixCoord(row_size, column_size));
    // the packed matrix's stride is the same with column size
    EXPECT_EQ(packed.stride()[0], column_size);
  }

  void test_column_major_layout(int row_size, int column_size, int ldm) {
    cutlass::layout::ColumnMajor column_major(ldm);

    // test pointer offset
    for (int row_idx = 0; row_idx < row_size; row_idx++) {
      for (int column_idx = 0; column_idx < column_size; column_idx++) {
        cutlass::MatrixCoord matrix_coord(row_idx, column_idx);
        auto ptr_offset = column_major(matrix_coord);
        decltype(ptr_offset) reference_offset = row_idx + column_idx * ldm;
        EXPECT_EQ(ptr_offset, reference_offset);
      }
    }

    // test stride
    EXPECT_EQ(column_major.stride()[0], ldm);

    // test capacity
    auto capacity = column_major.capacity(cutlass::MatrixCoord(row_size, column_size));
    decltype(capacity) reference_capacity = column_size * ldm;
    EXPECT_EQ(capacity, reference_capacity);

    // test packed
    auto packed = column_major.packed(cutlass::MatrixCoord(row_size, column_size));
    // the packed matrix's stride is the same with row size
    EXPECT_EQ(packed.stride()[0], row_size);
  }

} // namespace layout
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Layout_Matrix, row_major_32_53) {
  int const row_size = 32;
  int const column_size = 53;
  int const ldm = 55;
  test::layout::test_row_major_layout(row_size, column_size, ldm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Layout_Matrix, column_major_32_53) {
  int const row_size = 32;
  int const column_size = 53;
  int const ldm = 55;
  test::layout::test_column_major_layout(row_size, column_size, ldm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Layout_Matrix, general_matrix) {

  int M = 16;
  int N = 16;
  int interleave = 4;

  cutlass::layout::GeneralMatrix::TensorCoord extent = {M, N};

  cutlass::layout::GeneralMatrix layout =
      cutlass::layout::GeneralMatrix::packed(
          extent, cutlass::layout::Matrix::kColumnMajor, interleave);

  cutlass::HostTensor<int, cutlass::layout::ColumnMajor> tensor(extent);

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      tensor.host_data(m * N + n) = m * N + n;
    }
  }

  cutlass::TensorView<int, cutlass::layout::GeneralMatrix> canonical({tensor.host_data(), layout}, extent);

  // Uncomment this to view
  //
  //std::cout << canonical << std::endl;
  //
}

/////////////////////////////////////////////////////////////////////////////////////////////////
