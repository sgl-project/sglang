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
#include "../common/cutlass_unit_test.h"

#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TensorRef, basic_rank2) {
  int const M = 8;
  int const N = 16;

  int matrix_data[M * N] = {0};

  cutlass::TensorRef<
    int, 
    cutlass::IdentityTensorLayout<2> > matrix_ref(matrix_data, cutlass::make_Coord(N, 1));
  
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      matrix_ref.at(cutlass::make_Coord(m, n)) = m * N + n;
    }
  }

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      EXPECT_EQ(matrix_data[m * N + n], int(m * N + n));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TensorRef, rank2_column_major) {
  int const M = 8;
  int const N = 8;

  int matrix_data[M * N];

  cutlass::TensorRef<int, cutlass::layout::ColumnMajor> ref(matrix_data, M);

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      ref.at(cutlass::make_Coord(m, n)) = m * N + n;
    }
  }

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      EXPECT_EQ(matrix_data[m + n * M], int(m * N + n));
    }
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TensorRef, rank2_row_major) {
  int const M = 8;
  int const N = 16;

  int matrix_data[M * N] = { 0 };

  cutlass::TensorRef<int, cutlass::layout::RowMajor> ref(matrix_data, N);

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      ref.at(cutlass::make_Coord(m, n)) = m * N + n;
    }
  }

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      EXPECT_EQ(matrix_data[m * N + n], int(m * N + n));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TensorRef, rank2_contiguous_dynamic) {
  int const M = 8;
  int const N = 16;

  typedef cutlass::TensorRef<int, cutlass::layout::ContiguousMatrix> ContiguousTensorRef;

  cutlass::layout::Matrix layouts[] = {
    cutlass::layout::Matrix::kColumnMajor,
    cutlass::layout::Matrix::kRowMajor
  };

  for (int i = 0; i < 2; ++i) {

    int matrix_data[M * N] = { 0 };

    int row_stride;
    int col_stride;

    if (layouts[i] == cutlass::layout::Matrix::kColumnMajor) {
      row_stride = 1;
      col_stride = M;
    }
    else {
      row_stride = N;
      col_stride = 1;
    }

    // Use helper to determine stride vector from leading dimension
    ContiguousTensorRef ref(
      matrix_data,
      cutlass::layout::ContiguousMatrix::packed(cutlass::make_Coord(M, N), layouts[i]));

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        ref.at(cutlass::make_Coord(m, n)) = m * N + n;
      }
    }

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        EXPECT_EQ(matrix_data[m * row_stride + n * col_stride], int(m * N + n));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TensorRef, rank2_column_major_interleaved) {
  int const M = 16;
  int const N = 16;
  int const kInterleave = 4;

  int matrix_data[M * N] = {0};

  // Define the Layout for a column-major interleaved matrix format
  using Layout = cutlass::layout::ColumnMajorInterleaved<kInterleave>;

  // Construct a TensorRef
  cutlass::TensorRef<
    int,
    Layout> ref(matrix_data, Layout::packed(cutlass::make_Coord(M, N)));

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      ref.at(cutlass::make_Coord(m, n)) = m + n * M;
    }
  }

  // Verify
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; n += kInterleave) {
      for (int i = 0; i < kInterleave; ++i) {
        EXPECT_EQ(matrix_data[m * kInterleave + n * M + i], int(m + (n + i) * M));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TensorRef, rank2_row_major_interleaved) {
  int const M = 16;
  int const N = 16;
  int const kInterleave = 4;

  int matrix_data[M * N] = {0};

  // Define the Layout for a row-major interleaved matrix format
  using Layout = cutlass::layout::RowMajorInterleaved<kInterleave>;

  // Construct a TensorRef
  cutlass::TensorRef<
    int,
    Layout> ref(matrix_data, Layout::packed(cutlass::make_Coord(M, N)));

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      ref.at(cutlass::make_Coord(m, n)) = m + n * M;
    }
  }

  // Verify
  for (int m = 0; m < M; m += kInterleave) {
    for (int n = 0; n < N; ++n) {
      for (int i = 0; i < kInterleave; ++i) {
        EXPECT_EQ(matrix_data[m * N + i + n * kInterleave], int((m + i) + n * M));
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

