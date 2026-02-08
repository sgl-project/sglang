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

#include "cutlass/tensor_view.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TensorView, rank2_contiguous_dynamic) {
  int const M = 8;
  int const N = 16;

  typedef cutlass::TensorView<int, cutlass::layout::ContiguousMatrix> ContiguousTensorView;

  cutlass::layout::Matrix layouts[] = {
    cutlass::layout::Matrix::kColumnMajor,
    cutlass::layout::Matrix::kRowMajor
  };

  cutlass::Coord<2> bounds = cutlass::make_Coord(M - 2, N - 2);

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
    ContiguousTensorView view(
      matrix_data,
      cutlass::layout::ContiguousMatrix::packed(cutlass::make_Coord(M, N), layouts[i]),
      bounds);

    ASSERT_TRUE(view.good());

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        cutlass::Coord<2> coord = cutlass::make_Coord(m, n);
        if (view.contains(coord)) {
          view.at(coord) = m * N + n;
        }
      }
    }

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        int expected = 0;
        if (m < bounds[0] && n < bounds[1]) {
          expected = int(m * N + n);
        }
        EXPECT_EQ(matrix_data[m * row_stride + n * col_stride], expected);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Uncomment the following line to observe output from printing TensorView objects
//

// #define OBSERVE_TENSORVIEW_IO   // uncomment to enable printing

#ifdef OBSERVE_TENSORVIEW_IO

// This test construct a TensorView of rank=2 with matrix layouts known at runtime. This
// uses TensorRefMapFunc classes defined in cutlass/matrix_traits.h to define the mapping
// from logical tensor indices to storage in memory.
//
// Helpers in tools/util/tensor_view_io.h print both the logical TensorView and the
// linear memory of the tensor.
TEST(TensorView, contiguous) {

  int const M = 8;
  int const N = 16;

  typedef cutlass::TensorView<
    int32_t,
    cutlass::layout::ContiguousLayout> ContiguousTensorView;

  cutlass::layout::Matrix layouts[] = {
    cutlass::layout::Matrix::kColumnMajor,
    cutlass::layout::Matrix::kRowMajor
  };

  cutlass::Coord<2> bounds = cutlass::make_Coord(M, N);

  for (int i = 0; i < 2; ++i) {

    int matrix_data[M * N] = { 0 };

    int ldm;
    int row_stride;
    int col_stride;

    if (layouts[i] == cutlass::layout::Matrix::kColumnMajor) {
      row_stride = 1;
      col_stride = M;
      ldm = col_stride;
    }
    else {
      row_stride = N;
      col_stride = 1;
      ldm = row_stride;
    }

    // Use helper to determine stride vector from leading dimension
    ContiguousTensorView view(
      matrix_data,
      cutlass::layout::ContiguousLayout::stride(layouts[i], ldm),
      bounds);

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        cutlass::Coord<2> coord = cutlass::make_Coord(m, n);
        if (view.contains(coord)) {
          view.at(coord) = m * N + n;
        }
      }
    }

    std::cout << "---------\n";
    std::cout << (layouts[i] == cutlass::layout::Matrix::kColumnMajor ?
      "Column-major:" : "Row-major:") << "\n\n";

    std::cout << "Logical view:\n";
    std::cout.width(4);
    std::cout << view << "\n" << std::endl;   // Print TensorView object.

    std::cout << "Linear memory:";
    for (int idx = 0; idx < view.capacity(); ++idx) {
      if (!(idx % (layouts[i] == cutlass::layout::Matrix::kColumnMajor ? M : N))) {
        std::cout << std::endl;
      }
      std::cout << std::setw(4) << view.at(idx) << " ";
    }

    std::cout << "\n" << std::endl;
  }
}

// This test is similar to the previous except it uses a column-major, interleaved data
// layout. The test prints both the logical representation (a typical column-major matrix)
// and a representation of linear memory.
//
// Note, the interleave=4 structure implies that every four consecutive elements in the
// same row shall be adjacent in memory followed by the next row.
TEST(TensorView, rank2_column_major_interleaved) {
  int const M = 16;
  int const N = 16;
  int const kInterleave = 4;

  int matrix_data[M * N] = {0};

  cutlass::Coord<2> bounds = cutlass::make_Coord(M, N);

  // Define the TensorRefMapFunc for a column-major interleaved matrix format
  typedef cutlass::layout::ColumnMajorInterleaved<kInterleave> TensorRefMapFunc;

  // Define a TensorView of rank=2 using the column-major interleaved mapping function
  typedef cutlass::TensorView<
    int,
    TensorRefMapFunc> InterleavedTensorView;

  InterleavedTensorView view(
    matrix_data,
    TensorRefMapFunc::stride(M),
    bounds);

  // Initialize
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      view.at(cutlass::make_Coord(m, n)) = m + n * M;
    }
  }

  // Print logical view
  std::cout << "Column-major, interleave=" << kInterleave << " (logical view):\n";

  std::cout << std::setw(4) << view << "\n" << std::endl;

  // Now define a linear view of the same data in memory
  typedef cutlass::TensorView<int, 2, cutlass::layout::RowMajor> LinearTensorView;

  LinearTensorView linear_view(matrix_data, cutlass::make_Coord(N), bounds);

  std::cout << "Linear view in memory:\n";
  std::cout << std::setw(4) << linear_view << std::endl;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TensorView, int4) {

  int const M = 4;
  int const N = 8;

  using T = cutlass::int4b_t;

  cutlass::HostTensor<T, cutlass::layout::RowMajor> tensor({M, N});

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      T x = T(n ^ m); // some simple hash
      tensor.host_view().at({m, n}) = x;
    }
  }

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int x = (n ^ m); // some simple hash
      EXPECT_TRUE(int(tensor.host_view().at({m, n})) == x);
    }
  }

  EXPECT_EQ(tensor.size(), M * N);
}

TEST(TensorView, uint4) {

  int const M = 4;
  int const N = 8;

  using T = cutlass::uint4b_t;

  cutlass::HostTensor<T, cutlass::layout::RowMajor> tensor({M, N});

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      T x = T(n ^ m); // some simple hash
      tensor.host_view().at({m, n}) = x;
    }
  }

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int x = (n ^ m); // some simple hash
      EXPECT_TRUE(int(tensor.host_view().at({m, n})) == x);
    }
  }

  EXPECT_EQ(tensor.size(), M * N);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
