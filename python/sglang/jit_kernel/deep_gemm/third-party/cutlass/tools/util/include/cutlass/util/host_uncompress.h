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
    \brief uncompress sparse matrix from the host side 
*/
#pragma once

#include "cutlass/coord.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/tensor_view.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/gemm.h"

namespace cutlass {

// uncompress sparse tensor core A matrix
template <typename ElementA, typename LayoutA, typename ElementE,
          typename LayoutE>
void uncompress(TensorRef<ElementA, LayoutA> uncompressed_tensor_a,
                TensorRef<ElementA, LayoutA> tensor_a,
                TensorRef<ElementE, LayoutE> tensor_e, int row, int col) {
  // How many uncompressed data we can get with ElementE meta data
  int DecompressedElementsPerElementE =
      256 / cutlass::sizeof_bits<ElementA>::value;

  // Process 4bit meta data a time 
  int step;

  // 1:2 or 2:4 or 4:8
  int a, b;

  if (cutlass::sizeof_bits<ElementA>::value == 4) {
    step = 8;
    a = 4;
    b = 8;
  } else if (cutlass::sizeof_bits<ElementA>::value == 8) {
    step = 4;
    a = 2;
    b = 4;
  } else if (cutlass::sizeof_bits<ElementA>::value == 16) {
    step = 4;
    a = 2;
    b = 4;
  } else if (cutlass::sizeof_bits<ElementA>::value == 32) {
    step = 2;
    a = 1;
    b = 2;
  }

  int ElementsPerE = (cutlass::sizeof_bits<ElementA>::value == 4) ? 2 : 1;

  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < (col / DecompressedElementsPerElementE); ++c) {

      ElementE meta = tensor_e.at(MatrixCoord(r, c));

      for (int i = 0; i < DecompressedElementsPerElementE; i += step) {
        int e = (meta >> (i / step * 4)) & 0xf;
        int idx0 = e & 0x3;
        int idx1 = e >> 2;

        if (a == 1) idx0 = idx0 / 2;

        for (int ii = 0; ii < step; ii += ElementsPerE) {
          int real_col =
              c * DecompressedElementsPerElementE + i + ii;
          int compressed_col = (real_col / b) * a;

          if (ii == (idx0 * ElementsPerE)) {
            uncompressed_tensor_a.at(MatrixCoord(r, real_col)) =
                tensor_a.at(MatrixCoord(r, compressed_col));
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(MatrixCoord(r, real_col + 1)) =
                  tensor_a.at(MatrixCoord(r, compressed_col + 1));
          } else if ((ii == (idx1 * ElementsPerE)) && (a != 1)) {
            uncompressed_tensor_a.at(MatrixCoord(r, real_col)) =
                tensor_a.at(MatrixCoord(r, compressed_col + ElementsPerE));
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(MatrixCoord(r, real_col + 1)) =
                  tensor_a.at(
                      MatrixCoord(r, compressed_col + ElementsPerE + 1));
          } else {
            uncompressed_tensor_a.at(MatrixCoord(r, real_col)) =
                ElementA(0);
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(MatrixCoord(r, real_col + 1)) =
                  ElementA(0);
          }
        }
      }
    }
  }
}

// uncompress ELL block sparse matrix
template <typename ElementA, typename LayoutA,
          typename ElementE, typename LayoutE>
void uncompress_ell_block_sparse(
                TensorRef<ElementA, LayoutA> uncompressed_tensor_a,
                TensorRef<ElementA, LayoutA> tensor_a,
                TensorRef<ElementE, LayoutE> ell_idx,
                int rows, int cols,
                int ell_num_cols, int ell_blocksize) {

  for (int r = 0; r < rows / ell_blocksize; ++r) {
    for (int c = 0; c < ell_num_cols / ell_blocksize; ++c) {

      ElementE idx = ell_idx.at(MatrixCoord(r, c));

      if (idx != -1) {
        int row_begin = r * ell_blocksize;
        int col_begin_real = idx * ell_blocksize;
        int col_begin = c * ell_blocksize;
  
        for (int i = 0; i < ell_blocksize; ++i) {
          for (int j = 0; j < ell_blocksize; ++j) {
            uncompressed_tensor_a.at(MatrixCoord(row_begin + i, col_begin_real + j)) =
                tensor_a.at(
                    MatrixCoord(row_begin + i, col_begin +j));
          }
        }
      }
    }
  }
}

} // namespace cutlass

