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
    \brief reorder data from the host side 
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/tensor_view.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/gemm.h"

namespace cutlass {

/// This is needed for the interleaved integer tensor core kernels.  The purpose
/// is to use skip the shared memory part in the epilogue.
template <int Interleaved, typename Element, typename Layout>
void reorder_column(TensorRef<Element, Layout> dest,
                    TensorRef<Element, Layout> src,
                    cutlass::gemm::GemmCoord problem_size) {
  const int InstructionShapeCol = 8;
  // 4 threads per Quad
  const int ElementsPerThread = InstructionShapeCol / 4;
  // 4 threads per Quad
  const int ReorderedElementsPerThread =
      Interleaved / 4;

  for (int n = 0; n < problem_size.n(); n++) {
    for (int k = 0; k < problem_size.k(); k++) {
      dest.at({k, (n / Interleaved) * Interleaved +
                      ((n % ReorderedElementsPerThread) / ElementsPerThread) *
                          InstructionShapeCol +
                      ((n % Interleaved) / ReorderedElementsPerThread) *
                          ElementsPerThread +
                      (n % ElementsPerThread)}) = src.at({k, n});
    }
  }
}

template <int ColumnInterleaved, int LayoutInterleaved = ColumnInterleaved, typename Element, typename Layout>
void reorder_convK(TensorRef<Element, Layout> dest,
                    TensorRef<Element, Layout> src,
                    cutlass::gemm::GemmCoord problem_size) {

    TensorRef<Element, layout::RowMajorInterleaved<LayoutInterleaved>> mappedDest(dest.data(), dest.stride(0));
    TensorRef<Element, layout::RowMajorInterleaved<LayoutInterleaved>> mappedSrc(src.data(), src.stride(0));
    
    reorder_column<ColumnInterleaved>(
        mappedDest, mappedSrc, problem_size);
}

/// This is needed for the sparse tensor core kernels.  The purpose
/// is to use ldmatrix to load from shared memory to the register file.
template <typename Element, typename LayoutDest, typename LayoutSrc>
void reorder_meta(TensorRef<Element, LayoutDest> dest,
                  TensorRef<Element, LayoutSrc> src,
                  cutlass::gemm::GemmCoord problem_size) {
  for (int m = 0; m < problem_size.m(); m++) {
    for (int k = 0; k < problem_size.k(); k++) {
      // First reorder the rows.
      int group = (sizeof(Element) == 2) ? 32 : 16;
      int interweave = (sizeof(Element) == 2) ? 4 : 2;

      int dest_row = m / group * group + (m % 8) * interweave + (m % group) / 8;
      int dest_col = k;

      // Next swizzle the 2x2 blocks from Z to N.
      if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)) {
        ++dest_row;
        --dest_col;
      } else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)) {
        --dest_row;
        ++dest_col;
      }

      dest.at({dest_row, dest_col}) = src.at({m, k});
    }
  }
}
} // namespace cutlass
