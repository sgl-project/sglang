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
\brief unit tests for tensor layout
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace test {
namespace layout {
  void test_NHWC_layout(int n_size, int h_size, int w_size, int c_size) {
    int ldc = c_size + 1;
    int ldw = ldc * (w_size + 2);
    int ldh = ldw * (h_size + 3);

    cutlass::layout::TensorNHWC::Stride tensor_stride({ ldc, ldw, ldh });

    cutlass::layout::TensorNHWC tensor_nhwc(tensor_stride);

    // test pointer offset
    for (int n_idx = 0; n_idx < n_size; n_idx++) {
      for (int h_idx = 0; h_idx < h_size; h_idx++) {
        for (int w_idx = 0; w_idx < w_size; w_idx++) {
          for (int c_idx = 0; c_idx < c_size; c_idx++) {
            cutlass::Tensor4DCoord tensor_coord(n_idx, h_idx, w_idx, c_idx);
            auto ptr_offset = tensor_nhwc(tensor_coord);
            decltype(ptr_offset) reference_offset = c_idx +
              w_idx * ldc +
              h_idx * ldw +
              n_idx * ldh;
            EXPECT_EQ(ptr_offset, reference_offset);
          }
        }
      }
    }

    // test stride
    auto stride = tensor_nhwc.stride();
    EXPECT_EQ(stride, tensor_stride);

    // test capacity
    auto capacity = tensor_nhwc.capacity(cutlass::Tensor4DCoord(n_size, h_size, w_size, c_size));
    decltype(capacity) reference_capacity = ldh * n_size;
    EXPECT_EQ(capacity, reference_capacity);

    // test packed
    auto packed_tensor_layout = tensor_nhwc.packed(cutlass::Tensor4DCoord(n_size, h_size, w_size, c_size));
    auto packed_stride = packed_tensor_layout.stride();
    EXPECT_EQ(packed_stride, cutlass::layout::TensorNHWC::Stride({ c_size, w_size * c_size, h_size * w_size * c_size }));
  }


  void test_NCHW_layout(int n_size, int c_size, int h_size, int w_size) {
    int ldw = w_size + 1;
    int ldh = ldw * (h_size + 2);
    int ldc = ldh * (c_size + 1);

    cutlass::layout::TensorNCHW::Stride tensor_stride({ ldw, ldh, ldc });

    cutlass::layout::TensorNCHW tensor_nchw(tensor_stride);

    // test pointer offset
    for (int n_idx = 0; n_idx < n_size; n_idx++) {
      for (int c_idx = 0; c_idx < c_size; c_idx++) {
        for (int h_idx = 0; h_idx < w_size; h_idx++) {
          for (int w_idx = 0; w_idx < c_size; w_idx++) {
            // tensor4DCoord is always created in nhwc order
            cutlass::Tensor4DCoord tensor_coord(n_idx, h_idx, w_idx, c_idx);
            auto ptr_offset = tensor_nchw(tensor_coord);
            decltype(ptr_offset) reference_offset = w_idx +
              h_idx * ldw +
              c_idx * ldh +
              n_idx * ldc;
            EXPECT_EQ(ptr_offset, reference_offset);
          }
        }
      }
    }

    // test stride
    auto stride = tensor_nchw.stride();
    EXPECT_EQ(stride, tensor_stride);

    // test capacity
    auto capacity = tensor_nchw.capacity(cutlass::Tensor4DCoord(n_size, h_size, w_size, c_size));
    decltype(capacity) reference_capacity = ldc * n_size;
    EXPECT_EQ(capacity, reference_capacity);

    // test packed
    auto packed_tensor_layout = tensor_nchw.packed(cutlass::Tensor4DCoord(n_size, h_size, w_size, c_size));
    auto packed_stride = packed_tensor_layout.stride();
    EXPECT_EQ(packed_stride, cutlass::layout::TensorNHWC::Stride({ w_size, w_size * h_size, w_size * h_size * c_size }));
  }
} // namespace layout
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Layout_Tensor, NHWC_32_12_10_14) {
  int n_size = 32;
  int h_size = 12;
  int w_size = 10;
  int c_size = 14;
  test::layout::test_NHWC_layout(n_size, h_size, w_size, c_size);

}


/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Layout_Tensor, NCHW_32_12_10_14) {
  int n_size = 32;
  int c_size = 12;
  int h_size = 10;
  int w_size = 14;
  test::layout::test_NCHW_layout(n_size, c_size, h_size, w_size);

}

/////////////////////////////////////////////////////////////////////////////////////////////////

