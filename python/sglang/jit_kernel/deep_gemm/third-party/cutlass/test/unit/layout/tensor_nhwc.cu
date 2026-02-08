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
\brief unit tests for NHWC tensor layout
*/

#include "../common/cutlass_unit_test.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/util/device_memory.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace test {
namespace layout {

  void test_nhwc_layout(int n_size, int h_size, int w_size, int c_size) {
    int ldc = c_size + 1;
    int ldw = ldc * (w_size + 2);
    int ldh = ldw * (h_size + 3);

    typedef cutlass::layout::TensorNHWC Tensor;

    Tensor::Stride tensor_stride({ ldc, ldw, ldh });
    Tensor tensor_nhw_packed_c(tensor_stride);

    // test pointer offset
    for (int n_idx = 0; n_idx < n_size; n_idx++) {
      for (int p_idx = 0; p_idx < h_size; p_idx++) {
        for (int q_idx = 0; q_idx < w_size; q_idx++) {
          for (int c_idx = 0; c_idx < c_size; c_idx++) {
            cutlass::Tensor4DCoord tensor_coord(n_idx, p_idx, q_idx, c_idx);
            auto ptr_offset = tensor_nhw_packed_c(tensor_coord);
            decltype(ptr_offset) reference_offset = c_idx +
              q_idx * ldc +
              p_idx * ldw +
              n_idx * ldh;
            EXPECT_EQ(ptr_offset, reference_offset);
          }
        }
      }
    }

    // test stride
    auto stride = tensor_nhw_packed_c.stride();
    EXPECT_EQ(stride, tensor_stride);

    // test capacity
    auto capacity = tensor_nhw_packed_c.capacity(
            cutlass::Tensor4DCoord(n_size, h_size, w_size, c_size));
    decltype(capacity) reference_capacity = ldh * n_size;
    EXPECT_EQ(capacity, reference_capacity);

  }

  __global__ void test_nhwc_inverse(
          int *output, int n_size, int h_size, int w_size, int c_size) {
    int ldc = c_size;
    int ldw = ldc * w_size;
    int ldh = ldw * h_size;

    typedef cutlass::layout::TensorNHWC Tensor;

    Tensor::Stride tensor_stride({ ldc, ldw, ldh });
    Tensor tensor_nhw_packed_c(tensor_stride);

    for (int n_idx = 0; n_idx < n_size; n_idx++) {
      for (int p_idx = 0; p_idx < h_size; p_idx++) {
        for (int q_idx = 0; q_idx < w_size; q_idx++) {
            cutlass::Tensor4DCoord tensor_coord(n_idx, p_idx, q_idx, threadIdx.x);
            int ptr_offset = tensor_nhw_packed_c(tensor_coord);
            cutlass::Tensor4DCoord inv_coord = tensor_nhw_packed_c.inverse(ptr_offset);
            output[ptr_offset] = tensor_nhw_packed_c(inv_coord);
        }
      }
    }
  }

  class TestTensorNHWC {
    public:

  //
  // Data members
  //

  //
  // Methods
  //

  /// Ctor
  TestTensorNHWC() {

  }

  /// Runs the test
  void run(int n_size, int h_size, int w_size, int c_size) {

    size_t size = n_size * h_size * w_size * c_size;

    /// Device memory containing output
    cutlass::device_memory::allocation< int > output(size);
    int *output_host = (int *)malloc(sizeof(int) * size);

    dim3 grid(1,1);
    dim3 block(c_size, 1, 1);

    test::layout::test_nhwc_inverse<<< grid, block >>>(output.get(),
            n_size, h_size, w_size, c_size);

    cudaError_t result = cudaDeviceSynchronize();
    ASSERT_EQ(result, cudaSuccess) << "CUDA error: " << cudaGetErrorString(result);

    //
    // Verify output
    //

    cutlass::device_memory::copy_to_host(output_host, output.get(), size);

    result = cudaGetLastError();
    ASSERT_EQ(result, cudaSuccess) << "CUDA error: " << cudaGetErrorString(result);

    for (int n_idx = 0; n_idx < n_size; n_idx++) {
      for (int p_idx = 0; p_idx < h_size; p_idx++) {
        for (int q_idx = 0; q_idx < w_size; q_idx++) {
          for (int c_idx = 0; c_idx < c_size; c_idx++) {
            int reference_offset = c_idx +
              q_idx * c_size +
              p_idx * (c_size * w_size) +
              n_idx * (c_size * w_size * h_size);
              EXPECT_EQ(output_host[reference_offset], reference_offset);
          }
        }
      }
    }
  }
};


} // namespace layout
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Layout_TensorNHWC, NHWC_1_16_8_32) {
  int n_size = 1;
  int h_size = 16;
  int w_size = 8;
  int c_size = 32;
  test::layout::test_nhwc_layout(n_size, h_size, w_size, c_size);
  test::layout::TestTensorNHWC test_nhwc;
  test_nhwc.run(n_size, h_size, w_size, c_size);

}

TEST(Layout_TensorNHWC, NHWC_2_16_8_32) {
  int n_size = 2;
  int h_size = 16;
  int w_size = 8;
  int c_size = 32;
  test::layout::test_nhwc_layout(n_size, h_size, w_size, c_size);
  test::layout::TestTensorNHWC test_nhwc;
  test_nhwc.run(n_size, h_size, w_size, c_size);
}

TEST(Layout_TensorNHWC, NHWC_2_16_8_128) {
  int n_size = 2;
  int h_size = 16;
  int w_size = 8;
  int c_size = 128;
  test::layout::test_nhwc_layout(n_size, h_size, w_size, c_size);
  test::layout::TestTensorNHWC test_nhwc;
  test_nhwc.run(n_size, h_size, w_size, c_size);

}

TEST(Layout_TensorNHWC, NHWC_4_8_16_128) {
  int n_size = 4;
  int h_size = 8;
  int w_size = 16;
  int c_size = 128;
  test::layout::test_nhwc_layout(n_size, h_size, w_size, c_size);
  test::layout::TestTensorNHWC test_nhwc;
  test_nhwc.run(n_size, h_size, w_size, c_size);

}

/////////////////////////////////////////////////////////////////////////////////////////////////

