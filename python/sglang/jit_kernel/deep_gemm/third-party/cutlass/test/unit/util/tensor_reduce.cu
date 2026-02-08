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
#include <complex>

#include "../common/cutlass_unit_test.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/util/reference/device/tensor_reduce.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/host_tensor.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TensorReduce, norm_rowmajor_f32) {

  int const kM = 129;
  int const kN = 91;

  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor({kM, kN});

  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {

      float x = float(((m * kN + m + 7) % 8) - 4);

      tensor.at({m, n}) = x;
    }
  }

  tensor.sync_device();

  double device_norm = cutlass::reference::device::TensorNorm(tensor.device_view(), double());
  double host_norm = cutlass::reference::host::TensorNorm(tensor.host_view(), double());

  EXPECT_TRUE(std::abs(host_norm - device_norm) < 0.001);
}

TEST(TensorReduce, norm_nhwc_f32) {

  int const kN = 19;
  int const kH = 18;
  int const kW = 17;
  int const kC = 16;

  cutlass::HostTensor<float, cutlass::layout::TensorNHWC> tensor({kN, kH, kW, kC});

  int idx = 0;

  double computed_norm = double();

  for (int n = 0; n < kN; ++n) {
    for (int h = 0; h < kH; ++h) {
      for (int w = 0; w < kW; ++w) {
        for (int c = 0; c < kC; ++c, ++idx) {
      
          float x = float(((idx + 7) % 8) - 4);

          computed_norm += double(x) * double(x);

          tensor.at({n, h, w, c}) = x;
        }
      }
    }
  }

  computed_norm = std::sqrt(computed_norm);

  tensor.sync_device();

  double device_norm = cutlass::reference::device::TensorNorm(tensor.device_view(), double());
  double host_norm = cutlass::reference::host::TensorNorm(tensor.host_view(), double());

  EXPECT_TRUE(std::abs(host_norm - device_norm) < 0.001 && std::abs(computed_norm - host_norm) < 0.001)
    << "computed norm: " << computed_norm << "\n"
    << " host norm: " << host_norm << "\n"
    << "device norm: " << device_norm << "\n";
}

TEST(TensorReduce, norm_nhwc_f16) {

  int const kN = 69;
  int const kH = 68;
  int const kW = 67;
  int const kC = 66;

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::TensorNHWC> tensor({kN, kH, kW, kC});

  int idx = 0;

  double computed_norm = double();

  for (int n = 0; n < kN; ++n) {
    for (int h = 0; h < kH; ++h) {
      for (int w = 0; w < kW; ++w) {
        for (int c = 0; c < kC; ++c, ++idx) {
      
          float x = float(((idx + 7) % 8) - 4);
          computed_norm += double(x) * double(x);

          tensor.at({n, h, w, c}) = cutlass::half_t(x);
        }
      }
    }
  }
  
  computed_norm = std::sqrt(computed_norm);

  tensor.sync_device();

  double device_norm = cutlass::reference::device::TensorNorm(tensor.device_view(), double());
  double host_norm = cutlass::reference::host::TensorNorm(tensor.host_view(), double());

  EXPECT_TRUE(std::abs(host_norm - device_norm) < 0.001 && std::abs(computed_norm - host_norm) < 0.001)
    << "computed norm: " << computed_norm << "\n"
    << " host norm: " << host_norm << "\n"
    << "device norm: " << device_norm << "\n";
}

TEST(TensorReduce, norm_diff_nhwc_f32) {

  int const kN = 59;
  int const kH = 24;
  int const kW = 57;
  int const kC = 78;

  using Layout = cutlass::layout::TensorNHWC;

  cutlass::HostTensor<float, Layout> tensor_A({kN, kH, kW, kC});
  cutlass::HostTensor<float, Layout> tensor_B({kN, kH, kW, kC});


  int idx = 0;

  double sum_sq_diff = 0;

  for (int n = 0; n < kN; ++n) {
    for (int h = 0; h < kH; ++h) {
      for (int w = 0; w < kW; ++w) {
        for (int c = 0; c < kC; ++c, ++idx) {
      
          float a = float(((idx * 5 + 7) % 8) - 4);
          float b = float(((idx * 3 + 7) % 8) - 4);

          sum_sq_diff += double(a - b) * double(a - b);

          tensor_A.at({n, h, w, c}) = a;
          tensor_B.at({n, h, w, c}) = b;
        }
      }
    }
  }

  tensor_A.sync_device();
  tensor_B.sync_device();

  double device_norm = cutlass::reference::device::TensorNormDiff(
    tensor_A.device_view(), tensor_B.device_view(), double());

  double host_norm = std::sqrt(sum_sq_diff);
  
  EXPECT_TRUE(std::abs(host_norm - device_norm) < 0.001f)
    << "  host norm: " << host_norm << "\n"
    << "device norm: " << device_norm;
}


TEST(TensorReduce, norm_diff_nhwc_f16) {

  int const kN = 59;
  int const kH = 24;
  int const kW = 57;
  int const kC = 78;

  using Layout = cutlass::layout::TensorNHWC;

  cutlass::HostTensor<cutlass::half_t, Layout> tensor_A({kN, kH, kW, kC});
  cutlass::HostTensor<cutlass::half_t, Layout> tensor_B({kN, kH, kW, kC});

  int idx = 0;

  double sum_sq_diff = 0;

  for (int n = 0; n < kN; ++n) {
    for (int h = 0; h < kH; ++h) {
      for (int w = 0; w < kW; ++w) {
        for (int c = 0; c < kC; ++c, ++idx) {
      
          float a = float(((idx * 5 + 7) % 8) - 4);
          float b = float(((idx * 3 + 7) % 8) - 4);

          sum_sq_diff += double(a - b) * double(a - b);

          tensor_A.at({n, h, w, c}) = cutlass::half_t(a);
          tensor_B.at({n, h, w, c}) = cutlass::half_t(b);
        }
      }
    }
  }

  tensor_A.sync_device();
  tensor_B.sync_device();

  double device_norm = cutlass::reference::device::TensorNormDiff(
    tensor_A.device_view(), tensor_B.device_view(), double());

  double host_norm = std::sqrt(sum_sq_diff);
  
  EXPECT_TRUE(std::abs(host_norm - device_norm) < 0.001f)
    << "  host norm: " << host_norm << "\n"
    << "device norm: " << device_norm;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

