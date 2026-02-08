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

#include "cutlass/util/device_rmsnorm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/constants.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"

using ElementType = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;

void rmsnorm_host(cutlass::MatrixCoord tensor_size,
		  cutlass::TensorRef<ElementType, Layout> output,
		  cutlass::TensorRef<ElementType, Layout> input,
		  cutlass::TensorRef<ElementType, Layout> weight,
                  float epsilon) {
  const int M = tensor_size.row();
  const int N = tensor_size.column();

  for (int m = 0; m < M; ++m) {
    float square_sum{0};

    for (int n = 0; n < N; ++n) {
      float inp = static_cast<float>(input.at({m, n}));
      square_sum += inp * inp;
    }

    float sq_mean = square_sum / (float)N;
    float sqrt_var = cutlass::fast_sqrt(sq_mean + epsilon);

    for (int n = 0; n < N; ++n) {
      float inp = static_cast<float>(input.at({m, n}));
      float g = static_cast<float>(weight.at({0, n}));
      float res_fp32 = inp / sqrt_var * g;
      output.at({m, n}) = ElementType(res_fp32);
    }
  }
}

void run_test(int M, int N) {
  cutlass::HostTensor<ElementType, Layout> input, output_ref, output, weight;
  input.reset({M, N});
  output.reset({M, N});
  output_ref.reset({M, N});
  weight.reset({1, N});

  const unsigned seed = 2022;

  cutlass::reference::host::TensorFillRandomUniform(input.host_view(),
						    seed,
						    ElementType(5),
						    ElementType(-5),
						    0);

  cutlass::reference::host::TensorFillRandomUniform(weight.host_view(),
						    seed,
						    ElementType(5),
						    ElementType(-5),
						    0);

  input.sync_device();
  weight.sync_device();

  rmsnorm_host({M, N}, output_ref.host_ref(), input.host_ref(), weight.host_ref(), (float)1e-5);
  cutlass::rmsnorm({M, N}, output.device_ref(),
		   input.device_ref(), weight.device_ref(), NULL, (float)1e-5L);

  output.sync_host();

  float max_abs_diff = -1;
  float mean_abs_diff = 0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      auto diff = abs(static_cast<float>(output_ref.at({m, n}) - output.at({m, n})));
      mean_abs_diff += diff;
      max_abs_diff = cutlass::platform::max(max_abs_diff, diff);
    }
  }

  mean_abs_diff /= float(M * N);

  EXPECT_TRUE(max_abs_diff < 0.001f && mean_abs_diff < 0.001f)
    << "Max absolute difference  : " << max_abs_diff << "\n"
    << "Mean absolute difference: " << mean_abs_diff;
}

TEST(RMSNorm, 16x1024) {
  run_test(16, 1024);
}

TEST(RMSNorm, 1x127) {
  run_test(1, 127);
}
