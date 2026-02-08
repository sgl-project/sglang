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

#include "cutlass_unit_test.h"

#include <iostream>
#include <iomanip>
#include <utility>
#include <type_traits>
#include <vector>
#include <numeric>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

using namespace cute;

__global__ void
test(double const* g_in, double* g_out)
{
  extern __shared__ double smem[];

  smem[threadIdx.x] = g_in[threadIdx.x];

  __syncthreads();

  g_out[threadIdx.x] = 2 * smem[threadIdx.x];
}

__global__ void
test2(double const* g_in, double* g_out)
{
  using namespace cute;

  extern __shared__ double smem[];

  auto s_tensor = make_tensor(make_smem_ptr(smem + threadIdx.x), Int<1>{});
  auto g_tensor = make_tensor(make_gmem_ptr(g_in + threadIdx.x), Int<1>{});

  copy(g_tensor, s_tensor);

  __syncthreads();

  g_out[threadIdx.x] = 2 * smem[threadIdx.x];
}

TEST(SM80_CuTe_Ampere, CpSync)
{
  constexpr int count = 32;
  thrust::host_vector<double> h_in(count);
  for (int i = 0; i < count; ++i) {
    h_in[i] = double(i);
  }

  thrust::device_vector<double> d_in(h_in);

  thrust::device_vector<double> d_out(count, -1);
  test<<<1, count, sizeof(double) * count>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()));
  thrust::host_vector<double> h_result = d_out;

  thrust::device_vector<double> d_out_cp_async(count, -2);
  test2<<<1, count, sizeof(double) * count>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out_cp_async.data()));
  thrust::host_vector<double> h_result_cp_async = d_out_cp_async;

  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(h_result[i], h_result_cp_async[i]);
  }
}
