/******************************************************************************
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
 ******************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_utils.h"
#include <cfloat>

namespace cutlass {

__global__ void rmsnorm_twoPassAlgo_e8(float4 *output, const float4 *input,
                                       const float4 *weight,
                                       const int m, const int n, float epsilon) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean;
  float local_sums[1] = {0.0f};
  const int n_8 = n / 8;
  int offset = m_idx * n_8;
  input += offset;
  output += offset;

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = input[index];
    const half2 *h1 = (half2 *)&local_val.x;
    const half2 *h2 = (half2 *)&local_val.y;
    const half2 *h3 = (half2 *)&local_val.z;
    const half2 *h4 = (half2 *)&local_val.w;
    local_sums[0] += static_cast<float>(h1->x) * static_cast<float>(h1->x) +
                     static_cast<float>(h1->y) * static_cast<float>(h1->y) +
                     static_cast<float>(h2->x) * static_cast<float>(h2->x) +
                     static_cast<float>(h2->y) * static_cast<float>(h2->y) +
                     static_cast<float>(h3->x) * static_cast<float>(h3->x) +
                     static_cast<float>(h3->y) * static_cast<float>(h3->y) +
                     static_cast<float>(h4->x) * static_cast<float>(h4->x) +
                     static_cast<float>(h4->y) * static_cast<float>(h4->y);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = rsqrtf(local_sums[0] / n + epsilon);
  }
  __syncthreads();

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = input[index];
    const float4 weight_val = weight[index];

    const half2 *l1 = (half2 *)&local_val.x;
    const half2 *l2 = (half2 *)&local_val.y;
    const half2 *l3 = (half2 *)&local_val.z;
    const half2 *l4 = (half2 *)&local_val.w;

    const half2 *g1 = (half2 *)&weight_val.x;
    const half2 *g2 = (half2 *)&weight_val.y;
    const half2 *g3 = (half2 *)&weight_val.z;
    const half2 *g4 = (half2 *)&weight_val.w;

    float4 tmp;
    half2 *h1 = (half2 *)&tmp.x;
    half2 *h2 = (half2 *)&tmp.y;
    half2 *h3 = (half2 *)&tmp.z;
    half2 *h4 = (half2 *)&tmp.w;

    h1->x = half(static_cast<float>(l1->x) * s_mean * static_cast<float>(g1->x));
    h1->y = half(static_cast<float>(l1->y) * s_mean * static_cast<float>(g1->y));
    h2->x = half(static_cast<float>(l2->x) * s_mean * static_cast<float>(g2->x));
    h2->y = half(static_cast<float>(l2->y) * s_mean * static_cast<float>(g2->y));
    h3->x = half(static_cast<float>(l3->x) * s_mean * static_cast<float>(g3->x));
    h3->y = half(static_cast<float>(l3->y) * s_mean * static_cast<float>(g3->y));
    h4->x = half(static_cast<float>(l4->x) * s_mean * static_cast<float>(g4->x));
    h4->y = half(static_cast<float>(l4->y) * s_mean * static_cast<float>(g4->y));

    output[index] = tmp;
  }
}

template<typename T>
__global__ void rmsnorm_twoPassAlgo_e1(T* output,
                                       const T* input,
                                       const T* weight,
                                       const int m, const int n,
                                       float epsilon)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean;
  float local_sums[1] = {0.0f};
  int offset = m_idx * n;
  input += offset;
  output += offset;

  for (int index = tid ; index < n ; index += bdimx){
    float local_val = static_cast<float>(input[index]);
    local_sums[0] += local_val * local_val;
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = rsqrtf(local_sums[0] / n + epsilon);
  }
  __syncthreads();

  for (int index = tid ; index < n ; index += bdimx){
    const T weight_val = weight[index];
    const T local_val = input[index];
    output[index] = T(static_cast<float>(local_val) * s_mean * static_cast<float>(weight_val));
  }
}

template <typename T>
void rmsnorm(cutlass::MatrixCoord tensor_size,
             TensorRef<T, layout::RowMajor> ref_output,
             TensorRef<T, layout::RowMajor> ref_input,
             TensorRef<T, layout::RowMajor> ref_weight,
             cudaStream_t stream, float epsilon = 1e-5f){
  const int m = tensor_size.row();
  const int n = tensor_size.column();
  T* output = ref_output.data();
  const T* input = ref_input.data();
  const T* weight = ref_weight.data();
  dim3 grid(m);

  if (n % 8 == 0 && std::is_same<T, cutlass::half_t>::value) {
    dim3 block(cutlass::platform::min(1024, (n / 8 + 31) / 32 * 32));

    rmsnorm_twoPassAlgo_e8<<<grid, block, 0, stream>>>(
        (float4 *)output, (const float4 *)input, (const float4 *)weight, m, n, epsilon);
  } else {
    dim3 block(cutlass::platform::min(1024, ((n + 31)/32 + 31)/32*32));

    rmsnorm_twoPassAlgo_e1<<<grid, block, 0, stream>>>(
        output, input, weight, m, n, epsilon);
  }

  auto result = cudaGetLastError();
  if (result != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(result) << std::endl;
    abort();
  }
}

} // namespace cutlass
