/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>


#include "utils.h"

__device__ __forceinline__ float silu_f(float x) {
  return x / (1.f + __expf(-x));
}

template <typename T>
__global__ void GatedRMSNormFusedKernel(
    const T* __restrict__ x,
    const T* __restrict__ z,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int stride = blockDim.x;
  const T* x_row = x + row * hidden;
  const T* z_row = z + row * hidden;
  T* y_row = y + row * hidden;

  float sum_sq = 0.f;
  for (int i = tid; i < hidden; i += stride) {
    float xv = static_cast<float>(x_row[i]);
    sum_sq += xv * xv;
  }

  __shared__ float smem[32];
  float v = sum_sq;
  for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) v += SGLANG_SHFL_XOR_SYNC(0xffffffff, v, mask);
  if ((tid & (WARP_SIZE - 1)) == 0) smem[tid >> 5] = v;
  __syncthreads();
  float total = 0.f;
  if (tid < 32) {
    total = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? smem[tid] : 0.f;
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) total += SGLANG_SHFL_XOR_SYNC(0xffffffff, total, mask);
  }
  __shared__ float s_rstd;
  if (tid == 0) s_rstd = rsqrtf(total / hidden + eps);
  __syncthreads();
  float rstd = s_rstd;

  for (int i = tid; i < hidden; i += stride) {
    float xv = static_cast<float>(x_row[i]);
    float zv = static_cast<float>(z_row[i]);
    float wv = static_cast<float>(weight[i]);
    float normed = xv * rstd * wv;
    float gated = normed * silu_f(zv);
    y_row[i] = static_cast<T>(gated);
  }
}

template <typename T>
__global__ void GatedPointwiseKernel(
    const T* __restrict__ z,
    T* __restrict__ y,
    int hidden) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int stride = blockDim.x;
  const T* z_row = z + row * hidden;
  T* y_row = y + row * hidden;
  for (int i = tid; i < hidden; i += stride) {
    float zv = static_cast<float>(z_row[i]);
    float yv = static_cast<float>(y_row[i]);
    y_row[i] = static_cast<T>(yv * silu_f(zv));
  }
}

void gated_rmsnorm(
    at::Tensor& out,
    at::Tensor& x,
    at::Tensor& z,
    at::Tensor& weight,
    double eps,
    bool enable_pdl) {
  CHECK_INPUT(x);
  CHECK_INPUT(z);
  CHECK_INPUT(weight);
  CHECK_INPUT(out);
  auto device = x.device();
  CHECK_EQ(z.device(), device);
  CHECK_EQ(weight.device(), device);
  CHECK_EQ(out.device(), device);
  CHECK_DIM(2, x);
  CHECK_DIM(2, z);
  CHECK_DIM(2, out);
  CHECK_DIM(1, weight);
  CHECK_EQ(x.size(0), z.size(0));
  CHECK_EQ(x.size(1), z.size(1));
  CHECK_EQ(x.size(0), out.size(0));
  CHECK_EQ(x.size(1), out.size(1));
  CHECK_EQ(x.size(1), weight.size(0));

  const int rows = static_cast<int>(x.size(0));
  const int hidden = static_cast<int>(x.size(1));
  const int threads = 256;
  const dim3 grid(rows);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(x.scalar_type(), c_type, [&] {
    GatedRMSNormFusedKernel<c_type><<<grid, threads, 0, stream>>>(
        reinterpret_cast<const c_type*>(x.data_ptr()),
        reinterpret_cast<const c_type*>(z.data_ptr()),
        reinterpret_cast<const c_type*>(weight.data_ptr()),
        reinterpret_cast<c_type*>(out.data_ptr()),
        hidden,
        static_cast<float>(eps));
    return true;
  });
}


