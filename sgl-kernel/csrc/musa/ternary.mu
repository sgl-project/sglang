/*
 * Copyright (c) 2020-2026, Moore Threads Technology Co., Ltd("Moore Threads").
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include "musa/dispatch_utils.h"
#include "musa.h"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"


typedef __half float16_t;
typedef __mt_bfloat16 bfloat16_t;

__device__ __host__ __forceinline__ constexpr int64_t ceil_div(int64_t a,
                                                               int64_t b) {
  return (a + b - 1) / b;
}

template <typename scalar_t, int64_t vlen, int iobit = 128>
__global__ void FusedMulAdd(scalar_t *out, scalar_t *self, scalar_t *bias,
                            const scalar_t scale, const int64_t elem) {
  constexpr int bits_of_byte = 8;
  using Vec =
      at::musa::VecType<scalar_t, vlen * sizeof(scalar_t) * bits_of_byte>;

  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;
  int64_t grid_stride_vec = grid_stride * vlen;

  for (int64_t offset = tid * vlen; offset < elem; offset += grid_stride_vec) {
    Vec t = Vec::load(self, offset);
    Vec b = Vec::load(bias, offset);
    Vec o;
#pragma unroll
    for (int k = 0; k < vlen; ++k) {
      o.val_.elem[k] = b.val_.elem[k] + t.val_.elem[k] * scale;
    }
    Vec::store(out, offset, o);
  }
}

void fused_mul_add(torch::Tensor &output, torch::Tensor &self,
                    torch::Tensor &bias, const double scale) {
  TORCH_CHECK(self.sizes() == output.sizes(),
              "self and output shape don't match");
  TORCH_CHECK(self.sizes() == bias.sizes(), "self and bias shape don't match");
  TORCH_CHECK(self.scalar_type() == bias.scalar_type(),
              "self and bias should be same type");
  TORCH_CHECK(self.scalar_type() == output.scalar_type(),
              "self and output should be same type");
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Float ||
                  self.scalar_type() == at::ScalarType::BFloat16 ||
                  self.scalar_type() == at::ScalarType::Half,
              "self's dtype should be in float/half/bfloat16");

  // device guard
  const at::musa::OptionalMUSAGuard device_guard(device_of(self));

  // Suppose the uncontiguous elementwise is much slower
  if C10_UNLIKELY (!self.is_contiguous()) {
    self = self.contiguous();
  }
  if C10_UNLIKELY (!bias.is_contiguous()) {
    bias = bias.contiguous();
  }

  // follow mudnn config for arch==22
  const int64_t max_load_vec =
      self.scalar_type() == at::ScalarType::Float ? 4 : 8;
  const int64_t numel = self.numel();
  size_t thread_per_block = 512;
  if (ceil_div(numel, max_load_vec) <= 128) {
    thread_per_block = 128;
  } else if (ceil_div(numel, max_load_vec) <= 256) {
    thread_per_block = 256;
  }
  size_t nr_block = ceil_div(numel, max_load_vec * thread_per_block);

  const musaStream_t stream = at::musa::getCurrentMUSAStream();

  switch (self.scalar_type()) {
  case at::ScalarType::Float:
    FusedMulAdd<float, 4><<<nr_block, thread_per_block, 0, stream>>>(
        static_cast<float *>(output.data_ptr()),
        static_cast<float *>(self.data_ptr()),
        static_cast<float *>(bias.data_ptr()), scale, numel);
    break;
  case at::ScalarType::Half:
    FusedMulAdd<float16_t, 8>
        <<<nr_block, thread_per_block, 0, stream>>>(
            static_cast<float16_t *>(output.data_ptr()),
            static_cast<float16_t *>(self.data_ptr()),
            static_cast<float16_t *>(bias.data_ptr()),
            static_cast<float16_t>(scale), numel);
    break;
  case at::ScalarType::BFloat16:
    FusedMulAdd<bfloat16_t, 8>
        <<<nr_block, thread_per_block, 0, stream>>>(
            static_cast<bfloat16_t *>(output.data_ptr()),
            static_cast<bfloat16_t *>(self.data_ptr()),
            static_cast<bfloat16_t *>(bias.data_ptr()),
            static_cast<bfloat16_t>(scale), numel);
    break;
  default:
    break;
  }
}
