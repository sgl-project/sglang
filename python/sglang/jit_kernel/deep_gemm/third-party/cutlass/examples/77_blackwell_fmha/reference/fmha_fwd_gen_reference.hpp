/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <vector>
#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ElementAcc,
  class ProblemShape,
  class TensorQ,
  class TensorNewK,
  class TensorNewV,
  class TensorCacheK,
  class TensorCacheV,
  class TensorO
>
void __global__ fmha_fwd_gen_reference_kernel(
    ProblemShape problem_shape,
    const int* seqlen_kv, const int* cache_batch_idx, 
    TensorQ mQ, TensorNewK mNewK, TensorNewV mNewV,
    TensorCacheK mCacheK, TensorCacheV mCacheV, TensorO mO) {

  using namespace cute;
  extern __shared__ char mS_mem[];
  ElementAcc* mS = reinterpret_cast<ElementAcc*>(mS_mem);

  float scale = 1.0f / std::sqrt(float(get<2>(problem_shape)));

  if (mNewK.data() != nullptr) {
    // 1. copy in new_k to cache
    for (int idx_h = blockIdx.x; idx_h < size<3,0,1>(problem_shape); idx_h += gridDim.x) {
      for (int idx_b = blockIdx.z; idx_b < size<3,1>(problem_shape); idx_b += gridDim.z) {
        int idx_b_kv = cache_batch_idx != nullptr ? cache_batch_idx[idx_b] : idx_b;
        for (int idx_d = threadIdx.x; idx_d < size<2>(problem_shape); idx_d += blockDim.x) {
          mCacheK(seqlen_kv[idx_b], idx_d, make_coord(make_coord(_0{}, idx_h), idx_b_kv)) =
              mNewK(_0{}, idx_d, make_coord(make_coord(_0{}, idx_h), idx_b));
          mCacheV(seqlen_kv[idx_b], idx_d, make_coord(make_coord(_0{}, idx_h), idx_b_kv)) =
              mNewV(_0{}, idx_d, make_coord(make_coord(_0{}, idx_h), idx_b));
        }
      }
    }
  }

  // 2. compute attention
  for (int idx_h_kv = blockIdx.x; idx_h_kv < size<3,0,1>(problem_shape); idx_h_kv += gridDim.x) {
    for (int idx_h_qo = blockIdx.y; idx_h_qo < size<3,0,0>(problem_shape); idx_h_qo += gridDim.y) {
      int idx_h = idx_h_qo + size<3,0,0>(problem_shape) * idx_h_kv;
      for (int idx_b = blockIdx.z; idx_b < size<3,1>(problem_shape); idx_b += gridDim.z) {
        int idx_b_kv = cache_batch_idx != nullptr ? cache_batch_idx[idx_b] : idx_b;
        const int kDim = 128;
        ElementAcc reg_o[kDim] = {0};
        ElementAcc row_max = -INFINITY;
        ElementAcc row_sum = 0;
        auto iteration = [&](auto const& tK, auto const& tV) {
          ElementAcc reg_s = 0;
          for (int idx_d = 0; idx_d < kDim; idx_d++) {
            ElementAcc eQ = mQ(_0{}, idx_d, make_coord(idx_h, idx_b));
            ElementAcc eK = tK(idx_d);
            reg_s += eQ * eK;
          }

          ElementAcc old_row_max = row_max;
          row_max = std::max(row_max, reg_s);

          ElementAcc adjustment = std::exp(scale * (old_row_max - row_max));
          row_sum *= adjustment;
          for (int idx_d = 0; idx_d < kDim; idx_d++) {
            reg_o[idx_d] *= adjustment;
          }

          ElementAcc reg_p = std::exp(scale * (reg_s - row_max));
          row_sum += reg_p;

          for (int idx_d = 0; idx_d < kDim; idx_d++) {
            ElementAcc eV = tV(idx_d);
            reg_o[idx_d] += reg_p * eV;
          }
        };

        for (int idx_s = threadIdx.x; idx_s < seqlen_kv[idx_b]; idx_s += blockDim.x) {
          iteration(mCacheK(idx_s, _, make_coord(idx_h, idx_b_kv)), mCacheV(idx_s, _, make_coord(idx_h, idx_b_kv)));
        }

        if (mNewK.data() != nullptr && threadIdx.x == 0) {
          iteration(mNewK(_0{}, _, make_coord(idx_h, idx_b)), mNewV(_0{}, _, make_coord(idx_h, idx_b)));
        }

        mS[threadIdx.x] = row_max;
        __syncthreads();
        float old_row_max = row_max;
        for (int i = 0; i < blockDim.x; i++) {
          row_max = std::max(row_max, mS[i]);
        }
        __syncthreads();

        ElementAcc adjustment = std::exp(scale * (old_row_max - row_max));
        row_sum *= adjustment;
        for (int idx_d = 0; idx_d < kDim; idx_d++) {
          reg_o[idx_d] *= adjustment;
        }
        mS[threadIdx.x] = row_sum;
        __syncthreads();

        row_sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
          row_sum += mS[i];
        }
        __syncthreads();
        for (int idx_d = 0; idx_d < kDim; idx_d++) {
          mS[idx_d] = 0;
        }
        __syncthreads();

        for (int idx_d = 0; idx_d < kDim; idx_d++) {
          reg_o[idx_d] /= row_sum;
          atomicAdd(&mS[idx_d], reg_o[idx_d]);
        }

        __syncthreads();
        for (int idx_d = threadIdx.x; idx_d < kDim; idx_d += blockDim.x) {
          mO(_0{}, idx_d, make_coord(idx_h, idx_b)) = static_cast<typename TensorO::value_type>(mS[idx_d]);
        }
      }
    }
  }
}

template<
  class ElementAcc,
  class ProblemShape,
  class TensorQ,
  class TensorNewK,
  class TensorNewV,
  class TensorCacheK,
  class TensorCacheV,
  class TensorO
>
void fmha_fwd_gen_reference(
    ProblemShape problem_shape,
    const int* seqlen_kv, const int* cache_batch_idx, 
    TensorQ mQ, TensorNewK mNewK, TensorNewV mNewV,
    TensorCacheK mCacheK, TensorCacheV mCacheV, TensorO mO) {

  using namespace cute;

  dim3 grid(get<3,0,1>(problem_shape), get<3,0,0>(problem_shape), get<3,1>(problem_shape));
  dim3 block(128);
  int shared_mem = int(sizeof(ElementAcc)) * std::max<int>(128, block.x);
  assert(get<2>(problem_shape) == 128);
  fmha_fwd_gen_reference_kernel<ElementAcc><<<grid, block, shared_mem>>>(
      problem_shape, seqlen_kv, cache_batch_idx,
      mQ, mNewK, mNewV, mCacheK, mCacheV, mO
  );
}
