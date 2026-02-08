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

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ProblemShape,
  class TensorSeq,
  class TensorPageTable,
  class TensorQL,
  class TensorQR,
  class TensorCL,
  class TensorKR,
  class TensorO,
  class TensorLSE,
  class Scale
>
void __global__ fmha_mla_reference_kernel(
    ProblemShape problem_shape,
    TensorSeq mSeq, TensorPageTable mPT,
    TensorQL mQL, TensorQR mQR,
    TensorCL mCL, TensorKR mKR,
    TensorO mO, TensorLSE mLSE,
    Scale softmax_scale) {

  using namespace cute;

  auto [H, K, D, B] = problem_shape;
  auto [D_latent, D_rope] = D;

  using Element = typename TensorO::value_type;
  using ElementAcc = typename TensorLSE::value_type;

  extern __shared__ ElementAcc mS[];
  // ElementAcc* mS = reinterpret_cast<ElementAcc*>(mS_mem);

  for (int idx_B = blockIdx.y; idx_B < B; idx_B += gridDim.y) {
    if (mSeq.data() != nullptr) {
      K = mSeq(idx_B);
    }

    for (int idx_H = blockIdx.x; idx_H < H; idx_H += gridDim.x) {

      for (int idx_K = threadIdx.x; idx_K < K; idx_K += blockDim.x) {
        ElementAcc acc = 0;

        for (int idx_D = 0; idx_D < D_latent; idx_D++) {
          int page_idx_K = idx_K;
          int page_idx_B = idx_B;
          if (mPT.data() != nullptr) {
            page_idx_B = mPT(idx_K / size<0>(mCL), idx_B); 
            page_idx_K = idx_K % size<0>(mCL);
          }
          ElementAcc eQ = mQL(idx_H, idx_D, idx_B);
          ElementAcc eK = mCL(page_idx_K, idx_D, page_idx_B);
          acc += eQ * eK;
        }

        for (int idx_D = 0; idx_D < D_rope; idx_D++) {
          int page_idx_K = idx_K;
          int page_idx_B = idx_B;
          if (mPT.data() != nullptr) {
            page_idx_B = mPT(idx_K / size<0>(mCL), idx_B); 
            page_idx_K = idx_K % size<0>(mCL);
          }
          ElementAcc eQ = mQR(idx_H, idx_D, idx_B);
          ElementAcc eK = mKR(page_idx_K, idx_D, page_idx_B);
          acc += eQ * eK;
        }
        mS[idx_K] = acc;
      }

      __syncthreads();

      ElementAcc maxS = -std::numeric_limits<ElementAcc>::infinity();
      for (int idx_K = 0; idx_K < K; idx_K++) {
        maxS = std::max<ElementAcc>(maxS, mS[idx_K]);
      }
      if (maxS == -std::numeric_limits<ElementAcc>::infinity()) maxS = 0;

      __syncthreads();

      for (int idx_K = threadIdx.x; idx_K < K; idx_K += blockDim.x) {
        mS[idx_K] = expf(softmax_scale * (mS[idx_K] - maxS));
      }

      __syncthreads();

      ElementAcc sum = 0;
      for (int idx_K = 0; idx_K < K; idx_K++) {
        sum += mS[idx_K];
      }

      ElementAcc o_scale = 1.0f / sum;

      for (int idx_D = threadIdx.x; idx_D < D_latent; idx_D += blockDim.x) {
        ElementAcc acc = 0;
        for (int idx_K = 0; idx_K < K; idx_K++) {
          int page_idx_K = idx_K;
          int page_idx_B = idx_B;
          if (mPT.data() != nullptr) {
            page_idx_B = mPT(idx_K / size<0>(mCL), idx_B); 
            page_idx_K = idx_K % size<0>(mCL);
          }
          ElementAcc eV = mCL(page_idx_K, idx_D, page_idx_B);
          ElementAcc eK = static_cast<Element>(mS[idx_K]);
          acc += eK * eV;
        }
        mO(idx_H, idx_D, idx_B) = static_cast<typename TensorO::value_type>(acc * o_scale);
      }

      if (threadIdx.x == 0) {
        mLSE(idx_H, idx_B) = log(sum) + softmax_scale * maxS;
      }

    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ProblemShape,
  class TensorSeq,
  class TensorPageTable,
  class TensorQL,
  class TensorQR,
  class TensorCL,
  class TensorKR,
  class TensorO,
  class TensorLSE,
  class Scale
>
void fmha_mla_reference(
    ProblemShape problem_shape,
    TensorSeq mSeq, TensorPageTable mPT,
    TensorQL mQL, TensorQR mQR,
    TensorCL mCL, TensorKR mKR,
    TensorO mO, TensorLSE mLSE,
    Scale scale) {

  using namespace cute;

  auto [H, K, D, B] = problem_shape;
  auto [D_latent, D_rope] = D;

  dim3 grid(H, B, 1);
  dim3 block(256);
  int shared_mem = K * int(sizeof(typename TensorLSE::value_type)) + 16;
  cudaError_t result;
  if (shared_mem >= (48 << 10)) {
    result = cudaFuncSetAttribute(
        &fmha_mla_reference_kernel<ProblemShape, TensorSeq, TensorPageTable, TensorQL, TensorQR, TensorCL, TensorKR, TensorO, TensorLSE, Scale>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem);
    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      throw std::runtime_error("couldn't perform smem optin");
    }
  }
  fmha_mla_reference_kernel<<<grid, block, shared_mem>>>(
      problem_shape, mSeq, mPT, mQL, mQR, mCL, mKR, mO, mLSE, scale);
  cudaDeviceSynchronize();
  result = cudaGetLastError();
  if (cudaSuccess != result) {
    throw std::runtime_error("couldn't execute reference");
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
