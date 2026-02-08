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
  class TensorQ,
  class TensorK,
  class TensorV,
  class TensorO,
  class TensorLSE,
  class Fusion
>
void __global__ fmha_reference_kernel(
    ProblemShape problem_shape,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE,
    Fusion fusion
) {
  using namespace cute;

  using Element = typename TensorO::value_type;
  using ElementAccumulator = typename TensorLSE::value_type;
  
  extern __shared__ char mS_mem[];
  Element* mS = reinterpret_cast<Element*>(mS_mem);

  ElementAccumulator softmax_scale = static_cast<ElementAccumulator>(1.0 / sqrt(1.0 * size<1>(mO)));

  auto id = make_identity_tensor(make_shape(1, 1));
  for (int idx_L = blockIdx.y; idx_L < size<2>(mO); idx_L += gridDim.y) {
    for (int idx_Q = blockIdx.x; idx_Q < size<0>(mO); idx_Q += gridDim.x) {
      for (int idx_K = threadIdx.x; idx_K < size<0>(mK); idx_K += blockDim.x) {
        ElementAccumulator acc = 0;
        for (int idx_D = 0; idx_D < size<1>(mK); idx_D++) {
          acc += mQ(idx_Q, idx_D, idx_L) * mK(idx_K, idx_D, idx_L);
        }
        auto frag = make_tensor<ElementAccumulator>(Shape<_1, _1>{});
        frag(0) = acc;
        fusion.before_softmax(frag, make_tensor(id.data() + make_arithmetic_tuple(idx_Q, idx_K), id.layout()), problem_shape);
        mS[idx_K] = static_cast<Element>(frag(0) * softmax_scale);
      }

      __syncthreads();

      ElementAccumulator maxS = -std::numeric_limits<ElementAccumulator>::infinity();
      for (int idx_K = 0; idx_K < size<0>(mK); idx_K++) {
        maxS = std::max<ElementAccumulator>(maxS, mS[idx_K]);
      }
      if (maxS == -std::numeric_limits<ElementAccumulator>::infinity()) maxS = 0;

      __syncthreads();

      for (int idx_K = threadIdx.x; idx_K < size<0>(mK); idx_K += blockDim.x) {
        mS[idx_K] = static_cast<Element>(exp(mS[idx_K] - maxS));
      }

      __syncthreads();

      ElementAccumulator sum = 0;
      for (int idx_K = 0; idx_K < size<0>(mK); idx_K++) {
        sum += mS[idx_K];
      }

      Element scale = static_cast<Element>(1.0 / sum);

      for (int idx_D = threadIdx.x; idx_D < size<1>(mO); idx_D += blockDim.x) {
        ElementAccumulator acc = 0;
        for (int idx_K = 0; idx_K < size<0>(mK); idx_K++) {
          acc += mS[idx_K] * mV(idx_K, idx_D, idx_L) * scale;
        }
        mO(idx_Q, idx_D, idx_L) = static_cast<Element>(acc);
      }

      if (threadIdx.x == 0) {
        mLSE(idx_Q, idx_L) = log(sum) + maxS;
      }

    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ProblemShape,
  class TensorQ,
  class TensorK,
  class TensorV,
  class TensorO,
  class TensorLSE,
  class Fusion
>
void fmha_reference(
    ProblemShape problem_shape,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE,
    Fusion fusion
) {
  using namespace cute;

  dim3 grid(size<0>(mO), size<2>(mO), 1);
  dim3 block(256);
  int shared_mem = size<0>(mK) * sizeof(typename TensorO::value_type);

  if (shared_mem >= (48 << 10)) {
    CUTLASS_TRACE_HOST("  Setting smem size to " << shared_mem);
    auto result = cudaFuncSetAttribute(
        fmha_reference_kernel<ProblemShape, TensorQ, TensorK, TensorV, TensorO, TensorLSE, Fusion>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem);
    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      CUTLASS_TRACE_HOST(
        "  cudaFuncSetAttribute() returned error: "
        << cudaGetErrorString(result));
      return;
    }
  }

  fmha_reference_kernel<<<grid, block, shared_mem>>>(problem_shape, mQ, mK, mV, mO, mLSE, fusion);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
