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
#include "collective/fmha_fusion.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ProblemShapeIn,
  class TensorQ,
  class TensorK,
  class TensorV,
  class TensorO,
  class TensorLSE,
  class Mask
>
void __global__ fmha_reference_kernel(
    ProblemShapeIn problem_shape_in,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE,
    Mask mask) {

  using namespace cute;
  using namespace cutlass::fmha::collective;

  using Element = typename TensorO::value_type;
  using ElementAccumulator = typename TensorLSE::value_type;

  extern __shared__ char mS_mem[];
  ElementAccumulator* mS = reinterpret_cast<ElementAccumulator*>(mS_mem);

  ElementAccumulator softmax_scale = static_cast<ElementAccumulator>(1.0 / sqrt(1.0 * size<1>(mQ)));

  auto id = make_identity_tensor(make_shape(1, 1));

  for (int idx_L = blockIdx.y; idx_L < size<4>(problem_shape_in); idx_L += gridDim.y) {
    for (int idx_Q = blockIdx.x; idx_Q < size<0>(problem_shape_in); idx_Q += gridDim.x) {
      
      auto coord_L = idx2crd(idx_L, shape<4>(problem_shape_in));
      auto get_coord_in = [&]() {
        if constexpr (rank_v<decltype(get<2>(ProblemShapeIn{}))> == 2) {
          return cute::make_tuple(idx_Q, _0{}, cute::make_tuple(_0{}, _0{}), cute::make_tuple(_0{}, _0{}), coord_L);
        } else {
          return cute::make_tuple(idx_Q, _0{}, _0{}, _0{}, coord_L);
        }
      };
      auto coord_in = get_coord_in();
      auto [problem_shape, coord] = apply_variable_length(problem_shape_in, coord_in, get<4,1>(coord_in));

      int head_qk = 0;
      int head_v = 0;
      if constexpr (rank_v<decltype(get<2>(problem_shape))> == 2) {
        // MLA case: head_qk 192, head_v = 128
        head_qk = size<2, 0>(problem_shape) + size<2, 1>(problem_shape);
        head_v = size<2, 0>(problem_shape);
      } else {
        head_qk = size<3>(problem_shape);
        head_v = head_qk;
      }

      if (get<0,0>(coord) >= get<0>(problem_shape)) continue;

      int offset_Q = 0;
      if constexpr (rank<0>(decltype(coord){}) == 2) {
        offset_Q = get<0,1>(coord);
      }
  
      int offset_K = 0;
      if constexpr (rank<1>(decltype(coord){}) == 2) {
        offset_K = get<1,1>(coord);
      }

      if (get<1>(problem_shape) == 0) {
        for (int idx_D = threadIdx.x; idx_D < head_qk; idx_D += blockDim.x) {
          mO(idx_Q + offset_Q, idx_D, idx_L) = Element(0);
        }

        if (threadIdx.x == 0 && mLSE.data() != nullptr) {
          mLSE(idx_Q + offset_Q, idx_L) = -INFINITY;
        }
        continue;
      }
  
      for (int idx_K = threadIdx.x; idx_K < size<1>(problem_shape); idx_K += blockDim.x) {
        ElementAccumulator acc = 0;
        for (int idx_D = 0; idx_D < head_qk; idx_D++) {
          ElementAccumulator eQ = mQ(idx_Q + offset_Q, idx_D, idx_L);
          ElementAccumulator eK = mK(idx_K + offset_K, idx_D, idx_L);
          acc += eQ * eK;
        }
        auto frag = make_tensor<ElementAccumulator>(Shape<_1, _1>{});
        frag(0) = acc;
        mask.apply_mask(frag, make_tensor(id.data() + make_arithmetic_tuple(idx_Q, idx_K), id.layout()), problem_shape);
        mS[idx_K] = frag(0);
      }

      __syncthreads();

      ElementAccumulator maxS = -std::numeric_limits<ElementAccumulator>::infinity();
      for (int idx_K = 0; idx_K < size<1>(problem_shape); idx_K++) {
        maxS = std::max<ElementAccumulator>(maxS, mS[idx_K]);
      }
      if (maxS == -std::numeric_limits<ElementAccumulator>::infinity()) maxS = 0;

      __syncthreads();

      for (int idx_K = threadIdx.x; idx_K < size<1>(problem_shape); idx_K += blockDim.x) {
        mS[idx_K] = expf(softmax_scale * (mS[idx_K] - maxS));
      }

      __syncthreads();

      ElementAccumulator sum = 0;
      for (int idx_K = 0; idx_K < size<1>(problem_shape); idx_K++) {
        sum += mS[idx_K];
      }

      ElementAccumulator scale = 1.0f / sum;


      for (int idx_D = threadIdx.x; idx_D < head_v; idx_D += blockDim.x) {
        ElementAccumulator acc = 0;
        for (int idx_K = 0; idx_K < size<1>(problem_shape); idx_K++) {
          ElementAccumulator eV = mV(idx_K + offset_K, idx_D, idx_L);
          ElementAccumulator eK = static_cast<Element>(mS[idx_K]);
          acc += eK * eV;
        }
        mO(idx_Q + offset_Q, idx_D, idx_L) = static_cast<typename TensorO::value_type>(acc * scale);
      }


      if (threadIdx.x == 0 && mLSE.data() != nullptr) {
        mLSE(idx_Q + offset_Q, idx_L) = log(sum) + softmax_scale * maxS;
      }

    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ProblemShapeIn,
  class TensorQ,
  class TensorK,
  class TensorV,
  class TensorO,
  class TensorLSE,
  class Mask
>
void fmha_reference(
    ProblemShapeIn problem_shape_in,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE,
    Mask mask) {

  using namespace cute;

  dim3 grid(size<0>(mO), size<2>(mO), 1);
  dim3 block(256);
  int shared_mem = size<0>(mK) * int(sizeof(typename TensorLSE::value_type));
  fmha_reference_kernel<<<grid, block, shared_mem>>>(problem_shape_in, mQ, mK, mV, mO, mLSE, mask);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
