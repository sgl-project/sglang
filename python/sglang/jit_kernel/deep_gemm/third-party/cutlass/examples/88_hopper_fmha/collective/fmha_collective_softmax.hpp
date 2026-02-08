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

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "../collective/fmha_common.hpp"

namespace cutlass::fmha::collective {

template<
  class ElementAccumulator,
  class Fusion,
  class Params
>
struct CollectiveSoftmax {
  Params const& params;
  CUTLASS_DEVICE CollectiveSoftmax(Params const& params) : params(params) {}

  using SumType = float;
  using MaxType = ElementAccumulator;

  template<class AccPV, class TiledMmaPV>
  CUTLASS_DEVICE auto init(AccPV const& acc_pv, TiledMmaPV const& tiled_mma_pv) {
    Tensor s_max = make_fragment_like<MaxType>(size<0>(layout_acc_mn(tiled_mma_pv, acc_pv.layout())));
    Tensor a_sum = make_fragment_like<SumType>(s_max);
    return make_tuple(s_max, a_sum);
  }

  CUTLASS_DEVICE float overload_exp2(float f) {
    return ::exp2f(f);
  }

  CUTLASS_DEVICE cutlass::half_t overload_exp2(cutlass::half_t f) {
    auto a = f.raw();
    decltype(a) d;
    asm("ex2.approx.f16 %0, %1;" : "=h"(d) : "h"(a));
    return cutlass::half_t::bitcast(d);
  }


  CUTLASS_DEVICE float overload_max(float a, float b) {
    return ::max(a, b);
  }

  CUTLASS_DEVICE cutlass::half_t overload_max(cutlass::half_t a, cutlass::half_t b) {
    return cutlass::half_t{__hmax_nan(a.to_half(), b.to_half())};
  }

  CUTLASS_DEVICE half overload_to_native(cutlass::half_t f) {
    return f.to_half();
  }

  CUTLASS_DEVICE float overload_to_native(float f) {
    return f;
  }

  template<class AccQK, class TiledMmaQK, class CountQK, class State, class ProblemShape>
  CUTLASS_DEVICE auto step(AccQK& acc_qk, TiledMmaQK const& tiled_mma_qk, CountQK const& count_qk, State& state, ProblemShape const& problem_shape) {
    Fusion{}.before_softmax(acc_qk, count_qk, problem_shape);
    Tensor acc_qk_mn = make_tensor(acc_qk.data(), layout_acc_mn(tiled_mma_qk, acc_qk.layout()));
    auto reduction_target_qk = reduction_target_n(tiled_mma_qk);
    constexpr int red_rank = decltype(rank(reduction_target_qk))::value;

    auto& s_max = get<0>(state);
    auto& a_sum = get<1>(state);

    // Linear reduction is faster for the first iteration
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); i++) {
      s_max(i) = acc_qk_mn(i, 0);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int j = 1; j < size<1>(acc_qk_mn); j++) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<0>(acc_qk_mn); i++) {
        s_max(i) = overload_max(s_max(i), acc_qk_mn(i, j));
      }
    }

    for_each(make_seq<red_rank>{}, [&](auto r) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < shape<r>(reduction_target_qk); j *= 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(acc_qk_mn); i++) {
          s_max(i) = overload_max(s_max(i), MaxType{__shfl_xor_sync(uint32_t(-1), overload_to_native(s_max(i)), stride<r>(reduction_target_qk) * j)});
        }
      }
    });
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); i++) {
      MaxType local_max = s_max(i) == static_cast<MaxType>(-INFINITY) ? static_cast<MaxType>(0) : s_max(i);
      MaxType scale = static_cast<MaxType>(params.scale_softmax_log2);
      MaxType scale_max = scale * local_max;

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j++) {
        acc_qk_mn(i, j) = overload_exp2(scale * acc_qk_mn(i, j) - scale_max);
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); i++) {
      a_sum(i) = SumType{reduce(acc_qk_mn(i, _), cute::plus{})};
    }
  }

  template<bool kUseFusion=true, class AccQK, class TiledMmaQK, class CountQK, class State, class AccPV, class TiledMmaPV, class ProblemShape>
  CUTLASS_DEVICE auto step_interleave_begin(AccQK& acc_qk, TiledMmaQK const& tiled_mma_qk, CountQK const& count_qk, State& state, AccPV& acc_pv, TiledMmaPV const& tiled_mma_pv, ProblemShape const& problem_shape) {

    if constexpr (kUseFusion) {
      Fusion{}.before_softmax(acc_qk, count_qk, problem_shape);
    }

    Tensor acc_qk_mn = make_tensor(acc_qk.data(), layout_acc_mn(tiled_mma_qk, acc_qk.layout()));
    Tensor acc_pv_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma_pv, acc_pv.layout()));

    static_assert(size<0>(acc_qk_mn) == size<0>(acc_pv_mn));
    auto reduction_target_qk = reduction_target_n(tiled_mma_qk);
    constexpr int red_rank = decltype(rank(reduction_target_qk))::value;

    auto& s_max = get<0>(state);
    auto& a_sum = get<1>(state);

    Tensor s_max_prev = make_fragment_like(s_max);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); i++) {
      s_max_prev(i) = s_max(i);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); i++) {
      // Linear reduction is faster here, as well
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j++) {
        s_max(i) = overload_max(s_max(i), acc_qk_mn(i, j));
      }
    }
    // reduce max
    for_each(make_seq<red_rank>{}, [&](auto r) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < shape<r>(reduction_target_qk); j *= 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(acc_qk_mn); i++) {
          s_max(i) = overload_max(s_max(i), __shfl_xor_sync(uint32_t(-1), s_max(i), stride<r>(reduction_target_qk) * j));
        }
      }
    });
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_pv_mn); i++) {
      float s_max_cur = s_max(i) == -INFINITY ? 0.0f : s_max(i);
      float scale = ::exp2f((s_max_prev(i) - s_max_cur) * params.scale_softmax_log2);
      a_sum(i) *= scale;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_pv_mn); j++) {
        acc_pv_mn(i, j) *= scale;
      }
    }
  }

  template<class AccQK_MN, class State>
  CUTLASS_DEVICE auto step_interleave_step(AccQK_MN& acc_qk_mn, State& state) {

    auto& s_max = get<0>(state);
    auto& a_sum = get<1>(state);

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < size<0>(acc_qk_mn); j++) {
      float local_max = s_max(j) == -INFINITY ? 0.f : s_max(j);
      float scale_max = params.scale_softmax_log2 * local_max;

      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<1>(acc_qk_mn); k++) {
        acc_qk_mn(j, k) = ::exp2f(params.scale_softmax_log2 * acc_qk_mn(j, k) - scale_max);
        a_sum(j) += acc_qk_mn(j, k);
      }
    }
  }

  template<bool kUseFusion=true, class AccQK, class TiledMmaQK, class CountQK, class State, class AccPV, class TiledMmaPV, class ProblemShape>
  CUTLASS_DEVICE auto step(AccQK& acc_qk, TiledMmaQK const& tiled_mma_qk, CountQK const& count_qk, State& state, AccPV& acc_pv, TiledMmaPV const& tiled_mma_pv, ProblemShape const& problem_shape) {

    if constexpr (kUseFusion) {
      Fusion{}.before_softmax(acc_qk, count_qk, problem_shape);
    }

    Tensor acc_qk_mn = make_tensor(acc_qk.data(), layout_acc_mn(tiled_mma_qk, acc_qk.layout()));
    Tensor acc_pv_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma_pv, acc_pv.layout()));

    static_assert(size<0>(acc_qk_mn) == size<0>(acc_pv_mn));
    auto reduction_target_qk = reduction_target_n(tiled_mma_qk);
    constexpr int red_rank = decltype(rank(reduction_target_qk))::value;

    auto& s_max = get<0>(state);
    auto& a_sum = get<1>(state);

    Tensor s_max_prev = make_fragment_like(s_max);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); i++) {
      s_max_prev(i) = s_max(i);
      
      // Linear reduction is faster here, as well
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j++) {
        s_max(i) = overload_max(s_max(i), acc_qk_mn(i, j));
      }
      // reduce max
      for_each(make_seq<red_rank>{}, [&](auto r) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 1; j < shape<r>(reduction_target_qk); j *= 2) {
            s_max(i) = overload_max(s_max(i), MaxType{__shfl_xor_sync(uint32_t(-1), overload_to_native(s_max(i)), stride<r>(reduction_target_qk) * j)});
        }
      });

      MaxType local_max = s_max(i) == static_cast<MaxType>(-INFINITY) ? static_cast<MaxType>(0) : s_max(i);
      MaxType scale = static_cast<MaxType>(params.scale_softmax_log2);
      MaxType scale_max = scale * local_max;

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j++) {
        acc_qk_mn(i, j) = overload_exp2(scale * acc_qk_mn(i, j) - scale_max);
      }

      MaxType s_max_cur = s_max(i) == static_cast<MaxType>(-INFINITY) ? static_cast<MaxType>(0) : s_max(i);
      SumType scale_pv = overload_exp2((s_max_prev(i) - s_max_cur) * scale);
      a_sum(i) *= scale_pv;
      
      using ElementPV = typename AccPV::value_type;
      ElementPV scale_pv_ele = static_cast<ElementPV>(scale_pv);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_pv_mn); j++) {
        acc_pv_mn(i, j) *= scale_pv_ele;
      }
      a_sum(i) += SumType{reduce(acc_qk_mn(i, _), cute::plus{})};
    }
  }


  template<class State, class AccPV, class TiledMmaPV>
  CUTLASS_DEVICE auto tail(State& state, AccPV& acc_pv, TiledMmaPV const& tiled_mma_pv) {
    auto& s_max = get<0>(state);
    auto& a_sum = get<1>(state);

    Tensor acc_pv_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma_pv, acc_pv.layout()));

    auto reduction_target = reduction_target_n(tiled_mma_pv);
    constexpr int red_rank = decltype(rank(reduction_target))::value;
    for_each(make_seq<red_rank>{}, [&](auto r) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < shape<r>(reduction_target); j *= 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(acc_pv_mn); i++) {
          a_sum(i) = a_sum(i) + __shfl_xor_sync(uint32_t(-1), a_sum(i), stride<r>(reduction_target) * j);
        }
      }
    });

    Tensor acc_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma_pv, acc_pv.layout()));

    Tensor lse = make_fragment_like(a_sum);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_mn); i++) {
        float sum = a_sum(i);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : __frcp_rn(sum);
        lse(i) = (sum == 0.f || sum != sum) ? INFINITY : s_max(i) * params.scale_softmax + __logf(sum);
        float scale = params.rp_dropout * inv_sum;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<1>(acc_mn); j++) {
            acc_mn(i, j) *= scale;
        }
    }

    return lse;
  }
};

}  // namespace cutlass::fmha::collective
