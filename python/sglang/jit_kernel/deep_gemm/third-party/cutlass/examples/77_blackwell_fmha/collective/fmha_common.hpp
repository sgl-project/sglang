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

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cute/tensor.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template<typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void gemm_reset_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  constexpr int rA = decltype(rank(tA))::value;
  constexpr int rB = decltype(rank(tB))::value;
  constexpr int rC = decltype(rank(tC))::value;
  static_assert(rA == 3 && rB == 3 && rC == 3);

  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tA); k_block++) {
    cute::gemm(atom, tA(_,_,k_block), tB(_,_,k_block), tC);
    atom.accumulate_ = decltype(atom.accumulate_)::One;
  }
}

template<typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void gemm_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  atom.accumulate_ = decltype(atom.accumulate_)::Zero;
  gemm_reset_zero_acc(atom, tA, tB, tC);
}

template<class Layout, class Stages = _1>
CUTE_DEVICE constexpr auto unstageSmemLayout(Layout const& layout, Stages stages = {}) {
    return composition(layout, prepend<decltype(rank(layout))::value>(make_layout(stages), _));
}

template<class T>
CUTE_DEVICE T warp_uniform(T a) {
  return __shfl_sync(0xffffffff, a, 0);
}

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr
auto
to_tiled_mma_sm100_ts(
    TiledMMA<MMA_Atom<
      MMA_Traits<SM100_MMA_F8F6F4_SS, a_type, b_type, c_type,
                    cute::C<M>, cute::C<N>,
                    cute::integral_constant<UMMA::Major, a_major>,
                    cute::integral_constant<UMMA::Major, b_major>,
                    cute::integral_constant<UMMA::ScaleIn, a_neg>,
                    cute::integral_constant<UMMA::ScaleIn, b_neg>>,
      TAs...>, TMs...>) {

  return TiledMMA<MMA_Atom<
    MMA_Traits<SM100_MMA_F8F6F4_TS<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, UMMA::Saturate::False>>,
    TAs...>, TMs...>{};
}

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr
auto
to_tiled_mma_sm100_ts(
    TiledMMA<MMA_Atom<
      SM100_MMA_F16BF16_SS<a_type, b_type, c_type,
                    M, N,
                    a_major,
                    b_major,
                    a_neg,
                    b_neg>,
      TAs...>, TMs...>) {
  return TiledMMA<MMA_Atom<
    SM100_MMA_F16BF16_TS<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, UMMA::Saturate::False>,
    TAs...>, TMs...>{};
}

template<uint32_t RegCount>
CUTLASS_DEVICE
void warpgroup_reg_set() {
  if constexpr (RegCount < 128) {
    cutlass::arch::warpgroup_reg_dealloc<RegCount>();
  }
  else {
    cutlass::arch::warpgroup_reg_alloc<RegCount>();
  }
}

}  // namespace cutlass::fmha::collective
