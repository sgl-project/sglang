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
#include "cute/tensor.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template<typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void gemm_reset_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  constexpr int rA = decltype(rank(tA))::value;
  constexpr int rB = decltype(rank(tB))::value;
  constexpr int rC = decltype(rank(tC))::value;
  if constexpr (rA == 2 && rB == 2 && rC == 1) {
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<1>(tA); k_block++) {
      cute::gemm(atom, tA(_,k_block), tB(_,k_block), tC);
      atom.accumulate_ = GMMA::ScaleOut::One;
    }
  } else {
    static_assert(rA == 3 && rB == 3 && rC == 3);
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tA); k_block++) {
      cute::gemm(atom, tA(_,_,k_block), tB(_,_,k_block), tC);
      atom.accumulate_ = GMMA::ScaleOut::One;
    }
  }
}

template<typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void gemm_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  atom.accumulate_ = GMMA::ScaleOut::Zero;
  gemm_reset_zero_acc(atom, tA, tB, tC);
}

template<typename T, typename Fn>
CUTE_DEVICE constexpr typename T::value_type reduce(T const& t, Fn fn) {
  if constexpr (decltype(size(t) % _2{} == _0{})::value)  {
    auto partial = make_tensor<typename T::value_type>(size(t) / _2{});
    CUTE_UNROLL
    for (int i = 0; i < size(partial); i++) {
      partial(i) = fn(t(i), t(i + size(partial)));
    }
    return reduce(partial, fn);
  } else {
    auto result = t(_0{});
    CUTE_UNROLL
    for (int i = 1; i < size(t); i++) {
      result = fn(result, t(i));
    }
    return result;
  }
}

struct fmha_max {
  CUTE_DEVICE float operator()(float a, float b) { return ::max(a, b); }
};

template<typename Threshold, typename Source, typename Reference>
inline auto __device__ constexpr layout_separate(Threshold const& thr,
        Source const& src, Reference const& ref) {
    auto lt = filter(transform_layout(src, ref, [&](auto const& s, auto const& r) {
        if constexpr(decltype(r < thr)::value) {
            return s;
        } else {
            return make_layout(_1{}, _0{});
        }
    }));
    auto ge = filter(transform_layout(src, ref, [&](auto const& s, auto const& r) {
        if constexpr(decltype(r >= thr)::value) {
            return s;
        } else {
            return make_layout(_1{}, _0{});
        }
    }));
    return make_layout(lt, ge);
}

template<typename TiledMma, typename Acc>
inline auto __device__ constexpr layout_acc_mn(TiledMma const& tiled_mma, Acc const& acc) {
    auto separated = layout_separate(get<0>(typename TiledMma::Shape_MNK{}),
            get<0>(acc), stride<1>(typename TiledMma::LayoutC_TV{}));
    auto V_M = get<0>(separated);
    auto V_N = get<1>(separated);
    return make_layout(make_layout(V_M, get<1>(acc)), make_layout(V_N, get<2>(acc)));
}

template<typename TiledMma, typename Acc>
inline auto __device__ constexpr layout_op_mk_v(TiledMma const& tiled_mma, Acc const& acc) {
    return layout_separate(get<0>(typename TiledMma::Shape_MNK{}),
            get<0>(acc), stride<1>(typename TiledMma::LayoutA_TV{}));
}

template<typename TiledMma, typename Acc>
inline auto __device__ constexpr tensor_op_mk_v(TiledMma const& tiled_mma, Acc&& acc) {
    return make_tensor(acc.data(), layout_op_mk_v(tiled_mma, acc.layout()));
}

template<typename TiledMma>
inline auto __device__ constexpr reduction_target_n(TiledMma const& tiled_mma) {
    auto separated = layout_separate(get<0>(typename TiledMma::Shape_MNK{}),
            make_layout(shape<0>(typename TiledMma::LayoutC_TV{})),
            stride<0>(typename TiledMma::LayoutC_TV{}));
    return get<1>(separated);
}


template<template<cute::GMMA::Major, cute::GMMA::Major, cute::GMMA::ScaleIn, cute::GMMA::ScaleIn> class Primitive, cute::GMMA::Major tA, cute::GMMA::Major tB, cute::GMMA::ScaleIn sA, cute::GMMA::ScaleIn sB>
inline auto __device__ constexpr convert_to_gmma_rs(cute::MMA_Atom<Primitive<tA, tB, sA, sB>> const& tiled_mma) {
    using Atom = cute::MMA_Atom<Primitive<tA, tB, sA, sB>>;
    using ElementA = typename Atom::ValTypeA;
    using ElementB = typename Atom::ValTypeB;
    using ElementC = typename Atom::ValTypeC;
    using Shape_MNK = typename Atom::Shape_MNK;
    using RS = decltype(cute::GMMA::rs_op_selector<ElementA, ElementB, ElementC, Shape_MNK, tA, tB, sA, sB>());
    return cute::MMA_Atom<RS>{};
}

template<template<cute::GMMA::ScaleIn, cute::GMMA::ScaleIn> class Primitive, cute::GMMA::ScaleIn sA, cute::GMMA::ScaleIn sB>
inline auto __device__ constexpr convert_to_gmma_rs(cute::MMA_Atom<Primitive<sA, sB>> const& tiled_mma) {
    using Atom = cute::MMA_Atom<Primitive<sA, sB>>;
    using ElementA = typename Atom::ValTypeA;
    using ElementB = typename Atom::ValTypeB;
    using ElementC = typename Atom::ValTypeC;
    using Shape_MNK = typename Atom::Shape_MNK;
    constexpr auto tA = cute::GMMA::Major::K;
    constexpr auto tB = cute::GMMA::Major::K;
    using RS = decltype(cute::GMMA::rs_op_selector<ElementA, ElementB, ElementC, Shape_MNK, tA, tB, sA, sB>());
    return cute::MMA_Atom<RS>{};
}

template<class Atom, class... Args>
CUTE_DEVICE auto constexpr convert_to_gmma_rs(cute::TiledMMA<Atom, Args...> const& tiled_mma) {
    return cute::TiledMMA<decltype(convert_to_gmma_rs(Atom{})), Args...>{};
}

template<typename CLayout, typename AValueShape>
CUTE_DEVICE auto constexpr convert_c_layout_to_a_layout(CLayout const& c, AValueShape const& a) {
  return make_layout(
    make_shape(a, shape<1>(c), make_shape(shape<2>(c), size<0>(c) / size(a))),
    make_stride(stride<0>(c), stride<1>(c), make_stride(stride<2>(c), size<2>(a) * stride<0,2>(c))));
}

template<class Layout, class Stages = _1>
CUTE_DEVICE constexpr auto unstageSmemLayout(Layout const& layout, Stages stages = {}) {
    return composition(layout, make_tuple(_, _, make_layout(stages)));
}

template<class Element, class Accumulator, class OperandLayout_TV>
CUTE_DEVICE auto make_acc_into_op(Accumulator const& acc, OperandLayout_TV const& operand_layout_tv) {
  Tensor operand = make_fragment_like<Element>(convert_c_layout_to_a_layout(acc.layout(), shape<1>(operand_layout_tv)));
  Tensor operand_as_acc = make_tensor(operand.data(), acc.layout());

  cute::copy(acc, operand_as_acc);

  if constexpr (sizeof(Element) == 1) {
    
    // 00 11 22 33 00 11 22 33 acc layout
    // 00 00 11 11 22 22 33 33 operand layout
    // BB AA AA BB AA BB BB AA conflict-free exchange pattern
    //                         16-bit exchange; so process two at a time potentially
    int tid = threadIdx.x % 4;
    auto values_u32 = recast<uint32_t>(operand);

    CUTE_UNROLL
    for (int n = 0; n < size<1>(values_u32); n++) {
      CUTE_UNROLL
      for (int k = 0; k < size<2>(values_u32); k++) {
        CUTE_UNROLL
        for (int ii = 0; ii < 8; ii += 4) {

          uint32_t values_tmp_0 = values_u32(ii / 2 + 0, n, k);
          uint32_t values_tmp_1 = values_u32(ii / 2 + 1, n, k);

          // step A:
          // t 1 v 0 -> t 0 v 1
          // t 2 v 0 -> t 1 v 0
          // t 0 v 1 -> t 2 v 0
          // t 3 v 1 -> t 3 v 1

          int v_to_send = tid == 1 || tid == 2 ? 0 : 1;
          int v_to_recv = v_to_send;
          int t_to_recv_from = (0x3021 >> (tid * 4)) & 0xF;

          uint32_t values_tmp_a = v_to_send == 0 ? values_tmp_0 : values_tmp_1;

          values_tmp_a = __shfl_sync(0xFFFFFFFF, values_tmp_a, t_to_recv_from, 4);

          // step B:
          // t 0 v 0 -> t 0 v 0
          // t 3 v 0 -> t 1 v 1
          // t 1 v 1 -> t 2 v 1
          // t 2 v 1 -> t 3 v 0

          v_to_send = 1 - v_to_send;
          v_to_recv = 1 - v_to_recv;
          t_to_recv_from = (0x2130 >> (tid * 4)) & 0xF;

          uint32_t values_tmp_b = v_to_send == 0 ? values_tmp_0 : values_tmp_1;

          values_tmp_b = __shfl_sync(0xFFFFFFFF, values_tmp_b, t_to_recv_from, 4);

          values_u32(ii / 2 + 0, n, k) = __byte_perm(values_tmp_a, values_tmp_b, v_to_send == 0 ? 0x1054 : 0x5410);
          values_u32(ii / 2 + 1, n, k) = __byte_perm(values_tmp_a, values_tmp_b, v_to_send == 0 ? 0x3276 : 0x7632);
        }
      }
    }
  }

  return operand;
}

}  // namespace cutlass::fmha::collective
