/* Copyright 2026 SGLang Team. All Rights Reserved.

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

#pragma once

#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>

// CUTLASS exposes the generic F8F6F4 instruction traits, but the pinned
// versions used by SGLang have changed their public spelling over time. Keep
// the tiny instruction atom local so the JIT kernel has one stable interface.
namespace cute {

template <
    class AType,
    class BType,
    class CType,
    int M,
    int N,
    UMMA::Major AMajor,
    UMMA::Major BMajor,
    UMMA::ScaleIn AScale = UMMA::ScaleIn::One,
    UMMA::ScaleIn BScale = UMMA::ScaleIn::One>
struct SM100_MMA_F8F6F4_WS_SS_SGL {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scale_c,
      uint64_t const& idesc_e) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 "
        "[%0], %1, %2, %3, p, 0;\n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idesc_e >> 32)), "r"(scale_c));
  }
};

template <
    class AType,
    class BType,
    class CType,
    int M,
    int N,
    UMMA::Major AMajor,
    UMMA::Major BMajor,
    UMMA::ScaleIn AScale,
    UMMA::ScaleIn BScale>
struct MMA_Traits<SM100_MMA_F8F6F4_WS_SS_SGL<AType, BType, CType, M, N, AMajor, BMajor, AScale, BScale>> {
  using ValTypeD = CType;
  using ValTypeA = AType;
  using ValTypeB = BType;
  using ValTypeC = CType;

  using FrgTypeA = UMMA::smem_desc<AMajor>;
  using FrgTypeB = UMMA::smem_desc<BMajor>;
  using FrgTypeC = UMMA::tmem_frg_ws_1sm<CType>;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<32>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<32>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<32>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<AType, BType, CType, M, N, AMajor, BMajor, AScale, BScale>();

  template <class TD, class DL, class TA, class AL, class TB, class BL, class TC, class CL>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(
      MMA_Traits const& traits,
      Tensor<TD, DL>& d,
      Tensor<TA, AL> const& a,
      Tensor<TB, BL> const& b,
      Tensor<TC, CL> const&) {
    uint64_t const desc_a = a[0];
    uint64_t const desc_b = b[0];
    uint32_t const tmem_c = raw_pointer_cast(d.data());
    uint64_t const idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
    SM100_MMA_F8F6F4_WS_SS_SGL<AType, BType, CType, M, N, AMajor, BMajor, AScale, BScale>::fma(
        desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

}  // namespace cute
