/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute
{

namespace SM120::BLOCKSCALED {

template <class MMAOp,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr void
mma_unpack(MMA_Traits<MMAOp>   const& traits,
           Tensor<TD, DLayout>      & D,
           Tensor<TA, ALayout> const& A_zipped,
           Tensor<TB, BLayout> const& B_zipped,
           Tensor<TC, CLayout> const& C)
{
  static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");

  // Register value types from the MMA_Operation register arrays
  using          RegTypeD = typename remove_extent<typename MMAOp::DRegisters>::type;
  using          RegTypeA = typename remove_extent<typename MMAOp::ARegisters>::type;
  using          RegTypeB = typename remove_extent<typename MMAOp::BRegisters>::type;
  using          RegTypeC = typename remove_extent<typename MMAOp::CRegisters>::type;
  using        RegTypeSFA = typename remove_extent<typename MMAOp::SFARegisters>::type;
  using        RegTypeSFB = typename remove_extent<typename MMAOp::SFBRegisters>::type;

  constexpr int   RegNumD = extent<typename MMAOp::DRegisters>::value;
  constexpr int   RegNumA = extent<typename MMAOp::ARegisters>::value;
  constexpr int   RegNumB = extent<typename MMAOp::BRegisters>::value;
  constexpr int   RegNumC = extent<typename MMAOp::CRegisters>::value;
  constexpr int RegNumSFA = extent<typename MMAOp::SFARegisters>::value;
  constexpr int RegNumSFB = extent<typename MMAOp::SFBRegisters>::value;

  auto  [A, SFA] = unzip_tensor(A_zipped);
  auto  [B, SFB] = unzip_tensor(B_zipped);
  
  using Shape_MNK = typename MMA_Traits<MMAOp>::Shape_MNK;
  constexpr int SFVecSize = MMA_Traits<MMAOp>::SFVecSize;
  
  // Assert logical size
  CUTE_STATIC_ASSERT_V(size(SFA) == size<2>(Shape_MNK{}));
  CUTE_STATIC_ASSERT_V(size(SFB) == size<2>(Shape_MNK{})); 

  // Assert physical size
  CUTE_STATIC_ASSERT(decltype(cosize(layout(SFA))){} == size<2>(Shape_MNK{}) / SFVecSize); 
  CUTE_STATIC_ASSERT(decltype(cosize(layout(SFB))){} == size<2>(Shape_MNK{}) / SFVecSize); 

  Tensor rA = recast<RegTypeA>(A);
  Tensor rB = recast<RegTypeB>(B);
  CUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA>{});
  CUTE_STATIC_ASSERT_V(size(rB) == Int<RegNumB>{});

  Tensor rD = recast<RegTypeD>(D);
  Tensor rC = recast<RegTypeC>(C);
  CUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumD>{});
  CUTE_STATIC_ASSERT_V(size(rC) == Int<RegNumC>{});

  Tensor rSFA = recast<RegTypeSFA>(filter_zeros(SFA));
  Tensor rSFB = recast<RegTypeSFB>(filter_zeros(SFB));

  CUTE_STATIC_ASSERT_V(size(rSFA) == Int<RegNumSFA>{});
  CUTE_STATIC_ASSERT_V(size(rSFB) == Int<RegNumSFB>{});

  detail::explode(MMAOp::fma,
            rD,   make_int_sequence<RegNumD>{},
            rA,   make_int_sequence<RegNumA>{},
            rB,   make_int_sequence<RegNumB>{},
            rC,   make_int_sequence<RegNumC>{},
            rSFA, make_int_sequence<RegNumSFA>{},
            rSFB, make_int_sequence<RegNumSFB>{});
}
} // namespace SM120::BLOCKSCALED

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA F8F6F4 16x8x32 TN
template <class a_type, class b_type, class c_type>
struct MMA_Traits<SM120_16x8x32_TN<a_type, b_type, c_type>>
     : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>
{
  // The MMA accepts 8-bit inputs regardless of the types for A and B
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;

  using ValTypeD = c_type;
  using ValTypeC = c_type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA MXF8F6F4 16x8x64 TN
template <class a_type, class b_type, class c_type, class sf_type, int VS>
struct MMA_Traits<SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<a_type, b_type, c_type, sf_type, VS>>
{
  // The MMA accepts 4-bit inputs regardless of the types for A and B
  using ValTypeA = uint4_t;
  using ValTypeB = uint4_t;

  using ValTypeD = c_type;
  using ValTypeC = c_type;

  using ValTypeSF = sf_type;
  constexpr static int SFVecSize = VS;

  using Shape_MNK = Shape<_16,_8,_64>;
  using ThrID     = Layout<_32>;

  // (T32,V32) -> (M16,K64)
  using ALayout   = Layout<Shape <Shape <  _4,_8>,Shape < _8,_2,  _2>>,
                           Stride<Stride<_128,_1>,Stride<_16,_8,_512>>>;
  // (T32,V16) -> (M16,K64)
  using BLayout   = Layout<Shape <Shape < _4,_8>,Shape <_8,  _2>>,
                           Stride<Stride<_64,_1>,Stride<_8,_256>>>;
  // (T32,V64) -> (M16,K64)
  using SFALayout = Layout<Shape <Shape <_2,_2,_8>,_64>,  // Effectively 16 threads due to the 2:0 mode
                           Stride<Stride<_8,_0,_1>,_16>>;
  // (T32,V64) -> (N8,K64)
  using SFBLayout = Layout<Shape <Shape <_4,_8>,_64>,     // Effectively 8 threads due to the 4:0 mode
                           Stride<Stride<_0,_1>, _8>>;
  // (T32,V4)  -> (M16,N8)
  using CLayout   = SM80_16x8_Row;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA MXF8F6F4 16x8x32 TN
template <class a_type, class b_type, class c_type, class sf_type, int VS>
struct MMA_Traits<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<a_type, b_type, c_type, sf_type, VS>>
{
  using UnderlyingTraits = MMA_Traits<SM120_16x8x32_TN<a_type, b_type, c_type>>;

  // The MMA accepts 8-bit inputs regardless of the types for A and B
  using ValTypeA = typename UnderlyingTraits::ValTypeA;
  using ValTypeB = typename UnderlyingTraits::ValTypeB;

  using ValTypeD = typename UnderlyingTraits::ValTypeD;
  using ValTypeC = typename UnderlyingTraits::ValTypeC;

  using Shape_MNK = typename UnderlyingTraits::Shape_MNK;
  using ThrID     = typename UnderlyingTraits::ThrID;

  using ALayout   = typename UnderlyingTraits::ALayout;
  using BLayout   = typename UnderlyingTraits::BLayout;
  using CLayout   = typename UnderlyingTraits::CLayout;

  // Scaling factor
  using ValTypeSF = sf_type;
  constexpr static int SFVecSize = VS;

  // (T32,V32) -> (M16,K32)
  using SFALayout = Layout<Shape <Shape <_2,_2,_8>,_32>,  // Effectively 16 threads due to the 2:0 mode
                           Stride<Stride<_8,_0,_1>,_16>>;
  // (T32,V32) -> (N8,K32)
  using SFBLayout = Layout<Shape <Shape <_4,_8>,_32>,     // Effectively 8 threads due to the 4:0 mode
                           Stride<Stride<_0,_1>, _8>>;
};

// Transform if needed
template<class MMA_Op, class Tensor>
CUTLASS_DEVICE void
fp4_shift_A(MMA_Op const& op, Tensor&& tensor) {
}
template<class MMA_Op, class Tensor>
CUTLASS_DEVICE void
fp4_shift_B(MMA_Op const& op, Tensor&& tensor) {
}

// For SM120 MMA F8F6F4 input fp4, the operand A/B are load from ld.matrix. 
// ld.matrix b4x16_p64 places FP4 data at the first four bits in each
// eight-bit container, whereas MMA F8F6F4 expects the four-bit data to be in 
// the middle of the eight-bit container. Thus, e2m1 operands being fed
// to MMA F8F6F4 must be shifted left by two bits.
// 0b0000ABCD --> 0b00ABCD00
// NOTE: Same transformation is NOT needed for FP6 and FP8.
template<class AType, class BType, class... MMAArgs, class Tensor>
CUTLASS_DEVICE void
fp4_shift_A(SM120_16x8x32_TN<AType, BType, MMAArgs ...> const&, Tensor&& tensor) {
  using RegisterTypeA = typename remove_extent<typename
                        SM120_16x8x32_TN<AType, BType, MMAArgs ...>::ARegisters>::type;
  if constexpr (cute::is_same_v<AType, cutlass::float_e2m1_t>) {
    cute::transform(recast<RegisterTypeA>(tensor), [](RegisterTypeA& v){ return v << 2; });
  }
}
template<class AType, class BType, class... MMAArgs, class Tensor>
CUTLASS_DEVICE void
fp4_shift_B(SM120_16x8x32_TN<AType, BType, MMAArgs ...> const&, Tensor&& tensor) {
  using RegisterTypeB = typename remove_extent<typename
                        SM120_16x8x32_TN<AType, BType, MMAArgs ...>::BRegisters>::type;
  if constexpr (cute::is_same_v<BType, cutlass::float_e2m1_t>) {
    cute::transform(recast<RegisterTypeB>(tensor), [](RegisterTypeB& v){ return v << 2; });
  }
}

namespace SM120::BLOCKSCALED {

// Template function with scale factor needs to enmuerate types one by one, as template 
// arguments contatins two variadic lists, which cannot be deduced in one shot.
template<class AType, class BType, class CType, class SFType, int VS, class Tensor>
CUTLASS_DEVICE void
fp4_shift_A(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<AType, BType, CType, SFType, VS> const&, Tensor&& tensor) {
  using RegisterTypeA = typename remove_extent<typename
                        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<AType, BType, CType, SFType, VS>::ARegisters>::type;
  if constexpr (cute::is_same_v<AType, cutlass::float_e2m1_t>) {
    cute::transform(recast<RegisterTypeA>(tensor), [](RegisterTypeA& v){ return v << 2; });
  }
}
template<class AType, class BType, class CType, class SFType, int VS, class Tensor>
CUTLASS_DEVICE void
fp4_shift_B(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<AType, BType, CType, SFType, VS> const&, Tensor&& tensor) {
  using RegisterTypeB = typename remove_extent<typename
                        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<AType, BType, CType, SFType, VS>::BRegisters>::type;
  if constexpr (cute::is_same_v<BType, cutlass::float_e2m1_t>) {
    cute::transform(recast<RegisterTypeB>(tensor), [](RegisterTypeB& v){ return v << 2; });
  }
}

}

} // end namespace cute
