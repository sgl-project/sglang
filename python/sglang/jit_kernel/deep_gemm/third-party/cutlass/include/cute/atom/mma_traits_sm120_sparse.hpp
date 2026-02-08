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
#include <cute/arch/mma_sm120_sparse.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute
{

namespace {

// (T32,V4) -> (M16,N8)
using SM120_16x8_Row = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                             Stride<Stride<_32,_1>,Stride<_16,_8>>>;

}

namespace SM120::BLOCKSCALED::SPARSE
{

// Unpack explode/mma call with sparse and block scalaring inputs.
template <class MMAOp,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr void
mma_unpack(MMA_Traits<MMAOp>  const&,
          Tensor<TD, DLayout>      & D,
          Tensor<TA, ALayout> const& A,
          Tensor<TB, BLayout> const& B,
          Tensor<TC, CLayout> const& C)
{
  static_assert(is_rmem_v<TD>, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem_v<TA>, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem_v<TB>, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem_v<TC>, "Expected registers in MMA_Atom::call");
  using         DRegisters = typename MMAOp::DRegisters;
  using         ARegisters = typename MMAOp::ARegisters;
  using         ERegisters = typename MMAOp::ERegisters;
  using         BRegisters = typename MMAOp::BRegisters;
  using         CRegisters = typename MMAOp::CRegisters;
  using         SFARegisters = typename MMAOp::SFARegisters;
  using         SFBRegisters = typename MMAOp::SFBRegisters;
  // Register value types from the MMAOp register arrays
  using         RegTypeD   = typename remove_extent<DRegisters>::type;
  using         RegTypeA   = typename remove_extent<ARegisters>::type;
  using         RegTypeE   = typename remove_extent<ERegisters>::type;
  using         RegTypeB   = typename remove_extent<BRegisters>::type;
  using         RegTypeC   = typename remove_extent<CRegisters>::type;
  using         RegTypeSFA = typename remove_extent<SFARegisters>::type;
  using         RegTypeSFB = typename remove_extent<SFBRegisters>::type;
  constexpr int RegNumD    = extent<DRegisters>::value;
  constexpr int RegNumA    = extent<ARegisters>::value;
  constexpr int RegNumE    = extent<ERegisters>::value;
  constexpr int RegNumB    = extent<BRegisters>::value;
  constexpr int RegNumC    = extent<CRegisters>::value;
  constexpr int RegNumSFA  = extent<SFARegisters>::value;
  constexpr int RegNumSFB  = extent<SFBRegisters>::value;

  auto  [tA, tSFA, tE] = unzip_tensor(A);
  auto  [tB, tSFB    ] = unzip_tensor(B);
  Tensor rA      = recast<RegTypeA>(tA);
  Tensor rE      = recast<RegTypeE>(tE);
  Tensor rB      = recast<RegTypeB>(tB);
  Tensor rD      = recast<RegTypeD>(D);
  Tensor rC      = recast<RegTypeC>(C);
  Tensor rSFA    = recast<RegTypeSFA>(tSFA);
  Tensor rSFB    = recast<RegTypeSFB>(tSFB);

  CUTE_STATIC_ASSERT_V(size(rA)   == Int<RegNumA>{});
  CUTE_STATIC_ASSERT_V(size(rE)   == Int<RegNumE>{});
  CUTE_STATIC_ASSERT_V(size(rB)   == Int<RegNumB>{});
  CUTE_STATIC_ASSERT_V(size(rD)   == Int<RegNumD>{});
  CUTE_STATIC_ASSERT_V(size(rC)   == Int<RegNumC>{});
  CUTE_STATIC_ASSERT_V(size(filter_zeros(rSFA)) == Int<RegNumSFA>{});
  CUTE_STATIC_ASSERT_V(size(filter_zeros(rSFB)) == Int<RegNumSFB>{});

  detail::explode(MMAOp::fma,
                  rD, make_int_sequence<RegNumD>{},
                  rA, make_int_sequence<RegNumA>{},
                  rB, make_int_sequence<RegNumB>{},
                  rC, make_int_sequence<RegNumC>{},
                  rE, make_int_sequence<RegNumE>{},
                  rSFA, make_int_sequence<RegNumSFA>{},
                  rSFB, make_int_sequence<RegNumSFB>{});
}

} // end namespace SM120::BLOCKSCALED::SPARSE


namespace SM120::SPARSE
{

template <class MMAOp,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr void
mma_unpack(MMA_Traits<MMAOp>  const&,
          Tensor<TD, DLayout>      & D,
          Tensor<TA, ALayout> const& A,
          Tensor<TB, BLayout> const& B,
          Tensor<TC, CLayout> const& C)
{
  static_assert(is_rmem_v<TD>, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem_v<TA>, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem_v<TB>, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem_v<TC>, "Expected registers in MMA_Atom::call");
  using         DRegisters = typename MMAOp::DRegisters;
  using         ARegisters = typename MMAOp::ARegisters;
  using         ERegisters = typename MMAOp::ERegisters;
  using         BRegisters = typename MMAOp::BRegisters;
  using         CRegisters = typename MMAOp::CRegisters;
  // Register value types from the MMAOp register arrays
  using         RegTypeD   = typename remove_extent<DRegisters>::type;
  using         RegTypeA   = typename remove_extent<ARegisters>::type;
  using         RegTypeE   = typename remove_extent<ERegisters>::type;
  using         RegTypeB   = typename remove_extent<BRegisters>::type;
  using         RegTypeC   = typename remove_extent<CRegisters>::type;
  constexpr int RegNumD    = extent<DRegisters>::value;
  constexpr int RegNumA    = extent<ARegisters>::value;
  constexpr int RegNumE    = extent<ERegisters>::value;
  constexpr int RegNumB    = extent<BRegisters>::value;
  constexpr int RegNumC    = extent<CRegisters>::value;

  auto  [tA, tE] = unzip_tensor(A);
  Tensor rA      = recast<RegTypeA>(tA);
  Tensor rE      = recast<RegTypeE>(tE);
  Tensor rB      = recast<RegTypeB>(B);
  Tensor rD      = recast<RegTypeD>(D);
  Tensor rC      = recast<RegTypeC>(C);
  CUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA>{});
  CUTE_STATIC_ASSERT_V(size(rE) == Int<RegNumE>{});
  CUTE_STATIC_ASSERT_V(size(rB) == Int<RegNumB>{});
  CUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumD>{});
  CUTE_STATIC_ASSERT_V(size(rC) == Int<RegNumC>{});

  detail::explode(MMAOp::fma,
                  rD, make_int_sequence<RegNumD>{},
                  rA, make_int_sequence<RegNumA>{},
                  rB, make_int_sequence<RegNumB>{},
                  rC, make_int_sequence<RegNumC>{},
                  rE, make_int_sequence<RegNumE>{});
}

} // end namespace SM120::SPARSE

// sparse F8F6F4 without block-scaling
template <class a_type, class b_type, class c_type>
struct MMA_Traits<SM120::SPARSE::SM120_SPARSE_16x8x64_TN<a_type, b_type, c_type>>
{
  using ValTypeA = sparse_elem<2, a_type>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using FrgTypeA = sparse_elem<2, uint8_t>;
  using FrgTypeE = sparse_elem<8, uint8_t>;

  using ValTypeC = c_type;
  using ValTypeD = c_type;

  using Shape_MNK = Shape<_16, _8, _64>;
  using ThrID     = Layout<_32>;
  // (T32,V32) -> (M16,K64)
  using ALayout   = Layout<Shape <Shape <  _4,_8>,Shape < _8,_2,  _2>>,
                           Stride<Stride<_128,_1>,Stride<_16,_8,_512>>>;
  // (T32,V16) -> (N8,K64)
  using BLayout   = Layout<Shape <Shape < _4,_8>,Shape <_4,  _4>>,
                           Stride<Stride<_32,_1>,Stride<_8,_128>>>;
  // (T32,V4)  -> (M16,N8)
  using CLayout   = SM120_16x8_Row;

  // (T32, V32) -> (M16, K64) 
  using ELayout   = Layout<Shape <Shape <_2,  _2,_8>, _32>,
                           Stride<Stride<_8,_512,_1>,_16>>;
};

// sparse MXF8F6F4 with block-scaling.
template <class a_type, class b_type, class c_type, class sf_type, int VS>
struct MMA_Traits<SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<a_type, b_type, c_type, sf_type, VS>>
      : MMA_Traits<SM120::SPARSE::SM120_SPARSE_16x8x64_TN<a_type, b_type, c_type>>
{
  using ValTypeA = sparse_elem<2, a_type>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using FrgTypeA = sparse_elem<2, uint8_t>;
  using FrgTypeE = sparse_elem<8, uint8_t>;

  using ValTypeD = c_type;
  using ValTypeC = c_type;

  using ValTypeSF = sf_type;
  constexpr static int SFVecSize = VS;

  using UnderlyingSFTraits = MMA_Traits<SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<a_type, b_type, c_type, sf_type, VS>>;
  using SFALayout = typename UnderlyingSFTraits::SFALayout;
  using SFBLayout = typename UnderlyingSFTraits::SFBLayout;
};

template <class a_type, class b_type, class c_type, class sf_type, int VS> 
struct MMA_Traits<SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x128_TN_VS<a_type, b_type, c_type, sf_type, VS>>
{
  using ValTypeA = sparse_elem<4,  uint8_t>;
  using ValTypeE = sparse_elem<16, uint8_t>;
  using ValTypeB = uint4_t;
  using FrgTypeA = sparse_elem<4,  uint8_t>;
  using FrgTypeE = sparse_elem<16, uint8_t>;

  using ValTypeC = c_type;
  using ValTypeD = c_type;

  using ValTypeSF = sf_type;

  constexpr static int SFVecSize = VS;

  using Shape_MNK = Shape<_16, _8, _128>;
  using ThrID     = Layout<_32>;
  // (T32,V64) -> (M16,K128)
  using ALayout   = Layout<Shape <Shape <  _4,_8>,Shape <_16,_2,   _2>>,
                           Stride<Stride<_256,_1>,Stride<_16,_8,_1024>>>;
  // (T32,V32) -> (N8,K128)
  using BLayout   = Layout<Shape <Shape < _4,_8>,Shape <_8,  _4>>,
                           Stride<Stride<_64,_1>,Stride<_8,_256>>>;
  // (T32,V128) -> (M16,K128)
  using SFALayout = Layout<Shape <Shape <_2,_2,_8>,_128>,
                           Stride<Stride<_8,_0,_1>, _16>>;
  // (T32,V128) -> (N8,K128)
  using SFBLayout = Layout<Shape <Shape <_4,_8>,_128>,
                           Stride<Stride<_0,_1>,  _8>>;
  // (T32,V4)  -> (M16,N8)
  using CLayout   = SM120_16x8_Row;
  // (T32, V64) -> (M16, K128) 
  using ELayout   = Layout<Shape <Shape <_2,   _2,_8>, Shape< _64>>,
                           Stride<Stride<_8,_1024,_1>,Stride<_16>>>;
};

namespace SM120::SPARSE {

// For SM120 MMA F8F6F4 input fp4, the operand A/B are load from ld.matrix. 
// ld.matrix b4x16_p64 places FP4 data at the first four bits in each
// eight-bit container, whereas MMA F8F6F4 expects the four-bit data to be in 
// the middle of the eight-bit container. Thus, e2m1 operands being fed
// to MMA F8F6F4 must be shifted left by two bits.
// 0b0000ABCD --> 0b00ABCD00
// NOTE: Same transformation is NOT needed for FP6 and FP8.
template<class AType, class BType, class... MMAArgs, class Tensor>
CUTLASS_DEVICE void
fp4_shift_A(SM120_SPARSE_16x8x64_TN<AType, BType, MMAArgs ...> const&, Tensor&& tensor) {
  using RegisterTypeA = typename remove_extent<typename
                        SM120_SPARSE_16x8x64_TN<AType, BType, MMAArgs ...>::ARegisters>::type;
  if constexpr (cute::is_same_v<AType, cutlass::float_e2m1_t>) {
    cute::transform(recast<RegisterTypeA>(tensor), [](RegisterTypeA& v){ return v << 2; });
  }
}
template<class AType, class BType, class... MMAArgs, class Tensor>
CUTLASS_DEVICE void
fp4_shift_B(SM120_SPARSE_16x8x64_TN<AType, BType, MMAArgs ...> const&, Tensor&& tensor) {
  using RegisterTypeB = typename remove_extent<typename
                        SM120_SPARSE_16x8x64_TN<AType, BType, MMAArgs ...>::BRegisters>::type;
  if constexpr (cute::is_same_v<BType, cutlass::float_e2m1_t>) {
    cute::transform(recast<RegisterTypeB>(tensor), [](RegisterTypeB& v){ return v << 2; });
  }
}

} // end namespace SM120::SPARSE

namespace SM120::BLOCKSCALED::SPARSE {

// Template function with scale factor needs to enmuerate types one by one, as template 
// arguments contatins two variadic lists, which cannot be deduced in one shot.
template<class AType, class BType, class CType, class SFType, int VS, class Tensor>
CUTLASS_DEVICE void
fp4_shift_A(SM120_SPARSE_16x8x64_TN_VS<AType, BType, CType, SFType, VS> const&, Tensor&& tensor) {
  using RegisterTypeA = typename remove_extent<typename
                        SM120_SPARSE_16x8x64_TN_VS<AType, BType, CType, SFType, VS>::ARegisters>::type;
  if constexpr (cute::is_same_v<AType, cutlass::float_e2m1_t>) {
    cute::transform(recast<RegisterTypeA>(tensor), [](RegisterTypeA& v){ return v << 2; });
  }
}
template<class AType, class BType, class CType, class SFType, int VS, class Tensor>
CUTLASS_DEVICE void
fp4_shift_B(SM120_SPARSE_16x8x64_TN_VS<AType, BType, CType, SFType, VS> const&, Tensor&& tensor) {
  using RegisterTypeB = typename remove_extent<typename
                        SM120_SPARSE_16x8x64_TN_VS<AType, BType, CType, SFType, VS>::BRegisters>::type;
  if constexpr (cute::is_same_v<BType, cutlass::float_e2m1_t>) {
    cute::transform(recast<RegisterTypeB>(tensor), [](RegisterTypeB& v){ return v << 2; });
  }
}

} // end namespace SM120::BLOCKSCALED::SPARSE

} // end namespace cute
