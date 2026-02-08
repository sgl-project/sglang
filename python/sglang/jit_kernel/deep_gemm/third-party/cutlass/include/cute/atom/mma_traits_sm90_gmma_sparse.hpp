/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cute/pointer_sparse.hpp>             // cute::smem_sparse_ptr_flag
#include <cute/swizzle.hpp>                    // cute::Swizzle
#include <cute/tensor_impl.hpp>                // cute::Tensor
#include <cute/arch/mma_sm90_desc.hpp>         // cute::LayoutType
#include <cute/arch/mma_sm90_gmma_sparse.hpp>  // cute::SM90::SPARSE::GMMA_64x8x32_F16F16F16_SS, etc
#include <cute/atom/mma_traits_sm90_gmma.hpp>  // cute::GMMA::Layout_*
#include <cute/atom/mma_traits.hpp>            // cute::MMA_Traits
#include <cute/layout_composed.hpp>            // cute::ComposedLayout
#include <cute/numeric/integral_constant.hpp>  // cute::is_static

namespace cute {

namespace SM90::GMMA {

///////////////////////////////////////////
// Common layouts for GMMA Shared Memory //
///////////////////////////////////////////

// M|N-major layouts in units of Type and sparsity factor S
template <class Type, int S>
using Layout_MN_INTER_SpAtom = ComposedLayout<Swizzle<0,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                              decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_MN_INTER_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_MN_SW32_SpAtom  = ComposedLayout<Swizzle<1,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                              decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_MN_SW32_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_MN_SW64_SpAtom  = ComposedLayout<Swizzle<2,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                              decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_MN_SW64_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_MN_SW128_SpAtom = ComposedLayout<Swizzle<3,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                              decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_MN_SW128_Atom<Type>{}.layout_b()))>;

// K-major layouts in units of Type and sparsity factor S
template <class Type, int S>
using Layout_K_INTER_SpAtom = ComposedLayout<Swizzle<0,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                              decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_K_INTER_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_K_SW32_SpAtom  = ComposedLayout<Swizzle<1,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                              decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_K_SW32_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_K_SW64_SpAtom  = ComposedLayout<Swizzle<2,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                              decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_K_SW64_Atom<Type>{}.layout_b()))>;
template <class Type, int S>
using Layout_K_SW128_SpAtom = ComposedLayout<Swizzle<3,4,3>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                              decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_K_SW128_Atom<Type>{}.layout_b()))>;

// With GMMA::Major param
template <class Type, int S, GMMA::Major tnsp>
using Layout_INTER_SpAtom = typename conditional<tnsp == GMMA::Major::MN,
                                                 Layout_MN_INTER_SpAtom<Type,S>,
                                                 Layout_K_INTER_SpAtom<Type,S>>::type;
template <class Type, int S, GMMA::Major tnsp>
using Layout_SW32_SpAtom = typename conditional<tnsp == GMMA::Major::MN,
                                                Layout_MN_SW32_SpAtom<Type,S>,
                                                Layout_K_SW32_SpAtom<Type,S>>::type;
template <class Type, int S, GMMA::Major tnsp>
using Layout_SW64_SpAtom = typename conditional<tnsp == GMMA::Major::MN,
                                                Layout_MN_SW64_SpAtom<Type,S>,
                                                Layout_K_SW64_SpAtom<Type,S>>::type;
template <class Type, int S, GMMA::Major tnsp>
using Layout_SW128_SpAtom = typename conditional<tnsp == GMMA::Major::MN,
                                                 Layout_MN_SW128_SpAtom<Type,S>,
                                                 Layout_K_SW128_SpAtom<Type,S>>::type;

///////////////////////////////////////////////////////////////////////////////
// Higher level GMMA Descriptor utilities
///////////////////////////////////////////////////////////////////////////////

template <GMMA::Major>
struct sparse_smem_desc : DescriptorIterator {};

} // end namespace SM90::GMMA

// Customization point for creating a cute::GMMAsparse_smem_desc Tensor
template <SM90::GMMA::Major MajorMode>
struct MakeTensor<SM90::GMMA::sparse_smem_desc<MajorMode>>
{
  // Note that this is the exact same as cute::GMMAsmem_desc above, plus additional static checks.
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    static_assert(is_smem<TEngine>::value, "Expected SMEM Tensor to construct a GMMA Desc Tensor");
    static_assert(is_sparse<typename TEngine::value_type>::value, "Expected sparse value_type.");
    static_assert(is_sparse_ptr<TEngine>::value, "Expected sparse iter.");
    return make_tensor(SM90::GMMA::DescriptorIterator{SM90::GMMA::make_gmma_desc<MajorMode>(tensor<0>(smem_tensor))},
                       replace<0>(recast<uint128_t const>(smem_tensor).layout(), Layout<_1,_0>{}));
  }
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////////// MMA_TRAITS ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace SM90::GMMA {

// Metadata layouts
using ELayout_64x64  = Layout<Shape <Shape <_2,   _2,_8, _4>, Shape <_32>>, 
                              Stride<Stride<_8,_2048,_1,_16>, Stride<_64>>>;

using ELayout_64x32  = Layout<Shape <Shape <   _2,_2,_8, _4>, Shape <_16,_2>>, 
                              Stride<Stride<_1024,_0,_1,_16>, Stride<_64,_8>>>;

using ELayout_64x16  = Layout<Shape <Shape <  _2,_2,_8, _4>, Shape < _8,_2>>, 
                              Stride<Stride<_512,_0,_1,_16>, Stride<_64,_8>>>;

} // namespace SM90::GMMA

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace SM90::GMMA::SPARSE {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <class MMAOp,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr void
mma_unpack(MMA_Traits<MMAOp>   const& traits,
           Tensor<TD, DLayout>      & D,
           Tensor<TA, ALayout> const& A_zipped,
           Tensor<TB, BLayout> const& B,
           Tensor<TC, CLayout> const& C)
{
  static_assert(is_rmem_v<TD>, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem_v<TA>, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem_v<TB>, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem_v<TC>, "Expected registers in MMA_Atom::call");

  using DRegisters = typename MMAOp::DRegisters;
  using ARegisters = typename MMAOp::ARegisters;
  using ERegisters = typename MMAOp::ERegisters;
  using BRegisters = typename MMAOp::BRegisters;
  using CRegisters = typename MMAOp::CRegisters;

  // Register value types from the MMAOp register arrays
  using RegTypeD   = typename remove_extent<DRegisters>::type;
  using RegTypeA   = typename remove_extent<ARegisters>::type;
  using RegTypeE   = typename remove_extent<ERegisters>::type;
  using RegTypeB   = typename remove_extent<BRegisters>::type;
  using RegTypeC   = typename remove_extent<CRegisters>::type;

  constexpr int RegNumA = extent<ARegisters>::value;
  constexpr int RegNumE = extent<ERegisters>::value;
  constexpr int RegNumB = extent<BRegisters>::value;
  constexpr int RegNumC = extent<CRegisters>::value;

  auto [A, E] = unzip_tensor(A_zipped);
  Tensor rA   = recast<RegTypeA>(A);
  Tensor rE   = recast<RegTypeE>(E);
  Tensor rB   = recast<RegTypeB>(B);

  CUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA>{});
  CUTE_STATIC_ASSERT_V(size(rE) == Int<RegNumE>{});
  CUTE_STATIC_ASSERT_V(size(rB) == Int<RegNumB>{});

  static_assert(is_same<RegTypeD, void>::value, "GMMA DRegisters must have void type.");
  static_assert(is_same<typename TD::value_type, typename TC::value_type>::value, "GMMA C and D value_type must match.");
  static_assert(is_same<DLayout, CLayout>::value, "GMMA C and D layouts must match.");

  Tensor rC = recast<RegTypeC>(D);  // NOTE: D and C are same, so use mutable D

  CUTE_STATIC_ASSERT_V(size(rC) == Int<RegNumC>{});

  detail::explode(MMAOp::fma,
                  rA, make_int_sequence<RegNumA>{},
                  rB, make_int_sequence<RegNumB>{},
                  rC, make_int_sequence<RegNumC>{},
                  rE, make_int_sequence<RegNumE>{},
                  &(traits.accumulate_), seq<0>{});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace SM90::SPARSE

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<  8, 16>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<  8, 16>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 16, 16>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 16, 16>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 32, 16>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 32, 16>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 64, 16>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 64, 16>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 96, 16>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 96, 16>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<128, 16>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<128, 16>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<192, 16>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<192, 16>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<256, 16>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<256, 16>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x8x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<  8, 64>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x16x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 16, 64>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x32x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 32, 64>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x64x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 64, 64>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x96x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 96, 64>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x128x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<128, 64>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x192x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<192, 64>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x256x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<256, 64>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute

#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
#include "mma_traits_sm90_gmma_sparse_ext.hpp"
#endif
