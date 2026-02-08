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

#include <cute/pointer_flagged.hpp>            // cute::smem_ptr_flag
#include <cute/pointer_sparse.hpp>             // cute::smem_sparse_ptr_flag
#include <cute/swizzle.hpp>                    // cute::Swizzle
#include <cute/tensor_impl.hpp>                // cute::Tensor
#include <cute/arch/mma_sm90_desc.hpp>         // cute::LayoutType
#include <cute/arch/mma_sm90_gmma.hpp>         // cute::SM90_64x8x16_F16F16F16_SS, etc
#include <cute/atom/mma_traits.hpp>            // cute::MMA_Traits
#include <cute/layout_composed.hpp>            // cute::ComposedLayout
#include <cute/numeric/integral_constant.hpp>  // cute::is_static

namespace cute {

// Fence between the async destination accumulators of GMMA & source for their dependent use
template <class Engine, class Layout>
CUTE_HOST_DEVICE
void
warpgroup_fence_operand(Tensor<Engine, Layout>& frg) {
  CUTE_STATIC_ASSERT(is_static<Layout>::value);
  if constexpr (is_same_v<typename Engine::value_type, float>) {
    auto f32_frg = recast<float>(frg);
    CUTE_UNROLL
    for (int i = 0; i < size(f32_frg); ++i) {
      warpgroup_fence_operand(f32_frg(i));
    }
  }
  else {
    CUTE_STATIC_ASSERT(is_rmem<Engine>::value);
    auto u32_frg = recast<uint32_t>(frg);
    CUTE_UNROLL
    for (int i = 0; i < size(u32_frg); ++i) {
      warpgroup_fence_operand(u32_frg(i));
    }
  }
}

namespace SM90::GMMA {

///////////////////////////////////////////
// Common layouts for GMMA Shared Memory //
///////////////////////////////////////////

// M|N-major GMMA layouts in units of bits
using Layout_MN_INTER_Atom_Bits = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape< _128,_8>,Stride<_1, _128>>>;
using Layout_MN_SW32_Atom_Bits  = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape< _256,_8>,Stride<_1, _256>>>;
using Layout_MN_SW64_Atom_Bits  = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape< _512,_8>,Stride<_1, _512>>>;
using Layout_MN_SW128_Atom_Bits = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_1024,_8>,Stride<_1,_1024>>>;

// K-major GMMA layouts in units of bits
using Layout_K_INTER_Atom_Bits  = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape<_8, _128>,Stride< _128,_1>>>;
using Layout_K_SW32_Atom_Bits   = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8, _256>,Stride< _256,_1>>>;
using Layout_K_SW64_Atom_Bits   = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape<_8, _512>,Stride< _512,_1>>>;
using Layout_K_SW128_Atom_Bits  = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;

// M|N-major layouts in units of Type
template <class Type>
using Layout_MN_INTER_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_INTER_Atom_Bits{}));
template <class Type>
using Layout_MN_SW32_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW32_Atom_Bits{}));
template <class Type>
using Layout_MN_SW64_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW64_Atom_Bits{}));
template <class Type>
using Layout_MN_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW128_Atom_Bits{}));

// K-major layouts in units of Type
template <class Type>
using Layout_K_INTER_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_INTER_Atom_Bits{}));
template <class Type>
using Layout_K_SW32_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW32_Atom_Bits{}));
template <class Type>
using Layout_K_SW64_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW64_Atom_Bits{}));
template <class Type>
using Layout_K_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_Atom_Bits{}));

// With GMMA::Major param
template <class Type, Major tnsp>
using Layout_INTER_Atom = typename conditional<tnsp == Major::MN,
                                               Layout_MN_INTER_Atom<Type>,
                                               Layout_K_INTER_Atom<Type>>::type;
template <class Type, Major tnsp>
using Layout_SW32_Atom = typename conditional<tnsp == Major::MN,
                                              Layout_MN_SW32_Atom<Type>,
                                              Layout_K_SW32_Atom<Type>>::type;
template <class Type, Major tnsp>
using Layout_SW64_Atom = typename conditional<tnsp == Major::MN,
                                              Layout_MN_SW64_Atom<Type>,
                                              Layout_K_SW64_Atom<Type>>::type;
template <class Type, Major tnsp>
using Layout_SW128_Atom = typename conditional<tnsp == Major::MN,
                                               Layout_MN_SW128_Atom<Type>,
                                               Layout_K_SW128_Atom<Type>>::type;

//
// Tensor (position-dependent swizzle) to LayoutType utility
//

template <class Engine, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
LayoutType
layout_type(Tensor<Engine, Layout<Shape,Stride>> const&)
{
  static_assert(is_same<uint128_t, typename Engine::value_type>::value,
                "Expected uint128_t type in LayoutType conversion.");

  using Swizzle = get_swizzle_t<Engine>;
  constexpr int B = Swizzle::num_bits;
  constexpr int M = Swizzle::num_base;
  constexpr int S = Swizzle::num_shft;

  static_assert(M == 4,           "Unsupported layout swizzle");
  static_assert(0 <= B && B <= 3, "Unsupported layout swizzle");
  static_assert(S == 3,           "Unsupported layout swizzle");

  switch (B) {
    case 0: return LayoutType::INTERLEAVE;
    case 1: return LayoutType::B32;
    case 2: return LayoutType::B64;
    case 3: return LayoutType::B128;
  }
  return LayoutType::INTERLEAVE;  // ERROR
}

///////////////////////////////////////////////////////////////////////////////
// Construction method for GMMA Descriptors
///////////////////////////////////////////////////////////////////////////////

/**
* ///////////////////////////////
* // make_gmma_desc<Major::MN> //
* ///////////////////////////////
* Each GmmaDescriptor Major-MN describes a canonical layout of the form
*
* LayoutType::INTERLEAVE   : Swizzle<0,4,3> o smem_ptr o ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
* LayoutType::B32          : Swizzle<1,4,3> o smem_ptr o ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))
* LayoutType::B64          : Swizzle<2,4,3> o smem_ptr o ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))
* LayoutType::B128         : Swizzle<3,4,3> o smem_ptr o ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))
*
* where
*   T  : sizeof(uint128_t) / sizeof(value_type)
*   m  : integer in [1,16] corresponding to GMMA shape
*   k  : integer in [1,32] corresponding to GMMA shape
*   SBO: stride byte offset
*   LBO: leading byte offset
*
* See GMMA::Layout_MN_XXX_Atom<value_type> for building canonical GmmaDescriptor Major-MN layouts.
* For example,
*   auto smem_layout = tile_to_shape(Layout_MN_SW128_Atom<value_type>{}, Shape<_128,_64>{});
* is guaranteed to be accepted by make_gmma_desc<Major::MN> for appropriate value_type.
*
* //////////////////////////////
* // make_gmma_desc<Major::K> //
* //////////////////////////////
* Each GmmaDescriptor Major-K describes a canonical layout of the form
*
* LayoutType::INTERLEAVE : Swizzle<0,4,3> o smem_ptr o ((8,m),(T,2)):((1T,SBO),(1,LBO))
* LayoutType::B32        : Swizzle<1,4,3> o smem_ptr o ((8,m),(T,2)):((2T,SBO),(1, T ))
* LayoutType::B64        : Swizzle<2,4,3> o smem_ptr o ((8,m),(T,2)):((4T,SBO),(1, T ))
* LayoutType::B128       : Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2)):((8T,SBO),(1, T ))
*
* See GMMA::Layout_K_XXX_Atom<value_type> for building canonical GmmaDescriptor Major-K layouts.
* For example,
*   auto smem_layout = tile_to_shape(Layout_K_SW128_Atom<value_type>{}, Shape<_128,_64>{});
* is guaranteed to be accepted by make_gmma_desc<Major::K> for appropriate value_type.
*/
template <Major MajorMode, class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
GmmaDescriptor
make_gmma_desc(Tensor<TEngine,TLayout> const& tensor)
{
  static_assert(is_smem<TEngine>::value, "GMMA Descriptors can only be constructed on smem.");
  static_assert(TLayout::rank == 2, "GMMA Descriptors can only be constructed on rank-2 tensors.");
  using value_type = typename TEngine::value_type;

  Tensor u128_tensor = recast<uint128_t const>(tensor);

  // Result
  GmmaDescriptor desc;

  // Layout type
  constexpr LayoutType LAYOUT_TYPE = layout_type(u128_tensor);
  desc.bitfield.layout_type_ = uint8_t(LAYOUT_TYPE);

  // Start address (4LSB not included)
  uint32_t start_address = cast_smem_ptr_to_uint(raw_pointer_cast(u128_tensor.data()));
  desc.bitfield.start_address_ = static_cast<uint16_t>(start_address >> 4);

  constexpr uint8_t base_offset = 0;
  desc.bitfield.base_offset_ = base_offset;

  // LayoutType meta
  constexpr int W = LAYOUT_TYPE == LayoutType::INTERLEAVE ? 1 :
                    LAYOUT_TYPE == LayoutType::B32        ? 2 :
                    LAYOUT_TYPE == LayoutType::B64        ? 4 :
                    LAYOUT_TYPE == LayoutType::B128       ? 8 : -1;

  if constexpr (MajorMode == Major::MN)
  {
    /* In units of uint128_t, each GmmaDescriptor Major-MN describes a canonical layout of the form
     *
     * LayoutType::INTERLEAVE         : Swizzle<0,4,3> o smem_ptr o ((1,n),(8,k)):((X,SBO),(1,LBO))
     * LayoutType::B32                : Swizzle<1,4,3> o smem_ptr o ((2,n),(8,k)):((1,LBO),(2,SBO))
     * LayoutType::B64                : Swizzle<2,4,3> o smem_ptr o ((4,n),(8,k)):((1,LBO),(4,SBO))
     * LayoutType::B128               : Swizzle<3,4,3> o smem_ptr o ((8,n),(8,k)):((1,LBO),(8,SBO))
     */
    static_assert(size<1>(u128_tensor) == Int<(256 / cute::sizeof_bits<value_type>::value)>{} || // A and B in dense MMA
                  size<1>(u128_tensor) == Int<(128 / cute::sizeof_bits<value_type>::value)>{} || // A in sparse MMA
                  size<1>(u128_tensor) == Int<(512 / cute::sizeof_bits<value_type>::value)>{},   // B in sparse MMA
                         "Not a canonical GMMA_MN Layout: Expected K-size 256/sizeof_bits<T> for dense or (128|512)/sizeof_bits<T> for sparse.");

    // Construct the canonical GMMA T Layout with shape ((W,n),(8,2))
    Layout canonical_layout = logical_divide(layout(u128_tensor), Tile<Layout<Int<W>,_1>,Layout<Int<8>,_1>>{});

    // Check profile of canonical
    CUTE_STATIC_ASSERT_V(congruent(canonical_layout, Shape<Shape<_1,_1>,Shape<_1,_1>>{}), "Not a canonical GMMA_MN Layout: Expected profile failure.");
    // Check canonical mode strides
    constexpr uint32_t stride_00 = stride<0,0>(canonical_layout);
    constexpr uint32_t expected_stride_00 = LAYOUT_TYPE == LayoutType::INTERLEAVE ? stride<0,0>(canonical_layout) : 1;
    static_assert(stride_00 == expected_stride_00, "Not a canonical GMMA_MN Layout: Expected stride failure.");
    constexpr uint32_t stride_10 = stride<1,0>(canonical_layout);
    constexpr uint32_t expected_stride_10 = W;
    static_assert(stride_10 == expected_stride_10, "Not a canonical GMMA_MN Layout: Expected stride failure.");

    // stride dimension byte offset and leading dimension byte offset (4LSB not included == uint128_t units)
    constexpr uint32_t stride_01 = stride<0,1>(canonical_layout);
    constexpr uint32_t stride_11 = stride<1,1>(canonical_layout);

    desc.bitfield.stride_byte_offset_  = (LAYOUT_TYPE == LayoutType::INTERLEAVE) ? stride_01 : stride_11;
    desc.bitfield.leading_byte_offset_ = (LAYOUT_TYPE == LayoutType::INTERLEAVE) ? stride_11 : stride_01;
  }
  else if constexpr (MajorMode == Major::K)
  {
    /* In units of uint128_t, each GmmaDescriptor Major-K describes a canonical layout of the form
     *
     * LayoutType::INTERLEAVE    : Swizzle<0,4,3> o smem_ptr o ((8,n),2):((1,SBO),LBO)
     * LayoutType::B32           : Swizzle<1,4,3> o smem_ptr o ((8,n),2):((2,SBO),1)
     * LayoutType::B64           : Swizzle<2,4,3> o smem_ptr o ((8,n),2):((4,SBO),1)
     * LayoutType::B128          : Swizzle<3,4,3> o smem_ptr o ((8,n),2):((8,SBO),1)
     */
    CUTE_STATIC_ASSERT_V(size<0>(u128_tensor) % Int<8>{} == Int<0>{},          // N|M size
                         "Not a canonical GMMA_K Layout: Expected MN-size multiple of 8.");
    CUTE_STATIC_ASSERT_V(size<1>(u128_tensor) == Int<2>{} || size<1>(u128_tensor) == Int<4>{},      // K   size
                         "Not a canonical GMMA_K Layout: Expected K-size 2 for dense or 4 for sparse (in units of uint128_t).");

    // Construct the canonical GMMA N Layout with shape ((8,n),(2,1))
    Layout canonical_layout = logical_divide(layout(u128_tensor), Tile<Layout<_8,_1>,Layout<_2,_1>>{});

    // Check profile of canonical
    CUTE_STATIC_ASSERT_V(congruent(canonical_layout, Shape<Shape<_1,_1>,Shape<_1,_1>>{}), "Not a canonical GMMA_K Layout: Expected profile failure.");
    // Check canonical mode strides
    constexpr uint32_t stride_00 = stride<0,0>(canonical_layout);
    constexpr uint32_t expected_stride_00 = W;
    static_assert(stride_00 == expected_stride_00, "Not a canonical GMMA_K Layout: Expected stride failure.");
    constexpr uint32_t stride_10 = stride<1,0>(canonical_layout);
    constexpr uint32_t expected_stride_10 = (LAYOUT_TYPE == LayoutType::INTERLEAVE) ? stride<1,0>(canonical_layout) : 1;
    static_assert(stride_10 == expected_stride_10, "Not a canonical GMMA_K Layout: Expected stride failure.");

    // stride dimension byte offset and leading dimension byte offset (4LSB not included == uint128_t units)
    constexpr uint32_t stride_01 = stride<0,1>(canonical_layout);

    desc.bitfield.stride_byte_offset_  = stride_01;
    desc.bitfield.leading_byte_offset_ = stride_10;
  } else {
    static_assert(MajorMode != Major::MN && MajorMode != Major::K, "Unrecognized MajorMode!");
  }
  return desc;
}

///////////////////////////////////////////////////////////////////////////////
// Higher level GMMA Descriptor utilities
///////////////////////////////////////////////////////////////////////////////

struct DescriptorIterator
{
  using reference    = GmmaDescriptor;
  using element_type = GmmaDescriptor;
  using value_type   = GmmaDescriptor;

  GmmaDescriptor desc_;

  // Dereference returns the GmmaDescriptor
  CUTE_HOST_DEVICE constexpr
  reference operator*() const { return desc_; }

  // Advance and return a new GmmaDescriptor
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  // Return an advanced iterator
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  DescriptorIterator operator+(Index const& offset) const
  {
    // Use 32bit calculation rather than 64 bit calculation as we only update the part of desc
    GmmaDescriptor ret;
    ret.reg32_[0] = desc_.reg32_[0] + uint32_t(offset);
    ret.reg32_[1] = desc_.reg32_[1];
    return { ret };
  }
};

template <class T>
CUTE_HOST_DEVICE constexpr
GmmaDescriptor
raw_pointer_cast(DescriptorIterator const& ptr) {
  return ptr.desc_;
}

// Recast a DescriptorIterator Tensor to uint64_t, it's RegType in mma_unpack
template <class NewT>
CUTE_HOST_DEVICE constexpr
DescriptorIterator
recast_ptr(DescriptorIterator const& iter) {
  static_assert(is_same<NewT, uint64_t>::value, "Can only cast GmmaDescriptorIterator to uint64_t.");
  return iter;  // Do nothing, it will still dereference to GmmaDescriptor and decay to uint64_t
}

CUTE_HOST_DEVICE void
print(DescriptorIterator) {
  printf("GMMA::DescriptorIterator");
}

// The GMMA Traits below have custom fragment type flags for their smem desc tensors.
// These flags specialize a MakeTensor customization point to correctly make the fragment that is desired.
template <Major>
struct smem_desc : DescriptorIterator {};

} // end namespace SM90::GMMA

// Customization point for creating a GMMA::smem_desc Tensor
template <SM90::GMMA::Major MajorMode>
struct MakeTensor<SM90::GMMA::smem_desc<MajorMode>>
{
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    static_assert(is_smem<TEngine>::value, "Expected SMEM Tensor to construct a GMMA Desc Tensor");
    return make_tensor(SM90::GMMA::DescriptorIterator{SM90::GMMA::make_gmma_desc<MajorMode>(tensor<0>(smem_tensor))},
                       replace<0>(recast<uint128_t const>(smem_tensor).layout(), Layout<_1,_0>{}));
  }
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////////// MMA_TRAITS ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace SM90::GMMA {

//
// Specialized mma_unpack implementation for SM90 GMMA instructions
//

template <class MMA_Op, class... MMA_Args,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr
void
mma_unpack(MMA_Traits<MMA_Op, MMA_Args...> const& traits,
           Tensor<TD, DLayout>      & D,
           Tensor<TA, ALayout> const& A,
           Tensor<TB, BLayout> const& B,
           Tensor<TC, CLayout> const& C)
{
  static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");

  // Register value types from the MMA_Operation register arrays
  using RegTypeA = typename remove_extent<typename MMA_Op::ARegisters>::type;
  using RegTypeB = typename remove_extent<typename MMA_Op::BRegisters>::type;
  using RegTypeC = typename remove_extent<typename MMA_Op::CRegisters>::type;

  // SM90 GMMA take three arguments rather than four, try to assert C and D are aliased
  static_assert(is_same<typename TD::value_type, typename TC::value_type>::value, "GMMA C and D value_type must match.");
  static_assert(is_same<DLayout, CLayout>::value, "GMMA C and D layouts must match.");
  // assert((void*)&C == (void*)&D);

  Tensor rA = recast<RegTypeA>(A);
  Tensor rB = recast<RegTypeB>(B);
  Tensor rC = recast<RegTypeC>(D);  // NOTE: D and C are same, so use mutable D

  constexpr int RegNumA = extent<typename MMA_Op::ARegisters>::value;
  constexpr int RegNumB = extent<typename MMA_Op::BRegisters>::value;
  constexpr int RegNumC = extent<typename MMA_Op::CRegisters>::value;

  CUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA>{});
  CUTE_STATIC_ASSERT_V(size(rB) == Int<RegNumB>{});
  CUTE_STATIC_ASSERT_V(size(rC) == Int<RegNumC>{});

  detail::explode(MMA_Op::fma,
                  rA, make_int_sequence<RegNumA>{},
                  rB, make_int_sequence<RegNumB>{},
                  rC, make_int_sequence<RegNumC>{},
                  &(traits.accumulate_), seq<0>{});
}

// Accumulator layouts
template<int N>
using CLayout_64xN   = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,Int<N/8>>>,
                              Stride<Stride<_128,_1,_16>,Stride<_64,_8,   _512>>>;

using CLayout_64x8   = CLayout_64xN<  8>;
using CLayout_64x16  = CLayout_64xN< 16>;
using CLayout_64x32  = CLayout_64xN< 32>;
using CLayout_64x64  = CLayout_64xN< 64>;
using CLayout_64x96  = CLayout_64xN< 96>;
using CLayout_64x128 = CLayout_64xN<128>;
using CLayout_64x192 = CLayout_64xN<192>;
using CLayout_64x256 = CLayout_64xN<256>;

// Register source layout for 32-bit value types
using ALayout_64x8   = Layout<Shape <Shape <  _4,_8, _4>,Shape <    _2,  _2>>,
                              Stride<Stride< _64,_1,_16>,Stride<    _8,_256>>>;

// Register source layout for 16-bit (sparse 32-bit) value types
using ALayout_64x16  = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,  _2>>,
                              Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>;

// Register source layout for 8-bit (sparse 16-bit) value types
using ALayout_64x32  = Layout<Shape <Shape <  _4,_8, _4>,Shape < _4,_2,   _2>>,
                              Stride<Stride<_256,_1,_16>,Stride<_64,_8,_1024>>>;

// Register source layout for sparse 8-bit value types
using ALayout_64x64  = Layout<Shape <Shape <  _4,_8, _4>,Shape < _8,_2,   _2>>,
                              Stride<Stride<_512,_1,_16>,Stride<_64,_8,_2048>>>;

// Shared memory source layouts for any value type
template <int M, int K>
using ABLayout       = Layout<Shape <_128,Shape <Int<M>,Int<K>>>,
                              Stride<  _0,Stride<    _1,Int<M>>>>;

} // end namespace SM90::GMMA

using namespace SM90;

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x16_F16F16F16_SS = SM90::GMMA::MMA_64x8x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_8,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<  8, 16>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x16_F16F16F16_RS = SM90::GMMA::MMA_64x8x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_8,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<  8, 16>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x16_F16F16F16_SS = SM90::GMMA::MMA_64x16x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_16,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 16, 16>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x16_F16F16F16_RS = SM90::GMMA::MMA_64x16x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_16,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout< 16, 16>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x16_F16F16F16_SS = SM90::GMMA::MMA_64x32x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 32, 16>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x16_F16F16F16_RS = SM90::GMMA::MMA_64x32x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout< 32, 16>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x16_F16F16F16_SS = SM90::GMMA::MMA_64x64x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 64, 16>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x16_F16F16F16_RS = SM90::GMMA::MMA_64x64x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout< 64, 16>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x16_F16F16F16_SS = SM90::GMMA::MMA_64x96x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 96, 16>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x16_F16F16F16_RS = SM90::GMMA::MMA_64x96x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout< 96, 16>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x16_F16F16F16_SS = SM90::GMMA::MMA_64x128x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<128, 16>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x16_F16F16F16_RS = SM90::GMMA::MMA_64x128x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<128, 16>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x16_F16F16F16_SS = SM90::GMMA::MMA_64x192x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_192,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<192, 16>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x16_F16F16F16_RS = SM90::GMMA::MMA_64x192x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_192,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<192, 16>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x16_F16F16F16_SS = SM90::GMMA::MMA_64x256x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<256, 16>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x16_F16F16F16_RS = SM90::GMMA::MMA_64x256x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<256, 16>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x16_F32F16F16_SS = SM90::GMMA::MMA_64x8x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_8,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<  8, 16>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x16_F32F16F16_RS = SM90::GMMA::MMA_64x8x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_8,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<  8, 16>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x16_F32F16F16_SS = SM90::GMMA::MMA_64x16x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_16,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 16, 16>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x16_F32F16F16_RS = SM90::GMMA::MMA_64x16x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_16,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout< 16, 16>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x16_F32F16F16_SS = SM90::GMMA::MMA_64x32x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 32, 16>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x16_F32F16F16_RS = SM90::GMMA::MMA_64x32x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout< 32, 16>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x16_F32F16F16_SS = SM90::GMMA::MMA_64x64x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 64, 16>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x16_F32F16F16_RS = SM90::GMMA::MMA_64x64x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout< 64, 16>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x16_F32F16F16_SS = SM90::GMMA::MMA_64x96x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 96, 16>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x16_F32F16F16_RS = SM90::GMMA::MMA_64x96x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout< 96, 16>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x16_F32F16F16_SS = SM90::GMMA::MMA_64x128x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<128, 16>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x16_F32F16F16_RS = SM90::GMMA::MMA_64x128x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<128, 16>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x16_F32F16F16_SS = SM90::GMMA::MMA_64x192x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_192,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<192, 16>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x16_F32F16F16_RS = SM90::GMMA::MMA_64x192x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_192,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<192, 16>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x16_F32F16F16_SS = SM90::GMMA::MMA_64x256x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<256, 16>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x16_F32F16F16_RS = SM90::GMMA::MMA_64x256x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x16_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<256, 16>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x16_F32BF16BF16_SS = SM90::GMMA::MMA_64x8x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_8,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<  8, 16>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x16_F32BF16BF16_RS = SM90::GMMA::MMA_64x8x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_8,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<  8, 16>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x16_F32BF16BF16_SS = SM90::GMMA::MMA_64x16x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_16,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 16, 16>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x16_F32BF16BF16_RS = SM90::GMMA::MMA_64x16x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_16,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout< 16, 16>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x16_F32BF16BF16_SS = SM90::GMMA::MMA_64x32x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 32, 16>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x16_F32BF16BF16_RS = SM90::GMMA::MMA_64x32x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout< 32, 16>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x16_F32BF16BF16_SS = SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 64, 16>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x16_F32BF16BF16_RS = SM90::GMMA::MMA_64x64x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout< 64, 16>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x16_F32BF16BF16_SS = SM90::GMMA::MMA_64x96x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 96, 16>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x16_F32BF16BF16_RS = SM90::GMMA::MMA_64x96x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout< 96, 16>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x16_F32BF16BF16_SS = SM90::GMMA::MMA_64x128x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<128, 16>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x16_F32BF16BF16_RS = SM90::GMMA::MMA_64x128x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<128, 16>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x16_F32BF16BF16_SS = SM90::GMMA::MMA_64x192x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_192,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<192, 16>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x16_F32BF16BF16_RS = SM90::GMMA::MMA_64x192x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_192,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<192, 16>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x16_F32BF16BF16_SS = SM90::GMMA::MMA_64x256x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<256, 16>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x16_F32BF16BF16_RS = SM90::GMMA::MMA_64x256x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x16_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<256, 16>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x8_F32TF32TF32_SS_TN = SM90::GMMA::MMA_64x8x8_F32TF32TF32_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x8_F32TF32TF32_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64,  8>;
  using BLayout = GMMA::ABLayout<  8,  8>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x8_F32TF32TF32_RS_TN = SM90::GMMA::MMA_64x8x8_F32TF32TF32_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x8_F32TF32TF32_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x8;
  using BLayout = GMMA::ABLayout<  8,  8>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x8_F32TF32TF32_SS_TN = SM90::GMMA::MMA_64x16x8_F32TF32TF32_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x8_F32TF32TF32_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64,  8>;
  using BLayout = GMMA::ABLayout< 16,  8>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x8_F32TF32TF32_RS_TN = SM90::GMMA::MMA_64x16x8_F32TF32TF32_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x8_F32TF32TF32_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x8;
  using BLayout = GMMA::ABLayout< 16,  8>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x8_F32TF32TF32_SS_TN = SM90::GMMA::MMA_64x32x8_F32TF32TF32_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x8_F32TF32TF32_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64,  8>;
  using BLayout = GMMA::ABLayout< 32,  8>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x8_F32TF32TF32_RS_TN = SM90::GMMA::MMA_64x32x8_F32TF32TF32_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x8_F32TF32TF32_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x8;
  using BLayout = GMMA::ABLayout< 32,  8>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x8_F32TF32TF32_SS_TN = SM90::GMMA::MMA_64x64x8_F32TF32TF32_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x8_F32TF32TF32_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64,  8>;
  using BLayout = GMMA::ABLayout< 64,  8>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x8_F32TF32TF32_RS_TN = SM90::GMMA::MMA_64x64x8_F32TF32TF32_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x8_F32TF32TF32_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x8;
  using BLayout = GMMA::ABLayout< 64,  8>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x8_F32TF32TF32_SS_TN = SM90::GMMA::MMA_64x96x8_F32TF32TF32_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x8_F32TF32TF32_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64,  8>;
  using BLayout = GMMA::ABLayout< 96,  8>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x8_F32TF32TF32_RS_TN = SM90::GMMA::MMA_64x96x8_F32TF32TF32_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x8_F32TF32TF32_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x8;
  using BLayout = GMMA::ABLayout< 96,  8>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x8_F32TF32TF32_SS_TN = SM90::GMMA::MMA_64x128x8_F32TF32TF32_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x8_F32TF32TF32_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64,  8>;
  using BLayout = GMMA::ABLayout<128,  8>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x8_F32TF32TF32_RS_TN = SM90::GMMA::MMA_64x128x8_F32TF32TF32_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x8_F32TF32TF32_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x8;
  using BLayout = GMMA::ABLayout<128,  8>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x8_F32TF32TF32_SS_TN = SM90::GMMA::MMA_64x192x8_F32TF32TF32_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x8_F32TF32TF32_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64,  8>;
  using BLayout = GMMA::ABLayout<192,  8>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x8_F32TF32TF32_RS_TN = SM90::GMMA::MMA_64x192x8_F32TF32TF32_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x8_F32TF32TF32_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x8;
  using BLayout = GMMA::ABLayout<192,  8>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x8_F32TF32TF32_SS_TN = SM90::GMMA::MMA_64x256x8_F32TF32TF32_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x8_F32TF32TF32_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64,  8>;
  using BLayout = GMMA::ABLayout<256,  8>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x8_F32TF32TF32_RS_TN = SM90::GMMA::MMA_64x256x8_F32TF32TF32_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x8_F32TF32TF32_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_8>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x8;
  using BLayout = GMMA::ABLayout<256,  8>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32S8S8_SS_TN = SM90::GMMA::MMA_64x8x32_S32S8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x8x32_S32S8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32S8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x8x32_S32S8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x8x32_S32S8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32S8S8_SS_TN = SM90::GMMA::MMA_64x16x32_S32S8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x16x32_S32S8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32S8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x16x32_S32S8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x16x32_S32S8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32S8S8_SS_TN = SM90::GMMA::MMA_64x32x32_S32S8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x32x32_S32S8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32S8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x32x32_S32S8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x32x32_S32S8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32S8S8_SS_TN = SM90::GMMA::MMA_64x64x32_S32S8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x64x32_S32S8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32S8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x64x32_S32S8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x64x32_S32S8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32S8S8_SS_TN = SM90::GMMA::MMA_64x96x32_S32S8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x96x32_S32S8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32S8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x96x32_S32S8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x96x32_S32S8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32S8S8_SS_TN = SM90::GMMA::MMA_64x128x32_S32S8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x128x32_S32S8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32S8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x128x32_S32S8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x128x32_S32S8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32S8S8_SS_TN = SM90::GMMA::MMA_64x192x32_S32S8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x192x32_S32S8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32S8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x192x32_S32S8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x192x32_S32S8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32S8S8_SS_TN = SM90::GMMA::MMA_64x256x32_S32S8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x256x32_S32S8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32S8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x256x32_S32S8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x256x32_S32S8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32S8S8_RS_TN = SM90::GMMA::MMA_64x8x32_S32S8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x8x32_S32S8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32S8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x8x32_S32S8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x8x32_S32S8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32S8S8_RS_TN = SM90::GMMA::MMA_64x16x32_S32S8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x16x32_S32S8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32S8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x16x32_S32S8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x16x32_S32S8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32S8S8_RS_TN = SM90::GMMA::MMA_64x32x32_S32S8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x32x32_S32S8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32S8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x32x32_S32S8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x32x32_S32S8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32S8S8_RS_TN = SM90::GMMA::MMA_64x64x32_S32S8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x64x32_S32S8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32S8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x64x32_S32S8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x64x32_S32S8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32S8S8_RS_TN = SM90::GMMA::MMA_64x96x32_S32S8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x96x32_S32S8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32S8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x96x32_S32S8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x96x32_S32S8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32S8S8_RS_TN = SM90::GMMA::MMA_64x128x32_S32S8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x128x32_S32S8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32S8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x128x32_S32S8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x128x32_S32S8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32S8S8_RS_TN = SM90::GMMA::MMA_64x192x32_S32S8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x192x32_S32S8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32S8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x192x32_S32S8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x192x32_S32S8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32S8S8_RS_TN = SM90::GMMA::MMA_64x256x32_S32S8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x256x32_S32S8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32S8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x256x32_S32S8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x256x32_S32S8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32S8U8_SS_TN = SM90::GMMA::MMA_64x8x32_S32S8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x8x32_S32S8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32S8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x8x32_S32S8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x8x32_S32S8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32S8U8_SS_TN = SM90::GMMA::MMA_64x16x32_S32S8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x16x32_S32S8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32S8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x16x32_S32S8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x16x32_S32S8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32S8U8_SS_TN = SM90::GMMA::MMA_64x32x32_S32S8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x32x32_S32S8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32S8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x32x32_S32S8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x32x32_S32S8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32S8U8_SS_TN = SM90::GMMA::MMA_64x64x32_S32S8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x64x32_S32S8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32S8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x64x32_S32S8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x64x32_S32S8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32S8U8_SS_TN = SM90::GMMA::MMA_64x96x32_S32S8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x96x32_S32S8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32S8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x96x32_S32S8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x96x32_S32S8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32S8U8_SS_TN = SM90::GMMA::MMA_64x128x32_S32S8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x128x32_S32S8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32S8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x128x32_S32S8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x128x32_S32S8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32S8U8_SS_TN = SM90::GMMA::MMA_64x192x32_S32S8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x192x32_S32S8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32S8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x192x32_S32S8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x192x32_S32S8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32S8U8_SS_TN = SM90::GMMA::MMA_64x256x32_S32S8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x256x32_S32S8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32S8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x256x32_S32S8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x256x32_S32S8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32S8U8_RS_TN = SM90::GMMA::MMA_64x8x32_S32S8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x8x32_S32S8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32S8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x8x32_S32S8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x8x32_S32S8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32S8U8_RS_TN = SM90::GMMA::MMA_64x16x32_S32S8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x16x32_S32S8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32S8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x16x32_S32S8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x16x32_S32S8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32S8U8_RS_TN = SM90::GMMA::MMA_64x32x32_S32S8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x32x32_S32S8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32S8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x32x32_S32S8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x32x32_S32S8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32S8U8_RS_TN = SM90::GMMA::MMA_64x64x32_S32S8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x64x32_S32S8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32S8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x64x32_S32S8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x64x32_S32S8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32S8U8_RS_TN = SM90::GMMA::MMA_64x96x32_S32S8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x96x32_S32S8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32S8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x96x32_S32S8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x96x32_S32S8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32S8U8_RS_TN = SM90::GMMA::MMA_64x128x32_S32S8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x128x32_S32S8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32S8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x128x32_S32S8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x128x32_S32S8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32S8U8_RS_TN = SM90::GMMA::MMA_64x192x32_S32S8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x192x32_S32S8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32S8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x192x32_S32S8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x192x32_S32S8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32S8U8_RS_TN = SM90::GMMA::MMA_64x256x32_S32S8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x256x32_S32S8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32S8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x256x32_S32S8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x256x32_S32S8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32U8S8_SS_TN = SM90::GMMA::MMA_64x8x32_S32U8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x8x32_S32U8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32U8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x8x32_S32U8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x8x32_S32U8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32U8S8_SS_TN = SM90::GMMA::MMA_64x16x32_S32U8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x16x32_S32U8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32U8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x16x32_S32U8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x16x32_S32U8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32U8S8_SS_TN = SM90::GMMA::MMA_64x32x32_S32U8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x32x32_S32U8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32U8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x32x32_S32U8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x32x32_S32U8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32U8S8_SS_TN = SM90::GMMA::MMA_64x64x32_S32U8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x64x32_S32U8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32U8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x64x32_S32U8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x64x32_S32U8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32U8S8_SS_TN = SM90::GMMA::MMA_64x96x32_S32U8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x96x32_S32U8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32U8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x96x32_S32U8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x96x32_S32U8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32U8S8_SS_TN = SM90::GMMA::MMA_64x128x32_S32U8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x128x32_S32U8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32U8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x128x32_S32U8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x128x32_S32U8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32U8S8_SS_TN = SM90::GMMA::MMA_64x192x32_S32U8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x192x32_S32U8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32U8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x192x32_S32U8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x192x32_S32U8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32U8S8_SS_TN = SM90::GMMA::MMA_64x256x32_S32U8S8_SS_TN;

template <>
struct MMA_Traits<SM90_64x256x32_S32U8S8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32U8S8_SS_TN_SATURATE = SM90::GMMA::MMA_64x256x32_S32U8S8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x256x32_S32U8S8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32U8S8_RS_TN = SM90::GMMA::MMA_64x8x32_S32U8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x8x32_S32U8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32U8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x8x32_S32U8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x8x32_S32U8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32U8S8_RS_TN = SM90::GMMA::MMA_64x16x32_S32U8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x16x32_S32U8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32U8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x16x32_S32U8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x16x32_S32U8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32U8S8_RS_TN = SM90::GMMA::MMA_64x32x32_S32U8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x32x32_S32U8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32U8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x32x32_S32U8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x32x32_S32U8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32U8S8_RS_TN = SM90::GMMA::MMA_64x64x32_S32U8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x64x32_S32U8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32U8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x64x32_S32U8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x64x32_S32U8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32U8S8_RS_TN = SM90::GMMA::MMA_64x96x32_S32U8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x96x32_S32U8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32U8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x96x32_S32U8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x96x32_S32U8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32U8S8_RS_TN = SM90::GMMA::MMA_64x128x32_S32U8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x128x32_S32U8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32U8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x128x32_S32U8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x128x32_S32U8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32U8S8_RS_TN = SM90::GMMA::MMA_64x192x32_S32U8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x192x32_S32U8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32U8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x192x32_S32U8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x192x32_S32U8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32U8S8_RS_TN = SM90::GMMA::MMA_64x256x32_S32U8S8_RS_TN;

template <>
struct MMA_Traits<SM90_64x256x32_S32U8S8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32U8S8_RS_TN_SATURATE = SM90::GMMA::MMA_64x256x32_S32U8S8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x256x32_S32U8S8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32U8U8_SS_TN = SM90::GMMA::MMA_64x8x32_S32U8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x8x32_S32U8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32U8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x8x32_S32U8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x8x32_S32U8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32U8U8_SS_TN = SM90::GMMA::MMA_64x16x32_S32U8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x16x32_S32U8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32U8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x16x32_S32U8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x16x32_S32U8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32U8U8_SS_TN = SM90::GMMA::MMA_64x32x32_S32U8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x32x32_S32U8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32U8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x32x32_S32U8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x32x32_S32U8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32U8U8_SS_TN = SM90::GMMA::MMA_64x64x32_S32U8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x64x32_S32U8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32U8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x64x32_S32U8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x64x32_S32U8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32U8U8_SS_TN = SM90::GMMA::MMA_64x96x32_S32U8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x96x32_S32U8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32U8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x96x32_S32U8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x96x32_S32U8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32U8U8_SS_TN = SM90::GMMA::MMA_64x128x32_S32U8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x128x32_S32U8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32U8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x128x32_S32U8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x128x32_S32U8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32U8U8_SS_TN = SM90::GMMA::MMA_64x192x32_S32U8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x192x32_S32U8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32U8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x192x32_S32U8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x192x32_S32U8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32U8U8_SS_TN = SM90::GMMA::MMA_64x256x32_S32U8U8_SS_TN;

template <>
struct MMA_Traits<SM90_64x256x32_S32U8U8_SS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32U8U8_SS_TN_SATURATE = SM90::GMMA::MMA_64x256x32_S32U8U8_SS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x256x32_S32U8U8_SS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32U8U8_RS_TN = SM90::GMMA::MMA_64x8x32_S32U8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x8x32_S32U8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x8x32_S32U8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x8x32_S32U8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x8x32_S32U8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32U8U8_RS_TN = SM90::GMMA::MMA_64x16x32_S32U8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x16x32_S32U8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x16x32_S32U8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x16x32_S32U8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x16x32_S32U8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32U8U8_RS_TN = SM90::GMMA::MMA_64x32x32_S32U8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x32x32_S32U8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x32x32_S32U8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x32x32_S32U8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x32x32_S32U8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32U8U8_RS_TN = SM90::GMMA::MMA_64x64x32_S32U8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x64x32_S32U8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x64x32_S32U8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x64x32_S32U8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x64x32_S32U8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32U8U8_RS_TN = SM90::GMMA::MMA_64x96x32_S32U8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x96x32_S32U8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x96x32_S32U8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x96x32_S32U8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x96x32_S32U8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32U8U8_RS_TN = SM90::GMMA::MMA_64x128x32_S32U8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x128x32_S32U8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x128x32_S32U8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x128x32_S32U8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x128x32_S32U8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32U8U8_RS_TN = SM90::GMMA::MMA_64x192x32_S32U8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x192x32_S32U8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x192x32_S32U8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x192x32_S32U8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x192x32_S32U8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32U8U8_RS_TN = SM90::GMMA::MMA_64x256x32_S32U8U8_RS_TN;

template <>
struct MMA_Traits<SM90_64x256x32_S32U8U8_RS_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


using SM90_64x256x32_S32U8U8_RS_TN_SATURATE = SM90::GMMA::MMA_64x256x32_S32U8U8_RS_TN_SATURATE;

template <>
struct MMA_Traits<SM90_64x256x32_S32U8U8_RS_TN_SATURATE>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F16E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x8x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F16E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x8x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F32E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x8x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F32E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x8x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F16E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x16x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F16E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x16x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F32E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x16x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F32E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x16x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F16E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x32x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F16E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x32x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F32E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x32x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F32E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x32x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F16E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x64x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F16E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x64x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F32E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F32E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x64x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F16E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x96x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F16E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x96x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F32E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x96x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F32E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x96x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F16E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x128x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F16E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x128x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F32E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F32E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F16E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x192x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F16E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x192x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F32E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x192x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F32E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x192x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F16E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x256x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F16E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F16E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x256x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F16E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F32E4M3E4M3_SS_TN = SM90::GMMA::MMA_64x256x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F32E4M3E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F32E4M3E4M3_RS_TN = SM90::GMMA::MMA_64x256x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F32E4M3E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F16E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x8x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F16E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x8x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F32E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x8x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F32E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x8x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F16E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x16x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F16E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x16x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F32E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x16x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F32E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x16x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F16E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x32x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F16E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x32x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F32E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x32x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F32E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x32x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F16E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x64x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F16E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x64x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F32E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x64x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F32E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x64x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F16E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x96x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F16E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x96x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F32E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x96x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F32E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x96x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F16E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x128x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F16E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x128x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F32E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x128x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F32E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x128x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F16E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x192x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F16E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x192x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F32E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x192x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F32E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x192x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F16E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x256x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F16E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F16E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x256x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F16E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F32E4M3E5M2_SS_TN = SM90::GMMA::MMA_64x256x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F32E4M3E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F32E4M3E5M2_RS_TN = SM90::GMMA::MMA_64x256x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F32E4M3E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F16E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x8x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F16E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x8x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F32E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x8x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F32E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x8x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F16E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x16x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F16E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x16x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F32E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x16x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F32E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x16x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F16E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x32x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F16E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x32x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F32E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x32x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F32E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x32x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F16E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x64x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F16E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x64x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F32E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x64x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F32E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x64x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F16E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x96x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F16E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x96x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F32E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x96x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F32E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x96x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F16E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x128x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F16E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x128x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F32E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x128x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F32E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x128x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F16E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x192x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F16E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x192x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F32E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x192x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F32E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x192x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F16E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x256x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F16E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F16E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x256x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F16E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F32E5M2E4M3_SS_TN = SM90::GMMA::MMA_64x256x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F32E5M2E4M3_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F32E5M2E4M3_RS_TN = SM90::GMMA::MMA_64x256x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F32E5M2E4M3_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F16E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x8x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F16E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x8x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F32E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x8x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x8x32_F32E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x8x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x8x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_8,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<  8, 32>;
  using CLayout = GMMA::CLayout_64x8;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F16E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x16x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F16E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x16x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F32E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x16x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x16x32_F32E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x16x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x16x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_16,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 16, 32>;
  using CLayout = GMMA::CLayout_64x16;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F16E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x32x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F16E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x32x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F32E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x32x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x32x32_F32E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x32x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x32x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_32,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 32, 32>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F16E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x64x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F16E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x64x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F32E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x64x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x64x32_F32E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x64x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_64,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 64, 32>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F16E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x96x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F16E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x96x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F32E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x96x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x96x32_F32E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x96x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x96x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_96,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout< 96, 32>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F16E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x128x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F16E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x128x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F32E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x128x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x128x32_F32E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x128x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x128x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_128,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<128, 32>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F16E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x192x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F16E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x192x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F32E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x192x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x192x32_F32E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x192x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x192x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_192,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<192, 32>;
  using CLayout = GMMA::CLayout_64x192;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F16E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x256x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F16E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F16E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x256x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F16E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F32E5M2E5M2_SS_TN = SM90::GMMA::MMA_64x256x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F32E5M2E5M2_SS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x32_F32E5M2E5M2_RS_TN = SM90::GMMA::MMA_64x256x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x32_F32E5M2E5M2_RS_TN<scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_256,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using BLayout = GMMA::ABLayout<256, 32>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute

#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
#include "mma_traits_sm90_gmma_ext.hpp"
#endif
