/***************************************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cute/config.hpp>
#include <cute/pointer_sparse.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>         // cute::TMEM::

#include <cute/atom/mma_traits.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>         // cute::GMMA::
#include <cute/atom/mma_traits_sm90_gmma_sparse.hpp>  // cute::GMMA::
#include <cute/atom/copy_traits_sm100.hpp>            // UTCCP smem desc

#include <cute/numeric/numeric_types.hpp>

// Check that aggregate initialization in .with() initializes all fields
#if defined(__GNUG__)
#pragma GCC diagnostic warning "-Wmissing-field-initializers"
#pragma GCC diagnostic error "-Wmissing-field-initializers"
#endif

namespace cute {

namespace UMMA {

//////////////////////////////////////////////////
// Common layouts for UMMA Shared Memory //
//////////////////////////////////////////////////

using cute::GMMA::Layout_MN_INTER_Atom;
using cute::GMMA::Layout_MN_SW32_Atom;
using cute::GMMA::Layout_MN_SW64_Atom;
using cute::GMMA::Layout_MN_SW128_Atom;
using cute::GMMA::Layout_K_INTER_Atom;
using cute::GMMA::Layout_K_SW32_Atom;
using cute::GMMA::Layout_K_SW64_Atom;
using cute::GMMA::Layout_K_SW128_Atom;

using Layout_MN_SW128_32B_Atom_Bits = ComposedLayout<Swizzle<2,5,2>, smem_ptr_flag, Layout<Shape< _1024,_4>,Stride<_1, _1024>>>;

template <class Type>
using Layout_MN_SW128_32B_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW128_32B_Atom_Bits{}));

//////////////////////////////////////////////////
// Common layouts for Sparse UMMA Shared Memory //
//////////////////////////////////////////////////

using cute::GMMA::Layout_MN_INTER_SpAtom;
using cute::GMMA::Layout_MN_SW32_SpAtom;
using cute::GMMA::Layout_MN_SW64_SpAtom;
using cute::GMMA::Layout_MN_SW128_SpAtom;
using cute::GMMA::Layout_K_INTER_SpAtom;
using cute::GMMA::Layout_K_SW32_SpAtom;
using cute::GMMA::Layout_K_SW64_SpAtom;
using cute::GMMA::Layout_K_SW128_SpAtom;

template <class Type, int S>
using Layout_MN_SW128_32B_SpAtom = ComposedLayout<Swizzle<2,5,2>, smem_sparse_ptr_flag_bits<S,sizeof_bits_v<Type>>,
                                                  decltype(blocked_product(Layout<Shape<_1,Int<S>>>{}, Layout_MN_SW128_32B_Atom<Type>{}.layout_b()))>;

// With UMMA::Major param
template <class Type, int S, UMMA::Major tnsp>
using Layout_INTER_SpAtom = typename conditional<tnsp == UMMA::Major::MN,
                                                 Layout_MN_INTER_SpAtom<Type,S>,
                                                 Layout_K_INTER_SpAtom<Type,S>>::type;
template <class Type, int S, UMMA::Major tnsp>
using Layout_SW32_SpAtom = typename conditional<tnsp == UMMA::Major::MN,
                                                Layout_MN_SW32_SpAtom<Type,S>,
                                                Layout_K_SW32_SpAtom<Type,S>>::type;
template <class Type, int S, UMMA::Major tnsp>
using Layout_SW64_SpAtom = typename conditional<tnsp == UMMA::Major::MN,
                                                Layout_MN_SW64_SpAtom<Type,S>,
                                                Layout_K_SW64_SpAtom<Type,S>>::type;
template <class Type, int S, UMMA::Major tnsp>
using Layout_SW128_SpAtom = typename conditional<tnsp == UMMA::Major::MN,
                                                 Layout_MN_SW128_SpAtom<Type,S>,
                                                 Layout_K_SW128_SpAtom<Type,S>>::type;

// Tile a MN-logical layout atom to an MMA Tile Shape ((MMA_M,MMA_N),M_MMAs,N_MMAs,...)
template <class LayoutAtom, class MMATileShape, class ModeOrder = GenColMajor>
CUTE_HOST_DEVICE constexpr
auto
tile_to_mma_shape(LayoutAtom const& atom, MMATileShape const& mma_tile_shape, ModeOrder const& order = {})
{
  constexpr int R = decltype(rank(mma_tile_shape))::value;
  auto mn_shape = cute::tuple_cat(zip(shape<0>(mma_tile_shape), take<1,3>(mma_tile_shape)), take<3,R>(mma_tile_shape));
  auto mn_tiled = tile_to_shape(atom, mn_shape, order);                      // (BLK_M,BLK_N,...)
  return tiled_divide(mn_tiled, product_each(shape<0>(mma_tile_shape)));     // ((MMA_M,MMA_N),M_MMAs,N_MMAs,...)
}

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

  if constexpr (M == 4) {
    static_assert(S == 3, "Expected S = 3 when M == 4. Unsupported layout swizzle.");
    switch (B) {
      default: static_assert(0 <= B && B <= 3, "Expected B = 0,1,2, or 3 when M == 4. Unsupported layout swizzle.");
      case 0:  return LayoutType::SWIZZLE_NONE;
      case 1:  return LayoutType::SWIZZLE_32B;
      case 2:  return LayoutType::SWIZZLE_64B;
      case 3:  return LayoutType::SWIZZLE_128B;
    }
  } else
  if constexpr (M == 5) {
    static_assert(B == 2, "Expected B = 2 when M == 5. Unsupported layout swizzle.");
    static_assert(S == 2, "Expected S = 2 when M == 5. Unsupported layout swizzle.");
    return LayoutType::SWIZZLE_128B_BASE32B;
  } else {
    static_assert(M==5,   "Only 16B and 32B Atoms are supported for UMMA. Unsupported layout swizzle.");
    return LayoutType::SWIZZLE_NONE;  // ERROR
  }
}

///////////////////////////////////////////////////////////////////////////////
// Construction method for UMMA Descriptors
///////////////////////////////////////////////////////////////////////////////

/**
* ///////////////////////////////
* // make_umma_desc<Major::MN> //
* ///////////////////////////////
* Each UmmaDescriptor Major-MN describes a canonical layout of the form
*
* LayoutType::INTERLEAVE   : Swizzle<0,4,3> o smem_ptr o ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
* LayoutType::B32          : Swizzle<1,4,3> o smem_ptr o ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))
* LayoutType::B64          : Swizzle<2,4,3> o smem_ptr o ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))
* LayoutType::B128         : Swizzle<3,4,3> o smem_ptr o ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))
* LayoutType::128B_BASE32B : Swizzle<2,5,2> o smem_ptr o ((T,8,m),(4,k)):((1,T,LBO),(?T,SBO))
*
* where
*   T  : sizeof(uint128_t) / sizeof(value_type)
*   m  : integer in [1,16] corresponding to UMMA shape
*   k  : integer in [1,32] corresponding to UMMA shape
*   SBO: stride byte offset
*   LBO: leading byte offset
*
* See UMMA::Layout_MN_XXX_Atom<value_type> for building canonical UmmaDescriptor Major-MN layouts.
* For example,
*   auto smem_layout = tile_to_shape(Layout_MN_SW128_Atom<value_type>{}, Shape<_128,_64>{});
* is guaranteed to be accepted by make_umma_desc<Major::MN> for appropriate value_type.
*
* //////////////////////////////
* // make_umma_desc<Major::K> //
* //////////////////////////////
* Each UmmaDescriptor Major-K describes a canonical layout of the form
*
* LayoutType::INTERLEAVE : Swizzle<0,4,3> o smem_ptr o ((8,m),(T,2)):((1T,SBO),(1,LBO))
* LayoutType::B32        : Swizzle<1,4,3> o smem_ptr o ((8,m),(T,2)):((2T,SBO),(1, T ))
* LayoutType::B64        : Swizzle<2,4,3> o smem_ptr o ((8,m),(T,2)):((4T,SBO),(1, T ))
* LayoutType::B128       : Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2)):((8T,SBO),(1, T ))
*
* See UMMA::Layout_K_XXX_Atom<value_type> for building canonical UmmaDescriptor Major-K layouts.
* For example,
*   auto smem_layout = tile_to_shape(Layout_K_SW128_Atom<value_type>{}, Shape<_128,_64>{});
* is guaranteed to be accepted by make_umma_desc<Major::K> for appropriate value_type.
*/
template <UMMA::Major MajorMode, class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
SmemDescriptor
make_umma_desc(Tensor<TEngine,TLayout> const& tensor)
{
  static_assert(is_smem<TEngine>::value, "UMMA Descriptors can only be constructed on smem.");
  static_assert(TLayout::rank == 2, "UMMA Descriptors can only be constructed on rank-2 tensors.");
  using value_type = typename TEngine::value_type;

  Tensor u128_tensor = recast<uint128_t const>(tensor);

  // Result
  SmemDescriptor desc;
  desc.version_ = 1;     // Set the version for blackwell
  desc.lbo_mode_ = 0; // set to legacy mode by default

  // Layout type
  constexpr UMMA::LayoutType LAYOUT_TYPE = UMMA::layout_type(u128_tensor);
  desc.layout_type_ = uint8_t(LAYOUT_TYPE);

  // Start address (4LSB not included)
  uint32_t start_address = cast_smem_ptr_to_uint(raw_pointer_cast(u128_tensor.data()));
  desc.start_address_ = static_cast<uint16_t>(start_address >> 4);

  constexpr uint8_t base_offset = 0;
  desc.base_offset_ = base_offset;

  // LayoutType meta
  constexpr int SwizzleAtomMNSize = LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_NONE         ? 1 :
                                    LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_32B          ? 2 :
                                    LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_64B          ? 4 :
                                    LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_128B         ? 8 :
                                    LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 8 : -1;

  if constexpr (MajorMode == UMMA::Major::MN)
  {
    /* In units of uint128_t, each UmmaDescriptor Major-MN describes a canonical layout of the form
     *
     * LayoutType::INTERLEAVE         : Swizzle<0,4,3> o smem_ptr o ((1,n),(8,k)):((X,SBO),(1,LBO))
     * LayoutType::B32                : Swizzle<1,4,3> o smem_ptr o ((2,n),(8,k)):((1,LBO),(2,SBO))
     * LayoutType::B64                : Swizzle<2,4,3> o smem_ptr o ((4,n),(8,k)):((1,LBO),(4,SBO))
     * LayoutType::B128               : Swizzle<3,4,3> o smem_ptr o ((8,n),(8,k)):((1,LBO),(8,SBO))
     * LayoutType::B128_BASE32B       : Swizzle<2,5,2> o smem_ptr o ((8,n),(4,k)):((1,LBO),(4,SBO))
     */

    constexpr int SwizzleAtomKSize = LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 4 : 8;

    // Construct the canonical UMMA T Layout with shape ((SwizzleAtomMNSize,n),(SwizzleAtomKSize,2))
    Layout canonical_layout = logical_divide(layout(u128_tensor), Tile<Layout<Int<SwizzleAtomMNSize>>,Layout<Int<SwizzleAtomKSize>>>{});

    // Check profile of canonical
    CUTE_STATIC_ASSERT_V(congruent(canonical_layout, Shape<Shape<_1,_1>,Shape<_1,_1>>{}), "Not a canonical UMMA_MN Layout: Expected profile failure.");
    // Check canonical mode strides
    constexpr uint32_t stride_00 = stride<0,0>(canonical_layout);
    constexpr uint32_t expected_stride_00 = LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_NONE ? stride<0,0>(canonical_layout) : 1;
    static_assert(stride_00 == expected_stride_00, "Not a canonical UMMA_MN Layout: Expected stride failure.");
    constexpr uint32_t stride_10 = stride<1,0>(canonical_layout);
    constexpr uint32_t expected_stride_10 = SwizzleAtomMNSize;
    static_assert(stride_10 == expected_stride_10, "Not a canonical UMMA_MN Layout: Expected stride failure.");

    // stride dimension byte offset and leading dimension byte offset (4LSB not included == uint128_t units)
    constexpr uint32_t stride_01 = stride<0,1>(canonical_layout);
    constexpr uint32_t stride_11 = stride<1,1>(canonical_layout);

    desc.stride_byte_offset_  = (LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_NONE) ? stride_01 : stride_11;
    desc.leading_byte_offset_ = (LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_NONE) ? stride_11 : stride_01;
  } else
  if constexpr (MajorMode == UMMA::Major::K)
  {
    /* In units of uint128_t, each UmmaDescriptor Major-K describes a canonical layout of the form
     *
     * LayoutType::INTERLEAVE    : Swizzle<0,4,3> o smem_ptr o ((8,n),2):((1,SBO),LBO)
     * LayoutType::B32           : Swizzle<1,4,3> o smem_ptr o ((8,n),2):((2,SBO),1)
     * LayoutType::B64           : Swizzle<2,4,3> o smem_ptr o ((8,n),2):((4,SBO),1)
     * LayoutType::B128          : Swizzle<3,4,3> o smem_ptr o ((8,n),2):((8,SBO),1)
     * LayoutType::B128_BASE32B  : Not applicable for Major-K
     */

    static_assert(LAYOUT_TYPE != UMMA::LayoutType::SWIZZLE_128B_BASE32B, "SWIZZLE_128B_BASE32B is invalid for Major-K");
    CUTE_STATIC_ASSERT_V(size<0>(u128_tensor) % Int<8>{} == Int<0>{},          // N|M size
                         "Not a canonical UMMA_K Layout: Expected MN-size multiple of 8.");

    // Construct the canonical UMMA N Layout with shape ((8,n),(2,1))
    Layout canonical_layout = logical_divide(layout(u128_tensor), Tile<Layout<_8,_1>,Layout<_2,_1>>{});

    // Check profile of canonical
    CUTE_STATIC_ASSERT_V(congruent(canonical_layout, Shape<Shape<_1,_1>,Shape<_1,_1>>{}), "Not a canonical UMMA_K Layout: Expected profile failure.");
    // Check canonical mode strides
    constexpr uint32_t stride_00 = stride<0,0>(canonical_layout);
    constexpr uint32_t expected_stride_00 = SwizzleAtomMNSize;
    static_assert(stride_00 == expected_stride_00, "Not a canonical UMMA_K Layout: Expected stride failure.");
    constexpr uint32_t stride_10 = stride<1,0>(canonical_layout);
    constexpr uint32_t expected_stride_10 = (LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_NONE) ? stride<1,0>(canonical_layout) : 1;
    static_assert(stride_10 == expected_stride_10, "Not a canonical UMMA_K Layout: Expected stride failure.");

    // stride dimension byte offset and leading dimension byte offset (4LSB not included == uint128_t units)
    constexpr uint32_t stride_01 = stride<0,1>(canonical_layout);

    desc.stride_byte_offset_  = stride_01;
    desc.leading_byte_offset_ = stride_10;
  } else {
    static_assert(MajorMode != UMMA::Major::MN && MajorMode != UMMA::Major::K, "Unrecognized MajorMode!");
  }
  return desc;
}

///////////////////////////////////////////////////////////////////////////////
// Higher level UMMA Descriptor utilities
///////////////////////////////////////////////////////////////////////////////

struct DescriptorIterator
{
  using reference    = SmemDescriptor;
  using element_type = SmemDescriptor;
  using value_type   = SmemDescriptor;

  SmemDescriptor desc_;

  // Dereference returns the UmmaDescriptor
  CUTE_HOST_DEVICE constexpr
  reference operator*() const { return desc_; }

  // Advance and return a new UmmaDescriptor
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  // Return an advanced iterator
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  DescriptorIterator operator+(Index const& offset) const
  {
    // Use 32bit calculation rather than 64 bit calculation as we only update the part of desc
    SmemDescriptor ret;
    ret.lo = desc_.lo + uint32_t(offset);
    ret.hi = desc_.hi;
    return { ret };
  }
};

template <class T>
CUTE_HOST_DEVICE constexpr
SmemDescriptor
raw_pointer_cast(DescriptorIterator const& ptr) {
  return ptr.desc_;
}

CUTE_HOST_DEVICE void
print(DescriptorIterator const&) {
  printf("UMMA::DescriptorIterator");
}

// Flag for smem descriptor allocation/creation
template <UMMA::Major>
struct smem_desc : DescriptorIterator {};

template <UMMA::Major>
struct sparse_smem_desc : DescriptorIterator {};

} // end namespace UMMA

// Customization point for creating a UMMA::smem_desc Tensor
template <UMMA::Major MajorMode>
struct MakeTensor<UMMA::smem_desc<MajorMode>>
{
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    static_assert(is_smem<TEngine>::value, "Expected SMEM Tensor to construct a UMMA Desc Tensor");
    return make_tensor(UMMA::DescriptorIterator{UMMA::make_umma_desc<MajorMode>(tensor<0>(smem_tensor))},
                       replace<0>(recast<uint128_t const>(smem_tensor).layout(), Layout<_1,_0>{}));
  }
};

// Customization point for creating a UMMA::sparse_smem_desc Tensor
template <UMMA::Major MajorMode>
struct MakeTensor<UMMA::sparse_smem_desc<MajorMode>>
{
  // Note that this is the exact same as UMMA::smem_desc above.
  // Only the interface validates that we are passed a sparse_ptr, which is recast away to construct
  //   the smem desc tensor
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    static_assert(is_smem<TEngine>::value, "Expected SMEM Tensor to construct a UMMA Desc Tensor");
    static_assert(is_sparse<typename TEngine::value_type>::value, "Expected sparse value_type.");
    static_assert(is_sparse_ptr<TEngine>::value, "Expected sparse iter.");
    return make_tensor(UMMA::DescriptorIterator{UMMA::make_umma_desc<MajorMode>(tensor<0>(smem_tensor))},
                       replace<0>(recast<uint128_t const>(smem_tensor).layout(), Layout<_1,_0>{}));
  }
};

// Special smem_desc_iter tensor entry for UTCCP copy.
template <class UtccpOp, class TEngine, class TLayout>
constexpr auto get_utccp_smem_desc_tensor(Tensor<TEngine, TLayout> const& smem_utccp_partitioned_tensor) {
  using VecLayout = decltype(layout<0>(TLayout{}));
  static_assert(VecLayout::rank == 2 && shape<1>(VecLayout{}) == 1, "Mismatched vec_mode tensor.");
  static_assert(is_smem<TEngine>::value, "Expect vec_mode smem_tesnor.");
  static_assert(is_static<VecLayout>::value, "Utccp copy tensor's vec_mode should be static.");

  using value_type = typename TEngine::value_type;
  using UtccpTaits = Copy_Traits<UtccpOp>;

  // UtccpTaits::ValID: logical_bit_idx -> tmem_offset.
  // We arrange the logical_bit_idx in order of (core_matrix_strided, core_matrix_leading, repeat(only in 64dplw01), broadcast).
  // So we only need the first two modes for src smem_tensor.
  auto utccp_core_matrix_shape = take<0,2>(upcast<sizeof_bits_v<value_type>>(typename UtccpTaits::ValID{}).shape());
  // logical_bit_idx -> smem_addr
  Layout vec_v_layout = flatten(layout<0>(VecLayout{}));
  Layout utccp_core_matrix_layout = vec_v_layout.with_shape(utccp_core_matrix_shape);
  Tensor utccp_core_matrix_tensor = group_modes<0,2>(make_tensor(smem_utccp_partitioned_tensor.data(), utccp_core_matrix_layout));
  Tensor core_matrix_desc_tensor = make_tensor<UMMA::smem_desc<UMMA::Major::K>>(utccp_core_matrix_tensor);
  return make_tensor(core_matrix_desc_tensor.data(), recast_layout<value_type, uint128_t>(smem_utccp_partitioned_tensor.layout()));
}

namespace UMMA {

// Import TMEM constants
namespace TMEM = cute::TMEM;

enum class TmemAllocMode {
  // Default allocation mode.
  // If a TMEM Atom uses a half-subpartition (16DPs), then multiple atoms can be
  // interleaved by using the top-half-subpartition and the bottom-half-subpartition.
  // Full utilization of TMEM capacity.
  Interleaved = 0,
  // Prevents interleaving.
  // If a TMEM Atom uses a half-subpartition (16DPs), then multiple atoms will not be
  // interleaved.
  // Required for DP-address equivalence in TMEM-A and TMEM-C allocations in UMMA_TS.
  NonInterleaved = 1,
  // Duplicates the TMEM allocation across subpartitions.
  // E.g. UMMA_2SM_128xNx16_TS uses a "2x2 DP" TMEM Layout, but the TMEM allocation is
  // actually doubled and the input data must be duplicated between the
  // subpartitions [0,1]<->[2,3], i.e., each subpartition holds all columns
  // of the A matrix needed for a single UMMA operation.
  // For UMMA_2SM_128xNx16_TS, the distribution of the data is as follows.
  // SM0:
  //    Subpart0 = A[0:32, 0:16], Subpart1 = A[32:64, 0:16],
  //    Subpart2 = A[A:32, 0:16], Subpart3 = A[32:64, 0:16]
  // SM1:
  //    Subpart0 = A[64:96, 0:16], Subpart1 = A[96:128, 0:16],
  //    Subpart2 = A[64:96, 0:16], Subpart3 = A[96:128, 0:16]
  Duplicated = 2,
  // Duplicates the TMEM allocation across subpartitions for scale factor.
  // Scale factor TMEM allocation for 4x1 data path
  ScaleFactorDuplicated4by1 = 3,
  // Scale factor TMEM allocation for 2x2 data path
  ScaleFactorDuplicated2by2 = 4
};

struct tmem_frg_base {};

// The UMMA Traits below have custom fragment type flags for their tmem tensors.
// These flags specialize a MakeTensor customization point to correctly make the fragment that is desired.
template <class ValueType, class StorageType, int N_SM, UMMA::TmemAllocMode TmemAlloc = UMMA::TmemAllocMode::Interleaved>
struct tmem_frg : tmem_frg_base
{
  static_assert(sizeof_bits_v<ValueType> <= sizeof_bits_v<StorageType>, "TMEM MMA allocations require StorageType big enough for ValueType.");

  // UMMA TMEM Allocator
  //   Each UMMA expects a specific MxN layout of TMEM for accumulators
  //   and sometimes a specific MxK layout of TMEM for A-values.
  // @tparam ValueType The value type of the TMEM Tensor to allocate.
  // @tparam StorageType The storage type of the TMEM Tensor to allocate.
  //                     "Sparse" allocations often allocate ValueType=half_t within StorageType=uint32_t.
  //                     "Dense"  allocations often allocate ValueType=half_t within StorageType=half_t.
  // @tparam N_SM The number of SMs in this UMMA_XSM instruction.
  // @tparam TmemAlloc UMMA-specific allocation modifier for special cases.
  //                   Some UMMA instructions expect strange atoms or tilings of atoms.
  // @param tmem_shape ((M_MMA_SM,N_MMA_SM),MMA_M,MMA_N,...)
  //                   The post-MMA-partitioned shape of TMEM to allocate.
  //                   Note for UMMA_2SM_128xNx16, that M_MMA_SM will be 64, for example.
  template <class TmemShape>
  CUTE_HOST_DEVICE constexpr static auto
  make(TmemShape const& tmem_shape)
  {
    CUTE_STATIC_ASSERT_V(size(tmem_shape)*Int<int(sizeof_bits_v<StorageType>)>{} <= TMEM::MAX_CAPACITY_BITS{},
                        "Requesting more TMEM than is available.");
    CUTE_STATIC_ASSERT_V(rank<0>(tmem_shape) == Int<2>{}, "Expected post-partitioned shape ((M_MMA,N_MMA),...).");
    constexpr int R     = decltype(rank(tmem_shape))::value;
    constexpr int M_MMA = decltype(size<0,0>(tmem_shape))::value;
    constexpr int N_MMA = decltype(size<0,1>(tmem_shape))::value;

    // It's convenient to use "virtual tensor memory addressing"
    //   with DP_STRIDE=1, COL_STRIDE=128 to define the tmem_atom,
    //   then convert to "logical tensor memory addressing" on return.
    using COL_ADDR = C<sizeof_bits<StorageType>::value / sizeof_bits<ValueType>::value>;
    Layout tmem_restride = Layout<Shape <               _128,   _16384>,
                                  Stride<TMEM::DP<ValueType>, COL_ADDR>>{};

    static_assert(N_SM == 1 || N_SM == 2, "UMMA expects N_SM == 1 or N_SM == 2");
    if constexpr (N_SM == 1)
    {
      static_assert(TmemAlloc == UMMA::TmemAllocMode::Interleaved || TmemAlloc == UMMA::TmemAllocMode::NonInterleaved,
                    "UMMA_1SM only accepts Interleaved or NonInterleaved");
      static_assert(M_MMA == 64 || M_MMA == 128, "UMMA_1SM M-mode size should be 64 or 128.");

      if constexpr (M_MMA == 64)
      {
        // Half subpartitions layout atom: (M,N) -> tmem_addr
        Layout tmem_atom = Layout<Shape <Shape <_16,  _4>, Int<N_MMA>>,
                                  Stride<Stride< _1, _32>,      _128>>{};
        // tile_stride = 2 causes the tiling to "skip" the first tile in DPs
        constexpr int tile_stride = TmemAlloc == UMMA::TmemAllocMode::Interleaved ? 1 : 2;
        // This will tile in DPs first, then COLs
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape),
                                                                          compact_col_major(take<1,R>(tmem_shape),Int<tile_stride>{})));
        // Restride for the DP/COL addressing and return
        return make_tensor(make_tmem_ptr<ValueType>(), composition(tmem_restride, tmem_logical_layout));
      } else
      if constexpr (M_MMA == 128)
      {
        // For M_MMA = 128, all datapaths are occupied. TmemAllocMode doesn't change the allocation.
        // Full subpartitions layout atom: (M,N) -> tmem_addr
        Layout tmem_atom = Layout<Shape <_128,Int<N_MMA>>,
                                  Stride<  _1,     _128>>{};
        // This will tile in DPs first, then COLs
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Restride for the DP/COL addressing and return
        return make_tensor(make_tmem_ptr<ValueType>(), composition(tmem_restride, tmem_logical_layout));
      }

    } else
    if constexpr (N_SM == 2)
    {
      static_assert(TmemAlloc == UMMA::TmemAllocMode::Interleaved || TmemAlloc == UMMA::TmemAllocMode::Duplicated,
                    "UMMA_2SM only accepts Interleaved or Duplicated");
      static_assert(M_MMA == 32 || M_MMA == 64 || M_MMA == 128, "UMMA_2SM M-mode size should be 32 or 64 or 128.");

      if constexpr (M_MMA == 32)
      {
        static_assert(TmemAlloc == UMMA::TmemAllocMode::Interleaved, "Only TmemAllocMode::Interleaved is supported for UMMA_2SM M_MMA=32");
        // The "1x4" layout atom: (M,N) -> tmem_addr
        Layout tmem_atom = Layout<Shape <_32,Shape <Int<N_MMA/4>, _4>>,
                                  Stride< _1,Stride<        _128,_32>>>{};
        // This will tile in DPs first, then COLs
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Restride for the DP/COL addressing and return
        return make_tensor(make_tmem_ptr<ValueType>(), composition(tmem_restride, tmem_logical_layout));
      } else
      if constexpr (M_MMA == 64 && TmemAlloc == UMMA::TmemAllocMode::Interleaved)
      {
        // The "2x2" layout atom: (M,N) -> tmem_addr
        Layout tmem_atom = Layout<Shape <_64,Shape <Int<N_MMA/2>, _2>>,
                                  Stride< _1,Stride<        _128,_64>>>{};
        // This will tile in DPs first, then COLs
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Restride for the DP/COL addressing and return
        return make_tensor(make_tmem_ptr<ValueType>(), composition(tmem_restride, tmem_logical_layout));

      } else
      if constexpr (M_MMA == 64 && TmemAlloc == UMMA::TmemAllocMode::Duplicated)
      {
        // The "2x2" duplicated layout atom: (M,N) -> tmem_addr
        Layout tmem_atom = Layout<Shape <_128,Int<N_MMA>>,
                                  Stride< _1,      _128>>{};
        // This will tile in DPs first, then COLs
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Restride for the DP/COL addressing and return
        return make_tensor(make_tmem_ptr<ValueType>(), composition(tmem_restride, tmem_logical_layout));
      } else
      if constexpr (M_MMA == 128)
      {
        // For M_MMA = 128, all datapaths are occupied. TmemAllocMode doesn't change the allocation.
        // The "4x1" layout atom: (M,N) -> tmem_addr
        Layout tmem_atom = Layout<Shape <_128,Int<N_MMA>>,
                                  Stride<  _1,     _128>>{};
        // This will tile in DPs first, then COLs
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Restride for the DP/COL addressing and return
        return make_tensor(make_tmem_ptr<ValueType>(), composition(tmem_restride, tmem_logical_layout));
      }
    }

    CUTE_GCC_UNREACHABLE;
  }
};

// Convenient aliases for common cases in the UMMA::ElementXFrg below
template <class ValueType, class StorageType = uint32_t, UMMA::TmemAllocMode TmemAlloc = UMMA::TmemAllocMode::Interleaved>
using tmem_frg_1sm = tmem_frg<ValueType, StorageType, 1, TmemAlloc>;
template <class ValueType, class StorageType = uint32_t, UMMA::TmemAllocMode TmemAlloc = UMMA::TmemAllocMode::Interleaved>
using tmem_frg_2sm = tmem_frg<ValueType, StorageType, 2, TmemAlloc>;

// Make metadata TMEM fragments for sparse MMAs.
// Also note that the TMEM fragment addresses are assumed to be COL-4 aligned -- working with arch to remove this condition
template <class ValueType>
struct tmem_e_frg : tmem_frg_base
{
  template <class TmemShape>
  CUTE_HOST_DEVICE constexpr static auto
  make(TmemShape const& tmem_shape)
  {
    CUTE_STATIC_ASSERT_V(rank<0>(tmem_shape) == Int<2>{}, "Expected post-partitioned shape ((M_MMA,N_MMA),...).");
    constexpr int R     = decltype(rank(tmem_shape))::value;
    constexpr int M_MMA = decltype(size<0,0>(tmem_shape))::value;
    constexpr int N_MMA = decltype(size<0,1>(tmem_shape))::value;

    static_assert(M_MMA == 128, "Only 128 implemented right now.");

    // It's convenient to use "virtual tensor memory addressing"
    //   with DP_STRIDE=1, COL_STRIDE=128 to define the tmem_atom,
    //   then convert to "logical tensor memory addressing" on return.
    [[maybe_unused]] Layout tmem_restride = Layout<Shape <      _128, _16384>,
                                                   Stride<TMEM::DP_b,     _1>>{};

    if constexpr (sizeof_bits<ValueType>::value == 32)     // TF32: 128x16 atom
    {
      static_assert(N_MMA == 16);
      Layout tmem_atom = Layout<Shape <Shape <_8,   _2, _8>, Shape <  _8,_2>>,
                                Stride<Stride<_1,_1024,_16>, Stride<_128,_8>>>{};
      // Tile to MMA tiling
      Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
      // Address transformations with upcast<2> for 2-bit base types
      Layout tmem_layout = composition(upcast<2>(tmem_restride), tmem_logical_layout);
      // Sparsity wrap, no sparse_ptr because tmem_ptr handles it's own subword addressing
      return make_tensor(make_tmem_ptr<sparse_elem<4,uint8_t>>(), tmem_layout);
    } else
    if constexpr (sizeof_bits<ValueType>::value == 16)     // FP16: 128x32 atom
    {
      static_assert(N_MMA == 32);
      Layout tmem_atom = Layout<Shape <Shape <_8,   _2, _8>, Shape < _16,_2>>,
                                Stride<Stride<_1,_2048,_16>, Stride<_128,_8>>>{};
      // Tile to MMA tiling
      Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
      // Address transformations
      Layout tmem_layout = composition(tmem_restride, tmem_logical_layout);
      // Sparsity wrap, no sparse_ptr because tmem_ptr handles it's own subword addressing
      return make_tensor(make_tmem_ptr<sparse_elem<8,uint8_t>>(), tmem_layout);
    } else
    if constexpr (sizeof_bits<ValueType>::value ==  8)     // S8|Mix.F4/F6/F8: 128x64 atom
    {
      // For Mix 8bit f4/f6/f8, will pass in ValueType = uint8_t
      static_assert(N_MMA == 64);
      Layout tmem_atom = Layout<Shape <_128, _64>,
                                Stride<  _1,_128>>{};
      // Tile to MMA tiling
      Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
      // Address transformations
      Layout tmem_layout = composition(tmem_restride, tmem_logical_layout);
      // Sparsity wrap, no sparse_ptr because tmem_ptr handles it's own subword addressing
      return make_tensor(make_tmem_ptr<sparse_elem<8,uint8_t>>(), tmem_layout);
    }
    if constexpr (sizeof_bits<ValueType>::value ==  4)     // F4: 128x128 atom
    {
      // For F4, will pass in ValueType = fp4
      Layout tmem_restride1 = Layout<Shape <                     _128, Int<32768>>,
                                     Stride<cute::C<int32_t(1) << 22>,         _1>>{};
      // F4 has roughly same TMEM layout as Mix8bit.F4/F6/F8, the only difference is that K is multiplied by two
      static_assert(N_MMA == 128);
      Layout tmem_atom = Layout<Shape <_128, _128>,
                                Stride<  _1, _128>>{};
      // Tile to MMA tiling
      Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
      // Address transformations
      Layout tmem_layout = composition(tmem_restride1, tmem_logical_layout);
      // Sparsity wrap, no sparse_ptr because tmem_ptr handles it's own subword addressing
      return make_tensor(make_tmem_ptr<sparse_elem<16,uint8_t>>(), tmem_layout);
    }

    CUTE_GCC_UNREACHABLE;
  }
};

template <class ValueType>
struct tmem_e_frg_ws : tmem_frg_base
{
  template <class TmemShape>
  CUTE_HOST_DEVICE constexpr static auto
  make(TmemShape const& tmem_shape)
  {
    CUTE_STATIC_ASSERT_V(rank<0>(tmem_shape) == Int<2>{}, "Expected post-partitioned shape ((M_MMA,N_MMA),...).");
    constexpr int R     = decltype(rank(tmem_shape))::value;
    constexpr int M_MMA = decltype(size<0,0>(tmem_shape))::value;
    constexpr int N_MMA = decltype(size<0,1>(tmem_shape))::value;

    static_assert(M_MMA == 128 || M_MMA == 64 || M_MMA == 32, "Weight stationary UMMA_1SM M-mode size should be 32 or 64 or 128.");

    // It's convenient to use "virtual tensor memory addressing"
    //   with DP_STRIDE=1, COL_STRIDE=128 to define the tmem_atom,
    //   then convert to "logical tensor memory addressing" on return.
    Layout tmem_restride = Layout<Shape <      _128, _16384>,
                                  Stride<TMEM::DP_b,     _1>>{};

    if constexpr (sizeof_bits<ValueType>::value == 32)     // TF32
    {
      // MMA_M x MMA_K: 128x16 atom / 64x16 atom / 32x16 atom
      static_assert(N_MMA == 16);
      if constexpr (M_MMA == 128) {
        Layout tmem_atom = Layout<Shape <Shape <_8,   _2, _8>, Shape <  _8,_2>>,
                                  Stride<Stride<_1,_1024,_16>, Stride<_128,_8>>>{};
        // Tile to MMA tiling
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Address transformations with upcast<2> for 2-bit base types
        Layout tmem_layout = composition(upcast<2>(tmem_restride), tmem_logical_layout);
        // Sparsity wrap, no sparse_ptr because tmem_ptr handles it's own subword addressing
        return make_tensor(make_tmem_ptr<sparse_elem<4,uint8_t>>(), tmem_layout);
      }
      else if constexpr (M_MMA == 64) {
        Layout tmem_atom = Layout<Shape <Shape <_8,   _2, _4>, Shape <  _8,_2>, _2>,
                                  Stride<Stride<_1,_1024,_16>, Stride<_128,_8>,_64>>{};
        // Tile to MMA tiling
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Address transformations with upcast<2> for 2-bit base types
        Layout tmem_layout = composition(upcast<2>(tmem_restride), tmem_logical_layout);
        // Sparsity wrap, no sparse_ptr because tmem_ptr handles its own subword addressing
        return make_tensor(make_tmem_ptr<sparse_elem<4,uint8_t>>(), tmem_layout);
      }
      else if constexpr (M_MMA == 32) {
        Layout tmem_atom = Layout<Shape <Shape <_8,   _2, _2>, Shape <  _8,_2>, _4>,
                                  Stride<Stride<_1,_1024,_16>, Stride<_128,_8>,_32>>{};
        // Tile to MMA tiling
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Address transformations with upcast<2> for 2-bit base types
        Layout tmem_layout = composition(upcast<2>(tmem_restride), tmem_logical_layout);
        // Sparsity wrap, no sparse_ptr because tmem_ptr handles its own subword addressing
        return make_tensor(make_tmem_ptr<sparse_elem<4,uint8_t>>(), tmem_layout);
      }
      else {
        static_assert(dependent_false<TmemShape>, "Invalid M_MMA value");
      }
    }
    else if constexpr (sizeof_bits<ValueType>::value == 16)     // FP16
    {
      // MMA_M x MMA_K: 128x32 atom / 64x32 atom / 32x32 atom
      static_assert(N_MMA == 32);
      if constexpr (M_MMA == 128) {
        Layout tmem_atom = Layout<Shape <Shape <_8,   _2, _8>, Shape < _16,_2>>,
                                  Stride<Stride<_1,_2048,_16>, Stride<_128,_8>>>{};
        // Tile to MMA tiling
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Address transformations
        Layout tmem_layout = composition(tmem_restride, tmem_logical_layout);
        // Sparsity wrap, no sparse_ptr because tmem_ptr handles it's own subword addressing
        return make_tensor(make_tmem_ptr<sparse_elem<8,uint8_t>>(), tmem_layout);
      }
      else if constexpr (M_MMA == 64) {
        Layout tmem_atom = Layout<Shape <Shape <_8,   _2, _4>, Shape < _16,_2>, _2>,
                                  Stride<Stride<_1,_2048,_16>, Stride<_128,_8>,_64>>{};
        // Tile to MMA tiling
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Address transformations
        Layout tmem_layout = composition(tmem_restride, tmem_logical_layout);
        // Sparsity wrap, no sparse_ptr because tmem_ptr handles it's own subword addressing
        return make_tensor(make_tmem_ptr<sparse_elem<8,uint8_t>>(), tmem_layout);
      }
      else if constexpr (M_MMA == 32) {
        Layout tmem_atom = Layout<Shape <Shape <_8,   _2, _2>, Shape < _16,_2>, _4>,
                                  Stride<Stride<_1,_2048,_16>, Stride<_128,_8>,_32>>{};
        // Tile to MMA tiling
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Address transformations
        Layout tmem_layout = composition(tmem_restride, tmem_logical_layout);
        // Sparsity wrap, no sparse_ptr because tmem_ptr handles it's own subword addressing
        return make_tensor(make_tmem_ptr<sparse_elem<8,uint8_t>>(), tmem_layout);
      }
      else {
        static_assert(dependent_false<TmemShape>, "Invalid M_MMA value");
      }
    }
    else if constexpr (sizeof_bits<ValueType>::value ==  8)     // I8|F8
    {
      // MMA_M x MMA_K: 128x64 atom / 64x64 atom / 32x64 atom
      static_assert(N_MMA == 64);
      if constexpr (M_MMA == 128) {
        Layout tmem_atom = Layout<Shape <_128, _64>,
                                  Stride<  _1,_128>>{};
        // Tile to MMA tiling
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Address transformations
        Layout tmem_layout = composition(tmem_restride, tmem_logical_layout);
        // Sparsity wrap, no sparse_ptr because tmem_ptr handles it's own subword addressing
        return make_tensor(make_tmem_ptr<sparse_elem<8,uint8_t>>(), tmem_layout);
      }
      else if constexpr (M_MMA == 64) {
        Layout tmem_atom = Layout<Shape <_64, Shape < _64,  _2>>,
                                  Stride< _1, Stride<_128, _64>>>{};
        // Tile to MMA tiling
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Address transformations
        Layout tmem_layout = composition(tmem_restride, tmem_logical_layout);
        // Sparsity wrap, no sparse_ptr because tmem_ptr handles it's own subword addressing
        return make_tensor(make_tmem_ptr<sparse_elem<8,uint8_t>>(), tmem_layout);
      }
      else if constexpr (M_MMA == 32) {
        Layout tmem_atom = Layout<Shape <_32, Shape < _64,  _4>>,
                                  Stride< _1, Stride<_128, _32>>>{};
        // Tile to MMA tiling
        Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
        // Address transformations
        Layout tmem_layout = composition(tmem_restride, tmem_logical_layout);
        // Sparsity wrap, no sparse_ptr because tmem_ptr handles it's own subword addressing
        return make_tensor(make_tmem_ptr<sparse_elem<8,uint8_t>>(), tmem_layout);
      }
      else {
        static_assert(dependent_false<TmemShape>, "Invalid M_MMA value");
      }
    }
    else {
      static_assert(dependent_false<TmemShape>, "Invalid ValueType");
    }

    CUTE_GCC_UNREACHABLE;
  }
};

template <class ValueType, int SFVecSize, int N_SM, bool Is_SFA,
    UMMA::TmemAllocMode TmemAlloc = UMMA::TmemAllocMode::ScaleFactorDuplicated4by1>
struct tmem_sf_frg: tmem_frg_base
{
  // UMMA TMEM Allocator for Scale Factor A for Mxf4Nvf4 and Mxf8f6f4 instructions
  //  We expect a tensor that has the same layout as A matrix
  //  @tparam ValueType: data type of scaling factor
  //    Note that the StorageType is the same as ValueType, i.e., we always use a compact allocation
  //  @tparam SFVecSize: The number of values that is scaled by a single scaling factor.
  //    Valid values are (16, 32)
  //  @tparam N_SM: Number of SMs in UMMA instruction
  //  @param tmem_shape: An MMA partitioned shape where first mode encodes, A layout of the MMA instruction.
  //    Note that the shape doesn't match the actual allocation. size<0,1>(tmem_shape) will give us the number of
  //    elements in K-mode of MMA rather than the number of scaling factors.
  template <class TmemShape>
  CUTE_HOST_DEVICE constexpr static auto
  make(TmemShape const& tmem_shape)
  {
    CUTE_STATIC_ASSERT_V(rank<0>(tmem_shape) == Int<2>{}, "Expected post-partitioned shape ((M_MMA,N_MMA),...).");
    constexpr int MMA_MN  = decltype(size<0,0>(tmem_shape))::value;
    constexpr int MMA_VS  = decltype(size<0,1,0>(tmem_shape))::value;
    constexpr int MMA_NSF = decltype(size<0,1,1>(tmem_shape))::value;
    constexpr int R_MMA_K = decltype(rank(get<0,1>(tmem_shape)))::value;
    constexpr int R = decltype(rank(tmem_shape))::value;

    // We expect an MMA-SF partitioned tensor
    // ((MMA_MN, (VecSize, NSF)), num_MMA_MN, num_MMA_K, ...)
    //   where VecSize*NSF = MMA_K
    static_assert(R >= 3,       "Expected an MMA partitioned tensor");                            // ((MMA), num_MMA_MN, num_MMA_K, ...)
    static_assert(R_MMA_K == 2, "Expected an MMA-SF partitioned tensor");                         // (VecSize, NSF)
    using REP = _4;               // Replication factor. Data is always replicated across subpartitions
    constexpr int SUBPART_DPs = 32;      // Number of DPs in a subpartition

    using COL_ADDR = C<sizeof_bits<ValueType>::value / sizeof_bits<ValueType>::value>;
    Layout tmem_restride = Layout<Shape <               _128,   _16384>,
                                  Stride<TMEM::DP<ValueType>, COL_ADDR>>{};

    if constexpr (Is_SFA || (!Is_SFA && TmemAlloc == UMMA::TmemAllocMode::ScaleFactorDuplicated4by1)) {
      // SFA, 2x2 and 4x1 data path
      // SFB,         4x1 data path
      auto tmem_atom = Layout < Shape< Shape< Shape<Int<SUBPART_DPs>, Int<MMA_MN/SUBPART_DPs>>, REP>,  Shape<Int<MMA_VS>, Int<MMA_NSF>>>,
                              Stride<Stride<Stride<              _1,                    _512>, _32>, Stride<         _0,         _128>>>{};

      Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
      auto final_tmem_layout = composition(tmem_restride, tmem_logical_layout);
      return make_tensor(make_tmem_ptr<ValueType>(), final_tmem_layout);
    }
    else {
      // SFB, 2x2 datapath
      static_assert(!Is_SFA and TmemAlloc == UMMA::TmemAllocMode::ScaleFactorDuplicated2by2);
      static_assert(N_SM == 2, "Should be 2x2 Datapath");
      // 2x2 Datapth
      auto tmem_atom = Layout < Shape< Shape< Shape<Int<SUBPART_DPs>, Int<MMA_MN/2/SUBPART_DPs>>,     _2,  _2>,  Shape<Int<MMA_VS>, Int<MMA_NSF>>>,
                                Stride<Stride<Stride<    _1         ,                      _512>,    _64, _32>, Stride<         _0,        _128>>>{};

      Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
      auto final_tmem_layout = composition(tmem_restride, tmem_logical_layout);
      return make_tensor(make_tmem_ptr<ValueType>(), final_tmem_layout);
    }
  }
};

// Make C/D Tmem fragment for weight-stationary MMAs
template <class ValueType, class StorageType, int N_SM>
struct tmem_frg_ws : tmem_frg_base
{
  static_assert(sizeof_bits_v<ValueType> <= sizeof_bits_v<StorageType>, "TMEM MMA allocations require StorageType big enough for ValueType.");

  // UMMA TMEM Allocator
  //   Each UMMA expects a specific MxN layout of TMEM for accumulators
  //   and sometimes a specific MxK layout of TMEM for A-values.
  // @tparam ValueType The value type of the TMEM Tensor to allocate.
  // @tparam StorageType The storage type of the TMEM Tensor to allocate.
  //                     "Sparse" allocations often allocate ValueType=half_t within StorageType=uint32_t.
  //                     "Dense"  allocations often allocate ValueType=half_t within StorageType=half_t.
  // @tparam N_SM The number of SMs in this UMMA_XSM instruction.
  // @tparam TmemAlloc UMMA-specific allocation modifier for special cases.
  //                   Some UMMA instructions expect strange atoms or tilings of atoms.
  // @param tmem_shape ((M_MMA_SM,N_MMA_SM),MMA_M,MMA_N,...)
  //                   The post-MMA-partitioned shape of TMEM to allocate.
  //                   Note for UMMA_2SM_128xNx16, that M_MMA_SM will be 64, for example.
  template <class TmemShape>
  CUTE_HOST_DEVICE constexpr static auto
  make(TmemShape const& tmem_shape)
  {
    CUTE_STATIC_ASSERT_V(size(tmem_shape)*Int<int(sizeof_bits_v<StorageType>)>{} <= TMEM::MAX_CAPACITY_BITS{},
                        "Requesting more TMEM than is available.");
    CUTE_STATIC_ASSERT_V(rank<0>(tmem_shape) == Int<2>{}, "Expected post-partitioned shape ((M_MMA,N_MMA),...).");
    constexpr int R     = decltype(rank(tmem_shape))::value;
    constexpr int M_MMA = decltype(size<0,0>(tmem_shape))::value;
    constexpr int N_MMA = decltype(size<0,1>(tmem_shape))::value;

    // It's convenient to use "virtual tensor memory addressing"
    //   with DP_STRIDE=1, COL_STRIDE=128 to define the tmem_atom,
    //   then convert to "logical tensor memory addressing" on return.
    using COL_ADDR = C<sizeof_bits<StorageType>::value / sizeof_bits<ValueType>::value>;
    Layout tmem_restride = Layout<Shape <               _128,   _16384>,
                                  Stride<TMEM::DP<ValueType>, COL_ADDR>>{};

    static_assert(N_SM == 1, "UMMA.WS expects N_SM == 1");

    static_assert(M_MMA == 32 || M_MMA == 64 || M_MMA == 128,
                  "Weight stationary UMMA_1SM M-mode size should be 32 or 64 or 128.");
    static_assert(N_MMA == 64 || N_MMA == 128 || N_MMA == 256,
                  "Dense weight stationary UMMA_1SM N-mode size should be 64 or 128 or 256.");
    // Weight Stationary MMA config
    if constexpr (M_MMA == 32)
    {
      // 1x4 datapath
      Layout tmem_atom = Layout<Shape <_32, Shape<Int<N_MMA/4>, _4>>,
                                Stride< _1, Stride<       _128,_32>>
                              >{};
      constexpr int tile_stride = 1;
      // This will tile in DPs first, then COLs
      Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape),
                                                                        compact_col_major(take<1,R>(tmem_shape), Int<tile_stride>{})));
      // Restride for the DP/COL addressing and return
      return make_tensor(make_tmem_ptr<ValueType>(), composition(tmem_restride, tmem_logical_layout));
    } else
    if constexpr (M_MMA == 64)
    {
      // 2x2 datapath
      Layout tmem_atom = Layout<Shape <_64, Shape<Int<N_MMA/2>, _2>>,
                                Stride< _1, Stride<       _128,_64>>
                              >{};
      constexpr int tile_stride = 1;
      // This will tile in DPs first, then COLs
      Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape),
                                                                        compact_col_major(take<1,R>(tmem_shape), Int<tile_stride>{})));
      // Restride for the DP/COL addressing and return
      return make_tensor(make_tmem_ptr<ValueType>(), composition(tmem_restride, tmem_logical_layout));
    } else
    if constexpr (M_MMA == 128)
    {
      // For M_MMA = 128, all datapaths are occupied. TmemAllocMode doesn't change the allocation.
      // Full subpartitions layout atom: (M,N) -> tmem_addr
      Layout tmem_atom = Layout<Shape <_128,Int<N_MMA>>,
                                Stride<  _1,     _128>>{};
      // This will tile in DPs first, then COLs
      Layout tmem_logical_layout = tiled_product(tmem_atom, make_layout(take<1,R>(tmem_shape)));
      // Restride for the DP/COL addressing and return
      return make_tensor(make_tmem_ptr<ValueType>(), composition(tmem_restride, tmem_logical_layout));
    }

    CUTE_GCC_UNREACHABLE;
  }
};

// Convenient aliases for common cases in the UMMA::ElementXFrg below
template <class ValueType, class StorageType = uint32_t>
using tmem_frg_ws_1sm = tmem_frg_ws<ValueType, StorageType, 1>;

} // end namespace UMMA

// Customization point for creating a UMMA::tmem_frg Tensor
template <class ValueType, class StorageType, int N_SM, UMMA::TmemAllocMode TmemAlloc>
struct MakeTensor<UMMA::tmem_frg<ValueType, StorageType, N_SM, TmemAlloc>>
{
  template <class Shape>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Shape const& tmem_shape) {
    return UMMA::tmem_frg<ValueType, StorageType, N_SM, TmemAlloc>::make(shape(tmem_shape));
  }
};

template <class ValueType, class StorageType, int N_SM>
struct MakeTensor<UMMA::tmem_frg_ws<ValueType, StorageType, N_SM>>
{
  template <class Shape>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Shape const& tmem_shape) {
    return UMMA::tmem_frg_ws<ValueType, StorageType, N_SM>::make(shape(tmem_shape));
  }
};


// Customization point for creating a UMMA::tmem_frg Tensor
template <class ValueType>
struct MakeTensor<UMMA::tmem_e_frg<ValueType>>
{
  template <class Shape>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Shape const& tmem_shape) {
    return UMMA::tmem_e_frg<ValueType>::make(shape(tmem_shape));
  }
};

template <class ValueType>
struct MakeTensor<UMMA::tmem_e_frg_ws<ValueType>>
{
  template <class Shape>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Shape const& tmem_shape) {
    return UMMA::tmem_e_frg_ws<ValueType>::make(shape(tmem_shape));
  }
};

template <class ValueType, int SFVecSize, int N_SM, bool Is_SFA, UMMA::TmemAllocMode TmemAlloc>
struct MakeTensor<UMMA::tmem_sf_frg<ValueType, SFVecSize, N_SM, Is_SFA, TmemAlloc>>
{
  template <class Shape>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Shape const& tmem_shape) {
    return UMMA::tmem_sf_frg<ValueType, SFVecSize, N_SM, Is_SFA, TmemAlloc>::make(shape(tmem_shape));
  }
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////////// MMA_TRAITS ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_TF32_SS<a_type, b_type, c_type,
                                M, N, a_major, b_major,
                                a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 32, "SM100_MMA_TF32_SS supports 32bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  // Logical shape-K is always 256bits, transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_TF32_SS<a_type, b_type, c_type,
                   M, N, a_major, b_major,
                   a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_SS<a_type, b_type, c_type,
                                M, N, a_major, b_major,
                                a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_SS supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  // Logical shape-K is always 256bits, transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_SS<a_type, b_type, c_type,
                  M, N, a_major, b_major,
                  a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_TF32_TS<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 32, "SM100_MMA_TF32_TS supports 32bit types");

  using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  // Logical shape-K is always 256 bits; transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint32_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_TF32_TS<a_type, b_type, c_type,
                   M, N,
                   a_major, b_major,
                   a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F16BF16_TS<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_TS supports 16bit types");

  using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  // Logical shape-K is always 256 bits; transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint32_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_TS<a_type, b_type, c_type,
                  M, N,
                  a_major, b_major,
                  a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          uint32_t ScaleC, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_SS_SCALED<a_type, b_type, c_type,
                                M, N, a_major, b_major,
                                ScaleC, a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_SS_SCALED supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  // Logical shape-K is always 256bits, transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  static constexpr uint32_t ScalingFactor = ScaleC;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_SS_SCALED<a_type, b_type, c_type,
                         M, N, a_major, b_major,
                         ScaleC, a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);

  }

  template <uint32_t NewScaleC>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_F16BF16_SS_SCALED<a_type, b_type, c_type,
                                  M, N, a_major, b_major,
                                  NewScaleC, a_neg, b_neg>>
  with(UMMA::ScaleOut accumulate, cute::integral_constant<uint32_t, NewScaleC> scaleC) const {
    return {accumulate, idesc_};
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          uint32_t ScaleC, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F16BF16_TS_SCALED<a_type, b_type, c_type,
                                M, N, a_major, b_major,
                                ScaleC, a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_TS_SCALED supports 16bit types");

  using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  // Logical shape-K is always 256 bits; transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  static constexpr uint32_t ScalingFactor = ScaleC;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint32_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_TS_SCALED<a_type, b_type, c_type,
                         M, N, a_major, b_major,
                         ScaleC, a_neg, b_neg>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }

  template <uint32_t NewScaleC>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_F16BF16_TS_SCALED<a_type, b_type, c_type,
                                  M, N, a_major, b_major,
                                  NewScaleC, a_neg, b_neg, c_sat>>
  with(UMMA::ScaleOut accumulate, cute::integral_constant<uint32_t, NewScaleC> scaleC) const {
    return {accumulate, idesc_};
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          class... sparse_args>
struct MMA_Traits<SM100_MMA_TF32_SS_SPARSE<a_type, b_type, c_type,
                                       M, N, a_major, b_major,
                                       a_neg, b_neg>, sparse_args...>
{
  using ValTypeD = c_type;
  static_assert(sizeof(a_type) == 4);
  using ValTypeA = sparse_elem<2, a_type>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 32, "SM100_MMA_TF32_SS_SPARSE supports 32bit types");

  using FrgTypeA = UMMA::sparse_smem_desc<a_major>;
  using FrgTypeE = UMMA::tmem_e_frg<a_type>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  // SparseMma consume double mma-k bits
  static constexpr int K = 512 / cute::sizeof_bits<a_type>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  // uint32_t tmem_e: Metadata tmem address.
  cute::tuple<sparse_args...> sparse_args_;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, UMMA::Saturate::False, true>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_same<cute::tuple<sparse_args...>, cute::tuple<uint32_t>>::value,
                  "Params must be set via .with()?");
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());

    uint32_t tmem_e = get<0>(traits.sparse_args_);
    uint32_t id2 = tmem_e &  0x00000001;
    tmem_e       = tmem_e & ~0x00000001;

    uint64_t idesc = UMMA::make_runtime_instr_desc<true>(traits.idesc_, static_cast<uint16_t>(id2), tmem_e);

    SM100_MMA_TF32_SS_SPARSE<a_type, b_type, c_type,
                          M, N, a_major, b_major,
                          a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, tmem_e);
  }

  // Construct an executable sparse MMA_traits with sp into set.
  template <class TE, class ELayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_TF32_SS_SPARSE<a_type, b_type, c_type,
                                   M, N, a_major, b_major,
                                   a_neg, b_neg>, uint32_t>
  with(Tensor<TE, ELayout> const& E) const {
    // Check sparse_ptr, check sparsity, check shape/layout?
    uint32_t tmem_e_addr = raw_pointer_cast(E.data());     // Move to a CoupledTensor rather than a .with()?
    return {accumulate_, {tmem_e_addr}, idesc_};
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          class... sparse_args>
struct MMA_Traits<SM100_MMA_F16BF16_SS_SPARSE<a_type, b_type, c_type,
                                       M, N, a_major, b_major,
                                       a_neg, b_neg>, sparse_args...>
{
  using ValTypeD = c_type;
  static_assert(sizeof(a_type) == 2);
  using ValTypeA = sparse_elem<2, a_type>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_SS_SPARSE supports 16bit types");

  using FrgTypeA = UMMA::sparse_smem_desc<a_major>;
  using FrgTypeE = UMMA::tmem_e_frg<a_type>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  // SparseMma consume double mma-k bits
  static constexpr int K = 512 / cute::sizeof_bits<a_type>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  // uint32_t tmem_e: Metadata tmem address.
  cute::tuple<sparse_args...> sparse_args_;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, UMMA::Saturate::False, true>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_same<cute::tuple<sparse_args...>, cute::tuple<uint32_t>>::value,
                  "Params must be set via .with()?");
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());

    uint32_t tmem_e = get<0>(traits.sparse_args_);
    uint32_t id2 = tmem_e &  0x00000001;
    tmem_e       = tmem_e & ~0x00000001;

    uint64_t idesc = UMMA::make_runtime_instr_desc<true>(traits.idesc_, static_cast<uint16_t>(id2), tmem_e);

    SM100_MMA_F16BF16_SS_SPARSE<a_type, b_type, c_type,
                         M, N, a_major, b_major,
                         a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, tmem_e);
  }

  // Construct an executable sparse MMA_traits with sp into set.
  template <class TE, class ELayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_F16BF16_SS_SPARSE<a_type, b_type, c_type,
                                  M, N, a_major, b_major,
                                  a_neg, b_neg>, uint32_t>
  with(Tensor<TE, ELayout> const& E) const {
    // Check sparse_ptr, check sparsity, check shape/layout?
    uint32_t tmem_e_addr = raw_pointer_cast(E.data());     // Move to a CoupledTensor rather than a .with()?
    return {accumulate_, {tmem_e_addr}, idesc_};
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_TF32_2x1SM_SS<a_type, b_type, c_type,
                                     M, N, a_major, b_major,
                                     a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 32, "SM100_MMA_TF32_2x1SM_SS supports 32bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instructions's K extent is always 256bits, convert to units of element
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_TF32_2x1SM_SS<a_type, b_type, c_type,
                         M, N,
                         a_major, b_major,
                         a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS<a_type, b_type, c_type,
                                     M, N, a_major, b_major,
                                     a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_2x1SM_SS supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instructions's K extent is always 256bits, convert to units of element
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_2x1SM_SS<a_type, b_type, c_type,
                       M, N,
                       a_major, b_major,
                       a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_TF32_2x1SM_TS<a_type, b_type, c_type,
                                     M, N,
                                     a_major, b_major,
                                     a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 32, "SM100_MMA_TF32_2x1SM_TS supports 32bit types");

  using FrgTypeA = UMMA::tmem_frg_2sm<a_type, a_type, UMMA::TmemAllocMode::Duplicated>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instructions' K extent is always 256 bits; convert to units of element
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_TF32_2x1SM_TS<a_type, b_type, c_type,
                       M, N,
                       a_major, b_major,
                       a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F16BF16_2x1SM_TS<a_type, b_type, c_type,
                                     M, N,
                                     a_major, b_major,
                                     a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_2x1SM_TS supports 16bit types");

  using FrgTypeA = UMMA::tmem_frg_2sm<a_type, a_type, UMMA::TmemAllocMode::Duplicated>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instructions' K extent is always 256 bits; convert to units of element
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_2x1SM_TS<a_type, b_type, c_type,
                       M, N,
                       a_major, b_major,
                       a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          uint32_t ScaleC, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS_SCALED<a_type, b_type, c_type,
                                     M, N, a_major, b_major,
                                     ScaleC, a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_2x1SM_SS_SCALED supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instructions's K extent is always 256bits, convert to units of element
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;
  constexpr static uint32_t ScalingFactor = ScaleC;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_2x1SM_SS_SCALED<a_type, b_type, c_type,
                               M, N, a_major, b_major,
                               ScaleC, a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }

  template <uint32_t NewScaleC>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS_SCALED<a_type, b_type, c_type,
                                     M, N, a_major, b_major,
                                     NewScaleC, a_neg, b_neg>>
  with(UMMA::ScaleOut accumulate, cute::integral_constant<uint32_t, NewScaleC> scaleC) const {
    return {accumulate, idesc_};
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          uint32_t ScaleC, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F16BF16_2x1SM_TS_SCALED<a_type, b_type, c_type,
                                     M, N, a_major, b_major,
                                     ScaleC, a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_2x1SM_TS_SCALED supports 16bit types");

  using FrgTypeA = UMMA::tmem_frg_2sm<a_type, a_type, UMMA::TmemAllocMode::Duplicated>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instructions' K extent is always 256 bits; convert to units of element
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;
  constexpr static uint32_t ScalingFactor = ScaleC;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_2x1SM_TS_SCALED<a_type, b_type, c_type,
                               M, N, a_major, b_major,
                               ScaleC, a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }

  template <uint32_t NewScaleC>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_F16BF16_2x1SM_TS_SCALED<a_type, b_type, c_type,
                                        M, N, a_major, b_major,
                                        NewScaleC, a_neg, b_neg, c_sat>>
  with(UMMA::ScaleOut accumulate, cute::integral_constant<uint32_t, NewScaleC> scaleC) const {
    return {accumulate, idesc_};
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          class... sparse_args>
struct MMA_Traits<SM100_MMA_TF32_2x1SM_SS_SPARSE<a_type, b_type, c_type,
                                             M, N, a_major, b_major,
                                             a_neg, b_neg>, sparse_args...>
{
  using ValTypeD = c_type;
  static_assert(sizeof(a_type) == 4);
  using ValTypeA = sparse_elem<2, a_type>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 32, "SM100_MMA_TF32_2x1SM_SS_SPARSE supports 32bit types");

  using FrgTypeA = UMMA::sparse_smem_desc<a_major>;
  using FrgTypeE = UMMA::tmem_e_frg<a_type>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // SparseMma consume double mma-k bits
  constexpr static int K = 512 / cute::sizeof_bits<a_type>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  // uint32_t tmem_e: Metadata tmem address.
  cute::tuple<sparse_args...> sparse_args_;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, UMMA::Saturate::False, true>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_same<cute::tuple<sparse_args...>, cute::tuple<uint32_t>>::value,
                  "Params must be set via .with()?");
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());

    uint32_t tmem_e = get<0>(traits.sparse_args_);
    uint32_t id2 = tmem_e &  0x00000001;
    tmem_e       = tmem_e & ~0x00000001;

    uint64_t idesc = UMMA::make_runtime_instr_desc<true>(traits.idesc_, static_cast<uint16_t>(id2), tmem_e);

    SM100_MMA_TF32_2x1SM_SS_SPARSE<a_type, b_type, c_type,
                              M, N, a_major, b_major,
                              a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, tmem_e);
  }

  // Construct an executable sparse MMA_traits with sp into set.
  template <class TE, class ELayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_TF32_2x1SM_SS_SPARSE<a_type, b_type, c_type,
                                       M, N, a_major, b_major,
                                       a_neg, b_neg>, uint32_t>
  with(Tensor<TE, ELayout> const& E, uint32_t id2 = 0) const {
    // Check sparse_ptr, check sparsity, check shape/layout?
    uint32_t tmem_e_addr = raw_pointer_cast(E.data());
    return {accumulate_, {tmem_e_addr}, idesc_};
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          class... sparse_args>
struct MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS_SPARSE<a_type, b_type, c_type,
                                             M, N, a_major, b_major,
                                             a_neg, b_neg>, sparse_args...>
{
  using ValTypeD = c_type;
  static_assert(sizeof(a_type) == 2);
  using ValTypeA = sparse_elem<2, a_type>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 16, "SM100_MMA_F16BF16_2x1SM_SS_SPARSE supports 16bit types");

  using FrgTypeA = UMMA::sparse_smem_desc<a_major>;
  using FrgTypeE = UMMA::tmem_e_frg<a_type>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // SparseMma consume double mma-k bits
  constexpr static int K = 512 / cute::sizeof_bits<a_type>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  // uint32_t tmem_e: Metadata tmem address.
  cute::tuple<sparse_args...> sparse_args_;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, UMMA::Saturate::False, true>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_same<cute::tuple<sparse_args...>, cute::tuple<uint32_t>>::value,
                  "Params must be set via .with()?");
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());

    uint32_t tmem_e = get<0>(traits.sparse_args_);
    uint32_t id2 = tmem_e &  0x00000001;
    tmem_e       = tmem_e & ~0x00000001;

    uint64_t idesc = UMMA::make_runtime_instr_desc<true>(traits.idesc_, static_cast<uint16_t>(id2), tmem_e);

    SM100_MMA_F16BF16_2x1SM_SS_SPARSE<a_type, b_type, c_type,
                              M, N, a_major, b_major,
                              a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, tmem_e);
  }

  // Construct an executable sparse MMA_traits with sp into set.
  template <class TE, class ELayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS_SPARSE<a_type, b_type, c_type,
                                       M, N, a_major, b_major,
                                       a_neg, b_neg>, uint32_t>
  with(Tensor<TE, ELayout> const& E, uint32_t id2 = 0) const {
    // Check sparse_ptr, check sparsity, check shape/layout?
    uint32_t tmem_e_addr = raw_pointer_cast(E.data());
    return {accumulate_, {tmem_e_addr}, idesc_};
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_S8_SS<a_type, b_type, c_type,
                                M, N, a_major, b_major,
                                c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 8, "SM100_MMA_S8_SS supports 8bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  // Logical shape-K is always 256bits, transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, UMMA::ScaleIn::One, UMMA::ScaleIn::One, c_sat>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_S8_SS<a_type, b_type, c_type,
                  M, N, a_major, b_major,
                  c_sat>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_S8_TS<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 8, "SM100_MMA_S8_TS supports 8bit types");

  using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  // Logical shape-K is always 256 bits; transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, UMMA::ScaleIn::One, UMMA::ScaleIn::One, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_S8_TS<a_type, b_type, c_type,
                  M, N,
                  a_major, b_major,
                  a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::Saturate c_sat,
          class... sparse_args>
struct MMA_Traits<SM100_MMA_S8_SS_SPARSE<a_type, b_type, c_type,
                                       M, N, a_major, b_major,
                                       c_sat>, sparse_args...>
{
  using ValTypeD = c_type;
  static_assert(sizeof(a_type) == 1);
  using ValTypeA = sparse_elem<2, a_type>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 8, "SM100_MMA_S8_SS_SPARSE supports 8bit types");

  using FrgTypeA = UMMA::sparse_smem_desc<a_major>;
  using FrgTypeE = UMMA::tmem_e_frg<a_type>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  // SparseMma consume double mma-k bits
  static constexpr int K = 512 / cute::sizeof_bits<a_type>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  // uint32_t tmem_e: Metadata tmem address.
  cute::tuple<sparse_args...> sparse_args_;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, UMMA::ScaleIn::One, UMMA::ScaleIn::One, c_sat, true>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_same<cute::tuple<sparse_args...>, cute::tuple<uint32_t>>::value,
                  "Params must be set via .with()?");
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());

    uint32_t tmem_e = get<0>(traits.sparse_args_);
    uint32_t id2 = 0;

    uint64_t idesc = UMMA::make_runtime_instr_desc<true>(traits.idesc_, static_cast<uint16_t>(id2), tmem_e);

    SM100_MMA_S8_SS_SPARSE<a_type, b_type, c_type,
                         M, N, a_major, b_major,
                         c_sat>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, tmem_e);
  }

  // Construct an executable sparse MMA_traits with sp into set.
  template <class TE, class ELayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_S8_SS_SPARSE<a_type, b_type, c_type,
                                  M, N, a_major, b_major,
                                  c_sat>, uint32_t>
  with(Tensor<TE, ELayout> const& E, uint32_t id2 = 0) const {
    // Check sparse_ptr, check sparsity, check shape/layout?
    uint32_t tmem_e_addr = raw_pointer_cast(E.data());
    return {accumulate_, {tmem_e_addr}, idesc_};
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_S8_2x1SM_SS<a_type, b_type, c_type,
                                      M, N, a_major, b_major,
                                      c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 8, "SM100_MMA_S8_2x1SM_SS supports 8bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instructions's K extent is always 256bits, convert to units of element
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, UMMA::ScaleIn::One, UMMA::ScaleIn::One, c_sat>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_S8_2x1SM_SS<a_type, b_type, c_type,
                       M, N, a_major, b_major,
                       c_sat>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_S8_2x1SM_TS<a_type, b_type, c_type,
                                     M, N,
                                     a_major, b_major,
                                     a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 8, "SM100_MMA_S8_2x1SM_TS supports 8bit types");

  using FrgTypeA = UMMA::tmem_frg_2sm<a_type, a_type, UMMA::TmemAllocMode::Duplicated>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instructions' K extent is always 256 bits; convert to units of element
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
      a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_S8_2x1SM_TS<a_type, b_type, c_type,
                        M, N,
                        a_major, b_major,
                        a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::Saturate c_sat,
          class... sparse_args>
struct MMA_Traits<SM100_MMA_S8_2x1SM_SS_SPARSE<a_type, b_type, c_type,
                                             M, N, a_major, b_major,
                                             c_sat>, sparse_args...>
{
  using ValTypeD = c_type;
  static_assert(sizeof(a_type) == 1);
  using ValTypeA = sparse_elem<2, a_type>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 8, "SM100_MMA_S8_2x1SM_SS_SPARSE supports 8bit types");

  using FrgTypeA = UMMA::sparse_smem_desc<a_major>;
  using FrgTypeE = UMMA::tmem_e_frg<a_type>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // SparseMma consume double mma-k bits
  constexpr static int K = 512 / cute::sizeof_bits<a_type>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  // uint32_t tmem_e: Metadata tmem address.
  cute::tuple<sparse_args...> sparse_args_;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, UMMA::ScaleIn::One, UMMA::ScaleIn::One, c_sat, true>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_same<cute::tuple<sparse_args...>, cute::tuple<uint32_t>>::value,
                  "Params must be set via .with()?");
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());

    uint32_t tmem_e = get<0>(traits.sparse_args_);
    uint16_t id2    = 0u;

    uint64_t idesc = UMMA::make_runtime_instr_desc<true>(traits.idesc_, id2, tmem_e);

    SM100_MMA_S8_2x1SM_SS_SPARSE<a_type, b_type, c_type,
                              M, N, a_major, b_major,
                              c_sat>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, tmem_e);
  }

  // Construct an executable sparse MMA_traits with sp into set.
  template <class TE, class ELayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_S8_2x1SM_SS_SPARSE<a_type, b_type, c_type,
                                       M, N, a_major, b_major,
                                       c_sat>, uint32_t>
  with(Tensor<TE, ELayout> const& E, uint32_t id2 = 0) const {
    // Check sparse_ptr, check sparsity, check shape/layout?
    uint32_t tmem_e_addr = raw_pointer_cast(E.data());
    return {accumulate_, {tmem_e_addr}, idesc_};
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F8F6F4_SS, a_type, b_type, c_type,
                  cute::C<M>, cute::C<N>,
                  cute::integral_constant<UMMA::Major, a_major>,
                  cute::integral_constant<UMMA::Major, b_major>,
                  cute::integral_constant<UMMA::ScaleIn, a_neg>,
                  cute::integral_constant<UMMA::ScaleIn, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8, "SM100_MMA_F8F6F4_SS supports types with leq 8bit types");
  static_assert(M == 64 || M == 128, "SM100_MMA_F8F6F4_SS M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert(((b_major == UMMA::Major::K) && ((N % 8 == 0) && (8 <= N) && (N <= 256))) ||
                ((b_major == UMMA::Major::MN) && ((N % 16 == 0) && (16 <= N) && (N <= 256))), 
                "SM100_MMA_F8F6F4_SS N-mode size should be a multiple of 8 between 8 and 256 when B is K major. \
                 SM100_MMA_F8F6F4_SS N-mode size should be a multiple of 16 between 16 and 256 when B is MN major.");
  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  static_assert(sizeof_bits_v<ValTypeA> <= sizeof_bits_v<uint8_t> &&
                sizeof_bits_v<ValTypeB> <= sizeof_bits_v<uint8_t>);

  // Logical shape-K is always 256bits, transform to units of elements
  constexpr static int K = 32;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F8F6F4_SS::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};


template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_MXF8F6F4_SS<a_type, b_type, c_type, sf_type,
                                M, N, a_major, b_major,
                                a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8, "SM100_MMA_MXF8F6F4_SS supports types with leq 8bit types");

  // Logical shape-K is always 256bits, transform to units of elements
  constexpr static int K = 32;
  constexpr static int SFVecSize = 32;

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;
  using FrgTypeSFA = UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, true>;
  using FrgTypeSFB = UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, false>;

  static_assert(sizeof_bits_v<ValTypeA> <= sizeof_bits_v<uint8_t> &&
                sizeof_bits_v<ValTypeB> <= sizeof_bits_v<uint8_t>);


  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using MMA_ScaleFactor = SM100_MMA_MXF8F6F4_SS<a_type, b_type, c_type, sf_type,
                                M, (round_up(N, 128)), a_major, b_major,
                                a_neg, b_neg>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  uint32_t tsfa_addr_ = 0;
  uint32_t tsfb_addr_ = 0;

  UMMA::InstrDescriptorBlockScaled idesc_ = UMMA::make_instr_desc_block_scaled<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major, a_neg, b_neg>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc_block_scaled<>(traits.idesc_, traits.tsfa_addr_, traits.tsfb_addr_);

    SM100_MMA_MXF8F6F4_SS<a_type, b_type, c_type, sf_type,
                  M, N,
                  a_major, b_major,
                  a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, traits.tsfa_addr_, traits.tsfb_addr_);
  }

  // Construct an executable MMA_traits with sp into set.
  template <class TSFA, class TSFALayout, class TSFB, class TSFBLayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_MXF8F6F4_SS<a_type, b_type, c_type, sf_type,
                              M, N, a_major, b_major, a_neg, b_neg>>
  with(UMMA::ScaleOut accumulate, Tensor<TSFA, TSFALayout> const& SFA, Tensor<TSFB, TSFBLayout> const& SFB) const {
    uint32_t tmem_sfa_addr = raw_pointer_cast(SFA.data());
    uint32_t tmem_sfb_addr = raw_pointer_cast(SFB.data());
    return {accumulate, tmem_sfa_addr, tmem_sfb_addr, idesc_};
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          class... sparse_args>
struct MMA_Traits<SM100_MMA_MXF8F6F4_SS_SPARSE<a_type, b_type, c_type, sf_type,
                                          M, N, a_major, b_major,
                                          a_neg, b_neg>, sparse_args...>
{
  using ValTypeD = c_type;
  using ValTypeA = sparse_elem<2, a_type>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8, "SM100_MMA_MXF8F6F4_SS_SPARSE supports types with leq 8bit types");

  // Logical shape-K is always 512bits, transform to units of elements
  constexpr static int K = 64;
  constexpr static int SFVecSize = 64;

  using FrgTypeA = UMMA::sparse_smem_desc<a_major>;
  using FrgTypeE = UMMA::tmem_e_frg<uint8_t>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;
  using FrgTypeSFA = UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, true>;
  using FrgTypeSFB = UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, false>;

  static_assert(sizeof_bits_v<ValTypeA> <= sizeof_bits_v<uint8_t> &&
                sizeof_bits_v<ValTypeB> <= sizeof_bits_v<uint8_t>);


  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using MMA_ScaleFactor = SM100_MMA_MXF8F6F4_SS_SPARSE<a_type, b_type, c_type, sf_type,
                                M, (round_up(N, 128)), a_major, b_major,
                                a_neg, b_neg>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  uint32_t tsfa_addr_ = 0;
  uint32_t tsfb_addr_ = 0;

  // uint32_t tmem_e: Metadata tmem address.
  cute::tuple<sparse_args...> sparse_args_;

  UMMA::InstrDescriptorBlockScaled idesc_ = UMMA::make_instr_desc_block_scaled<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major, a_neg, b_neg, true>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_same<cute::tuple<sparse_args...>, cute::tuple<uint32_t>>::value,
              "Params must be set via .with()?");
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());

    uint32_t tmem_e = get<0>(traits.sparse_args_);
    uint16_t id2    = 0u;

    uint64_t idesc = UMMA::make_runtime_instr_desc_block_scaled<true>(traits.idesc_, traits.tsfa_addr_, traits.tsfb_addr_, id2, tmem_e);

    SM100_MMA_MXF8F6F4_SS_SPARSE<a_type, b_type, c_type, sf_type,
                            M, N,
                            a_major, b_major,
                            a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, traits.tsfa_addr_, traits.tsfb_addr_, tmem_e);
  }

  // Construct an executable MMA_traits with sp into set.
  template <class TE, class TELayout, class TSFA, class TSFALayout, class TSFB, class TSFBLayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_MXF8F6F4_SS_SPARSE<a_type, b_type, c_type, sf_type,
                              M, N, a_major, b_major, a_neg, b_neg>, uint32_t>
  with(UMMA::ScaleOut accumulate, Tensor<TE, TELayout> const& E, Tensor<TSFA, TSFALayout> const& SFA, Tensor<TSFB, TSFBLayout> const& SFB) const {
    uint32_t tmem_e_addr = raw_pointer_cast(E.data());
    uint32_t tmem_sfa_addr = raw_pointer_cast(SFA.data());
    uint32_t tmem_sfb_addr = raw_pointer_cast(SFB.data());
    return {accumulate, tmem_sfa_addr, tmem_sfb_addr, {tmem_e_addr}, idesc_};
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F8F6F4_TS<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8, "SM100_MMA_F8F6F4_TS supports types with leq 8bit types");

  using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  static_assert(sizeof_bits_v<ValTypeA> <= sizeof_bits_v<uint8_t> &&
                sizeof_bits_v<ValTypeB> <= sizeof_bits_v<uint8_t>);
  // Logical shape-K is always 256 bits; transform to units of elements
  static constexpr int K = 32;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F8F6F4_TS<a_type, b_type, c_type,
                  M, N,
                  a_major, b_major,
                  a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          class... sparse_args>
struct MMA_Traits<SM100_MMA_F8F6F4_SS_SPARSE<a_type, b_type, c_type,
                                       M, N, a_major, b_major,
                                       a_neg, b_neg>, sparse_args...>
{
  using ValTypeD = c_type;
  static_assert(sizeof(a_type) == 1);
  using ValTypeA = sparse_elem<2, a_type>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8, "SM100_MMA_F8F6F4_SS_SPARSE supports types with leq 8bit types");

  using FrgTypeA = UMMA::sparse_smem_desc<a_major>;
  using FrgTypeE = UMMA::tmem_e_frg<uint8_t>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  // SparseMma consume double mma-k bits
  static constexpr int K = 512 / cute::sizeof_bits<uint8_t>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  // uint32_t tmem_e: Metadata tmem address.
  cute::tuple<sparse_args...> sparse_args_;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, UMMA::Saturate::False, true>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_same<cute::tuple<sparse_args...>, cute::tuple<uint32_t>>::value,
                  "Params must be set via .with()?");
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());

    uint32_t tmem_e = get<0>(traits.sparse_args_);
    uint16_t id2    = 0u;

    uint64_t idesc = UMMA::make_runtime_instr_desc<true>(traits.idesc_, id2, tmem_e);

    SM100_MMA_F8F6F4_SS_SPARSE<a_type, b_type, c_type,
                         M, N, a_major, b_major,
                         a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, tmem_e);
  }

  // Construct an executable sparse MMA_traits with sp into set.
  template <class TE, class ELayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_F8F6F4_SS_SPARSE<a_type, b_type, c_type,
                                  M, N, a_major, b_major,
                                  a_neg, b_neg>, uint32_t>
  with(Tensor<TE, ELayout> const& E, uint32_t id2 = 0) const {
    // Check sparse_ptr, check sparsity, check shape/layout?
    uint32_t tmem_e_addr = raw_pointer_cast(E.data());
    return {accumulate_, {tmem_e_addr}, idesc_};
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F8F6F4_2x1SM_SS, a_type, b_type, c_type,
                  cute::C<M>, cute::C<N>,
                  cute::integral_constant<UMMA::Major, a_major>,
                  cute::integral_constant<UMMA::Major, b_major>,
                  cute::integral_constant<UMMA::ScaleIn, a_neg>,
                  cute::integral_constant<UMMA::ScaleIn, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8, "SM100_MMA_F8F6F4_2x1SM_SS supports types with leq 8bit types");
  static_assert(M == 128 || M == 256, "SM100_MMA_F8F6F4_2x1SM_SS M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert(((b_major == UMMA::Major::K) && ((N % 16 == 0) && (16 <= N) && (N <= 256))) ||
                ((b_major == UMMA::Major::MN) && ((N % 32 == 0) && (32 <= N) && (N <= 256))), 
                "SM100_MMA_F8F6F4_2x1SM_SS N-mode size should be a multiple of 16 between 16 and 256 when B is K major. \
                 SM100_MMA_F8F6F4_2x1SM_SS N-mode size should be a multiple of 32 between 32 and 256 when B is MN major.");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  static_assert(sizeof_bits_v<ValTypeA> <= sizeof_bits_v<uint8_t> &&
                sizeof_bits_v<ValTypeB> <= sizeof_bits_v<uint8_t>);
  // Size of instructions's K extent is always 256bits, convert to units of element
  constexpr static int K = 32;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F8F6F4_2x1SM_SS::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F8F6F4_2x1SM_TS<a_type, b_type, c_type,
                                     M, N,
                                     a_major, b_major,
                                     a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8, "SM100_MMA_F8F6F4_2x1SM_TS supports types with leq 8bit types");

  using FrgTypeA = UMMA::tmem_frg_2sm<a_type, a_type, UMMA::TmemAllocMode::Duplicated>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  static_assert(sizeof_bits_v<ValTypeA> <= sizeof_bits_v<uint8_t> &&
                sizeof_bits_v<ValTypeB> <= sizeof_bits_v<uint8_t>);
  // Size of instructions' K extent is always 256 bits; convert to units of element
  constexpr static int K = 32;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F8F6F4_2x1SM_TS<a_type, b_type, c_type,
                       M, N,
                       a_major, b_major,
                       a_neg, b_neg, c_sat>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          class... sparse_args>
struct MMA_Traits<SM100_MMA_F8F6F4_2x1SM_SS_SPARSE<a_type, b_type, c_type,
                                             M, N, a_major, b_major,
                                             a_neg, b_neg>, sparse_args...>
{
  using ValTypeD = c_type;
  static_assert(sizeof(a_type) == 1);
  using ValTypeA = sparse_elem<2, a_type>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8, "SM100_MMA_F8F6F4_2x1SM_SS_SPARSE supports types with leq 8bit types");

  using FrgTypeA = UMMA::sparse_smem_desc<a_major>;
  using FrgTypeE = UMMA::tmem_e_frg<uint8_t>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // SparseMma consume double mma-k bits
  constexpr static int K = 512 / cute::sizeof_bits<uint8_t>::value;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  // uint32_t tmem_e: Metadata tmem address.
  cute::tuple<sparse_args...> sparse_args_;

  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, UMMA::Saturate::False, true>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_same<cute::tuple<sparse_args...>, cute::tuple<uint32_t>>::value,
                  "Params must be set via .with()?");
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());

    uint32_t tmem_e = get<0>(traits.sparse_args_);
    uint16_t id2    = 0u;

    uint64_t idesc = UMMA::make_runtime_instr_desc<true>(traits.idesc_, id2, tmem_e);

    SM100_MMA_F8F6F4_2x1SM_SS_SPARSE<a_type, b_type, c_type,
                              M, N, a_major, b_major,
                              a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, tmem_e);
  }

  // Construct an executable sparse MMA_traits with sp into set.
  template <class TE, class ELayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_F8F6F4_2x1SM_SS_SPARSE<a_type, b_type, c_type,
                                       M, N, a_major, b_major,
                                       a_neg, b_neg>, uint32_t>
  with(Tensor<TE, ELayout> const& E, uint32_t id2 = 0) const {
    // Check sparse_ptr, check sparsity, check shape/layout?
    uint32_t tmem_e_addr = raw_pointer_cast(E.data());
    return {accumulate_, {tmem_e_addr}, idesc_};
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_MXF8F6F4_2x1SM_SS<a_type, b_type, c_type, sf_type,
                                      M, N, a_major, b_major,
                                      a_neg, b_neg>>
{
  using ValTypeD   = c_type;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = c_type;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8, "SM100_MMA_MXF8F6F4_2x1SM_SS supports types with leq 8bit types");

  using FrgTypeA   = UMMA::smem_desc<a_major>;
  using FrgTypeB   = UMMA::smem_desc<b_major>;
  using FrgTypeC   = UMMA::tmem_frg_2sm<c_type>;

  // Logical shape-K is always 256bits, transform to units of elements
  constexpr static int K = 32;
  constexpr static int SFVecSize = 32;

  constexpr static UMMA::TmemAllocMode TmemAlloc = M == 128 ?
      UMMA::TmemAllocMode::ScaleFactorDuplicated2by2 : UMMA::TmemAllocMode::ScaleFactorDuplicated4by1;
  using FrgTypeSFA = UMMA::tmem_sf_frg<sf_type, SFVecSize, 2,  true, TmemAlloc>;
  using FrgTypeSFB = UMMA::tmem_sf_frg<sf_type, SFVecSize, 2, false, TmemAlloc>;

  static_assert(sizeof_bits_v<ValTypeA> <= sizeof_bits_v<uint8_t> &&
                sizeof_bits_v<ValTypeB> <= sizeof_bits_v<uint8_t>);

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using MMA_ScaleFactor = SM100_MMA_MXF8F6F4_SS<a_type, b_type, c_type, sf_type,
                                (M/2 > 64 ? M/2 : M), (round_up(N, 128)), a_major, b_major,
                                a_neg, b_neg>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  uint32_t tsfa_addr_ = 0;
  uint32_t tsfb_addr_ = 0;

  UMMA::InstrDescriptorBlockScaled idesc_ = UMMA::make_instr_desc_block_scaled<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major, a_neg, b_neg>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc_block_scaled<>(traits.idesc_, traits.tsfa_addr_, traits.tsfb_addr_);

    SM100_MMA_MXF8F6F4_2x1SM_SS<a_type, b_type, c_type, sf_type,
                          M, N,
                          a_major, b_major,
                          a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, traits.tsfa_addr_, traits.tsfb_addr_);
  }

  // Construct an executable MMA_traits with sp into set.
  template <class TSFA, class TSFALayout, class TSFB, class TSFBLayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_MXF8F6F4_2x1SM_SS<a_type, b_type, c_type, sf_type,
                                M, N, a_major, b_major, a_neg, b_neg>>
  with(UMMA::ScaleOut accumulate, Tensor<TSFA, TSFALayout> const& SFA, Tensor<TSFB, TSFBLayout> const& SFB) const {
    uint32_t tmem_sfa_addr = raw_pointer_cast(SFA.data());
    uint32_t tmem_sfb_addr = raw_pointer_cast(SFB.data());
    return {accumulate, tmem_sfa_addr, tmem_sfb_addr, idesc_};
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          class... sparse_args>
struct MMA_Traits<SM100_MMA_MXF8F6F4_2x1SM_SS_SPARSE<a_type, b_type, c_type, sf_type,
                                      M, N, a_major, b_major,
                                      a_neg, b_neg>, sparse_args...>
{
  using ValTypeD = c_type;
  static_assert(sizeof(a_type) == 1);
  using ValTypeA = sparse_elem<2, a_type>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8, "SM100_MMA_MXF8F6F4_2x1SM_SS_SPARSE supports types with leq 8bit types");

  using FrgTypeA = UMMA::sparse_smem_desc<a_major>;
  using FrgTypeE = UMMA::tmem_e_frg<uint8_t>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // SparseMma consume double mma-k bits
  constexpr static int K = 64;
  constexpr static int SFVecSize = 64;

  constexpr static UMMA::TmemAllocMode TmemAlloc = M == 128 ?
      UMMA::TmemAllocMode::ScaleFactorDuplicated2by2 : UMMA::TmemAllocMode::ScaleFactorDuplicated4by1;
  using FrgTypeSFA = UMMA::tmem_sf_frg<sf_type, SFVecSize, 2,  true, TmemAlloc>;
  using FrgTypeSFB = UMMA::tmem_sf_frg<sf_type, SFVecSize, 2, false, TmemAlloc>;

  static_assert(sizeof_bits_v<ValTypeA> <= sizeof_bits_v<uint8_t> &&
                sizeof_bits_v<ValTypeB> <= sizeof_bits_v<uint8_t>);

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using MMA_ScaleFactor = SM100_MMA_MXF8F6F4_SS_SPARSE<a_type, b_type, c_type, sf_type,
                                (M/2 > 64 ? M/2 : M), (round_up(N, 128)), a_major, b_major,
                                a_neg, b_neg>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  uint32_t tsfa_addr_ = 0;
  uint32_t tsfb_addr_ = 0;

  // uint32_t tmem_e: Metadata tmem address.
  cute::tuple<sparse_args...> sparse_args_;

  UMMA::InstrDescriptorBlockScaled idesc_ = UMMA::make_instr_desc_block_scaled<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major, a_neg, b_neg, true>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());

    uint32_t tmem_e = get<0>(traits.sparse_args_);
    uint16_t id2    = 0u;

    uint64_t idesc = UMMA::make_runtime_instr_desc_block_scaled<true>(traits.idesc_, traits.tsfa_addr_, traits.tsfb_addr_, id2, tmem_e);

    SM100_MMA_MXF8F6F4_2x1SM_SS_SPARSE<a_type, b_type, c_type, sf_type,
                          M, N,
                          a_major, b_major,
                          a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, traits.tsfa_addr_, traits.tsfb_addr_, tmem_e);
  }

  // Construct an executable MMA_traits with sp into set.
  template <class TE, class TELayout, class TSFA, class TSFALayout, class TSFB, class TSFBLayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_MXF8F6F4_2x1SM_SS_SPARSE<a_type, b_type, c_type, sf_type,
                                M, N, a_major, b_major, a_neg, b_neg>, uint32_t>
  with(UMMA::ScaleOut accumulate, Tensor<TE, TELayout> const& E, Tensor<TSFA, TSFALayout> const& SFA, Tensor<TSFB, TSFBLayout> const& SFB) const {
    uint32_t tmem_e_addr = raw_pointer_cast(E.data());
    uint32_t tmem_sfa_addr = raw_pointer_cast(SFA.data());
    uint32_t tmem_sfb_addr = raw_pointer_cast(SFB.data());
    return {accumulate, tmem_sfa_addr, tmem_sfb_addr, {tmem_e_addr}, idesc_};
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, int VS, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_MXF4_SS<a_type, b_type, c_type, sf_type,
                                M, N, VS, a_major, b_major,
                                a_neg, b_neg>>
{
  using ValTypeD   = c_type;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = c_type;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;
  static_assert(cute::sizeof_bits_v<a_type> == 4 && cute::sizeof_bits_v<b_type> == 4, "SM100_MMA_MXF4_SS supports 4bit types");

  // Logical shape-K is always 256bits, transform to units of elements
  constexpr static int K = 64;
  constexpr static int SFVecSize = VS;

  using FrgTypeA   = UMMA::smem_desc<a_major>;
  using FrgTypeB   = UMMA::smem_desc<b_major>;
  using FrgTypeC   = UMMA::tmem_frg_1sm<c_type>;
  using FrgTypeSFA = UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, true>;
  using FrgTypeSFB = UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, false>;

  static_assert((VS == 32 && ((is_same_v<a_type, cutlass::float_e2m1_t> || is_same_v<a_type, cutlass::type_erased_dynamic_float4_t>) &&
                              (is_same_v<b_type, cutlass::float_e2m1_t> || is_same_v<b_type, cutlass::type_erased_dynamic_float4_t>))
                          &&   is_same_v<sf_type, cutlass::float_ue8m0_t>)
             || (VS == 16),
       "2x mode (VectorSize=32) only supports a_type and b_type=float_e2m1_t or cutlass::type_erased_dynamic_float4_t and sf_type=ue8m0_t");

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using MMA_ScaleFactor = SM100_MMA_MXF4_SS<a_type, b_type, c_type, sf_type,
                                M, (round_up(N, 128)), VS, a_major, b_major,
                                a_neg, b_neg>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  uint32_t tsfa_addr_ = 0;
  uint32_t tsfb_addr_ = 0;

  UMMA::InstrDescriptorBlockScaled idesc_ = UMMA::make_instr_desc_block_scaled<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major, a_neg, b_neg>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc_block_scaled<>(traits.idesc_, traits.tsfa_addr_, traits.tsfb_addr_);

    SM100_MMA_MXF4_SS<a_type, b_type, c_type, sf_type,
                  M, N, VS,
                  a_major, b_major,
                  a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, traits.tsfa_addr_, traits.tsfb_addr_);
  }

  // Construct an executable sparse MMA_traits with sp into set.
  template <class TSFA, class TSFALayout, class TSFB, class TSFBLayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_MXF4_SS<a_type, b_type, c_type, sf_type,
                              M, N, VS, a_major, b_major, a_neg, b_neg>>
  with(UMMA::ScaleOut accumulate, Tensor<TSFA, TSFALayout> const& SFA, Tensor<TSFB, TSFBLayout> const& SFB) const {
    uint32_t tmem_sfa_addr = raw_pointer_cast(SFA.data());     // Move to a CoupledTensor rather than a .with()?
    uint32_t tmem_sfb_addr = raw_pointer_cast(SFB.data());     // Move to a CoupledTensor rather than a .with()?
    return {accumulate, tmem_sfa_addr, tmem_sfb_addr, idesc_};
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, int VS, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          class... sparse_args>
struct MMA_Traits<SM100_MMA_MXF4NVF4_SS_SPARSE<a_type, b_type, c_type, sf_type,
                                M, N, VS, a_major, b_major,
                                a_neg, b_neg>, sparse_args...>
{
  using ValTypeD   = c_type;
  using ValTypeA   = sparse_elem<4, uint8_t>;
  using ValTypeE   = sparse_elem<16, uint8_t>;
  using ValTypeB   = b_type;
  using ValTypeC   = c_type;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;
  static_assert(cute::sizeof_bits_v<a_type> == 4 && cute::sizeof_bits_v<b_type> == 4, "SM100_MMA_MXF4NVF4_SS_SPARSE supports 4bit types");

  // Logical shape-K is always 256bits, transform to units of elements
  constexpr static int K = 128;
  constexpr static int SFVecSize = VS;

  using FrgTypeA   = UMMA::sparse_smem_desc<a_major>;
  using FrgTypeE   = UMMA::tmem_e_frg<a_type>;
  using FrgTypeB   = UMMA::smem_desc<b_major>;
  using FrgTypeC   = UMMA::tmem_frg_1sm<c_type>;
  using FrgTypeSFA = UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, true>;
  using FrgTypeSFB = UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, false>;

  static_assert((VS == 64 && ((is_same_v<a_type, cutlass::float_e2m1_t> || is_same_v<a_type, cutlass::type_erased_dynamic_float4_t>) &&
                              (is_same_v<b_type, cutlass::float_e2m1_t> || is_same_v<b_type, cutlass::type_erased_dynamic_float4_t>))
                          &&   is_same_v<sf_type, cutlass::float_ue8m0_t>)
             || (VS == 32),
       "2x mode (VectorSize=64) only supports a_type and b_type=float_e2m1_t or cutlass::type_erased_dynamic_float4_t and sf_type=ue8m0_t");

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using MMA_ScaleFactor = SM100_MMA_MXF4NVF4_SS_SPARSE<a_type, b_type, c_type, sf_type,
                                M, (round_up(N, 128)), VS, a_major, b_major,
                                a_neg, b_neg>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  uint32_t tsfa_addr_ = 0;
  uint32_t tsfb_addr_ = 0;

  // uint32_t tmem_e: Metadata tmem address.
  cute::tuple<sparse_args...> sparse_args_;

  UMMA::InstrDescriptorBlockScaled idesc_ = UMMA::make_instr_desc_block_scaled<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major, a_neg, b_neg, true>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());

    uint32_t tmem_e = get<0>(traits.sparse_args_);
    uint16_t id2    = 0u;

    uint64_t idesc = UMMA::make_runtime_instr_desc_block_scaled<true>(traits.idesc_, traits.tsfa_addr_, traits.tsfb_addr_, id2, tmem_e);

    SM100_MMA_MXF4NVF4_SS_SPARSE<a_type, b_type, c_type, sf_type,
                  M, N, VS,
                  a_major, b_major,
                  a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, traits.tsfa_addr_, traits.tsfb_addr_, tmem_e);
  }

  // Construct an executable sparse MMA_traits with sp into set.
  template <class TE, class TELayout, class TSFA, class TSFALayout, class TSFB, class TSFBLayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_MXF4NVF4_SS_SPARSE<a_type, b_type, c_type, sf_type,
                              M, N, VS, a_major, b_major, a_neg, b_neg>, uint32_t>
  with(UMMA::ScaleOut accumulate, Tensor<TE, TELayout> const& E, Tensor<TSFA, TSFALayout> const& SFA, Tensor<TSFB, TSFBLayout> const& SFB) const {
    uint32_t tmem_e_addr   = raw_pointer_cast(E.data());
    uint32_t tmem_sfa_addr = raw_pointer_cast(SFA.data());     // Move to a CoupledTensor rather than a .with()?
    uint32_t tmem_sfb_addr = raw_pointer_cast(SFB.data());     // Move to a CoupledTensor rather than a .with()?
    return {accumulate, tmem_sfa_addr, tmem_sfb_addr, {tmem_e_addr}, idesc_};
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, int VS, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_MXF4_2x1SM_SS<a_type, b_type, c_type, sf_type,
                                      M, N, VS, a_major, b_major,
                                      a_neg, b_neg>>
{
  using ValTypeD   = c_type;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = c_type;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;
  static_assert(cute::sizeof_bits_v<a_type> == 4 && cute::sizeof_bits_v<b_type> == 4, "SM100_MMA_MXF4_2x1SM_SS supports 4bit types");

  // Logical shape-K is always 256bits, transform to units of elements
  constexpr static int K = 64;
  constexpr static int SFVecSize = VS;

  using FrgTypeA   = UMMA::smem_desc<a_major>;
  using FrgTypeB   = UMMA::smem_desc<b_major>;
  using FrgTypeC   = UMMA::tmem_frg_2sm<c_type>;

  constexpr static UMMA::TmemAllocMode TmemAlloc = M == 128 ?
      UMMA::TmemAllocMode::ScaleFactorDuplicated2by2 : UMMA::TmemAllocMode::ScaleFactorDuplicated4by1;
  using FrgTypeSFA = UMMA::tmem_sf_frg<sf_type, SFVecSize, 2,  true, TmemAlloc>;
  using FrgTypeSFB = UMMA::tmem_sf_frg<sf_type, SFVecSize, 2, false, TmemAlloc>;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using MMA_ScaleFactor = SM100_MMA_MXF4_SS<a_type, b_type, c_type, sf_type,
                                (M/2 > 64 ? M/2 : M), (round_up(N, 128)), VS, a_major, b_major,
                                a_neg, b_neg>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  uint32_t tsfa_addr_ = 0;
  uint32_t tsfb_addr_ = 0;

  UMMA::InstrDescriptorBlockScaled idesc_ = UMMA::make_instr_desc_block_scaled<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major, a_neg, b_neg>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc_block_scaled<>(traits.idesc_, traits.tsfa_addr_, traits.tsfb_addr_);

    SM100_MMA_MXF4_2x1SM_SS<a_type, b_type, c_type, sf_type,
                          M, N, VS,
                          a_major, b_major,
                          a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, traits.tsfa_addr_, traits.tsfb_addr_);
  }

  // Construct an executable sparse MMA_traits with sp into set.
  template <class TSFA, class TSFALayout, class TSFB, class TSFBLayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_MXF4_2x1SM_SS<a_type, b_type, c_type, sf_type,
                                M, N, VS, a_major, b_major, a_neg, b_neg>>
  with(UMMA::ScaleOut accumulate, Tensor<TSFA, TSFALayout> const& SFA, Tensor<TSFB, TSFBLayout> const& SFB) const {
    // Check sparse_ptr, check sparsity, check shape/layout?
    uint32_t tmem_sfa_addr = raw_pointer_cast(SFA.data());     // Move to a CoupledTensor rather than a .with()?
    uint32_t tmem_sfb_addr = raw_pointer_cast(SFB.data());     // Move to a CoupledTensor rather than a .with()?
    return {accumulate, tmem_sfa_addr, tmem_sfb_addr, idesc_};
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, int VS, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          class... sparse_args>
struct MMA_Traits<SM100_MMA_MXF4NVF4_2x1SM_SS_SPARSE<a_type, b_type, c_type, sf_type,
                                M, N, VS, a_major, b_major,
                                a_neg, b_neg>, sparse_args...>
{
  using ValTypeD   = c_type;
  using ValTypeA = sparse_elem<4, uint8_t>;
  using ValTypeE = sparse_elem<16, uint8_t>;
  using ValTypeB   = b_type;
  using ValTypeC   = c_type;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;
  static_assert(cute::sizeof_bits_v<a_type> == 4 && cute::sizeof_bits_v<b_type> == 4, "SM100_MMA_MXF4NVF4_2x1SM_SS_SPARSE supports 4bit types");

  // Logical shape-K is always 256bits, transform to units of elements
  constexpr static int K = 128;
  constexpr static int SFVecSize = VS;

  constexpr static UMMA::TmemAllocMode TmemAlloc = M == 128 ?
      UMMA::TmemAllocMode::ScaleFactorDuplicated2by2 : UMMA::TmemAllocMode::ScaleFactorDuplicated4by1;
  using FrgTypeA   = UMMA::sparse_smem_desc<a_major>;
  // using FrgTypeE = UMMA::tmem_e_frg<uint8_t>;
  using FrgTypeE   = UMMA::tmem_e_frg<a_type>;
  using FrgTypeB   = UMMA::smem_desc<b_major>;
  using FrgTypeC   = UMMA::tmem_frg_2sm<c_type>;
  using FrgTypeSFA = UMMA::tmem_sf_frg<sf_type, SFVecSize, 2,  true, TmemAlloc>;
  using FrgTypeSFB = UMMA::tmem_sf_frg<sf_type, SFVecSize, 2, false, TmemAlloc>;

  static_assert((VS == 64 && ((is_same_v<a_type, cutlass::float_e2m1_t> || is_same_v<a_type, cutlass::type_erased_dynamic_float4_t>) &&
                              (is_same_v<b_type, cutlass::float_e2m1_t> || is_same_v<b_type, cutlass::type_erased_dynamic_float4_t>))
                          &&   is_same_v<sf_type, cutlass::float_ue8m0_t>)
             || (VS == 32),
       "2x mode (VectorSize=64) only supports a_type and b_type=float_e2m1_t or cutlass::type_erased_dynamic_float4_t and sf_type=ue8m0_t");

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using MMA_ScaleFactor = SM100_MMA_MXF4NVF4_SS_SPARSE<a_type, b_type, c_type, sf_type,
                                (M/2 > 64 ? M/2 : M), (round_up(N, 128)), VS, a_major, b_major,
                                a_neg, b_neg>;


  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  uint32_t tsfa_addr_ = 0;
  uint32_t tsfb_addr_ = 0;

  // uint32_t tmem_e: Metadata tmem address.
  cute::tuple<sparse_args...> sparse_args_;

  UMMA::InstrDescriptorBlockScaled idesc_ = UMMA::make_instr_desc_block_scaled<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major, a_neg, b_neg, true>();

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());

    uint32_t tmem_e = get<0>(traits.sparse_args_);
    uint16_t id2    = 0u;

    uint64_t idesc = UMMA::make_runtime_instr_desc_block_scaled<true>(traits.idesc_, traits.tsfa_addr_, traits.tsfb_addr_, id2, tmem_e);

    SM100_MMA_MXF4NVF4_2x1SM_SS_SPARSE<a_type, b_type, c_type, sf_type,
                  M, N, VS,
                  a_major, b_major,
                  a_neg, b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc, traits.tsfa_addr_, traits.tsfb_addr_, tmem_e);
  }

  // Construct an executable sparse MMA_traits with sp into set.
  template <class TE, class TELayout, class TSFA, class TSFALayout, class TSFB, class TSFBLayout>
  CUTE_HOST_DEVICE constexpr
  MMA_Traits<SM100_MMA_MXF4NVF4_2x1SM_SS_SPARSE<a_type, b_type, c_type, sf_type,
                              M, N, VS, a_major, b_major, a_neg, b_neg>, uint32_t>
  with(UMMA::ScaleOut accumulate, Tensor<TE, TELayout> const& E, Tensor<TSFA, TSFALayout> const& SFA, Tensor<TSFB, TSFBLayout> const& SFB) const {
    uint32_t tmem_e_addr   = raw_pointer_cast(E.data());
    uint32_t tmem_sfa_addr = raw_pointer_cast(SFA.data());     // Move to a CoupledTensor rather than a .with()?
    uint32_t tmem_sfb_addr = raw_pointer_cast(SFB.data());     // Move to a CoupledTensor rather than a .with()?
    return {accumulate, tmem_sfa_addr, tmem_sfb_addr, {tmem_e_addr}, idesc_};
  }
};

/**
 * Specialization for a vectorized FMA per thread.
 */
template <>
struct MMA_Traits<SM100_2x1x1_F32F32F32F32>
{
  using ValTypeD = float;
  using ValTypeA = float;
  using ValTypeB = float;
  using ValTypeC = float;

  using Shape_MNK = Shape<_2,_1,_1>;
  using ThrID   = Layout<_1>;

  using ALayout = Layout<Shape<_1,_2>>;
  using BLayout = Layout<Shape<_1,_1>>;
  using CLayout = Layout<Shape<_1,_2>>;
};

template <>
struct MMA_Traits<SM100_1x2x1_F32F32F32F32>
{
  using ValTypeD = float;
  using ValTypeA = float;
  using ValTypeB = float;
  using ValTypeC = float;

  using Shape_MNK = Shape<_1,_2,_1>;
  using ThrID   = Layout<_1>;

  using ALayout = Layout<Shape<_1,_1>>;
  using BLayout = Layout<Shape<_1,_2>>;
  using CLayout = Layout<Shape<_1,_2>>;
};

namespace SM103 {
  // Common mma_unpack for all MMA_Ops in cute::SM103
template <class MMA_Op,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr
void
mma_unpack(MMA_Traits<MMA_Op> const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& zA,
             Tensor<TB, BLayout> const& zB,
             Tensor<TC, CLayout> const& C)
  {
    auto [A, next_A, SFA] = unzip_tensor(zA);
    auto [B, next_B, SFB] = unzip_tensor(zB);

    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_next_a = next_A[0];
    uint64_t desc_b = B[0];
    uint64_t desc_next_b = next_B[0];

    auto desc_a_temp = reinterpret_cast<UMMA::SmemDescriptor &>(desc_a);
    auto desc_next_a_temp = reinterpret_cast<UMMA::SmemDescriptor &>(desc_next_a);
    desc_a_temp.lbo_mode_ = 1;
    desc_a_temp.leading_byte_offset_ = desc_next_a_temp.start_address_;

    auto desc_b_temp = reinterpret_cast<UMMA::SmemDescriptor &>(desc_b);
    auto desc_next_b_temp = reinterpret_cast<UMMA::SmemDescriptor &>(desc_next_b);
    desc_b_temp.lbo_mode_ = 1;
    desc_b_temp.leading_byte_offset_ = desc_next_b_temp.start_address_;

    uint32_t tmem_c = raw_pointer_cast(D.data());
    UMMA::InstrDescriptorBlockScaled instr_desc =  traits.idesc_;
    instr_desc.k_size_ = 1;
    auto tsfa_addr = raw_pointer_cast(SFA.data());
    auto tsfb_addr = raw_pointer_cast(SFB.data());

    uint64_t idesc = UMMA::make_runtime_instr_desc_block_scaled<>(instr_desc, tsfa_addr, tsfb_addr);
    // print("a: "); print(A); print("\n");
    // print("b: "); print(B); print("\n");

    MMA_Op::fma(reinterpret_cast<uint64_t &>(desc_a_temp), reinterpret_cast<uint64_t &>(desc_b_temp), tmem_c, uint32_t(traits.accumulate_), idesc, tsfa_addr, tsfb_addr);
  }
} // end namespace SM103


template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, int VS, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM103::SM103_MXF4_ULTRA_SS_VS<a_type, b_type, c_type, sf_type,
                                M, N, VS, a_major, b_major,
                                a_neg, b_neg>>
{
  using ValTypeD   = c_type;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = c_type;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  // Logical shape-K is always 256bits, transform to units of elements
  constexpr static int K = 96;
  constexpr static int SFVecSize = VS;

  static_assert(a_major == UMMA::Major::K && b_major == UMMA::Major::K, "This MMA does not support transpose");

  using FrgTypeA   = UMMA::smem_desc<a_major>;
  using FrgTypeB   = UMMA::smem_desc<b_major>;
  using FrgTypeC   = UMMA::tmem_frg_1sm<c_type>;
  using FrgTypeSFA = UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, true>;
  using FrgTypeSFB = UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, false>;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  using MMA_ScaleFactor = SM100_MMA_MXF4_SS<a_type, b_type, c_type, sf_type,
                              M, (round_up(N, 128)), VS, a_major, b_major,
                              a_neg, b_neg>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptorBlockScaled idesc_ = UMMA::make_instr_desc_block_scaled<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major, a_neg, b_neg>();
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, int VS, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM103::SM103_MXF4_ULTRA_2x1SM_SS_VS<a_type, b_type, c_type, sf_type,
                                M, N, VS, a_major, b_major,
                                a_neg, b_neg>>
{
  using ValTypeD   = c_type;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = c_type;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  // Logical shape-K is always 256bits, transform to units of elements
  constexpr static int K = 96;
  constexpr static int SFVecSize = VS;

  static_assert(a_major == UMMA::Major::K && b_major == UMMA::Major::K, "This MMA does not support transpose");

  using FrgTypeA   = UMMA::smem_desc<a_major>;
  using FrgTypeB   = UMMA::smem_desc<b_major>;
  using FrgTypeC   = UMMA::tmem_frg_2sm<c_type>;
  constexpr static UMMA::TmemAllocMode TmemAlloc = M == 128 ?
      UMMA::TmemAllocMode::ScaleFactorDuplicated2by2 : UMMA::TmemAllocMode::ScaleFactorDuplicated4by1;
  using FrgTypeSFA = UMMA::tmem_sf_frg<sf_type, SFVecSize, 2,  true, TmemAlloc>;
  using FrgTypeSFB = UMMA::tmem_sf_frg<sf_type, SFVecSize, 2, false, TmemAlloc>;

  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_2>;
  using ALayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<K>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;
  using BLayout = Layout<Shape <      _2,Shape <Int<N/2>,Int<K>>>,
                         Stride<Int<N/2>,Stride<      _1,Int<N>>>>;
  using CLayout = Layout<Shape <      _2,Shape <Int<M/2>,Int<N>>>,
                         Stride<Int<M/2>,Stride<      _1,Int<M>>>>;

  using MMA_ScaleFactor = SM100_MMA_MXF4_SS<a_type, b_type, c_type, sf_type,
                                (M/2 > 64 ? M/2 : M), (round_up(N, 128)), VS, a_major, b_major,
                                a_neg, b_neg>;


  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptorBlockScaled idesc_ = UMMA::make_instr_desc_block_scaled<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major, a_neg, b_neg>();
};

} // end namespace cute
