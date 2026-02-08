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

#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/partitioner.hpp>
#include <cute/numeric/numeric_types.hpp>

#include <cute/layout.hpp>

namespace cute
{
template <>
struct Copy_Traits<SM100_LOAD_256bit_CACHE_NOALLOCATION>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,_256>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,_256>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template <>
struct Copy_Traits<SM100_STORE_256bit_CACHE_NOALLOCATION>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,_256>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,_256>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template <>
struct Copy_Traits<SM100_U8x8_LDSM_T>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape <  _2,   _2,  _4,_2>,_128>,
                           Stride<Stride<_128,_1024,_256,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape <  _4,_8>,Shape <_8,  _2, _2,   _2>>,
                           Stride<Stride<_256,_8>,Stride<_1,_128,_64,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<SM100_U8x16_LDSM_T>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape <  _2,   _2,  _4,   _2>,_128>,
                           Stride<Stride<_128,_1024,_256,_2048>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape <  _4,_8>,Shape <_8,  _2, _2,   _4>>,
                           Stride<Stride<_256,_8>,Stride<_1,_128,_64,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<SM100_SU4_DU8x16_x1_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape <  _8,_4>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,_32>,
                           Stride<_32, _1>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<SM100_SU6_DU8x16_x1_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape <  _8,_4>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,_32>,
                           Stride<_32, _1>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<SM100_SU4_DU8x16_x2_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape < _16,_2>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _2>>,
                           Stride<_32,Stride< _1,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<SM100_SU6_DU8x16_x2_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape < _16,_2>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _2>>,
                           Stride<_32,Stride< _1,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<SM100_SU4_DU8x16_x4_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _4>>,
                           Stride<_32,Stride< _1,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<SM100_SU6_DU8x16_x4_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _4>>,
                           Stride<_32,Stride< _1,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<SM100_U8x4_STSM_T>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape <  _4,_8>,Shape <_8,  _2, _2>>,
                           Stride<Stride<_256,_8>,Stride<_1,_128,_64>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape <  _8,_4>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template <>
struct Copy_Traits<SM100_U8x8_STSM_T>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape <  _4,_8>,Shape <_8,  _2, _2,   _2>>,
                           Stride<Stride<_256,_8>,Stride<_1,_128,_64,_1024>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _16,_2>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template <>
struct Copy_Traits<SM100_U8x16_STSM_T>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape <  _4,_8>,Shape <_8,  _2, _2,   _4>>,
                           Stride<Stride<_256,_8>,Stride<_1,_128,_64,_1024>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMEM Traits and Utilities
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class... Args>
struct Copy_Atom;

/** Generate a TiledCopy from a CopyAtom and a TMEM tensor
 * Example:
 *   Tensor gmem_tensor = ...                                            // (M,N,...)
 *   Tensor tmem_tensor = ...                                            // (M,N,...)
 *   auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD_Operation, tmem_tensor);
 *   auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);
 *
 *   Tensor tDtC = thr_tmem_load.partition_S(tmem_tensor);                    // (TMEM_LOAD,TMEM_LOAD_M,TMEM_LOAD_N,...)
 *   Tensor tDgC = thr_tmem_load.partition_D(gmem_tensor);                    // (TMEM_LOAD,TMEM_LOAD_M,TMEM_LOAD_N,...)
 *   Tensor tDrC = make_tensor<ElementAccumulator>(shape(tDgD));         // (TMEM_LOAD,TMEM_LOAD_M,TMEM_LOAD_N,...)
 *
 *   copy(tiled_tmem_load, tDtC, tDrC);       // tmem -> rmem
 *   copy(tDrC, tDgC);                   // rmem -> gmem
 */
template <class CopyOp, class CopyT,
          class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
auto
make_tmem_copy(Copy_Atom<CopyOp,CopyT> const& atom,
               Tensor<TEngine,TLayout> const& tmem)
{
  static_assert(is_tmem<TEngine>::value, "Expected TMEM tensor.");
  using T      = typename TEngine::value_type;
  using Traits = typename Copy_Atom<CopyOp, CopyT>::Traits;
  static_assert(sizeof_bits_v<CopyT> == sizeof_bits_v<T>,
                "Expected a CopyAtom with the same type-width as the Tensor.");

  // atom thr idx -> tmem addr    4warps where each warp points to the same position within it's own subpartition
  auto atom_t_layout = Layout<Shape<_32,_4>, Stride<_0, decltype(Int<32>{} * TMEM::DP<T>{})>>{};
  // atom val idx -> tmem addr    Cast the CopyOp's value ids to the proper data width
  auto atom_v_layout = coalesce(upcast<sizeof_bits<T>::value>(typename Traits::ValID{}));

  return make_cotiled_copy(atom, make_layout(atom_t_layout, atom_v_layout), tmem.layout());
}

template <class CopyOp,
          class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
auto
make_tmem_copy(CopyOp const&,
               Tensor<TEngine,TLayout> const& tmem)
{
  return make_tmem_copy(Copy_Atom<CopyOp, typename TEngine::value_type>{}, tmem);
}

/** Generate a TV_Tiler from a TMEM tensor
 * Example:
 *   Tensor gmem_tensor = ...                                            // (M,N,...)
 *   Tensor tmem_tensor = ...                                            // (M,N,...)
 *   auto tmem_tiler = make_tmem_warp_partitioner(tmem_tensor);
 *   auto warp_tiler  = tmem_tiler.get_slice(warp_idx);
 *
 *   Tensor tWtC = warp_tiler.partition(tmem_tensor);                    // (WARP_M,WARP_N,...)
 *   Tensor tWgC = warp_tiler.partition(gmem_tensor);                    // (WARP_M,WARP_N,...)
 */
template <class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
auto
make_tmem_warp_partitioner(Tensor<TEngine,TLayout> const& tmem)
{
  static_assert(is_tmem<TEngine>::value, "Expected TMEM tensor.");
  using T = typename TEngine::value_type;

  // warp idx -> tmem addr    This is the T in the Layout_TV
  auto atom_t_layout = Layout<_4, decltype(Int<32>{} * TMEM::DP<T>{})>{};

  // tmem coord -> tmem addr
  auto tmem_layout = tmem.layout();
  // tmem addr -> tmem coord    Append 1:0 so off-the-ends get the stride-0
  auto inv_tmem_layout = make_layout(left_inverse(tmem_layout), Layout<_1,_0>{});

  // wid -> tmem_coord
  auto layout_t_tmem = composition(inv_tmem_layout, atom_t_layout);
  //
  // Tiler -- Find the active elements in the TMEM tensor and generate a tiler to extract them
  //

  // Convert to the awkward by-mode tiler to preserve the modes of the tiled TMEM
  auto flat_tmem_shape = product_each(shape(tmem_layout));
  auto flat_tmem_zeros = repeat<rank(flat_tmem_shape)>(Int<0>{});

  auto tiler = transform(make_seq<rank(flat_tmem_shape)>{}, [&](auto i) {
    return filter(composition(make_layout(flat_tmem_shape, replace<i>(flat_tmem_zeros, Int<1>{})), layout_t_tmem));
  });

  //
  // Layout_TV -- Find the (tid,vid) -> tile coord transformation
  //

  // Apply the tiler to a reference and transform the codomain
  // tile_coord -> tmem_coord
  auto tile2tmem = composition(make_layout(flat_tmem_shape), tiler);

  // wid -> tile_coord
  auto layout_tv = composition(left_inverse(tile2tmem), layout_t_tmem);
  return make_tiler_impl(layout_tv, tiler);
}

namespace SM100::TMEM::LOAD {

//
// Specialized copy_unpack implementation for SM100::TMEM::LOAD instructions
//

template <class CopyOp,
          class TS, class SLayout,
          class TD, class DLayout>
CUTE_HOST_DEVICE constexpr
void
copy_unpack(Copy_Traits<CopyOp> const& traits,
            Tensor<TS,SLayout>  const& src,
            Tensor<TD,DLayout>       & dst)
{
  static_assert(is_tmem<TS>::value, "Expected TMEM src.");
  static_assert(is_rmem<TD>::value, "Expected RMEM dst.");

  using SrcType = typename TS::value_type;
  CUTE_STATIC_ASSERT_V((coalesce(layout(src)) == coalesce(upcast<sizeof_bits<SrcType>::value>(typename Copy_Traits<CopyOp>::ValID{}))),
    "Expected src to have the specific TMEM layout required by CopyOp.");

  uint32_t tmem_addr = raw_pointer_cast(src.data());

  using RegTypeDst = typename remove_extent<typename CopyOp::DRegisters>::type;
  Tensor rD = recast<RegTypeDst>(dst);

  constexpr int RegNumDst = extent<typename CopyOp::DRegisters>::value;
  CUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumDst>{},
    "In CopyAtom, dst layout doesn't vectorize into registers. This dst layout is incompatible with this CopyOp.");

  // thread idx <=> DP lane assert.
  // ASSERT TMEM_LOAD thread attemping to access DP lane within sub-partition.
#if defined(__CUDA_ARCH__) && !defined(NDEBUG)
  assert(((uint32_t(threadIdx.x) / 32) % 4) == (((tmem_addr >> 16) / 32) % 4));
#endif

  detail::explode(CopyOp::copy,
                  &tmem_addr, seq<0>{},
                  rD, make_seq<RegNumDst>{});
}

} // end namespace SM100::TMEM::LOAD

namespace SM100::TMEM::STORE {

//
// Specialized copy_unpack implementation for SM100::TMEM::STORE instructions
//

template <class CopyOp,
          class TS, class SLayout,
          class TD, class DLayout>
CUTE_HOST_DEVICE constexpr
void
copy_unpack(Copy_Traits<CopyOp> const& traits,
            Tensor<TS,SLayout>  const& src,
            Tensor<TD,DLayout>       & dst)
{
  static_assert(is_rmem<TS>::value, "Expected RMEM src.");
  static_assert(is_tmem<TD>::value, "Expected TMEM dst.");

  using RegTypeSrc = typename remove_extent<typename CopyOp::SRegisters>::type;
  Tensor rS = recast<RegTypeSrc>(src);

  constexpr int RegNumSrc = extent<typename CopyOp::SRegisters>::value;
  CUTE_STATIC_ASSERT_V(size(rS) == Int<RegNumSrc>{},
    "In CopyAtom, src layout doesn't vectorize into registers. This src layout is incompatible with this tiled copy.");

  using DstType = typename TD::value_type;
  CUTE_STATIC_ASSERT_V((coalesce(layout(dst)) == coalesce(upcast<sizeof_bits<DstType>::value>(typename Copy_Traits<CopyOp>::ValID{}))),
    "Expected dst to have the specific TMEM layout required by CopyOp.");

  uint32_t tmem_addr = raw_pointer_cast(dst.data());

  // thread idx <=> DP lane assert.
  // ASSERT TMEM_LOAD thread attemping to access DP lane within sub-partition.
#if defined(__CUDA_ARCH__) && !defined(NDEBUG)
  assert(((uint32_t(threadIdx.x) / 32) % 4) == (((tmem_addr >> 16) / 32) % 4));
#endif

  detail::explode(CopyOp::copy,
                  rS, make_seq<RegNumSrc>{},
                  &tmem_addr, seq<0>{});
}

} // end namespace SM100::TMEM::STORE

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMEM_LOAD Copy Traits
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b1x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp256b1x>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;
  // Logical bit id to bit idx (address)
  using ValID = Layout<Shape <_256,       _16>,
                       Stride<  _1,TMEM::DP_b>>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_32,_4096>,
                           Stride< _0,   _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _4,  _8>,Shape <_64,   _2>>,
                           Stride<Stride<_64,_256>,Stride< _1,_2048>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b1x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp256b1x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_16>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_4096>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _4,  _8>,Shape <_64,   _2>>,
                           Stride<Stride<_64,_256>,Stride< _1,_2048>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b2x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp256b2x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_512,       _16>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_8192>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _4,  _8>,Shape <_64,   _2,  _2>>,
                           Stride<Stride<_64,_512>,Stride< _1,_4096,_256>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b2x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp256b2x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_32>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_8192>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _4,  _8>,Shape <_64,   _2,  _2>>,
                           Stride<Stride<_64,_512>,Stride< _1,_4096,_256>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b4x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp256b4x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_1024,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_16384>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_64,   _2,  _4>>,
                           Stride<Stride<_64,_1024>,Stride< _1,_8192,_256>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b4x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp256b4x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_64>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_16384>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_64,   _2,  _4>>,
                           Stride<Stride<_64,_1024>,Stride< _1,_8192,_256>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b8x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp256b8x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_2048,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_32768>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_64,    _2,  _8>>,
                           Stride<Stride<_64,_2048>,Stride< _1,_16384,_256>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b8x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp256b8x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_128>,       _16>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_32768>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_64,    _2,  _8>>,
                           Stride<Stride<_64,_2048>,Stride< _1,_16384,_256>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b16x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp256b16x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_4096,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_65536>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_64,    _2, _16>>,
                           Stride<Stride<_64,_4096>,Stride< _1,_32768,_256>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b16x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp256b16x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_256>,       _16>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_65536>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_64,    _2, _16>>,
                           Stride<Stride<_64,_4096>,Stride< _1,_32768,_256>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b32x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp256b32x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_8192,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_131072>,
                           Stride< _0,     _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_64,    _2, _32>>,
                           Stride<Stride<_64,_8192>,Stride< _1,_65536,_256>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b32x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp256b32x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_512>,       _16>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_131072>,
                           Stride< _0,     _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_64,    _2, _32>>,
                           Stride<Stride<_64,_8192>,Stride< _1,_65536,_256>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b1x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b1x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_128,       _16>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_2048>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _4,  _8>,Shape <_32,   _2>>,
                           Stride<Stride<_32,_128>,Stride< _1,_1024>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b1x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b1x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16, _8>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_2048>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _4,  _8>,Shape <_32,   _2>>,
                           Stride<Stride<_32,_128>,Stride< _1,_1024>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b2x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b2x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_256,       _16>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_4096>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _4,  _8>,Shape <_32,   _2,  _2>>,
                           Stride<Stride<_32,_256>,Stride< _1,_2048,_128>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b2x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b2x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_16>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_4096>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _4,  _8>,Shape <_32,   _2,  _2>>,
                           Stride<Stride<_32,_256>,Stride< _1,_2048,_128>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b4x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b4x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_512,       _16>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_8192>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _4,  _8>,Shape <_32,   _2,  _4>>,
                           Stride<Stride<_32,_512>,Stride< _1,_4096,_128>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b4x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b4x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_32>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_8192>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _4,  _8>,Shape <_32,   _2,  _4>>,
                           Stride<Stride<_32,_512>,Stride< _1,_4096,_128>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b8x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b8x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_1024,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_16384>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_32,   _2,  _8>>,
                           Stride<Stride<_32,_1024>,Stride< _1,_8192,_128>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b8x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b8x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_64>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_16384>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_32,   _2,  _8>>,
                           Stride<Stride<_32,_1024>,Stride< _1,_8192,_128>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b16x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b16x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_2048,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_32768>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_32,    _2, _16>>,
                           Stride<Stride<_32,_2048>,Stride< _1,_16384,_128>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b16x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b16x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_128>,       _16>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_32768>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_32,    _2, _16>>,
                           Stride<Stride<_32,_2048>,Stride< _1,_16384,_128>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b32x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b32x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_4096,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_65536>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_32,    _2, _32>>,
                           Stride<Stride<_32,_4096>,Stride< _1,_32768,_128>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b32x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b32x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_256>,       _16>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_65536>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_32,    _2, _32>>,
                           Stride<Stride<_32,_4096>,Stride< _1,_32768,_128>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b64x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b64x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_8192,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_131072>,
                           Stride< _0,     _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_32,    _2, _64>>,
                           Stride<Stride<_32,_8192>,Stride< _1,_65536,_128>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp128b64x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp128b64x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_512>,       _16>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_131072>,
                           Stride< _0,     _1>>;
  using DstLayout = Layout<Shape <Shape < _4,   _8>,Shape <_32,    _2, _64>>,
                           Stride<Stride<_32,_8192>,Stride< _1,_65536,_128>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b1x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b1x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_64,       _16>,
                       Stride< _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_1024>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape <  _2, _2, _8>,_32>,
                           Stride<Stride<_512,_32,_64>, _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b1x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b1x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16, _4>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_1024>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape <  _2, _2, _8>,_32>,
                           Stride<Stride<_512,_32,_64>, _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b2x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b2x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_128,       _16>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_2048>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape <   _2, _2,  _8>,Shape <_32, _2>>,
                           Stride<Stride<_1024,_32,_128>,Stride< _1,_64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b2x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b2x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16, _8>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_2048>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape <   _2, _2,  _8>,Shape <_32, _2>>,
                           Stride<Stride<_1024,_32,_128>,Stride< _1,_64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b4x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b4x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_256,       _16>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_4096>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape <   _2, _2,  _8>,Shape <_32, _4>>,
                           Stride<Stride<_2048,_32,_256>,Stride< _1,_64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b4x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b4x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_16>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_4096>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape <   _2, _2,  _8>,Shape <_32, _4>>,
                           Stride<Stride<_2048,_32,_256>,Stride< _1,_64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b8x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b8x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_512,       _16>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_8192>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape <   _2, _2,  _8>,Shape <_32, _8>>,
                           Stride<Stride<_4096,_32,_512>,Stride< _1,_64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b8x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b8x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_32>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_8192>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape <   _2, _2,  _8>,Shape <_32, _8>>,
                           Stride<Stride<_4096,_32,_512>,Stride< _1,_64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b16x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b16x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_1024,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_16384>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape <   _2, _2,   _8>,Shape <_32,_16>>,
                           Stride<Stride<_8192,_32,_1024>,Stride< _1,_64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b16x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b16x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_64>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_16384>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape <   _2, _2,   _8>,Shape <_32,_16>>,
                           Stride<Stride<_8192,_32,_1024>,Stride< _1,_64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b32x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b32x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_2048,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_32768>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape <    _2, _2,   _8>,Shape <_32,_32>>,
                           Stride<Stride<_16384,_32,_2048>,Stride< _1,_64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b32x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b32x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_128>,       _16>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_32768>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape <    _2, _2,   _8>,Shape <_32,_32>>,
                           Stride<Stride<_16384,_32,_2048>,Stride< _1,_64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b64x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b64x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_4096,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_65536>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape <    _2, _2,   _8>,Shape <_32,_64>>,
                           Stride<Stride<_32768,_32,_4096>,Stride< _1,_64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b64x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b64x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_256>,       _16>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_65536>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape <    _2, _2,   _8>,Shape <_32,_64>>,
                           Stride<Stride<_32768,_32,_4096>,Stride< _1,_64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b128x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b128x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_8192,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_131072>,
                           Stride< _0,     _1>>;
  using DstLayout = Layout<Shape <Shape <    _2, _2,   _8>,Shape <_32,_128>>,
                           Stride<Stride<_65536,_32,_8192>,Stride< _1, _64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp64b128x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp64b128x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_512>,       _16>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_131072>,
                           Stride< _0,     _1>>;
  using DstLayout = Layout<Shape <Shape <    _2, _2,   _8>,Shape <_32,_128>>,
                           Stride<Stride<_65536,_32,_8192>,Stride< _1, _64>>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b1x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b1x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_64,       _16>,
                       Stride< _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_1024>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape <_16, _2>,_32>,
                           Stride<Stride<_64,_32>, _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b1x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b1x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16, _4>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_1024>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape <_16, _2>,_32>,
                           Stride<Stride<_64,_32>, _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b2x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b2x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_128,       _16>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_2048>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _16, _2>,_64>,
                           Stride<Stride<_128,_64>, _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b2x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b2x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16, _8>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_2048>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _16, _2>,_64>,
                           Stride<Stride<_128,_64>, _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b4x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b4x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_256,       _16>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_4096>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _16,  _2>,_128>,
                           Stride<Stride<_256,_128>,  _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b4x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b4x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_16>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_4096>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _16,  _2>,_128>,
                           Stride<Stride<_256,_128>,  _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b8x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b8x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_512,       _16>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_8192>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _16,  _2>,_256>,
                           Stride<Stride<_512,_256>,  _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b8x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b8x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_32>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_8192>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <Shape < _16,  _2>,_256>,
                           Stride<Stride<_512,_256>,  _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b16x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b16x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_1024,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_16384>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape <  _16,  _2>,_512>,
                           Stride<Stride<_1024,_512>,  _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b16x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b16x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_64>,       _16>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_16384>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape <  _16,  _2>,_512>,
                           Stride<Stride<_1024,_512>,  _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b32x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b32x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_2048,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_32768>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape <  _16,   _2>,_1024>,
                           Stride<Stride<_2048,_1024>,   _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b32x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b32x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_128>,       _16>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_32768>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape <  _16,   _2>,_1024>,
                           Stride<Stride<_2048,_1024>,   _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b64x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b64x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_4096,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_65536>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape <  _16,   _2>,_2048>,
                           Stride<Stride<_4096,_2048>,   _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b64x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b64x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_256>,       _16>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_65536>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <Shape <  _16,   _2>,_2048>,
                           Stride<Stride<_4096,_2048>,   _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b128x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b128x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_8192,       _16>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_131072>,
                           Stride< _0,     _1>>;
  using DstLayout = Layout<Shape <Shape <  _16,   _2>,_4096>,
                           Stride<Stride<_8192,_4096>,   _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b128x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_16dp32b128x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_512>,       _16>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_131072>,
                           Stride< _0,     _1>>;
  using DstLayout = Layout<Shape <Shape <  _16,   _2>,_4096>,
                           Stride<Stride<_8192,_4096>,   _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b1x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b1x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_32,       _32>,
                       Stride< _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_1024>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <_32,_32>,
                           Stride<_32, _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b1x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b1x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16, _2>,       _32>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_1024>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <_32,_32>,
                           Stride<_32, _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b2x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b2x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_64,       _32>,
                       Stride< _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_2048>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <_32,_64>,
                           Stride<_64, _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b2x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b2x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16, _4>,       _32>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_2048>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape <_32,_64>,
                           Stride<_64, _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b4x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b4x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_128,       _32>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_4096>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b4x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b4x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16, _8>,       _32>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_4096>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b8x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b8x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_256,       _32>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_8192>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape < _32,_256>,
                           Stride<_256,  _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b8x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b8x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_16>,       _32>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_8192>,
                           Stride< _0,   _1>>;
  using DstLayout = Layout<Shape < _32,_256>,
                           Stride<_256,  _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b16x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b16x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_512,       _32>,
                       Stride<  _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_16384>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape < _32,_512>,
                           Stride<_512,  _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b16x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b16x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_32>,       _32>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_16384>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape < _32,_512>,
                           Stride<_512,  _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b32x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b32x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_1024,       _32>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_32768>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <  _32,_1024>,
                           Stride<_1024,   _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b32x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b32x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_64>,       _32>,
                       Stride<Stride< _1,_32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_32768>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <  _32,_1024>,
                           Stride<_1024,   _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b64x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b64x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_2048,       _32>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_65536>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <  _32,_2048>,
                           Stride<_2048,   _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b64x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b64x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_128>,       _32>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_65536>,
                           Stride< _0,    _1>>;
  using DstLayout = Layout<Shape <  _32,_2048>,
                           Stride<_2048,   _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b128x;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b128x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_4096,       _32>,
                       Stride<   _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_131072>,
                           Stride< _0,     _1>>;
  using DstLayout = Layout<Shape <  _32,_4096>,
                           Stride<_4096,   _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b128x_16b;

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b128x_16b>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <Shape <_16,_256>,       _32>,
                       Stride<Stride< _1, _32>,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32,_131072>,
                           Stride< _0,     _1>>;
  using DstLayout = Layout<Shape <  _32,_4096>,
                           Stride<_4096,   _1>>;
  using RefLayout = SrcLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMEM_STORE Copy Traits
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b1x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp256b1x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b1x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b1x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b1x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b1x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b1x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b1x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp256b1x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b1x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b1x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b1x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b1x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b1x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b2x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp256b2x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b2x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b2x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b2x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b2x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b2x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b2x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp256b2x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b2x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b2x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b2x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b2x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b2x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b4x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp256b4x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b4x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b4x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b4x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b4x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b4x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b4x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp256b4x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b4x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b4x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b4x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b4x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b4x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b8x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp256b8x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b8x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b8x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b8x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b8x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b8x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b8x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp256b8x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b8x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b8x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b8x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b8x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b8x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b16x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp256b16x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b16x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b16x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b16x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b16x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b16x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b16x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp256b16x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b16x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b16x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b16x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b16x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b16x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b32x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp256b32x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b32x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b32x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b32x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b32x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b32x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b32x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp256b32x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b32x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b32x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b32x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b32x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp256b32x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b1x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b1x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b1x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b1x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b1x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b1x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b1x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b1x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b1x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b1x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b1x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b1x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b1x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b1x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b2x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b2x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b2x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b2x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b2x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b2x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b2x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b2x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b2x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b2x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b2x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b2x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b2x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b2x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b4x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b4x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b4x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b4x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b4x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b4x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b4x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b4x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b4x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b4x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b4x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b4x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b4x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b4x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b8x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b8x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b8x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b8x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b8x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b8x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b8x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b8x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b8x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b8x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b8x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b8x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b8x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b8x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b16x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b16x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b16x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b16x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b16x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b16x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b16x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b16x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b16x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b16x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b16x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b16x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b16x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b16x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b32x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b32x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b32x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b32x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b32x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b32x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b32x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b32x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b32x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b32x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b32x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b32x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b32x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b32x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b64x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b64x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b64x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b64x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b64x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b64x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b64x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp128b64x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp128b64x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b64x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b64x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b64x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b64x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp128b64x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b1x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b1x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b1x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b1x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b1x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b1x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b1x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b1x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b1x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b1x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b1x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b1x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b1x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b1x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b2x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b2x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b2x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b2x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b2x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b2x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b2x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b2x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b2x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b2x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b2x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b2x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b2x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b2x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b4x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b4x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b4x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b4x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b4x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b4x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b4x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b4x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b4x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b4x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b4x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b4x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b4x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b4x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b8x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b8x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b8x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b8x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b8x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b8x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b8x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b8x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b8x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b8x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b8x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b8x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b8x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b8x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b16x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b16x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b16x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b16x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b16x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b16x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b16x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b16x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b16x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b16x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b16x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b16x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b16x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b16x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b32x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b32x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b32x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b32x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b32x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b32x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b32x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b32x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b32x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b32x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b32x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b32x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b32x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b32x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b64x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b64x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b64x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b64x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b64x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b64x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b64x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b64x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b64x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b64x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b64x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b64x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b64x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b64x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b128x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b128x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b128x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b128x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b128x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b128x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b128x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp64b128x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp64b128x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b128x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b128x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b128x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b128x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp64b128x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b1x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b1x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b1x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b1x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b1x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b1x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b1x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b1x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b1x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b1x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b1x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b1x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b1x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b1x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b2x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b2x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b2x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b2x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b2x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b2x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b2x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b2x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b2x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b2x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b2x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b2x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b2x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b2x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b4x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b4x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b4x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b4x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b4x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b4x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b4x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b4x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b4x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b4x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b4x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b4x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b4x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b4x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b8x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b8x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b8x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b8x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b8x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b8x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b8x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b8x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b8x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b8x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b8x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b8x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b8x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b8x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b16x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b16x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b16x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b16x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b16x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b16x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b16x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b16x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b16x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b16x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b16x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b16x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b16x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b16x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b32x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b32x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b32x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b32x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b32x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b32x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b32x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b32x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b32x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b32x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b32x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b32x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b32x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b32x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b64x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b64x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b64x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b64x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b64x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b64x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b64x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b64x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b64x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b64x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b64x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b64x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b64x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b64x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b128x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b128x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b128x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b128x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b128x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b128x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b128x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_16dp32b128x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_16dp32b128x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b128x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b128x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b128x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b128x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_16dp32b128x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b1x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b1x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b1x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b1x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b1x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b1x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b1x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b1x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b1x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b1x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b1x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b1x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b1x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b1x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b2x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b2x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b2x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b2x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b2x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b2x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b2x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b2x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b2x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b2x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b2x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b2x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b2x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b2x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b4x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b4x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b4x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b4x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b4x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b4x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b4x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b4x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b4x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b4x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b4x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b4x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b4x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b4x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b8x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b8x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b8x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b8x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b8x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b8x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b8x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b8x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b8x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b8x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b8x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b8x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b8x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b8x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b16x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b16x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b16x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b16x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b16x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b16x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b16x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b16x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b16x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b16x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b16x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b16x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b16x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b16x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b32x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b32x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b32x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b32x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b32x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b32x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b32x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b32x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b32x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b32x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b32x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b32x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b32x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b32x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b64x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b64x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b64x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b64x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b64x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b64x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b64x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b64x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b64x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b64x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b64x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b64x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b64x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b64x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b128x;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b128x>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b128x>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b128x>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b128x>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b128x>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b128x>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::STORE::SM100_TMEM_STORE_32dp32b128x_16b;

template <>
struct Copy_Traits<SM100_TMEM_STORE_32dp32b128x_16b>
{
  using ThrID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b128x_16b>::ThrID;
  using ValID = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b128x_16b>::ValID;
  using SrcLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b128x_16b>::DstLayout;
  using DstLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b128x_16b>::SrcLayout;
  using RefLayout = typename Copy_Traits<SM100_TMEM_LOAD_32dp32b128x_16b>::RefLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace TMEM {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Given a 1x tmem copy op, returns the widest repeated variant that divides the specified bits in the N-mode
template <class CopyOp, int bits_n>
CUTE_HOST_DEVICE constexpr
auto
op_repeater()
{
  if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b1x>) {
    if constexpr (bits_n % (256 * 32) == 0) {
      return SM100_TMEM_LOAD_16dp256b32x{};
    }
    else if constexpr (bits_n % (256 * 16) == 0) {
      return SM100_TMEM_LOAD_16dp256b16x{};
    }
    else if constexpr (bits_n % (256 *  8) == 0) {
      return SM100_TMEM_LOAD_16dp256b8x{};
    }
    else if constexpr (bits_n % (256 *  4) == 0) {
      return SM100_TMEM_LOAD_16dp256b4x{};
    }
    else if constexpr (bits_n % (256 *  2) == 0) {
      return SM100_TMEM_LOAD_16dp256b2x{};
    }
    else if constexpr (bits_n % (256 *  1) == 0) {
      return SM100_TMEM_LOAD_16dp256b1x{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b1x_16b>) {
    if constexpr (bits_n % (256 * 32) == 0) {
      return SM100_TMEM_LOAD_16dp256b32x_16b{};
    }
    else if constexpr (bits_n % (256 * 16) == 0) {
      return SM100_TMEM_LOAD_16dp256b16x_16b{};
    }
    else if constexpr (bits_n % (256 *  8) == 0) {
      return SM100_TMEM_LOAD_16dp256b8x_16b{};
    }
    else if constexpr (bits_n % (256 *  4) == 0) {
      return SM100_TMEM_LOAD_16dp256b4x_16b{};
    }
    else if constexpr (bits_n % (256 *  2) == 0) {
      return SM100_TMEM_LOAD_16dp256b2x_16b{};
    }
    else if constexpr (bits_n % (256 *  1) == 0) {
      return SM100_TMEM_LOAD_16dp256b1x_16b{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b1x>) {
    if constexpr (bits_n % (128 * 64) == 0) {
      return SM100_TMEM_LOAD_16dp128b64x{};
    }
    else if constexpr (bits_n % (128 * 32) == 0) {
      return SM100_TMEM_LOAD_16dp128b32x{};
    }
    else if constexpr (bits_n % (128 * 16) == 0) {
      return SM100_TMEM_LOAD_16dp128b16x{};
    }
    else if constexpr (bits_n % (128 *  8) == 0) {
      return SM100_TMEM_LOAD_16dp128b8x{};
    }
    else if constexpr (bits_n % (128 *  4) == 0) {
      return SM100_TMEM_LOAD_16dp128b4x{};
    }
    else if constexpr (bits_n % (128 *  2) == 0) {
      return SM100_TMEM_LOAD_16dp128b2x{};
    }
    else if constexpr (bits_n % (128 *  1) == 0) {
      return SM100_TMEM_LOAD_16dp128b1x{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b1x_16b>) {
    if constexpr (bits_n % (128 * 64) == 0) {
      return SM100_TMEM_LOAD_16dp128b64x_16b{};
    }
    else if constexpr (bits_n % (128 * 32) == 0) {
      return SM100_TMEM_LOAD_16dp128b32x_16b{};
    }
    else if constexpr (bits_n % (128 * 16) == 0) {
      return SM100_TMEM_LOAD_16dp128b16x_16b{};
    }
    else if constexpr (bits_n % (128 *  8) == 0) {
      return SM100_TMEM_LOAD_16dp128b8x_16b{};
    }
    else if constexpr (bits_n % (128 *  4) == 0) {
      return SM100_TMEM_LOAD_16dp128b4x_16b{};
    }
    else if constexpr (bits_n % (128 *  2) == 0) {
      return SM100_TMEM_LOAD_16dp128b2x_16b{};
    }
    else if constexpr (bits_n % (128 *  1) == 0) {
      return SM100_TMEM_LOAD_16dp128b1x_16b{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b1x>) {
    if constexpr (bits_n % (64 * 128) == 0) {
      return SM100_TMEM_LOAD_16dp64b128x{};
    }
    else if constexpr (bits_n % (64 * 64) == 0) {
      return SM100_TMEM_LOAD_16dp64b64x{};
    }
    else if constexpr (bits_n % (64 * 32) == 0) {
      return SM100_TMEM_LOAD_16dp64b32x{};
    }
    else if constexpr (bits_n % (64 * 16) == 0) {
      return SM100_TMEM_LOAD_16dp64b16x{};
    }
    else if constexpr (bits_n % (64 *  8) == 0) {
      return SM100_TMEM_LOAD_16dp64b8x{};
    }
    else if constexpr (bits_n % (64 *  4) == 0) {
      return SM100_TMEM_LOAD_16dp64b4x{};
    }
    else if constexpr (bits_n % (64 *  2) == 0) {
      return SM100_TMEM_LOAD_16dp64b2x{};
    }
    else if constexpr (bits_n % (64 *  1) == 0) {
      return SM100_TMEM_LOAD_16dp64b1x{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b1x_16b>) {
    if constexpr (bits_n % (64 * 128) == 0) {
      return SM100_TMEM_LOAD_16dp64b128x_16b{};
    }
    else if constexpr (bits_n % (64 * 64) == 0) {
      return SM100_TMEM_LOAD_16dp64b64x_16b{};
    }
    else if constexpr (bits_n % (64 * 32) == 0) {
      return SM100_TMEM_LOAD_16dp64b32x_16b{};
    }
    else if constexpr (bits_n % (64 * 16) == 0) {
      return SM100_TMEM_LOAD_16dp64b16x_16b{};
    }
    else if constexpr (bits_n % (64 *  8) == 0) {
      return SM100_TMEM_LOAD_16dp64b8x_16b{};
    }
    else if constexpr (bits_n % (64 *  4) == 0) {
      return SM100_TMEM_LOAD_16dp64b4x_16b{};
    }
    else if constexpr (bits_n % (64 *  2) == 0) {
      return SM100_TMEM_LOAD_16dp64b2x_16b{};
    }
    else if constexpr (bits_n % (64 *  1) == 0) {
      return SM100_TMEM_LOAD_16dp64b1x_16b{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b1x>) {
    if constexpr (bits_n % (64 * 128) == 0) {
      return SM100_TMEM_LOAD_16dp32b128x{};
    }
    else if constexpr (bits_n % (64 * 64) == 0) {
      return SM100_TMEM_LOAD_16dp32b64x{};
    }
    else if constexpr (bits_n % (64 * 32) == 0) {
      return SM100_TMEM_LOAD_16dp32b32x{};
    }
    else if constexpr (bits_n % (64 * 16) == 0) {
      return SM100_TMEM_LOAD_16dp32b16x{};
    }
    else if constexpr (bits_n % (64 *  8) == 0) {
      return SM100_TMEM_LOAD_16dp32b8x{};
    }
    else if constexpr (bits_n % (64 *  4) == 0) {
      return SM100_TMEM_LOAD_16dp32b4x{};
    }
    else if constexpr (bits_n % (64 *  2) == 0) {
      return SM100_TMEM_LOAD_16dp32b2x{};
    }
    else if constexpr (bits_n % (64 *  1) == 0) {
      return SM100_TMEM_LOAD_16dp32b1x{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b1x_16b>) {
    if constexpr (bits_n % (64 * 128) == 0) {
      return SM100_TMEM_LOAD_16dp32b128x_16b{};
    }
    else if constexpr (bits_n % (64 * 64) == 0) {
      return SM100_TMEM_LOAD_16dp32b64x_16b{};
    }
    else if constexpr (bits_n % (64 * 32) == 0) {
      return SM100_TMEM_LOAD_16dp32b32x_16b{};
    }
    else if constexpr (bits_n % (64 * 16) == 0) {
      return SM100_TMEM_LOAD_16dp32b16x_16b{};
    }
    else if constexpr (bits_n % (64 *  8) == 0) {
      return SM100_TMEM_LOAD_16dp32b8x_16b{};
    }
    else if constexpr (bits_n % (64 *  4) == 0) {
      return SM100_TMEM_LOAD_16dp32b4x_16b{};
    }
    else if constexpr (bits_n % (64 *  2) == 0) {
      return SM100_TMEM_LOAD_16dp32b2x_16b{};
    }
    else if constexpr (bits_n % (64 *  1) == 0) {
      return SM100_TMEM_LOAD_16dp32b1x_16b{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b1x>) {
    if constexpr (bits_n % (32 * 128) == 0) {
      return SM100_TMEM_LOAD_32dp32b128x{};
    }
    else if constexpr (bits_n % (32 * 64) == 0) {
      return SM100_TMEM_LOAD_32dp32b64x{};
    }
    else if constexpr (bits_n % (32 * 32) == 0) {
      return SM100_TMEM_LOAD_32dp32b32x{};
    }
    else if constexpr (bits_n % (32 * 16) == 0) {
      return SM100_TMEM_LOAD_32dp32b16x{};
    }
    else if constexpr (bits_n % (32 *  8) == 0) {
      return SM100_TMEM_LOAD_32dp32b8x{};
    }
    else if constexpr (bits_n % (32 *  4) == 0) {
      return SM100_TMEM_LOAD_32dp32b4x{};
    }
    else if constexpr (bits_n % (32 *  2) == 0) {
      return SM100_TMEM_LOAD_32dp32b2x{};
    }
    else if constexpr (bits_n % (32 *  1) == 0) {
      return SM100_TMEM_LOAD_32dp32b1x{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b1x_16b>) {
    if constexpr (bits_n % (32 * 128) == 0) {
      return SM100_TMEM_LOAD_32dp32b128x_16b{};
    }
    else if constexpr (bits_n % (32 * 64) == 0) {
      return SM100_TMEM_LOAD_32dp32b64x_16b{};
    }
    else if constexpr (bits_n % (32 * 32) == 0) {
      return SM100_TMEM_LOAD_32dp32b32x_16b{};
    }
    else if constexpr (bits_n % (32 * 16) == 0) {
      return SM100_TMEM_LOAD_32dp32b16x_16b{};
    }
    else if constexpr (bits_n % (32 *  8) == 0) {
      return SM100_TMEM_LOAD_32dp32b8x_16b{};
    }
    else if constexpr (bits_n % (32 *  4) == 0) {
      return SM100_TMEM_LOAD_32dp32b4x_16b{};
    }
    else if constexpr (bits_n % (32 *  2) == 0) {
      return SM100_TMEM_LOAD_32dp32b2x_16b{};
    }
    else if constexpr (bits_n % (32 *  1) == 0) {
      return SM100_TMEM_LOAD_32dp32b1x_16b{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_STORE_16dp256b1x>) {
    if constexpr (bits_n % (256 * 32) == 0) {
      return SM100_TMEM_STORE_16dp256b32x{};
    }
    else if constexpr (bits_n % (256 * 16) == 0) {
      return SM100_TMEM_STORE_16dp256b16x{};
    }
    else if constexpr (bits_n % (256 *  8) == 0) {
      return SM100_TMEM_STORE_16dp256b8x{};
    }
    else if constexpr (bits_n % (256 *  4) == 0) {
      return SM100_TMEM_STORE_16dp256b4x{};
    }
    else if constexpr (bits_n % (256 *  2) == 0) {
      return SM100_TMEM_STORE_16dp256b2x{};
    }
    else if constexpr (bits_n % (256 *  1) == 0) {
      return SM100_TMEM_STORE_16dp256b1x{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_STORE_16dp256b1x_16b>) {
    if constexpr (bits_n % (256 * 32) == 0) {
      return SM100_TMEM_STORE_16dp256b32x_16b{};
    }
    else if constexpr (bits_n % (256 * 16) == 0) {
      return SM100_TMEM_STORE_16dp256b16x_16b{};
    }
    else if constexpr (bits_n % (256 *  8) == 0) {
      return SM100_TMEM_STORE_16dp256b8x_16b{};
    }
    else if constexpr (bits_n % (256 *  4) == 0) {
      return SM100_TMEM_STORE_16dp256b4x_16b{};
    }
    else if constexpr (bits_n % (256 *  2) == 0) {
      return SM100_TMEM_STORE_16dp256b2x_16b{};
    }
    else if constexpr (bits_n % (256 *  1) == 0) {
      return SM100_TMEM_STORE_16dp256b1x_16b{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_STORE_16dp128b1x>) {
    if constexpr (bits_n % (128 * 64) == 0) {
      return SM100_TMEM_STORE_16dp128b64x{};
    }
    else if constexpr (bits_n % (128 * 32) == 0) {
      return SM100_TMEM_STORE_16dp128b32x{};
    }
    else if constexpr (bits_n % (128 * 16) == 0) {
      return SM100_TMEM_STORE_16dp128b16x{};
    }
    else if constexpr (bits_n % (128 *  8) == 0) {
      return SM100_TMEM_STORE_16dp128b8x{};
    }
    else if constexpr (bits_n % (128 *  4) == 0) {
      return SM100_TMEM_STORE_16dp128b4x{};
    }
    else if constexpr (bits_n % (128 *  2) == 0) {
      return SM100_TMEM_STORE_16dp128b2x{};
    }
    else if constexpr (bits_n % (128 *  1) == 0) {
      return SM100_TMEM_STORE_16dp128b1x{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_STORE_16dp128b1x_16b>) {
    if constexpr (bits_n % (128 * 64) == 0) {
      return SM100_TMEM_STORE_16dp128b64x_16b{};
    }
    else if constexpr (bits_n % (128 * 32) == 0) {
      return SM100_TMEM_STORE_16dp128b32x_16b{};
    }
    else if constexpr (bits_n % (128 * 16) == 0) {
      return SM100_TMEM_STORE_16dp128b16x_16b{};
    }
    else if constexpr (bits_n % (128 *  8) == 0) {
      return SM100_TMEM_STORE_16dp128b8x_16b{};
    }
    else if constexpr (bits_n % (128 *  4) == 0) {
      return SM100_TMEM_STORE_16dp128b4x_16b{};
    }
    else if constexpr (bits_n % (128 *  2) == 0) {
      return SM100_TMEM_STORE_16dp128b2x_16b{};
    }
    else if constexpr (bits_n % (128 *  1) == 0) {
      return SM100_TMEM_STORE_16dp128b1x_16b{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_STORE_16dp64b1x>) {
    if constexpr (bits_n % (64 * 128) == 0) {
      return SM100_TMEM_STORE_16dp64b128x{};
    }
    else if constexpr (bits_n % (64 * 64) == 0) {
      return SM100_TMEM_STORE_16dp64b64x{};
    }
    else if constexpr (bits_n % (64 * 32) == 0) {
      return SM100_TMEM_STORE_16dp64b32x{};
    }
    else if constexpr (bits_n % (64 * 16) == 0) {
      return SM100_TMEM_STORE_16dp64b16x{};
    }
    else if constexpr (bits_n % (64 *  8) == 0) {
      return SM100_TMEM_STORE_16dp64b8x{};
    }
    else if constexpr (bits_n % (64 *  4) == 0) {
      return SM100_TMEM_STORE_16dp64b4x{};
    }
    else if constexpr (bits_n % (64 *  2) == 0) {
      return SM100_TMEM_STORE_16dp64b2x{};
    }
    else if constexpr (bits_n % (64 *  1) == 0) {
      return SM100_TMEM_STORE_16dp64b1x{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_STORE_16dp64b1x_16b>) {
    if constexpr (bits_n % (64 * 128) == 0) {
      return SM100_TMEM_STORE_16dp64b128x_16b{};
    }
    else if constexpr (bits_n % (64 * 64) == 0) {
      return SM100_TMEM_STORE_16dp64b64x_16b{};
    }
    else if constexpr (bits_n % (64 * 32) == 0) {
      return SM100_TMEM_STORE_16dp64b32x_16b{};
    }
    else if constexpr (bits_n % (64 * 16) == 0) {
      return SM100_TMEM_STORE_16dp64b16x_16b{};
    }
    else if constexpr (bits_n % (64 *  8) == 0) {
      return SM100_TMEM_STORE_16dp64b8x_16b{};
    }
    else if constexpr (bits_n % (64 *  4) == 0) {
      return SM100_TMEM_STORE_16dp64b4x_16b{};
    }
    else if constexpr (bits_n % (64 *  2) == 0) {
      return SM100_TMEM_STORE_16dp64b2x_16b{};
    }
    else if constexpr (bits_n % (64 *  1) == 0) {
      return SM100_TMEM_STORE_16dp64b1x_16b{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_STORE_16dp32b1x>) {
    if constexpr (bits_n % (64 * 128) == 0) {
      return SM100_TMEM_STORE_16dp32b128x{};
    }
    else if constexpr (bits_n % (64 * 64) == 0) {
      return SM100_TMEM_STORE_16dp32b64x{};
    }
    else if constexpr (bits_n % (64 * 32) == 0) {
      return SM100_TMEM_STORE_16dp32b32x{};
    }
    else if constexpr (bits_n % (64 * 16) == 0) {
      return SM100_TMEM_STORE_16dp32b16x{};
    }
    else if constexpr (bits_n % (64 *  8) == 0) {
      return SM100_TMEM_STORE_16dp32b8x{};
    }
    else if constexpr (bits_n % (64 *  4) == 0) {
      return SM100_TMEM_STORE_16dp32b4x{};
    }
    else if constexpr (bits_n % (64 *  2) == 0) {
      return SM100_TMEM_STORE_16dp32b2x{};
    }
    else if constexpr (bits_n % (64 *  1) == 0) {
      return SM100_TMEM_STORE_16dp32b1x{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_STORE_16dp32b1x_16b>) {
    if constexpr (bits_n % (64 * 128) == 0) {
      return SM100_TMEM_STORE_16dp32b128x_16b{};
    }
    else if constexpr (bits_n % (64 * 64) == 0) {
      return SM100_TMEM_STORE_16dp32b64x_16b{};
    }
    else if constexpr (bits_n % (64 * 32) == 0) {
      return SM100_TMEM_STORE_16dp32b32x_16b{};
    }
    else if constexpr (bits_n % (64 * 16) == 0) {
      return SM100_TMEM_STORE_16dp32b16x_16b{};
    }
    else if constexpr (bits_n % (64 *  8) == 0) {
      return SM100_TMEM_STORE_16dp32b8x_16b{};
    }
    else if constexpr (bits_n % (64 *  4) == 0) {
      return SM100_TMEM_STORE_16dp32b4x_16b{};
    }
    else if constexpr (bits_n % (64 *  2) == 0) {
      return SM100_TMEM_STORE_16dp32b2x_16b{};
    }
    else if constexpr (bits_n % (64 *  1) == 0) {
      return SM100_TMEM_STORE_16dp32b1x_16b{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_STORE_32dp32b1x>) {
    if constexpr (bits_n % (32 * 128) == 0) {
      return SM100_TMEM_STORE_32dp32b128x{};
    }
    else if constexpr (bits_n % (32 * 64) == 0) {
      return SM100_TMEM_STORE_32dp32b64x{};
    }
    else if constexpr (bits_n % (32 * 32) == 0) {
      return SM100_TMEM_STORE_32dp32b32x{};
    }
    else if constexpr (bits_n % (32 * 16) == 0) {
      return SM100_TMEM_STORE_32dp32b16x{};
    }
    else if constexpr (bits_n % (32 *  8) == 0) {
      return SM100_TMEM_STORE_32dp32b8x{};
    }
    else if constexpr (bits_n % (32 *  4) == 0) {
      return SM100_TMEM_STORE_32dp32b4x{};
    }
    else if constexpr (bits_n % (32 *  2) == 0) {
      return SM100_TMEM_STORE_32dp32b2x{};
    }
    else if constexpr (bits_n % (32 *  1) == 0) {
      return SM100_TMEM_STORE_32dp32b1x{};
    }
  }
  else if constexpr (cute::is_same_v<CopyOp, SM100_TMEM_STORE_32dp32b1x_16b>) {
    if constexpr (bits_n % (32 * 128) == 0) {
      return SM100_TMEM_STORE_32dp32b128x_16b{};
    }
    else if constexpr (bits_n % (32 * 64) == 0) {
      return SM100_TMEM_STORE_32dp32b64x_16b{};
    }
    else if constexpr (bits_n % (32 * 32) == 0) {
      return SM100_TMEM_STORE_32dp32b32x_16b{};
    }
    else if constexpr (bits_n % (32 * 16) == 0) {
      return SM100_TMEM_STORE_32dp32b16x_16b{};
    }
    else if constexpr (bits_n % (32 *  8) == 0) {
      return SM100_TMEM_STORE_32dp32b8x_16b{};
    }
    else if constexpr (bits_n % (32 *  4) == 0) {
      return SM100_TMEM_STORE_32dp32b4x_16b{};
    }
    else if constexpr (bits_n % (32 *  2) == 0) {
      return SM100_TMEM_STORE_32dp32b2x_16b{};
    }
    else if constexpr (bits_n % (32 *  1) == 0) {
      return SM100_TMEM_STORE_32dp32b1x_16b{};
    }
  }
  else {
    static_assert(dependent_false<CopyOp>, "Must pass 1x tmem copy operator");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Select TMEM store corresponding to the provided TMEM load
template <class CopyOp>
CUTE_HOST_DEVICE constexpr auto
tmem_load_to_store(CopyOp) {
  if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b1x>) {
    return SM100_TMEM_STORE_16dp256b1x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b1x_16b>) {
    return SM100_TMEM_STORE_16dp256b1x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b2x>) {
    return SM100_TMEM_STORE_16dp256b2x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b2x_16b>) {
    return SM100_TMEM_STORE_16dp256b2x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b4x>) {
    return SM100_TMEM_STORE_16dp256b4x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b4x_16b>) {
    return SM100_TMEM_STORE_16dp256b4x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b8x>) {
    return SM100_TMEM_STORE_16dp256b8x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b8x_16b>) {
    return SM100_TMEM_STORE_16dp256b8x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b16x>) {
    return SM100_TMEM_STORE_16dp256b16x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b16x_16b>) {
    return SM100_TMEM_STORE_16dp256b16x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b32x>) {
    return SM100_TMEM_STORE_16dp256b32x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp256b32x_16b>) {
    return SM100_TMEM_STORE_16dp256b32x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b1x>) {
    return SM100_TMEM_STORE_16dp128b1x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b1x_16b>) {
    return SM100_TMEM_STORE_16dp128b1x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b2x>) {
    return SM100_TMEM_STORE_16dp128b2x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b2x_16b>) {
    return SM100_TMEM_STORE_16dp128b2x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b4x>) {
    return SM100_TMEM_STORE_16dp128b4x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b4x_16b>) {
    return SM100_TMEM_STORE_16dp128b4x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b8x>) {
    return SM100_TMEM_STORE_16dp128b8x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b8x_16b>) {
    return SM100_TMEM_STORE_16dp128b8x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b16x>) {
    return SM100_TMEM_STORE_16dp128b16x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b16x_16b>) {
    return SM100_TMEM_STORE_16dp128b16x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b32x>) {
    return SM100_TMEM_STORE_16dp128b32x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b32x_16b>) {
    return SM100_TMEM_STORE_16dp128b32x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b64x>) {
    return SM100_TMEM_STORE_16dp128b64x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp128b64x_16b>) {
    return SM100_TMEM_STORE_16dp128b64x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b1x>) {
    return SM100_TMEM_STORE_16dp64b1x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b1x_16b>) {
    return SM100_TMEM_STORE_16dp64b1x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b2x>) {
    return SM100_TMEM_STORE_16dp64b2x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b2x_16b>) {
    return SM100_TMEM_STORE_16dp64b2x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b4x>) {
    return SM100_TMEM_STORE_16dp64b4x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b4x_16b>) {
    return SM100_TMEM_STORE_16dp64b4x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b8x>) {
    return SM100_TMEM_STORE_16dp64b8x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b8x_16b>) {
    return SM100_TMEM_STORE_16dp64b8x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b16x>) {
    return SM100_TMEM_STORE_16dp64b16x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b16x_16b>) {
    return SM100_TMEM_STORE_16dp64b16x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b32x>) {
    return SM100_TMEM_STORE_16dp64b32x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b32x_16b>) {
    return SM100_TMEM_STORE_16dp64b32x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b64x>) {
    return SM100_TMEM_STORE_16dp64b64x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b64x_16b>) {
    return SM100_TMEM_STORE_16dp64b64x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b128x>) {
    return SM100_TMEM_STORE_16dp64b128x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp64b128x_16b>) {
    return SM100_TMEM_STORE_16dp64b128x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b1x>) {
    return SM100_TMEM_STORE_16dp32b1x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b1x_16b>) {
    return SM100_TMEM_STORE_16dp32b1x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b2x>) {
    return SM100_TMEM_STORE_16dp32b2x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b2x_16b>) {
    return SM100_TMEM_STORE_16dp32b2x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b4x>) {
    return SM100_TMEM_STORE_16dp32b4x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b4x_16b>) {
    return SM100_TMEM_STORE_16dp32b4x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b8x>) {
    return SM100_TMEM_STORE_16dp32b8x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b8x_16b>) {
    return SM100_TMEM_STORE_16dp32b8x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b16x>) {
    return SM100_TMEM_STORE_16dp32b16x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b16x_16b>) {
    return SM100_TMEM_STORE_16dp32b16x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b32x>) {
    return SM100_TMEM_STORE_16dp32b32x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b32x_16b>) {
    return SM100_TMEM_STORE_16dp32b32x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b64x>) {
    return SM100_TMEM_STORE_16dp32b64x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b64x_16b>) {
    return SM100_TMEM_STORE_16dp32b64x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b128x>) {
    return SM100_TMEM_STORE_16dp32b128x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_16dp32b128x_16b>) {
    return SM100_TMEM_STORE_16dp32b128x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b1x>) {
    return SM100_TMEM_STORE_32dp32b1x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b1x_16b>) {
    return SM100_TMEM_STORE_32dp32b1x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b2x>) {
    return SM100_TMEM_STORE_32dp32b2x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b2x_16b>) {
    return SM100_TMEM_STORE_32dp32b2x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b4x>) {
    return SM100_TMEM_STORE_32dp32b4x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b4x_16b>) {
    return SM100_TMEM_STORE_32dp32b4x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b8x>) {
    return SM100_TMEM_STORE_32dp32b8x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b8x_16b>) {
    return SM100_TMEM_STORE_32dp32b8x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b16x>) {
    return SM100_TMEM_STORE_32dp32b16x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b16x_16b>) {
    return SM100_TMEM_STORE_32dp32b16x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b32x>) {
    return SM100_TMEM_STORE_32dp32b32x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b32x_16b>) {
    return SM100_TMEM_STORE_32dp32b32x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b64x>) {
    return SM100_TMEM_STORE_32dp32b64x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b64x_16b>) {
    return SM100_TMEM_STORE_32dp32b64x_16b{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b128x>) {
    return SM100_TMEM_STORE_32dp32b128x{};
  }
  else if constexpr (is_same_v<CopyOp, SM100_TMEM_LOAD_32dp32b128x_16b>) {
    return SM100_TMEM_STORE_32dp32b128x_16b{};
  }
  else {
    static_assert(dependent_false<CopyOp>, "No TMEM_STORE matching for provided TMEM_LOAD");
  }
}

} // namespace TMEM

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// UTCCP Copy Traits
//
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace SM100::TMEM::UTCCP {

//
// Specialized copy_unpack implementation for SM100::TMEM::UTCCP instructions
//

template <class CopyOp,
          class TS, class SLayout,
          class TD, class DLayout>
CUTE_HOST_DEVICE constexpr
void
copy_unpack(Copy_Traits<CopyOp> const&,
            Tensor<TS,SLayout>  const& src,
            Tensor<TD,DLayout>       & dst)
{
  static_assert(is_rmem<TS>::value, "Expected smem_desc src for SM100_UTCCP");
  static_assert(is_tmem<TD>::value, "Expected tmem dst for SM100_UTCCP");
  CopyOp::copy(src[0], raw_pointer_cast(dst.data()));
}

} // end namespace SM100::TMEM::UTCCP

// In the following UTCCP traits, the ValID is representing:
// logical_bit_idx -> tmem_addr_offset.
// And the logical_bit_idx is numbered in the order of:
// [core_matrix_strided, core_matrix_leading, broadcast, repeat].
// The first two modes provide convenience for smem_desc construtction.
// The last two modes provide boradcast transformation for 4x32DP and 2x64DP.
// With above, the strides of first two modes are neccessary to be TMEM::DP_b and 1.
// And the stride of the third mode in the SrcLayout must be zero.

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::UTCCP::SM100_UTCCP_128dp256bit_1cta;

template <>
struct Copy_Traits<SM100_UTCCP_128dp256bit_1cta>
{
  using ThrID = Layout<_1>;
  using ValID = Layout<Shape <_128,      _256>,
                       Stride<TMEM::DP_b, _1>>;
  using SrcLayout = Layout<Shape<_1, _32768>,
                           Stride<_0, _1>>;
  using DstLayout = Layout<Shape<_1, _32768>,
                           Stride<_0,_1>>;
  using RefLayout = DstLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::UTCCP::SM100_UTCCP_128dp256bit_2cta;

template <>
struct Copy_Traits<SM100_UTCCP_128dp256bit_2cta>
{
  using ThrID = Layout<_2>;
  using ValID = typename Copy_Traits<SM100_UTCCP_128dp256bit_1cta>::ValID;
  using SrcLayout = Layout<Shape <_2, _32768>,
                           Stride<_0, _1>>;
  using DstLayout = Layout<Shape <_2, _32768>,
                           Stride<_0, _1>>;
  using RefLayout = DstLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::UTCCP::SM100_UTCCP_128dp128bit_1cta;

template <>
struct Copy_Traits<SM100_UTCCP_128dp128bit_1cta>
{
  using ThrID = Layout<_1>;
  using ValID = Layout<Shape <_128,      _128>,
                       Stride<TMEM::DP_b, _1>>;
  using SrcLayout = Layout<Shape<_1, _16384>,
                           Stride<_0, _1>>;
  using DstLayout = Layout<Shape<_1, _16384>,
                           Stride<_0,_1>>;
  using RefLayout = DstLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::UTCCP::SM100_UTCCP_128dp128bit_2cta;

template <>
struct Copy_Traits<SM100_UTCCP_128dp128bit_2cta>
{
  using ThrID = Layout<_2>;
  using ValID = typename Copy_Traits<SM100_UTCCP_128dp128bit_1cta>::ValID;
  using SrcLayout = Layout<Shape <_2, _16384>,
                           Stride<_0, _1>>;
  using DstLayout = Layout<Shape <_2, _16384>,
                           Stride<_0, _1>>;
  using RefLayout = DstLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::UTCCP::SM100_UTCCP_4dp256bit_1cta;

template <>
struct Copy_Traits<SM100_UTCCP_4dp256bit_1cta>
{
  /*
  4DP is really hard to model if we consider this instruction as a "copy" instruction.
  But, if we take it as "TMEM refresh" instruction, then everything goes out naturally.
  4DP utccp is designed to refresh the last 4 lanes of each tmem subpartition.
  So, in the kernel implementation, we usually only don't need to iterate on MMA_M dimension,
  but only need to iterate on MMA_K dimension.
  And in each refresh, logically we are refreshing MMA's 128 rows M + 256bit K.
  So the "atom_v" should be (refresh_m, refresh_k) instead of (copy_m, copy_k).
  And the Src/DstLayout below is: copy_bits -> logical_refresh_bits.
  */

  using ThrID = Layout<_1>;
  using ValID = Layout<Shape <_128,    _256>,
                       Stride<TMEM::DP_b,_1>>;
  using SrcLayout = Layout<Shape <_1,Shape <_4, _256>>,
                           Stride<_0,Stride<_32,_128>>>;
  using DstLayout = Layout<Shape <_1,Shape <_4, _256>>,
                           Stride<_0,Stride<_32,_128>>>;
  using RefLayout = DstLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::UTCCP::SM100_UTCCP_4dp256bit_2cta;

template <>
struct Copy_Traits<SM100_UTCCP_4dp256bit_2cta>
{
  using ThrID = Layout<_2>;
  using ValID = typename Copy_Traits<SM100_UTCCP_4dp256bit_1cta>::ValID;
  using SrcLayout = Layout<Shape <_2,Shape <_4, _256>>,
                           Stride<_0,Stride<_32,_128>>>;
  using DstLayout = Layout<Shape <_2,Shape <_4, _256>>,
                           Stride<_0,Stride<_32,_128>>>;
  using RefLayout = DstLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta;

template <>
struct Copy_Traits<SM100_UTCCP_4x32dp128bit_1cta>
{
  using _DP = TMEM::DP_b;
  using _DPx32 = Int<_DP{}*32>;

  using ThrID = Layout<_1>;
  // logical bit_idx -> tmem_addr
  // [core_matrix_strided, core_matrix_leading, broadcast]
  using ValID = Layout<Shape <_32,_128,_4>,
                       Stride<_DP,_1,  _DPx32>>;
  using SrcLayout = Layout<Shape <_1,Shape <_32,_128,_4>>,
                           Stride<_0,Stride<_1, _32, _0>>>;
  using DstLayout = Layout<Shape <_1,_16384>,
                           Stride<_0,_1>>;
  using RefLayout = DstLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_2cta;

template <>
struct Copy_Traits<SM100_UTCCP_4x32dp128bit_2cta>
{
  using ThrID = Layout<_2>;
  using ValID = typename Copy_Traits<SM100_UTCCP_4x32dp128bit_1cta>::ValID;
  using SrcLayout = Layout<Shape <_2,Shape <_32,_128,_4>>,
                           Stride<_0,Stride<_1, _32, _0>>>;
  using DstLayout = Layout<Shape<_2, _16384>,
                           Stride<_0,_1>>;
  using RefLayout = DstLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::UTCCP::SM100_UTCCP_2x64dp128bitlw0213_1cta;

template <>
struct Copy_Traits<SM100_UTCCP_2x64dp128bitlw0213_1cta>
{
  using _DP = TMEM::DP_b;
  using _DPx64 = Int<_DP{}*64>;

  using ThrID = Layout<_1>;
  // logical bit_idx -> tmem_addr
  // [core_matrix_strided, core_matrix_leading, broadcast]
  using ValID = Layout<Shape <_64,_128,_2>,
                       Stride<_DP,_1,  _DPx64>>;
  using SrcLayout = Layout<Shape <_1,Shape <_64,_128,_2>>,
                           Stride<_0,Stride<_1, _64, _0>>>;
  using DstLayout = Layout<Shape<_1, _16384>,
                           Stride<_0, _1>>;
  using RefLayout = DstLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::UTCCP::SM100_UTCCP_2x64dp128bitlw0213_2cta;

template <>
struct Copy_Traits<SM100_UTCCP_2x64dp128bitlw0213_2cta>
{
  using ThrID = Layout<_2>;
  using ValID = typename Copy_Traits<SM100_UTCCP_2x64dp128bitlw0213_1cta>::ValID;

  using SrcLayout = Layout<Shape <_2,Shape <_64,_128,_2>>,
                           Stride<_0,Stride<_1, _64, _0>>>;
  using DstLayout = Layout<Shape<_2, _16384>,
                           Stride<_0, _1>>;
  using RefLayout = DstLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::UTCCP::SM100_UTCCP_2x64dp128bitlw0123_1cta;

template <>
struct Copy_Traits<SM100_UTCCP_2x64dp128bitlw0123_1cta>
{
  using _DP = TMEM::DP_b;
  using _DPx32 = Int<_DP{}*32>;
  using _DPx64 = Int<_DP{}*64>;

  using ThrID = Layout<_1>;
  // logical bit_idx -> tmem_addr
  // [core_matrix_strided, core_matrix_leading, repeat, broadcast]
  using ValID = Layout<Shape <_32,_128,_2,    _2>,
                       Stride<_DP,_1  ,_DPx64,_DPx32>>;

  using SrcLayout = Layout<Shape <_1,Shape <_32,_128,_2,_2>>,
                           Stride<_0,Stride<_1, _32,_4096,_0>>>;
  using DstLayout = Layout<Shape<_1, _16384>,
                           Stride<_0, _1>>;
  using RefLayout = DstLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using SM100::TMEM::UTCCP::SM100_UTCCP_2x64dp128bitlw0123_2cta;

template <>
struct Copy_Traits<SM100_UTCCP_2x64dp128bitlw0123_2cta>
{
  using ThrID = Layout<_2>;
  using ValID = typename Copy_Traits<SM100_UTCCP_2x64dp128bitlw0123_1cta>::ValID;
  using SrcLayout = Layout<Shape <_2,Shape <_32,_128,_2,_2>>,
                           Stride<_0,Stride<_1, _32, _4096,_0>>>;
  using DstLayout = Layout<Shape <_2,_16384>,
                           Stride<_0,_1>>;
  using RefLayout = DstLayout;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <class CopyOp,
          class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
auto
make_utccp_copy(CopyOp const&,
                Tensor<TEngine,TLayout> const& tmem)
{
  static_assert(is_tmem<TEngine>::value, "Expected TMEM tensor.");
  using T      = typename TEngine::value_type;
  using Traits = Copy_Traits<CopyOp>;
  using Atom   = Copy_Atom<Traits, T>;

  // atom thr idx -> tmem addr    This is the T in the Layout_TV
  auto atom_t_layout = make_layout(size(typename Traits::ThrID{}), Int<0>{});
  // atom val idx -> tmem addr    Cast the CopyOp's value ids to the proper data width
  auto atom_v_layout = coalesce(upcast<sizeof_bits<T>::value>(typename Traits::ValID{}));

  return make_cotiled_copy(Atom{}, make_layout(atom_t_layout, atom_v_layout), tmem.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cute

