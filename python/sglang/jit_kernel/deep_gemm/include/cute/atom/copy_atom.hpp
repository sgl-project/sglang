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

#include <cute/config.hpp>                     // CUTE_HOST_DEVICE
#include <cute/tensor_impl.hpp>                // cute::Tensor
#include <cute/util/type_traits.hpp>           // cute::__CUTE_REQUIRES
#include <cute/container/tuple.hpp>            // cute::is_tuple
#include <cute/numeric/integral_constant.hpp>  // cute::is_constant, cute::is_integral
#include <cute/atom/copy_traits.hpp>           // cute::Copy_Traits
#include <cute/atom/mma_atom.hpp>              // cute::TiledMMA

namespace cute
{

template <class... Args>
struct Copy_Atom;

template <class CopyOperation, class CopyInternalType>
struct Copy_Atom<CopyOperation, CopyInternalType> : Copy_Atom<Copy_Traits<CopyOperation>, CopyInternalType>
{};

template <class... Args, class CopyInternalType>
struct Copy_Atom<Copy_Traits<Args...>, CopyInternalType>
  : Copy_Traits<Args...>
{
  using Traits = Copy_Traits<Args...>;

  // Bit and Thr layouts from the Copy_Traits
  using ThrID        = typename Traits::ThrID;
  using BitLayoutSrc = typename Traits::SrcLayout;
  using BitLayoutDst = typename Traits::DstLayout;
  using BitLayoutRef = typename Traits::RefLayout;

  using ValType = CopyInternalType;

  using ValLayoutSrc = decltype(recast_layout<uint1_t, ValType>(BitLayoutSrc{}));
  using ValLayoutDst = decltype(recast_layout<uint1_t, ValType>(BitLayoutDst{}));
  using ValLayoutRef = decltype(recast_layout<uint1_t, ValType>(BitLayoutRef{}));

  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutSrc{}) == size(ThrID{}), "CopyOperation is not valid for Src of ValType.");
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutDst{}) == size(ThrID{}), "CopyOperation is not valid for Dst of ValType.");
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutRef{}) == size(ThrID{}), "CopyOperation is not valid for Ref of ValType.");

  static constexpr int NumValSrc = size<1>(ValLayoutSrc{});
  static constexpr int NumValDst = size<1>(ValLayoutDst{});

  // Additional Trait parameters/transformations
  template <class... TraitsArgs>
  CUTE_HOST_DEVICE
  auto
  with(TraitsArgs&&... args) const {
    auto traits = Traits::with(static_cast<TraitsArgs&&>(args)...);
    return Copy_Atom<decltype(traits), CopyInternalType>{traits};
  }

  //
  // Tensor call interfaces
  //

  // Check and call instruction, or recurse
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_HOST_DEVICE
  void
  call(Tensor<SEngine,SLayout> const& src,
       Tensor<DEngine,DLayout>      & dst) const
  {
    static_assert(SLayout::rank == 1, "Expected rank-1 src tensor");
    static_assert(DLayout::rank == 1, "Expected rank-1 dst tensor");

    if constexpr (is_constant<NumValSrc, decltype(size(src))>::value ||
                  is_constant<NumValDst, decltype(size(dst))>::value) {
      // Dispatch to unpack to execute instruction
      return copy_unpack(static_cast<Traits const&>(*this), src, dst);
    } else if constexpr (is_tuple<decltype(shape(src))>::value &&
                         is_tuple<decltype(shape(dst))>::value) {
      // If the size of the src/dst doesn't match the instruction,
      //   recurse this rank-1 layout by peeling off the mode
      //   ((A,B,C,...)) -> (A,B,C,...)
      return copy(*this, tensor<0>(src), tensor<0>(dst));
    } else {
      static_assert(dependent_false<SEngine>,
                    "CopyAtom: Src/Dst partitioning does not match the instruction requirement.");
    }
  }

  // Accept mutable temporaries
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_HOST_DEVICE
  void
  call(Tensor<SEngine,SLayout> const& src,
       Tensor<DEngine,DLayout>     && dst) const
  {
    return call(src, dst);
  }

  // Check and call instruction, or recurse
  template <class PEngine, class PLayout,
            class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_HOST_DEVICE
  void
  call(Tensor<PEngine,PLayout> const& prd,
       Tensor<SEngine,SLayout> const& src,
       Tensor<DEngine,DLayout>      & dst) const
  {
    static_assert(PLayout::rank == 1, "Expected rank-1 prd tensor");
    static_assert(SLayout::rank == 1, "Expected rank-1 src tensor");
    static_assert(DLayout::rank == 1, "Expected rank-1 dst tensor");

    if constexpr (is_constant<NumValSrc, decltype(size(src))>::value ||
                  is_constant<NumValDst, decltype(size(dst))>::value) {
      // Dispatch to unpack to execute instruction
      Traits const& traits = static_cast<Traits const&>(*this);
      auto has_with_bool = cute::is_valid([](auto t)->void_t<decltype(t.with(true))>{}, traits);
      if constexpr (has_with_bool) {
        copy_unpack(traits.with(prd(Int<0>{})), src, dst);
      } else {
        if (prd(Int<0>{})) { copy_unpack(traits, src, dst); }
      }
    } else if constexpr (is_tuple<decltype(shape(prd))>::value &&
                         is_tuple<decltype(shape(src))>::value &&
                         is_tuple<decltype(shape(dst))>::value) {
      // If the size of the src/dst doesn't match the instruction,
      //   recurse this rank-1 layout by peeling off the mode
      //   ((A,B,C,...)) -> (A,B,C,...)
      return copy_if(*this, tensor<0>(prd), tensor<0>(src), tensor<0>(dst));
    } else {
      static_assert(dependent_false<SEngine>,
                    "CopyAtom: Src/Dst partitioning does not match the instruction requirement.");
    }
  }

  // Accept mutable temporaries
  template <class PEngine, class PLayout,
            class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_HOST_DEVICE
  void
  call(Tensor<PEngine,PLayout> const& prd,
       Tensor<SEngine,SLayout> const& src,
       Tensor<DEngine,DLayout>     && dst) const
  {
    return call(prd, src, dst);
  }
};

//
// A tiling of copy atoms
//

template <class TiledCopy, class ThrIdx>
struct ThrCopy;

template <class Copy_Atom,
          class LayoutCopy_TV,  // (tid,vid) -> coord   [Need not be 2D...]
          class ShapeTiler_MN>  // coord space
struct TiledCopy : Copy_Atom
{
  // Layout information from the CopyAtom
  using AtomThrID     = typename Copy_Atom::ThrID;        // thrid -> thr_idx
  using AtomLayoutSrc = typename Copy_Atom::ValLayoutSrc; // (thr,val) -> offset
  using AtomLayoutDst = typename Copy_Atom::ValLayoutDst; // (thr,val) -> offset
  using AtomLayoutRef = typename Copy_Atom::ValLayoutRef; // (thr,val) -> offset

  using AtomNumThr = decltype(size<0>(AtomLayoutRef{}));
  using AtomNumVal = decltype(size<1>(AtomLayoutRef{}));

  // Layout information for the TiledCopy
  using Tiler_MN       = ShapeTiler_MN;
  using TiledLayout_TV = LayoutCopy_TV;
  using TiledNumThr    = decltype(size<0>(TiledLayout_TV{}));
  using TiledNumVal    = decltype(size<1>(TiledLayout_TV{}));

  CUTE_STATIC_ASSERT_V(TiledNumThr{} % AtomNumThr{} == Int<0>{}, "TiledCopy uses too few thrs for selected CopyAtom");
  CUTE_STATIC_ASSERT_V(TiledNumVal{} % AtomNumVal{} == Int<0>{}, "TiledCopy uses too few vals for selected CopyAtom");

  // Tile a tensor or a layout from shape
  //   (M,N,...)
  // to shape
  //   (Thr,(FrgV,FrgX),(RestM,RestN,...))
  // where
  //   Thr:   The logical threads within the tiled copy.
  //   FrgV:  The values local to a COPY_ATOM Src.
  //   FrgX:  The values tiled across COPY_ATOMs Src.
  //   RestM: The values tiled in M.
  //   RestN: The values tiled in N.
  template <class STensor>
  CUTE_HOST_DEVICE constexpr static
  auto
  tidfrg_S(STensor&& stensor)
  {
    CUTE_STATIC_ASSERT_V(rank(stensor) >= rank(Tiler_MN{}), "Rank of tensor to be partitioned too small.");

    // Tile the stensor and compute the (src-thr, src-val) -> (ref-thr, ref-val) layout
    return tile2thrfrg(zipped_divide(stensor,Tiler_MN{}), right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{}));
  }

  // Tile a tensor or a layout from shape
  //   (M,N,...)
  // to shape
  //   (Thr,(FrgV,FrgX),(RestM,RestN,...))
  // where
  //   Thr:   The logical threads within the tiled copy.
  //   FrgV:  The values local to a COPY_ATOM Dst.
  //   FrgX:  The values tiled across COPY_ATOMs Dst.
  //   RestM: The values tiled in M.
  //   RestN: The values tiled in N.
  template <class DTensor>
  CUTE_HOST_DEVICE constexpr static
  auto
  tidfrg_D(DTensor&& dtensor)
  {
    CUTE_STATIC_ASSERT_V(rank(dtensor) >= rank(Tiler_MN{}), "Rank of tensor to be partitioned too small.");

    // Tile the dtensor and compute the (dst-thr, dst-val) -> (ref-thr, ref-val) layout
    return tile2thrfrg(zipped_divide(dtensor,Tiler_MN{}), right_inverse(AtomLayoutRef{}).compose(AtomLayoutDst{}));
  }

  // Tile a tensor or a layout from shape
  //   ((TileM,TileN,...), (RestM,RestN,...))
  // to shape
  //   (Thr,(FrgV,FrgX),(RestM,RestN,...))
  template <class Tensor, class Ref2TrgLayout>
  CUTE_HOST_DEVICE constexpr static
  auto
  tile2thrfrg(Tensor&& tensor, Ref2TrgLayout const& ref2trg)
  {
    // Take the thrs/vals that the atom is interested in
    // NOTE: Assumes the AtomNumThr are contiguous and identity within TiledThrID
    auto atom_layout_TV = zipped_divide(TiledLayout_TV{}, make_shape(AtomNumThr{}, AtomNumVal{}));
    // ((atom_tid,atom_val),(rest_tid,rest_val)) -> (m,n)

    // Transform to the trg layout
    auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);
    // ((trg_tid,trg_val),(rest_tid,rest_val)) -> (m,n)

    // Transform the thrs mode from thrid to thr_idx
    // NOTE: Assumes the AtomNumThr are contiguous and identity within TiledThrID
    auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1,Shape<_1,_1>>{});
    // ((trg_tid,rest_tid),(trg_val,rest_val)) -> (m,n)

    /// ==================

    // Transform the tile mode
    auto tv_tensor = tensor.compose(thrval2mn, _);
    // ((thrid,val),(RestM,RestN,...))

    // Unfold and return
    return tv_tensor(make_coord(_,_), _);
  }

  // retile_S and retile_D assume they are working with the reference layout -- they are the same
  template <class Tensor>
  CUTE_HOST_DEVICE constexpr static
  auto
  retile(Tensor&& tensor)
  {
    constexpr int R = remove_cvref_t<Tensor>::rank;
    // Assert that AtomLayoutSrc|Dst is identity so we can skip the Ref transformation

    // Assume the first size<0>(tensor) elements are the first val_ids in TiledLayout_TV.
    // Then, we only need the shape+layout of those size<0>(tensor) elements in TiledLayout_TV
    //   and that shape is what we gather from the other modes of tensor

    auto V = size<0>(tensor);

    auto frg_layout_mn = upcast<TiledNumThr{} * V>(right_inverse(TiledLayout_TV{}).with_shape(shape(Tiler_MN{})));
    // (m,n) -> v_idx -- The shape and order of the V inside of TiledLayout_TV

    auto frg_layout_v = zipped_divide(logical_product(make_layout(V), right_inverse(frg_layout_mn)), make_layout(AtomNumVal{}));
    // (atom_vals,rest_vals) -> (v,m,n)

    /// =======

    // Tile the tensor for TileFrg
    auto t_tensor = zipped_divide(tensor, prepend(product_each(shape(frg_layout_mn)), V));
    // ((TileV,TileM,TileN,...),(1,RestM,RestN,...))

    // Transform the tile mode
    auto v_tensor = t_tensor.compose(frg_layout_v, _);
    // ((atom_vals,rest_vals),(1,RM,RN,...))

    // Unfold and return
    return v_tensor(_, append<R>(Int<0>{},_));
  }

  CUTE_HOST_DEVICE constexpr static
  auto
  get_layoutS_TV()
  {
    // (M,N) -> (M,N)
    auto ref_S = make_layout(make_shape(shape(Tiler_MN{}), Int<1>{}));
    // (thr_idx,val_idx) -> (M,N)
    return tile2thrfrg(ref_S, right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{}))(_,_,Int<0>{});
  }

  CUTE_HOST_DEVICE constexpr static
  auto
  get_layoutD_TV()
  {
    // (M,N) -> (M,N)
    auto ref_D = make_layout(make_shape(shape(Tiler_MN{}), Int<1>{}));
    // (thr_idx,val_idx) -> (M,N)
    return tile2thrfrg(ref_D, right_inverse(AtomLayoutRef{}).compose(AtomLayoutDst{}))(_,_,Int<0>{});
  }

  template <class ThrIdx,
            __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE static
  auto
  get_slice(ThrIdx const& thr_idx)
  {
    return ThrCopy<TiledCopy, ThrIdx>(thr_idx);
  }

  template <class ThrIdx,
            __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE  static
  auto
  get_thread_slice(ThrIdx const& thr_idx)
  {
    return get_slice(thr_idx);
  }
};

template <class TiledCopy, class ThrIdx>
struct ThrCopy
{
  ThrIdx thr_idx_;

  CUTE_HOST_DEVICE
  ThrCopy(ThrIdx const& thr_idx) : thr_idx_(thr_idx) {}

  template <class STensor>
  CUTE_HOST_DEVICE
  auto
  partition_S(STensor&& stensor) const {
    //static_assert(sizeof(typename remove_cvref_t<STensor>::value_type) == sizeof(typename TiledCopy::ValType),
    //              "Expected ValType for tiling SrcTensor.");
    auto thr_tensor = make_tensor(static_cast<STensor&&>(stensor).data(), TiledCopy::tidfrg_S(stensor.layout()));
    return thr_tensor(thr_idx_, _, repeat<rank_v<STensor>>(_));
  }

  template <class DTensor>
  CUTE_HOST_DEVICE
  auto
  partition_D(DTensor&& dtensor) const {
    //static_assert(sizeof(typename remove_cvref_t<DTensor>::value_type) == sizeof(typename TiledCopy::ValType),
    //              "Expected ValType for tiling DstTensor.");
    auto thr_tensor = make_tensor(static_cast<DTensor&&>(dtensor).data(), TiledCopy::tidfrg_D(dtensor.layout()));
    return thr_tensor(thr_idx_, _, repeat<rank_v<DTensor>>(_));
  }

  template <class STensor>
  CUTE_HOST_DEVICE static
  auto
  retile_S(STensor&& stensor) {
    // static_assert(sizeof(typename remove_cvref_t<STensor>::value_type) == sizeof(typename TiledCopy::ValType),
    //               "Expected ValType for tiling SrcTensor.");
    return make_tensor(static_cast<STensor&&>(stensor).data(), TiledCopy::retile(stensor.layout()));
  }

  template <class DTensor>
  CUTE_HOST_DEVICE static
  auto
  retile_D(DTensor&& dtensor) {
    // static_assert(sizeof(typename remove_cvref_t<DTensor>::value_type) == sizeof(typename TiledCopy::ValType),
    //               "Expected ValType for tiling DstTensor.");
    return make_tensor(static_cast<DTensor&&>(dtensor).data(), TiledCopy::retile(dtensor.layout()));
  }
};


template <class... Args,
          class LayoutCopy_TV,
          class Tiler>
CUTE_HOST_DEVICE
auto
make_tiled_copy_impl(Copy_Atom<Args...> const& atom,
                     LayoutCopy_TV      const&,
                     Tiler              const&)
{
  return TiledCopy<Copy_Atom<Args...>, LayoutCopy_TV, Tiler>{atom};
}

//
// These tile the Copy_Atom as a whole
//

template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE
auto
make_tiled_copy_A(Copy_Atom<CArgs...> const& copy_atom,
                  TiledMMA<MArgs...>  const& mma)
{
  return make_tiled_copy_impl(copy_atom, mma.get_layoutA_TV(), make_shape(tile_size<0>(mma),tile_size<2>(mma)));
}

template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE
auto
make_tiled_copy_B(Copy_Atom<CArgs...> const& copy_atom,
                  TiledMMA<MArgs...>  const& mma)
{
  return make_tiled_copy_impl(copy_atom, mma.get_layoutB_TV(), make_shape(tile_size<1>(mma),tile_size<2>(mma)));
}

template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE
auto
make_tiled_copy_C(Copy_Atom<CArgs...> const& copy_atom,
                  TiledMMA<MArgs...>  const& mma)
{
  return make_tiled_copy_impl(copy_atom, mma.get_layoutC_TV(), make_shape(tile_size<0>(mma),tile_size<1>(mma)));
}

// returns the smallest tiled copy that can retile LayoutC_TV
// for use with pipelined epilogues with subtiled stores
template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE
auto
make_tiled_copy_C_atom(Copy_Atom<CArgs...> const& copy_atom,
                       TiledMMA<MArgs...>  const& mma)
{
  // Truncate the V-layout to just the Copy_Atom, keep the V-order
  auto layoutC_TV = mma.get_layoutC_TV();
  auto copy_V     = Int<Copy_Atom<CArgs...>::NumValSrc>{};
  CUTE_STATIC_ASSERT_V(copy_V <= size<1>(layoutC_TV));
  auto layout_TV  = composition(layoutC_TV, make_layout(make_shape(size<0>(layoutC_TV), copy_V)));

  // Recompute tiler and restride the TV layout for the new tiler

  // Tiler -- Find the active elements in the MMA tensor and generate a tiler to extract them
  // Convert to the awkward by-mode tiler to preserve the modes of the tiled MMA
  auto mma_tiler = make_shape(tile_size<0>(mma),tile_size<1>(mma));
  auto mma_zeros = repeat_like(mma_tiler, Int<0>{});

  auto tiler = transform(make_seq<rank(mma_tiler)>{}, [&](auto i) {
    return filter(composition(make_layout(mma_tiler, replace<i>(mma_zeros, Int<1>{})), layout_TV));
  });

  // Layout_TV -- Find the (tid,vid) -> tile coord transformation
  // Apply the tiler to a reference and transform the codomain
  // tile_coord -> mma_coord
  auto tile2mma = composition(make_layout(mma_tiler), tiler);

  // (tid,vid) -> tile_coord
  auto layout_tv = composition(left_inverse(tile2mma), layout_TV);

  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}

/** Produce a TiledCopy from logical thread and values layouts.
 * The thread and value layouts map coordinates to thr_idx and val_idx.
 *    The product of these layouts is taken to produce the TV layout and the Tiler.
 * Useful when threads and values need very specific mappings onto coordinates
 *    in the target tensors.
 */
template <class... Args,
          class ThrLayout,
          class ValLayout = Layout<_1>>
CUTE_HOST_DEVICE
auto
make_tiled_copy(Copy_Atom<Args...> const& copy_atom,
                ThrLayout          const& thr_layout = {},     // (m,n) -> thr_idx
                ValLayout          const& val_layout = {})     // (m,n) -> val_idx
{
  // Take the raked_products to compute the Layout_MN
  // (M,N) -> (thr_idx, val_idx)
  auto layout_mn = raked_product(thr_layout, val_layout);
  // (thr_idx, val_idx) -> (M,N)
  auto layout_tv = right_inverse(layout_mn).with_shape(make_shape(size(thr_layout), size(val_layout)));
  // Tiler for extracting relevant elements
  // (M,N) -> tensor coord
  auto tiler = product_each(shape(layout_mn));

#if 0
  print("thr_layout: "); print(thr_layout); print("\n");
  print("val_layout: "); print(val_layout); print("\n");
  print("layout_mn : "); print(layout_mn);  print("\n");
  print("layout_tv : "); print(layout_tv);  print("\n");
  print("tiler     : "); print(tiler);      print("\n");
#endif

  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}

/** Produce a TiledCopy from thread and value offset maps.
 * The TV Layout maps threads and values to the codomain of the data_layout.
 * It is verified that the intended codomain is valid within data_layout.
 * Useful when threads and values don't care about owning specific coordinates, but
 *   care more about the vector-width and offsets between them.
 */
template <class... Args, class AtomTVLayout, class DataLayout>
CUTE_HOST_DEVICE constexpr
auto
make_cotiled_copy(Copy_Atom<Args...> const& copy_atom,
                  AtomTVLayout const& atom_tv_layout,   // atom (thr,val) -> data addr
                  DataLayout   const& data_layout)      // coord          -> data addr    The target layout
{
  static_assert(is_static<AtomTVLayout>::value);
  static_assert(is_static<DataLayout>::value);

  // data addr -> data coord    Append 1:0 so off-the-ends get the stride-0
  auto inv_data_layout = make_layout(left_inverse(data_layout), Layout<_1,_0>{});

  // (tid,vid) -> data_coord
  auto layout_tv_data = composition(inv_data_layout, atom_tv_layout);

  // Check validity
  // Append 1:0 to data_layout so that OOB coordinates get the stride-0
  CUTE_STATIC_ASSERT_V(coalesce(composition(make_layout(data_layout, Layout<_1,_0>{}), layout<1>(layout_tv_data))) == coalesce(layout<1>(atom_tv_layout)),
                       "The memory pointed to by AtomTVLayout does not exist in the DataLayout.");
  //
  // Tiler -- Find the active elements in the DATA tensor and generate a tiler to extract them
  //

  // Convert to the awkward by-mode tiler to preserve the modes of the tiled DATA
  auto flat_data_shape = product_each(shape(data_layout));
  auto flat_data_zeros = repeat<rank(flat_data_shape)>(Int<0>{});

  auto tiler = transform(make_seq<rank(flat_data_shape)>{}, [&](auto i) {
    return filter(composition(make_layout(flat_data_shape, replace<i>(flat_data_zeros, Int<1>{})), layout_tv_data));
  });

  //
  // Layout_TV -- Find the (tid,vid) -> tile coord transformation
  //

  // Apply the tiler to a reference and transform the codomain
  // tile_coord -> data_coord
  auto tile2data = composition(make_layout(flat_data_shape), tiler);

  // (tid,vid) -> tile_coord
  auto layout_tv = composition(left_inverse(tile2data), layout_tv_data);
  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}

// Make a TiledCopy out of the copy_atom that matches the Src-Layout of tiled_copy
template <class... Args,
          class TiledCopy>
CUTE_HOST_DEVICE
auto
make_tiled_copy_S(Copy_Atom<Args...> const& copy_atom,
                  TiledCopy          const& tiled_copy)
{
  return make_tiled_copy_impl(copy_atom, tiled_copy.get_layoutS_TV(), typename TiledCopy::Tiler_MN{});
}

// Make a TiledCopy out of the copy_atom that matches the Dst-Layout of tiled_copy
template <class... Args,
          class TiledCopy>
CUTE_HOST_DEVICE
auto
make_tiled_copy_D(Copy_Atom<Args...> const& copy_atom,
                  TiledCopy          const& tiled_copy)
{
  return make_tiled_copy_impl(copy_atom, tiled_copy.get_layoutD_TV(), typename TiledCopy::Tiler_MN{});
}

//
// Size
//

// The logical size of a TileCopy
template <int... I, class... Args>
CUTE_HOST_DEVICE constexpr
auto
tile_size(TiledCopy<Args...> const&)
{
  return size<I...>(typename TiledCopy<Args...>::Tiler_MN{});
}

// The number of threads involved in a TiledCopy
template <class... Args>
CUTE_HOST_DEVICE constexpr
auto
size(TiledCopy<Args...> const&)
{
  return typename TiledCopy<Args...>::TiledNumThr{};
}

//
// Display utilities
//

template <class... Args, class T>
CUTE_HOST_DEVICE
void
print(Copy_Atom<Copy_Traits<Args...>, T> const&)
{
  using Atom = Copy_Atom<Copy_Traits<Args...>, T>;
  print("Copy_Atom\n");
  print("  ThrID:        "); print(typename Atom::ThrID{});        print("\n");
  print("  ValLayoutSrc: "); print(typename Atom::ValLayoutSrc{}); print("\n");
  print("  ValLayoutDst: "); print(typename Atom::ValLayoutDst{}); print("\n");
  print("  ValLayoutRef: "); print(typename Atom::ValLayoutRef{}); print("\n");
  print("  ValueType:    "); print(sizeof_bits<typename Atom::ValType>::value); print("b\n");
}

template <class Atom, class... Args>
CUTE_HOST_DEVICE
void
print(TiledCopy<Atom, Args...> const& copy, char const* pad = "")
{
  using Copy = TiledCopy<Atom, Args...>;
  print("TiledCopy\n");
  print("  Tiler_MN:       "); print(typename Copy::Tiler_MN{});       print("\n");
  print("  TiledLayout_TV: "); print(typename Copy::TiledLayout_TV{}); print("\n");
  print(static_cast<Atom const&>(copy));
}

template <class TiledCopy, class ThrIdx>
CUTE_HOST_DEVICE
void
print(ThrCopy<TiledCopy, ThrIdx> const& thr_copy)
{
  print("ThrCopy\n");
  print("  ThrIdx: "); print(thr_copy.thr_idx_); print("\n");
  print(TiledCopy{});
}

} // end namespace cute

////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cute/atom/copy_traits_sm50.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/copy_traits_sm80.hpp>
#include <cute/atom/copy_traits_sm90.hpp>
#include <cute/atom/copy_traits_sm100.hpp>


// Config
#if (__CUDACC_VER_MAJOR__ >= 12)
#  define CUTE_COPY_ATOM_TMA_SM90_ENABLED
#  define CUTE_COPY_ATOM_TMA_SM100_ENABLED
#endif


#if (!defined(CUTE_COPY_ATOM_TMA_SM90_ENABLED))
#  define CUTE_COPY_ATOM_TMA_SM90_ENABLED
#endif

#if (!defined(CUTE_COPY_ATOM_TMA_SM100_ENABLED))
#  define CUTE_COPY_ATOM_TMA_SM100_ENABLED
#endif


#if defined(CUTE_COPY_ATOM_TMA_SM90_ENABLED)
#include <cute/atom/copy_traits_sm90_tma.hpp>
#endif


#if defined(CUTE_COPY_ATOM_TMA_SM100_ENABLED)
#include <cute/atom/copy_traits_sm100_tma.hpp>
#endif


////////////////////////////////////////////////////////////////////////////////////////////////////
