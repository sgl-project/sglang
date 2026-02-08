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

#include <cute/config.hpp>            // CUTE_HOST_DEVICE
#include <cute/tensor_impl.hpp>       // cute::Tensor
#include <cute/atom/copy_atom.hpp>    // cute::Copy_Atom

namespace cute
{

//
// copy_if -- Predicated Copy
//

template <class PrdTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(PrdTensor                    const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>      & dst)
{
  using SrcType = typename SrcEngine::value_type;
  using DstType = typename DstEngine::value_type;

  CUTE_UNROLL
  for (int i = 0; i < size(dst); ++i) {
    if (pred(i)) {
      dst(i) = static_cast<DstType>(static_cast<SrcType>(src(i)));
    }
  }
}

//
// copy_if -- Predicated CopyAtom
//

// Predicate Tensor is an Actual Tensor
template <class... CopyArgs,
          class PrdEngine, class PrdLayout,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(Copy_Atom<CopyArgs...>       const& copy_atom,
        Tensor<PrdEngine, PrdLayout> const& prd,       // ([V],Rest...)
        Tensor<SrcEngine, SrcLayout> const& src,       // ( V, Rest...)
        Tensor<DstEngine, DstLayout>      & dst)       // ( V, Rest...)
{
  if constexpr (PrdLayout::rank == SrcLayout::rank - 1) {
    // Back-compat ONLY -- Delete?
    copy_if(copy_atom, make_tensor(prd.data(), prepend(prd.layout(), Layout<_1,_0>{})), src, dst);
  } else {
    static_assert(SrcLayout::rank == DstLayout::rank, "CopyAtom rank-mismatch.");
    static_assert(SrcLayout::rank == PrdLayout::rank, "CopyAtom rank-mismatch.");

    if constexpr (SrcLayout::rank == 1) {   // Dispatch the copy
      copy_atom.call(prd, src, dst);
    } else {                                // Loop over all but the first mode
      constexpr int R = SrcLayout::rank;
      Tensor prd_v = group_modes<1,R>(prd);
      Tensor src_v = group_modes<1,R>(src);
      Tensor dst_v = group_modes<1,R>(dst);
      CUTE_UNROLL
      for (int i = 0; i < size<1>(dst_v); ++i) {
        copy_atom.call(prd_v(_,i), src_v(_,i), dst_v(_,i));
      }
    }
  }
}

template <class... CopyArgs,
          class PredTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
[[deprecated("Use a bool-tensor or transform-tensor as predication.")]]
CUTE_HOST_DEVICE
void
copy_if(Copy_Atom<CopyArgs...>       const& copy_atom,
        PredTensor                   const& pred,      // (Rest...)
        Tensor<SrcEngine, SrcLayout> const& src,       // (V,Rest...)
        Tensor<DstEngine, DstLayout>      & dst)       // (V,Rest...)
{
  Tensor tpred = cute::lazy::transform(make_tensor(counting_iterator<int>{}, replace<0>(shape(dst), _1{})), pred);
  return copy_if(copy_atom, tpred, src, dst);
}

//
// copy_if -- AutoCopyAsync
//

template <class PrdTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(AutoCopyAsync                const& cpy,
        PrdTensor                    const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>      & dst)
{
  using SrcElemWithConst = remove_reference_t<typename SrcEngine::reference>;
  using SrcType = typename SrcEngine::value_type;
  using DstType = typename DstEngine::value_type;

  auto copy_op = []() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    if constexpr (is_gmem<SrcEngine>::value && is_smem<DstEngine>::value &&
                  sizeof(SrcType) == sizeof(DstType)) {
      if constexpr (is_const_v<SrcElemWithConst> && sizeof(SrcType) == 16) {
          return SM80_CP_ASYNC_CACHEGLOBAL<SrcType,DstType>{};
      } else if constexpr (sizeof(SrcType) == 4 || sizeof(SrcType) == 8 || sizeof(SrcType) == 16) {
          return SM80_CP_ASYNC_CACHEALWAYS<SrcType,DstType>{};
      } else {
          return UniversalCopy<SrcType,DstType>{};
      }
    } else {
        return UniversalCopy<SrcType,DstType>{};
    }

    CUTE_GCC_UNREACHABLE;
#else
    return UniversalCopy<SrcType,DstType>{};
#endif
  }();

  CUTE_UNROLL
  for (int i = 0; i < size(dst); ++i) {
    if (pred(i)) {
      copy_op.copy(src(i), dst(i));
    }
  }
}

//
// copy -- AutoCopyAsync
//

template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(AutoCopyAsync                const& cpy,
     Tensor<SrcEngine, SrcLayout> const& src,       // (V,Rest...)
     Tensor<DstEngine, DstLayout>      & dst)       // (V,Rest...)
{
  copy_if(cpy, constant_fn<true_type>{}, src, dst);
}

//
// copy -- CopyAtom
//

template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,       // (V,Rest...)
     Tensor<DstEngine, DstLayout>      & dst)       // (V,Rest...)
{
  static_assert(SrcLayout::rank == DstLayout::rank, "CopyAtom rank-mismatch.");

  if constexpr (SrcLayout::rank == 1) {   // Dispatch the copy
    copy_atom.call(src, dst);
  } else {                                // Loop over all but the first mode
    constexpr int R = SrcLayout::rank;
    Tensor src_v = group_modes<1,R>(src);
    Tensor dst_v = group_modes<1,R>(dst);

    if constexpr (is_static<decltype(shape(src_v))>::value && is_static<decltype(shape(dst_v))>::value) {
      CUTE_STATIC_ASSERT_V(size<1>(src_v) == size<1>(dst_v));

      // AutoFilter on the Rest-mode
      auto dst_null = nullspace(layout<1>(dst_v));

      Tensor dst_n = zipped_divide(dst_v, make_tile(shape<0>(dst_v), dst_null));  // ((V, NLL), (_1, Rest))
      Tensor src_n = zipped_divide(src_v, make_tile(shape<0>(src_v), dst_null));  // ((V, NLL), (_1, Rest))

      CUTE_STATIC_ASSERT_V(size<1>(src_n) == size<1>(dst_n));
      CUTE_STATIC_ASSERT_V((cosize<0,1>(dst_n.layout()) == Int<1>{}), "Nullspace definition error");
      CUTE_STATIC_ASSERT_V((cosize<0,1>(src_n.layout()) == Int<1>{}), "Error: Ambiguous scatter detected in copy");
      CUTE_STATIC_ASSERT_V((size<1,0>(dst_n) == Int<1>{}));
      CUTE_STATIC_ASSERT_V((size<1,0>(src_n) == Int<1>{}));

      Tensor dst_c = dst_n(make_coord(_,Int<0>{}),make_coord(Int<0>{},_));        // (V, Rest)
      Tensor src_c = src_n(make_coord(_,Int<0>{}),make_coord(Int<0>{},_));        // (V, Rest)

      CUTE_STATIC_ASSERT_V( size<1>(src_c) ==  size<1>(dst_c));
      CUTE_STATIC_ASSERT_V(shape<0>(dst_c) == shape<0>(dst));
      CUTE_STATIC_ASSERT_V(shape<0>(src_c) == shape<0>(src));

      CUTE_UNROLL
      for (int i = 0; i < size<1>(dst_c); ++i) {
        copy_atom.call(src_c(_,i), dst_c(_,i));
      }
    } else {
      CUTE_UNROLL
      for (int i = 0; i < size<1>(dst_v); ++i) {
        copy_atom.call(src_v(_,i), dst_v(_,i));
      }
    }
  }
}

////////////////////////////////////////////////////////
// Special Auto-Vectorizing, Auto-Filtering Overloads //
////////////////////////////////////////////////////////

// Specialization for AutoVectorizingCopyAssumedAlignment<MaxVecBits>
template <int MaxVecBits,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(AutoVectorizingCopyWithAssumedAlignment<MaxVecBits> const&,
     Tensor<SrcEngine, SrcLayout>                        const& src,
     Tensor<DstEngine, DstLayout>                             & dst)
{
  constexpr int common_elem = CUTE_STATIC_V(max_common_vector(src, dst));
  static_assert(is_integral<decltype(Int<common_elem>{} * sizeof_bits_v<typename DstEngine::value_type>)>::value, "Error: Attempting a subbit write!");

  if constexpr (common_elem > 1)
  {
    constexpr int align_bits = CUTE_STATIC_V(gcd(max_alignment(src), max_alignment(dst), Int<MaxVecBits>{}));
    constexpr int vec_bits   = gcd(common_elem * sizeof_bits_v<typename DstEngine::value_type>, align_bits);

    if constexpr ((vec_bits % 8) == 0 && sizeof_bits_v<typename DstEngine::value_type> < Int<vec_bits>{})
    {
      // If more than one element vectorizes to a multiple of 8bits that is larger than the value_type, then recast and copy
      using VecType = uint_bit_t<vec_bits>;

      // Recast
      Tensor src_v = recast<VecType>(src);
      Tensor dst_v = recast<VecType>(dst);
      return copy_if(constant_fn<true_type>{}, src_v, dst_v);
    } else {
      return copy_if(constant_fn<true_type>{}, src, dst);
    }
  } else {
    return copy_if(constant_fn<true_type>{}, src, dst);
  }
}

template <class Base>
struct AutoFilter {
  Base const& base;
  CUTE_HOST_DEVICE AutoFilter(Base const& b) : base(b) {}
};

// Specialization for AutoFilter
template <class CopyOp,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(AutoFilter<CopyOp>           const& copy_op,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst)
{
  if constexpr (is_constant<true, decltype(size(src) == size(dst))>::value) {
    auto dst_null = nullspace(dst.layout());

    Tensor dst_n = zipped_divide(dst, dst_null);
    Tensor src_n = zipped_divide(src, dst_null);

    CUTE_STATIC_ASSERT_V(cosize<0>(dst_n.layout()) == Int<1>{}, "Nullspace definition error");
    CUTE_STATIC_ASSERT_V(cosize<0>(src_n.layout()) == Int<1>{}, "Error: Ambiguous race-condition detected.");

    copy(copy_op.base, src_n(Int<0>{},_), dst_n(Int<0>{},_));
  } else {
    copy(copy_op.base, src, dst);
  }
}

// Auto-vectorizing copy for static layouts
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst)
{
  if constexpr (is_static<SrcLayout>::value && is_static<DstLayout>::value) {
    // Assume Tensors with static layouts (e.g. registers) have pointers that are 128b aligned
    return copy(AutoFilter(AutoVectorizingCopyWithAssumedAlignment<128>{}), src, dst);
  } else
  if constexpr (is_static<decltype(shape(src))>::value && is_static<decltype(shape(dst))>::value) {
    // Tensors with static shapes can be filtered, but do not assume that dynamic layouts are aligned.
    return copy(AutoFilter(AutoVectorizingCopyWithAssumedAlignment<8>{}), src, dst);
  } else {
    // Do not assume that dynamic layouts are aligned.
    return copy(AutoVectorizingCopyWithAssumedAlignment<8>{}, src, dst);
  }
}

// Auto-vectorizing copy with assumed alignment up to 128bit.
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_aligned(Tensor<SrcEngine, SrcLayout> const& src,
             Tensor<DstEngine, DstLayout>      & dst)
{
  if constexpr (is_static<decltype(shape(src))>::value && is_static<decltype(shape(dst))>::value) {
    // Tensors with static shapes can be filtered
    return copy(AutoFilter(AutoVectorizingCopyWithAssumedAlignment<128>{}), src, dst);
  } else {
    return copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, src, dst);
  }
}

// Specializaton for Atom AutoVectorizingCopyAssumedAlignment
template <int MaxVecBits, class... Args,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<MaxVecBits>, Args...> const&,
     Tensor<SrcEngine, SrcLayout>                                            const& src,
     Tensor<DstEngine, DstLayout>                                                 & dst)
{
  return copy(AutoVectorizingCopyWithAssumedAlignment<MaxVecBits>{}, src, dst);
}

template <int MaxVecBits, class... Args,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<Copy_Traits<AutoVectorizingCopyWithAssumedAlignment<MaxVecBits>>, Args...> const&,
     Tensor<SrcEngine, SrcLayout>                                                         const& src,
     Tensor<DstEngine, DstLayout>                                                              & dst)
{
  return copy(AutoVectorizingCopyWithAssumedAlignment<MaxVecBits>{}, src, dst);
}

#if defined(CUTE_COPY_ATOM_TMA_SM90_ENABLED)
template <class... CT_Args,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Traits<SM90_BULK_COPY_AUTO, CT_Args...> const& atom,  // Copy_Traits may or may not have the memory barrier in it already
     Tensor<SrcEngine, SrcLayout>                 const& src,
     Tensor<DstEngine, DstLayout>                      & dst)
{
  using SrcType = typename SrcEngine::value_type;
  using DstType = typename DstEngine::value_type;
  static_assert(cute::is_same<SrcType, DstType>::value);
  static_assert((is_gmem<SrcEngine>::value && is_smem<DstEngine>::value) ||
                (is_smem<SrcEngine>::value && is_gmem<DstEngine>::value),
                "Bulk Copy only supports gmem -> smem or smem -> gmem movement.");
  // G2S or S2G dispatch
  using BULK_COPY_OP = conditional_t<is_gmem<SrcEngine>::value,
                                     SM90_BULK_COPY_G2S,
                                     SM90_BULK_COPY_S2G>;

  // Find the common subtensor of src and dst
  auto tiler = max_common_layout(src, dst);
  constexpr int vec_elem = decltype(size(tiler))::value;
  constexpr int vec_bits = vec_elem * sizeof_bits_v<SrcType>;
  static_assert(vec_bits >= 128, "Expected at least 128-bits for BLKCP");

  // Construct a new concrete Atom of the vector size
  using BulkAtom = Copy_Atom<Copy_Traits<BULK_COPY_OP, Int<vec_bits>, CT_Args...>, SrcType>;
  auto bulk_atom = apply(atom.opargs_, [](auto const&... args) { return BulkAtom{args...}; });
  return copy(bulk_atom, logical_divide(src, tiler), logical_divide(dst, tiler));
}

// Backwards-compat. Throw out any extra Copy_Atom args.
template <class... CT_Args, class... CA_Args,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<Copy_Traits<SM90_BULK_COPY_AUTO, CT_Args...>, CA_Args...> const& atom,
     Tensor<SrcEngine, SrcLayout>                                        const& src,
     Tensor<DstEngine, DstLayout>                                             & dst)
{
  return copy(static_cast<Copy_Traits<SM90_BULK_COPY_AUTO, CT_Args...> const&>(atom), src, dst);
}
#endif // #if defined(CUTE_COPY_ATOM_TMA_SM90_ENABLED)

//
// Decay TiledCopy to CopyAtom
//

template <class CopyAtom, class TV, class Tiler,
          class PrdTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(TiledCopy<CopyAtom, TV, Tiler> const& tiled_copy,
        PrdTensor                      const& pred,
        Tensor<SrcEngine, SrcLayout>   const& src,
        Tensor<DstEngine, DstLayout>        & dst)
{
  return copy_if(static_cast<CopyAtom const&>(tiled_copy), pred, src, dst);
}

template <class CopyAtom, class TV, class Tiler,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(TiledCopy<CopyAtom, TV, Tiler> const& tiled_copy,
     Tensor<SrcEngine, SrcLayout>   const& src,
     Tensor<DstEngine, DstLayout>        & dst)
{
  return copy(static_cast<CopyAtom const&>(tiled_copy), src, dst);
}

template <class TiledCopy, class ThrIdx,
          class PrdTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(ThrCopy<TiledCopy, ThrIdx>   const& thr_copy,
        PrdTensor                    const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>      & dst) = delete;

template <class TiledCopy, class ThrIdx,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(ThrCopy<TiledCopy, ThrIdx>   const& thr_copy,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst) = delete;

//
// Catch uncaught policies
//

template <class CopyPolicy,
          class PredTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(CopyPolicy                   const& cpy,
        PredTensor                   const& prd,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>      & dst)
{
  static_assert(dependent_false<CopyPolicy>, "Unrecognized CopyPolicy.");
}

template <class CopyPolicy,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(CopyPolicy                   const& cpy,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst)
{
  static_assert(dependent_false<CopyPolicy>, "Unrecognized CopyPolicy.");
}

//
// Accept mutable temporaries
//

template <class PrdTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(PrdTensor                    const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>     && dst)
{
  return copy_if(pred, src, dst);
}

template <class CopyPolicy,
          class PrdTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(CopyPolicy                   const& copy_policy,
        PrdTensor                    const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>     && dst)
{
  return copy_if(copy_policy, pred, src, dst);
}

template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>     && dst)
{
  return copy(src, dst);
}

template <class CopyPolicy,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(CopyPolicy                   const& copy_policy,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>     && dst)
{
  return copy(copy_policy, src, dst);
}

template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_aligned(Tensor<SrcEngine, SrcLayout> const& src,
             Tensor<DstEngine, DstLayout>     && dst)
{
  return copy_aligned(src, dst);
}

} // end namespace cute
