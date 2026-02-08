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

#include <cute/config.hpp>          // CUTE_HOST_DEVICE
#include <cute/tensor_impl.hpp>     // cute::Tensor
#include <cute/atom/copy_atom.hpp>  // cute::Copy_Atom

namespace cute
{

//
// Prefetch global tensors into L2
//

template <uint32_t NumThreads, uint32_t FetchBytes = 64,
          class GEngine, class GLayout>
CUTE_HOST_DEVICE
void
cooperative_prefetch(uint32_t                 const& tid,
                     Tensor<GEngine, GLayout> const& src)
{
  static_assert(is_gmem<GEngine>::value, "Expected global tensor for prefetch");

  constexpr int V = decltype(max_common_vector(src, src))::value;

  if constexpr (V > 1) {
    // L2 sector is 32B, default fetch granularity is 64B
    using VecType = conditional_t<(V * sizeof_bits_v<typename GEngine::value_type>) < (FetchBytes * 8),
                                  ArrayEngine<typename GEngine::value_type, V>,
                                  uint8_t[FetchBytes]                         >;

    Tensor src_v = recast<VecType const>(src);
    CUTE_UNROLL
    for (int i = tid; i < size(src_v); i += NumThreads) {
      prefetch(raw_pointer_cast(&src_v(i)));
    }
  } else {
    CUTE_UNROLL
    for (int i = tid; i < size(src); i += NumThreads) {
      prefetch(raw_pointer_cast(&src(i)));
    }
  }
}

template <class GEngine, class GLayout>
CUTE_HOST_DEVICE
void
prefetch(Tensor<GEngine, GLayout> const& src)
{
  return cooperative_prefetch<1>(0, src);
}

// Prefetch with copy atom
namespace detail {

template <class CopyOp, class = void>
constexpr bool has_prefetch = false;

template <class CopyOp>
constexpr bool has_prefetch<CopyOp, void_t<typename CopyOp::PREFETCH>> = true;

} // end namespace detail

template <class CopyOp, class... CT_Args, class CopyType,
          class GEngine, class GLayout>
CUTE_HOST_DEVICE
void
prefetch(Copy_Atom<Copy_Traits<CopyOp, CT_Args...>, CopyType> const& atom,
         Tensor<GEngine, GLayout>                             const& src)
{
  if constexpr (detail::has_prefetch<CopyOp>) {
    using Prefetch_Traits = Copy_Traits<typename CopyOp::PREFETCH, CT_Args...>;
    using Prefetch_Atom = Copy_Atom<Prefetch_Traits, CopyType>;
    Prefetch_Atom prefetch_atom{atom};
    //auto& dst = const_cast<Tensor<GEngine, GLayout>&>(src); // dst is ignored for prefetch atoms
    Tensor dst = make_tensor(make_smem_ptr<CopyType>(nullptr), shape(src));
    return copy(prefetch_atom, src, dst);
  } else {
    return prefetch(src);
  }
}

#if defined(CUTE_COPY_ATOM_TMA_SM90_ENABLED)
template <class... CT_Args,
          class SrcEngine, class SrcLayout>
CUTE_HOST_DEVICE
void
prefetch(Copy_Traits<SM90_BULK_COPY_AUTO, CT_Args...> const& atom,
         Tensor<SrcEngine, SrcLayout>                 const& src)
{
  using SrcType = typename SrcEngine::value_type;
  static_assert(is_gmem<SrcEngine>::value, "Expected global tensor for L2 prefetch");

  auto tiler = max_common_layout(src, src);
  constexpr int vec_elem = decltype(size(tiler))::value;
  constexpr int vec_bits = vec_elem * sizeof_bits_v<SrcType>;
  static_assert(vec_bits >= 128, "Expected at least 128-bits for BLKCP");

  // Construct a new concrete Atom of the vector size
  auto bulk_atom = Copy_Atom<Copy_Traits<SM90_BULK_COPY_G2S, Int<vec_bits>>, SrcType>{};

  return prefetch(bulk_atom, logical_divide(src, tiler));
}

// Backwards-compat. Throw out any extra Copy_Atom args.
template <class... CT_Args, class... CA_Args,
          class SrcEngine, class SrcLayout>
CUTE_HOST_DEVICE
void
prefetch(Copy_Atom<Copy_Traits<SM90_BULK_COPY_AUTO, CT_Args...>, CA_Args...> const& atom,
         Tensor<SrcEngine, SrcLayout>                                        const& src)
{
  return prefetch(static_cast<Copy_Traits<SM90_BULK_COPY_AUTO, CT_Args...> const&>(atom), src);
}
#endif // #if defined(CUTE_COPY_ATOM_TMA_SM90_ENABLED)

} // end namespace cute
