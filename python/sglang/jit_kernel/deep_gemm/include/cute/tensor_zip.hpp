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

#include <cute/config.hpp>           // CUTE_HOST_DEVICE
#include <cute/tensor_impl.hpp>      // cute::Tensor
#include <cute/container/tuple.hpp>  // cute::tuple

namespace cute
{

// A tuple of Iterators that can be offset asymmetrically
// Note that this only accepts op+(tuple<Index...>) and op[tuple<Index...>]
//   where each iterator will be offset by its respective index only.
// READ-ONLY for now until cute::tuple can be constructed with references.
template <class... Iters>
struct ZipIterator
{
  using value_type   = cute::tuple<iter_value_t<Iters>...>;
  using element_type = cute::tuple<iter_element_t<Iters>...>;
  // NOTE: cute::tuple does not support constructions with references at the moment.
  //       Consider fixes and/or an implementation of std::forward_as_tuple.
  //       For now, use a cute::tuple of value_types instead, which makes this Iterator READ-ONLY.
  //using reference    = cute::tuple<iter_reference_t<Iters>...>;
  using reference  = value_type;

  ZipIterator() = delete;

  CUTE_HOST_DEVICE constexpr
  ZipIterator(Iters... iters)
    : iters_(iters...)
  {}

  CUTE_HOST_DEVICE constexpr
  ZipIterator(cute::tuple<Iters...> const& iters)
    : iters_(iters)
  {}

  CUTE_HOST_DEVICE constexpr
  reference operator*() const {
    return cute::apply(iters_, [](auto&&... args) { return reference(*args...); });
  }

  template <class... Index>
  CUTE_HOST_DEVICE constexpr
  ZipIterator operator+(cute::tuple<Index...> const& idxs) const {
    static_assert(sizeof...(Index) == sizeof...(Iters), "Expect same number of offsets as iterators.");
    return cute::transform(iters_, idxs, [](auto&& iter, auto&& idx) { return iter + idx; });
  }

  template <class... Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](cute::tuple<Index...> const& idxs) const {
    return *(*this + idxs);
  }

  cute::tuple<Iters...> iters_;
};

//------------------------------------------------------------------------------
// type traits

template <class... Iters>
struct is_rmem<ZipIterator<Iters...>> : conjunction<is_rmem<Iters>...> {};
template <class... Iters>
struct is_smem<ZipIterator<Iters...>> : conjunction<is_smem<Iters>...> {};
template <class... Iters>
struct is_gmem<ZipIterator<Iters...>> : conjunction<is_gmem<Iters>...> {};
template <class... Iters>                                                   
struct is_tmem<ZipIterator<Iters...>> : conjunction<is_tmem<Iters>...> {};  

// A tuple of Layouts that operates on each Layout symmetrically
// The Layouts need to have compatible shapes and ranks.
// The ZipLayout presents the intersection of the domain of its component Layouts.
//   E.g. all Layouts accept 1D coords and ZipLayout does as well.
// The ZipLayout returns the union of the codomain of its component Layouts.
//   E.g. all Layouts return an integer so ZipLayout returns a tuple of integers.
template <class... Layouts>
struct ZipLayout
{
  static constexpr int rank = (int(0) | ... | Layouts::rank);

  static_assert((is_layout<Layouts>::value && ...), "All template parameters must be layouts");
  static_assert(((Layouts::rank == rank) && ...),   "All layouts must have the same rank");

  CUTE_HOST_DEVICE constexpr
  ZipLayout(Layouts const&... layouts)
    : layouts_(layouts...)
  {}

  CUTE_HOST_DEVICE constexpr
  ZipLayout(cute::tuple<Layouts...> const& layouts)
    : layouts_(layouts)
  {}

  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(Coord const& coord) const {
    if constexpr (has_underscore<Coord>::value) {
      return ZipLayout(cute::transform(layouts_, [&] (auto layout) { return layout(coord); }));
    } else {
      return cute::transform(layouts_, [&] (auto layout) { return layout(coord); });
    }

    CUTE_GCC_UNREACHABLE;
  }

  // op() convenience function for multi-dimensional coordinates
  template <class Coord0, class Coord1, class... Coords>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator()(Coord0 const& c0, Coord1 const& c1, Coords const&... cs) const {
    return operator()(make_coord(c0,c1,cs...));
  }

  cute::tuple<Layouts...> layouts_;
};

template <class... Layouts>
struct is_layout<ZipLayout<Layouts...>> : true_type {};

//
// make_zip_tensor and unzip_tensor
//

template <class... Engines, class... Layouts>
CUTE_HOST_DEVICE constexpr
auto
make_zip_tensor(Tensor<Engines,Layouts> const&... tensors)
{
  return make_tensor(ZipIterator(tensors.data()...),
                     ZipLayout(tensors.layout()...));
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
unzip_tensor(Tensor<Engine,Layout> const& tensor)
{
  return cute::transform(tensor.data().iters_, tensor.layout().layouts_,
                         [](auto iter, auto layout) { return make_tensor(iter, layout); });
}

//
// Utilities
//

template <int... Is, class... Layouts>
CUTE_HOST_DEVICE constexpr
auto
rank(ZipLayout<Layouts...> const& layouts)
{
  return rank<Is...>(get<0>(layouts.layouts_));
}

template <int... Is, class... Layouts>
CUTE_HOST_DEVICE constexpr
auto
size(ZipLayout<Layouts...> const& layouts)
{
  return size<Is...>(get<0>(layouts.layouts_));
}

//
// Manipulation
//

// Extend each component layout to rank-N by appending Layout @a x.
template <int N, class... Layouts, class ShapeX = _1, class StrideX = _0>
CUTE_HOST_DEVICE constexpr
auto
append(ZipLayout<Layouts...>  const& layouts,
       Layout<ShapeX,StrideX> const& x = {})
{
  return ZipLayout(cute::transform(layouts.layouts_, [&](auto t){ return append<N>(t, x); }));
}

// Extend each component layout to rank-N by prepending Layout @a x.
template <int N, class... Layouts, class ShapeX = _1, class StrideX = _0>
CUTE_HOST_DEVICE constexpr
auto
prepend(ZipLayout<Layouts...>  const& layouts,
        Layout<ShapeX,StrideX> const& x = {})
{
  return ZipLayout(cute::transform(layouts.layouts_, [&](auto t){ return prepend<N>(t, x); }));
}

template <class... Layouts, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
logical_divide(ZipLayout<Layouts...> const& layouts,
               Tiler                 const& tiler)
{
  return ZipLayout(cute::transform(layouts.layouts_, [&](auto t){ return logical_divide(t, tiler); }));
}

template <class... Layouts, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
zipped_divide(ZipLayout<Layouts...> const& layouts,
              Tiler                 const& tiler)
{
  return ZipLayout(cute::transform(layouts.layouts_, [&](auto t){ return zipped_divide(t, tiler); }));
}

// Return <SlicedZipLayout, ZipOffsets> by calling slice_and_offset and all component layouts.
template <class Coord, class... Layouts>
CUTE_HOST_DEVICE constexpr
auto
slice_and_offset(Coord const& c, ZipLayout<Layouts...> const& layouts)
{
  auto result = cute::zip(cute::transform(layouts.layouts_, [&c](auto const& layout) { return slice_and_offset(c, layout); }));
  return cute::make_tuple(ZipLayout(get<0>(result)), get<1>(result));
}

} // end namespace cute
