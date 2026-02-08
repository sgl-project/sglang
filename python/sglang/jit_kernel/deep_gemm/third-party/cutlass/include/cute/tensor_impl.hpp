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
/*! \file
    \brief This file contains the definition of Tensor as well as classes/functions most closely associated with it.

    For backwards-compatibility, "tensor.hpp" is the "entrypoint" header for a collection of classes and utilities
    that are adjacent to Tensor, e.g. fill(). Whereas this file contains the actual definition of Tensor and
    a small set of functions central to its usage.

    Within the CUTLASS codebase, favor not including "tensor.hpp" wherever possible; instead include "tensor_impl.hpp"
    along with other specific headers that you need. This helps to avoid circular includes and to reduce build time.
*/

#pragma once

#include <cute/config.hpp>                     // CUTE_HOST_DEVICE
#include <cute/layout.hpp>                     // cute::Shape
#include <cute/layout_composed.hpp>            // cute::is_composed_layout
#include <cute/pointer.hpp>                    // cute::recast_ptr
#include <cute/pointer_base.hpp>               // cute::iterator_traits
#include <cute/container/array_aligned.hpp>    // cute::array_aligned
#include <cute/container/array_subbyte.hpp>    // cute::array_subbyte
#include <cute/container/tuple.hpp>            // cute::tuple
#include <cute/numeric/integral_constant.hpp>  // cute::is_integral
#include <cute/util/type_traits.hpp>           // __CUTE_REQUIRES

namespace cute
{

//
// Engine -- owning or non-owning data store
//

// concept Engine {
//   using iterator     = ;
//   using value_type   = ;
//   using element_type = ;
//   using reference    = ;
//   iterator begin();
// };

template <class T, size_t N>
struct ArrayEngine
{
  using Storage = typename conditional<(sizeof_bits<T>::value % 8 == 0),
                                       array_aligned<T,N>,
                                       array_subbyte<T,N>>::type;
  using iterator     = typename Storage::iterator;
  using reference    = typename iterator_traits<iterator>::reference;
  using element_type = typename iterator_traits<iterator>::element_type;
  using value_type   = typename iterator_traits<iterator>::value_type;
  Storage storage_;

  CUTE_HOST_DEVICE constexpr auto begin() const { return storage_.begin(); }
  CUTE_HOST_DEVICE constexpr auto begin()       { return storage_.begin(); }
};

// Specialization for sparse_elem<S,T> tensor allocation/iteration
// NOTE: This can and should be used for allocation of SMEM as well!
//       Fuse these two ArrayEngines?
template <int S, class T, size_t N>
struct ArrayEngine<sparse_elem<S,T>, N>
{
  static_assert(N % S == 0, "Expected a multiple of the sparsity.");
  using value_type   = sparse_elem<S,T>;
  using Storage      = typename conditional<(sizeof_bits<T>::value % 8 == 0),
                                            array_aligned<T,N/S>,
                                            array_subbyte<T,N/S>>::type;
  using iterator     = sparse_ptr<S,sparse_elem<S,T>*>;
  using reference    = typename iterator_traits<iterator>::reference;
  using element_type = typename iterator_traits<iterator>::element_type;
  Storage storage_;

  CUTE_HOST_DEVICE constexpr auto begin() const { return recast_ptr<value_type>(storage_.begin()); }
  CUTE_HOST_DEVICE constexpr auto begin()       { return recast_ptr<value_type>(storage_.begin()); }
};

template <class Iterator>
struct ViewEngine
{
  using iterator     = Iterator;
  using reference    = typename iterator_traits<iterator>::reference;
  using element_type = typename iterator_traits<iterator>::element_type;
  using value_type   = typename iterator_traits<iterator>::value_type;
  iterator storage_;

  CUTE_HOST_DEVICE constexpr iterator const& begin() const { return storage_; }
  CUTE_HOST_DEVICE constexpr iterator      & begin()       { return storage_; }
};

template <class Iterator>
struct ConstViewEngine
{
  using iterator     = Iterator;
  using reference    = typename iterator_traits<iterator>::reference;
  using element_type = typename iterator_traits<iterator>::element_type;
  using value_type   = typename iterator_traits<iterator>::value_type;
  iterator storage_;

  CUTE_HOST_DEVICE constexpr iterator const& begin() const { return storage_; }
};

//
// Tensor
//

template <class Engine, class Layout>
struct Tensor
{
  using iterator     = typename Engine::iterator;
  using value_type   = typename Engine::value_type;
  using element_type = typename Engine::element_type;
  using reference    = typename Engine::reference;

  using engine_type  = Engine;
  using layout_type  = Layout;

  CUTE_HOST_DEVICE constexpr
  Tensor() {}

  CUTE_HOST_DEVICE constexpr
  Tensor(Engine const& engine, Layout const& layout)
      : rep_(layout, engine) {
  }

  //
  // Accessors
  //

  static constexpr int rank  = Layout::rank;

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  tensor() const {
    return *this;
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  engine() const {
    return get<1>(rep_);
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  engine() {
    return get<1>(rep_);
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  data() const {
    return engine().begin();
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  data() {
    return engine().begin();
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  layout() const {
    return get<0>(rep_);
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  shape() const {
    return layout().shape();
  }

  CUTE_HOST_DEVICE constexpr
  auto
  size() const {
    return cute::size(shape());
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  stride() const {
    return layout().stride();
  }

  //
  // Indexing op() and op[]
  //

  // Index into this tensor like an array by computing the offset via layout()
  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator[](Coord const& coord) {
    return data()[layout()(coord)];
  }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator[](Coord const& coord) const {
    return data()[layout()(coord)];
  }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator()(Coord const& coord) {
    if constexpr (has_underscore<Coord>::value) {
      auto [sliced_layout,offset] = slice_and_offset(coord, layout());
      return make_tensor(data() + offset, sliced_layout);
    } else {
      return data()[layout()(coord)];
    }

    CUTE_GCC_UNREACHABLE;
  }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator()(Coord const& coord) const {
    if constexpr (has_underscore<Coord>::value) {
      auto [sliced_layout,offset] = slice_and_offset(coord, layout());
      return make_tensor(data() + offset, sliced_layout);
    } else {
      return data()[layout()(coord)];
    }

    CUTE_GCC_UNREACHABLE;
  }

  // op() convenience function for multi-dimensional coordinates
  template <class Coord0, class Coord1, class... Coords>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator()(Coord0 const& c0, Coord1 const& c1, Coords const&... cs) {
    return operator()(make_coord(c0,c1,cs...));
  }

  template <class Coord0, class Coord1, class... Coords>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator()(Coord0 const& c0, Coord1 const& c1, Coords const&... cs) const {
    return operator()(make_coord(c0,c1,cs...));
  }

  //
  // Compose
  //

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr
  auto
  compose(Layouts const&... layouts) {
    return make_tensor(data(), layout().compose(layouts...));
  }

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr
  auto
  compose(Layouts const&... layouts) const {
    return make_tensor(data(), layout().compose(layouts...));
  }

  //
  // Tile
  //

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr
  auto
  tile(Layouts const&... layouts) {
    return make_tensor(data(), layout().tile(layouts...));
  }

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr
  auto
  tile(Layouts const&... layouts) const {
    return make_tensor(data(), layout().tile(layouts...));
  }

  //
  // Utility
  //

  template <class Int,
            __CUTE_REQUIRES(is_integral<Int>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_1d_coord(Int const& linear_idx) const {
    return layout().get_1d_coord(linear_idx);
  }

  template <class Int,
            __CUTE_REQUIRES(is_integral<Int>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_hier_coord(Int const& linear_idx) const {
    return layout().get_hier_coord(linear_idx);
  }

  template <class Int,
            __CUTE_REQUIRES(is_integral<Int>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_flat_coord(Int const& linear_idx) const {
    return layout().get_flat_coord(linear_idx);
  }

  cute::tuple<layout_type, engine_type> rep_;
};

template <class T>
struct is_tensor : false_type {};
template <class Engine, class Layout>
struct is_tensor<Tensor<Engine,Layout>> : true_type {};
template <class T>
constexpr bool is_tensor_v = is_tensor<T>::value;

// Customization point for creation of owning and non-owning Tensors
template <class T>
struct MakeTensor
{
  template <class Arg0, class... Args>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Arg0 const& arg0, Args const&... args) const
  {
    if constexpr (has_dereference<Arg0>::value) {
      // Construct a non-owning Tensor
      using Engine = ViewEngine<Arg0>;
      if constexpr (sizeof...(Args) == 1 && (is_layout<Args>::value && ...)) {
        // Forward a Layout
        return Tensor{Engine{arg0}, args...};
      } else {
        // Construct a Layout from Args
        return Tensor{Engine{arg0}, make_layout(args...)};
      }
    } else {
      // Construct an owning Tensor
      static_assert((is_static<Arg0>::value && ... && is_static<Args>::value),
                    "Dynamic owning tensors not supported");
      if constexpr (sizeof...(Args) == 0 && is_layout<Arg0>::value) {
        // Forward a Layout
        using Layout = Arg0;
        using Engine = ArrayEngine<T, cosize_v<Layout>>;
        return Tensor<Engine,Layout>();
      } else {
        // Construct a Layout from Args
        using Layout = decltype(make_layout(arg0, args...));
        using Engine = ArrayEngine<T, cosize_v<Layout>>;
        return Tensor<Engine,Layout>();
      }
    }

    CUTE_GCC_UNREACHABLE;
  }
};

//
// make_tensor
//

// Make an owning Tensor that will allocate a static array
// e.g. make_tensor<float>(Int<12>{})
template <class T, class... Args>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Args const&... args)
{
  static_assert((not has_dereference<Args>::value && ...), "Expected layout args... in make_tensor<T>(args...)");
  return MakeTensor<T>{}(args...);
}

// Make a non-owning Tensor that will use a pointer (view)
// e.g. make_tensor(vec.data(), 12)
template <class Iterator, class... Args>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Iterator const& iter, Args const&... args)
{
  static_assert(has_dereference<Iterator>::value, "Expected iterator iter in make_tensor(iter, args...)");
  static_assert((not has_dereference<Args>::value && ...), "Expected layout args... in make_tensor(iter, args...)");
  return MakeTensor<Iterator>{}(iter, args...);
}

//
// make_tensor_like
//   Make a register tensor the same type and shape and (if possible) order as another tensor
//

template <class NewT, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_tensor_like(Layout const& layout)
{
  return make_tensor<NewT>(make_layout_like(layout));
}

template <class NewT, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_tensor_like(Tensor<Engine,Layout> const& tensor)
{
  return make_tensor_like<NewT>(tensor.layout());
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_tensor_like(Tensor<Engine,Layout> const& tensor)
{
  return make_tensor_like<typename Engine::value_type>(tensor.layout());
}

//
// make_fragment_like
//   Make a tensor the same shape and (if possible) order as another tensor, with special
//   consideration of the 0th mode. The 0th mode is commonly used for MMA_Atoms or Copy_Atoms
//   so this allocates the 0th mode with LayoutLeft regardless of the reference layout.
//

template <class NewT, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_fragment_like(Layout const& layout)
{
  return make_tensor<NewT>(make_fragment_like(layout));
}

template <class NewT, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_fragment_like(Tensor<Engine,Layout> const& tensor)
{
  return make_fragment_like<NewT>(tensor.layout());
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_fragment_like(Tensor<Engine,Layout> const& tensor)
{
  return make_fragment_like<typename Engine::value_type>(tensor.layout());
}

//
// make_coord_tensor
//   Make a tensor from a layout by binding it to a counting iter with 0-offset of the same profile as the codomain.
//

template <class Layout, __CUTE_REQUIRES(is_layout<Layout>::value)>
CUTE_HOST_DEVICE constexpr
auto
make_coord_tensor(Layout const& layout)
{
  return make_tensor(make_inttuple_iter(coprofile(layout)), layout);
}

//
// make_identity_tensor
//   Make a tensor that maps coordinates within a shape to themselves.
//

template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_identity_tensor(Shape const& shape)
{
  return make_coord_tensor(make_identity_layout(shape));
}

//
// Utilities
//

// Return the subtensor of a mode
template <int... Is, class Tensor>
CUTE_HOST_DEVICE constexpr
auto
tensor(Tensor&& tensor)
{
  if constexpr (sizeof...(Is) == 0) {
    return tensor;
  } else {
    return make_tensor(tensor.data(), get<Is...>(tensor.layout()));
  }

  CUTE_GCC_UNREACHABLE;
}

// Return the layout of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
layout(Tensor<Engine,Layout> const& tensor)
{
  return layout<Is...>(tensor.layout());
}

// Return the shape of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
shape(Tensor<Engine,Layout> const& tensor)
{
  return shape<Is...>(tensor.layout());
}

// Return the stride of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
stride(Tensor<Engine,Layout> const& tensor)
{
  return stride<Is...>(tensor.layout());
}

// Return the number of elements in a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
size(Tensor<Engine,Layout> const& tensor)
{
  return size<Is...>(tensor.layout());
}

// Return the rank of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
rank(Tensor<Engine,Layout> const& tensor)
{
  return rank<Is...>(tensor.layout());
}

// Return the depth of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
depth(Tensor<Engine, Layout> const& tensor)
{
  return depth<Is...>(tensor.layout());
}

//
// Operations to manipulate Tensors like a Layout or IntTuple
//   These are implemented with explicit modifier overloads because these
//   methods likely also have a general IntTuple overload that can shadow.
//

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
flatten(Tensor<Engine,Layout> const& tensor) {
  return make_tensor(tensor.data(), flatten(tensor.layout()));
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
flatten(Tensor<Engine,Layout>& tensor) {
  return make_tensor(tensor.data(), flatten(tensor.layout()));
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
flatten(Tensor<Engine,Layout>&& tensor) {
  return make_tensor(tensor.data(), flatten(tensor.layout()));
}

template <class Engine, class Layout, class Profile = Int<1>>
CUTE_HOST_DEVICE constexpr
auto
coalesce(Tensor<Engine,Layout> const& tensor, Profile const& profile = {}) {
  return make_tensor(tensor.data(), coalesce(tensor.layout(), profile));
}

template <class Engine, class Layout, class Profile = Int<1>>
CUTE_HOST_DEVICE constexpr
auto
coalesce(Tensor<Engine,Layout>& tensor, Profile const& profile = {}) {
  return make_tensor(tensor.data(), coalesce(tensor.layout(), profile));
}

template <class Engine, class Layout, class Profile = Int<1>>
CUTE_HOST_DEVICE constexpr
auto
coalesce(Tensor<Engine,Layout>&& tensor, Profile const& profile = {}) {
  return make_tensor(tensor.data(), coalesce(tensor.layout(), profile));
}

// Replace the modes in layout that have a 0-stride with a 1-size
template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
filter_zeros(Tensor<Engine,Layout> const& tensor) {
  return make_tensor(tensor.data(), filter_zeros(tensor.layout()));
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
filter_zeros(Tensor<Engine,Layout>& tensor) {
  return make_tensor(tensor.data(), filter_zeros(tensor.layout()));
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
filter_zeros(Tensor<Engine,Layout>&& tensor) {
  return make_tensor(tensor.data(), filter_zeros(tensor.layout()));
}

template <class Engine, class Layout, class Profile>
CUTE_HOST_DEVICE constexpr
auto
filter_zeros(Tensor<Engine,Layout> const& tensor, Profile const& profile)
{
  return make_tensor(tensor.data(), filter_zeros(tensor.layout(), profile));
}

template <class Engine, class Layout, class Profile>
CUTE_HOST_DEVICE constexpr
auto
filter_zeros(Tensor<Engine,Layout>& tensor, Profile const& profile)
{
  return make_tensor(tensor.data(), filter_zeros(tensor.layout(), profile));
}

template <class Engine, class Layout, class Profile>
CUTE_HOST_DEVICE constexpr
auto
filter_zeros(Tensor<Engine,Layout>&& tensor, Profile const& profile)
{
  return make_tensor(tensor.data(), filter_zeros(tensor.layout(), profile));
}

// Remove all of the 0-strides and 1-sizes
template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
filter(Tensor<Engine,Layout> const& tensor) {
  return make_tensor(tensor.data(), filter(tensor.layout()));
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
filter(Tensor<Engine,Layout>& tensor) {
  return make_tensor(tensor.data(), filter(tensor.layout()));
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
filter(Tensor<Engine,Layout>&& tensor) {
  return make_tensor(tensor.data(), filter(tensor.layout()));
}

// Group the modes [B,E) into a single mode
// e.g. group<2,4>(make_tensor<int>(Layout<Shape<_1,_2,_3,_4,_5,_6>>{}))
//      => make_tensor<int>(Layout<Shape<_1,_2,Shape<_3,_4>,_5,_6>>{})
template <int B, int E, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
group_modes(Tensor<Engine,Layout> const& tensor) {
  return make_tensor(tensor.data(), group<B,E>(tensor.layout()));
}

template <int B, int E, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
group_modes(Tensor<Engine,Layout>& tensor) {
  return make_tensor(tensor.data(), group<B,E>(tensor.layout()));
}

template <int B, int E, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
group_modes(Tensor<Engine,Layout>&& tensor) {
  return make_tensor(tensor.data(), group<B,E>(tensor.layout()));
}

// Return the subtensor of a range of modes
template <int B, int E, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
take(Tensor<Engine,Layout> const& tensor) {
  return make_tensor(tensor.data(), take<B,E>(tensor.layout()));
}

template <int B, int E, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
take(Tensor<Engine,Layout>& tensor) {
  return make_tensor(tensor.data(), take<B,E>(tensor.layout()));
}

template <int B, int E, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
take(Tensor<Engine,Layout>&& tensor) {
  return make_tensor(tensor.data(), take<B,E>(tensor.layout()));
}

// Return a tensor with the same shape as input but offset by a given coordinate
template <class Coord, class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
domain_offset(Coord const& coord, Tensor&& tensor)
{
  auto [layout, ptr_offset] = domain_offset(coord, tensor.layout());
  return make_tensor(static_cast<Tensor&&>(tensor).data() + ptr_offset, layout);
}

//
// Recast
//

// NOTE: This is very dangerous to do
//   -- doesn't check dynamic integer divisibility
//   -- doesn't check alignment

template <class NewType_, class Tensor>
CUTE_HOST_DEVICE constexpr
auto
recast(Tensor&& tensor)
{
  using OldType = typename remove_cvref_t<Tensor>::element_type;
  using NewType = copy_cv_t<OldType, NewType_>;

  if constexpr (is_same<NewType, OldType>::value) {
    return make_tensor(static_cast<Tensor&&>(tensor).data(), tensor.layout());
  } else {
    auto old_layout = tensor.layout();
    auto new_layout = recast_layout<OldType,NewType>(old_layout);

    // If this is an upcast of a normal Layout with static negative strides, then offset as well
    if constexpr (sizeof(OldType) < sizeof(NewType) && not is_composed_layout<decltype(old_layout)>::value) {
      auto shape_diff = transform(flatten(old_layout.shape()), flatten(new_layout.shape()), minus{});
      auto extent_diff = transform(shape_diff, flatten(old_layout.stride()), multiplies{});
      auto offset = fold(extent_diff, Int<0>{}, [](auto const& i, auto const& a) { return i + cute::min(a,Int<0>{}); });

      return make_tensor(recast_ptr<NewType>(static_cast<Tensor&&>(tensor).data() + offset), new_layout);
    } else {
      return make_tensor(recast_ptr<NewType>(static_cast<Tensor&&>(tensor).data()         ), new_layout);
    }
  }

  CUTE_GCC_UNREACHABLE;
}

//
// max_common_vector
//

/* Return Int<N> such that N is the maximum number of contiguous elements
 * that logically correspond in the tensors of @a a and @a b. This is,
 * the number of elements that could reasonably be vectorized into a single load/store.
 *
 * @returns Int<N> with N >= 0
 *
 * A return value of Int<0> indicates that no such conclusion can be made and no
 * vectorization should be attempted.
 *
 * Note that the return value does NOT include alignment concerns such as the pointer value and
 * the divisibility of dynamic strides.
 */
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr
auto
max_common_vector(Tensor<SrcEngine,SrcLayout> const& a,
                  Tensor<DstEngine,DstLayout> const& b)
{
  using SrcType = typename SrcEngine::value_type;
  using SrcRef  = typename SrcEngine::reference;
  using DstType = typename DstEngine::value_type;
  using DstRef  = typename DstEngine::reference;

  // Determine if vectorization candidates at all
  if constexpr (// Should be the same value_types, else the copy is also performing a cast
                cute::is_same<SrcType, DstType>::value &&
                // The types should be trivially copyable so that vectorization is valid
                is_trivially_copyable<SrcType>::value &&
                is_trivially_copyable<DstType>::value &&
                // Should be load/storing real data, rather than implicit iterators or such
                is_reference<SrcRef>::value &&
                is_reference<DstRef>::value)
  {
    return max_common_vector(a.layout(), b.layout());
  } else {
    return Int<0>{};
  }

  CUTE_GCC_UNREACHABLE;
}

/* Return a layout that points to the maximum number of contiguous elements
 * that logically correspond in the tensors of @a a and @a b. This is,
 * the elements that could reasonably be "vectorized" into a single load/store.
 *
 * @returns Layout R such that composition(a.layout(), R) and composition(b.layout(), R)
 *          are both identity Layouts.
 *
 * Note that the returned layout does NOT include alignment concerns such as the pointer value and
 * the divisibility of dynamic strides.
 */
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr
auto
max_common_layout(Tensor<SrcEngine,SrcLayout> const& a,
                  Tensor<DstEngine,DstLayout> const& b)
{
  using SrcType = typename SrcEngine::value_type;
  using SrcRef  = typename SrcEngine::reference;
  using DstType = typename DstEngine::value_type;
  using DstRef  = typename DstEngine::reference;

  // Determine if vectorization candidates at all
  if constexpr (// Should be the same value_types, else the copy is also performing a cast
                cute::is_same<SrcType, DstType>::value &&
                // The types should be trivially copyable so that vectorization is valid
                is_trivially_copyable<SrcType>::value &&
                is_trivially_copyable<DstType>::value &&
                // Should be load/storing real data, rather than implicit iterators or such
                is_reference<SrcRef>::value &&
                is_reference<DstRef>::value)
  {
    return max_common_layout(a.layout(), b.layout());
  } else {
    return Layout<_1,_0>{};
  }

  CUTE_GCC_UNREACHABLE;
}

/* Return the maximum (statically known) alignment of a Tensor in the number of bits
 */
template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
max_alignment(Tensor<Engine,Layout> const& t)
{
  return gcd(max_alignment(t.data()),
             max_alignment(t.layout()) * static_value<sizeof_bits<typename Engine::value_type>>());
}

//
// Key algebraic operations -- Composition, Divide, and Product
//

// Apply a Tiler to the Tensor via composition.
template <class Tensor, class Tiler,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
composition(Tensor    && tensor,
            Tiler const& tiler)   // Layout or Tile<Layout...> or Shape
{
  return make_tensor(static_cast<Tensor&&>(tensor).data(),
                     composition(tensor.layout(), tiler));
}

// Apply a Tiler to the Tensor.
//
// Consider a Tensor with shape (A,B,x,y)
// And a Tiler that is:
//
// * A Layout with shape (BLK_A,BLK_B)
// ** Result Tensor shape ((BLK_A,BLK_B),Rest).
// ** That is, the Tensor and Tile are treated as 1D for the tiling.
// ** See logical_divide(Layout,Layout)
//
// * A Tile<Layout...> with shape <BLK_A,BLK_B>
// ** Result Tensor shape ((BLK_A,a),(BLK_B,b),x,y).
// ** Each mode of the Tile<Layout...> is applied to the corresponding mode of the Tensor.
// ** See logical_divide(Layout,Tuple)
//
// * A Shape (BLK_A,BLK_B)
// ** Result Tensor shape ((BLK_A,a),(BLK_B,b),x,y).
// ** Equivalent to applying Tile<BLK_A:_1,BLK_B:_1>.
// ** See logical_divide(Layout,Tuple) and logical_divide(Layout,Int)
//
// Note that the Tile<Layout...>/Shape Tilers must be weakly_congruent to the Tensor
template <class Tensor, class Tiler,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
logical_divide(Tensor    && tensor,
               Tiler const& tiler)   // Layout or Tile<Layout...> or Shape
{
  return make_tensor(static_cast<Tensor&&>(tensor).data(),
                     logical_divide(tensor.layout(), tiler));
}

// zipped_divide is logical_divide with Tiler modes and Rest modes gathered together: (Tiler,Rest)
// When Tiler is Layout, this has no effect as logical_divide results in the same.
// When Tiler is Tile<Layout...> or Shape, this zips modes into standard form ((BLK_A,BLK_B),(a,b,x,y))
template <class Tensor, class Tiler,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
zipped_divide(Tensor    && tensor,
              Tiler const& tiler)    // Layout or Tile<Layout...> or Shape
{
  return make_tensor(static_cast<Tensor&&>(tensor).data(),
                     zipped_divide(tensor.layout(), tiler));
}

// tiled_divide is zipped_divide with the second output mode flattened ((BLK_A,BLK_B),a,b,x,y)
template <class Tensor, class Tiler,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
tiled_divide(Tensor    && tensor,
             Tiler const& tiler)     // Layout or Tile<Layout...> or Shape
{
  return make_tensor(static_cast<Tensor&&>(tensor).data(),
                     tiled_divide(tensor.layout(), tiler));
}

// flat_divide is zipped_divide with the both modes flattened (BLK_A,BLK_B,a,b,x,y)
template <class Tensor, class Tiler,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
flat_divide(Tensor    && tensor,
            Tiler const& tiler)      // Layout or Tile<Layout...> or Shape
{
  return make_tensor(static_cast<Tensor&&>(tensor).data(),
                     flat_divide(tensor.layout(), tiler));
}

// logical_product on a Tensor doesn't make sense since it often increases cosize
//   though this might make sense for creating Tensors with broadcasted (stride-0) modes

//
// Tensor partitioning utilities
//

// Apply a Tiler to the Tensor, then slice out one of those tiles by slicing into the "Rest" modes.
// With an inner_partition, you get everything that's inside the Tiler. Everything that the Tiler is pointing to.
// Split the modes of tensor according to the Tiler
//   zipped_divide returns something like ((BLK_A,BLK_B,...),(a,b,...,x,y))
// Then slice into the second mode (the "Rest" mode) with Coord
template <class Tensor, class Tiler, class Coord,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
inner_partition(Tensor    && tensor,
                Tiler const& tiler,
                Coord const& coord)
{
  auto tensor_tiled = zipped_divide(static_cast<Tensor&&>(tensor), tiler);
  constexpr int R0 = decltype(rank<0>(tensor_tiled))::value;

  // The coord slices into the second mode (the "rest" mode), flatten the first
  if constexpr (is_tuple<Coord>::value) {
    // Append trailing modes if coord is tuple
    constexpr int R1 = decltype(rank<1>(tensor_tiled))::value;
    return tensor_tiled(repeat<R0>(_), append<R1>(coord,_));
  } else {
    // Flat indexing if coord is not tuple
    return tensor_tiled(repeat<R0>(_), coord);
  }
}

// Apply a Tiler to the Tensor, then slice out the remainder by slicing into the "Tile" modes.
// With an outer_partition, you get everything that's outside the Tiler. The layout of the Tile in the Tensor.
// Split the modes of tensor according to the Tiler
//   zipped_divide returns something like ((BLK_A,BLK_B,...),(a,b,...,x,y))
// Then slice into the first mode (the "Tile" mode) with Coord
template <class Tensor, class Tiler, class Coord,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
outer_partition(Tensor    && tensor,
                Tiler const& tiler,
                Coord const& coord)
{
  auto tensor_tiled = zipped_divide(static_cast<Tensor&&>(tensor), tiler);
  constexpr int R1 = decltype(rank<1>(tensor_tiled))::value;

  // The coord slices into the first mode (the "tile" mode), flatten the second
  if constexpr (is_tuple<Coord>::value) {
    // Append trailing modes if coord is tuple
    constexpr int R0 = decltype(rank<0>(tensor_tiled))::value;
    return tensor_tiled(append<R0>(coord,_), repeat<R1>(_));
  } else {
    // Flat indexing if coord is not tuple
    return tensor_tiled(coord, repeat<R1>(_));
  }
}

// Tile a tensor according to @a tiler and use @a coord to index into the remainder, keeping the tile.
// This is typical at the CTA level where tiles of data are extracted:
//   Tensor data = ...                                                                         // (  M,  N)
//   Tensor cta_data = local_tile(data, Shape<_32,_64>{}, make_coord(blockIdx.x,blockIdx.y));  // (_32,_64)
template <class Tensor, class Tiler, class Coord,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
local_tile(Tensor    && tensor,
           Tiler const& tiler,   // tiler to apply
           Coord const& coord)   // coord to slice into "remainder"
{
  return inner_partition(static_cast<Tensor&&>(tensor),
                         tiler,
                         coord);
}

// Same as above, but with a projection parameter to strip out unwanted tiling modes for convenience
//   when using projections of the same tiler.
// This is typical at the CTA level where tiles of data are extracted as projections:
//   Tensor dataA = ...                                                        // (M,K)
//   Tensor dataB = ...                                                        // (N,K)
//   Tensor dataC = ...                                                        // (M,N)
//   auto cta_tiler = Shape<_32, _64, _4>{};
//   auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
//   Tensor ctaA = local_tile(dataA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (_32,_4,k)
//   Tensor ctaB = local_tile(dataB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (_64,_4,k)
//   Tensor ctaC = local_tile(dataC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (_32,_64)
template <class Tensor, class Tiler, class Coord, class Proj,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE
auto
local_tile(Tensor    && tensor,
           Tiler const& tiler,   // tiler to apply
           Coord const& coord,   // coord to slice into "remainder"
           Proj  const& proj)    // projection to apply to tiler and coord
{
  return local_tile(static_cast<Tensor&&>(tensor),
                    dice(proj, tiler),
                    dice(proj, coord));
}

// Tile a tensor according to the flat shape of a layout that provides the coordinate of the target index.
// This is typical at the Thread level where data is partitioned across repeated patterns of threads:
//   Tensor data = ...                                                            // (_16,_64)
//   Tensor thr_data = local_partition(data, Layout<Shape<_2,_16>>{}, thr_idx);   // ( _8, _4)
template <class Tensor, class LShape, class LStride, class Index,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE
auto
local_partition(Tensor                     && tensor,
                Layout<LShape,LStride> const& tile,    // coord -> index
                Index                  const& index)   // index to slice for
{
  static_assert(is_integral<Index>::value);
  return outer_partition(static_cast<Tensor&&>(tensor),
                         product_each(shape(tile)),
                         tile.get_flat_coord(index));
}

// Same as above, but with a projection parameter to strip out unwanted tiling modes for convenience
//   when using projections of the same tiler.
// This is typical at the Thread level where data is partitioned across projected layouts of threads:
//   Tensor dataA = ...                                                            // (M,K)
//   Tensor dataB = ...                                                            // (N,K)
//   Tensor dataC = ...                                                            // (M,N)
//   auto thr_layout = Layout<Shape<_2,_16,_1>, Stride<_16,_1,_0>>{};
//   Tensor thrA = local_partition(dataA, thr_layout, thr_idx, Step<_1, X,_1>{});  // (M/2,K/1)
//   Tensor thrB = local_partition(dataB, thr_layout, thr_idx, Step< X,_1,_1>{});  // (N/16,K/1)
//   Tensor thrC = local_partition(dataC, thr_layout, thr_idx, Step<_1,_1, X>{});  // (M/2,N/16)
template <class Tensor, class LShape, class LStride, class Index, class Projection,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE
auto
local_partition(Tensor                     && tensor,
                Layout<LShape,LStride> const& tile,   // coord -> index
                Index                  const& index,  // index to slice for
                Projection             const& proj)
{
  return local_partition(static_cast<Tensor&&>(tensor),
                         dice(proj, tile),
                         index);
}

//
// Display utilities
//

template <class Engine, class Layout>
CUTE_HOST_DEVICE void print(Tensor<Engine,Layout> const& tensor)
{
  print(tensor.data()); print(" o "); print(tensor.layout());
}

} // end namespace cute

