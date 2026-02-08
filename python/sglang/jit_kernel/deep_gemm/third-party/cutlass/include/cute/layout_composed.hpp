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

#include <cute/config.hpp>                     // CUTE_HOST_DEVICE, CUTE_GCC_UNREACHABLE
#include <cute/layout.hpp>                     // cute::tuple
#include <cute/numeric/integral_constant.hpp>  // cute::true_type, cute::false_type, cute::Int

/* This implements a ComposedLayout of the form
 *   LayoutA o Offset o LayoutB
 * and is useful in cases where composition() does not or cannot apply to LayoutA and LayoutB.
 * For example, when the "divisibility condition" is violated in composition(LayoutA, LayoutB).
 *
 * This ComposedLayout provides similar functionality to Layout including tiling, partitioning,
 * coordinate-to-index mapping and layout manipulations, but is not considered a "normal" layout.
 * For example, this layout provides shape() and size() functions, but does not provide stride() functions.
 * Mostly, the similar functionality is accomplished by applying each operation to LayoutB only
 * as LayoutB defines the domain.
 */

namespace cute
{

// A Layout of non-trivially composable functions: F o I o L
template <class LayoutA, class Offset, class LayoutB>
struct ComposedLayout : private cute::tuple<LayoutA, Offset, LayoutB>  // EBO for static layouts
{
  CUTE_HOST_DEVICE constexpr
  ComposedLayout(LayoutA const& layoutA = {},
                 Offset  const& offset  = {},
                 LayoutB const& layoutB = {})
      : cute::tuple<LayoutA, Offset, LayoutB>(layoutA, offset, layoutB)
  {}

  //
  // Accessors
  //

  static constexpr int rank  = LayoutB::rank;

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  layout_a() const {
    return get<0>(static_cast<cute::tuple<LayoutA, Offset, LayoutB> const&>(*this));
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  offset() const {
    return get<1>(static_cast<cute::tuple<LayoutA, Offset, LayoutB> const&>(*this));
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  layout_b() const {
    return get<2>(static_cast<cute::tuple<LayoutA, Offset, LayoutB> const&>(*this));
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  layout() const {
    return *this;
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  shape() const {
    return layout_b().shape();
  }

  // Doesn't really make sense to ask for the strides of this "layout"
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  stride() const = delete;

  //
  // Mappings
  //

  // Map a logical coordinate to a linear index (Coord has no Underscore slice operators)
  // OR
  // Slice the layout and return the sublayout (Coord has an Underscore slice op)
  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(Coord const& coord) const {
    if constexpr (has_underscore<Coord>::value) {
      return slice(coord, *this);
    } else {
      return layout_a()(offset() + layout_b()(coord));    // (A o O o B)(c)
    }

    CUTE_GCC_UNREACHABLE;
  }

  // Convenience function for multi-dimensional coordinates
  template <class Coord0, class Coord1, class... Coords>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(Coord0 const& c0, Coord1 const& c1, Coords const&... cs) const {
    return operator()(make_coord(c0,c1,cs...));
  }

  //
  // Compose
  //

  template <class OtherLayout>
  CUTE_HOST_DEVICE constexpr
  auto
  compose(OtherLayout const& other) const {
    return composition(*this, other);
  }

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr
  auto
  compose(Layouts const&... layouts) const {
    return composition(*this, make_tile(layouts...));
  }

  template <class OtherShape>
  CUTE_HOST_DEVICE constexpr
  auto
  with_shape(OtherShape const& shape) const {
    return composition(*this, make_layout(shape));
  }

  template <class... Shapes>
  CUTE_HOST_DEVICE constexpr
  auto
  with_shape(Shapes const&... shapes) const {
    return composition(*this, make_layout(make_shape(shapes...)));
  }

  //
  // Tile
  //

  template <class OtherLayout>
  CUTE_HOST_DEVICE constexpr
  auto
  tile(OtherLayout const& other) const {
    return tiled_divide(*this, other);
  }

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr
  auto
  tile(Layouts const&... layouts) const {
    return tiled_divide(*this, make_tile(layouts...));
  }

  // Equality, return a static or dynamic boolean
  template <class... Args>
  CUTE_HOST_DEVICE constexpr
  auto
  operator==(ComposedLayout<Args...> const& other) const {
    return this->layout_a() == other.layout_a() &&
           this->layout_b() == other.layout_b() &&
           this->offset()   == other.offset();
  }
};

template <class A, class O, class B>
struct is_layout<ComposedLayout<A,O,B>> : true_type {};

template <class T>
struct is_composed_layout : false_type {};
template <class A, class O, class B>
struct is_composed_layout<ComposedLayout<A,O,B>> : true_type {};

//
// Constructors
//

template <class LayoutA, class Offset, class LayoutB>
CUTE_HOST_DEVICE constexpr
auto
make_composed_layout(LayoutA const& layoutA,
                     Offset  const& offset,
                     LayoutB const& layoutB)
{
  return ComposedLayout<LayoutA, Offset, LayoutB>{layoutA, offset, layoutB};
}

//
// Utilities
//

// Return the layout of a mode
template <int... Is, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
decltype(auto)
layout(ComposedLayout<A,O,B> const& clayout)
{
  return composition(clayout.layout_a(), clayout.offset(), layout<Is...>(clayout.layout_b()));
}

// Return the shape of a mode
template <int... Is, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
decltype(auto)
shape(ComposedLayout<A,O,B> const& layout)
{
  return shape<Is...>(layout.layout_b());
}

// Doesn't make sense to directly ask for the strides of this "layout"
template <int... Is, class Fn, class O, class Layout>
CUTE_HOST_DEVICE constexpr
decltype(auto)
stride(ComposedLayout<Fn,O,Layout> const& layout) = delete;

// Return the number of elements in a mode
template <int... Is, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
decltype(auto)
size(ComposedLayout<A,O,B> const& layout)
{
  return size<Is...>(layout.layout_b());
}

// Return the number of modes
template <int... Is, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
rank(ComposedLayout<A,O,B> const& layout)
{
  return rank<Is...>(layout.layout_b());
}

// Return the depth of the layout
template <int... Is, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
depth(ComposedLayout<A,O,B> const& layout)
{
  return depth<Is...>(layout.layout_b());
}

// Return the codomain size of a mode
template <int... Is, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
cosize(ComposedLayout<A,O,B> const& layout)
{
  return cosize<Is...>(layout.layout_b());
}

//
// Operations to manipulate Layouts like a tuple of pairs
//

template <size_t I, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
get(ComposedLayout<A,O,B> const& a)
{
  return composition(a.layout_a(), a.offset(), get<I>(a.layout_b()));
}

template <int Begin, int End, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
take(ComposedLayout<A,O,B> const& a)
{
  return composition(a.layout_a(), a.offset(), take<Begin,End>(a.layout_b()));
}

template <class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
flatten(ComposedLayout<A,O,B> const& a)
{
  return composition(a.layout_a(), a.offset(), flatten(a.layout_b()));
}

template <int N, class A, class O, class B, class X>
CUTE_HOST_DEVICE constexpr
auto
append(ComposedLayout<A,O,B> const& a, X const& x)
{
  return composition(a.layout_a(), a.offset(), append<N>(a.layout_b(), x));
}

template <int Begin, int End, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
group(ComposedLayout<A,O,B> const& a)
{
  return composition(a.layout_a(), a.offset(), group<Begin,End>(a.layout_b()));
}

//
// Slice a ComposedLayout
//

template <class Coord, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
slice_and_offset(Coord const& coord, ComposedLayout<A,O,B> const& layout)
{
  auto [slice, offset] = slice_and_offset(coord, layout.layout_b());
  return cute::make_tuple(ComposedLayout{layout.layout_a(), layout.offset() + offset, slice}, Int<0>{});
}

template <class Coord, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
slice(Coord const& coord, ComposedLayout<A,O,B> const& layout)
{
  return get<0>(slice_and_offset(coord, layout));
}

// Compute a pointer offset and (potentially modified) layout from a coordinate
// For composed layout tensors the offset is accumulated in the layout itself while pointer is not updated
template <class Coord, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
domain_offset(Coord const& coord, ComposedLayout<A,O,B> const& layout)
{
  return cute::make_tuple(ComposedLayout{layout.layout_a(), layout.offset() + layout.layout_b()(coord), layout.layout_b()}, Int<0>{});
}

//
// composition
//

template <class LayoutA,
          class Offset,
          class LayoutB>
CUTE_HOST_DEVICE constexpr
auto
composition(LayoutA const& layoutA,
            Offset  const& offset,
            LayoutB const& layoutB)
{
  return ComposedLayout<LayoutA, Offset, LayoutB>{layoutA, offset, layoutB};
}

template <class A, class O, class B, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
composition(ComposedLayout<A,O,B> const& a,
            Tiler                 const& b)
{
  return composition(a.layout_a(), a.offset(), composition(a.layout_b(), b));
}

template <class ShapeA, class StrideA,
          class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
composition(Layout<ShapeA,StrideA> const& a,
            ComposedLayout<A,O,B>  const& b)
{
  CUTE_STATIC_ASSERT_V(b.offset() == Int<0>{}, "Require offset == 0.");

  return composition(composition(a, b.layout_a()), b.layout_b());
}

//
// complement
//

template <class A, class O, class B, class CoTarget>
CUTE_HOST_DEVICE constexpr
auto
complement(ComposedLayout<A,O,B> const& layout, CoTarget const& cotarget)
{
  return complement(layout.layout_b(), cotarget);
}

template <class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
complement(ComposedLayout<A,O,B> const& layout)
{
  return complement(layout, cosize(layout));
}

//
// inverse
//

template <class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
right_inverse(ComposedLayout<A,O,B> const& layout)
{
  return composition(right_inverse(layout.layout_b()), right_inverse(layout.offset()), right_inverse(layout.layout_a()));
}

template <class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
left_inverse(ComposedLayout<A,O,B> const& layout)
{
  return composition(left_inverse(layout.layout_b()), left_inverse(layout.offset()), left_inverse(layout.layout_a()));
}

//
// Other operations
//

template <class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
zip(ComposedLayout<A,O,B> const& a)
{
  return composition(a.layout_a(), a.offset(), zip(a.layout_b()));
}

// Partitions

template <class A, class O, class B, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
logical_divide(ComposedLayout<A,O,B> const& a,
               Tiler                 const& b)
{
  return composition(a.layout_a(), a.offset(), logical_divide(a.layout_b(), b));
}

template <class A, class O, class B, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
tile_unzip(ComposedLayout<A,O,B> const& a,
           Tiler                 const& b)
{
  return composition(a.layout_a(), a.offset(), tile_unzip(a.layout_b(), b));
}

template <class A, class O, class B, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
tiled_divide(ComposedLayout<A,O,B> const& a,
             Tiler                 const& b)
{
  return composition(a.layout_a(), a.offset(), tiled_divide(a.layout_b(), b));
}

template <class A, class O, class B, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
zipped_divide(ComposedLayout<A,O,B> const& a,
              Tiler                 const& b)
{
  return composition(a.layout_a(), a.offset(), zipped_divide(a.layout_b(), b));
}

template <class A, class O, class B, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
flat_divide(ComposedLayout<A,O,B> const& a,
            Tiler                 const& b)
{
  return composition(a.layout_a(), a.offset(), flat_divide(a.layout_b(), b));
}

template <class A, class O, class B, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
logical_product(ComposedLayout<A,O,B> const& a,
                Tiler                 const& b)
{
  return composition(a.layout_a(), a.offset(), logical_product(a.layout_b(), b));
}

template <class A, class O, class B, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
zipped_product(ComposedLayout<A,O,B> const& a,
               Tiler                 const& b)
{
  return composition(a.layout_a(), a.offset(), zipped_product(a.layout_b(), b));
}

template <class A, class O, class B, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
tiled_product(ComposedLayout<A,O,B> const& a,
              Tiler                 const& b)
{
  return composition(a.layout_a(), a.offset(), tiled_product(a.layout_b(), b));
}

template <class A, class O, class B, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
flat_product(ComposedLayout<A,O,B> const& a,
             Tiler                 const& b)
{
  return composition(a.layout_a(), a.offset(), flat_product(a.layout_b(), b));
}

template <class A, class O, class B, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
blocked_product(ComposedLayout<A,O,B> const& a,
                Tiler                 const& b)
{
  return composition(a.layout_a(), a.offset(), blocked_product(a.layout_b(), b));
}

template <class A, class O, class B, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
raked_product(ComposedLayout<A,O,B> const& a,
              Tiler                 const& b)
{
  return composition(a.layout_a(), a.offset(), raked_product(a.layout_b(), b));
}

template <class A, class O, class B,
          class Shape, class ModeOrder = GenColMajor>
CUTE_HOST_DEVICE constexpr
auto
tile_to_shape(ComposedLayout<A,O,B> const& layout,
              Shape                 const& trg_shape,
              ModeOrder             const& ord_shape = {})
{
  return composition(layout.layout_a(), layout.offset(), tile_to_shape(layout.layout_b(), trg_shape, ord_shape));
}

template <class A, class O, class B,
          class Shape>
CUTE_HOST_DEVICE constexpr
auto
filter(ComposedLayout<A,O,B> const& layout, Shape const& trg_profile)
{
  return composition(layout.layout_a(), layout.offset(), filter(layout.layout_b(), trg_profile));
}

template <class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
coalesce(ComposedLayout<A,O,B> const& layout)
{
  return composition(layout.layout_a(), layout.offset(), coalesce(layout.layout_b()));
}

template <class A, class O, class B, class Shape>
CUTE_HOST_DEVICE constexpr
auto
coalesce(ComposedLayout<A,O,B> const& layout, Shape const& trg_profile)
{
  return composition(layout.layout_a(), layout.offset(), coalesce(layout.layout_b(), trg_profile));
}


//
// Upcast and Downcast
//

template <int N, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
upcast(ComposedLayout<A,O,B> const& layout)
{
  return composition(upcast<N>(layout.layout_a()), upcast<N>(layout.offset()), upcast<N>(layout.layout_b()));
}

template <int N, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
downcast(ComposedLayout<A,O,B> const& layout)
{
  return composition(downcast<N>(layout.layout_a()), downcast<N>(layout.offset()), downcast<N>(layout.layout_b()));
}


template <class OldType, class NewType,
          class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
recast_layout(ComposedLayout<A,O,B> const& layout)
{
  using scale = decltype(trait_ratio(sizeof_bits<NewType>{}, sizeof_bits<OldType>{}));
  if constexpr (scale::num == 1 && scale::den == 1) {
    return layout;
  }
  else if constexpr (scale::num == 1) {
    return downcast<scale::den>(layout);
  }
  else if constexpr (scale::den == 1) {
    return upcast<scale::num>(layout);
  }
  else {
    return downcast<scale::den>(upcast<scale::num>(layout));
  }
  CUTE_GCC_UNREACHABLE;
}

template <class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
max_alignment(ComposedLayout<A,O,B> const& layout)
{
  // Do not attempt for general ComposedLayouts
  //return gcd(max_alignment(layout.layout_a()), max_alignment(layout.offset()), max_alignment(layout.layout_b()));
  return Int<1>{};
}

template <class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto
nullspace(ComposedLayout<A,O,B> const& layout)
{
  // Do not attempt for general ComposedLayouts
  return Layout<_1,_0>{};
}

//
// Display utilities
//

template <class A, class O, class B>
CUTE_HOST_DEVICE void print(ComposedLayout<A,O,B> const& layout)
{
  print(layout.layout_a()); print(" o "); print(layout.offset()); print(" o "); print(layout.layout_b());
}

#if !defined(__CUDACC_RTC__)
template <class A, class O, class B>
CUTE_HOST std::ostream& operator<<(std::ostream& os, ComposedLayout<A,O,B> const& layout)
{
  return os << layout.layout_a() << " o " << layout.offset() << " o " << layout.layout_b();
}
#endif

} // end namespace cute
