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
    \brief Additional permutation information for the example.
*/

#include "cutlass/layout/permute.h"
#include "cutlass/gemm/gemm.h"

namespace example
{

using namespace cute;

// This struct is specialized below for different CUTLASS 2.x permutation ops
// to describe the operation in terms of target CuTe shape and stride order.
template<class Permute>
struct PermuteTraits {};

// Use X as a placeholder for shape division result
using X = Underscore;

// Reshape a rank-2 shape into a multidimensional shape.
// Input:
//   shape = (A, B, ...)
//   target_shape = ((A1, ..., X, ..., Am), (B1, ..., X, ..., Bn), ...)
// Output:
//   ((A1, ..., A/prod(A1..Am), ..., Am), (B1, ..., B/prod(B1..Bn), ..., Bn), ...)
template<class Shape, class TargetShape>
constexpr auto
reshape(Shape const& shape, TargetShape const& target_shape)
{
  if constexpr (is_tuple<Shape>::value) {
    return cute::transform(shape, target_shape, [](auto && s, auto && t){ return reshape(s, t); });
  }
  else {
    auto idx = find_if(target_shape, [](auto x){ return is_underscore<decltype(x)>{}; });
    constexpr int I = decltype(idx)::value;
    static_assert(I < tuple_size_v<TargetShape>, "Each mode of TargetShape must contain a placeholder X");
    auto divisors = remove<I>(target_shape);
    assert(shape % product(divisors) == 0);
    return replace<I>(target_shape, shape / product(divisors));
  }
}

// Given a tensor layout, compute a permutation layout consisting of:
// - sub-modes corresponding to the implied multidimensional shape of the source tensor
// - strides accounting for the permutation operation being performed
template<class Permute, bool Transpose, class Shape, class Stride>
constexpr auto
make_permute_layout(Layout<Shape,Stride> const& layout) {
  static_assert(cute::rank(Shape{}) == 3, "Only rank-3 layouts are supported");
  if constexpr (Transpose) {
    // Deal with tensor B by transposing appropriately before and after computing the permute layout.
    // Its CuTe-canonical mode order is [N,K,L], while permute operations expect [row,col,batch].
    return select<1,0,2>(make_permute_layout<Permute, false>(select<1,0,2>(layout)));
  }
  else {
    if constexpr (cutlass::layout::is_trivial_permute<Permute>) {
      // Special case for NoPermute. Use a depth-2 layout for consistency with other permutations.
      using ShapeProfile = tuple<tuple<X>, tuple<X>, tuple<X>>;
      return unflatten(layout, ShapeProfile{});
    }
    else {
      // Here's where the permutation layout is actually built
      using ShapeProfile = typename PermuteTraits<Permute>::ShapeProfile;
      using StrideOrder  = typename PermuteTraits<Permute>::StrideOrder;
      return make_ordered_layout(reshape(layout.shape(), ShapeProfile{}), StrideOrder{});
    }
  }
}

namespace detail
{

template<int I>
struct is_constant_pred {
  template <class T>
  constexpr auto operator()(T) {
    return is_constant<I, T>{};
  }
};

template<class Permutation, int... I>
constexpr auto
inverse_impl(Permutation const & perm, seq<I...>) {
  return cute::make_tuple(Int<find_if(Permutation{}, is_constant_pred<I>{})>{}...);
}

} // namespace detail

// Compute an inverse of a permutation represented as a tuple of cute::Int<>
template<class Permutation>
constexpr auto
inverse(Permutation const & perm) {
  auto flat_perm = flatten(perm);
  return unflatten(detail::inverse_impl(flat_perm, tuple_seq<decltype(flat_perm)>{}), perm);
}

template<class T>
using inverse_t = decltype(inverse(T{}));

// Given a rank-2 layout of tensor that is assumed to have been permuted,
// compute the original rank-2 layout of the tensor prior to the permutation.
// This is needed to form the correct input to the standalone permutation kernel.
template<class Permute, bool Transpose, class Shape, class Stride>
constexpr auto
make_original_layout(Layout<Shape,Stride> const& layout) {
  static_assert(cute::rank(Shape{}) == 3, "Only rank-3 layouts are supported");
  if constexpr (Transpose) {
    // Deal with tensor B by transposing appropriately before and after computing the permute layout.
    // Its CuTe-canonical mode order is [N,K,L], while permute operations expect [row,col,batch].
    return select<1,0,2>(make_original_layout<Permute, false>(select<1,0,2>(layout)));
  }
  else {
    using ShapeProfile = typename PermuteTraits<Permute>::ShapeProfile;
    auto re_shape   = flatten(reshape(layout.shape(), ShapeProfile{}));
    using IndexOrder   = typename PermuteTraits<Permute>::IndexOrder;
    auto orig_shape = transform_leaf(IndexOrder{}, [&](auto i){ return get<i>(re_shape); });
    using OrigOrder    = conditional_t<cutlass::gemm::detail::is_major<0,Stride>(), seq<0,1,2>, seq<1,0,2>>;
    // print("Permuted shape: "); print(reshape(layout.shape(), ShapeProfile{})); print("\n");
    // print("Original shape: "); print(orig_shape); print("\n");
    return make_ordered_layout(product_each(orig_shape), OrigOrder{});
  }
}

/////////////// Tensor4DPermute0213 ////////////////////

template<int D1, int D2>
struct PermuteTraits<cutlass::layout::Tensor4DPermute0213ColumnMajor<D1, D2>>
{
  static constexpr bool kBatched = false;
  using ShapeProfile = Shape<Shape<X,Int<D1>>, Shape<Int<D2>,X>, Shape<X>>;
  using IndexOrder   = Step<Step<_0,_2>, Step<_1,_3>, Step<_4>>;
  using StrideOrder = inverse_t<IndexOrder>; // Step<Step<_0,_2>, Step<_1,_3>, Step<_4>>;
};

template<int D1, int D2>
struct PermuteTraits<cutlass::layout::Tensor4DPermute0213ColumnMajorInverse<D1, D2>>
{
  static constexpr bool kBatched = false;
  using ShapeProfile = Shape<Shape<X,Int<D2>>, Shape<Int<D1>,X>, Shape<X>>;
  using IndexOrder   = Step<Step<_0,_2>, Step<_1,_3>, Step<_4>>;
  using StrideOrder  = inverse_t<IndexOrder>; // Step<Step<_0,_2>, Step<_1,_3>, Step<_4>>;
};

template<int D1, int D2>
struct PermuteTraits<cutlass::layout::Tensor4DPermute0213RowMajor<D1, D2>>
{
  static constexpr bool kBatched = false;
  using ShapeProfile = Shape<Shape<Int<D1>,X>, Shape<X,Int<D2>>, Shape<X>>;
  using IndexOrder   = Step<Step<_1,_3>, Step<_0,_2>, Step<_4>>;
  using StrideOrder  = Step<Step<_1,_3>, Step<_0,_2>, Step<_4>>;
};

template<int D1, int D2>
struct PermuteTraits<cutlass::layout::Tensor4DPermute0213RowMajorInverse<D1, D2>>
{
  static constexpr bool kBatched = false;
  using ShapeProfile = Shape<Shape<Int<D2>,X>, Shape<X,Int<D1>>, Shape<X>>;
  using IndexOrder   = Step<Step<_1,_3>, Step<_0,_2>, Step<_4>>;
  using StrideOrder  = Step<Step<_1,_3>, Step<_0,_2>, Step<_4>>;
};

/////////////// Tensor4DPermuteBMM0321 ////////////////////

template<int D>
struct PermuteTraits<cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D>>
{
  static constexpr bool kBatched = true;
  using ShapeProfile = Shape<Shape<X>, Shape<X>, Shape<Int<D>,X>>;
  using IndexOrder   = Step<Step<_0,_2>, Step<_1>, Step<_3>>;
  using StrideOrder  = Step<Step<_0>, Step<_2>, Step<_1,_3>>;
};

template<int D>
struct PermuteTraits<cutlass::layout::Tensor4DPermuteBMM0321ColumnMajorInverse<D>>
{
  static constexpr bool kBatched = true;
  using ShapeProfile = Shape<Shape<X,Int<D>>, Shape<X>, Shape<X>>;
  using IndexOrder   = Step<Step<_0>, Step<_2>, Step<_1,_3>>;
  using StrideOrder  = Step<Step<_0,_2>, Step<_1>, Step<_3>>;
};

/////////////// Tensor4DPermuteBMM0213 ////////////////////

template<int D>
struct PermuteTraits<cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D>>
{
  static constexpr bool kBatched = true;
  using ShapeProfile = Shape<Shape<X>, Shape<X>, Shape<Int<D>,X>>;
  using IndexOrder   = Step<Step<_0>, Step<_1,_2>, Step<_3>>;
  using StrideOrder  = Step<Step<_2>, Step<_0>, Step<_1,_3>>;
};

template<int D>
struct PermuteTraits<cutlass::layout::Tensor4DPermuteBMM0213RowMajorInverse<D>>
{
  static constexpr bool kBatched = true;
  using ShapeProfile = Shape<Shape<X>, Shape<X,Int<D>>, Shape<X>>;
  using IndexOrder   = Step<Step<_0>, Step<_1>, Step<_2,_3>>;
  using StrideOrder  = Step<Step<_1>, Step<_0,_2>, Step<_3>>;
};

/////////////// Tensor5DPermute02413 ////////////////////

template<int D1, int D2, int D3>
struct PermuteTraits<cutlass::layout::Tensor5DPermute02413ColumnMajor<D1, D2, D3>>
{
  static constexpr bool kBatched = false;
  using ShapeProfile = Shape<Shape<X,Int<D1>>, Shape<Int<D2>,Int<D3>,X>, Shape<X>>;
  using IndexOrder   = Step<Step<_0,_2>, Step<_4,_1,_3>, Step<_5>>;
  using StrideOrder  = inverse_t<IndexOrder>; // Step<Step<_0,_3>, Step<_1,_4,_2>, Step<_5>>;
};

template<int D1, int D2, int D3>
struct PermuteTraits<cutlass::layout::Tensor5DPermute02413ColumnMajorInverse<D1, D2, D3>>
{
  static constexpr bool kBatched = false;
  using ShapeProfile = Shape<Shape<X,Int<D2>>, Shape<X,Int<D1>,Int<D3>>, Shape<X>>;
  using IndexOrder   = Step<Step<_0,_3>, Step<_1,_4,_2>, Step<_5>>;
  using StrideOrder  = inverse_t<IndexOrder>; // Step<Step<_0,_2>, Step<_4,_1,_3>, Step<_5>>;
};

/////////////// Tensor5DPermute20314 ////////////////////

template<int D1, int D2, int D3>
struct PermuteTraits<cutlass::layout::Tensor5DPermute20314RowMajor<D1, D2, D3>>
{
  static constexpr bool kBatched = false;
  using ShapeProfile = Shape<Shape<Int<D1>,X>, Shape<X,Int<D3>,Int<D2>>, Shape<X>>;
  using IndexOrder   = Step<Step<_2,_0>, Step<_3,_1,_4>, Step<_5>>;
  using StrideOrder  = Step<Step<_1,_3>, Step<_0,_2,_4>, Step<_5>>;
};

template<int D1, int D2, int D3>
struct PermuteTraits<cutlass::layout::Tensor5DPermute20314RowMajorInverse<D1, D2, D3>>
{
  static constexpr bool kBatched = false;
  using ShapeProfile = Shape<Shape<X,Int<D2>>, Shape<X,Int<D1>,Int<D3>>, Shape<X>>;
  using IndexOrder   = Step<Step<_3,_0>, Step<_2,_4,_1>, Step<_5>>;
  using StrideOrder  = Step<Step<_4,_2>, Step<_0,_3,_1>, Step<_5>>;
};

} // namespace example
