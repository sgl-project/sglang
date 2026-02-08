/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Kernel performing a reduction over densely packed tensors in global memory
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reduction {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Mixed-precision reduction
template <
  typename ElementAccumulator_,
  typename Element_,
  int Count = 1
>
struct ReduceAdd {

  //
  // Type definitions
  //

  using ElementAccumulator = ElementAccumulator_;
  using Element = Element_;
  static int const kCount = Count;

  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentElement = cutlass::Array<Element, kCount>;

  struct Params { };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  ReduceAdd(Params params_ = Params()): params(params_) { }

  /// Operator
  CUTLASS_HOST_DEVICE
  FragmentAccumulator operator()(
    FragmentAccumulator accumulator, 
    FragmentElement element) const {

    plus<FragmentAccumulator> op;

    NumericArrayConverter<
      ElementAccumulator, 
      Element, 
      kCount, 
      PreferredRoundingMode<ElementAccumulator, Element>::kRound> converter;

    return op(accumulator, converter(element));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Special handling for binary operators
template <typename ReductionOp, typename Element, int N>
struct VectorizeArrayOperation {

  using ValueType = Array<Element, N>;

  CUTLASS_HOST_DEVICE
  ValueType operator()(
    ReductionOp const &reduction_op, 
    ValueType const &lhs, 
    ValueType const &rhs) const {

    ValueType result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = reduction_op(lhs[i], rhs[i]);
    }

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ReductionOp, typename Element, int N>
struct ReduceArrayOperation {

  using ArrayType = Array<Element, N>;

  CUTLASS_HOST_DEVICE
  Element operator()(
    ReductionOp const &reduction_op, 
    ArrayType const &array) const {

    Element item = reduction_op(array[0], array[1]);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 2; i < N; ++i) {
      item = reduction_op(item, array[i]);
    }

    return item;
  }
};

template <int N>
struct ReduceArrayOperation<logical_and<uint1b_t>, uint1b_t, N> {

  using ArrayType = Array<uint1b_t, N>;

  CUTLASS_HOST_DEVICE
  uint1b_t operator()(
    logical_and<uint1b_t> const &reduction_op, 
    ArrayType const &array) const {

    uint8_t const *ptr = reinterpret_cast<uint8_t const *>(&array);
    bool item = false;

    CUTLASS_PRAGMA_UNROLL
    for (int byte = 0; byte < (N + 7) / 8; ++byte) {
      uint8_t bits = ptr[byte];
      item = (item || !bits);
    }

    return uint1b_t{!item};
  }
};

template <int N>
struct ReduceArrayOperation<logical_or<uint1b_t>, uint1b_t, N> {

  using ArrayType = Array<uint1b_t, N>;

  CUTLASS_HOST_DEVICE
  uint1b_t operator()(
    logical_and<uint1b_t> const &reduction_op, 
    ArrayType const &array) const {

    uint8_t const *ptr = reinterpret_cast<uint8_t const *>(&array);
    bool item = true;

    CUTLASS_PRAGMA_UNROLL
    for (int byte = 0; byte < (N + 7) / 8; ++byte) {
      uint8_t bits = ptr[byte];
      item = (item || bits);
    }

    return uint1b_t{item};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper function to infer template argument types
template <typename ReductionOp, typename Element, int N>
CUTLASS_HOST_DEVICE
Array<Element, N> ApplyArrayOperator(
  ReductionOp const &reduction_op,
  Array<Element, N> const &lhs, 
  Array<Element, N> const &rhs) {

  VectorizeArrayOperation<ReductionOp, Element, N> vectorize_op;

  return vectorize_op(reduction_op, lhs, rhs);
}

/// Helper to reduce an array
template <typename ReductionOp, typename Element, int N>
Element ReduceArray(ReductionOp const &reduction_op, Array<Element, N> const &array) {
  ReduceArrayOperation<ReductionOp, Element, N> reduce_array_op;

  return reduce_array_op(reduction_op, array);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace reduction
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
