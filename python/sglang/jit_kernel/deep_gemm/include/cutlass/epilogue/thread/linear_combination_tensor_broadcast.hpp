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
  \brief Functor performing linear combination operation, bias addition, and tensor-tensor
  elementwise operations
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/detail.hpp"
#include "cutlass/epilogue/thread/scale_type.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

namespace detail {

/// Returns whether a source operand is needed for a combination of binary operation and scale
/// type. Simple specialized checks are made for cases in which 0 is an identity element of
/// the binary operation.
template <class BinaryOp, class ElementCompute, ScaleType::Kind Scale>
CUTLASS_HOST_DEVICE
bool is_binary_op_source_needed(ElementCompute scale) {
  if constexpr (cute::is_same_v<BinaryOp, NoOp<ElementCompute>>) {
    return false;
  }
  else if constexpr (cute::is_same_v<BinaryOp, plus<ElementCompute>> || cute::is_same_v<BinaryOp, minus<ElementCompute>>) {
    // Cases for binary operators for which 0 is an identity element
    if constexpr (Scale == ScaleType::NoBetaScaling) return true;

    if constexpr (Scale == ScaleType::OnlyAlphaScaling) return false;

    if constexpr (Scale == ScaleType::Nothing) return false;

    return scale != ElementCompute(0);
  }

  return true;
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/** Compute a tensor-tensor broadcast epilogue.
 *
 * @param ElementOutput_ Data type used to load and store tensors
 * @param ElementAccumulator_ Accumulator data type
 * @param ElementCompute_ Data type used to compute linear combination
 * @param ElementBias_ Data type of Bias elements
 * @param ActivationFunctor_ Fused Activation
 * @param BinaryOp0_ Binary operation to perform on O0 and C0. detail::NoOp means no operation
 * @param BinaryOp1_ Binary operation to perform on O1 and C1. detail::NoOp means no operation
 * @param UnaryOp_ Unary operation to perform on final result
 * @param Scale Controls the type of Alpha and Beta scaling to perform
 * @param Round How values should be rounded in conversions
 * @param ElementSource_ Data type used for source operands
 *
 *  Computes the following:
 *      O0 = alpha * accumulator + bias
 *      O1 = BinaryOp0(O0, beta * C0)
 *      O2 = BinaryOp1(O1, beta * C1)
 *      D  = UnaryOp(O2)
 */
template <
  class ElementOutput_,
  class ElementAccumulator_ = ElementOutput_,
  class ElementCompute_ = ElementOutput_,
  class ElementBias_ = ElementCompute_,
  template <class T> class ActivationFunctor_ = Identity,
  template <class T> class BinaryOp0_ = plus,
  template <class T> class BinaryOp1_ = detail::NoOp,
  template <class T> class UnaryOp_ = Identity,
  ScaleType::Kind Scale = ScaleType::Default,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  class ElementSource_ = ElementOutput_
>
class LinearCombinationTensorBroadcast {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementBias = ElementBias_;
  using ElementC = ElementSource_;
  using ElementD = ElementOutput_;
  using ElementScalingFactor = ElementAccumulator_;

  using UnaryOp = UnaryOp_<ElementCompute>;
  using BinaryOp0 = BinaryOp0_<ElementCompute>;
  using BinaryOp1 = BinaryOp1_<ElementCompute>;
  using ActivationFunctor = ActivationFunctor_<ElementCompute>;

  static constexpr int kCount = 1;
  static constexpr ScaleType::Kind kScale = Scale;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;
  using FragmentBias = Array<ElementBias, kCount>;

  static constexpr FloatRoundStyle kRound = Round;
  using NoOpType = detail::NoOp<ElementCompute>;
  static constexpr bool IsBinaryOp0Enabled = !cute::is_same_v<BinaryOp0, NoOpType>;
  static constexpr bool IsBinaryOp1Enabled = !cute::is_same_v<BinaryOp1, NoOpType>;
  static constexpr bool IsUnaryOpEnabled = !cute::is_same_v<UnaryOp, NoOpType> && !cute::is_same_v<UnaryOp, Identity<ElementCompute>>;

  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha{};                          ///< scales accumulators
    ElementCompute beta{};                           ///< scales source tensor
    ElementCompute const* alpha_ptr = nullptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const* beta_ptr = nullptr;        ///< pointer to source scalar - if not null, loads it from memory

    //
    // Methods
    //
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const* alpha_ptr, ElementCompute const* beta_ptr)
        : alpha_ptr(alpha_ptr),
          beta_ptr(beta_ptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const* alpha_ptr)
        : alpha_ptr(alpha_ptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha,
           ElementCompute beta)
        : alpha(alpha),
          beta(beta) {}
  };

private:
  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationTensorBroadcast(Params const& params)
      : alpha_(params.alpha_ptr ? *params.alpha_ptr : params.alpha),
        beta_(params.beta_ptr ? *params.beta_ptr : params.beta) {}

  /// Returns true if source 0 is needed
  CUTLASS_HOST_DEVICE
  bool is_source0_needed() const {
    return detail::is_binary_op_source_needed<BinaryOp0, ElementCompute, Scale>(beta_);
  }

  /// Returns true if source 1 is needed
  CUTLASS_HOST_DEVICE
  bool is_source1_needed() const {
    return detail::is_binary_op_source_needed<BinaryOp1, ElementCompute, Scale>(beta_);
  }

  //
  // Specialization for scalar
  //
  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator, ElementC const source0, ElementC source1, ElementBias const bias) {
    // Convert everything to Compute type, do compute, and then store to output type
    NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    NumericConverter<ElementCompute, ElementBias, Round> bias_converter;
    NumericConverter<ElementCompute, ElementC, Round> source_converter;
    NumericConverter<ElementD, ElementCompute, Round> destination_converter;

    ActivationFunctor act;
    multiplies<ElementCompute> mul;
    multiply_add<ElementCompute> madd;

    ElementCompute intermediate = accumulator_converter(accumulator);
    intermediate = madd(alpha_, intermediate, bias_converter(bias));
    intermediate = act(intermediate);

    // Apply BinaryOp0, if needed
    if constexpr (IsBinaryOp0Enabled) {
      BinaryOp0 bin0;
      ElementCompute converted_source = source_converter(source0);
      intermediate = bin0(intermediate, mul(beta_, converted_source));
    }

    // Apply BinaryOp1, if needed
    if constexpr (IsBinaryOp1Enabled) {
      BinaryOp1 bin1;
      ElementCompute converted_source = source_converter(source1);
      intermediate = bin1(intermediate, mul(beta_, converted_source));
    }

    // Apply UnaryOp, if needed
    if constexpr (IsUnaryOpEnabled) {
      UnaryOp unary;
      intermediate = unary(intermediate);
    }

    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
