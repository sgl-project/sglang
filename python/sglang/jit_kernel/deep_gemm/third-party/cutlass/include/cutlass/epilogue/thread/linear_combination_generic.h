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
  \brief Functor performing linear combination operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Activation, class = void>
struct GenericActivationTraits {
  static constexpr bool IsArgumentsNeeded = false;
  struct Arguments {};
};

template <class Activation>
struct GenericActivationTraits<Activation, decltype(typename Activation::Arguments(), void())> {
  static constexpr bool IsArgumentsNeeded = true;
  using Arguments = typename Activation::Arguments;
};

template <typename T>
struct LinearCombinationGenericParams {
  T alpha;                  ///< scales accumulators
  T beta;                   ///< scales source tensor
  T const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
  T const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  LinearCombinationGenericParams():
    alpha(T(1)),
    beta(T(0)),
    alpha_ptr(nullptr),
    beta_ptr(nullptr) { }

  CUTLASS_HOST_DEVICE
  LinearCombinationGenericParams(
    T alpha,
    T beta = T(0)
  ): alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) { }

  CUTLASS_HOST_DEVICE
  LinearCombinationGenericParams(
    T const *alpha_ptr,
    T const *beta_ptr = nullptr
  ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator followed by an activation function to an array of elements.
///
/// D = activation(alpha * accumulator + beta * source + uniform)
///
template <
  template<typename T> class ActivationFunctor,
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  bool IsHeavy = false
>
class LinearCombinationGeneric {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static bool const kIsHeavy = IsHeavy;
  static int const kCount = Count;
  static const ScaleType::Kind kScale = Scale;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentSource = Array<ElementOutput, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params
    : LinearCombinationGenericParams<ElementCompute>,
      GenericActivationTraits<ActivationFunctor<ElementCompute>>::Arguments {
    using LinearCombinationGenericParams<ElementCompute>::LinearCombinationGenericParams;
  };

private:

  //
  // Data members
  //

  Params params_;
  bool skip_elementwise_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationGeneric(Params const &params) {
    params_ = params;
    params_.alpha = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    params_.beta = (params.beta_ptr ? *params.beta_ptr : params.beta);
    skip_elementwise_ = false;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    if (Scale == ScaleType::Nothing) return false;

    return params_.beta != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      params_.beta = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      skip_elementwise_ = true;
    }
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    FragmentOutput const &source) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations

    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    multiply_add<FragmentCompute> mul_add_accumulator;
    ActivationFunctor<FragmentCompute> activation;

    if (Scale == ScaleType::NoBetaScaling) {
      intermediate = converted_source;
      intermediate = mul_add_accumulator(params_.alpha, converted_accumulator, intermediate);    // D = alpha * Accum + X
    }  else if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate = mul_add_source(params_.beta, converted_source);                             // X =  beta * C + uniform
      intermediate = mul_add_accumulator(params_.alpha, converted_accumulator, intermediate);    // D = alpha * Accum + X
    }

    if constexpr (GenericActivationTraits<ActivationFunctor<ElementCompute>>::IsArgumentsNeeded) {
      intermediate = skip_elementwise_ ? intermediate : activation(intermediate, params_);
    } else {
      intermediate = skip_elementwise_ ? intermediate : activation(intermediate);
    }

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations

    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_accumulator;
    ActivationFunctor<FragmentCompute> activation;

    if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate = mul_add_accumulator(params_.alpha, converted_accumulator);    // D = alpha * Accum
    }

    if constexpr (GenericActivationTraits<ActivationFunctor<FragmentCompute>>::IsArgumentsNeeded) {
      intermediate = skip_elementwise_ ? intermediate : activation(intermediate, params_);
    } else {
      intermediate = skip_elementwise_ ? intermediate : activation(intermediate);
    }

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass
