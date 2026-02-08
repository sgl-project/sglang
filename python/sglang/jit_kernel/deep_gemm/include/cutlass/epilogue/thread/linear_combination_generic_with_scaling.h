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

/*! \file
  \brief Functor performing linear combination operations with a generic element-wise activation
  function. Scaling factors are applied to operands A, B, and C. The pre-activation auxiliary
  output is also returned.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
///
/// Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias
///   D = activation(Aux)
///
template <
  template<typename T> class ActivationFunctor,
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  typename ElementAuxOutput_,                          ///< Data type used to store auxiliary output
  int Count,                                           ///< Number of elements computed per operation
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  bool IsHeavy = false
>
class LinearCombinationGenericWithScalingAndAbsMax {
public:

  using ElementOutput = ElementOutput_;
  using ElementAuxOutput = ElementAuxOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalingFactor = ElementAccumulator_;

  /// Data type used for absolute maximum value
  using ElementAbsmax = float;

  static bool const kIsScalingAndAmaxAuxOutputNeeded = (platform::is_same<ElementAuxOutput, cutlass::float_e4m3_t>::value ||
                                                        platform::is_same<ElementAuxOutput, cutlass::float_e5m2_t>::value);
  static bool const kIsScalingAndAmaxOutputNeeded    = (platform::is_same<ElementOutput, cutlass::float_e4m3_t>::value ||
                                                        platform::is_same<ElementOutput, cutlass::float_e5m2_t>::value);

  static bool const kIsHeavy = IsHeavy;
  static int const kCount = Count;
  static const ScaleType::Kind kScale = Scale;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAuxOutput = Array<ElementAuxOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {
    struct ActivationParams
      : LinearCombinationGenericParams<ElementCompute>,
        GenericActivationTraits<ActivationFunctor<ElementCompute>>::Arguments {
      using LinearCombinationGenericParams<ElementCompute>::LinearCombinationGenericParams;
    };

    ActivationParams activation;
    ElementScalingFactor const* scale_a_ptr = nullptr;   ///< pointer to a scalar - if not null, loads it from memory
    ElementScalingFactor const* scale_b_ptr = nullptr;   ///< pointer to b scalar - if not null, loads it from memory
    ElementScalingFactor const* scale_c_ptr = nullptr;   ///< pointer to c scalar - if not null, loads it from memory
    ElementScalingFactor const* scale_d_ptr = nullptr;   ///< pointer to d scalar - if not null, loads it from memory
    ElementScalingFactor const* scale_aux_ptr = nullptr; ///< pointer to aux scalar - if not null, loads it from memory

    ElementAbsmax * abs_max_aux_ptr = nullptr;      ///< pointer to location to store amax of Aux
    ElementAbsmax * abs_max_D_ptr   = nullptr;      ///< pointer to location to store amax of D

    CUTLASS_HOST_DEVICE
    Params() :
      scale_a_ptr(nullptr),
      scale_b_ptr(nullptr),
      scale_c_ptr(nullptr),
      scale_d_ptr(nullptr),
      scale_aux_ptr(nullptr),
      abs_max_aux_ptr(nullptr),
      abs_max_D_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ActivationParams activation_params,
           ElementScalingFactor const* scale_a_ptr,
           ElementScalingFactor const* scale_b_ptr,
           ElementScalingFactor const* scale_c_ptr,
           ElementScalingFactor const* scale_d_ptr,
           ElementScalingFactor const* scale_aux_ptr,
           ElementAbsmax * abs_max_aux_ptr,
           ElementAbsmax * abs_max_D_ptr) :
           activation(activation_params),
           scale_a_ptr(scale_a_ptr),
           scale_b_ptr(scale_b_ptr),
           scale_c_ptr(scale_c_ptr),
           scale_d_ptr(scale_d_ptr),
           scale_aux_ptr(scale_aux_ptr),
           abs_max_aux_ptr(abs_max_aux_ptr),
           abs_max_D_ptr(abs_max_D_ptr) {}
  };

private:

  //
  // Data members
  //

  Params params_;
  bool skip_elementwise_;

  // Scaling factors for output and auxiliary output
  ElementCompute scale_d_;
  ElementCompute scale_aux_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationGenericWithScalingAndAbsMax(Params const &params) :
    params_(params),
    skip_elementwise_(false),
    scale_d_(ElementCompute(params.scale_d_ptr ? *(params.scale_d_ptr) : ElementScalingFactor(1))),
    scale_aux_(ElementCompute(params.scale_aux_ptr ? *(params.scale_aux_ptr) : ElementScalingFactor(1)))
  {
    params_.activation.alpha = (params.activation.alpha_ptr ? *params.activation.alpha_ptr : params.activation.alpha);
    params_.activation.beta = (params.activation.beta_ptr ? *params.activation.beta_ptr : params.activation.beta);
    auto scale_a =
        ElementCompute(params.scale_a_ptr ? *(params.scale_a_ptr) : ElementScalingFactor(1));
    auto scale_b =
        ElementCompute(params.scale_b_ptr ? *(params.scale_b_ptr) : ElementScalingFactor(1));
    auto scale_c =
        ElementCompute(params.scale_c_ptr ? *(params.scale_c_ptr) : ElementScalingFactor(1));

    multiplies<ElementCompute> multiply;
    params_.activation.alpha = multiply(params.activation.alpha, multiply(scale_a, scale_b));
    params_.activation.beta = multiply(params.activation.beta, scale_c);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    if (Scale == ScaleType::Nothing) return false;

    return params_.activation.beta != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      params_.activation.beta = ElementCompute(1);
    }

    // Only the final partition should perform the activation function
    // and scale the output and auxiliary output values.
    if (k_partition != k_partition_count - 1) {
      skip_elementwise_ = true;
      scale_d_ = ElementCompute(1.);
      scale_aux_ = ElementCompute(1.);
    }
  }

  /// Computes linear scaling:
  ///    Aux = (alpha * scale_a * scale_b * accumulator) + (beta * scale_c * source) + bias
  ///      D = activation(Aux)
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentCompute& output,
    FragmentCompute& aux_output,
    FragmentAccumulator const &accumulator,
    FragmentCompute const& bias,
    FragmentOutput const &source) {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations

    FragmentCompute intermediate;

    multiplies<FragmentCompute> multiply;
    plus<FragmentCompute> add;
    multiply_add<FragmentCompute> mul_add_accumulator;
    ActivationFunctor<FragmentCompute> activation;

    if (Scale == ScaleType::NoBetaScaling) {
      intermediate = converted_source;
      intermediate = mul_add_accumulator(params_.activation.alpha, converted_accumulator, intermediate);
    }  else if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate = multiply(params_.activation.beta, converted_source);
      intermediate = mul_add_accumulator(params_.activation.alpha, converted_accumulator, intermediate);
    }

    intermediate = add(intermediate, bias);

    aux_output = intermediate;
    if constexpr (GenericActivationTraits<ActivationFunctor<ElementCompute>>::IsArgumentsNeeded) {
      output = skip_elementwise_ ? intermediate : activation(intermediate, params_.activation);
    } else {
      output = skip_elementwise_ ? intermediate : activation(intermediate);
    }
  }

  /// Computes linear scaling:
  ///    Aux = (alpha * scale_a * scale_b * accumulator) + bias
  ///      D = activation(Aux)
  CUTLASS_DEVICE
  void operator()(
    FragmentCompute& output,
    FragmentCompute& aux_output,
    FragmentAccumulator const &accumulator,
    FragmentCompute const& bias) {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations

    FragmentCompute intermediate;

    multiplies<FragmentCompute> multiply;
    plus<FragmentCompute> add;
    ActivationFunctor<FragmentCompute> activation;

    if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate = multiply(params_.activation.alpha, converted_accumulator);
    }

    intermediate = add(intermediate, bias);

    aux_output = intermediate;
    if constexpr (GenericActivationTraits<ActivationFunctor<FragmentCompute>>::IsArgumentsNeeded) {
      output = skip_elementwise_ ? intermediate : activation(intermediate, params_.activation);
    } else {
      output = skip_elementwise_ ? intermediate : activation(intermediate);
    }
  }

  CUTLASS_HOST_DEVICE
  ElementAbsmax* get_ptr_output_abs_max() const {
    return params_.abs_max_D_ptr;
  }

  CUTLASS_HOST_DEVICE
  ElementAbsmax* get_ptr_aux_output_abs_max() const {
    return params_.abs_max_aux_ptr;
  }

  CUTLASS_HOST_DEVICE
  ElementCompute get_scale_d() const {
    return scale_d_;
  }

  CUTLASS_HOST_DEVICE
  ElementCompute get_scale_aux() const {
    return scale_aux_;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass
