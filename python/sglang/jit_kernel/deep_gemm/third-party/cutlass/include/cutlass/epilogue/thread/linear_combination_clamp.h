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
  \brief Functor performing linear scaling operations used by epilogues. Values are clamped before
         converting to the output element type.
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

namespace detail {

/// Single source of truth for whether to unroll for `LinearCombinationClamp()`
constexpr bool LinearCombinationClampIsHeavy() {
  return false;
}

}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements then clamps the output before
/// converting to the output element type.
///
/// D = alpha * accumulator + beta * source + uniform
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class LinearCombinationClamp {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;
  using FragmentSource = Array<ElementOutput, kCount>;

  static FloatRoundStyle const kRound = Round;

  static bool const kIsHeavy = detail::LinearCombinationClampIsHeavy();

  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta;                   ///< scales source tensor
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): 
      alpha(ElementCompute(1)), 
      beta(ElementCompute(0)), 
      alpha_ptr(nullptr), 
      beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta
    ): alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha
    ): alpha(alpha), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(nullptr) {

    }
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
  LinearCombinationClamp(Params const &params) {

    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    if (Scale == ScaleType::Nothing) return false;

    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator, 
    FragmentOutput const &source,
    ElementCompute uniform = ElementCompute(0)) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_source = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations

    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_add_source;
    multiply_add<ComputeFragment> mul_add_accumulator;
    
    minimum<ComputeFragment> min_accumulator;
    maximum<ComputeFragment> max_accumulator;

    if (Scale == ScaleType::NoBetaScaling) {
      intermediate = converted_source;
      intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X
    } else if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate = mul_add_source(beta_, converted_source);                             // X =  beta * C + uniform
      intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X
    }

    /// Clamping constant value
    ElementCompute const kClampMax =
        ElementCompute(cutlass::platform::numeric_limits<ElementOutput>::max());

    ElementCompute const kClampMin =
        ElementCompute(cutlass::platform::numeric_limits<ElementOutput>::lowest());

    intermediate = max_accumulator(intermediate, kClampMin);
    intermediate = min_accumulator(intermediate, kClampMax);

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

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations

    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_accumulator;
    
    minimum<ComputeFragment> min_accumulator;
    maximum<ComputeFragment> max_accumulator;

    if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate = mul_accumulator(alpha_, converted_accumulator);    // D = alpha * Accum
    }

    /// Clamping constant value
    ElementCompute const kClampMax =
        ElementCompute(cutlass::platform::numeric_limits<ElementOutput>::max());

    ElementCompute const kClampMin =
        ElementCompute(cutlass::platform::numeric_limits<ElementOutput>::lowest());

    intermediate = max_accumulator(intermediate, kClampMin);
    intermediate = min_accumulator(intermediate, kClampMax);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Conditional guards to enable partial specialization for packed integers
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 720) && ((__CUDACC_VER_MAJOR__ > 10) || ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2)))

/// Applies a linear combination operator to an array of elements then clamps the output before
/// converting to the output element type.
///
/// D = alpha * accumulator + beta * source + uniform
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
  ScaleType::Kind Scale,                               ///< Control Alpha and Beta scaling
  FloatRoundStyle Round
>
class LinearCombinationClamp<ElementOutput_, Count, int, float, Scale, Round> {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = int;
  using ElementCompute = float;

  static_assert(
      cutlass::platform::numeric_limits<ElementOutput>::is_integer,
      "This elementwise op expects the output to be int.");

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  static bool const kIsHeavy = detail::LinearCombinationClampIsHeavy();

  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta;                   ///< scales source tensor
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): 
      alpha(ElementCompute(1)), 
      beta(ElementCompute(0)), 
      alpha_ptr(nullptr), 
      beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta
    ): alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha
    ): alpha(alpha), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(nullptr) {

    }
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
  LinearCombinationClamp(Params const &params) {

    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    if (Scale == ScaleType::Nothing) return false;

    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }
  
  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator, 
    FragmentOutput const &source,
    ElementCompute uniform = ElementCompute(0)) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_source = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Compute linear scaling in floating point
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_add_source;
    multiply_add<ComputeFragment> mul_add_accumulator;
    
    // Float min-max
    if (Scale == ScaleType::NoBetaScaling) {
      intermediate = converted_source;
      intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X
    } else if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate = mul_add_source(beta_, converted_source);                             // X =  beta * C + uniform
      intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X
    }

    //
    // Convert float => ElementOutput_ with clamping
    //
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Compute linear scaling in floating point
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_add_accumulator;
    
    // Float min-max
    if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate = mul_add_accumulator(alpha_, converted_accumulator);    // D = alpha * Accum
    }

    //
    // Convert float => ElementOutput_ with clamping
    //
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }
};

#endif // Conditional guards to enable partial specialization for packed integers

////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements then clamps
/// the output before converting to the output element type.
///
/// D = alpha * accumulator + beta * source + uniform
///
/// Note: The below method only when problem_size_K <= 256 for signed int8 gemm
/// or problem_size_K <= 128 for unsigned int8 gemm. The default approach is
/// above.
template <
    /// Data type used to load and store< tensors
    typename ElementOutput_,
    /// Number of elements computed per operation
    int Count,
    ///< Control Alpha and Beta scaling
    ScaleType::Kind Scale = ScaleType::Default,
    /// Rounding mode
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class FastLinearCombinationClamp {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = int;
  using ElementCompute = float;

  static_assert(
      cutlass::platform::numeric_limits<ElementOutput>::is_integer,
      "This elementwise op expects the output to be int.");

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  static bool const kIsHeavy = false;

  /// Host-constructable parameters structure
  struct Params {
    /// scales accumulators
    ElementCompute alpha;
    /// scales source tensor
    ElementCompute beta;
    /// pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *alpha_ptr;
    /// pointer to source scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : alpha(ElementCompute(1)),
          beta(ElementCompute(0)),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta)
        : alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha)
        : alpha(alpha), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr, ElementCompute const *beta_ptr)
        : alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr)
        : alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(nullptr) {}
  };

 private:
  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;

 public:
  /// Constructs the function object, possibly loading from pointers in host
  /// memory
  CUTLASS_HOST_DEVICE
  FastLinearCombinationClamp(Params const &params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    if (Scale == ScaleType::Nothing) return false;

    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }
  
  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentOutput const &source,
                            ElementCompute uniform = ElementCompute(0)) const {
    // Convert source to interal compute numeric type
    FastNumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>
        source_converter;
    FastNumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    ComputeFragment converted_source = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Compute linear scaling in floating point
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_add_source;
    multiply_add<ComputeFragment> mul_add_accumulator;

    minimum<ComputeFragment> min_accumulator;
    maximum<ComputeFragment> max_accumulator;

    // Float min-max
    if (Scale == ScaleType::NoBetaScaling) {
      intermediate = converted_source;
      intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X
    } else if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate =
          mul_add_source(beta_, converted_source);  // X =  beta * C + uniform
      intermediate = mul_add_accumulator(alpha_, converted_accumulator,
                                         intermediate);  // D = alpha * Accum + X
    }

    /// Clamping constant value
    ElementCompute const kClamp =
        ElementCompute(1 << (sizeof_bits<ElementOutput>::value - 1));

    intermediate = max_accumulator(intermediate, -kClamp);
    intermediate = min_accumulator(intermediate, kClamp - ElementCompute(1));

    // Convert to destination numeric type
    FastNumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {

    // Convert source to interal compute numeric type
    FastNumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Compute linear scaling in floating point
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_accumulator;

    minimum<ComputeFragment> min_accumulator;
    maximum<ComputeFragment> max_accumulator;

    // Float min-max
    if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate = mul_accumulator(alpha_, converted_accumulator);
    }

    /// Clamping constant value
    ElementCompute const kClamp =
        ElementCompute(1 << (sizeof_bits<ElementOutput>::value - 1));

    intermediate = max_accumulator(intermediate, -kClamp);
    intermediate = min_accumulator(intermediate, kClamp - ElementCompute(1));

    // Convert to destination numeric type
    FastNumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    return destination_converter(intermediate);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass
