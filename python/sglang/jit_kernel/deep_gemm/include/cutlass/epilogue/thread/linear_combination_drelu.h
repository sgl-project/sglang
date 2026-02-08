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
  \brief Functor performing linear combination with a maximum operation used by epilogues.
*/

#pragma once

#include "cutlass/half.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/activation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source + uniform
///
template <
  typename ElementCompute_,                            ///< Data type returned by this functor
  typename ElementAccumulator_,                        ///< Data type of accumulators
  typename ElementSource_,                             ///< Data type of source tensor
  typename ElementTensor_,                             ///< Data type of additional tensor
  int Count,                                           ///< Number of elements computed per operation
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class LinearCombinationDRelu {
public:

  using ElementOutput = ElementSource_;
  using ElementCompute = ElementCompute_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementSource = ElementSource_;
  using ElementTensor = ElementTensor_;

  static int const kCount = Count;

  using FragmentCompute = Array<ElementCompute, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentTensor = Array<ElementTensor, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta;                   ///< scales source tensor
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory
    ElementCompute threshold;              ///< minimum value that is output 
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): 
      alpha(ElementCompute(1)), 
      beta(ElementCompute(0)),
      threshold(ElementCompute(0)), 
      alpha_ptr(nullptr), 
      beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta,
      ElementCompute threshold = ElementCompute(0)
    ): alpha(alpha), beta(beta), threshold(threshold), alpha_ptr(nullptr), beta_ptr(nullptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr,
      ElementCompute threshold = ElementCompute(0)
    ): alpha(0), beta(0), threshold(threshold), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {

    }
  };

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;
  ElementTensor threshold_;
  bool participates_in_reduction_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationDRelu(Params const &params) {

    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    threshold_ = ElementTensor(params.threshold);
    participates_in_reduction_  = true;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ != ElementCompute(0);
  }

  /// Returns true if the threadblock computes the reduction
  CUTLASS_HOST_DEVICE
  bool participates_in_reduction() const {
    return participates_in_reduction_;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      // set to NaN to make ReLU no-op for all except last k partitions
      int64_t allones = -1;
      threshold_ = reinterpret_cast<ElementTensor const &>(allones);
      participates_in_reduction_ = false;
    }
  }
  
  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentCompute operator()(
    FragmentAccumulator const &accumulator, 
    FragmentSource const &source,
    FragmentTensor const &tensor) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementSource, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    multiply_add<FragmentCompute> mul_add_accumulator;

    intermediate = mul_add_source(beta_, converted_source);                             // X =  beta * C
    intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X

    // dReLU = (cond ? dy : 0)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      ElementTensor cond = tensor[i];
      if (cond <= threshold_) {
        intermediate[i] = ElementCompute();
      }
    }

    return intermediate;
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentCompute operator()(
    FragmentAccumulator const &accumulator,
    FragmentTensor const &tensor) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_accumulator;

    intermediate = mul_accumulator(alpha_, converted_accumulator);    // D = alpha * Accum

    // dReLU = (cond ? dy : 0)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      ElementTensor cond = tensor[i];
      if (cond <= threshold_) {
        intermediate[i] = ElementCompute();
      }
    }

    return intermediate;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source + uniform
///
template <
  typename ElementCompute_,                            ///< Data type returned by this functor
  typename ElementAccumulator_,                        ///< Data type of accumulators
  typename ElementSource_,                             ///< Data type of source tensor
  int Count,                                           ///< Number of elements computed per operation
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class LinearCombinationDReluConditionalBits {
public:

  using ElementOutput = ElementSource_;
  using ElementCompute = ElementCompute_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementSource = ElementSource_;
  using ElementTensor = uint1b_t;

  static bool const kIsHeavy = false;

  static int const kCount = Count;

  using FragmentCompute = Array<ElementCompute, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentTensor = Array<ElementTensor, kCount>;

  static FloatRoundStyle const kRound = Round;

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
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {

    }
  };

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;
  FragmentTensor predicate_mask_;
  bool participates_in_reduction_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationDReluConditionalBits(Params const &params) {

    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    participates_in_reduction_ = true;
    predicate_mask_.clear();
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ != ElementCompute(0);
  }

  /// Returns true if the threadblock computes the reduction
  CUTLASS_HOST_DEVICE
  bool participates_in_reduction() const {
    return participates_in_reduction_;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    predicate_mask_.clear();

    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      // Avoid computing the reduction if this isn't the final Split-K slice
      participates_in_reduction_ = false;
      
      bit_not<FragmentTensor> not_op;
      predicate_mask_ = not_op(predicate_mask_);
    }
  }
  
  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_DEVICE
  FragmentCompute operator()(
    FragmentAccumulator const &accumulator, 
    FragmentSource const &source,
    FragmentTensor const &tensor) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementSource, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    multiply_add<FragmentCompute> mul_add_accumulator;

    intermediate = mul_add_source(beta_, converted_source);                             // X =  beta * C + uniform
    intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X

    bit_or<FragmentTensor> or_op;

    FragmentTensor predicates = or_op(tensor, predicate_mask_);

    // Obtain from packed bits
    bool conditions[kCount];
    UnpackPredicates<kCount> unpack_predicates;

    unpack_predicates(conditions, predicates);

    // dReLU = (cond ? dy : 0)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      if (!conditions[i]) {
        intermediate[i] = ElementCompute();
      }
    }

    return intermediate;
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentCompute operator()(
    FragmentAccumulator const &accumulator,
    FragmentTensor const &tensor) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_accumulator;

    intermediate = mul_accumulator(alpha_, converted_accumulator);    // D = alpha * Accum

    bit_or<FragmentTensor> or_op;

    FragmentTensor predicates = or_op(tensor, predicate_mask_);

    // Obtain from packed bits
    bool conditions[kCount];
    UnpackPredicates<kCount> unpack_predicates;

    unpack_predicates(conditions, predicates);

    // dReLU = (cond ? dy : 0)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      if (!conditions[i]) {
        intermediate[i] = ElementCompute();
      }
    }

    return intermediate;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
