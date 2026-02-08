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
#include "cutlass/platform/platform.h"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/scale_type.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

struct EmptyArguments {};

template<class T, class = void>
struct ElementwiseOpDispatcher {
  using Arguments = EmptyArguments;

  T op;

  CUTLASS_HOST_DEVICE
  ElementwiseOpDispatcher(Arguments) {}

  template <typename ValueType>
  CUTLASS_HOST_DEVICE
  ValueType operator()(ValueType value) {
    return op(value);
  }
};

template<class T>
struct ElementwiseOpDispatcher<T, std::void_t<typename T::Arguments>> {
  using Arguments = typename T::Arguments;

  Arguments args;
  T op;

  CUTLASS_HOST_DEVICE
  ElementwiseOpDispatcher(Arguments args_):args(args_) {}

  template <typename ValueType>
  CUTLASS_HOST_DEVICE
  ValueType operator()(ValueType value) {
    return op(value, args);
  }
};

}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This base class is meant to define the concept required of the
/// EpilogueWithBroadcast::OutputOp
template <
  typename ElementC_,
  typename ElementAccumulator_,
  typename ElementCompute_,
  typename ElementZ_,
  typename ElementT_,
  int ElementsPerAccess,
  typename ElementwiseOp_ = Identity<ElementCompute_>,
  typename BinaryOp_ = plus<ElementCompute_>,
  bool StoreT_ = true,
  typename ElementVector_ = ElementC_
>
class LinearCombinationBiasElementwise {
public:

  using ElementOutput = ElementC_;
  using ElementD = ElementOutput;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementZ = ElementZ_;
  using ElementT = ElementT_;
  using ElementVector = ElementVector_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;

  /// Follow cutlass3x EVT aliases
  static bool const IsEltActSupported = true;

  using ElementwiseOp = ElementwiseOp_;
  using BinaryOp = BinaryOp_;

  using ElementwiseOpDispatcher = detail::ElementwiseOpDispatcher<ElementwiseOp>;
  using ElementwiseArguments = typename ElementwiseOpDispatcher::Arguments;

  // Indicates that this epilogue applies only one binary operation
  static bool const kIsSingleSource = true;


  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = Array<ElementC, kElementsPerAccess>;
  using FragmentZ = Array<ElementZ, kElementsPerAccess>;
  using FragmentT = Array<ElementT, kElementsPerAccess>;

  // Definitions needed for collective epilogue
  using FragmentSource = FragmentC;
  using FragmentOutput = FragmentZ;
  using ElementBias = ElementVector;
  using FragmentBias = Array<ElementBias, kElementsPerAccess>;
  using ActivationFn = ElementwiseOp;
  static const ScaleType::Kind kScale = ScaleType::Default;

  static bool const kIsHeavy = kIsHeavy_member_or_false<ElementwiseOp>::value;

  /// If true, the 'Z' tensor is stored
  static bool const kStoreZ = true;

  /// If true, the 'T' tensor is stored
  static bool const kStoreT = StoreT_;

  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta;                   ///< scales source tensor
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory
    ElementwiseArguments  elementwise;     ///< Arguments for elementwise operation

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
      ElementCompute beta,
      ElementwiseArguments  elementwise_ = ElementwiseArguments{}
    ): alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr), elementwise(elementwise_) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha
    ): alpha(alpha), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr,
      ElementwiseArguments  elementwise_ = ElementwiseArguments{}
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr), elementwise(elementwise_) {

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
  ElementwiseArguments const &elementwise_;
  bool skip_elementwise_;

public:

  //
  // Methods
  //

  /// Constructor from Params
  CUTLASS_HOST_DEVICE
  LinearCombinationBiasElementwise(Params const &params): elementwise_(params.elementwise) {

    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    skip_elementwise_ = false;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      skip_elementwise_ = true;
    }
  }

  /// Applies the operation when elementwise_op require arguments and is_source_needed() is true
  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentC const &frag_C,
    FragmentCompute const &V,
    ElementwiseArgs const &elementwise_args) const {

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C = NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(alpha_ * tmp_Accum[i] + beta_ * tmp_C[i], V[i]);
      result_T[i] = z;
      result_Z[i] = skip_elementwise_ ? z : elementwise_op(z, elementwise_args);
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  /// Applies the operation when elementwise_op require arguments and is_source_needed() is false
  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentCompute const &V,
    ElementwiseArgs const &elementwise_args) const {

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(alpha_ * tmp_Accum[i], V[i]);
      result_T[i] = z;
      result_Z[i] = skip_elementwise_ ? z : elementwise_op(z, elementwise_args);
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  /// Applies the operation when is_source_needed() is true
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentC const &frag_C,
    FragmentCompute const &V) const {

    ElementwiseOpDispatcher elementwise_op(elementwise_);
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C = NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(alpha_ * tmp_Accum[i] + beta_ * tmp_C[i], V[i]);
      result_T[i] = z;
      result_Z[i] = skip_elementwise_ ? z : elementwise_op(z);
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  /// Applies the operation when is_source_needed() is false
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentCompute const &V) const {

    ElementwiseOpDispatcher elementwise_op(elementwise_);
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(alpha_ * tmp_Accum[i], V[i]);
      result_T[i] = z;
      result_Z[i] = skip_elementwise_ ? z : elementwise_op(z);
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  /// Applies the operation when elementwise_op require arguments and is_source_needed() is true
  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
    ElementZ &Z,
    ElementT &T,
    ElementAccumulator const &AB,
    ElementC const &C,
    ElementCompute const &V,
    ElementwiseArgs const &elementwise_args) const {

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    ElementCompute tmp_Accum = NumericConverter<ElementCompute, ElementAccumulator>()(AB);
    ElementCompute tmp_C = NumericConverter<ElementCompute, ElementC>()(C);

    ElementCompute z = binary_op(alpha_ * tmp_Accum + beta_ * tmp_C, V);
    ElementCompute result_Z = skip_elementwise_ ? z : elementwise_op(z, elementwise_args);

    NumericConverter<ElementZ, ElementCompute> convert_z;
    Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      ElementCompute result_T = z;
      NumericConverter<ElementT, ElementCompute> convert_t;
      T = convert_t(result_T);
    }
  }

  /// Applies the operation when elementwise_op require arguments and is_source_needed() is false
  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
    ElementZ &Z,
    ElementT &T,
    ElementAccumulator const &AB,
    ElementCompute const &V,
    ElementwiseArgs const &elementwise_args) const {

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    ElementCompute tmp_Accum = NumericConverter<ElementCompute, ElementAccumulator>()(AB);

    ElementCompute z = binary_op(alpha_ * tmp_Accum, V);
    ElementCompute result_Z = skip_elementwise_ ? z : elementwise_op(z, elementwise_args);

    NumericConverter<ElementZ, ElementCompute> convert_z;
    Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      ElementCompute result_T = z;
      NumericConverter<ElementT, ElementCompute> convert_t;
      T = convert_t(result_T);
    }
  }

  /// Applies the operation when is_source_needed() is true
  CUTLASS_HOST_DEVICE
  void operator()(
    ElementZ &Z,
    ElementT &T,
    ElementAccumulator const &AB,
    ElementC const &C,
    ElementCompute const &V) const {

    ElementwiseOpDispatcher elementwise_op(elementwise_);
    BinaryOp binary_op;

    ElementCompute tmp_Accum = NumericConverter<ElementCompute, ElementAccumulator>()(AB);
    ElementCompute tmp_C = NumericConverter<ElementCompute, ElementC>()(C);

    ElementCompute z = binary_op(alpha_ * tmp_Accum + beta_ * tmp_C, V);
    ElementCompute result_Z = skip_elementwise_ ? z : elementwise_op(z);

    NumericConverter<ElementZ, ElementCompute> convert_z;
    Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      ElementCompute result_T = z;
      NumericConverter<ElementT, ElementCompute> convert_t;
      T = convert_t(result_T);
    }
  }

  /// Applies the operation when is_source_needed() is false
  CUTLASS_HOST_DEVICE
  void operator()(
    ElementZ &Z,
    ElementT &T,
    ElementAccumulator const &AB,
    ElementCompute const &V) const {

    ElementwiseOpDispatcher elementwise_op(elementwise_);
    BinaryOp binary_op;

    ElementCompute tmp_Accum = NumericConverter<ElementCompute, ElementAccumulator>()(AB);

    ElementCompute z = binary_op(alpha_ * tmp_Accum, V);
    ElementCompute result_Z = skip_elementwise_ ? z : elementwise_op(z);

    NumericConverter<ElementZ, ElementCompute> convert_z;
    Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      ElementCompute result_T = z;
      NumericConverter<ElementT, ElementCompute> convert_t;
      T = convert_t(result_T);
    }
  }
};


/// This base class is meant to define the concept required of the
/// EpilogueWithBroadcast::OutputOp
template <
  typename ElementC_,
  typename ElementAccumulator_,
  typename ElementCompute_,
  typename ElementZ_,
  typename ElementT_,
  int ElementsPerAccess,
  typename ElementwiseOp_ = Identity<ElementCompute_>,
  typename BinaryOp_ = plus<ElementCompute_>,
  bool StoreT_ = true,
  typename ElementVector_ = ElementC_
>
class LinearCombinationPerChannelScalingBiasElementwise {
public:

  using ElementOutput = ElementC_;
  using ElementD = ElementOutput;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementZ = ElementZ_;
  using ElementT = ElementT_;
  using ElementVector = ElementVector_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;

  /// Follow cutlass3x EVT aliases
  static bool const IsEltActSupported = true;
  static bool const IsPerChannelScalingSupported = true;

  using ElementwiseOp = ElementwiseOp_;
  using BinaryOp = BinaryOp_;

  using ElementwiseOpDispatcher = detail::ElementwiseOpDispatcher<ElementwiseOp>;
  using ElementwiseArguments = typename ElementwiseOpDispatcher::Arguments;

  // Indicates that this epilogue applies only one binary operation
  static bool const kIsSingleSource = true;


  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = Array<ElementC, kElementsPerAccess>;
  using FragmentZ = Array<ElementZ, kElementsPerAccess>;
  using FragmentT = Array<ElementT, kElementsPerAccess>;

  // Definitions needed for collective epilogue
  using FragmentSource = FragmentC;
  using FragmentOutput = FragmentZ;
  using ElementBias = ElementVector;
  using FragmentBias = Array<ElementBias, kElementsPerAccess>;
  using ActivationFn = ElementwiseOp;
  static const ScaleType::Kind kScale = ScaleType::PerChannelScaling;

  static bool const kIsHeavy = kIsHeavy_member_or_false<ElementwiseOp>::value;

  /// If true, the 'Z' tensor is stored
  static bool const kStoreZ = true;

  /// If true, the 'T' tensor is stored
  static bool const kStoreT = StoreT_;

  /// Host-constructable parameters structure
  struct Params {
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory
    ElementCompute beta;                   ///< scales source tensor
    ElementwiseArguments  elementwise;     ///< Arguments for elementwise operation

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): 
      alpha_ptr(nullptr), 
      beta_ptr(nullptr),
      beta(ElementCompute(0)) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr,
      ElementwiseArguments  elementwise_ = ElementwiseArguments{}
    ): beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr), elementwise(elementwise_) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr
    ): beta(0), alpha_ptr(alpha_ptr), beta_ptr(nullptr) {

    }
  };

private:

  //
  // Data members
  //

  ElementCompute const* beta_ptr_ = nullptr;
  ElementCompute beta_ = 0;
  ElementwiseArguments const &elementwise_;
  bool skip_elementwise_;

public:

  //
  // Methods
  //

  /// Constructor from Params
  CUTLASS_HOST_DEVICE
  LinearCombinationPerChannelScalingBiasElementwise(Params const &params): elementwise_(params.elementwise) {
    if (params.beta_ptr) {
      beta_ptr_ = params.beta_ptr;
    }
    else {
      beta_ = params.beta;
    }
    skip_elementwise_ = false;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ptr_ != nullptr || beta_ != ElementCompute(0);
  }

  CUTLASS_HOST_DEVICE
  bool is_beta_vector() const {
    return beta_ptr_ != nullptr;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      skip_elementwise_ = true;
    }
  }

  /// Applies the operation when elementwise_op require arguments and is_source_needed() is true
  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentC const &frag_C,
    FragmentCompute const & valpha,
    FragmentCompute const & vbias,
    ElementwiseArgs const &elementwise_args) const {

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C = NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(valpha[i] * tmp_Accum[i] + beta_ * tmp_C[i], vbias[i]);
      result_T[i] = z;
      result_Z[i] = skip_elementwise_ ? z : elementwise_op(z, elementwise_args);
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  /// Applies the operation when elementwise_op require arguments and is_source_needed() is true
  /// D = elementwise_op(vector_alpha * accumulator + vector_beta * source + bias)
  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentC const &frag_C,
    FragmentCompute const & valpha,
    FragmentCompute const & vbeta,
    FragmentCompute const & vbias,
    ElementwiseArgs const &elementwise_args) const {

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C = NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(valpha[i] * tmp_Accum[i] + vbeta[i] * tmp_C[i], vbias[i]);
      result_T[i] = z;
      result_Z[i] = skip_elementwise_ ? z : elementwise_op(z, elementwise_args);
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  /// Applies the operation when elementwise_op require arguments and is_source_needed() is false
  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentCompute const & valpha,
    FragmentCompute const & vbias,
    ElementwiseArgs const &elementwise_args) const {

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(valpha[i] * tmp_Accum[i], vbias[i]);
      result_T[i] = z;
      result_Z[i] = skip_elementwise_ ? z : elementwise_op(z, elementwise_args);
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  /// Applies the operation when is_source_needed() is true
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentC const &frag_C,
    FragmentCompute const & valpha,
    FragmentCompute const & vbias) const {

    ElementwiseOpDispatcher elementwise_op(elementwise_);
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C = NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(valpha[i] * tmp_Accum[i] + beta_ * tmp_C[i], vbias[i]);
      result_T[i] = z;
      result_Z[i] = skip_elementwise_ ? z : elementwise_op(z);
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  /// Applies the operation when is_source_needed() is false
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentCompute const & valpha,
    FragmentCompute const & vbias) const {

    ElementwiseOpDispatcher elementwise_op(elementwise_);
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(valpha[i] * tmp_Accum[i], vbias[i]);
      result_T[i] = z;
      result_Z[i] = skip_elementwise_ ? z : elementwise_op(z);
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  /// Applies the operation when elementwise_op require arguments and is_source_needed() is true
  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
    ElementZ &Z,
    ElementT &T,
    ElementAccumulator const &AB,
    ElementC const &C,
    ElementCompute const & valpha,
    ElementCompute const & vbias,
    ElementwiseArgs const &elementwise_args) const {

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    ElementCompute tmp_Accum = NumericConverter<ElementCompute, ElementAccumulator>()(AB);
    ElementCompute tmp_C = NumericConverter<ElementCompute, ElementC>()(C);

    ElementCompute z = binary_op(valpha * tmp_Accum + beta_ * tmp_C, vbias);
    ElementCompute result_Z = skip_elementwise_ ? z : elementwise_op(z, elementwise_args);

    NumericConverter<ElementZ, ElementCompute> convert_z;
    Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      ElementCompute result_T = z;
      NumericConverter<ElementT, ElementCompute> convert_t;
      T = convert_t(result_T);
    }
  }

  /// Applies the operation when elementwise_op require arguments and is_source_needed() is true
  /// D = elementwise_op(vector_alpha * accumulator + vector_beta * source + bias)
  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
    ElementZ &Z,
    ElementT &T,
    ElementAccumulator const &AB,
    ElementC const &C,
    ElementCompute const & valpha,
    ElementCompute const & vbeta,
    ElementCompute const & vbias,
    ElementwiseArgs const &elementwise_args) const {

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    ElementCompute tmp_Accum = NumericConverter<ElementCompute, ElementAccumulator>()(AB);
    ElementCompute tmp_C = NumericConverter<ElementCompute, ElementC>()(C);

    ElementCompute z = binary_op(valpha * tmp_Accum + vbeta * tmp_C, vbias);
    ElementCompute result_Z = skip_elementwise_ ? z : elementwise_op(z, elementwise_args);

    NumericConverter<ElementZ, ElementCompute> convert_z;
    Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      ElementCompute result_T = z;
      NumericConverter<ElementT, ElementCompute> convert_t;
      T = convert_t(result_T);
    }
  }

  /// Applies the operation when elementwise_op require arguments and is_source_needed() is false
  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
    ElementZ &Z,
    ElementT &T,
    ElementAccumulator const &AB,
    ElementCompute const & valpha,
    ElementCompute const & vbias,
    ElementwiseArgs const &elementwise_args) const {

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    ElementCompute tmp_Accum = NumericConverter<ElementCompute, ElementAccumulator>()(AB);

    ElementCompute z = binary_op(valpha * tmp_Accum, vbias);
    ElementCompute result_Z = skip_elementwise_ ? z : elementwise_op(z, elementwise_args);

    NumericConverter<ElementZ, ElementCompute> convert_z;
    Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      ElementCompute result_T = z;
      NumericConverter<ElementT, ElementCompute> convert_t;
      T = convert_t(result_T);
    }
  }

  /// Applies the operation when is_source_needed() is true
  CUTLASS_HOST_DEVICE
  void operator()(
    ElementZ &Z,
    ElementT &T,
    ElementAccumulator const &AB,
    ElementC const &C,
    ElementCompute const & valpha,
    ElementCompute const & vbias) const {

    ElementwiseOpDispatcher elementwise_op(elementwise_);
    BinaryOp binary_op;

    ElementCompute tmp_Accum = NumericConverter<ElementCompute, ElementAccumulator>()(AB);
    ElementCompute tmp_C = NumericConverter<ElementCompute, ElementC>()(C);

    ElementCompute z = binary_op(valpha * tmp_Accum + beta_ * tmp_C, vbias);
    ElementCompute result_Z = skip_elementwise_ ? z : elementwise_op(z);

    NumericConverter<ElementZ, ElementCompute> convert_z;
    Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      ElementCompute result_T = z;
      NumericConverter<ElementT, ElementCompute> convert_t;
      T = convert_t(result_T);
    }
  }

  /// Applies the operation when is_source_needed() is false
  CUTLASS_HOST_DEVICE
  void operator()(
    ElementZ &Z,
    ElementT &T,
    ElementAccumulator const &AB,
    ElementCompute const & valpha,
    ElementCompute const & vbias) const {

    ElementwiseOpDispatcher elementwise_op(elementwise_);
    BinaryOp binary_op;

    ElementCompute tmp_Accum = NumericConverter<ElementCompute, ElementAccumulator>()(AB);

    ElementCompute z = binary_op(valpha * tmp_Accum, vbias);
    ElementCompute result_Z = skip_elementwise_ ? z : elementwise_op(z);

    NumericConverter<ElementZ, ElementCompute> convert_z;
    Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      ElementCompute result_T = z;
      NumericConverter<ElementT, ElementCompute> convert_t;
      T = convert_t(result_T);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
