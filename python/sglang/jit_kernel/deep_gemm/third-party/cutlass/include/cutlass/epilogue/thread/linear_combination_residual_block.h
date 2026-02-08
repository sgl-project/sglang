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
  \brief Epilogue functor specialized for residual blocks in deep neural networks.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/detail.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/// Models a residual block of the form: UnaryOp(BinaryOp(BinaryOp(ActivationOp(TensorOp(X) + bias), residual1), residual2))
template <typename ElementOutput_, typename ElementAccumulator_,
          typename ElementCompute_, typename ElementC_, int ElementsPerAccess,
          template <typename T> class ActivationOp_,
          template <typename T> class BinaryOp1_,
          template <typename T> class UnaryOp_,
          template <typename T> class BinaryOp2_ = detail::NoOp,
          bool StoreT_ = false,
          typename ElementVector_ = ElementC_>
class LinearCombinationResidualBlock {
public:
  static bool const kIsSingleSource = false;

  using ElementOutput = ElementC_;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementVector = ElementVector_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;

  using UnaryOp = UnaryOp_<Array<ElementCompute, kCount>>;
  using BinaryOp1 = BinaryOp1_<Array<ElementCompute, kCount>>;
  using BinaryOp2 = BinaryOp2_<Array<ElementCompute, kCount>>;
  using ActivationOp = ActivationOp_<Array<ElementCompute, kCount>>;

  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = Array<ElementC, kElementsPerAccess>;
  using FragmentOutput = Array<ElementOutput, kElementsPerAccess>;

  using ElementZ = ElementOutput_;
  using ElementT = ElementZ;
  using FragmentZ = Array<ElementZ, kElementsPerAccess>;
  using FragmentT = Array<ElementT, kElementsPerAccess>;

  static bool const kIsHeavy = true;
  static bool const kStoreZ = true;
  static bool const kStoreT = StoreT_;

  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta;                   ///< scales residual input
    ElementCompute const *alpha_ptr{nullptr};       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr{nullptr};        ///< pointer to residual scalar - if not null, loads it from memory

    CUTLASS_HOST_DEVICE
    Params() : alpha(ElementCompute(1)), beta(ElementCompute(1)) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta)
        : alpha(alpha), beta(beta) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr, ElementCompute const *beta_ptr)
        : alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {}
  };

private:

  ElementCompute alpha_;
  ElementCompute beta_;
  bool skip_elementwise_;

public:

  /// Constructor from Params
  CUTLASS_HOST_DEVICE
  LinearCombinationResidualBlock(Params const &params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    skip_elementwise_ = false;
  }

  /// The "source" tensor corresponds to the residual input
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  /// Functionally required for serial reduction in the epilogue
  /// IMPORTANT: Split-k is supported only when ActivationOp is Identity.
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      skip_elementwise_ = true;
    }
  }

  /// Applies the operation UnaryOp(BinaryOp(BinaryOp(ActivationOp(AB + bias), residual1), residual2))
  CUTLASS_HOST_DEVICE
  void operator()(FragmentOutput &frag_Z, FragmentOutput &, FragmentAccumulator const &AB,
                  FragmentC const &residual1, FragmentC const &residual2,
                  FragmentCompute const &bias) const {
    UnaryOp unary_op;
    BinaryOp1 binary_op1;
    BinaryOp2 binary_op2;
    ActivationOp activation;

    FragmentCompute tmp_Accum =
        NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_residual1 =
        NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(residual1);
    FragmentCompute tmp_residual2 =
        NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(residual2);

    FragmentCompute z =
        binary_op2(binary_op1(activation(alpha_ * tmp_Accum + bias), beta_ * tmp_residual1), beta_ * tmp_residual2);
    FragmentCompute result_Z = skip_elementwise_ ? z : unary_op(z);

    NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);
  }

  /// Should never be called
  CUTLASS_HOST_DEVICE
  void operator()(FragmentOutput &, FragmentOutput &, FragmentAccumulator const &,
                  FragmentCompute const &) const {}
};

/// Models a residual block of the form: UnaryOp(BinaryOp(ActivationOp(TensorOp(X) + bias), residual))
template <typename ElementOutput_, typename ElementAccumulator_,
          typename ElementCompute_, typename ElementC_, int ElementsPerAccess,
          template <typename T> class ActivationOp_,
          template <typename T> class BinaryOp1_,
          template <typename T> class UnaryOp_,
          bool StoreT_,
          typename ElementVector_>
class LinearCombinationResidualBlock<ElementOutput_, ElementAccumulator_,
          ElementCompute_, ElementC_, ElementsPerAccess,
          ActivationOp_, BinaryOp1_, UnaryOp_,
          detail::NoOp, StoreT_, ElementVector_> {
public:
  static bool const kIsSingleSource = true;

  using ElementOutput = ElementC_;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementVector = ElementVector_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;

  using UnaryOp = UnaryOp_<Array<ElementCompute, kCount>>;
  using BinaryOp = BinaryOp1_<Array<ElementCompute, kCount>>;
  using ActivationOp = ActivationOp_<Array<ElementCompute, kCount>>;

  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = Array<ElementC, kElementsPerAccess>;
  using FragmentOutput = Array<ElementOutput, kElementsPerAccess>;

  using ElementZ = ElementOutput_;
  using ElementT = ElementZ;
  using FragmentZ = Array<ElementZ, kElementsPerAccess>;
  using FragmentT = Array<ElementT, kElementsPerAccess>;

  static bool const kIsHeavy = true;
  static bool const kStoreZ = true;
  static bool const kStoreT = StoreT_;

  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta;                   ///< scales residual input
    ElementCompute const *alpha_ptr{nullptr};       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr{nullptr};        ///< pointer to residual scalar - if not null, loads it from memory

    CUTLASS_HOST_DEVICE
    Params() : alpha(ElementCompute(1)), beta(ElementCompute(1)) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta)
        : alpha(alpha), beta(beta) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr, ElementCompute const *beta_ptr)
        : alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {}
  };

private:

  ElementCompute alpha_;
  ElementCompute beta_;
  bool skip_elementwise_;

public:

  /// Constructor from Params
  CUTLASS_HOST_DEVICE
  LinearCombinationResidualBlock(Params const &params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    skip_elementwise_ = false;
  }

  /// The "source" tensor corresponds to the residual input
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  /// Functionally required for serial reduction in the epilogue
  /// IMPORTANT: Split-k is supported only when ActivationOp is Identity.
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      skip_elementwise_ = true;
    }
  }

  /// Applies the operation UnaryOp(BinaryOp(ActivationOp(AB + bias), residual))
  CUTLASS_HOST_DEVICE
  void operator()(FragmentOutput &frag_Z, FragmentOutput &, FragmentAccumulator const &AB,
                  FragmentC const &residual,
                  FragmentCompute const &bias) const {
    UnaryOp unary_op;
    BinaryOp binary_op;
    ActivationOp activation;

    FragmentCompute tmp_Accum =
        NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_residual =
        NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(residual);

    FragmentCompute z =
        binary_op(activation(alpha_ * tmp_Accum + bias), beta_ * tmp_residual);
    FragmentCompute result_Z = skip_elementwise_ ? z : unary_op(z);

    NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);
  }

  /// Should never be called
  CUTLASS_HOST_DEVICE
  void operator()(FragmentOutput &, FragmentOutput &, FragmentAccumulator const &,
                  FragmentCompute const &) const {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
