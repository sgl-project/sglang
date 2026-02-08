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

#include <cuda_fp16.h>

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

namespace detail {

template <typename Element, int ElementsPerAccess>
struct ArrayMaximum {

  CUTLASS_HOST_DEVICE
  Array<Element, ElementsPerAccess> operator()(
    Array<Element, ElementsPerAccess>  const &lhs,
    Array<Element, ElementsPerAccess>  const &rhs) const {

    Array<Element, ElementsPerAccess> result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      result[i] = platform::max(lhs[i].get(), rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<Element, ElementsPerAccess> operator()(
    Array<Element, ElementsPerAccess>  const &lhs,
    Element                                   rhs) const {

    Array<Element, ElementsPerAccess> result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      result[i] = platform::max(lhs[i].get(), rhs);
    }

    return result;
  }
};


/// Partial specialization: Element=float
template <int ElementsPerAccess>
struct ArrayMaximum<float, ElementsPerAccess> {

  CUTLASS_HOST_DEVICE
  Array<float, ElementsPerAccess> operator()(
    Array<float, ElementsPerAccess>  const &lhs,
    Array<float, ElementsPerAccess>  const &rhs) const {

    Array<float, ElementsPerAccess> result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      result[i] = fmax(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<float, ElementsPerAccess> operator()(
    Array<float, ElementsPerAccess>  const &lhs,
    float rhs) const {

    Array<float, ElementsPerAccess> result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      result[i] = fmax(lhs[i], rhs);
    }

    return result;
  }
};

/// Partial specialization: Element=half
template <int ElementsPerAccess>
struct ArrayMaximum<half_t, ElementsPerAccess> {

  CUTLASS_DEVICE
  Array<half_t, ElementsPerAccess> operator()(
    Array<half_t, ElementsPerAccess>  const &lhs,
    Array<half_t, ElementsPerAccess>  const &rhs) const {

    Array<half_t, ElementsPerAccess> result;

    #if __CUDA_ARCH__ >= 800
    int const kVectorCount = ElementsPerAccess / 2;


    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(lhs.raw_data());
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(rhs.raw_data());
    __half2       *res_ptr = reinterpret_cast<__half2 *>(result.raw_data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVectorCount; ++i) {
      res_ptr[i] = __hmax2(lhs_ptr[i], rhs_ptr[i]);
    }

    static_assert(!(ElementsPerAccess % 2), "Output array must be divisible by vector length.");

    #else
    __half const *lhs_ptr = reinterpret_cast<__half const *>(lhs.raw_data());
    __half const *rhs_ptr = reinterpret_cast<__half const *>(rhs.raw_data());
    __half       *res_ptr = reinterpret_cast<__half       *>(result.raw_data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      res_ptr[i] = ((lhs_ptr[i] < rhs_ptr[i]) ? rhs_ptr[i] : lhs_ptr[i]);
    }

    #endif

    return result;
  }

  CUTLASS_DEVICE
  Array<half_t, ElementsPerAccess> operator()(
    Array<half_t, ElementsPerAccess>  const &lhs,
    half_t const &rhs) const {

    Array<half_t, ElementsPerAccess> result;

    #if __CUDA_ARCH__ >= 800
    int const kVectorCount = ElementsPerAccess / 2;


    __half rhs_raw = reinterpret_cast<__half const &>(rhs);
    __half2 rhs_pair = __half2half2(rhs_raw);

    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(lhs.raw_data());
    __half2       *res_ptr = reinterpret_cast<__half2 *>(result.raw_data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVectorCount; ++i) {
      res_ptr[i] = __hmax2(lhs_ptr[i], rhs_pair);
    }

    static_assert(!(ElementsPerAccess % 2), "Output array must be divisible by vector length.");

    #else

    __half const *lhs_ptr = reinterpret_cast<__half const *>(lhs.raw_data());
    __half const  rhs_raw = reinterpret_cast<__half const &>(rhs);
    __half       *res_ptr = reinterpret_cast<__half       *>(result.raw_data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      res_ptr[i] = ((lhs_ptr[i] < rhs_raw) ? rhs_raw : lhs_ptr[i]);
    }

    #endif

    return result;
  }
};

/// Partial specialization: Element=bfloat16_t
template <int ElementsPerAccess>
struct ArrayMaximum<bfloat16_t, ElementsPerAccess> {

  using NvType   = __nv_bfloat16;
  using NvTypeV2 = __nv_bfloat162;

  CUTLASS_DEVICE
  Array<bfloat16_t, ElementsPerAccess> operator()(
    Array<bfloat16_t, ElementsPerAccess>  const &lhs,
    Array<bfloat16_t, ElementsPerAccess>  const &rhs) const {

    Array<bfloat16_t, ElementsPerAccess> result;

    #if __CUDA_ARCH__ >= 800
    int const kVectorCount = ElementsPerAccess / 2;


    NvTypeV2 const *lhs_ptr = reinterpret_cast<NvTypeV2 const *>(lhs.raw_data());
    NvTypeV2 const *rhs_ptr = reinterpret_cast<NvTypeV2 const *>(rhs.raw_data());
    NvTypeV2       *res_ptr = reinterpret_cast<NvTypeV2 *>(result.raw_data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVectorCount; ++i) {
      res_ptr[i] = __hmax2(lhs_ptr[i], rhs_ptr[i]);
    }

    #else
    NvType const *lhs_ptr = reinterpret_cast<NvType const *>(lhs.raw_data());
    NvType const *rhs_ptr = reinterpret_cast<NvType const *>(rhs.raw_data());
    NvType       *res_ptr = reinterpret_cast<NvType       *>(result.raw_data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      res_ptr[i] = ((lhs_ptr[i] < rhs_ptr[i]) ? rhs_ptr[i] : lhs_ptr[i]);
    }

    #endif

    return result;
  }

  CUTLASS_DEVICE
  Array<bfloat16_t, ElementsPerAccess> operator()(
    Array<bfloat16_t, ElementsPerAccess>  const &lhs,
    bfloat16_t                                   rhs) const {

    Array<bfloat16_t, ElementsPerAccess> result;

    #if __CUDA_ARCH__ >= 800
    int const kVectorCount = ElementsPerAccess / 2;


    NvType rhs_raw = reinterpret_cast<NvType const &>(rhs);
    NvTypeV2 rhs_pair = __bfloat162bfloat162(rhs_raw);

    NvTypeV2 const *lhs_ptr = reinterpret_cast<NvTypeV2 const *>(lhs.raw_data());
    NvTypeV2       *res_ptr = reinterpret_cast<NvTypeV2 *>(result.raw_data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVectorCount; ++i) {
      res_ptr[i] = __hmax2(lhs_ptr[i], rhs_pair);
    }

    static_assert(!(ElementsPerAccess % 2), "Output array must be divisible by vector length.");

    #else

    NvType const *lhs_ptr = reinterpret_cast<NvType const *>(lhs.raw_data());
    NvType const  rhs_raw = reinterpret_cast<NvType const &>(rhs);
    NvType       *res_ptr = reinterpret_cast<NvType       *>(result.raw_data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      res_ptr[i] = ((lhs_ptr[i] < rhs_raw) ? rhs_raw : lhs_ptr[i]);
    }

    #endif

    return result;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, int ElementsPerAccess>
struct ReluConditional {

  CUTLASS_HOST_DEVICE
  void operator()(
    bool conditional[],
    Array<Element, ElementsPerAccess> const &fragment, 
    Element threshold) const {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      conditional[i] = !(fragment[i] < threshold);
    }
  }
};

template <int ElementsPerAccess>
struct ReluConditional<half_t, ElementsPerAccess> {

  CUTLASS_DEVICE
  void operator()(
    bool conditional[],
    Array<half_t, ElementsPerAccess> const &fragment, 
    half_t threshold) const {

    __half y = reinterpret_cast<__half const &>(threshold);
    __half const *x = reinterpret_cast<__half const *>(fragment.raw_data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      conditional[i] = !__hlt(x[i], y);
    }
  }
};

template <int ElementsPerAccess>
struct ReluConditional<bfloat16_t, ElementsPerAccess> {

  CUTLASS_DEVICE
  void operator()(
    bool conditional[],
    Array<bfloat16_t, ElementsPerAccess> const &fragment,
    bfloat16_t threshold) const {

    __nv_bfloat16 y = reinterpret_cast<__nv_bfloat16 const &>(threshold);
    __nv_bfloat16 const *x = reinterpret_cast<__nv_bfloat16 const *>(fragment.raw_data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      conditional[i] = !__hlt(x[i], y);
    }
  }
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This is a partial specialization for fused Bias and ReLU. It supports the option of packing
/// ReLU conditionals in a bit vector that may be used by backwards passes as an optimization.
///
/// This class can only be used with cutlass::epilogue::threadblock::EpilogueWithBroadcast<>.
///
/// This base class is meant to define the concept required of the
/// EpilogueWithBroadcast::OutputOp
template <
  typename ElementC_,
  typename ElementAccumulator_,
  typename ElementCompute_,
  typename ElementZ_,
  int ElementsPerAccess,
  bool StoreT_ = true,
  typename ElementVector_ = ElementC_
>
class LinearCombinationBiasRelu {
public:

  using ElementOutput = ElementC_;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementZ = ElementZ_;
  using ElementVector = ElementVector_;

  using ElementT = uint1b_t;

  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;

  using ElementwiseOp = ReLu<ElementCompute>;
  using BinaryOp = plus<ElementCompute>;

  // Indicates that this epilogue applies only one binary operation
  static bool const kIsSingleSource = true;

  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = Array<ElementOutput, kElementsPerAccess>;
  using FragmentZ = Array<ElementZ, kElementsPerAccess>;
  using FragmentT = Array<ElementT, kElementsPerAccess>;

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
    ElementZ threshold;                    ///< ReLu threshold

    //
    // Methods
    //
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): 
      alpha(ElementCompute(1)), 
      beta(ElementCompute()), 
      alpha_ptr(nullptr), 
      beta_ptr(nullptr),
      threshold(ElementCompute()) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta,
      ElementCompute threshold_ = ElementCompute()
    ): 
      alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) {

      NumericConverter<ElementZ, ElementCompute> convert_threshold;

      threshold = convert_threshold(threshold_);
    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha
    ): alpha(alpha), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr), threshold(ElementZ()) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr,
      ElementCompute threshold_ = ElementCompute()
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {

      NumericConverter<ElementZ, ElementCompute> convert_threshold;

      threshold = convert_threshold(threshold_);
    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(nullptr), threshold(ElementZ()) {
    }

  };

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;
  ElementZ threshold_;

public:

  //
  // Methods
  //

  /// Constructor from Params
  CUTLASS_HOST_DEVICE
  LinearCombinationBiasRelu(Params const &params) {

    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    threshold_ = params.threshold;
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
      // set to NaN to make ReLU no-op for all except last k partitions
      int64_t allones = -1;
      threshold_ = reinterpret_cast<ElementZ const &>(allones);
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

    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C = NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result_Z;

    bool conditions[kElementsPerAccess];

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {

      ElementCompute z = alpha_ * tmp_Accum[i];
      z += beta_ * tmp_C[i];

      z = binary_op(z, V[i]);
      result_Z[i] = z;
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    //
    // Compute condition
    //

    detail::ReluConditional<ElementZ, kElementsPerAccess> relu_conditional;
    relu_conditional(conditions, frag_Z, threshold_);

    detail::ArrayMaximum<ElementZ, kElementsPerAccess> maximum_op;
    frag_Z = maximum_op(frag_Z, threshold_);

    if (kStoreT) {
      PackPredicates<kElementsPerAccess> pack_predicates;
      frag_T = pack_predicates(conditions); 
    }
  }

  /// Applies the operation when is_source_needed() is false
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z, 
    FragmentT &frag_T, 
    FragmentAccumulator const &AB,
    FragmentCompute const &V) const {

    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute result_Z;

    bool conditions[kElementsPerAccess];

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(alpha_ * tmp_Accum[i], V[i]);
      result_Z[i] = z;
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    //
    // Compute condition
    //

    detail::ReluConditional<ElementZ, kElementsPerAccess> relu_conditional;
    relu_conditional(conditions, frag_Z, threshold_);

    detail::ArrayMaximum<ElementZ, kElementsPerAccess> maximum_op;
    frag_Z = maximum_op(frag_Z, threshold_);

    // 
    // Compute conditions
    //

    //
    // Store
    //
    if (kStoreT) {
      PackPredicates<kElementsPerAccess> pack_predicates;
      frag_T = pack_predicates(conditions);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
