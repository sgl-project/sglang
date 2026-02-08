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
/*! 
  \file

  \brief CUTLASS Library is an object-oriented approach to managing operations implemented by CUTLASS.

  Generally,
    
    description   - compile-time constant parameters used to instantiate an operation

    configuration - runtime parameters with computationally expensive initialization 
    
    arguments     - runtime parameters that may be passed to an initialized operation with low
                    computational overhead
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/complex.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/library/library.h"
#include "cutlass/library/arch_mappings.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct NumericTypeMap;

template <> struct NumericTypeMap<void> {
  static NumericTypeID const kId = NumericTypeID::kVoid;
};

template <> struct NumericTypeMap<cutlass::uint1b_t> {
  static NumericTypeID const kId = NumericTypeID::kB1;
};

template <> struct NumericTypeMap<cutlass::int2b_t> {
  static NumericTypeID const kId = NumericTypeID::kS2;
};

template <> struct NumericTypeMap<cutlass::int4b_t> {
  static NumericTypeID const kId = NumericTypeID::kS4;
};

template <> struct NumericTypeMap<int8_t> {
  static NumericTypeID const kId = NumericTypeID::kS8;
};

template <> struct NumericTypeMap<int16_t> {
  static NumericTypeID const kId = NumericTypeID::kS16;
};

template <> struct NumericTypeMap<int32_t> {
  static NumericTypeID const kId = NumericTypeID::kS32;
};

template <> struct NumericTypeMap<int64_t> {
  static NumericTypeID const kId = NumericTypeID::kS64;
};

template <> struct NumericTypeMap<cutlass::uint2b_t> {
  static NumericTypeID const kId = NumericTypeID::kU2;
};

template <> struct NumericTypeMap<cutlass::uint4b_t> {
  static NumericTypeID const kId = NumericTypeID::kU4;
};

template <> struct NumericTypeMap<uint8_t> {
  static NumericTypeID const kId = NumericTypeID::kU8;
};

template <> struct NumericTypeMap<cutlass::float_e4m3_t> {
  static NumericTypeID const kId = NumericTypeID::kFE4M3;
};

template <> struct NumericTypeMap<cutlass::float_e5m2_t> {
  static NumericTypeID const kId = NumericTypeID::kFE5M2;
};


template <> struct NumericTypeMap<cutlass::float_e2m3_t> {
  static NumericTypeID const kId = NumericTypeID::kFE2M3;
};

template <> struct NumericTypeMap<cutlass::float_e3m2_t> {
  static NumericTypeID const kId = NumericTypeID::kFE3M2;
};

template <> struct NumericTypeMap<cutlass::float_e2m1_t> {
  static NumericTypeID const kId = NumericTypeID::kFE2M1;
};
template <> struct NumericTypeMap<cutlass::float_ue8m0_t> {
  static NumericTypeID const kId = NumericTypeID::kFUE8M0;
};

template <> struct NumericTypeMap<cutlass::float_ue4m3_t> {
  static NumericTypeID const kId = NumericTypeID::kFUE4M3;
};


template <> struct NumericTypeMap<uint16_t> {
  static NumericTypeID const kId = NumericTypeID::kU16;
};

template <> struct NumericTypeMap<uint32_t> {
  static NumericTypeID const kId = NumericTypeID::kU32;
};

template <> struct NumericTypeMap<uint64_t> {
  static NumericTypeID const kId = NumericTypeID::kU64;
};

template <> struct NumericTypeMap<cutlass::half_t> {
  static NumericTypeID const kId = NumericTypeID::kF16;
};

template <> struct NumericTypeMap<float> {
  static NumericTypeID const kId = NumericTypeID::kF32;
};

template <> struct NumericTypeMap<double> {
  static NumericTypeID const kId = NumericTypeID::kF64;
};

template <> struct NumericTypeMap<cutlass::complex<cutlass::half_t> > {
  static NumericTypeID const kId = NumericTypeID::kCF16;
};

template <> struct NumericTypeMap<cutlass::complex<float> > {
  static NumericTypeID const kId = NumericTypeID::kCF32;
};

template <> struct NumericTypeMap<cutlass::complex<double> > {
  static NumericTypeID const kId = NumericTypeID::kCF64;
};

template <> struct NumericTypeMap<cutlass::bfloat16_t> {
  static NumericTypeID const kId = NumericTypeID::kBF16;
};

template <> struct NumericTypeMap<cutlass::tfloat32_t> {
  static NumericTypeID const kId = NumericTypeID::kTF32;
};




template <> struct NumericTypeMap<cutlass::type_erased_dynamic_float8_t> {
  static NumericTypeID const kId = NumericTypeID::kF8;
};

template <> struct NumericTypeMap<cutlass::type_erased_dynamic_float6_t> {
  static NumericTypeID const kId = NumericTypeID::kF6;
};

template <> struct NumericTypeMap<cutlass::type_erased_dynamic_float4_t> {
  static NumericTypeID const kId = NumericTypeID::kF4;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct MathOperationMap {
  static MathOperationID const kId = MathOperationID::kInvalid;
};

template <> struct MathOperationMap<cutlass::arch::OpMultiplyAdd> {
  static MathOperationID const kId = MathOperationID::kMultiplyAdd;
};

template <> struct MathOperationMap<cutlass::arch::OpMultiplyAddFastBF16> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddFastBF16;
};

template <> struct MathOperationMap<cutlass::arch::OpMultiplyAddFastF16> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddFastF16;
};

template <> struct MathOperationMap<cutlass::arch::OpMultiplyAddSaturate> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddSaturate;
};

template <> struct MathOperationMap<cutlass::arch::OpMultiplyAddMixedInputUpcast> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddMixedInputUpcast;
};

template <> struct MathOperationMap<cutlass::arch::OpMultiplyAddComplex> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddComplex;
};

template <> struct MathOperationMap<cutlass::arch::OpMultiplyAddGaussianComplex> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddGaussianComplex;
};

template <> struct MathOperationMap<cutlass::arch::OpXorPopc> {
  static MathOperationID const kId = MathOperationID::kXorPopc;
};


template <> struct MathOperationMap<cutlass::arch::OpMultiplyAddFastF32> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddFastF32;
};

template <> struct MathOperationMap<cutlass::arch::OpMultiplyAddComplexFastF32> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddComplexFastF32;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct LayoutMap;

template <> struct LayoutMap<cutlass::layout::ColumnMajor> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajor;
};

template <> struct LayoutMap<cutlass::layout::RowMajor> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajor;
};

template <> struct LayoutMap<cutlass::layout::ColumnMajorInterleaved<2>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK2;
};

template <> struct LayoutMap<cutlass::layout::RowMajorInterleaved<2>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK2;
};

template <> struct LayoutMap<cutlass::layout::ColumnMajorInterleaved<4>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK4;
};

template <> struct LayoutMap<cutlass::layout::RowMajorInterleaved<4>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK4;
};

template <> struct LayoutMap<cutlass::layout::ColumnMajorInterleaved<16>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK16;
};

template <> struct LayoutMap<cutlass::layout::RowMajorInterleaved<16>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK16;
};

template <> struct LayoutMap<cutlass::layout::ColumnMajorInterleaved<32>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK32;
};

template <> struct LayoutMap<cutlass::layout::RowMajorInterleaved<32>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK32;
};

template <> struct LayoutMap<cutlass::layout::ColumnMajorInterleaved<64>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK64;
};

template <> struct LayoutMap<cutlass::layout::RowMajorInterleaved<64>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK64;
};

template <> struct LayoutMap<cutlass::layout::TensorNHWC> {
  static LayoutTypeID const kId = LayoutTypeID::kTensorNHWC;
};

template <> struct LayoutMap<cutlass::layout::TensorNDHWC> {
  static LayoutTypeID const kId = LayoutTypeID::kTensorNDHWC;
};

template <> struct LayoutMap<cutlass::layout::TensorNCxHWx<32>> {
  static LayoutTypeID const kId = LayoutTypeID::kTensorNC32HW32;
};

template <> struct LayoutMap<cutlass::layout::TensorNCxHWx<64>> {
  static LayoutTypeID const kId = LayoutTypeID::kTensorNC64HW64;
};

template <> struct LayoutMap<cutlass::layout::TensorCxRSKx<32>> {
  static LayoutTypeID const kId = LayoutTypeID::kTensorC32RSK32;
};

template <> struct LayoutMap<cutlass::layout::TensorCxRSKx<64>> {
  static LayoutTypeID const kId = LayoutTypeID::kTensorC64RSK64;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct OpcodeClassMap;

template <> struct OpcodeClassMap<arch::OpClassSimt> {
  static OpcodeClassID const kId = OpcodeClassID::kSimt;
};

template <> struct OpcodeClassMap<arch::OpClassTensorOp> {
  static OpcodeClassID const kId = OpcodeClassID::kTensorOp;
};

template <> struct OpcodeClassMap<arch::OpClassSparseTensorOp> {
  static OpcodeClassID const kId = OpcodeClassID::kSparseTensorOp;
};


template <> struct OpcodeClassMap<arch::OpClassBlockScaledTensorOp> {
  static OpcodeClassID const kId = OpcodeClassID::kBlockScaledOp;
};


template <> struct OpcodeClassMap<arch::OpClassWmmaTensorOp> {
  static OpcodeClassID const kId = OpcodeClassID::kWmmaTensorOp;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <cutlass::ComplexTransform Transform> struct ComplexTransformMap;

template <> struct ComplexTransformMap<cutlass::ComplexTransform::kNone> {
  static cutlass::library::ComplexTransform const kId = cutlass::library::ComplexTransform::kNone;
};

template <> struct ComplexTransformMap<cutlass::ComplexTransform::kConjugate> {
  static cutlass::library::ComplexTransform const kId = cutlass::library::ComplexTransform::kConjugate;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <cutlass::conv::Mode  T> struct ConvModeMap;

template <> struct ConvModeMap<conv::Mode::kCrossCorrelation> {
  static ConvModeID const kId = ConvModeID::kCrossCorrelation;
};

template <> struct ConvModeMap<conv::Mode::kConvolution> {
  static ConvModeID const kId = ConvModeID::kConvolution;
};


template <cutlass::conv::Operator  T> struct ConvKindMap;

template <> struct ConvKindMap<conv::Operator::kFprop> {
  static ConvKind const kId = ConvKind::kFprop;
};

template <> struct ConvKindMap<conv::Operator::kDgrad> {
  static ConvKind const kId = ConvKind::kDgrad;
};

template <> struct ConvKindMap<conv::Operator::kWgrad> {
  static ConvKind const kId = ConvKind::kWgrad;
};


template <cutlass::conv::IteratorAlgorithm  T> struct IteratorAlgorithmMap;

template <> struct IteratorAlgorithmMap<conv::IteratorAlgorithm::kAnalytic> {
  static IteratorAlgorithmID const kId = IteratorAlgorithmID::kAnalytic;
};

template <> struct IteratorAlgorithmMap<conv::IteratorAlgorithm::kOptimized> {
  static IteratorAlgorithmID const kId = IteratorAlgorithmID::kOptimized;
};

template <> struct IteratorAlgorithmMap<conv::IteratorAlgorithm::kFixedChannels> {
  static IteratorAlgorithmID const kId = IteratorAlgorithmID::kFixedChannels;
};

template <> struct IteratorAlgorithmMap<conv::IteratorAlgorithm::kFewChannels> {
  static IteratorAlgorithmID const kId = IteratorAlgorithmID::kFewChannels;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, typename Layout>
TensorDescription make_TensorDescription(int alignment = 1) {
  TensorDescription desc;

  desc.element = NumericTypeMap<Element>::kId;
  desc.layout = LayoutMap<Layout>::kId;
  desc.alignment = alignment;
  desc.log_extent_range = int(sizeof(typename Layout::TensorCoord::Index) - 1) * 8;
  desc.log_stride_range = int(sizeof(typename Layout::Stride::Index) - 1) * 8;

  return desc;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
