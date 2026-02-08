/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

 #pragma once

 /////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Layout type identifier
enum class LayoutTypeID {
  kUnknown,
  kColumnMajor,
  kRowMajor,
  kBlockScalingTensor,          
  kColumnMajorInterleavedK2,
  kRowMajorInterleavedK2,
  kColumnMajorInterleavedK4,
  kRowMajorInterleavedK4,
  kColumnMajorInterleavedK16,
  kRowMajorInterleavedK16,
  kColumnMajorInterleavedK32,
  kRowMajorInterleavedK32,
  kColumnMajorInterleavedK64,
  kRowMajorInterleavedK64,
  kTensorNCHW,
  kTensorNCDHW,
  kTensorNHWC,
  kTensorNDHWC,
  kTensorNC32HW32,
  kTensorC32RSK32,
  kTensorNC64HW64,
  kTensorC64RSK64,
  kInvalid
};
  
/// Numeric data type
enum class NumericTypeID {
  kUnknown,
  kVoid,
  kB1,
  kU2,
  kU4,
  kU8,
  kU16,
  kU32,
  kU64,
  kS2,
  kS4,
  kS8,
  kS16,
  kS32,
  kS64,
  kFE4M3,
  kFE5M2,
  
  kFE2M3,
  kFE3M2,
  kFE2M1,
  kFUE8M0, 
  kFUE4M3, 
  kF8,
  kF6,
  kF4,
  
  kF16,
  kBF16, 
  kTF32,
  kF32,
  kF64,
  kCF16,
  kCBF16,
  kCF32,
  kCTF32,
  kCF64,
  kCS2,
  kCS4,
  kCS8,
  kCS16,
  kCS32,
  kCS64,
  kCU2,
  kCU4,
  kCU8,
  kCU16,
  kCU32,
  kCU64,
  kInvalid
};

/// Enumerated type describing a transformation on a complex value.
enum class ComplexTransform {
  kNone,
  kConjugate,
  kInvalid
};

/// Providers
enum class Provider {
  kNone,
  kCUTLASS,
  kReferenceHost,
  kReferenceDevice,
  kCUBLAS,
  kCUDNN,
  kInvalid
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumeration indicating the kind of operation
enum class OperationKind {
  kGemm,
  kBlockScaledGemm,
  kBlockwiseGemm,
  kRankK,
  kRank2K,
  kTrmm,
  kSymm,
  kConv2d,
  kConv3d,
  kEqGemm,
  kSparseGemm,
  kReduction,
  kGroupedGemm,
  kInvalid
};

/// Enumeration indicating whether scalars are in host or device memory
enum class ScalarPointerMode {
  kHost,
  kDevice,
  kInvalid
};

/// Describes how reductions are performed across threadblocks
enum class SplitKMode {
  kNone,
  kSerial,
  kParallel,
  kParallelSerial,
  kInvalid
};

/// Indicates the classificaition of the math instruction
enum class OpcodeClassID {
  kSimt,
  kTensorOp,
  kWmmaTensorOp,
  kSparseTensorOp,
  kBlockScaledOp,                
  kInvalid
};

enum class MathOperationID {
  kAdd,
  kMultiplyAdd,
  kMultiplyAddSaturate,
  kMultiplyAddMixedInputUpcast,
  kMultiplyAddFastBF16,
  kMultiplyAddFastF16,
  kMultiplyAddFastF32,
  kMultiplyAddComplex,
  kMultiplyAddComplexFastF32,
  kMultiplyAddGaussianComplex,
  kXorPopc,
  kInvalid
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumeration indicating what kind of GEMM operation to perform
enum class GemmKind {
  kGemm,
  kBlockScaledGemm,                
  kSparse,
  kUniversal,
  kPlanarComplex,
  kPlanarComplexArray,
  kGrouped,
  kInvalid
};

/// Enumeration indicating what kind of RankK update operation to perform
enum class RankKKind {
  kUniversal,
  kInvalid
};

/// Enumeration indicating what kind of TRMM operation to perform
enum class TrmmKind {
  kUniversal,
  kInvalid
};

/// Enumeration indicating what kind of SYMM/HEMM operation to perform
enum class SymmKind {
  kUniversal,
  kInvalid
};

/// Enumeration indicating what kind of Conv2d operation to perform
enum class ConvKind {
  kUnknown,
  kFprop,
  kDgrad,
  kWgrad,
  kInvalid
};

enum class ConvModeID {
  kCrossCorrelation,
  kConvolution,
  kInvalid
};

// Iterator algorithm enum in order of general performance-efficiency
enum class IteratorAlgorithmID {
  kNone,
  kAnalytic,
  kOptimized,
  kFixedChannels,
  kFewChannels,
  kInvalid
};


enum class EpilogueKind {
  kUnknown,
  kConversion,
  kLinearCombination,
  kLinearCombinationClamp,
  kLinearCombinationPlanarComplex,
  kLinearCombinationRelu,
  kLinearCombinationSigmoid,
  kInvalid
};


enum class RuntimeDatatype {
  kStatic,
  kE4M3,
  kE5M2,
  kE3M2,
  kE2M3,
  kE2M1,
  
  kInvalid
};


enum class RasterOrder {
  kAlongN,
  kAlongM,
  kHeuristic,
  kInvalid
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
