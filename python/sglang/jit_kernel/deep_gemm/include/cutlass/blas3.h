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
  \brief Basic include for CUTLASS BLAS3/HPC code.
    
  
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/blas3_types.h"
#include "cutlass/coord.h"
#include "cutlass/complex.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_types.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines FillMode inversions
template <FillMode kFillMode>
struct InvertFillMode;

/// Invert FillMode lower to upper
template <>
struct InvertFillMode<FillMode::kLower> {
  static FillMode const mode = FillMode::kUpper;
};

/// Invert FillMode upper to lower
template <>
struct InvertFillMode<FillMode::kUpper> {
  static FillMode const mode = FillMode::kLower;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines SideMode inversions
template <SideMode kSideMode>
struct InvertSideMode;

/// Invert SideMode left to right
template <>
struct InvertSideMode<SideMode::kLeft> {
  static SideMode const mode = SideMode::kRight;
};

/// Invert SideMode right to left
template <>
struct InvertSideMode<SideMode::kRight> {
  static SideMode const mode = SideMode::kLeft;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines correct compare operation for Triangular matrix boundary
template <FillMode kFillMode, DiagType kDiagType = DiagType::kNonUnit>
struct TrMatrixCompareOp {
  using Index = int32_t;
  using Type = typename platform::conditional<
                        (kFillMode == FillMode::kLower), 
                        greater_equal<Index>, 
                        less_equal<Index>>::type;
};

template <FillMode kFillMode>
struct TrMatrixCompareOp <kFillMode, DiagType::kUnit> {
   using Index = int32_t;
   using Type = typename platform::conditional<
                        (kFillMode == FillMode::kLower), 
                        greater_equal<Index>, 
                        less_equal<Index>>::type;
};

template <FillMode kFillMode>
struct TrMatrixCompareOp <kFillMode, DiagType::kZero> {
   using Index = int32_t;
   using Type = typename platform::conditional<
                        (kFillMode == FillMode::kLower), 
                        greater<Index>, 
                        less<Index>>::type;
};
////////////////////////////////////////////////////////////////////////////////////////////////////
// Returns precision in terms of bits (based on datatype) to fill tensors with.
// Defaults to 5 bits of mantissa for TF32 and FP32 (with implicit round-offs).
// Also defines acceptable mantissa result variance/error.
template <typename Element>
struct MantissaInBits {
  static int constexpr bits = 5;
  static double constexpr error = 1.0e-7;
};

// Full precision is supported for FP64
template <>
struct MantissaInBits<double> {
  static int constexpr bits = 30;
  static double constexpr error = 1.0e-15;
};

template <>
struct MantissaInBits<cutlass::complex<double>> {
  static int constexpr bits = 30;
  static double constexpr error = 1.0e-14;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////

