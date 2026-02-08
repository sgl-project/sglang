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
    \brief Templates exposing SIMD operators
*/

#pragma once

#include "cutlass/arch/array.h"
#include "cutlass/arch/numeric_types.h"

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Element-wise operators
//

CUTLASS_HOST_DEVICE
template <typename T, int N>
Array<T, N> operator*(Array<T, N> const &a, Array<T, N> const &b) {
  Array<T, N> d;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; ++i) {
    d[i] = a[i] * b[i];
  }
  return d;
}

CUTLASS_HOST_DEVICE
template <typename T, int N>
Array<T, N> operator+(Array<T, N> const &a, Array<T, N> const &b) {
  Array<T, N> d;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; ++i) {
    d[i] = a[i] + b[i];
  }
  return d;
}

CUTLASS_HOST_DEVICE
template <typename T, int N>
Array<T, N> operator-(Array<T, N> const &a, Array<T, N> const &b) {
  Array<T, N> d;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; ++i) {
    d[i] = a[i] - b[i];
  }
  return d;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Multiply-accumulate operators
//

CUTLASS_HOST_DEVICE
template <typename T, int N>
Array<T, N> mac(Array<T, N> const &a, Array<T, N> const &b, Array<T, N> const &c) {
  Array<T, N> d;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; ++i) {
    d[i] = a[i] * b[i] + c[i];
  }
  return d;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Dot product operator
//

CUTLASS_HOST_DEVICE
template <typename Element, typename Accumulator, int N>
Accumulator dot(Array<T, N> const &a, Array<T, N> const &b, Accumulator accum) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; ++i) {
    accum += a[i] * b[i];
  }
  return accum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "simd_sm60.h"
#include "simd_sm61.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
