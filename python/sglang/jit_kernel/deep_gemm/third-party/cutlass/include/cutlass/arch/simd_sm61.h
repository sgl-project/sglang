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
    \brief Templates exposing SIMD operators for SM61
*/

#pragma once

#include "simd.h"

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Dot product operator - specialized for int32_t <- (int8_t * int8_t) x 4 + int32_t
CUTLASS_HOST_DEVICE
template <>
int32_t dot(Array<int8_t, 4> const &a, Array<int8_t, 4> const &b, int32_t accum) {

  return accum;
}

/// Dot product operator - specialized for int32_t <- (uint8_t * int8_t) x 4 + int32_t
CUTLASS_HOST_DEVICE
template <>
int32_t dot(Array<uint8_t, 4> const &a, Array<int8_t, 4> const &b, int32_t accum) {

  return accum;
}

/// Dot product operator - specialized for int32_t <- (int8_t * uint8_t) x 4 + int32_t
CUTLASS_HOST_DEVICE
template <>
int32_t dot(Array<int8_t, 4> const &a, Array<uint8_t, 4> const &b, int32_t accum) {

  return accum;
}

/// Dot product operator - specialized for int32_t <- (uint8_t * uint8_t) x 4 + int32_t
CUTLASS_HOST_DEVICE
template <>
int32_t dot(Array<uint8_t, 4> const &a, Array<uint8_t, 4> const &b, int32_t accum) {

  return accum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Dot product operator - specialized for int32_t <- (int16_t * int8_t) x 2 + int32_t
CUTLASS_HOST_DEVICE
template <>
int32_t dot(Array<int16_t, 2> const &a, Array<int8_t, 2> const &b, int32_t accum) {

  return accum;
}

/// Dot product operator - specialized for int32_t <- (uint16_t * int8_t) x 2 + int32_t
CUTLASS_HOST_DEVICE
template <>
int32_t dot(Array<uint16_t, 2> const &a, Array<int8_t, 2> const &b, int32_t accum) {

  return accum;
}

/// Dot product operator - specialized for int32_t <- (int16_t * int8_t) x 2 + int32_t
CUTLASS_HOST_DEVICE
template <>
int32_t dot(Array<int16_t, 2> const &a, Array<uint8_t, 2> const &b, int32_t accum) {

  return accum;
}

/// Dot product operator - specialized for int32_t <- (uint16_t * int8_t) x 2 + int32_t
CUTLASS_HOST_DEVICE
template <>
int32_t dot(Array<uint16_t, 2> const &a, Array<uint8_t, 2> const &b, int32_t accum) {

  return accum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Dot product operator - specialized for int32_t <- (int16_t * int16_t) x 2 + int32_t
CUTLASS_HOST_DEVICE
template <>
int32_t dot(Array<int16_t, 2> const &a, Array<int16_t, 2> const &b, int32_t accum) {

  return accum;
}

/// Dot product operator - specialized for int32_t <- (uint16_t * int16_t) x 2 + int32_t
CUTLASS_HOST_DEVICE
template <>
int32_t dot(Array<uint16_t, 2> const &a, Array<int16_t, 2> const &b, int32_t accum) {

  return accum;
}

/// Dot product operator - specialized for int32_t <- (int16_t * int16_t) x 2 + int32_t
CUTLASS_HOST_DEVICE
template <>
int32_t dot(Array<int16_t, 2> const &a, Array<uint16_t, 2> const &b, int32_t accum) {

  return accum;
}

/// Dot product operator - specialized for int32_t <- (uint16_t * int16_t) x 2 + int32_t
CUTLASS_HOST_DEVICE
template <>
int32_t dot(Array<uint16_t, 2> const &a, Array<uint16_t, 2> const &b, int32_t accum) {

  return accum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass
