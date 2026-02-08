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
    \brief Top-level include for all CUTLASS numeric types.
*/

#pragma once

#include "cutlass/cutlass.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the size of an element in bits
template <typename T>
struct sizeof_bits {
  static constexpr int value = int(sizeof(T) * 8);
};

template <typename T>
struct sizeof_bits<T const> : sizeof_bits<T> {};

template <typename T>
struct sizeof_bits<T volatile> : sizeof_bits<T> {};

template <typename T>
struct sizeof_bits<T const volatile> : sizeof_bits<T> {};

template <>
struct sizeof_bits<void> {
  static constexpr int value = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns the number of bytes required to hold a specified number of bits
template <class R = int, class T>
CUTLASS_HOST_DEVICE
constexpr
R
bits_to_bytes(T bits) {
  return (R(bits) + R(7)) / R(8);
}

/// Returns the number of bits required to hold a specified number of bytes
template <class R = int, class T>
CUTLASS_HOST_DEVICE
constexpr
R
bytes_to_bits(T bytes) {
  return R(bytes) * R(8);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
struct is_subbyte {
  static constexpr bool value = sizeof_bits<T>::value < 8;
};

template <class T>
struct is_subbyte<T const> : is_subbyte<T> {};

}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
