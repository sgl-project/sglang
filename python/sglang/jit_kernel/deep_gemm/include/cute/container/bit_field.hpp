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
/*! \file
    \brief Portable bit field that supports byte and word straddling that can
           be used in unions to bit-wise define parameters.
*/

#pragma once

#include <cute/config.hpp>                  // CUTE_HOST_DEVICE
#include <cute/numeric/numeric_types.hpp>   // uint_bit_t
#include <cute/util/type_traits.hpp>        // cute::is_same

namespace cute
{

class dummy_type {};

template <uint32_t BitStart, uint32_t NumBits, class OtherValueType = dummy_type>
struct bit_field
{
  static_assert(0 < NumBits && NumBits <= 64, "bit_fields with more than 64 bits are not supported.");

  // value_type: Use the smallest value type that fits NumBits
  static constexpr uint32_t value_type_bits = (NumBits <=  8) ?  8 :
                                              (NumBits <= 16) ? 16 :
                                              (NumBits <= 32) ? 32 : 64;
  using value_type   = cute::uint_bit_t<value_type_bits>;
  // storage_type: Use the smallest storage_type that avoids boundary crossing
  static constexpr uint32_t storage_type_bits = (BitStart /  8 == (BitStart + NumBits - 1) /  8) ?  8 :
                                                (BitStart / 16 == (BitStart + NumBits - 1) / 16) ? 16 :
                                                (BitStart / 32 == (BitStart + NumBits - 1) / 32) ? 32 : 64;
  using storage_type = cute::uint_bit_t<storage_type_bits>;

  static_assert(sizeof(OtherValueType) == sizeof(value_type) || is_same<OtherValueType,dummy_type>::value,
                "sizeof(OtherValueType) must be same as sizeof(value_type).");

  // Number of storage values needed: ceil_div(BitStart + NumBits, storage_type_bits)
  static constexpr uint32_t N      = (BitStart + NumBits + storage_type_bits - 1) / storage_type_bits;
  // Index of storage value for BitStart
  static constexpr uint32_t idx    = BitStart / storage_type_bits;
  // Bit of data_[idx] for BitStart
  static constexpr uint32_t bit_lo = BitStart % storage_type_bits;
  // Number of bits in data_[idx] used for NumBits if straddling, else 0
  static constexpr uint32_t bit_hi = (idx + 1 < N) ? (storage_type_bits - bit_lo) : 0;

public:

  // NumBits mask
  static constexpr value_type   mask    = value_type(uint64_t(-1) >> (64u - NumBits));
  // NumBits mask for BitStart
  static constexpr storage_type mask_lo = storage_type(mask) << bit_lo;
  // NumBits mask for leftover bits in data_[idx+1] if straddling, else 0
  static constexpr storage_type mask_hi = (idx + 1 < N) ? (storage_type(mask) >> bit_hi) : 0;

  storage_type data_[N];

  // Get value
  CUTE_HOST_DEVICE constexpr
  value_type get() const {
    storage_type result = (data_[idx] & mask_lo) >> bit_lo;
    if constexpr (bit_hi != 0) {
      result |= (data_[idx+1] & mask_hi) << bit_hi;
    }
    return static_cast<value_type>(result);
  }

  // Set value
  CUTE_HOST_DEVICE constexpr
  void set(value_type x) {
    storage_type item = static_cast<storage_type>(x & mask);
    data_[idx] = static_cast<storage_type>((data_[idx] & ~mask_lo) | (item << bit_lo));
    if constexpr (bit_hi != 0) {
      data_[idx+1] = static_cast<storage_type>((data_[idx+1] & ~mask_hi) | (item >> bit_hi));
    }
  }

  // Assign value
  CUTE_HOST_DEVICE constexpr
  bit_field& operator=(value_type x) {
    set(x);
    return *this;
  }

  // Cast to value
  CUTE_HOST_DEVICE constexpr
  operator value_type () const {
    return get();
  }

  // Assign OtherValueType
  CUTE_HOST_DEVICE constexpr
  bit_field& operator=(OtherValueType x) {
    return *this = *reinterpret_cast<value_type*>(&x);
  }

  // Cast to OtherValueType
  CUTE_HOST_DEVICE constexpr
  operator OtherValueType () const {
    value_type x = get();
    return *reinterpret_cast<OtherValueType*>(&x);
  }
};

} // end namespace cute
