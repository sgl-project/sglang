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



/*!
  \file
  \brief Generic floating-point type for ExMy format
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_size.h"
#include "cutlass/platform/platform.h"

// #define CUTLASS_DEBUG_TRACE_LEVEL 2
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
 // Helper functions
namespace detail {

template <class Src, class Dst>
CUTLASS_HOST_DEVICE
Dst copy_bits(Src src)
{
  Dst dst;
  static_assert(sizeof(Src) <= sizeof(Dst), "Dst type should be at least the same size as Src type");
  static_assert(cutlass::platform::is_trivially_copyable<Dst>::value, "Dst type should be trivially copyable");
  static_assert(cutlass::platform::is_trivially_copyable<
    /*cutlass::platform::remove_cvref_t< */ Dst /* > */ >::value, "Dst type should be trivially copyable");
  memcpy(&dst, &src, sizeof(src));
  return dst;
}

enum class NanInfEncoding
{
  // IEEE-754 style NaN. Exponent bits are
  // all ones, and at least one bit of mantissa is one
  IEEE_754,
  // Canonical NaN. There is only one value representing NaN and
  // no Inf is defined.
  CANONICAL_ONLY,
  // No NaN or Inf encoded.
  NONE
};

enum class FpEncoding
{
  E11M52, // double
  E8M23,  // float
  E5M2,   // FP8
  E4M3,   // FP8
  UE4M3,  // FP8 
  UE8M0,  // FP8
  E3M2,   // FP6
  E2M3,   // FP6
  E2M1,   // FP4
};

//////

#if (CUTLASS_CXX17_OR_LATER)
template<uint32_t NumExpBits, uint32_t NumMantissaBits>
constexpr int exponent_bias_cxx17() {
  if CUTLASS_CONSTEXPR_IF_CXX17 (NumExpBits == 0) {
    static_assert(NumMantissaBits <= static_cast<uint32_t>(cutlass::platform::numeric_limits<int32_t>::max()));
    return -1 * static_cast<int>(NumMantissaBits);
  }
  else {
    return static_cast<int>((1 << (NumExpBits - 1))) - 1;
  }

  CUTLASS_GCC_UNREACHABLE;
}
#endif

namespace impl {
template<uint32_t NumExpBitsMinusOne>
constexpr int shift_num_bits_expression_cxx11() {
#if (CUTLASS_CXX17_OR_LATER)
  static_assert(NumExpBitsMinusOne <= 31u);
#endif
  return NumExpBitsMinusOne > 31u ? 31u : NumExpBitsMinusOne;
}

template<uint32_t NumExpBitsMinusOne>
constexpr int inner_shift_expression_cxx11() {
  return static_cast<int>((1u << shift_num_bits_expression_cxx11<NumExpBitsMinusOne>()) - 1u);
}

} // namespace impl

// C++11 equivalent of exponent_bias_cxx17()
template<uint32_t NumExpBits, uint32_t NumMantissaBits>
constexpr int exponent_bias_cxx11() {
#if (CUTLASS_CXX17_OR_LATER)
  return exponent_bias_cxx17<NumExpBits, NumMantissaBits>();
#else
  return (NumExpBits == 0) ?
    -1 * static_cast<int>(NumMantissaBits) : impl::inner_shift_expression_cxx11<NumExpBits - 1u>();
#endif
}

// C++11 equivalent of maximum_exponent_cxx17()
template<uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding>
constexpr int maximum_exponent_cxx11() {
  return
    ((NumExpBits == 0) ?
      (0 - exponent_bias_cxx11<NumExpBits, NumMantissaBits>()) :
      ((NaNEncoding == NanInfEncoding::IEEE_754) ?
        ((static_cast<int>((1 << NumExpBits)) - 2) - exponent_bias_cxx11<NumExpBits, NumMantissaBits>()) :
        ((NaNEncoding == NanInfEncoding::CANONICAL_ONLY) ?
          ((NumMantissaBits > 0) ?
            static_cast<int>((1 << NumExpBits)) - 1 - exponent_bias_cxx11<NumExpBits, NumMantissaBits>() :
            static_cast<int>((1 << NumExpBits)) - 2 - exponent_bias_cxx11<NumExpBits, NumMantissaBits>()
          ) :
          (static_cast<int>((1 << NumExpBits)) - 1 - exponent_bias_cxx11<NumExpBits, NumMantissaBits>())
        )
      )
    );
}

#if (CUTLASS_CXX17_OR_LATER)
template<uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding>
constexpr int maximum_exponent_cxx17() {
  constexpr int exp_bias = exponent_bias_cxx17<NumExpBits, NumMantissaBits>();
  if CUTLASS_CONSTEXPR_IF_CXX17 (NumExpBits == 0) {
    // If no exponent bits, return fixed hidden bias
    return 0 - exp_bias;
  }
  else {
    if CUTLASS_CONSTEXPR_IF_CXX17 (NaNEncoding == NanInfEncoding::IEEE_754) {
      // We have IEEE style NaN and infinity
      // All values when exp_bits = 1...1s are used.
      int max_exp_bits = static_cast<int>((1 << NumExpBits)) - 2;
      return max_exp_bits - exp_bias;
    }
    else {
      // There are no cases where we have Inf without IEEE_754_Nan

      // If we have a canonical NaN. Only exp=1..1 and mantissa=1..1
      // value has a special meaning. If we also have at least one mantissa
      // bit, then maximum exponent is 1...1 - exponent_bias
      if CUTLASS_CONSTEXPR_IF_CXX17 (NaNEncoding == NanInfEncoding::CANONICAL_ONLY) {
        if CUTLASS_CONSTEXPR_IF_CXX17 (NumMantissaBits > 0) {
          int max_exp_bits = static_cast<int>((1 << NumExpBits)) - 1;
          return max_exp_bits - exp_bias;
        }
        else { // no mantissa bits
          int max_exp_bits = static_cast<int>((1 << NumExpBits)) - 2;
          return max_exp_bits - exp_bias;
        }
      }
      // No NaNs or infs
      int max_exp_bits = static_cast<int>((1 << NumExpBits)) - 1;
      return max_exp_bits - exp_bias;
    }
  }

  CUTLASS_GCC_UNREACHABLE;
}
#endif

template<uint32_t NumExpBits, uint32_t NumMantissaBits>
constexpr int minimum_exponent_cxx11() {
  return
    ((NumExpBits == 0) ?
      0 - exponent_bias_cxx11<NumExpBits, NumMantissaBits>() :
      ((NumMantissaBits > 0) ?
        1 - exponent_bias_cxx11<NumExpBits, NumMantissaBits>() :
        0 - exponent_bias_cxx11<NumExpBits, NumMantissaBits>())
    );
}

#if (CUTLASS_CXX17_OR_LATER)
template<uint32_t NumExpBits, uint32_t NumMantissaBits>
constexpr int minimum_exponent_cxx17() {
  constexpr int exp_bias = exponent_bias_cxx17<NumExpBits, NumMantissaBits>();
  constexpr bool has_denorm = (NumMantissaBits > 0);
  if CUTLASS_CONSTEXPR_IF_CXX17 (NumExpBits == 0) {
    // If no exponent bits, return fixed hidden bias
    // Note that minimum and maximum exponents are the same.
    return 0 - exp_bias;
  }

  if CUTLASS_CONSTEXPR_IF_CXX17 (has_denorm) {
    // Exp = 0...0s is reserved for denorm values.
    return 1 - exp_bias;
  }
  return 0 - exp_bias;
}
#endif

template<class Storage, uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding>
constexpr Storage max_pos_denormal_value_cxx11() {
  static_assert(NumExpBits > 0 || NumMantissaBits > 0, "Both NumExpBits and NumMantissaBits can't be zero");
  return
    (!(NumMantissaBits > 0) ? Storage(0) : Storage((1ull << NumMantissaBits) - 1));
}

#if (CUTLASS_CXX17_OR_LATER)
template<class Storage, uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding>
constexpr Storage max_pos_denormal_value_cxx17() {
  static_assert(NumExpBits > 0 || NumMantissaBits > 0, "Both NumExpBits and NumMantissaBits can't be zero");
  constexpr bool has_denorm = (NumMantissaBits > 0);
  if CUTLASS_CONSTEXPR_IF_CXX17 (!has_denorm) {
    // If we don't have denormal values, return all 0s
    return Storage(0);
  }
  else {
    // Case: (NumExpBits > 0 && NumMantissaBits > 0) or (NumExpBits == 0 && NumMantissaBits > 0)
    return Storage((1ull << NumMantissaBits) - 1);
  }

  CUTLASS_GCC_UNREACHABLE;
}
#endif


template<class Storage, uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding>
constexpr Storage min_pos_denormal_value_cxx11() {
  return (!(NumMantissaBits > 0) ? Storage(0) : Storage(1));
}

#if (CUTLASS_CXX17_OR_LATER)
template<class Storage, uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding>
constexpr Storage min_pos_denormal_value_cxx17() {
  constexpr bool has_denorm = (NumMantissaBits > 0);
  if CUTLASS_CONSTEXPR_IF_CXX17 (!has_denorm) {
    // If we don't have denormal values, return all 0s
    return Storage(0);
  }
  // Case: (NumExpBits > 0 && NumMantissaBits > 0) or (NumExpBits == 0 && NumMantissaBits > 0)
  return Storage(1);
}
#endif

template<class Storage, uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding>
constexpr Storage max_pos_normal_value_cxx11() {
  return
    ((NumExpBits == 0) ?
      Storage(0) :
      ((NumMantissaBits == 0) ?
        0 :
        (((NaNEncoding == NanInfEncoding::IEEE_754 || NaNEncoding == NanInfEncoding::NONE) ?
          ((1ull << NumMantissaBits) - 1) :
          ((1ull << NumMantissaBits) - 2)))
      ) | (static_cast<Storage>(
            maximum_exponent_cxx11<NumExpBits, NumMantissaBits, NaNEncoding>() +
            exponent_bias_cxx11<NumExpBits, NumMantissaBits>()
          ) << NumMantissaBits)
    );
}

#if (CUTLASS_CXX17_OR_LATER)
template<class Storage, uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding>
constexpr Storage max_pos_normal_value_cxx17() {
  if CUTLASS_CONSTEXPR_IF_CXX17 (NumExpBits == 0) {
    // if there are no exponent bits, we don't have normal values.
    return Storage(0);
  }
  constexpr int exp_bias = exponent_bias_cxx17<NumExpBits, NumMantissaBits>();
  constexpr int max_exp = maximum_exponent_cxx17<NumExpBits, NumMantissaBits, NaNEncoding>();
  constexpr int exp = max_exp + exp_bias;

  // place the exponent
  Storage val = static_cast<Storage>(exp) << NumMantissaBits;
  // If there are no mantissa bits return the exponent
  if CUTLASS_CONSTEXPR_IF_CXX17 (NumMantissaBits == 0) {
    return val;
  }
  else {
    // If the NaN Inf encoding follows IEEE 754 or there is no (NaN and Inf) then mantissa can be all 1..1s
    if CUTLASS_CONSTEXPR_IF_CXX17 (NaNEncoding == NanInfEncoding::IEEE_754 ||
                  NaNEncoding == NanInfEncoding::NONE  ) {
      Storage mantissa = (1ull << NumMantissaBits) - 1;
      val |= mantissa;
    }
    else {
      // If we have a canonical NaN, then the exponent can be the maximum bit value
      // but mantissa=1..1s is reserved for NaN.
      Storage mantissa = (1ull << NumMantissaBits) - 2;
      val |= mantissa;
    }
    return val;
  }

  CUTLASS_GCC_UNREACHABLE;
}
#endif

template<class Storage, uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding>
constexpr Storage min_pos_normal_value_cxx11() {
  return
    ((NumExpBits == 0) ?
      Storage(0) :
      (Storage((NumMantissaBits > 0) ? 1 : 0) << NumMantissaBits)
    );
}

#if (CUTLASS_CXX17_OR_LATER)
template<class Storage, uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding>
constexpr Storage min_pos_normal_value_cxx17() {
  constexpr bool has_denorm = (NumMantissaBits > 0);

  if CUTLASS_CONSTEXPR_IF_CXX17 (NumExpBits == 0) {
    // if there are no exponent bits, we don't have normal values.
    return Storage(0);
  }
  Storage exp = 0;
  if CUTLASS_CONSTEXPR_IF_CXX17 (has_denorm) {
    exp = 1;
  }
  return static_cast<Storage>(exp << NumMantissaBits);
}
#endif

template<class Storage, uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding>
constexpr Storage max_value_cxx11() {
  return
    ((NumExpBits > 0) ?
      max_pos_normal_value_cxx11<Storage, NumExpBits, NumMantissaBits, NaNEncoding>() :
      max_pos_denormal_value_cxx11<Storage, NumExpBits, NumMantissaBits, NaNEncoding>()
    );
}

#if (CUTLASS_CXX17_OR_LATER)
template<class Storage, uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding>
constexpr Storage max_value_cxx17() {
  constexpr bool has_normal = (NumExpBits > 0);
  if CUTLASS_CONSTEXPR_IF_CXX17 (has_normal) {
    return max_pos_normal_value_cxx17<Storage, NumExpBits, NumMantissaBits, NaNEncoding>();
  }
  else {
    return max_pos_denormal_value_cxx17<Storage, NumExpBits, NumMantissaBits, NaNEncoding>();
  }

  CUTLASS_GCC_UNREACHABLE;
}
#endif

template<class Storage, uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding, bool IsSigned>
constexpr Storage min_value_cxx11() {
  return
    (IsSigned ?
      Storage(1ull << (NumExpBits + NumMantissaBits)) | max_value_cxx11<Storage, NumExpBits, NumMantissaBits, NaNEncoding>() :
      Storage(0)
    );
}

#if (CUTLASS_CXX17_OR_LATER)
template<class Storage, uint32_t NumExpBits, uint32_t NumMantissaBits, NanInfEncoding NaNEncoding, bool IsSigned>
constexpr Storage min_value_cxx17() {
  if (IsSigned) {
    return Storage(1ull << (NumExpBits + NumMantissaBits)) | max_value_cxx17<Storage, NumExpBits, NumMantissaBits, NaNEncoding>();
  }
  else { // Unsigned number
    return Storage(0);
  }

  CUTLASS_GCC_UNREACHABLE;
}
#endif

template <
    class StorageType,
    uint32_t NumBits, uint32_t NumExpBits, uint32_t NumMantissaBits,
    NanInfEncoding Nan = NanInfEncoding::IEEE_754, bool IsSigned = true>
struct FpBitRepresentation {
public:

  using Storage = StorageType;

#if (CUTLASS_CXX17_OR_LATER)
  static_assert(cutlass::platform::is_unsigned_v<Storage>, "Use an unsigned integer for StorageType");
#endif
  static constexpr bool IS_SIGNED = IsSigned;
  // Canonical NaN is always represented as exponent=11...11 and mantissa=11...11, if it exists
  static constexpr NanInfEncoding NAN_TYPE = Nan;
  // Inf is always represented as exponent=11...11 and mantissa=00...00, if it exists
  static constexpr bool HAS_INF = (NAN_TYPE == NanInfEncoding::IEEE_754);
  static constexpr bool HAS_NAN = (NAN_TYPE != NanInfEncoding::NONE);

  static constexpr bool HAS_DENORM = (NumMantissaBits > 0);
  static constexpr bool HAS_NORMAL = !HAS_DENORM;

  static constexpr uint32_t NUM_BITS = NumBits;
  static constexpr uint32_t NUM_EXPONENT_BITS = NumExpBits;
  static constexpr uint32_t NUM_MANTISSA_BITS = NumMantissaBits;
  static_assert(NUM_BITS >= (NUM_EXPONENT_BITS + NUM_MANTISSA_BITS + uint32_t(IS_SIGNED)), "Number of bits do not match");

  static constexpr Storage ONE = Storage(1);
  static constexpr Storage ZERO = Storage(0);

  // Note: Don't rely on operator precedence. Use parenthesis.
  static constexpr Storage EXPONENT_MASK = (Storage(1) << Storage(NUM_EXPONENT_BITS)) - ONE;
  static constexpr Storage MANTISSA_MASK = (Storage(1) << Storage(NUM_MANTISSA_BITS)) - ONE;
  static constexpr Storage EXPONENT_SHIFT = Storage(NUM_MANTISSA_BITS);
  static constexpr Storage SIGN_SHIFT = (IS_SIGNED) ? Storage(NUM_MANTISSA_BITS + NUM_EXPONENT_BITS) : Storage(0);

  // Note: All biased/real exponent calculation are done with signed ints
  // Use unsigned to represent data not exponent.
  static constexpr int EXP_BIAS = detail::exponent_bias_cxx11<NUM_EXPONENT_BITS, NUM_MANTISSA_BITS>();
  static constexpr int MAX_EXP = detail::maximum_exponent_cxx11<NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>();
  static constexpr int MIN_EXP = detail::minimum_exponent_cxx11<NUM_EXPONENT_BITS, NUM_MANTISSA_BITS>();

  // Floating-point Limits
  static constexpr Storage MAX_POS_NORMAL_VAL = detail::max_pos_normal_value_cxx11<Storage, NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>();
  static constexpr Storage MAX_POS_DENORMAL_VAL = detail::max_pos_denormal_value_cxx11<Storage, NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>();
  static constexpr Storage MIN_POS_NORMAL_VAL = detail::min_pos_normal_value_cxx11<Storage, NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>();
  static constexpr Storage MIN_POS_DENORMAL_VAL = detail::min_pos_denormal_value_cxx11<Storage, NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>();

  static constexpr Storage MAX_VALUE = max_value_cxx11<Storage, NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>();
  static constexpr Storage MIN_VALUE = min_value_cxx11<Storage, NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE, IS_SIGNED>();

  //
  // C++17 Verification
  //
#if (CUTLASS_CXX17_OR_LATER)
  static_assert(EXP_BIAS == detail::exponent_bias_cxx17<NUM_EXPONENT_BITS, NUM_MANTISSA_BITS>(),                "Error");
  static_assert(MAX_EXP  == detail::maximum_exponent_cxx17<NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>(),   "Error");
  static_assert(MIN_EXP  == detail::minimum_exponent_cxx17<NUM_EXPONENT_BITS, NUM_MANTISSA_BITS>(),             "Error");

  static_assert(MAX_POS_NORMAL_VAL   == detail::max_pos_normal_value_cxx17<Storage, NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>(), "Error");
  static_assert(MAX_POS_DENORMAL_VAL == detail::max_pos_denormal_value_cxx17<Storage, NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>(), "Error");
  static_assert(MIN_POS_NORMAL_VAL   == detail::min_pos_normal_value_cxx17<Storage, NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>(), "Error");
  static_assert(MIN_POS_DENORMAL_VAL == detail::min_pos_denormal_value_cxx17<Storage, NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>(), "Error");
  static_assert(MAX_VALUE            == max_value_cxx17<Storage, NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>(), "Error");
  static_assert(MIN_VALUE            == min_value_cxx17<Storage, NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE, IS_SIGNED>(), "Error");
#endif

  // If we don't have INF defined, set the largest number. Gives us .satfinite behavior.
  static constexpr Storage INF_MASK = (HAS_INF) ?
      (Storage(EXPONENT_MASK) << Storage(NUM_MANTISSA_BITS)) : MAX_VALUE;
  static constexpr Storage NAN_MASK = (Storage(EXPONENT_MASK) << Storage(NUM_MANTISSA_BITS)) | MANTISSA_MASK;

  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 bool is_inf(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (!HAS_INF) {
      return false;
    }
    bool exp_all_ones = (exponent_bits(flt) ^ EXPONENT_MASK) == 0;
    bool mantissa_all_zeros = mantissa_bits(flt) == 0;
    return exp_all_ones && mantissa_all_zeros;
  }

  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 bool is_canonical_nan(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (NAN_TYPE == NanInfEncoding::NONE) {
      return false;
    }
    bool exp_all_ones = (exponent_bits(flt) ^ EXPONENT_MASK) == ZERO;
    bool mantissa_all_ones = (mantissa_bits(flt) ^ MANTISSA_MASK) == ZERO;
    return exp_all_ones && mantissa_all_ones;
  }

  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 bool is_nan(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (NAN_TYPE == NanInfEncoding::NONE) {
      return false;
    }

    if CUTLASS_CONSTEXPR_IF_CXX17 (NAN_TYPE == NanInfEncoding::CANONICAL_ONLY) {
      return is_canonical_nan(flt);
    }

    bool exp_all_ones = (exponent_bits(flt) ^ EXPONENT_MASK) == ZERO;
    bool mantissa_has_ones = mantissa_bits(flt) > ZERO;
    return exp_all_ones && mantissa_has_ones;
  }

  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 bool is_denorm(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (!HAS_DENORM) {
      return false;
    }
    else if (exponent_bits(flt) == ZERO) {
      // Exponent bits are all 0s
      return true;
    }
    return false;
  }

  template<typename T = Storage>
  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 T sign_bit(T flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (!IS_SIGNED) {
      return T(0);
    }
    return static_cast<T>(flt >> T(SIGN_SHIFT));
  }

  template<typename T = Storage>
  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 T set_sign_bit(T flt, T sign) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (!IS_SIGNED) {
      return flt;
    }
    return static_cast<T>(flt | (sign << T(SIGN_SHIFT)));
  }

  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 Storage exponent_bits(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (NUM_EXPONENT_BITS == ZERO) {
      return ZERO;
    }
    return (flt >> (NUM_MANTISSA_BITS)) & EXPONENT_MASK;
  }

  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 int exponent(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (NUM_EXPONENT_BITS == ZERO) {
      return -int(EXP_BIAS);
    }

    if (HAS_DENORM && (exponent_bits(flt) == ZERO)) {
      return 1 - int(EXP_BIAS);
    }

    return int(flt >> (NUM_MANTISSA_BITS) & EXPONENT_MASK) - int(EXP_BIAS);
  }

  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 Storage mantissa_bits(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (NUM_MANTISSA_BITS == ZERO) {
      return ZERO;
    }
    return (flt & MANTISSA_MASK);
  }

  template <class FpType>
  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 Storage to_bits(FpType flt) {
    return copy_bits<FpType, Storage>(flt);
  }

  template <class DstFpBits>
  CUTLASS_HOST_DEVICE static typename DstFpBits::Storage convert_to(
      Storage src_val,
      DstFpBits dst_encoding) {
    return convert(FpBitRepresentation{}, src_val, dst_encoding);
  }

  template <class SrcFpBits>
  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 Storage convert_from(
      typename SrcFpBits::Storage src_val,
      SrcFpBits src_encoding) {
    return convert(src_encoding, src_val, FpBitRepresentation{});
  }

private:

  template<typename T = Storage>
  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 T make_fp_from_bits(T sign, T exp, T mantissa) {
    T fp_bits = T(ZERO);
    CUTLASS_UNUSED(sign);
    if CUTLASS_CONSTEXPR_IF_CXX17 (IS_SIGNED) {
      fp_bits = sign << SIGN_SHIFT;
    }
    fp_bits |= (exp << T(NUM_MANTISSA_BITS));
    fp_bits |= (mantissa);
    return fp_bits;
  }

  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 Storage nan_with_sign(Storage sign) {
    Storage fp_bits = NAN_MASK;
    return set_sign_bit(fp_bits, sign);
  }

  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 Storage inf_with_sign(Storage sign) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (HAS_INF) {
      Storage fp_bits = INF_MASK;
      return set_sign_bit(fp_bits, sign);
    }
    else {
      // If INF is not defined assume satfinite behavior
      return (sign == ZERO) ? MAX_VALUE : MIN_VALUE;
    }

    CUTLASS_GCC_UNREACHABLE;
  }

  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 Storage significand(Storage flt) {
    if (is_denorm(flt)) {
      return mantissa_bits(flt);
    }
    else {
      return (ONE << Storage(NUM_MANTISSA_BITS)) | mantissa_bits(flt);
    }

    CUTLASS_GCC_UNREACHABLE;
  }

  template<typename T>
  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 T significand_hidden_bits(T significand) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (NUM_MANTISSA_BITS == 0) {
      return T(1);
    }
    return ((T(0b11) << T(NUM_MANTISSA_BITS)) & significand) >> T(NUM_MANTISSA_BITS);
  }

  // Current assumption round to nearest even
  template<class T>
  CUTLASS_HOST_DEVICE
  static CUTLASS_CONSTEXPR_IF_CXX17 T round_significand(T src, int shift_amount) {
    T dst_mantissa = src;
    // If the shift amount is positive, we are shifting left
    // Type with less mantissa bits is rounded to a type with more
    // mantissa bits.
    if (shift_amount > 0) {
      dst_mantissa = (dst_mantissa << (shift_amount));
    }
    else {
      // There are fewer mantissa bits in the target type
      // we need to round the destination number up for all
      // lower precision bits removed.
      // We assume round-to-nearest-even here.
      int pos_shift_amount = -shift_amount;

      // Too large shift return all zeros to prevent undefined behavior for shift.
      if (pos_shift_amount >= static_cast<int>(sizeof(T) * 8)) {
        return T(0);
      }

      T guard_bit_mask = (T(1) << T(pos_shift_amount));            // Last bit to remain in mantissa
      T sticky_mask    = (T(1) << T(pos_shift_amount - 1)) - T(1); // Remaining bits
      T round_bit_mask = (T(1) << T(pos_shift_amount - 1));        // First bit removed from mantissa

      bool sticky_bit = (src & sticky_mask) >= T(1);                      // ORing all sticky bits
      bool round_bit = (src & round_bit_mask) >= T(1);
      bool guard_bit = (src & guard_bit_mask) >= T(1);

      // Shift mantissa bits to right to remove lowest precision bits
      dst_mantissa = dst_mantissa >> pos_shift_amount;

      if ((sticky_bit && round_bit) || (guard_bit && round_bit && !sticky_bit)) {
        dst_mantissa += 1;
      }
    }
    return dst_mantissa;
  }

  template <class SrcFpBits, class DstFpBits>
  CUTLASS_HOST_DEVICE
  static typename DstFpBits::Storage convert(
      SrcFpBits src_encoding,
      typename SrcFpBits::Storage src_val,
      DstFpBits dst_encoding) {

    using SrcT = typename SrcFpBits::Storage;
    using DstT = typename DstFpBits::Storage;
    using LargeStorage = typename cutlass::platform::conditional<(sizeof(SrcT) > sizeof(DstT)), SrcT, DstT>::type;

    LargeStorage src_sign_bit = src_encoding.sign_bit(src_val);

    // If the source is NaN, set the destination to NaN carrying the sign bit
    if (src_encoding.is_nan(src_val)) {
      return dst_encoding.nan_with_sign(DstT(src_sign_bit));
    }
    // If the source is INF, set the destination to INF carrying the sign bit
    else if (src_encoding.is_inf(src_val)) {
      return dst_encoding.set_sign_bit(DstFpBits::INF_MASK, DstT(src_sign_bit));
    }
    // Number is not NaN or INF: Zero and others

    LargeStorage src_exp_bits = src_encoding.exponent_bits(src_val);
    LargeStorage src_significand = src_encoding.significand(src_val);
    int src_exp = src_encoding.exponent(src_val);

    // The source value is 0. Return a signed 0.
    if (src_exp_bits == LargeStorage(0) && src_significand == LargeStorage(0)) {
      return dst_encoding.set_sign_bit(DstT(0), DstT(src_sign_bit));
    }

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    printf("(1) src_sign: %llu src_exp_bits %llx src_exp %d src_significand %llx\n",
      static_cast<unsigned long long>(src_sign_bit), static_cast<unsigned long long>(src_exp_bits), src_exp, static_cast<unsigned long long>(src_significand));
#endif
    // Normalize the number: Left shift the significand bits until hidden "1" appears.
    // Only needed if the src value is denormal.
    // Conditions:
    //  If the exponent is 0, then the significand can't be 0 (src_val==0 case handled above):
    //    there is at least one "1" bit in the significand. Loop executes.
    //  If the exponent is not 0, then the number is normal:
    //    significand has hidden bit set. Loop doesn't execute.
    // Assumption: Zero is always defined for the floating point types and detected above

    while (src_encoding.significand_hidden_bits(src_significand) == LargeStorage(0)) {
      src_significand <<= LargeStorage(1);
      src_exp--;
    }

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    printf("(2) src_sign: %llu src_exp_bits %llx src_exp %d src_significand %llx\n",
      static_cast<unsigned long long>(src_sign_bit), static_cast<unsigned long long>(src_exp_bits), src_exp, static_cast<unsigned long long>(src_significand));
#endif
    // The exponent exceeds DstFormat's exponent capacity
    // Return positive/negative infinity.
    // If no INF is defined, return positive/negative largest value.
    if (src_exp > DstFpBits::MAX_EXP) {
      return dst_encoding.set_sign_bit(DstFpBits::INF_MASK, DstT(src_sign_bit));
    }
    else if (src_exp <= DstFpBits::MAX_EXP && src_exp >= DstFpBits::MIN_EXP) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      printf("(3) Exp match: src_sign: %d src_exp_bits: %x src_exp: %d src_significand: %x\n",
        src_sign_bit, src_exp_bits, src_exp, src_significand);
#endif

      int shift_amount = int(DstFpBits::NUM_MANTISSA_BITS) - int(SrcFpBits::NUM_MANTISSA_BITS);
      int dst_exponent = src_exp + DstFpBits::EXP_BIAS;
      LargeStorage dst_mantissa = src_significand;

      // if we have an M0 case, the floating point number is always denormal.
      // Therefore, if exponents are equal, we need to check whether it is inf
      if (DstFpBits::NUM_EXPONENT_BITS == 0) {
        if (dst_mantissa > DstFpBits::INF_MASK) {
          return dst_encoding.inf_with_sign(DstT(src_sign_bit));
        }
      }

      // Round to nearest even
      dst_mantissa = round_significand(dst_mantissa, shift_amount);

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      printf("(4) after rounding src_sign: %d dst_exponent: %d dst_mantissa: %x\n",
        src_sign_bit, dst_exponent, dst_mantissa);
#endif

      if (dst_encoding.significand_hidden_bits(dst_mantissa) > 0b1) {
        // Significant became larger than 01.X...X. Divide significand by 2 and multiply exp by 2
        while (dst_exponent < (DstFpBits::MAX_EXP+DstFpBits::EXP_BIAS) &&
               dst_encoding.significand_hidden_bits(dst_mantissa) > LargeStorage(0b1)) {
          dst_mantissa >>= LargeStorage(1);
          dst_exponent++;
        }

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        printf("(5) after rounding  max_exp: %d src_sign: %d dst_exponent: %d dst_mantissa: %x\n",
          DstFpBits::MAX_EXP,src_sign_bit, dst_exponent, dst_mantissa);
#endif

        if (dst_encoding.significand_hidden_bits(dst_mantissa) > LargeStorage(0b1)) {
          return dst_encoding.set_sign_bit(DstFpBits::INF_MASK, DstT(src_sign_bit));
        }
      }

      dst_mantissa = dst_mantissa & DstFpBits::MANTISSA_MASK;
      static_assert(sizeof(LargeStorage) >= sizeof(decltype(dst_exponent)),
        "sizeof(LargeStorage) must be greater than or equal to sizeof(decltype(dst_exponent))");
      LargeStorage dst_exponent_bits = static_cast<LargeStorage>(dst_exponent);

      DstT final_val = static_cast<DstT>(dst_encoding.template make_fp_from_bits<LargeStorage>(src_sign_bit, dst_exponent_bits, dst_mantissa));

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      printf("(6) Final Value src_sign: %d dst_exp_bits: %x dst_mantissa: %x\n",
        src_sign_bit, dst_exponent_bits, dst_mantissa);
#endif

      if (DstFpBits::is_nan(final_val)) {
        // This NAN is generated when:
        //  Src is not an Nan
        //  the exp of Src == the max_exp of Dst.
        //  The mantissa becomes all-1s after rounding.
        // Return max value of Dst (not NAN) as it just couldn't be represented in the range of Dst.
        return dst_encoding.set_sign_bit(DstFpBits::INF_MASK, DstT(src_sign_bit));
      }
      else {
        return final_val;
      }
    }
    else {
      // Result is denormal
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      printf("(7) Denormal case src_sign: %d src_exp: %d src_significand: %x MIN_EXP: %d\n",
        src_sign_bit, src_exp, src_significand, DstFpBits::MIN_EXP);
#endif

      int exp_diff = src_exp - DstFpBits::MIN_EXP;
      int shift_amount = int(DstFpBits::NUM_MANTISSA_BITS) - int(SrcFpBits::NUM_MANTISSA_BITS);
      shift_amount += exp_diff;
      LargeStorage dst_mantissa = src_significand;
      dst_mantissa = round_significand(dst_mantissa, shift_amount);

      if (dst_encoding.significand_hidden_bits(dst_mantissa) >= LargeStorage(0b1)) {
        if CUTLASS_CONSTEXPR_IF_CXX17 (DstFpBits::NUM_EXPONENT_BITS == 0) {
          return dst_encoding.inf_with_sign(DstT(src_sign_bit));
        }
        else {
          LargeStorage dst_exp_bits = 1;
          dst_mantissa &= DstFpBits::MANTISSA_MASK;
          DstT final_val = static_cast<DstT>(dst_encoding.template make_fp_from_bits<LargeStorage>(src_sign_bit, dst_exp_bits, dst_mantissa));
          return final_val;
        }
      }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      printf("(7.1) Denormal case exp_diff: %d shift_amount: %d dst_mantissa %d\n", exp_diff, shift_amount, dst_mantissa);
#endif
      dst_mantissa &= DstFpBits::MANTISSA_MASK;

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      printf("(8) Final Value src_sign: %d src_exp: %d dst_mantissa: %x\n",
        src_sign_bit, src_exp, dst_mantissa);
#endif

      DstT final_val = static_cast<DstT>(dst_encoding.template make_fp_from_bits<LargeStorage>(src_sign_bit, LargeStorage(0), dst_mantissa));
      return final_val;
    }

    return DstT(0);
  }

  template <class StorageType_, uint32_t NumBits_, uint32_t NumExpBits_,
            uint32_t NumMantissaBits_, NanInfEncoding Nan_, bool IsSigned_>
            friend struct FpBitRepresentation;
};

#if (CUTLASS_CXX17_OR_LATER)

template<FpEncoding FpExMyCode>
CUTLASS_CONSTEXPR_IF_CXX17 auto fp_encoding_selector() {
  if CUTLASS_CONSTEXPR_IF_CXX17      (FpExMyCode == FpEncoding::E11M52) { // double
    return cutlass::detail::FpBitRepresentation<uint64_t, 64, 11, 52, cutlass::detail::NanInfEncoding::IEEE_754>{};
  }
  else if CUTLASS_CONSTEXPR_IF_CXX17 (FpExMyCode == FpEncoding::E8M23)  { // float
    return cutlass::detail::FpBitRepresentation<uint32_t, 32, 8, 23, cutlass::detail::NanInfEncoding::IEEE_754>{};
  }
  else if CUTLASS_CONSTEXPR_IF_CXX17 (FpExMyCode == FpEncoding::E5M2)   {   // FP8
    return cutlass::detail::FpBitRepresentation<uint8_t, 8, 5, 2, cutlass::detail::NanInfEncoding::IEEE_754>{};
  }
  else if CUTLASS_CONSTEXPR_IF_CXX17 (FpExMyCode == FpEncoding::E4M3)   {   // FP8
    return cutlass::detail::FpBitRepresentation<uint8_t, 8, 4, 3, cutlass::detail::NanInfEncoding::CANONICAL_ONLY>{};
  }
  
  else if CUTLASS_CONSTEXPR_IF_CXX17 (FpExMyCode == FpEncoding::UE4M3)   {   // FP8
    return cutlass::detail::FpBitRepresentation<uint8_t, 8, 4, 3, cutlass::detail::NanInfEncoding::CANONICAL_ONLY, false>{};
  }
  
  else if CUTLASS_CONSTEXPR_IF_CXX17 (FpExMyCode == FpEncoding::UE8M0)   {   // FP8
    return cutlass::detail::FpBitRepresentation<uint8_t, 8, 8, 0, cutlass::detail::NanInfEncoding::CANONICAL_ONLY, false>{};
  }
  else if CUTLASS_CONSTEXPR_IF_CXX17 (FpExMyCode == FpEncoding::E3M2)   {   // FP6
    return cutlass::detail::FpBitRepresentation<uint8_t, 6, 3, 2, cutlass::detail::NanInfEncoding::NONE>{};
  }
  else if CUTLASS_CONSTEXPR_IF_CXX17 (FpExMyCode == FpEncoding::E2M3)   {   // FP6
    return cutlass::detail::FpBitRepresentation<uint8_t, 6, 2, 3, cutlass::detail::NanInfEncoding::NONE>{};
  }
  else if CUTLASS_CONSTEXPR_IF_CXX17 (FpExMyCode == FpEncoding::E2M1)   {   // FP4
    return cutlass::detail::FpBitRepresentation<uint8_t, 4, 2, 1, cutlass::detail::NanInfEncoding::NONE>{};
  }
  CUTLASS_GCC_UNREACHABLE;
}

#else
//
// Definitions for floating point encodings.
//

template <FpEncoding FpExMyCode> struct FpEncodingSelector {
  using type = void;
};

template <> struct FpEncodingSelector<FpEncoding::E11M52> {
  using type = cutlass::detail::FpBitRepresentation<uint64_t, 64, 11, 52, cutlass::detail::NanInfEncoding::IEEE_754>;
};

template <> struct FpEncodingSelector<FpEncoding::E8M23> {
  using type = cutlass::detail::FpBitRepresentation<uint32_t, 32, 8, 23, cutlass::detail::NanInfEncoding::IEEE_754>;
};
template <> struct FpEncodingSelector<FpEncoding::E5M2> {
  using type = cutlass::detail::FpBitRepresentation<uint8_t, 8, 5, 2, cutlass::detail::NanInfEncoding::IEEE_754>;
};

template <> struct FpEncodingSelector<FpEncoding::E4M3> {
  using type = cutlass::detail::FpBitRepresentation<uint8_t, 8, 4, 3, cutlass::detail::NanInfEncoding::CANONICAL_ONLY>;
};

template <> struct FpEncodingSelector<FpEncoding::UE4M3> {
  using type = cutlass::detail::FpBitRepresentation<uint8_t, 8, 4, 3, cutlass::detail::NanInfEncoding::CANONICAL_ONLY, false>;
};

template <> struct FpEncodingSelector<FpEncoding::UE8M0> {
  using type = cutlass::detail::FpBitRepresentation<uint8_t, 8, 8, 0, cutlass::detail::NanInfEncoding::CANONICAL_ONLY, false>;
};

template <> struct FpEncodingSelector<FpEncoding::E3M2> {
  using type = cutlass::detail::FpBitRepresentation<uint8_t, 6, 3, 2, cutlass::detail::NanInfEncoding::NONE>;
};

template <> struct FpEncodingSelector<FpEncoding::E2M3> {
  using type = cutlass::detail::FpBitRepresentation<uint8_t, 6, 2, 3, cutlass::detail::NanInfEncoding::NONE>;
};

template <> struct FpEncodingSelector<FpEncoding::E2M1> {
  using type = cutlass::detail::FpBitRepresentation<uint8_t, 4, 2, 1, cutlass::detail::NanInfEncoding::NONE>;
};
#endif

} // namespace detail

template <detail::FpEncoding T, class Derived>
struct float_exmy_base
{

  static constexpr detail::FpEncoding Encoding = T;
  using BitRepresentation =
    #if (CUTLASS_CXX17_OR_LATER)
      decltype(detail::fp_encoding_selector<T>())
    #else
      typename detail::FpEncodingSelector<T>::type
    #endif
      ;

  using FP32BitRepresentation =
    #if (CUTLASS_CXX17_OR_LATER)
      decltype(cutlass::detail::fp_encoding_selector<cutlass::detail::FpEncoding::E8M23>())
    #else
      typename detail::FpEncodingSelector<cutlass::detail::FpEncoding::E8M23>::type
    #endif
      ;

  using Storage = typename BitRepresentation::Storage;

  //
  // Data members
  //

  /// Data container
  Storage storage;

  /// Ctors.
  float_exmy_base() = default;

  CUTLASS_HOST_DEVICE
  float_exmy_base(Storage s) : storage(s) {
  }

  /// Is finite implementation
  CUTLASS_HOST_DEVICE
  static bool isfinite(float_exmy_base flt) {
    return !BitRepresentation::is_inf(flt.storage);
  }

  /// Is NaN implementation
  CUTLASS_HOST_DEVICE
  static bool isnan(float_exmy_base flt) {
    return BitRepresentation::is_nan(flt.storage);
  }

  /// Is infinite implementation
  CUTLASS_HOST_DEVICE
  static bool isinf(float_exmy_base flt) {
    return BitRepresentation::is_inf(flt.storage);
  }

  /// Is infinite implementation
  CUTLASS_HOST_DEVICE
  static bool isnormal(float_exmy_base flt) {
    return !BitRepresentation::is_denorm(flt.storage);
  }

  CUTLASS_HOST_DEVICE
  static float_exmy_base<T, Derived> bitcast(Storage x) {
    float_exmy_base f;
    f.storage = x;
    return f;
  }

  CUTLASS_HOST_DEVICE
  float_exmy_base convert_from_float(float const &flt) const {
    FP32BitRepresentation::Storage fp32_bits = FP32BitRepresentation::to_bits(flt);
    float_exmy_base float_exmy;
    float_exmy.storage = BitRepresentation::convert_from(fp32_bits, FP32BitRepresentation{});
    return float_exmy;
  }

  CUTLASS_HOST_DEVICE
  float convert_to_float(float_exmy_base<T, Derived> const &x) const {
    FP32BitRepresentation::Storage fp32_bits;
    fp32_bits = BitRepresentation::convert_to(x.storage, FP32BitRepresentation{});
    return detail::copy_bits<FP32BitRepresentation::Storage, float>(fp32_bits);
  }

  // Note: Only consider float/int conversions in this Base class
  // Types inheriting from this class should define their own constructors and
  // specialized type conversions

  /// Floating point conversion
  CUTLASS_HOST_DEVICE
  explicit float_exmy_base<T, Derived>(float x) {
    storage = static_cast<Derived*>(this)->convert_from_float(x).storage;
  }

  // Integer conversion
  CUTLASS_HOST_DEVICE
  explicit float_exmy_base<T, Derived>(int x) {
    storage = static_cast<Derived*>(this)->convert_from_float(float(x)).storage;
  }

  CUTLASS_HOST_DEVICE
  explicit float_exmy_base<T, Derived>(unsigned x) {
    storage = static_cast<Derived*>(this)->convert_from_float(float(x)).storage;
  }

  /// Converts to float
  CUTLASS_HOST_DEVICE
  operator float() const {
    return static_cast<const Derived*>(this)->convert_to_float(*this);
  }

  /// Converts to int
  CUTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(static_cast<const Derived*>(this)->convert_to_float(*this));
  }

  /// Accesses raw internal state
  CUTLASS_HOST_DEVICE
  Storage &raw() {
    return storage;
  }

  /// Accesses raw internal state
  CUTLASS_HOST_DEVICE
  Storage raw() const {
    return storage;
  }

  /// Returns the sign bit
  CUTLASS_HOST_DEVICE
  bool signbit() const {
    return bool(BitRepresentation::sign_bit(storage));
  }

  /// Returns the biased exponent
  CUTLASS_HOST_DEVICE
  int exponent_biased() const {
    return int(BitRepresentation::exponent_bits(storage));
  }

  /// Returns the unbiased exponent
  CUTLASS_HOST_DEVICE
  int exponent() const {
    return int(BitRepresentation::exponent(storage));
  }

  /// Returns the mantissa
  CUTLASS_HOST_DEVICE
  int mantissa() const {
    return int(BitRepresentation::mantissa_bits(storage));
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Arithmetic operators
  //
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // Note: Almost all data types cast to float then do the arithmetic operations
  // Types inheriting from this class can overload them if specialized instructions are available
  // in HW (e.g. half_t)


  CUTLASS_HOST_DEVICE
  friend bool operator==(float_exmy_base const &lhs, float_exmy_base const &rhs) {
    return float(lhs) == float(rhs);
  }

  CUTLASS_HOST_DEVICE
  friend bool operator!=(float_exmy_base const &lhs, float_exmy_base const &rhs) {
    return float(lhs) != float(rhs);
  }

  CUTLASS_HOST_DEVICE
  friend bool operator<(float_exmy_base const &lhs, float_exmy_base const &rhs) {
    return float(lhs) < float(rhs);
  }

  CUTLASS_HOST_DEVICE
  friend bool operator<=(float_exmy_base const &lhs, float_exmy_base const &rhs) {
    return float(lhs) <= float(rhs);
  }

  CUTLASS_HOST_DEVICE
  friend bool operator>(float_exmy_base const &lhs, float_exmy_base const &rhs) {
    return float(lhs) > float(rhs);
  }

  CUTLASS_HOST_DEVICE
  friend bool operator>=(float_exmy_base const &lhs, float_exmy_base const &rhs) {
    return float(lhs) >= float(rhs);
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base operator+(float_exmy_base const &lhs, float_exmy_base const &rhs) {
    return float_exmy_base(float(lhs) + float(rhs));
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base operator-(float_exmy_base const &lhs) {
    return float_exmy_base(-float(lhs));
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base operator-(float_exmy_base const &lhs, float_exmy_base const &rhs) {
    return float_exmy_base(float(lhs) - float(rhs));
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base operator*(float_exmy_base const &lhs, float_exmy_base const &rhs) {
    return float_exmy_base(float(lhs) * float(rhs));
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base operator/(float_exmy_base const &lhs, float_exmy_base const &rhs) {
    return float_exmy_base(float(lhs) / float(rhs));
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base &operator+=(float_exmy_base &lhs, float_exmy_base const &rhs) {
    lhs = float_exmy_base(float(lhs) + float(rhs));
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base &operator-=(float_exmy_base &lhs, float_exmy_base const &rhs) {
    lhs = float_exmy_base(float(lhs) - float(rhs));
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base &operator*=(float_exmy_base &lhs, float_exmy_base const &rhs) {
    lhs = float_exmy_base(float(lhs) * float(rhs));
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base &operator/=(float_exmy_base &lhs, float_exmy_base const &rhs) {
    lhs = float_exmy_base(float(lhs) / float(rhs));
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base &operator++(float_exmy_base &lhs) {
    float tmp(lhs);
    ++tmp;
    lhs = float_exmy_base(tmp);
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base &operator--(float_exmy_base &lhs) {
    float tmp(lhs);
    --tmp;
    lhs = float_exmy_base(tmp);
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base operator++(float_exmy_base &lhs, int) {
    float_exmy_base ret(lhs);
    float tmp(lhs);
    tmp++;
    lhs = float_exmy_base(tmp);
    return ret;
  }

  CUTLASS_HOST_DEVICE
  friend float_exmy_base operator--(float_exmy_base &lhs, int) {
    float_exmy_base ret(lhs);
    float tmp(lhs);
    tmp--;
    lhs = float_exmy_base(tmp);
    return ret;
  }

};

template <detail::FpEncoding T, class Derived>
CUTLASS_HOST_DEVICE
cutlass::float_exmy_base<T, Derived> abs(cutlass::float_exmy_base<T, Derived> const& h) {
  using BitRepresentation = typename cutlass::float_exmy_base<T, Derived>::BitRepresentation;
  using Storage = typename cutlass::float_exmy_base<T, Derived>::Storage;
  return BitRepresentation::IS_SIGNED ?
      cutlass::float_exmy_base<T, Derived>(Storage(h.raw() & Storage((1<<BitRepresentation::SIGN_SHIFT) - 1))) :
      cutlass::float_exmy_base<T, Derived>(h.raw());
}
} // namespace cutlass
