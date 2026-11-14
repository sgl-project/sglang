/*
 * Copyright (c) 2020-2026, Moore Threads Technology Co., Ltd("Moore Threads").
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <limits>
#include <type_traits>

namespace musa::dnn {

// cutlass integer_subbyte class
template <int Bits, bool Signed = true>
struct integer_subbyte {
  using Storage = uint8_t;

  static_assert(Bits <= 8 * sizeof(Storage), "Require a subbyte of bits in integer_subbyte");

  using xint_t = typename std::conditional<Signed, int, unsigned>::type;

  static constexpr Storage bits_mask_ = Storage((1 << Bits) - 1);

  static constexpr Storage sign_mask_ = Storage((Signed ? 1 : 0) << (Bits - 1));

  Storage storage;

  __host__ __device__ constexpr integer_subbyte() {}

  __host__ __device__ constexpr integer_subbyte(int value)
      : storage(reinterpret_cast<Storage const&>(value) & bits_mask_) {}

  __host__ __device__ constexpr integer_subbyte(unsigned value)
      : storage(reinterpret_cast<Storage const&>(value) & bits_mask_) {}
};

}  // namespace musa::dnn
