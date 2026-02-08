/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Defines an unsigned 256b integer.
*/

#pragma once
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(cstdint)
#else
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#endif
#include "cutlass/uint128.h"

namespace cutlass {

///! Unsigned 256b integer type
struct alignas(32) uint256_t {
  /// Size of one part of the uint's storage in bits
  static constexpr int storage_bits_ = 128;

  struct hilo {
    uint128_t lo;
    uint128_t hi;
  };

  // Use a union to store either low and high parts.
  union {
    struct hilo hilo_;
  };

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  uint256_t() : hilo_{uint128_t{}, uint128_t{}} {}

  /// Constructor from uint128
  CUTLASS_HOST_DEVICE
  uint256_t(uint128_t lo_) : hilo_{lo_, uint128_t{}} {}

  /// Constructor from two 128b unsigned integers
  CUTLASS_HOST_DEVICE
  uint256_t(uint128_t lo_, uint128_t hi_) : hilo_{lo_, hi_} {}

  /// Lossily cast to uint128_t
  CUTLASS_HOST_DEVICE
  explicit operator uint128_t() const {
    return hilo_.lo;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
