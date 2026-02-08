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

#include <cute/config.hpp>

#include <cute/arch/util.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute
{

//
// Direct Copy for any specific types
//

template <class S, class D = S>
struct UniversalCopy
{
  using SRegisters = S[1];
  using DRegisters = D[1];

  // Sanity
  static_assert(sizeof_bits_v<S> >= 8);
  static_assert(sizeof_bits_v<D> >= 8);

  CUTE_HOST_DEVICE static constexpr void
  copy(S const& src,
       D      & dst)
  {
    dst = src;
  }
};

//
// Placeholder for the copy algorithm's stronger auto-vectorizing behavior
//   that assumes alignment of pointers and dynamic layouts up to MaxVecBits
//

template <int MaxVecBits = 128>
struct AutoVectorizingCopyWithAssumedAlignment
     : UniversalCopy<uint_bit_t<MaxVecBits>>
{
  static_assert(MaxVecBits == 8 || MaxVecBits == 16 || MaxVecBits == 32 || MaxVecBits == 64 || MaxVecBits == 128,
                "Expected MaxVecBits to be 8 or 16 or 32 or 64 or 128 for alignment and performance.");
};

//
// AutoVectorizingCopy alias assumes maximal alignment of pointers and dynamic strides.
//   If this is not the case then AutoVectorizingCopyWithAssumedAlignment should be used instead
//

using AutoVectorizingCopy = AutoVectorizingCopyWithAssumedAlignment<128>;

//
// DefaultCopy alias does not assume alignment of pointers or dynamic strides.
//

using DefaultCopy = AutoVectorizingCopyWithAssumedAlignment<8>;

//
// Copy policy automatically selecting between
// UniversalCopy and cp.async , based on type and memory space.
//
struct AutoCopyAsync {};

//
// Global memory prefetch into L2
//

CUTE_HOST_DEVICE static void
prefetch(void const* gmem_ptr)
{
#if defined(__CUDA_ARCH__)
  asm volatile("prefetch.global.L2 [%0];\n" : : "l"(gmem_ptr) : "memory");
#endif
}

} // end namespace cute
