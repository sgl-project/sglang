/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cute/arch/copy.hpp>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 500
  #define CUTE_ARCH_WARP_SHUFFLE_ENABLED 1
#endif

namespace cute
{
// Shuffle data between thread pair (0, 1), (2, 3), etc.
struct SM50_Shuffle_U32_2x2Trans_XOR1
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_WARP_SHUFFLE_ENABLED)
    uint32_t x0 = src0;
    uint32_t y0 = __shfl_xor_sync(0xffffffff, x0, 1);

    uint32_t x1 = src1;
    uint32_t y1 = __shfl_xor_sync(0xffffffff, x1, 1);

    if (threadIdx.x % 2 == 0) {
      dst1 = y0;
    } 
    else {
      dst0 = y1;
    }
#else 
    CUTE_INVALID_CONTROL_PATH("Trying to use __shfl_xor_sync without CUTE_ARCH_WARP_SHUFFLE_ENABLED.");
#endif
  }
};

// Shuffle data between thread pair (0, 4), (1, 5), etc.
struct SM50_Shuffle_U32_2x2Trans_XOR4
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_WARP_SHUFFLE_ENABLED)
    uint32_t x0 = threadIdx.x & 4  ? src0 : src1;
    uint32_t y0 = __shfl_xor_sync(0xffffffff, x0, 4);

    // Replace detination register with shuffle result.
    if (threadIdx.x & 0x4) {
      dst0 = y0;
    } 
    else {
      dst1 = y0;
    }
#else 
    CUTE_INVALID_CONTROL_PATH("Trying to use __shfl_xor_sync without CUTE_ARCH_WARP_SHUFFLE_ENABLED.");
#endif
  }
};


} // end namespace cute
