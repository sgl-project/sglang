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
    \brief Defines tags for architecture-specific configurations.
*/

#pragma once

#include "cutlass/cutlass.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

constexpr int sm100_smem_capacity_bytes = 232448;  
constexpr int sm120_smem_capacity_bytes = 101376;

#if defined(__NVCC__) || defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))

/// Computes laneId within a warp
CUTLASS_DEVICE
int LaneId() {
  int ret;
  asm ("mov.u32 %0, %%laneid;" : "=r"(ret) : );
  return ret;
}

/// Computes SM number the thread is running on
CUTLASS_DEVICE
int SmId() {
  int ret;
  asm ("mov.u32 %0, %%smid;" : "=r"(ret) : );
  return ret;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
struct Sm50 {
  static int const kMinComputeCapability = 50;
}; 
struct Sm60 {
  static int const kMinComputeCapability = 60;
}; 
struct Sm61 {
  static int const kMinComputeCapability = 61;
};
struct Sm70 {
  static int const kMinComputeCapability = 70;
};
struct Sm72 {
  static int const kMinComputeCapability = 72;
};
struct Sm75 {
  static int const kMinComputeCapability = 75;
};
struct Sm80 {
  static int const kMinComputeCapability = 80; 
};
struct Sm86 {
  static int const kMinComputeCapability = 86;
};
struct Sm89 {
  static int const kMinComputeCapability = 89;
};
struct Sm90 {
  static int const kMinComputeCapability = 90; 
};


struct Sm100 {
  static int const kMinComputeCapability = 100; 
};

struct Sm101 {
  static int const kMinComputeCapability = 101; 
};

struct Sm120 {
  static int const kMinComputeCapability = 120;
};

struct Sm103 {
  static int const kMinComputeCapability = 103; 
};

/// Triggers a breakpoint on the device
CUTLASS_DEVICE
void device_breakpoint() {
#if defined(__CUDA_ARCH__)
  asm volatile ("  brkpt;\n");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
