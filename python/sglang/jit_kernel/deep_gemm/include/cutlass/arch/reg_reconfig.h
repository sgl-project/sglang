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
  \brief PTX for CTA Reconfiguration
*/

#pragma once

#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include <cuda/std/cstdint>
#else
#include <cstdint>
#endif

#ifndef CUDA_CTA_RECONFIG_ACTIVATED
  #if defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 12 && (             \
         (__CUDA_ARCH__ ==  900 && defined(__CUDA_ARCH_FEAT_SM90_ALL))      \
      || (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))     \
      || (__CUDA_ARCH__ == 1010 && defined(__CUDA_ARCH_FEAT_SM101_ALL))     \
      || (__CUDA_ARCH__ == 1030 && defined(__CUDA_ARCH_FEAT_SM103_ALL))     \
      || (__CUDA_ARCH__ == 1200 && defined(__CUDA_ARCH_FEAT_SM120_ALL))     \
      || (__CUDA_ARCH__ == 1210 && defined(__CUDA_ARCH_FEAT_SM121_ALL))     \
    )
    #define CUDA_CTA_RECONFIG_ACTIVATED 1
  #endif

  #if defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 12 && (          \
         (__CUDA_ARCH__ == 1000 && CUDA_ARCH_FAMILY(1000))  \
      || (__CUDA_ARCH__ == 1010 && CUDA_ARCH_FAMILY(1010))  \
      || (__CUDA_ARCH__ == 1030 && CUDA_ARCH_FAMILY(1030))  \
      || (__CUDA_ARCH__ == 1200 && CUDA_ARCH_FAMILY(1200))  \
      || (__CUDA_ARCH__ == 1210 && CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210))  \
    )
    #define CUDA_CTA_RECONFIG_ACTIVATED 1
  #endif

#endif

namespace cutlass {
namespace arch {

template<uint32_t RegCount>
CUTLASS_DEVICE
void warpgroup_reg_alloc(){
#if CUDA_CTA_RECONFIG_ACTIVATED
  asm volatile( "setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount) );
#endif
}

template<uint32_t RegCount>
CUTLASS_DEVICE
void warpgroup_reg_dealloc(){
#if CUDA_CTA_RECONFIG_ACTIVATED
  asm volatile( "setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount) );
#endif
}

} // namespace arch
} // namespace cutlass
