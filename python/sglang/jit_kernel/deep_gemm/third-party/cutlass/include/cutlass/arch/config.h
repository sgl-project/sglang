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

/*! \file
    \brief Definitions for architecture macros
*/

#pragma once

#include "cutlass/platform/platform.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

// SM90
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 0))
  #define CUTLASS_ARCH_MMA_SM90_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM90_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900)
    #define CUTLASS_ARCH_MMA_SM90_ENABLED 1

    #if (!defined(CUTLASS_ARCH_MMA_SM90A_ENABLED) && defined(__CUDA_ARCH_FEAT_SM90_ALL))
      #define CUTLASS_ARCH_MMA_SM90A_ENABLED 1
    #endif
  #endif
#endif

#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 2))
  #define CUTLASS_ARCH_MMA_SPARSE_SM90_SUPPORTED
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// Modifiable TMA
// tensormap.replace is arch conditional
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3))
  #define CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_ENABLED) && \
    (defined(__CUDA_ARCH_FEAT_SM90_ALL)  || defined(__CUDA_ARCH_FEAT_SM100_ALL) || \
     defined(__CUDA_ARCH_FEAT_SM101_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL)))
    #define CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_ENABLED 1
  #endif
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////

// SM90 F64
#if (__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 8))
  #define CUTLASS_ARCH_MMA_SM90_F64_MMA_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM90_F64_MMA_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    #define CUTLASS_ARCH_MMA_SM90_F64_MMA_ENABLED 1
  #endif
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////

// SM100, SM100a
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
  #define CUTLASS_ARCH_MMA_SM100_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM100_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1000)
    #define CUTLASS_ARCH_MMA_SM100_ENABLED 1

    #if (!defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) && defined(__CUDA_ARCH_FEAT_SM100_ALL))
      #define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
    #endif

    // SM100f
    #if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9))
    #define CUTLASS_ARCH_MMA_SM100F_SUPPORTED 1
    #endif

    #if (!defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) && CUDA_ARCH_FAMILY(1000))
      #define CUTLASS_ARCH_MMA_SM100F_ENABLED CUTLASS_ARCH_MMA_SM100F_SUPPORTED
    #endif
  #endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// SM101 and SM101a
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)
  #define CUTLASS_ARCH_MMA_SM101_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM101_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1010)
    #define CUTLASS_ARCH_MMA_SM101_ENABLED 1

    #if (!defined(CUTLASS_ARCH_MMA_SM101A_ENABLED) && defined(__CUDA_ARCH_FEAT_SM101_ALL))
      #define CUTLASS_ARCH_MMA_SM101A_ENABLED 1
    #endif

    // SM101f
    #if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
    #define CUTLASS_ARCH_MMA_SM101F_SUPPORTED 1
    #endif

    #if (!defined(CUTLASS_ARCH_MMA_SM101F_ENABLED) && CUDA_ARCH_FAMILY(1010))
      #define CUTLASS_ARCH_MMA_SM101F_ENABLED CUTLASS_ARCH_MMA_SM101F_SUPPORTED
    #endif
  #endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// SM110 and SM110a only on 13.0 and above
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 0))
  #define CUTLASS_ARCH_MMA_SM110_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM110_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1100)
    #define CUTLASS_ARCH_MMA_SM110_ENABLED 1

    #if (!defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) && defined(__CUDA_ARCH_FEAT_SM110_ALL))
      #define CUTLASS_ARCH_MMA_SM110A_ENABLED 1
    #endif

    // SM110f
    #if (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 0))
    #define CUTLASS_ARCH_MMA_SM110F_SUPPORTED 1
    #endif

    #if (!defined(CUTLASS_ARCH_MMA_SM110F_ENABLED) && CUDA_ARCH_FAMILY(1100))
      #define CUTLASS_ARCH_MMA_SM110F_ENABLED CUTLASS_ARCH_MMA_SM110F_SUPPORTED
    #endif
  #endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// SM120 and SM120a
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
  #define CUTLASS_ARCH_MMA_SM120_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM120_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200)
    #define CUTLASS_ARCH_MMA_SM120_ENABLED 1

    #if (!defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) && defined(__CUDA_ARCH_FEAT_SM120_ALL))
      #define CUTLASS_ARCH_MMA_SM120A_ENABLED 1
    #endif

    // SM120f
    #if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9))
    #define CUTLASS_ARCH_MMA_SM120F_SUPPORTED 1
    #endif

    #if (!defined(CUTLASS_ARCH_MMA_SM120F_ENABLED) && CUDA_ARCH_FAMILY(1200))
      #define CUTLASS_ARCH_MMA_SM120F_ENABLED CUTLASS_ARCH_MMA_SM120F_SUPPORTED
    #endif
  #endif
#endif

// SM103 and SM103a
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9))
  #define CUTLASS_ARCH_MMA_SM103_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM103_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030)
    #define CUTLASS_ARCH_MMA_SM103_ENABLED 1

    #if (!defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) && defined(__CUDA_ARCH_FEAT_SM103_ALL))
      #define CUTLASS_ARCH_MMA_SM103A_ENABLED 1
    #endif

    // SM103f
    #if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9))
    #define CUTLASS_ARCH_MMA_SM103F_SUPPORTED 1
    #endif

    #if (!defined(CUTLASS_ARCH_MMA_SM103F_ENABLED) && CUDA_ARCH_FAMILY(1030))
      #define CUTLASS_ARCH_MMA_SM103F_ENABLED CUTLASS_ARCH_MMA_SM103F_SUPPORTED
    #endif
  #endif
#endif

// SM121 and SM121a
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9))
  #define CUTLASS_ARCH_MMA_SM121_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM121_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210)
    #define CUTLASS_ARCH_MMA_SM121_ENABLED 1

    #if (!defined(CUTLASS_ARCH_MMA_SM121A_ENABLED) &&\
         (defined(__CUDA_ARCH_FEAT_SM121_ALL) || CUDA_ARCH_CONDITIONAL(1210)))
      #define CUTLASS_ARCH_MMA_SM121A_ENABLED 1
    #endif

    // SM121f
    #if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9))
    #define CUTLASS_ARCH_MMA_SM121F_SUPPORTED 1
    #endif

    #if (!defined(CUTLASS_ARCH_MMA_SM121F_ENABLED) && CUDA_ARCH_FAMILY(1210))
      #define CUTLASS_ARCH_MMA_SM121F_ENABLED CUTLASS_ARCH_MMA_SM121F_SUPPORTED
    #endif
  #endif
#endif


#if (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM101A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM101F_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM103F_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM110F_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM120F_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM121A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121F_ENABLED))
#  define CUTLASS_ARCH_CLC_ENABLED
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////

