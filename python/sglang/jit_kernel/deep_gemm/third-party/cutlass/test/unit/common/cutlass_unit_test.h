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

#pragma once
#pragma warning (disable : 4068 ) /* disable unknown pragma warnings for visual studio */

#pragma nv_diag_suppress boolean_controlling_expr_is_constant
#include <gtest/gtest.h>
#pragma nv_diag_warning boolean_controlling_expr_is_constant
#pragma warning( disable : 4503)

#include <cstdlib>
#include <string>

#include <cuda_runtime_api.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gets a CUDA device
cudaDeviceProp GetCudaDevice();

/// Prints device properties
std::ostream &operator<<(std::ostream &out, cudaDeviceProp const &device);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Sets flags for Unit test
void FilterArchitecture();

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Reads environment variable `CUTLASS_UNIT_TEST_PROBLEM_COUNT` to control the number and order
//  of problem sizes run by CUTLASS unit tests
int CutlassUnitTestProblemCount();

/////////////////////////////////////////////////////////////////////////////////////////////////

// active test macro
#define CUTLASS_TEST_LEVEL_ACTIVE(LEVEL,NAME_STATIC,NAME_DYNAMIC,...) \
    TEST(NAME_STATIC,L##LEVEL##_##NAME_DYNAMIC) __VA_ARGS__

// disabled test macro
#define CUTLASS_TEST_LEVEL_DISABLED(LEVEL,NAME_STATIC,NAME_DYNAMIC,...) \
    TEST(NAME_STATIC,DISABLED_L##LEVEL##_##NAME_DYNAMIC) {}

#if CUTLASS_TEST_LEVEL == 0
#define CUTLASS_TEST_L0(NAME_STATIC,NAME_DYNAMIC,...)   CUTLASS_TEST_LEVEL_ACTIVE(0,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#define CUTLASS_TEST_L1(NAME_STATIC,NAME_DYNAMIC,...) CUTLASS_TEST_LEVEL_DISABLED(1,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#define CUTLASS_TEST_L2(NAME_STATIC,NAME_DYNAMIC,...) CUTLASS_TEST_LEVEL_DISABLED(2,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#elif CUTLASS_TEST_LEVEL == 1
#define CUTLASS_TEST_L0(NAME_STATIC,NAME_DYNAMIC,...)   CUTLASS_TEST_LEVEL_ACTIVE(0,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#define CUTLASS_TEST_L1(NAME_STATIC,NAME_DYNAMIC,...)   CUTLASS_TEST_LEVEL_ACTIVE(1,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#define CUTLASS_TEST_L2(NAME_STATIC,NAME_DYNAMIC,...) CUTLASS_TEST_LEVEL_DISABLED(2,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#else
#define CUTLASS_TEST_L0(NAME_STATIC,NAME_DYNAMIC,...)   CUTLASS_TEST_LEVEL_ACTIVE(0,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#define CUTLASS_TEST_L1(NAME_STATIC,NAME_DYNAMIC,...)   CUTLASS_TEST_LEVEL_ACTIVE(1,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#define CUTLASS_TEST_L2(NAME_STATIC,NAME_DYNAMIC,...)   CUTLASS_TEST_LEVEL_ACTIVE(2,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#endif

#if !defined(CUTLASS_TEST_UNIT_ENABLE_WARNINGS)
#define CUTLASS_TEST_UNIT_ENABLE_WARNINGS false
#endif

#if (__CUDACC_VER_MAJOR__ >= 12)
  #define CUDA_12_0_SM90_FEATURES_SUPPORTED true
#else
  #define CUDA_12_0_SM90_FEATURES_SUPPORTED false
#endif

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>

/////////////////////////////////////////////////////////////////////////////////////////////////
