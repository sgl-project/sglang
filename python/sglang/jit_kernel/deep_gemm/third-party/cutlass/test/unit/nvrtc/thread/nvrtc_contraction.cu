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
    \brief Unit tests for GETT
*/

#include <cuda.h>
#include <gtest/gtest.h>

#include "testbed.h"

#include "nvrtc_config.hpp"

#ifndef CUDA_INCLUDE_DIR
static_assert(0, "CUDA include path is not defined");
#endif

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
TEST(SM90_nvrtc_kernel, Contraction) {
  static const char* nvrtc_opts[] = {
    "-w",
    "-default-device",
    "-std=c++17",
    "-arch=sm_90",
    "-I" CUDA_INCLUDE_DIR,
#if (__CUDACC_VER_MAJOR__ >= 13)
    "-I" CUDA_INCLUDE_DIR "/cccl",
#endif // __CUDACC_VER_MAJOR__ >= 13
  };

  EXPECT_TRUE(test::nvrtc::thread::TestbedKernel::compile(
    "nvrtc::thread::ContractionKernel<"
        "cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,"
        "cute::Shape<cute::Shape<cute::_8, cute::_8>, cute::Shape<cute::_16, cute::_8>, cute::Shape<cute::_8, cute::_8>>,"
        "cute::Shape<cute::_1, cute::_2, cute::_1>,"
        "true, true,"
        "10, 10, 10, 10>::Kernel",
    { std::begin(nvrtc_opts), std::end(nvrtc_opts) }
  ));
}
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////
