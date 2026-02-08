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
    \brief Tests for grouped GEMM problem visitors
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

#include "testbed_grouped_scheduler.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

// Run a series of tests on the testbed
template <typename Testbed>
void run_tests() {
  for (int scale_factor : {8, 16, 32, 64}) {
    for (int threadblock_count : {54, 108, 216, 324, 432}) {
      for (int problems : {1, 27, 180, 300}) {
        Testbed testbed;
        testbed.run(problems, threadblock_count, scale_factor);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGroupedScheduler_p128_t128, 64x64x32) {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  static int const kNumPrefetch = 128;
  static int const kThreadCount = 128;
  static bool const kTranspose = false;

  using Testbed = test::gemm::device::TestbedGroupedGemmScheduler<
                              ThreadblockShape,
                              kNumPrefetch,
                              kThreadCount,
                              kTranspose,
                              // List of GroupScheduleModes to compare. List must contain at least two.
                              cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
                              cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>;
  run_tests<Testbed>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGroupedScheduler_p128_t128_transpose, 64x64x32) {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  static int const kNumPrefetch = 128;
  static int const kThreadCount = 128;
  static bool const kTranspose = true;

  using Testbed = test::gemm::device::TestbedGroupedGemmScheduler<
                              ThreadblockShape,
                              kNumPrefetch,
                              kThreadCount,
                              kTranspose,
                              // List of GroupScheduleModes to compare. List must contain at least two.
                              cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
                              cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>;
  run_tests<Testbed>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGroupedScheduler_p256_t256, 64x64x32) {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  static int const kNumPrefetch = 256;
  static int const kThreadCount = 256;
  static bool const kTranspose = false;

  using Testbed = test::gemm::device::TestbedGroupedGemmScheduler<
                              ThreadblockShape,
                              kNumPrefetch,
                              kThreadCount,
                              kTranspose,
                              // List of GroupScheduleModes to compare. List must contain at least two.
                              cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
                              cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>;
  run_tests<Testbed>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGroupedScheduler_p256_t128, 64x64x32) {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  static int const kNumPrefetch = 256;
  static int const kThreadCount = 128;
  static bool const kTranspose = false;

  using Testbed = test::gemm::device::TestbedGroupedGemmScheduler<
                              ThreadblockShape,
                              kNumPrefetch,
                              kThreadCount,
                              kTranspose,
                              // List of GroupScheduleModes to compare. List must contain at least two.
                              cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
                              cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>;
  run_tests<Testbed>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGroupedScheduler_p256_t256, 64x32x32) {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 32>;
  static int const kNumPrefetch = 256;
  static int const kThreadCount = 256;
  static bool const kTranspose = false;

  using Testbed = test::gemm::device::TestbedGroupedGemmScheduler<
                              ThreadblockShape,
                              kNumPrefetch,
                              kThreadCount,
                              kTranspose,
                              // List of GroupScheduleModes to compare. List must contain at least two.
                              cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
                              cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>;
  run_tests<Testbed>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGroupedScheduler_p256_t256_transpose, 64x32x32) {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 32>;
  static int const kNumPrefetch = 256;
  static int const kThreadCount = 256;
  static bool const kTranspose = true;

  using Testbed = test::gemm::device::TestbedGroupedGemmScheduler<
                              ThreadblockShape,
                              kNumPrefetch,
                              kThreadCount,
                              kTranspose,
                              // List of GroupScheduleModes to compare. List must contain at least two.
                              cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
                              cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>;
  run_tests<Testbed>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGroupedScheduler_p256_t256, 32x64x32) {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 32>;
  static int const kNumPrefetch = 256;
  static int const kThreadCount = 256;
  static bool const kTranspose = false;

  using Testbed = test::gemm::device::TestbedGroupedGemmScheduler<
                              ThreadblockShape,
                              kNumPrefetch,
                              kThreadCount,
                              kTranspose,
                              // List of GroupScheduleModes to compare. List must contain at least two.
                              cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
                              cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>;
  run_tests<Testbed>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmGroupedScheduler_p256_t256_transpose, 32x64x32) {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 32>;
  static int const kNumPrefetch = 256;
  static int const kThreadCount = 256;
  static bool const kTranspose = true;

  using Testbed = test::gemm::device::TestbedGroupedGemmScheduler<
                              ThreadblockShape,
                              kNumPrefetch,
                              kThreadCount,
                              kTranspose,
                              // List of GroupScheduleModes to compare. List must contain at least two.
                              cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
                              cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>;
  run_tests<Testbed>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
