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
    \brief Common Testbed file shared by Pipeline unit tests
*/

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cutlass/gemm/gemm.h>

#include "cutlass/util/command_line.h"
#include "../common/cutlass_unit_test.h"

#if CUDA_12_0_SM90_FEATURES_SUPPORTED
  #define CUTLASS_UNIT_TEST_PIPELINE true
#else
  #define CUTLASS_UNIT_TEST_PIPELINE false
#endif

// Command line test options
struct Options {
  //
  // Data Members
  // 
  bool help;
  bool verification_enabled;
  int SM_count;
  int clock_MHz;

  //
  // Methods
  // 
  Options():
    help(false),
    verification_enabled(true),
    SM_count(116),
    clock_MHz(1477)
  { }

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("verification-enabled", verification_enabled, true);
    cmd.get_cmd_line_argument("sm-count", SM_count, 116);
    cmd.get_cmd_line_argument("clock", clock_MHz, 1477);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "Options:\n\n"
      << "  --help                          If specified, displays this usage statement.\n\n"
      << "  --verification-enabled=<bool>   Enable/Disable verification\n"
      << "  --sm-count=<int>                Number of SMs on the chip\n"
      << "  --clock=<int>                   Locked clock value in Mhz\n";

    return out;
  }
};

//
// Testbed
//

template<typename Pipeline>
struct Testbed {
private:
  // Commandline options
  Options options;

  void run_test(uint32_t const kNumIters) {

    // Run CuTe Gemm 
    Pipeline pipeline;

    cudaError_t result = pipeline.run(kNumIters);

    CUTE_CHECK_LAST();
  }


public:
  Testbed(Options const &options_) : options(options_) {
    int device_id = 0;
    cudaDeviceProp device_prop;
    CUTE_CHECK_ERROR(cudaSetDevice(device_id));
    CUTE_CHECK_ERROR(cudaGetDeviceProperties(&device_prop, device_id));
  
    if (device_prop.major < 1) {
      fprintf(stderr, "Device does not support CUDA.\n");
      exit(1);
    }
  }

  /// Run verification Gemm problem sizes
  bool verification() {

    std::array<uint32_t, 5> kNumIters;

    for (size_t i = 0; i < kNumIters.size(); ++i) {
      kNumIters[i] = static_cast<uint32_t>( (rand() % 1000) + 1 );
    }

    for (int n : kNumIters) {
      std::cout << "Stages = " << Pipeline::Stages << " kNumIters = " << n << "\n";
      run_test(n);
    }

    return true;
  }
};
