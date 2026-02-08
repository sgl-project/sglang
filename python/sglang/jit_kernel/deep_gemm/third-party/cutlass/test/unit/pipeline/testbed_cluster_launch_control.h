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
    \brief Testbed file used by cluster launch control pipeline unit test
*/

//

//

#if CUDA_12_0_SM90_FEATURES_SUPPORTED
  #define CUTLASS_UNIT_TEST_PIPELINE true
#else
  #define CUTLASS_UNIT_TEST_PIPELINE false
#endif

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cutlass/gemm/gemm.h>

#include "cutlass/util/command_line.h"

// Command line test options
struct OptionsClusterLaunch {
  //
  // Data Members
  // 
  bool help = false;
  bool verification_enabled = true;
  int SM_count = 116;
  int clock_MHz = 1477;
  dim3 grid_dim = {0,0,0};

  //
  // Methods
  // 

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("verification-enabled", verification_enabled, verification_enabled);
    cmd.get_cmd_line_argument("sm-count", SM_count, SM_count);
    cmd.get_cmd_line_argument("clock", clock_MHz, clock_MHz);
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
class TestbedClusterLaunch {
private:
  // Commandline options
  OptionsClusterLaunch options;

  bool run_test() {

    // Run CuTe Gemm 
    Pipeline pipeline;

    bool success = false;
    cudaError_t result = pipeline.run(success, this->options.grid_dim);
    
    CUTE_CHECK_LAST();
    return success;
  }


public:
  TestbedClusterLaunch(OptionsClusterLaunch const &options_) : options(options_) {
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

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  printf(
    "CUTLASS_ARCH_MMA_SM100_SUPPORTED must be set, but it is not. \n"
    "This test is waived.\n"
  );
  return true;
#endif

#if 0
    bool is_success = false;
    for (int i = 0; i< 10; i++){
      printf("iteration = %d\n", i);
      is_success = run_test();
      if ( not is_success )
        return is_success;
    }
    return is_success;
#else
    // Run the test with single launch
    return run_test();
#endif
  }
};
