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


#include <iostream>

// Run tests on GPUs 

int testRun(int arch, std::vector<bool (*)()> & test_funcs, const std::string & test_name) {

  bool supported = false;

  int arch_major = arch / 10;
  int arch_minor = arch - arch / 10 * 10;  

  if(arch_major >= 8) {
    // Ampere Tensor Core operations exposed with mma.sync are first available in CUDA 11.0.
    //
    // CUTLASS must be compiled with CUDA 11 Toolkit to run Conv2dFprop examples.
    if (__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0)) {
      supported = true;
    }
  }
  else if(arch_major >= 7) {
    // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
    //
    // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
    if (__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)) {
      supported = true;
    }
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!(props.major == arch_major && props.minor == arch_minor)) {
    supported = false;
  }

  if (!supported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    std::cout << "This example isn't supported on current architecture" << std::endl;
    return 0;
  }

  bool pass = true;
 
  std::cout << "Device: " << props.name << std::endl;
  std::cout << "Arch: SM" << arch << std::endl;
  std::cout << "Test: " << test_name << std::endl;
  for(auto func : test_funcs) {
    pass &= func();
  }


  if(pass)
    return 0;
  else
    return -1;

}

