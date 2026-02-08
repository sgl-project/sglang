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
    \brief Tests for device-wide GEMM interface
*/

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm_complex.h"

#include "testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
struct TestbedComplex : public Testbed<Gemm> {

  using Base = Testbed<Gemm>;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;


  //
  // Methods
  //

  TestbedComplex(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    Base(init_A_, init_B_, init_C_, seed_) { }


  /// Verifies the result is a GEMM
  bool verify(
    cutlass::gemm::GemmCoord problem_size, 
    ElementCompute alpha, 
    ElementCompute beta) {

    //
    // Verify
    //

    cutlass::reference::host::GemmComplex(
      problem_size,
      alpha, 
      this->tensor_A.host_ref(),
      Gemm::kTransformA,
      this->tensor_B.host_ref(), 
      Gemm::kTransformB,
      beta, 
      this->tensor_C.host_ref(), 
      this->reference_D.host_ref(), 
      ElementAccumulator(0)
    );

    return this->compare_reference(problem_size, alpha, beta);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //
    
    size_t smem_size = sizeof(typename Gemm::GemmKernel::SharedStorage);
    
    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);
    
    if (result != cudaSuccess) {
    	throw std::runtime_error("cudaGetDevice() API call failed.");
    }
    
    result = cudaGetDeviceProperties(&properties, device_idx);
    
    if (result != cudaSuccess) {
    	throw std::runtime_error("cudaGetDeviceProperties() failed");
    }
    
    if (properties.sharedMemPerBlockOptin < smem_size) {
    	return false;
    }

    return true;
  }

  /// Executes one test
  bool run(
    cutlass::gemm::GemmCoord problem_size, 
    int split_k_slices = 1,
    ElementCompute alpha = ElementCompute(1), 
    ElementCompute beta = ElementCompute(0)) {

    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient CUDA device." << std::endl;
      }
      return true;
    }

    //
    // Initialize workspace
    //

    this->initialize(problem_size);
		

    //
    // Initialize the GEMM operator
    //

    typename Gemm::Arguments arguments{
      problem_size,
      this->tensor_A.device_ref(),
      this->tensor_B.device_ref(),
      this->tensor_C.device_ref(),
      this->tensor_D.device_ref(),
      {alpha, beta},
      split_k_slices
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Run the GEMM
    //

    status = gemm_op();

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Verify
    //

    bool passed = this->verify(problem_size, alpha, beta);

    if (!passed) {
      std::cout << "Error with split_k_slices = " << split_k_slices << ", alpha: " << alpha << std::endl;
    }

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
bool TestAllGemmComplex() {
  bool passed = true;

  using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;

  int const kMinimumOperandElementSize = 
    std::min(
      int(cutlass::sizeof_bits<typename Gemm::ElementA>::value), 
      int(cutlass::sizeof_bits<typename Gemm::ElementB>::value));

  int const kAlignment = 
    cutlass::platform::is_same<
      typename Gemm::OperatorClass, 
      cutlass::arch::OpClassSimt>::value ? 1 : 128 / kMinimumOperandElementSize;

  int problem_size_m[] = {
    kAlignment, 512 - 3*kAlignment
  };

  int problem_size_n[] = {
    kAlignment, 512 - 2*kAlignment
  };

  int problem_size_k[] = {
    kAlignment, 128 - kAlignment
  };

  int split_k_slices[] = {
    1, 2, 3
  };

  double problem_alpha[] = {
    1
  };

  double problem_beta[] = {
    2.0
  };

  TestbedComplex<Gemm> testbed;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        for (int split_k : split_k_slices) {

          if (!Gemm::kSplitKSerial && split_k > 1) {
            continue;
          }

          for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {

              cutlass::gemm::GemmCoord problem_size(m, n, k);

              passed = testbed.run(
                problem_size, 
                split_k,
                cutlass::from_real<ElementCompute>(alpha), 
                cutlass::from_real<ElementCompute>(beta)
              );

              if (!passed) {
                return false;
              }
            }
          }
        }
      }
    }
  }

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

