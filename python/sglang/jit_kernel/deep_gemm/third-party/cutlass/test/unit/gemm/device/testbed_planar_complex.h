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
#include <fstream>
#include <sstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/gemm_planar_complex.h"
#include "cutlass/util/host_tensor_planar_complex.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"

////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
class TestbedPlanarComplex {
public:

  using ElementA = typename Gemm::ElementA;
  using LayoutA = typename Gemm::LayoutA;
  using ElementB = typename Gemm::ElementB;
  using LayoutB = typename Gemm::LayoutB;
  using ElementC = typename Gemm::ElementC;
  using LayoutC = typename Gemm::LayoutC;
  using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  //
  // Data members
  //

  cutlass::gemm::GemmCoord problem_size;
  cutlass::HostTensorPlanarComplex<ElementA, LayoutA> tensor_A;
  cutlass::HostTensorPlanarComplex<ElementB, LayoutB> tensor_B;
  cutlass::HostTensorPlanarComplex<ElementC, LayoutC> tensor_C;
  cutlass::HostTensorPlanarComplex<ElementC, LayoutC> tensor_D;
  cutlass::HostTensorPlanarComplex<ElementC, LayoutC> tensor_D_ref;

  //
  // Methods
  //

  TestbedPlanarComplex(cutlass::gemm::GemmCoord const & problem_size): problem_size(problem_size) {

    tensor_A.reset({problem_size.m(), problem_size.k()});
    tensor_B.reset({problem_size.k(), problem_size.n()});
    tensor_C.reset({problem_size.m(), problem_size.n()});
    tensor_D.reset({problem_size.m(), problem_size.n()});
    tensor_D_ref.reset({problem_size.m(), problem_size.n()}, false);
  }

  void initialize() {

    uint64_t seed = 1073;

    int scope_max = 8;
    int scope_min = -8;

    cutlass::reference::host::TensorFillRandomUniform(
        tensor_A.host_view(), seed, scope_max, scope_min, 0);

    cutlass::reference::host::TensorFillRandomUniform(
        tensor_B.host_view(), seed * 2019, scope_max, scope_min, 0);

    cutlass::reference::host::TensorFillRandomUniform(
        tensor_C.host_view(), seed * 2020, scope_max, scope_min, 0);

    cutlass::reference::host::TensorFill(tensor_D.host_view(), cutlass::complex<ElementC>());
    cutlass::reference::host::TensorFill(tensor_D_ref.host_view(), cutlass::complex<ElementC>());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
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
  
  bool run(
      cutlass::complex<ElementCompute> alpha = {1, 0},
      cutlass::complex<ElementCompute> beta = {0, 0}) {

    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient CUDA device." << std::endl;
      }
      return true;
    }

    initialize();

    int batch_count = 1;

    ElementA *ptr_A = tensor_A.device_data();
    ElementB *ptr_B = tensor_B.device_data();
    ElementC *ptr_C = tensor_C.device_data();
    ElementC *ptr_D = tensor_D.device_data();

    typename LayoutA::Stride::Index lda = tensor_A.layout().stride(0);
    typename LayoutB::Stride::Index ldb = tensor_B.layout().stride(0);
    typename LayoutC::Stride::Index ldc = tensor_C.layout().stride(0);
    typename LayoutC::Stride::Index ldd = tensor_D.layout().stride(0);

    int64_t imag_stride_A = tensor_A.imaginary_stride();
    int64_t imag_stride_B = tensor_B.imaginary_stride();
    int64_t imag_stride_C = tensor_C.imaginary_stride();
    int64_t imag_stride_D = tensor_D.imaginary_stride();

    //
    // Launch device kernel
    //

    Gemm gemm_op;

    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      batch_count,
      {alpha, beta},
      ptr_A,
      ptr_A + imag_stride_A,
      ptr_B,
      ptr_B + imag_stride_B,
      ptr_C,
      ptr_C + imag_stride_C,
      ptr_D,
      ptr_D + imag_stride_D,
      lda,
      lda,
      ldb,
      ldb,
      ldc,
      ldc,
      ldd,
      ldd
    };

    cutlass::Status status = gemm_op(args);

    EXPECT_EQ(status, cutlass::Status::kSuccess);

    cudaError_t error = cudaDeviceSynchronize();

    tensor_D.sync_host();

    //
    // Compute reference
    //

    cutlass::reference::host::GemmPlanarComplex<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementAccumulator
    >(
      problem_size,
      alpha,
      tensor_A.host_ref(),
      Gemm::kTransformA,
      tensor_B.host_ref(),
      Gemm::kTransformB,
      beta,
      tensor_C.host_ref(),
      tensor_D_ref.host_ref()
    );
    
    bool passed = cutlass::reference::host::TensorEquals(
      tensor_D.host_view(), 
      tensor_D_ref.host_view()
    );

    EXPECT_TRUE(passed);

    if (!passed) {
      std::ofstream output("gemm_planar_complex.txt");

      output
        << "A:\n" << tensor_A.host_view() << "\n"
        << "B:\n" << tensor_B.host_view() << "\n"
        << "C:\n" << tensor_C.host_view() << "\n"
        << "Reference:\n"
        << tensor_D_ref.host_view() << "\n"
        << "Computed:\n"
        << tensor_D.host_view() << "\n";
    }

    return passed;
  }
};

template <typename Gemm>
bool TestOneGemmPlanarComplex(cutlass::gemm::GemmCoord problem_size) {

  TestbedPlanarComplex<Gemm> testbed(problem_size);

  return testbed.run();
}

template <typename Gemm>
bool TestAllGemmPlanarComplex() {

  int M[] = {
    16, 64, 72, 144, 264, 520,
  };

  int N[] = {
    16, 64, 72, 144, 248, 264, 520
  };

  int K[] = {
    8, 64, 72, 96,  264, 520
  };

  using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;

  cutlass::complex<ElementCompute> alpha_values[] = {
    {ElementCompute(1.25), ElementCompute(-0.5)}
  };

  cutlass::complex<ElementCompute> beta_values[] = {
    {ElementCompute(-2.25), ElementCompute(1.5)}
  };

  for (int m : M) {
    for (int n : N) {
      for (int k : K) {
        
        test::gemm::device::TestbedPlanarComplex<Gemm> testbed({m, n, k});

        for (auto const &alpha : alpha_values) {
          for (auto const &beta : beta_values) {

            bool passed = testbed.run(alpha, beta);
            if (!passed) {
              return false;
            }            
          }
        }
      }
    }
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////


