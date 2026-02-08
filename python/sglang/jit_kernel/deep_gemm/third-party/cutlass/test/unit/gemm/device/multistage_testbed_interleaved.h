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

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/host_reorder.h"

namespace test {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

template <typename Gemm, int InterleavedK>
struct MultistageInterleavedTestbed {

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  //
  // Methods
  //

  MultistageInterleavedTestbed(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view, 
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, 2, -2, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    } 
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    } 
    else {
      EXPECT_TRUE(false) << "Not implemented";
      return false;
    }

    return true;
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

    if (properties.sharedMemPerMultiprocessor < smem_size) {
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(
    cutlass::gemm::GemmCoord problem_size, 
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
    // Allocate the GEMM workspace
    //

    cutlass::HostTensor<
      typename Gemm::ElementA, 
      typename Gemm::LayoutA> tensor_A(problem_size.mk());

    cutlass::HostTensor<
      typename Gemm::ElementB, 
      typename Gemm::LayoutB> tensor_B(problem_size.kn());

    cutlass::HostTensor<
      typename Gemm::ElementB, 
      typename Gemm::LayoutB> tensor_B_reordered(problem_size.kn());

    cutlass::HostTensor<
      typename Gemm::ElementC, 
      typename Gemm::LayoutC> tensor_C(problem_size.mn());

    cutlass::HostTensor<
      typename Gemm::ElementC, 
      typename Gemm::LayoutC> tensor_D(problem_size.mn());

    cutlass::HostTensor<
      typename Gemm::ElementC, 
      typename Gemm::LayoutC> reference_D(problem_size.mn(), false);

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2019));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2018));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2017));

    cutlass::reorder_column<InterleavedK>(
        tensor_B_reordered.host_ref(), tensor_B.host_ref(), problem_size);

    cutlass::reference::host::TensorCopy(
      reference_D.host_view(), 
      tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B_reordered.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();

    //
    // Initialize the GEMM operator
    //

    typename Gemm::Arguments arguments{
      problem_size,
      tensor_A.device_ref(),
      tensor_B_reordered.device_ref(),
      tensor_C.device_ref(),
      tensor_D.device_ref(),
      {alpha, beta}
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.initialize(arguments);

    EXPECT_TRUE(status == cutlass::Status::kSuccess);

    //
    // Run the GEMM
    //

    status = gemm_op();

    EXPECT_TRUE(status == cutlass::Status::kSuccess);

    //
    // Verify
    //

    cutlass::reference::host::Gemm<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB,
        typename Gemm::ElementC, typename Gemm::LayoutC, ElementCompute,
        ElementAccumulator, typename Gemm::Operator>
        reference_gemm;

    reference_gemm(
      problem_size,
      alpha, 
      tensor_A.host_ref(), 
      tensor_B.host_ref(), 
      beta, 
      reference_D.host_ref(), 
      ElementAccumulator(0)
    );
    
    tensor_D.sync_host();

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);

    bool passed = cutlass::reference::host::TensorEquals(
      reference_D.host_view(), 
      tensor_D.host_view());

    EXPECT_TRUE(passed);
    if (!passed) {

      std::stringstream fname;

      fname << "error_Gemm_device_" 
        << problem_size.m() << "x"
        << problem_size.n() << "x"
        << problem_size.k() << "_"
        << Gemm::ThreadblockShape::kM << "x"  
        << Gemm::ThreadblockShape::kN << "x"  
        << Gemm::ThreadblockShape::kK << "_"
        << Gemm::WarpShape::kM << "x"  
        << Gemm::WarpShape::kN << "x"  
        << Gemm::WarpShape::kK << ".txt";

      std::ofstream file(fname.str());

      file
        << "problem: " << problem_size 
        << ", alpha: " << alpha << ", beta: " << beta << "\n\n";

      file 
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nB_reordered =\n" << tensor_B_reordered.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\n\nReference =\n" << reference_D.host_view()
        << "\nComputed =\n" << tensor_D.host_view();
    }

    return passed;
  }

  /// Runs a set of problem sizes
  bool run_all() {
    bool passed = true;

    int problem_size_m[] = {
      InterleavedK, 512 + InterleavedK
    };

    int problem_size_n[] = {
      InterleavedK, 512 + InterleavedK
    };

    int problem_size_k[] = {
      InterleavedK, Gemm::ThreadblockShape::kK * Gemm::kStages + InterleavedK
    };

    double problem_alpha[] = {
      1.0
    };

    double problem_beta[] = {
      0.0
    };

    for (int m : problem_size_m) {
      for (int n : problem_size_n) {
        for (int k : problem_size_k) {
          for (double alpha : problem_alpha) {
            for (double beta : problem_beta) {
 
              passed = run(
                {m, n, k}, 
                ElementCompute(alpha), 
                ElementCompute(beta)
              );

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
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace test

////////////////////////////////////////////////////////////////////////////////
