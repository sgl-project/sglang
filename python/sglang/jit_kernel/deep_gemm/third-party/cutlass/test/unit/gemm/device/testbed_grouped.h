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

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
struct TestbedGrouped {

  //
  // Type definitions
  //

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;

  using MatrixCoord = typename LayoutC::TensorCoord;

  //
  // Data members
  //

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint32_t seed;

  int problem_count;

  std::vector<cutlass::gemm::GemmCoord>               problem_sizes_host;
  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;

  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;

  std::vector<int64_t> lda_host;
  std::vector<int64_t> ldb_host;
  std::vector<int64_t> ldc_host;
  std::vector<int64_t> ldd_host;

  cutlass::DeviceAllocation<int64_t> lda;
  cutlass::DeviceAllocation<int64_t> ldb;
  cutlass::DeviceAllocation<int64_t> ldc;
  cutlass::DeviceAllocation<int64_t> ldd;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementC> block_D;

  cutlass::DeviceAllocation<ElementA *> ptr_A;
  cutlass::DeviceAllocation<ElementB *> ptr_B;
  cutlass::DeviceAllocation<ElementC *> ptr_C;
  cutlass::DeviceAllocation<ElementC *> ptr_D;

  //
  // Methods
  //

  TestbedGrouped(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint32_t seed_ = 3080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view, 
    cutlass::Distribution::Kind dist_kind,
    uint32_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Gemm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        if (cutlass::sizeof_bits<ElementAccumulator>::value <= 16) {
          scope_max = 5;
          scope_min = -5;
        }
        else {
          scope_max = 8;
          scope_min = -8;
        }
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    } 
    else {
      // no fill - remain zero
    }

    return true;
  }

  /// Initializes data structures
  void initialize() {

    //
    // Choose random problem sizes
    //

    // construct a few problems of random sizes
    srand(seed);

    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;
    int64_t total_elements_D = 0;


    lda_host.resize(problem_count);
    ldb_host.resize(problem_count);
    ldc_host.resize(problem_count);
    ldd_host.resize(problem_count);

    problem_sizes_host.clear();
    problem_sizes_host.resize(problem_count);

    for (int32_t i = 0; i < problem_count; ++i) {

      cutlass::gemm::GemmCoord problem(
        8 * (rand() % 64) + 24,
        8 * (rand() % 64) + 24,
        8 * (rand() % 64) + 24);

      if (!i) {
        problem = cutlass::gemm::GemmCoord(48, 16, 8);
      }

      problem_sizes_host.at(i) = problem;

      // std::cout << "Problem[" << i << "]: " << problem << std::endl;

      lda_host.at(i) = LayoutA::packed({problem.m(), problem.k()}).stride(0);
      ldb_host.at(i) = LayoutB::packed({problem.k(), problem.n()}).stride(0);
      ldc_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);
      ldd_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);

      offset_A.push_back(total_elements_A);
      offset_B.push_back(total_elements_B);
      offset_C.push_back(total_elements_C);
      offset_D.push_back(total_elements_D);

      int64_t elements_A = problem.m() * problem.k();
      int64_t elements_B = problem.k() * problem.n();
      int64_t elements_C = problem.m() * problem.n();
      int64_t elements_D = problem.m() * problem.n();

      total_elements_A += elements_A;
      total_elements_B += elements_B;
      total_elements_C += elements_C;
      total_elements_D += elements_D;

      // Random strides between problems?
    }

    problem_sizes_device.reset(problem_count);
    problem_sizes_device.copy_from_host(problem_sizes_host.data());

    lda.reset(problem_count);
    ldb.reset(problem_count);
    ldc.reset(problem_count);
    ldd.reset(problem_count);

    lda.copy_from_host(lda_host.data());
    ldb.copy_from_host(ldb_host.data());
    ldc.copy_from_host(ldc_host.data());
    ldd.copy_from_host(ldd_host.data());

    //
    // Assign pointers
    //

    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_D);

    std::vector<ElementA *> ptr_A_host(problem_count);
    std::vector<ElementB *> ptr_B_host(problem_count);
    std::vector<ElementC *> ptr_C_host(problem_count);
    std::vector<ElementC *> ptr_D_host(problem_count);

    for (int32_t i = 0; i < problem_count; ++i) {
      ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
      ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
      ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    }

    ptr_A.reset(problem_count);
    ptr_A.copy_from_host(ptr_A_host.data());
    
    ptr_B.reset(problem_count);
    ptr_B.copy_from_host(ptr_B_host.data());
    
    ptr_C.reset(problem_count);
    ptr_C.copy_from_host(ptr_C_host.data());
    
    ptr_D.reset(problem_count);
    ptr_D.copy_from_host(ptr_D_host.data());

    //
    // Initialize the problems of the workspace
    //

    for (int32_t i = 0; i < problem_count; ++i) {
      cutlass::gemm::GemmCoord problem = problem_sizes_host.at(i);

      LayoutA layout_A(lda_host.at(i));
      LayoutB layout_B(ldb_host.at(i));
      LayoutC layout_C(ldc_host.at(i));
      LayoutC layout_D(ldd_host.at(i));

      MatrixCoord extent_A{problem.m(), problem.k()};
      MatrixCoord extent_B{problem.k(), problem.n()};
      MatrixCoord extent_C{problem.m(), problem.n()};
      
      std::vector<ElementA> matrix_A(layout_A.capacity(extent_A));
      std::vector<ElementB> matrix_B(layout_B.capacity(extent_B));
      std::vector<ElementC> matrix_C(layout_C.capacity(extent_C));
      std::vector<ElementC> matrix_D(layout_D.capacity(extent_C));

      initialize_tensor(cutlass::TensorView<ElementA, LayoutA>(matrix_A.data(), layout_A, extent_A), init_A, seed * 2021);
      initialize_tensor(cutlass::TensorView<ElementB, LayoutB>(matrix_B.data(), layout_B, extent_B), init_B, seed * 2022);
      initialize_tensor(cutlass::TensorView<ElementC, LayoutC>(matrix_C.data(), layout_C, extent_C), init_C, seed * 2023);

      cutlass::device_memory::copy_to_device(ptr_A_host.at(i), matrix_A.data(), matrix_A.size());
      cutlass::device_memory::copy_to_device(ptr_B_host.at(i), matrix_B.data(), matrix_B.size());
      cutlass::device_memory::copy_to_device(ptr_C_host.at(i), matrix_C.data(), matrix_C.size());
      cutlass::device_memory::copy_to_device(ptr_D_host.at(i), matrix_D.data(), matrix_D.size());
    }
  }

  /// Verifies the result is a GEMM
  bool verify(
    ElementCompute alpha, 
    ElementCompute beta) {

    bool passed = true;

    for (int32_t i = 0; i < problem_count; ++i) {
      cutlass::gemm::GemmCoord problem = problem_sizes_host.at(i);

      LayoutA layout_A(lda_host.at(i));
      LayoutB layout_B(ldb_host.at(i));
      LayoutC layout_C(ldc_host.at(i));
      LayoutC layout_D(ldd_host.at(i));

      MatrixCoord extent_A{problem.m(), problem.k()};
      MatrixCoord extent_B{problem.k(), problem.n()};
      MatrixCoord extent_C{problem.m(), problem.n()};
      
      std::vector<ElementA> matrix_A(layout_A.capacity(extent_A));
      std::vector<ElementB> matrix_B(layout_B.capacity(extent_B));
      std::vector<ElementC> matrix_C(layout_C.capacity(extent_C));
      std::vector<ElementC> matrix_D(layout_D.capacity(extent_C));
      std::vector<ElementC> matrix_Ref(layout_D.capacity(extent_C));

      cutlass::device_memory::copy_to_host(matrix_A.data(), block_A.get() + offset_A.at(i), matrix_A.size());
      cutlass::device_memory::copy_to_host(matrix_B.data(), block_B.get() + offset_B.at(i), matrix_B.size());
      cutlass::device_memory::copy_to_host(matrix_C.data(), block_C.get() + offset_C.at(i), matrix_C.size());
      cutlass::device_memory::copy_to_host(matrix_D.data(), block_D.get() + offset_D.at(i), matrix_D.size());

      cutlass::TensorView<ElementA, LayoutA> view_A(matrix_A.data(), layout_A, extent_A);
      cutlass::TensorView<ElementB, LayoutB> view_B(matrix_B.data(), layout_B, extent_B);
      cutlass::TensorView<ElementC, LayoutC> view_C(matrix_C.data(), layout_C, extent_C);
      cutlass::TensorView<ElementC, LayoutC> view_D(matrix_D.data(), layout_D, extent_C);
      cutlass::TensorView<ElementC, LayoutC> view_Ref(matrix_Ref.data(), layout_D, extent_C);

      // Reference GEMM
      cutlass::reference::host::GemmComplex<
          ElementA, LayoutA,
          ElementB, LayoutB,
          ElementC, LayoutC, 
          ElementCompute, ElementAccumulator
      >(
        problem,
        alpha, 
        view_A,
        Gemm::kTransformA,
        view_B,
        Gemm::kTransformB,
        beta, 
        view_C, 
        view_Ref, 
        ElementAccumulator(0)
      );

      // Ensure that no input or output is entirely zero
      EXPECT_GT(cutlass::reference::host::TensorNorm(view_A), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(view_B), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(view_C), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(view_D), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(view_Ref), 0);

      // Compare against reference
      passed = cutlass::reference::host::TensorEquals(view_D, view_Ref);

      if (!passed) {
        std::ofstream file("testbed_grouped_errors.txt");

        file
          << "problem: " << problem << "  [group: " << i << "]\n" 
          << ", alpha: " << alpha << ", beta: " << beta << "\n\n";

        file 
          << "A =\n" << view_A
          << "\nB =\n" << view_B
          << "\nC =\n" << view_C
          << "\n\nReference =\n" << view_Ref
          << "\nComputed =\n" << view_D;

        return passed;
      }
    }

    return passed;
  }

  /// Executes one test
  bool run(
    int problem_count,
    ElementCompute alpha = ElementCompute(1), 
    ElementCompute beta = ElementCompute(0)) {

    this->problem_count = problem_count;

    // Initialize the problem
    initialize();

    int threadblock_count = Gemm::sufficient(problem_sizes_host.data(), problem_count);

    // Early exit
    if (!threadblock_count) {
      if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient CUDA device resources." << std::endl;
      }
      return true;
    }

    // Configure the GEMM arguments
    typename EpilogueOutputOp::Params epilogue_op(alpha, beta);

    // Configure GEMM arguments
    typename Gemm::Arguments args(
      problem_sizes_device.get(),
      problem_count,
      threadblock_count,
      epilogue_op,
      ptr_A.get(),
      ptr_B.get(),
      ptr_C.get(),
      ptr_D.get(),
      lda.get(),
      ldb.get(),
      ldc.get(),
      ldd.get(),
      problem_sizes_host.data()
    );

    // Initialize the GEMM object
    Gemm gemm;

    size_t workspace_size = gemm.get_workspace_size(args);
    cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm.initialize(args, workspace.get());

    if (status != cutlass::Status::kSuccess) {
      return false;
    }

    // Run the GEMM object
    status = gemm.run();

    if (status != cutlass::Status::kSuccess) {
      return false;
    }

    // Wait for completion
    cudaError_t result = cudaDeviceSynchronize();

    EXPECT_EQ(result, cudaSuccess) 
      << "Kernel execution error: " << cudaGetErrorString(result);

    if (result != cudaSuccess) {
      return false;
    }

    // Verify correctness
    return verify(alpha, beta);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // device
} // gemm
} // test

/////////////////////////////////////////////////////////////////////////////////////////////////
