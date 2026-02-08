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
#include "cutlass/util/reference/host/gemm_complex.h"

#include "testbed_utils.h"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
struct GemmWithBroadcastReferenceOp {

  using OutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;

  using ElementCompute = typename OutputOp::ElementCompute;
  using ElementZ = typename OutputOp::ElementZ;
  using ElementT = typename OutputOp::ElementT;

  typename OutputOp::BinaryOp binary_op;
  typename OutputOp::ElementwiseOp elementwise_op;

  GemmWithBroadcastReferenceOp() { }

  void operator()(ElementZ &Z, ElementT &T, ElementCompute gemm, ElementCompute bias) {

    ElementCompute t_full = binary_op(gemm, bias);

    if (OutputOp::kStoreT) {
      T = ElementT(t_full);
    }

    if (OutputOp::kStoreZ) {
      ElementCompute z_full = elementwise_op(t_full);
      Z = ElementZ(z_full);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Fused testbed
//
//  Y = GEMM(AB, C)
//
//  T[i, j] = BinaryOp(Y[i, j], Broadcast[i])
//
//  Z[i, j] = Elementwise(T[i, j])
//

template <
  typename Gemm, 
  typename ReferenceOp = GemmWithBroadcastReferenceOp<Gemm>
>
struct TestbedGemmWithBroadcast {

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using OutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename OutputOp::ElementCompute;
  using ElementVector = typename OutputOp::ElementVector;
  using ElementZ = typename OutputOp::ElementZ;
  using ElementT = typename OutputOp::ElementT;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;          // Input A
  cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;          // Input B
  cutlass::HostTensor<ElementC, typename Gemm::LayoutC> tensor_C;                         // Input C
  cutlass::HostTensor<ElementVector, typename Gemm::LayoutC> tensor_Broadcast;            // Input Broadcast

  cutlass::HostTensor<ElementZ, typename Gemm::LayoutC> tensor_Z;
  cutlass::HostTensor<ElementT, typename Gemm::LayoutC> tensor_T;

  cutlass::HostTensor<ElementAccumulator, typename Gemm::LayoutC> tensor_C_ref;
  cutlass::HostTensor<ElementAccumulator, typename Gemm::LayoutC> tensor_Y_ref;
  cutlass::HostTensor<ElementZ, typename Gemm::LayoutC> tensor_Z_ref;
  cutlass::HostTensor<ElementT, typename Gemm::LayoutC> tensor_T_ref;


  //
  // Methods
  //

  TestbedGemmWithBroadcast(
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

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Gemm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 1;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
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
      EXPECT_TRUE(false) << "Not implemented";
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void initialize(cutlass::gemm::GemmCoord problem_size) {
    //
    // Allocate the GEMM workspace
    //

    tensor_A.resize(problem_size.mk());
    tensor_B.resize(problem_size.kn());
    tensor_C.resize(problem_size.mn());
    tensor_Z.resize(problem_size.mn());
    tensor_T.resize(problem_size.mn());
    tensor_Broadcast.resize({
      problem_size.m(), 
      1
    });

    tensor_C_ref.resize(problem_size.mn());
    tensor_Y_ref.resize(problem_size.mn());
    tensor_Z_ref.resize(problem_size.mn());
    tensor_T_ref.resize(problem_size.mn());

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2019));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2018));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2017));
    EXPECT_TRUE(initialize_tensor(tensor_Broadcast.host_view(), init_C, seed + 2020));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = typename Gemm::ElementA(1);
    tensor_B.host_view().at({0, 0}) = typename Gemm::ElementB(1);
    tensor_C.host_view().at({0, 0}) = typename Gemm::ElementC(1);

    for (int m = 0; m < tensor_C_ref.extent().row(); ++m) {
      for (int n = 0; n < tensor_C_ref.extent().column(); ++n) {
        tensor_C_ref.at({m, n}) = ElementAccumulator(tensor_C.at({m, n}));
      }
    }

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_Broadcast.sync_device();

    tensor_Z.sync_device();
    tensor_T.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
    cutlass::gemm::GemmCoord problem_size, 
    ElementAccumulator alpha, 
    ElementAccumulator beta) {

    tensor_Z.sync_host();
    tensor_T.sync_host();

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()), 0);

    if (OutputOp::kStoreZ) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_Z.host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_Z_ref.host_view()), 0);
    }

    if (OutputOp::kStoreT) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_T.host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_T_ref.host_view()), 0);
    }

    bool passed = true;
    float norm_diff = 0;

    if (OutputOp::kStoreZ) {
      norm_diff = cutlass::reference::host::TensorNormDiff(tensor_Z_ref.host_view(), tensor_Z.host_view(), float());
      passed = (norm_diff <= 0.1f);
      EXPECT_LT(norm_diff, 0.1f) << " tensor_Z is incorrect";
    }

    if (OutputOp::kStoreT) {

      norm_diff = cutlass::reference::host::TensorNormDiff(tensor_T_ref.host_view(), tensor_T.host_view(), float());
      passed = (passed && (norm_diff <= 0.1f));

      EXPECT_LT(norm_diff, 0.1f) << " tensor_T is incorrect"; 
    }


    if (!passed) {

      /*
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
      */

      std::ofstream file("errors_testbed_gemm_with_broadcast.txt");


      file
        << "problem: " << problem_size 
        << ", alpha: " << alpha << ", beta: " << beta << "\n\n";

      file 
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\nZ =\n" << tensor_Z.host_view()
        << "\nT =\n" << tensor_T.host_view()
        << "\n\n"
        << "\nY_ref =\n" << tensor_Y_ref.host_view()
        << "\nZ_ref =\n" << tensor_Z_ref.host_view()
        << "\nT_ref =\n" << tensor_T_ref.host_view();
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
    cutlass::gemm::GemmCoord problem_size, 
    ElementAccumulator alpha, 
    ElementAccumulator beta) {

    //
    // Verify
    //

    cutlass::reference::host::GemmComplex<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB,
        ElementAccumulator, typename Gemm::LayoutC, 
        ElementAccumulator, ElementAccumulator
    >(
      problem_size,
      alpha, 
      tensor_A.host_ref(),
      Gemm::kTransformA,
      tensor_B.host_ref(),
      Gemm::kTransformB,
      beta, 
      tensor_C_ref.host_ref(), 
      tensor_Y_ref.host_ref(), 
      ElementAccumulator(0)
    );

    using ElementC = typename Gemm::ElementC;

    ReferenceOp reference_op;

    // compute tensor Z and tensor T
    for (int m = 0; m < problem_size.m(); ++m) {
      for (int n = 0; n < problem_size.n(); ++n) {

        ElementZ z;
        ElementT t;

        reference_op(z, t, tensor_Y_ref.at({m, n}), tensor_Broadcast.at({m, 0}));

        if (OutputOp::kStoreZ) {
          tensor_Z_ref.at({m, n}) = z;
        }

        if (OutputOp::kStoreT) {
          tensor_T_ref.at({m, n}) = t;
        }
      }
    }

    return compare_reference(problem_size, alpha, beta);
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
    cutlass::gemm::GemmUniversalMode mode,
    cutlass::gemm::GemmCoord problem_size, 
    int batch_count = 1,
    ElementAccumulator alpha = ElementAccumulator(1), 
    ElementAccumulator beta = ElementAccumulator(0)) {

    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient CUDA device." << std::endl;
      }
      return true;
    }

    this->initialize(problem_size);

    //
    // Initialize the GEMM operator
    //

    typename Gemm::Arguments arguments{
      mode,
      problem_size,
      batch_count,
      {alpha, beta},
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_C.device_data(),
      tensor_Z.device_data(),
      tensor_Broadcast.device_data(),
      tensor_T.device_data(),
      problem_size.m() * problem_size.k(),
      problem_size.n() * problem_size.k(),
      problem_size.m() * problem_size.n(),
      problem_size.m() * problem_size.n(),
      problem_size.m(),
      problem_size.m() * problem_size.n(),
      tensor_A.layout().stride(0),
      tensor_B.layout().stride(0),
      tensor_C.layout().stride(0),
      tensor_Z.layout().stride(0),
      0,                                    // This must be zero
      tensor_T.layout().stride(0),
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

    bool passed = true;

    passed = this->verify(problem_size, alpha, beta);

    if (!passed) {
      std::cout << "Failed with batch_count/split_k_slices = " << batch_count << std::endl;
    }

    //
    // Profile
    //

    #if 0 // profiling disabled for now.

    int const kWorkspaces = 100;

    cutlass::DeviceAllocation<typename Gemm::ElementA> profiling_tensor_A(tensor_A.capacity() * kWorkspaces);
    cutlass::DeviceAllocation<typename Gemm::ElementB> profiling_tensor_B(tensor_B.capacity() * kWorkspaces);
    cutlass::DeviceAllocation<ElementC> profiling_tensor_C(tensor_C.capacity() * kWorkspaces);
    cutlass::DeviceAllocation<ElementC> profiling_tensor_Broadcast(tensor_Broadcast.capacity() * kWorkspaces);
    cutlass::DeviceAllocation<ElementZ> profiling_tensor_Z(tensor_Z.capacity() * kWorkspaces);
    cutlass::DeviceAllocation<ElementT> profiling_tensor_T(tensor_T.capacity() * kWorkspaces);

    cudaEvent_t events[2];
    for (auto & event : events) {
      cudaError_t result = cudaEventCreate(&event);
      if (result != cudaSuccess) {
        EXPECT_EQ(result, cudaSuccess) << " cudaEventCreate() failed with error " << cudaGetErrorString(result);
        return false;
        break;
      }
    }

    int const kWarmupIterations = 5;
    int const kProfilingIterations = 100;

    for (int i = 0; i < kWarmupIterations; ++i) {
      status = gemm_op();
      EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);
    }
    

    cudaError_t result = cudaEventRecord(events[0]);
    EXPECT_EQ(result, cudaSuccess);

    for (int i = 0; i < kProfilingIterations; ++i) {

      typename Gemm::Arguments arguments{
        mode,
        problem_size,
        batch_count,
        {alpha, beta},
        profiling_tensor_A.get() + tensor_A.capacity() * (i % kWorkspaces),
        profiling_tensor_B.get() + tensor_B.capacity() * (i % kWorkspaces),
        profiling_tensor_C.get() + tensor_C.capacity() * (i % kWorkspaces),
        profiling_tensor_Z.get() + tensor_Z.capacity() * (i % kWorkspaces),
        profiling_tensor_Broadcast.get() + tensor_Broadcast.capacity() * (i % kWorkspaces),
        profiling_tensor_T.get() + tensor_T.capacity() * (i % kWorkspaces),
        problem_size.m() * problem_size.k(),
        problem_size.n() * problem_size.k(),
        problem_size.m() * problem_size.n(),
        problem_size.m() * problem_size.n(),
        problem_size.m(),
        problem_size.m() * problem_size.n(),
        tensor_A.layout().stride(0),
        tensor_B.layout().stride(0),
        tensor_C.layout().stride(0),
        tensor_Z.layout().stride(0),
        0,                                    // This must be zero
        tensor_T.layout().stride(0),
      };

      gemm_op.initialize(arguments, workspace.get());
      status = gemm_op();
      EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);
    }

    result = cudaEventRecord(events[1]);
    EXPECT_EQ(result, cudaSuccess);

    result = cudaDeviceSynchronize();
    EXPECT_EQ(result, cudaSuccess);

    float elapsed_time = 0;
    result = cudaEventElapsedTime(&elapsed_time, events[0], events[1]);
    EXPECT_EQ(result, cudaSuccess);

    double average_time = double(elapsed_time) / double(kProfilingIterations);

    std::cout << problem_size << ": " << average_time << " ms" << std::endl;

    for (auto & event : events) {
      cudaEventDestroy(event);
    }
    #endif

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Gemm, 
  typename ReferenceOp = GemmWithBroadcastReferenceOp<Gemm>
>
bool TestGemmWithBroadcast(
  cutlass::gemm::GemmCoord const & problem_size,
  cutlass::gemm::GemmUniversalMode mode,
  int batch_count,
  double alpha = 1.0, 
  double beta = 2.0) {

  bool passed = true;

  TestbedGemmWithBroadcast<Gemm, ReferenceOp> testbed;
  
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  passed = testbed.run(
    mode,
    problem_size, 
    batch_count,
    cutlass::from_real<ElementAccumulator>(alpha), 
    cutlass::from_real<ElementAccumulator>(beta)
  );

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Gemm, 
  typename ReferenceOp = GemmWithBroadcastReferenceOp<Gemm>
>
bool TestAllGemmWithBroadcast() {

  int M_problems[] = {8, 136, 264, 520};
  int N_problems[] = {8, 136, 264, 520};
  int K_problems[] = {8, 136, 264, 520};
  double alpha_problems[] = {1.25, 2.25};
  double beta_problems[] = {0, 1, 2.0};

  bool passed = true;

  for (int M : M_problems) {
    for (int N : N_problems) {
      for (int K : K_problems) {
        for (double alpha : alpha_problems) {
          for (double beta : beta_problems) {

            TestbedGemmWithBroadcast<Gemm, ReferenceOp> testbed;
            
            using ElementAccumulator = typename Gemm::ElementAccumulator;

            passed = testbed.run(
              cutlass::gemm::GemmUniversalMode::kGemm,
              {M, N, K}, 
              1,
              cutlass::from_real<ElementAccumulator>(alpha), 
              cutlass::from_real<ElementAccumulator>(beta)
            );

            EXPECT_TRUE(passed) 
              << "M: " << M << ", N: " << N << ", K: " << K << ", alpha: " << alpha << ", beta: " << beta;

            if (!passed) {

              return passed;
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

