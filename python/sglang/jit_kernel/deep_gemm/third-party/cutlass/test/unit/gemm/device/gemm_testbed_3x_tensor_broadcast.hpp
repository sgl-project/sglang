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
    \brief Tests for device-wide GEMM interface with elementwise tensor-tensor broadcast epilogue
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "../../common/cutlass_unit_test.h"

#include "testbed_utils.h"
#include "gemm_testbed_3x.hpp"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
struct Testbed3xTensorBroadcast {

  using TestBedImpl = typename detail::TestbedImpl<Gemm>;
  using Kernel      = typename Gemm::GemmKernel;
  using Epilogue    = typename Gemm::GemmKernel::CollectiveEpilogue;

  using ElementA = typename Kernel::ElementA;
  using StrideA  = typename Kernel::StrideA;
  using ElementB = typename Kernel::ElementB;
  using StrideB  = typename Kernel::StrideB;
  using ElementC = typename Kernel::ElementC;
  using StrideC  = typename Kernel::StrideC;
  using ElementD = typename Kernel::ElementD;
  using StrideD  = typename Kernel::StrideD;

  using ElementAccumulator   = typename Kernel::ElementAccumulator;
  using ElementCompute       = typename Epilogue::ElementCompute;
  using ElementScalar        = typename Epilogue::ElementScalar;
  using ProblemShapeType     = typename Kernel::ProblemShape;
  using ElementBias          = typename Epilogue::ElementBias;
  using ActivationFunctor    = typename Epilogue::ActivationFunctor;

  static constexpr bool IsBinaryOp0Enabled = Epilogue::IsBinaryOp0Enabled;
  static constexpr bool IsBinaryOp1Enabled = Epilogue::IsBinaryOp1Enabled;
  static constexpr bool IsUnaryOpEnabled   = Epilogue::IsUnaryOpEnabled;

  static constexpr bool PerColBias = Epilogue::PerColumnBias;

  using LayoutTagA = typename TestBedImpl::LayoutTagA;
  using LayoutTagB = typename TestBedImpl::LayoutTagB;
  using LayoutTagC = typename TestBedImpl::LayoutTagC;
  using LayoutTagD = typename TestBedImpl::LayoutTagD;
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;

  cutlass::HostTensor<ElementBias, LayoutTagVector> bias;
  cutlass::HostTensor<ElementC, LayoutTagC> tensor_C1;
  // tensor_C0 is taken from TestbedImpl's tensor_C


  // Detail Implementation
  TestBedImpl impl_;

  //
  // Methods
  //
  Testbed3xTensorBroadcast(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = TestBedImpl::kDefaultSeed
  ) :
    impl_(CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED,
          init_A_, init_B_, init_C_, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform, seed_) { }

  Testbed3xTensorBroadcast(
    typename LayoutTagA::Stride stride_factor_A_,
    typename LayoutTagB::Stride stride_factor_B_,
    typename LayoutTagC::Stride stride_factor_C_,
    typename LayoutTagD::Stride stride_factor_D_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = TestBedImpl::kDefaultSeed
  ) :
    impl_(stride_factor_A_,
          stride_factor_B_,
          stride_factor_C_,
          stride_factor_D_,
          CheckEquality::EXACT, ScalarLoc::ON_HOST, VectorScale::ENABLED,
          init_A_,
          init_B_,
          init_C_,
          cutlass::Distribution::Uniform,
          cutlass::Distribution::Uniform,
          seed_) { }

  /// Initializes data structures
  void initialize(ProblemShapeType problem_size) {
    //
    // Allocate the GEMM workspace for A/B/C/D tensor
    //
    impl_.initialize(problem_size);
  }

  void initialize_bias(ProblemShapeType problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto bias_size = PerColBias ? cute::get<1>(problem_shape_MNKL) : cute::get<0>(problem_shape_MNKL);
    bias.resize(cutlass::Coord<1>(bias_size));

    EXPECT_TRUE(detail::initialize_tensor(bias.host_view(), cutlass::Distribution::Uniform, impl_.collective_mma_inputs.seed + 2023));
    bias.sync_device();
  }

  void initialize_c1(ProblemShapeType problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::get<0>(problem_shape_MNKL);
    auto N = cute::get<1>(problem_shape_MNKL);
    auto L = cute::get<3>(problem_shape_MNKL);

    auto c_coord = cutlass::make_Coord(M * L, N);

    tensor_C1.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, impl_.collective_epilogue.stride_factor_C));
    EXPECT_TRUE(detail::initialize_tensor(tensor_C1.host_view(), cutlass::Distribution::Uniform, impl_.collective_mma_inputs.seed + 2024));
    tensor_C1.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL,
      ElementScalar alpha,
      ElementScalar beta,
      bool use_bias)
  {
    auto [M, N, K, L] = problem_shape_MNKL;

    impl_.collective_epilogue.tensor_D.sync_host();
    EXPECT_GT(cutlass::reference::host::TensorNorm(impl_.collective_mma_inputs.tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(impl_.collective_mma_inputs.tensor_B.host_view()), 0);

    if (impl_.collective_epilogue.tensor_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(impl_.collective_epilogue.tensor_D.host_view()), 0);
    }

    if (impl_.collective_epilogue.reference_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(impl_.collective_epilogue.reference_D.host_view()), 0);
    }

    bool passed = cutlass::reference::host::TensorEquals(impl_.collective_epilogue.reference_D.host_view(), impl_.collective_epilogue.tensor_D.host_view());

    EXPECT_TRUE(passed);

    if (!passed) {
      std::stringstream fname;
      fname << "error_Gemm_device_broadcast"
        << M << "x" << N << "x" << K << "x" << L << "_"
        << cute::get<0>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<1>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<2>(typename Gemm::GemmKernel::TileShape{}) << ".txt";

      std::ofstream file(fname.str());
      file
        << "problem: " << ' ' << M << "x" << N << "x" << K << ", Batch count = " << L
        << ", alpha: " << float(alpha) << ", beta: " << float(beta) << ", use_bias: " << use_bias 
        << ", per-col bias: " << PerColBias << "\n\n";

      if (use_bias){
        file << "Bias = \n" << bias.host_view()<< "\n\n";
      }

      file
        << "A =\n" << impl_.collective_mma_inputs.tensor_A.host_view()
        << "\nB =\n" << impl_.collective_mma_inputs.tensor_B.host_view()
        << "\nC0 =\n" << impl_.collective_epilogue.tensor_C.host_view()
        << "\nC1 =\n" << tensor_C1.host_view()
        << "\n\nReference =\n" << impl_.collective_epilogue.reference_D.host_view()
        << "\n\nComputed =\n" <<impl_.collective_epilogue.tensor_D.host_view();
    }

    return passed;
  }

  /// Verifies the result matches the GEMM with elementwise tensor-tensor
  /// broadcast operation
  bool verify(
    ProblemShapeType problem_size,
    ElementScalar alpha,
    ElementScalar beta,
    bool use_bias)
  {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::get<0>(problem_shape_MNKL);
    auto N = cute::get<1>(problem_shape_MNKL);
    auto K = cute::get<2>(problem_shape_MNKL);
    auto L = cute::get<3>(problem_shape_MNKL);

    auto A = cute::make_tensor(impl_.collective_mma_inputs.tensor_A.host_data(),
        cute::make_layout(cute::make_shape(M, K, L), impl_.collective_mma_inputs.stride_a));
    auto B = cute::make_tensor(impl_.collective_mma_inputs.tensor_B.host_data(),
        cute::make_layout(cute::make_shape(N, K, L), impl_.collective_mma_inputs.stride_b));
    auto D = cute::make_tensor(impl_.collective_epilogue.reference_D.host_data(),
        cute::make_layout(cute::make_shape(M, N, L), impl_.collective_epilogue.stride_d));
    auto Bias = cute::make_tensor(static_cast<ElementBias*>(use_bias ? bias.host_data() : nullptr),
        cute::make_layout(PerColBias ? cute::make_shape(1, N) : cute::make_shape(M, 1)));
    auto C0 = cute::make_tensor(impl_.collective_epilogue.tensor_C.host_data(),
        cute::make_layout(cute::make_shape(M, N, L), impl_.collective_epilogue.stride_c));
    auto C1 = cute::make_tensor(tensor_C1.host_data(),
        cute::make_layout(cute::make_shape(M, N, L), impl_.collective_epilogue.stride_c));

    // Create host workspace for output of testbed. This computes a portion of the epilogue:
    //    ref_compute_out = Activation(alpha * (A @ B) + bias)
    cutlass::HostTensor<ElementCompute, LayoutTagC> ref_compute_out;
    auto c_coord = cutlass::make_Coord(M * L, N);
    ref_compute_out.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, impl_.collective_epilogue.stride_factor_C), false);
    auto RefComputeOut = cute::make_tensor(ref_compute_out.host_data(),
        cute::make_layout(cute::make_shape(M, N, L), impl_.collective_epilogue.stride_c));

    cutlass::reference::host::GettMainloopParams<ElementAccumulator, decltype(A), decltype(B)> mainloop_params{A, B};

    // Use a dummy null tensor for operand C because the epilogue overrides C.
    auto dummy_C = cute::make_tensor(static_cast<ElementC*>(nullptr),
        cute::make_layout(cute::make_shape(M, N, L), impl_.collective_epilogue.stride_c));
    ElementCompute dummy_beta(0);
    auto dummy_Aux = cute::make_tensor(static_cast<ElementD*>(nullptr),
        cute::make_layout(cute::make_shape(M, N, L), impl_.collective_epilogue.stride_d));
    auto dummy_Valpha = cute::make_tensor(static_cast<ElementCompute*>(nullptr),
        cute::make_layout(cute::make_shape(M, N, 1), cute::make_stride(cute::_1{}, cute::_0{}, M)));
    auto dummy_Vbeta = cute::make_tensor(static_cast<ElementCompute*>(nullptr),
        cute::make_layout(cute::make_shape(M, N, 1), cute::make_stride(cute::_1{}, cute::_0{}, M)));
    
    auto dummy_SFD = cute::make_tensor(static_cast<ElementD*>(nullptr),
        cute::make_layout(cute::make_shape(M, N, L), impl_.collective_epilogue.stride_c));
    using DummySFDVectorSize = cute::Int<0>;
    

    cutlass::reference::host::GettEpilogueParams<
        ElementScalar,
        ElementScalar,
        ElementAccumulator,
        ElementCompute,
        decltype(dummy_C),
        decltype(RefComputeOut),
        decltype(Bias),
        decltype(dummy_Aux),      
        decltype(dummy_Valpha),
        decltype(dummy_Vbeta),
        ActivationFunctor,
        decltype(dummy_SFD),            
        DummySFDVectorSize,             
        cutlass::plus<ElementCompute>,
        PerColBias> epilogue_params{
          alpha,
          dummy_beta,
          dummy_C,
          RefComputeOut,
          Bias,
          dummy_Aux,
          dummy_Valpha,
          dummy_Vbeta
        };

    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    cutlass::NumericConverter<ElementCompute, ElementC, Epilogue::ThreadEpilogueOp::kRound> source_converter;
    cutlass::NumericConverter<ElementD, ElementCompute, Epilogue::ThreadEpilogueOp::kRound> destination_converter;
    cutlass::multiplies<ElementCompute> mul;

    // Compute broadcast operations atop the reference
    #pragma omp parallel for collapse(3)
    for (int64_t l = 0; l < cute::size<2>(A.layout()); ++l) {
      for (int64_t m = 0; m < cute::size<0>(A.layout()); ++m) {
        for (int64_t n = 0; n < cute::size<0>(B.layout()); ++n) {
          ElementCompute intermediate = RefComputeOut(m, n, l);
          // Apply BinaryOp0, if needed
          if constexpr (IsBinaryOp0Enabled) {
            typename Epilogue::ThreadEpilogueOp::BinaryOp0 bin0;
            ElementCompute converted_source = source_converter(C0(m, n, l));
            intermediate = bin0(intermediate, mul(beta, converted_source));
          }

          // Apply BinaryOp1, if needed
          if constexpr (IsBinaryOp1Enabled) {
            typename Epilogue::ThreadEpilogueOp::BinaryOp1 bin1;
            ElementCompute converted_source = source_converter(C1(m, n, l));
            intermediate = bin1(intermediate, mul(beta, converted_source));
          }

          // Apply UnaryOp, if needed
          if constexpr (IsUnaryOpEnabled) {
            typename Epilogue::ThreadEpilogueOp::UnaryOp unary;
            intermediate = unary(intermediate);
          }

          D(m, n, l) = destination_converter(intermediate);
        }
      }
    }

    return compare_reference(problem_shape_MNKL, alpha, beta, use_bias);
  }

  /// Executes one test
  bool run(
      ProblemShapeType problem_size,
      ElementScalar alpha = ElementScalar(1),
      ElementScalar beta = ElementScalar(0),
      bool profiling = false,
      int iterations = 20,
      bool use_bias = true)
  {
    // Fail test if insufficient CUDA device
    if (!impl_.sufficient()) {
      std::cout << "Test failed due to insufficient CUDA device." << std::endl;
      return false;
    }
    //
    // Initialize the GEMM operator
    //

    typename Gemm::Arguments arguments;
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    if (not profiling) {
      impl_.sm_count = std::min(impl_.MaxSmCount, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id));
      hw_info.sm_count = impl_.sm_count;
    }
    else {
      impl_.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
      hw_info.sm_count = impl_.sm_count;
    }

    /// Initializes data structures
    /// A/B/C0/D Tensor
    initialize(problem_size);
    initialize_bias(problem_size);

    if constexpr (IsBinaryOp1Enabled) {
      initialize_c1(problem_size);
    }

    arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        { impl_.collective_mma_inputs.tensor_A.device_data(), impl_.collective_mma_inputs.stride_a,
          impl_.collective_mma_inputs.tensor_B.device_data(), impl_.collective_mma_inputs.stride_b,
          impl_.mma_promotion_interval
        },
        { // Epilogue arguments
          { alpha, beta }, // ThreadOp arguments
          impl_.collective_epilogue.stride_c,
          impl_.collective_epilogue.tensor_D.device_data(),
          impl_.collective_epilogue.stride_d,
          use_bias ? bias.device_data() : nullptr,
          impl_.collective_epilogue.tensor_C.device_data(),
          tensor_C1.device_data()
        }, // Epilogue arguments end
        hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);

    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
      return true;
    }

    //
    // Run the GEMM
    //

    if (profiling) {
      return impl_.profile(problem_size, iterations, gemm_op, arguments, workspace);
    }
    else {
      cudaError_t result;
      status = gemm_op.initialize(arguments, workspace.get());
      status = gemm_op.run();
      result = cudaDeviceSynchronize();
      if (result != cudaSuccess) {
        EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
        return false;
      }

      EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

      //
      // Verify
      //
      bool passed = this->verify(problem_size, alpha, beta, use_bias);
      if (!passed) {
        std::cout << "Error : Failed : with alpha: " << float(alpha)
                  << ", beta: " << float(beta)
                  << ", use_bias: " << use_bias
                  << "\n";
      }

      return passed;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
bool TestAllTensorBroadcast(bool use_bias=true) {
  using ElementScalar = typename Gemm::GemmKernel::CollectiveEpilogue::ElementScalar;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  int max_alignment = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);
  std::vector<int> problem_size_m = {max_alignment, 512 - 3 * max_alignment};
  std::vector<int> problem_size_n = {max_alignment, 512 - 2 * max_alignment};

  if constexpr (cute::is_same_v<typename Gemm::GemmKernel::DispatchPolicy::Schedule,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong>) {
    problem_size_m.push_back(768);
    problem_size_n.push_back(768);
  }

  constexpr int Stages = Gemm::GemmKernel::DispatchPolicy::Stages;
  constexpr int TileShapeK = cute::size<2>(typename Gemm::GemmKernel::TileShape{});

  std::vector<int> problem_size_k = {max_alignment, TileShapeK * (Stages + 1) - max_alignment};

  Testbed3xTensorBroadcast<Gemm> testbed;
  bool passed = true;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        ProblemShapeType problem_size;
        if constexpr (cute::rank(ProblemShapeType{}) == 4) {
          problem_size = ProblemShapeType{m, n, k, /* l */ 1};
        }
        else {
          problem_size = ProblemShapeType{m, n, k};
        }

        for (bool use_bias : {true, false}) {
          passed = testbed.run(
            problem_size,
            cutlass::from_real<ElementScalar>(1),
            cutlass::from_real<ElementScalar>(1),
            false,  // profiling
            20,     // iterations
            use_bias
          );

          if (!passed) {
            return false;
          }
        }
      }
    }
  }

  if constexpr (cute::rank(ProblemShapeType{}) == 4) {
    auto problem_size = ProblemShapeType{256 + max_alignment, 256 + max_alignment, 160 + max_alignment, /* l */ 3};
    passed = testbed.run(
      problem_size,
      cutlass::from_real<ElementScalar>(1),
      cutlass::from_real<ElementScalar>(1),
      false,  // profiling
      20      // iterations
    );
    if (!passed) {
      return false;
    }
  }
  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
