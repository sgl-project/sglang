/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Testbed for running device-level GEMMs with absolute maximum calculation and scaling
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed.h"
#include "testbed_sparse.h"
#include "testbed_utils.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Gemm,
  typename GemmTestbed,
  template<typename T> class ActivationFunctor
>
struct TestbedWithAmax {

  static_assert(std::is_same_v<GemmTestbed, Testbed<Gemm>> || std::is_same_v<GemmTestbed, SparseTestbed<Gemm>>);
  static constexpr bool IsSparseTestbed = std::is_same_v<GemmTestbed, SparseTestbed<Gemm>>;

  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;
  using ElementScalingFactor = typename Gemm::EpilogueOutputOp::ElementScalingFactor;
  using ElementAbsmax = typename Gemm::EpilogueOutputOp::ElementAbsmax;

  static bool const kScaleAux = Gemm::EpilogueOutputOp::kIsScalingAndAmaxAuxOutputNeeded;
  static bool const kScaleOutput = Gemm::EpilogueOutputOp::kIsScalingAndAmaxOutputNeeded;
  bool doScaleA;
  bool doScaleB;
  bool doScaleC;

  GemmTestbed underlying_testbed;

  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementAuxOutput, typename Gemm::LayoutC> tensor_Aux;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_Vector;
  cutlass::HostTensor<ElementAccumulator, typename Gemm::LayoutC> tmp_D;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, typename Gemm::LayoutC> reference_D;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementAuxOutput, typename Gemm::LayoutC> reference_Aux;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_A;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_B;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_C;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_D;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_Aux;
  cutlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> abs_max_Aux;
  cutlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> abs_max_D;
  cutlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> reference_abs_max_Aux;
  cutlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> reference_abs_max_D;

  //
  // Methods
  //

  TestbedWithAmax(
    bool scaleA = true,
    bool scaleB = true,
    bool scaleC = true,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform
  ):
    doScaleA(scaleA), doScaleB(scaleB), doScaleC(scaleC),
    underlying_testbed(init_A_, init_B_, init_C_) { }

  /// Helper to initialize scaling factors
  template <typename Element, typename Layout>
  bool initialize_scale_factor(cutlass::TensorView<Element, Layout> view, uint64_t seed, int bits=0) {
    cutlass::reference::host::TensorFillRandomUniform(view, seed, double(1.), double(0.), bits);
    return true;
  }

  /// Initializes data structures
  void initialize(cutlass::gemm::GemmCoord problem_size) {
    //
    // Allocate the GEMM workspace
    //
    underlying_testbed.initialize(problem_size);

    tensor_Vector.resize({1, problem_size.n()});
    reference_D.resize(problem_size.mn(), false);
    tmp_D.resize(problem_size.mn(), false);

    EXPECT_TRUE(
      underlying_testbed.initialize_tensor(tensor_Vector.host_view(), underlying_testbed.init_C, underlying_testbed.seed + 2020)
    );

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    cutlass::Coord<2> origin(0);
    tensor_Vector.host_view().at(origin) = typename Gemm::ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), underlying_testbed.tensor_C.host_view());

    tensor_Vector.sync_device();

    int scale_bits = 2;
    if (doScaleA) {
      scale_A.resize({1, 1});
      EXPECT_TRUE(initialize_scale_factor(scale_A.host_view(), underlying_testbed.seed + 2021, scale_bits));
      scale_A.sync_device();
    }

    if (doScaleB) {
      scale_B.resize({1, 1});
      EXPECT_TRUE(initialize_scale_factor(scale_B.host_view(), underlying_testbed.seed + 2022, scale_bits));
      scale_B.sync_device();
    }

    if (doScaleC) {
      scale_C.resize({1, 1});
      EXPECT_TRUE(initialize_scale_factor(scale_C.host_view(), underlying_testbed.seed + 2023, scale_bits));
      scale_C.sync_device();
    }

    if (kScaleOutput) {
      scale_D.resize({1, 1});
      EXPECT_TRUE(initialize_scale_factor(scale_D.host_view(), underlying_testbed.seed + 2024, scale_bits));
      scale_D.sync_device();

      abs_max_D.resize({1, 1});
      cutlass::reference::host::TensorFill(abs_max_D.host_view());
      abs_max_D.sync_device();

      reference_abs_max_D.resize({1, 1});
    }

    if (kScaleAux) {
      tensor_Aux.resize(problem_size.mn());
      cutlass::reference::host::TensorFill(tensor_Aux.host_view());
      tensor_Aux.sync_device();

      scale_Aux.resize({1, 1});
      EXPECT_TRUE(initialize_scale_factor(scale_Aux.host_view(), underlying_testbed.seed + 2025, scale_bits));
      scale_Aux.sync_device();

      abs_max_Aux.resize({1, 1});
      cutlass::reference::host::TensorFill(abs_max_Aux.host_view());
      abs_max_Aux.sync_device();

      reference_Aux.resize(problem_size.mn(), false);
      reference_abs_max_Aux.resize({1, 1});
    }
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
    cutlass::gemm::GemmCoord problem_size,
    ElementCompute alpha,
    ElementCompute beta) {

    underlying_testbed.tensor_D.sync_host();

    EXPECT_GT(cutlass::reference::host::TensorNorm(underlying_testbed.tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(underlying_testbed.tensor_B.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(underlying_testbed.tensor_C.host_view()), 0);

    EXPECT_GT(cutlass::reference::host::TensorNorm(underlying_testbed.tensor_D.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);
    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), underlying_testbed.tensor_D.host_view());
    if (!passed) {
      std::cout << "Comparison of D failed" << std::endl;
    }

    if (kScaleAux) {
      tensor_Aux.sync_host();
      abs_max_Aux.sync_host();
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_Aux.host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(abs_max_Aux.host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_Aux.host_view()), 0);
      if (!cutlass::reference::host::TensorEquals(reference_Aux.host_view(), tensor_Aux.host_view())) {
        passed = false;
        std::cout << "Comparison of Aux failed" << std::endl;
      }
      if (!cutlass::reference::host::TensorEquals(abs_max_Aux.host_view(), reference_abs_max_Aux.host_view())) {
        passed = false;
        std::cout << "Comparison of Aux absmax failed" << std::endl;
      }
    }

    if (kScaleOutput) {
      abs_max_D.sync_host();
      EXPECT_GT(cutlass::reference::host::TensorNorm(abs_max_D.host_view()), 0);
      if (!cutlass::reference::host::TensorEquals(abs_max_D.host_view(), reference_abs_max_D.host_view())) {
        passed = false;
        std::cout << "Comparison of D absmax failed" << std::endl;
      }
    }

    EXPECT_TRUE(passed) << " mismatched reference";

    if (!passed) {

      std::ofstream file("testbed_with_amax_errors.txt");

      file
        << "problem: " << problem_size
        << ", alpha: " << alpha << ", beta: " << beta << "\n\n";

      file
        << "A =\n" << underlying_testbed.tensor_A.host_view()
        << "\nB =\n" << underlying_testbed.tensor_B.host_view()
        << "\nC =\n" << underlying_testbed.tensor_C.host_view()
        << "\nVector =\n" << tensor_Vector.host_view()
        << "\nScaleA = " << scale_A.host_view()
        << "\nScaleB = " << scale_B.host_view()
        << "\nScaleC = " << scale_C.host_view()
        << "\nScaleD = " << scale_D.host_view()
        << "\nScaleAux = " << scale_Aux.host_view()
        << "\n\nReference D =\n" << reference_D.host_view()
        << "\nComputed D =\n" << underlying_testbed.tensor_D.host_view();
      if (kScaleAux) {
        file
          << "\n\nReference Aux =\n" << reference_Aux.host_view()
          << "\nComputed Aux =\n" << tensor_Aux.host_view()
          << "\n\nReference Absmax Aux = " << reference_abs_max_Aux.host_view()
          << "\nComputed Absmax Aux = " << abs_max_Aux.host_view();
      }
      if (kScaleOutput) {
        file
          << "\n\nReference Absmax D = " << reference_abs_max_D.host_view()
          << "\nComputed Absmax D = " << abs_max_D.host_view();
      }
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
    cutlass::gemm::GemmCoord problem_size,
    ElementCompute alpha,
    ElementCompute beta) {

    cutlass::Coord<2> origin(0);
    ElementCompute scaled_alpha = alpha;
    if (doScaleA) {
      scaled_alpha *= scale_A.host_view().at(origin);
    }
    if (doScaleB) {
      scaled_alpha *= scale_B.host_view().at(origin);
    }

    ElementCompute scaled_beta = beta;
    if (doScaleC) {
      scaled_beta *= scale_C.host_view().at(origin);
    }

    //
    // Verify
    //

    auto ref_tA = [&](){
      if constexpr (IsSparseTestbed) {
        cutlass::uncompress(
          underlying_testbed.tensor_A_uncompressed.host_ref(),
          underlying_testbed.tensor_A.host_ref(),
          underlying_testbed.tensor_E.host_ref(),
          problem_size.m(),
          problem_size.k()
        );
        return underlying_testbed.tensor_A_uncompressed.host_ref();
      }
      else {
        return underlying_testbed.tensor_A.host_ref();
      }
    }();

    // Run reference kernel with ElementOutput of type ElementAccumulator
    // so that we can compute the absmax epilogue on data that is of type
    // ElementAccumulator (which is what the GEMM we are testing will do).
    cutlass::reference::host::GemmComplex<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB,
        typename Gemm::ElementC, typename Gemm::LayoutC,
        ElementCompute, ElementAccumulator, ElementAccumulator
    >(
      problem_size,
      scaled_alpha,
      ref_tA,
      Gemm::kTransformA,
      underlying_testbed.tensor_B.host_ref(),
      Gemm::kTransformB,
      scaled_beta,
      underlying_testbed.tensor_C.host_ref(),
      tmp_D.host_ref(),
      ElementAccumulator(0)
    );

    ElementCompute tmp_abs_max_Aux(0.);
    ElementCompute tmp_abs_max_D(0.);

    cutlass::NumericConverter<ElementCompute, typename Gemm::ElementC> cvt_c_to_compute;
    cutlass::NumericConverter<ElementCompute, ElementAccumulator> cvt_accum_to_compute;
    cutlass::NumericConverter<ElementAbsmax, ElementCompute> cvt_compute_to_absmax;
    cutlass::NumericConverter<typename Gemm::EpilogueOutputOp::ElementOutput, ElementCompute> cvt_compute_to_d;
    cutlass::NumericConverter<typename Gemm::EpilogueOutputOp::ElementAuxOutput, ElementCompute> cvt_compute_to_aux;

    cutlass::absolute_value_op<ElementCompute> abs;
    cutlass::maximum_with_nan_propogation<ElementCompute> max;
    ActivationFunctor<ElementCompute> act;

    ElementScalingFactor d_scale = kScaleOutput ? scale_D.host_view().at(origin) : ElementScalingFactor(1.);

    for (int m = 0; m < problem_size.m(); ++m) {
      for (int n = 0; n < problem_size.n(); ++n) {
        ElementCompute intermediate = cvt_accum_to_compute(tmp_D.host_view().at({m, n}));
        ElementCompute bias = cvt_c_to_compute(tensor_Vector.host_view().at({0, n}));
        ElementCompute aux = intermediate + bias;
        ElementCompute d = act(aux);
        tmp_abs_max_Aux = max(abs(aux), tmp_abs_max_Aux);
        tmp_abs_max_D = max(abs(d), tmp_abs_max_D);
        reference_D.host_view().at({m, n}) = cvt_compute_to_d(d * d_scale);

        if (kScaleAux) {
          reference_Aux.host_view().at({m, n}) = cvt_compute_to_aux(aux * scale_Aux.host_view().at(origin));
        }
      }
    }

    if (kScaleAux) {
      reference_abs_max_Aux.host_view().at(origin) = cvt_compute_to_absmax(tmp_abs_max_Aux);
    }

    if (kScaleOutput) {
      reference_abs_max_D.host_view().at(origin) = cvt_compute_to_absmax(tmp_abs_max_D);
    }

    return compare_reference(problem_size, alpha, beta);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //
    return underlying_testbed.sufficient();
  }

  /// Executes one test
  bool run(
    cutlass::gemm::GemmUniversalMode mode,
    cutlass::gemm::GemmCoord problem_size,
    int batch_count = 1,
    ElementCompute alpha = ElementCompute(1),
    ElementCompute beta = ElementCompute(0))
  {

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

    typename Gemm::EpilogueOutputOp::Params::ActivationParams activation_params{alpha, beta};
    typename Gemm::EpilogueOutputOp::Params epilogue_params{
      activation_params,
      scale_A.device_data(),
      scale_B.device_data(),
      scale_C.device_data(),
      scale_D.device_data(),
      scale_Aux.device_data(),
      abs_max_Aux.device_data(),
      abs_max_D.device_data()
    };

    auto arguments = [&]() {
      if constexpr (IsSparseTestbed) {
        return typename Gemm::Arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          problem_size,
          batch_count,
          epilogue_params,
          underlying_testbed.tensor_A.device_data(),
          underlying_testbed.tensor_B.device_data(),
          underlying_testbed.tensor_C.device_data(),
          underlying_testbed.tensor_D.device_data(),
          underlying_testbed.tensor_E_reordered.device_data(),
          tensor_Aux.device_data(),
          tensor_Vector.device_data(),
          int64_t(),
          int64_t(),
          int64_t(),
          int64_t(),
          int64_t(),
          int64_t(),
          int64_t(),
          underlying_testbed.tensor_A.layout().stride(0),
          underlying_testbed.tensor_B.layout().stride(0),
          underlying_testbed.tensor_C.layout().stride(0),
          underlying_testbed.tensor_D.layout().stride(0),
          underlying_testbed.tensor_E_reordered.layout().stride(0),
          tensor_Aux.layout().stride(0),
          0 // stride vector
        };
      }
      else {
        return typename Gemm::Arguments{
          mode,
          problem_size,
          batch_count,
          epilogue_params,
          underlying_testbed.tensor_A.device_data(),
          underlying_testbed.tensor_B.device_data(),
          underlying_testbed.tensor_C.device_data(),
          underlying_testbed.tensor_D.device_data(),
          tensor_Aux.device_data(),
          tensor_Vector.device_data(),
          problem_size.m() * problem_size.k(),
          problem_size.n() * problem_size.k(),
          problem_size.m() * problem_size.n(),
          problem_size.m() * problem_size.n(),
          0, // stride vector
          underlying_testbed.tensor_A.layout().stride(0),
          underlying_testbed.tensor_B.layout().stride(0),
          underlying_testbed.tensor_C.layout().stride(0),
          underlying_testbed.tensor_D.layout().stride(0),
          (int64_t)0 // Leading dimension of vector. This must be 0
        };
      }
    }();

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Run the GEMM
    //

    status = gemm_op();

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    cudaError_t cuda_error = cudaDeviceSynchronize();
    EXPECT_TRUE(cuda_error == cudaSuccess) << cudaGetErrorString(cuda_error);

    //
    // Verify
    //

    bool passed = this->verify(problem_size, alpha, beta);

    if (!passed) {
      std::cout << "Failed with batch_count/split_k_slices = " << batch_count << std::endl;
    }

    return passed;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Gemm,
  typename GemmTestbed,
  template<typename T> class ActivationFunctor = cutlass::epilogue::thread::Identity
>
bool TestAllGemmWithAbsmax(bool scaleA=true, bool scaleB=true, bool scaleC=true) {

  int const kMinimumOperandElementSize =
    std::min(
      int(cutlass::sizeof_bits<typename Gemm::ElementA>::value),
      int(cutlass::sizeof_bits<typename Gemm::ElementB>::value));

  int constexpr kAlignmentM = [&]() {
    if constexpr (std::is_same_v<GemmTestbed, SparseTestbed<Gemm>>) {
      // M dimension has to be multiple of 32 (sparse float) or 16 (sparse int)
      // because of the reordering of operand E
      return std::max(((sizeof(typename Gemm::ElementE) == 2) ? 32 : 16),
                                   kMinimumOperandElementSize);
    }
    else {
      return 128 / kMinimumOperandElementSize;
    }
  }();

  int const kAlignmentN = 128 / kMinimumOperandElementSize;

  int M_problems[] = {kAlignmentM, 128 + 32};
  int N_problems[] = {kAlignmentN, 512 - 2 * kAlignmentN};
  int K_problems[] = {Gemm::ThreadblockShape::kK * 2};
  double alpha_problems[] = {1.};
  double beta_problems[] = {0.};
  int split_k_slices[] = {
    1, 2
  };

  bool passed = true;

  for (int M : M_problems) {
    for (int N : N_problems) {
      for (int K : K_problems) {
        for (int split_k : split_k_slices) {
          if (cutlass::sizeof_bits_v<typename Gemm::EpilogueOutputOp::ElementOutput> <= 8 && split_k > 1) {
            // Don't test split-K with FP8 output. The kernel being tested will writie partial accumulations
            // for different splits to global memory in FP8, while the reference kernel will not. This leads
            // to mismatches that are difficult to capture without a permissive relative equality check threshold.
            continue;
          }

          for (double alpha : alpha_problems) {
            for (double beta : beta_problems) {
              TestbedWithAmax<Gemm, GemmTestbed, ActivationFunctor> testbed(scaleA, scaleB, scaleC);

              using ElementAccumulator = typename Gemm::ElementAccumulator;

              passed = testbed.run(
                cutlass::gemm::GemmUniversalMode::kGemm,
                {M, N, K},
                split_k,
                cutlass::from_real<ElementAccumulator>(alpha),
                cutlass::from_real<ElementAccumulator>(beta)
              );

              EXPECT_TRUE(passed)
                << "M: " << M << ", N: " << N << ", K: " << K << ", alpha: " << alpha << ", beta: " << beta << ", split_k:" << split_k;

              if (!passed) {

                return passed;
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
