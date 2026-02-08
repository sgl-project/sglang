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
    \brief Example of running an Ada FP8 GEMM.

    In addition to using FP8 Tensor Core instructions, the Ada FP8 GEMM uses a distinct epilogue
    that enables additional scaling of operands/outputs, storing a pre-activation-function output
    tensor (called the "auxiliary" output), and computing the absolute maximum value of the
    outputs.

    Pseudocode for this epilogue is as follows:

    Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias
    D = activation(Aux)

    if Aux is fp8 type:
        abs_max_output = max( abs(aux) | (for every aux in Aux))
        Aux = scale_aux * Aux
    endif

    if D is fp8 type:
        abs_max_output = max( abs(d) | (for every d in D))
        D = scale_d * D
    endif

    Parameter Aux is optionally stored to global memory
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"


using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementOutput = cutlass::float_e4m3_t;
using ElementAuxOutput = ElementOutput;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
static int const kStages = 3;
static int const kAlignmentA = 16;
static int const kAlignmentB = 16;

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::ReLu,
    ElementOutput,
    ElementAuxOutput,
    8,
    ElementAccumulator,
    ElementAccumulator
    >;

template <typename MathOperator>
using Gemm_ = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 64, 128>, cutlass::gemm::GemmShape<64, 32, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages,
    kAlignmentA, kAlignmentB, MathOperator
  >;

using ElementAbsmax = typename EpilogueOutputOp::ElementAbsmax;


// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool reference_check;
  cutlass::gemm::GemmCoord problem_size;

  int iterations;
  int warmup_iterations;

  bool scale_A;
  bool scale_B;
  bool scale_C;

  float alpha;
  float beta;

  Options():
    help(false),
    error(false),
    reference_check(false),
    iterations(20),
    warmup_iterations(5),
    scale_A(true),
    scale_B(true),
    scale_C(true),
    alpha(1.f),
    beta(0.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("warmup_iterations", warmup_iterations, 5);
    cmd.get_cmd_line_argument("reference-check", reference_check, false);
    cmd.get_cmd_line_argument("scale-A", scale_A, true);
    cmd.get_cmd_line_argument("scale-B", scale_B, true);
    cmd.get_cmd_line_argument("scale-C", scale_C, true);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);

    int m, n, k;
    cmd.get_cmd_line_argument("m", m, 1024);
    cmd.get_cmd_line_argument("n", n, 1024);
    cmd.get_cmd_line_argument("k", k, 1024);

    problem_size = cutlass::gemm::GemmCoord{m, n, k};
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "58_ada_fp8_gemm\n\n"
      << "  This example executes a GEMM using Ada FP8 Tensor Core operations. In addition to performing\n"
      << "  a normal GEMM, the kernel performs the following operations:\n"
      << "      Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias\n"
      << "        D = activation(Aux)\n\n"
      << "      if Aux is fp8:\n"
      << "         abs_max_output = max( abs(aux) | (for every aux in Aux) )\n"
      << "         Aux = scale_aux * Aux\n\n"
      << "      if D is fp8 type:\n"
      << "         abs_max_output = max( abs(d) | (for every d in D) )\n"
      << "         D = scale_d * D\n\n"
      << "Options:\n\n"
      << "  --help                           If specified, displays this usage statement\n\n"
      << "  --m=<int>                        Sets the M dimension of the GEMM\n"
      << "  --n=<int>                        Sets the N dimension of the GEMM\n"
      << "  --k=<int>                        Sets the K dimension of the GEMM\n"
      << "  --scale-A=<bool>                 Whether to apply a scaling factor to operand A (default: true)\n"
      << "  --scale-B=<bool>                 Whether to apply a scaling factor to operand B (default: true)\n"
      << "  --scale-C=<bool>                 Whether to apply a scaling factor to operand C (default: true)\n"
      << "  --iterations=<int>               Number of profiling iterations to perform\n"
      << "  --warmup-iterations=<int>        Number of warmup iterations to perform\n"
      << "  --reference-check=<bool>         If true, performs reference check\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  float gflops(float runtime_s) const {
    // Two flops per multiply-add
    return 2.0f * float(problem_size.product()) / float(1.0e9) / runtime_s;
  }
};

/// Helper class to run the kernel
template <typename Gemm>
struct TestbedRunner {

  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;
  using ElementScalingFactor = typename Gemm::EpilogueOutputOp::ElementScalingFactor;

  static bool const kScaleAux = Gemm::EpilogueOutputOp::kIsScalingAndAmaxAuxOutputNeeded;
  static bool const kScaleOutput = Gemm::EpilogueOutputOp::kIsScalingAndAmaxOutputNeeded;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementAuxOutput, typename Gemm::LayoutC> tensor_Aux;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, typename Gemm::LayoutC> tensor_D;
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

  TestbedRunner(
    bool scaleA = true,
    bool scaleB = true,
    bool scaleC = true,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  /// Helper to initialize scaling factors
  template <typename Element, typename Layout>
  bool initialize_scale_factor(cutlass::TensorView<Element, Layout> view, uint64_t seed, int bits=0) {
    cutlass::reference::host::TensorFillRandomUniform(view, seed, double(1.), double(0.), bits);
    return true;
  }

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
        scope_max = 2;
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
      std::cerr << "Not implemented";
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void initialize(const Options& options) {
    //
    // Allocate the GEMM workspace
    //

    tensor_A.resize(options.problem_size.mk());
    tensor_B.resize(options.problem_size.kn());
    tensor_C.resize(options.problem_size.mn());
    tensor_D.resize(options.problem_size.mn());
    tensor_Vector.resize({1, options.problem_size.n()});
    reference_D.resize(options.problem_size.mn(), false);
    tmp_D.resize(options.problem_size.mn(), false);

    initialize_tensor(tensor_A.host_view(), init_A, seed + 2019);
    initialize_tensor(tensor_B.host_view(), init_B, seed + 2018);
    initialize_tensor(tensor_C.host_view(), init_C, seed + 2017);
    initialize_tensor(tensor_Vector.host_view(), init_C, seed + 2020);

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    cutlass::Coord<2> origin(0);
    tensor_A.host_view().at(origin) = typename Gemm::ElementA(1);
    tensor_B.host_view().at(origin) = typename Gemm::ElementB(1);
    tensor_C.host_view().at(origin) = typename Gemm::ElementC(1);
    tensor_Vector.host_view().at(origin) = typename Gemm::ElementC(1);

    cutlass::reference::host::TensorFill(tensor_D.host_view());
    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
    tensor_Vector.sync_device();

    int scale_bits = 2;
    if (options.scale_A) {
      scale_A.resize({1, 1});
      initialize_scale_factor(scale_A.host_view(), seed + 2021, scale_bits);
      scale_A.sync_device();
    }

    if (options.scale_B) {
      scale_B.resize({1, 1});
      initialize_scale_factor(scale_B.host_view(), seed + 2022, scale_bits);
      scale_B.sync_device();
    }

    if (options.scale_C) {
      scale_C.resize({1, 1});
      initialize_scale_factor(scale_C.host_view(), seed + 2023, scale_bits);
      scale_C.sync_device();
    }

    if (kScaleOutput) {
      scale_D.resize({1, 1});
      initialize_scale_factor(scale_D.host_view(), seed + 2024, scale_bits);
      scale_D.sync_device();

      abs_max_D.resize({1, 1});
      cutlass::reference::host::TensorFill(abs_max_D.host_view());
      abs_max_D.sync_device();

      reference_abs_max_D.resize({1, 1});
    }

    if (kScaleAux) {
      tensor_Aux.resize(options.problem_size.mn());
      cutlass::reference::host::TensorFill(tensor_Aux.host_view());
      tensor_Aux.sync_device();

      scale_Aux.resize({1, 1});
      initialize_scale_factor(scale_Aux.host_view(), seed + 2025, scale_bits);
      scale_Aux.sync_device();

      abs_max_Aux.resize({1, 1});
      cutlass::reference::host::TensorFill(abs_max_Aux.host_view());
      abs_max_Aux.sync_device();

      reference_Aux.resize(options.problem_size.mn(), false);
      reference_abs_max_Aux.resize({1, 1});
    }
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(const Options& options) {

    tensor_D.sync_host();

    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    if (kScaleAux) {
      tensor_Aux.sync_host();
      abs_max_Aux.sync_host();
      passed &= cutlass::reference::host::TensorEquals(reference_Aux.host_view(), tensor_Aux.host_view());
      passed &= cutlass::reference::host::TensorEquals(abs_max_Aux.host_view(), reference_abs_max_Aux.host_view());
    }

    if (kScaleOutput) {
      abs_max_D.sync_host();
      passed &= cutlass::reference::host::TensorEquals(abs_max_D.host_view(), reference_abs_max_D.host_view());
    }

    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;

      std::string output_file = "testbed_with_amax_errors.txt";
      std::ofstream file(output_file);

      file
        << "problem: " << options.problem_size
        << ", alpha: " << options.alpha << ", beta: " << options.beta << "\n\n";

      file
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\nVector =\n" << tensor_Vector.host_view()
        << "\nScaleA = " << scale_A.host_view()
        << "\nScaleB = " << scale_B.host_view()
        << "\nScaleC = " << scale_C.host_view()
        << "\nScaleD = " << scale_D.host_view()
        << "\nScaleAux = " << scale_Aux.host_view()
        << "\n\nReference D =\n" << reference_D.host_view()
        << "\nComputed D =\n" << tensor_D.host_view();
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

      std::cerr << "Dumped results to " << output_file << std::endl;

    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(const Options& options) {

    cutlass::Coord<2> origin(0);
    ElementCompute scaled_alpha = options.alpha;
    if (options.scale_A) {
      scaled_alpha *= scale_A.host_view().at(origin);
    }
    if (options.scale_B) {
      scaled_alpha *= scale_B.host_view().at(origin);
    }

    ElementCompute scaled_beta = options.beta;
    if (options.scale_C) {
      scaled_beta *= scale_C.host_view().at(origin);
    }

    //
    // Verify
    //

    cutlass::reference::host::GemmComplex<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB,
        typename Gemm::ElementC, typename Gemm::LayoutC,
        ElementCompute, ElementAccumulator, ElementAccumulator
    >(
      options.problem_size,
      scaled_alpha,
      tensor_A.host_ref(),
      Gemm::kTransformA,
      tensor_B.host_ref(),
      Gemm::kTransformB,
      scaled_beta,
      tensor_C.host_ref(),
      tmp_D.host_ref(),
      ElementAccumulator(0)
    );

    ElementCompute tmp_abs_max_Aux(0.);
    ElementCompute tmp_abs_max_D(0.);

    cutlass::NumericConverter<ElementCompute, typename Gemm::ElementC> cvt_c_to_compute;
    cutlass::NumericConverter<ElementCompute, ElementAccumulator> cvt_accum_to_compute;
    cutlass::NumericConverter<ElementAccumulator, ElementCompute> cvt_compute_to_accum;
    cutlass::NumericConverter<typename Gemm::EpilogueOutputOp::ElementOutput, ElementCompute> cvt_compute_to_d;
    cutlass::NumericConverter<typename Gemm::EpilogueOutputOp::ElementAuxOutput, ElementCompute> cvt_compute_to_aux;

    cutlass::absolute_value_op<ElementCompute> abs;
    cutlass::maximum_with_nan_propogation<ElementCompute> max;
    cutlass::epilogue::thread::ReLu<ElementCompute> act;

    ElementScalingFactor d_scale = kScaleOutput ? scale_D.host_view().at(origin) : ElementScalingFactor(1.);

    for (int m = 0; m < options.problem_size.m(); ++m) {
      for (int n = 0; n < options.problem_size.n(); ++n) {
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
      reference_abs_max_Aux.host_view().at(origin) = cvt_compute_to_accum(tmp_abs_max_Aux);
    }

    if (kScaleOutput) {
      reference_abs_max_D.host_view().at(origin) = cvt_compute_to_accum(tmp_abs_max_D);
    }

    return compare_reference(options);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {

    if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 4)) {
      std::cerr << "This example requires CUDA 12.4 or greater." << std::endl;
      return false;
    }

    size_t smem_size = sizeof(typename Gemm::GemmKernel::SharedStorage);

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      std::cerr << "cudaGetDevice() failed with error: " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      std::cerr << "cudaGetDeviceProperties() failed with error: " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    if (properties.major < 8 || (properties.major == 8 && properties.minor < 9)) {
      std::cerr << "CUTLASS's Ada FP8 GEMM example requires a device of compute capability 89 or higher.\n" << std::endl;
      return false;
    }

    if (properties.sharedMemPerBlockOptin < smem_size) {
      std::cerr << "Insufficient shared memory. Need " << smem_size
                << ", but device only has " << properties.sharedMemPerBlockOptin << std::endl;
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(Options& options)
  {

    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      std::cerr << "Insufficient resources to run the kernel." << std::endl;
      return false;
    }

    this->initialize(options);

    //
    // Initialize the GEMM operator
    //

    typename Gemm::EpilogueOutputOp::Params::ActivationParams activation_params{
      ElementCompute(options.alpha),
      ElementCompute(options.beta)
    };
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

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      options.problem_size,
      /* batch_count = */ 1,
      epilogue_params,
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_C.device_data(),
      tensor_D.device_data(),
      tensor_Aux.device_data(),
      tensor_Vector.device_data(),
      options.problem_size.m() * options.problem_size.k(),
      options.problem_size.n() * options.problem_size.k(),
      options.problem_size.m() * options.problem_size.n(),
      options.problem_size.m() * options.problem_size.n(),
      (int)options.problem_size.m(), // Batch stride vector
      tensor_A.layout().stride(0),
      tensor_B.layout().stride(0),
      tensor_C.layout().stride(0),
      tensor_D.layout().stride(0),
      (int64_t)0 // Leading dimension of vector. This must be 0
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::can_implement() failed" << std::endl;
      return false;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::initialize() failed" << std::endl;
      return false;
    }

    //
    // Run the GEMM
    //

    status = gemm_op();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::run() failed" << std::endl;
      return false;
    }

    cudaError_t cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
      return false;
    }

    //
    // Verify
    //

    bool passed = true;
    if (options.reference_check) {
      passed &= this->verify(options);
    } else {
      std::cout << "Skipped reference check" << std::endl;
    }

    //
    // Warm up
    //

    for (int i = 0; i < options.warmup_iterations; ++i) {
      gemm_op();
    }

    //
    // Profile
    //

    cudaEvent_t events[2];
    cudaError_t error;
    for (auto & event : events) {
      error = cudaEventCreate(&event);
      if (error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(error) << std::endl;
        return false;
      }
    }

    // Record an event at the start of a series of GEMM operations
    error = cudaEventRecord(events[0]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    // Run profiling loop
    for (int iter = 0; iter < options.iterations; ++iter) {
      gemm_op();
    }

    // Record an event when the GEMM operations have been launched.
    error = cudaEventRecord(events[1]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    // Wait for work on the device to complete.
    error = cudaEventSynchronize(events[1]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    // Compute average runtime and GFLOPs.
    runtime_ms = runtime_ms / float(options.iterations);
    float gflops = options.gflops(runtime_ms / 1000.0f);

    std::cout << "Problem size: " << options.problem_size.m() << 'x' << options.problem_size.n() << 'x' << options.problem_size.k() << std::endl;
    std::cout << "Runtime (ms): " << runtime_ms << std::endl;
    std::cout << "GFLOPs/sec:   " << gflops << std::endl;

    // Cleanup
    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    return passed;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const** argv) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  bool satisfied;
  if (props.major < 10) {
  }
  else {
    satisfied = (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8);
  }

  if (!satisfied) {
    //
    // This example requires an NVIDIA GPU with compute capability 8.9 or greater.
    //

    std::cout
      << "CUTLASS's FP8 SM89 example requires an NVIDIA GPU with compute capability 8.9 or greater "
      << "and CUDA toolkit version 12.4 or later"
      << std::endl;

    return 0;
  }

  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  std::cout << "Running GEMM with staged accumulation (OpMultiplyAdd)" << std::endl;
  std::cout << "=====================================================" << std::endl;
  TestbedRunner<Gemm_<cutlass::arch::OpMultiplyAdd>> testbed_staged_accum;
  bool passed = testbed_staged_accum.run(options);

  if (passed) {
    std::cout << "Passed" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
  }

  std::cout << "\nRunning GEMM with fast accumulation (OpMultiplyAddFastAccum)" << std::endl;
  std::cout << "============================================================" << std::endl;
  TestbedRunner<Gemm_<cutlass::arch::OpMultiplyAddFastAccum>> testbed_fast_accum;
  passed = testbed_fast_accum.run(options);

  if (passed) {
    std::cout << "Passed" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
  }

  return 0;
}
