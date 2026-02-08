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
    \brief CUTLASS Layernorm Example.

    This workload provides a layer normalization example using a one-pass, square-sum-based
    variance calculation. Specifically, we fuse the reduction operation to find 
    local mean and local square sum mean in the epilogue of 1st GEMM. After a light 
    full reduction kernel, the mean / variance values are readily calculated for element-wise
    operations which are fused into the 2nd GEMM.

    As stated in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data,
    the square-sum based one-pass implementation may raise concerns on numerical stability issues. 
    That being said, though this fully fused layernorm example almost perfectly hides all the memory cost to 
    access the intermediate matrix for layernorm computation, the numerical issue might hinder a persuasive 
    usage in real-world scenarios. If that is the case, a user may turn to the stand-alone CUTLASS layernorm
    example in tools/util/include/cutlass/util/device_layernorm.h

    Examples:

      # Run a CUTLASS layernorm example with default setup , 
      # using the language of the transformer model as an example,
      (Column Major output matrix, hidden dimension = 768, valid word number = 4096, intermediate_scale = 4)
      $ ./examples/37_gemm_layernorm_gemm_fusion/37_gemm_layernorm_gemm_fusion

      # Run an attention example with hidden dimension = 512
      $ ./examples/37_gemm_layernorm_gemm_fusion/37_gemm_layernorm_gemm_fusion --hidden_dim=512

*/

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/device/gemm_complex.h"
#include "cutlass/epilogue/thread/scale_type.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/fast_math.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "gemm_with_layernorm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

enum class Disposition {
  kPassed,
  kIncorrect,
  kNotVerified
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
template<typename LayoutOutput_>
struct Options {

  using LayoutOutput = LayoutOutput_;

  static bool const kIsColumnMajorOutput = cutlass::platform::is_same<LayoutOutput, cutlass::layout::ColumnMajor>::value;

  bool help;
  cutlass::gemm::GemmCoord problem_size0;
  cutlass::gemm::GemmCoord problem_size1;
  int hidden_dim;
  int valid_word_num;
  int intermediate_scale;
  int iterations;
  unsigned seed;
  float alpha;
  float beta;
  bool verification_enabled;
  double tolerance;

  Options():
    help(false),
    iterations(20),
    seed(2022),
    hidden_dim(768),
    valid_word_num(4096),
    intermediate_scale(4),
    alpha(1),
    beta(0),
    verification_enabled(true),
    tolerance(0.01),
    problem_size1(problem_size0.m() * 4, problem_size0.n(), problem_size0.m())
  { }

  bool valid() {

    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("hidden_dim", hidden_dim, 768);
    cmd.get_cmd_line_argument("valid_word_num", valid_word_num, 4096);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("verify", verification_enabled);
    cmd.get_cmd_line_argument("seed", seed);
    cmd.get_cmd_line_argument("tolerance", tolerance);

    if (kIsColumnMajorOutput) {
      // column major output setup
      problem_size0.m() = hidden_dim;
      problem_size0.n() = valid_word_num;
      problem_size0.k() = hidden_dim;

      problem_size1.m() = hidden_dim * intermediate_scale;
      problem_size1.n() = valid_word_num;
      problem_size1.k() = hidden_dim;
    }else{
      // row major output setup
      problem_size0.m() = valid_word_num;
      problem_size0.n() = hidden_dim;
      problem_size0.k() = hidden_dim;

      problem_size1.m() = valid_word_num;
      problem_size1.n() = hidden_dim * intermediate_scale;
      problem_size1.k() = hidden_dim;
    }

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "37_gemm_layernorm_gemm_fusion example\n\n"
      << "  This example uses the CUTLASS Library to compute GEMM + Layernorm for arbitrary problem sizes.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --hidden_dim=<int>          Hidden dimension\n"
      << "  --valid_word_num=<int>      Valid word number\n"
      << "  --seed=<int>                Random number seed (1*)\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform (0 to disable profiling).\n\n"
      << "  --verify=<bool>             If true, performs reference calculation.\n\n"
      << "  --tolerance <float>         Error tolerance\n"
    ;

    out << "\n\nExamples:\n\n"
      << "$ ./examples/37_gemm_layernorm_gemm_fusion/37_gemm_layernorm_gemm_fusion \\\n"
      << "     --hidden_dim=768 --valid_word_num=1024 \n\n";

    return out;
  }

  /// Returns true if the environment and Toolkit support this
  bool supported(bool verbose = true) const {

    // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
    // in CUDA 11.0.
    //
    // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ >= 11)) {
      if (verbose) {
        std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
      }
      return false;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
      if (verbose) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
      }
      return false;
    }

    if (!((props.major * 10 + props.minor) >= 80)) {
      if (verbose) {
        std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
                  << std::endl;
      }
      return false;
    }

    //
    // CUTLASS attempts to load 128b vectors of cutlass::half_t (F16) elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 8 elements.
    //
    int const kAlignment = 8;

    if ((problem_size0.m() % kAlignment) ||
        (problem_size0.n() % kAlignment) ||
        (problem_size0.k() % kAlignment)) {
      if (verbose) {
        std::cerr << "Misaligned input in 1st GEMM." << std::endl;
      }
      // misaligned tensors for Gemm1
      return false;
    }

    if ((problem_size1.m() % kAlignment) ||
        (problem_size1.n() % kAlignment) ||
        (problem_size1.k() % kAlignment)) {
      if (verbose) {
        std::cerr << "Misaligned input in 2nd GEMM." << std::endl;
      }
      // misaligned tensors for Gemm2
      return false;
    }

    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  typename LayoutOutput_>
struct Testbed {

  //
  // Type definitions
  //

  // User-defined data types
  using ElementInputA0 = cutlass::half_t;
  using ElementInputB0 = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  using LayoutInputA0 = cutlass::layout::RowMajor;
  using LayoutInputB0 = cutlass::layout::ColumnMajor;
  using LayoutOutput = LayoutOutput_;

  static bool const kIsColumnMajorOutput = cutlass::platform::is_same<LayoutOutput, cutlass::layout::ColumnMajor>::value;
  // turn of shifted K by default
  static bool const kIsShiftedVariance = false;

  /// Linear scaling operator
  using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementCompute,
    ElementCompute
  >;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  
  static int const kStages0  = 3;
  static int const kStages1  = 4;

  using GemmLayernorm = cutlass::GemmLayernorm<
    ElementInputA0,
    LayoutInputA0,
    ElementInputB0,
    LayoutInputB0,
    ElementOutput,
    LayoutOutput,
    ElementCompute,
    EpilogueFunctorOp,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    kStages0,
    kStages1,
    kIsShiftedVariance
  >;
  
  using ElementInputA1 = typename GemmLayernorm::ElementInputA1;
  using ElementOutputC1 = typename GemmLayernorm::ElementOutputC1;
  using ElementInputScaleBias = typename GemmLayernorm::ElementInputScaleBias;
  using ElementLayernormCompute = typename GemmLayernorm::ElementLayernormCompute;

  using LayoutInputA1 = typename GemmLayernorm::LayoutInputA1;
  using LayoutOutputC0 = typename GemmLayernorm::LayoutOutputC0;
  using LayoutOutputC1 = typename GemmLayernorm::LayoutOutputC1;
  using LayoutInputScaleBias = typename GemmLayernorm::LayoutInputScaleBias;

  //
  // Data members
  //

  Options<LayoutOutput> const &options;

  cutlass::HostTensor<ElementInputA0, LayoutInputA0>                 tensor_A0;
  cutlass::HostTensor<ElementInputB0, LayoutInputB0>                 tensor_B0;
  cutlass::HostTensor<ElementOutput, LayoutOutputC0>                 tensor_C0;
  cutlass::HostTensor<ElementInputA1, LayoutInputA1>                 tensor_A1;
  cutlass::HostTensor<ElementOutputC1, LayoutOutputC1>               tensor_C1;

  cutlass::HostTensor<ElementOutput, LayoutOutputC0>                 reference_C0;
  cutlass::HostTensor<ElementOutputC1, LayoutOutputC1>               reference_C1;

  cutlass::HostTensor<ElementInputScaleBias, LayoutInputScaleBias>   tensor_Variance;
  cutlass::HostTensor<ElementInputScaleBias, LayoutInputScaleBias>   tensor_Mean;
  cutlass::HostTensor<ElementInputScaleBias, LayoutInputScaleBias>   tensor_Beta;
  cutlass::HostTensor<ElementInputScaleBias, LayoutInputScaleBias>   tensor_Gamma;

  cutlass::HostTensor<ElementInputScaleBias, LayoutInputScaleBias>   reference_Mean;
  cutlass::HostTensor<ElementInputScaleBias, LayoutInputScaleBias>   reference_Variance;

  // shifted K tensor to better ensure the numerical stability
  // According to https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  // the closer shifted K to the actual mean, the better numerical stability we'll observe
  cutlass::HostTensor<ElementOutput, LayoutOutputC0>                 tensor_Shifted_K;

  //
  // Methods
  //

  Testbed(
    Options<LayoutOutput> const &options_
  ):
    options(options_)
  {

    tensor_A0.reset({options.problem_size0.m(), options.problem_size0.k()});
    tensor_B0.reset({options.problem_size0.k(), options.problem_size0.n()});

    tensor_C0.reset({options.problem_size0.m(), options.problem_size0.n()});

    tensor_A1.reset({options.problem_size1.m(), options.problem_size1.k()});
    tensor_C1.reset({options.problem_size1.m(), options.problem_size1.n()});

    reference_C0.reset({options.problem_size0.m(), options.problem_size0.n()});
    reference_C1.reset({options.problem_size1.m(), options.problem_size1.n()});

    int leading_dim_0 = kIsColumnMajorOutput ? options.problem_size0.n() : options.problem_size0.m();
    int leading_dim_1 = kIsColumnMajorOutput ? options.problem_size0.m() : options.problem_size0.n();

    int block_num = (leading_dim_1 + GemmLayernorm::ThreadblockShape::kM - 1) / GemmLayernorm::ThreadblockShape::kM;

    tensor_Variance.reset({block_num, leading_dim_0});
    tensor_Mean.reset({block_num, leading_dim_0});
    tensor_Shifted_K.reset({1, leading_dim_0});

    tensor_Beta.reset({1, leading_dim_1});
    tensor_Gamma.reset({1, leading_dim_1});

    reference_Mean.reset({1, leading_dim_0}, false);
    reference_Variance.reset({1, leading_dim_0}, false);
    
  }

  /// Run
  Disposition run() {

    Disposition disposition = Disposition::kNotVerified;

    //
    // Initialize the workspace
    //

    initialize();

    //
    // Launch device kernel
    //
    cutlass::Status status = cutlass::Status::kSuccess;

    status = execute_device_kernel();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Device execution failed." << std::endl;
      return disposition;
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Device synchronize failed with error "
        << cudaGetErrorString(result) << std::endl;
      return disposition;
    }

    //
    // Compute the reference
    //
    compute_reference();

    //
    // Verify
    //

    if (options.verification_enabled) {

      bool passed = verify();

      if (passed) {
        disposition = Disposition::kPassed;
      }
      else {
        disposition = Disposition::kIncorrect;
      }
    }

    //
    // Profiling
    //
    if (options.iterations) {
      profile();
    }

    return disposition;
  }

  /// Random initialization
  void initialize() {

    cutlass::reference::host::TensorFillRandomUniform(
      tensor_A0.host_view(),
        options.seed,
        ElementInputA0(4),
        ElementInputA0(-4),
        0
      );

    cutlass::reference::host::TensorFillRandomUniform(
      tensor_B0.host_view(),
        options.seed + 1,
        ElementInputB0(4),
        ElementInputB0(-4),
        0
      );

    cutlass::reference::host::TensorFillRandomUniform(
      tensor_A1.host_view(),
        options.seed + 2,
        ElementInputA1(4),
        ElementInputA1(-4),
        0
      );

    cutlass::reference::host::TensorFillRandomUniform(
      tensor_Beta.host_view(),
        options.seed + 3,
        ElementInputScaleBias(4),
        ElementInputScaleBias(-4),
        0
      );

    cutlass::reference::host::TensorFillRandomUniform(
      tensor_Gamma.host_view(),
        options.seed + 4,
        ElementInputScaleBias(4),
        ElementInputScaleBias(-4),
        0
      );

    cutlass::reference::host::TensorFillRandomUniform(
      tensor_Shifted_K.host_view(),
        options.seed + 5,
        ElementOutput(4),
        ElementOutput(-5),
        0
      );

    tensor_A0.sync_device();
    tensor_B0.sync_device();
    tensor_A1.sync_device();
    tensor_Beta.sync_device();
    tensor_Gamma.sync_device();

  }



  cutlass::Status execute_device_kernel() {

    cutlass::Status status = cutlass::Status::kSuccess;

    //
    // Setup arguments
    //

    typename GemmLayernorm::Arguments args(
      options.problem_size0,
      options.problem_size1,
      tensor_A0.device_ref().data(),
      tensor_B0.device_ref().data(),
      tensor_C0.device_ref().data(),
      tensor_C0.device_ref().data(),
      tensor_A1.device_ref().data(),
      tensor_C1.device_ref().data(),
      tensor_A0.device_ref().stride(0),
      tensor_B0.device_ref().stride(0),
      tensor_C0.device_ref().stride(0),
      tensor_C0.device_ref().stride(0),
      tensor_A1.device_ref().stride(0),
      tensor_C1.device_ref().stride(0),
      {
        ElementCompute(options.alpha),
        ElementCompute(options.beta)
      },
      tensor_Variance.device_ref(),
      tensor_Mean.device_ref(),
      tensor_Gamma.device_ref(),
      tensor_Beta.device_ref(),
      tensor_Shifted_K.device_ref().data()
    );

    //
    // Launch
    //

    GemmLayernorm gemm_layernorm;

    // Initialize
    status = gemm_layernorm.initialize(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    // Run
    status = gemm_layernorm();

    return status;
  }

  /// Reference calculation
  void compute_reference() {

    cutlass::reference::device::Gemm<
      ElementInputA0,
      LayoutInputA0,
      ElementInputB0,
      LayoutInputB0,
      ElementOutput,
      LayoutOutputC0,
      ElementCompute,
      ElementCompute
    > gemm_device0;

    cutlass::reference::device::Gemm<
      ElementInputA1,
      LayoutInputA1,
      ElementOutput,
      LayoutOutputC0,
      ElementOutputC1,
      LayoutOutputC1,
      ElementCompute,
      ElementCompute
    > gemm_device1;

    // Compute 1st GEMM
    gemm_device0(
      options.problem_size0,
      ElementCompute(options.alpha),
      tensor_A0.device_ref(),
      tensor_B0.device_ref(),
      ElementCompute(options.beta),
      tensor_C0.device_ref(),
      reference_C0.device_ref()
    );

    reference_C0.sync_host();

    tensor_Mean.sync_host();
    tensor_Variance.sync_host();
    tensor_Gamma.sync_host();
    tensor_Beta.sync_host();
    tensor_Shifted_K.sync_host();

    // Compute the sum and square sum for verification purpose
    if (kIsColumnMajorOutput) {
      for (int n = 0; n < options.problem_size0.n(); ++n) {
      
        ElementLayernormCompute sum = ElementLayernormCompute(0);
        ElementLayernormCompute square_sum = ElementLayernormCompute(0);
        for (int m = 0; m < options.problem_size0.m(); ++m) {
          sum += ElementLayernormCompute(reference_C0.at({m, n}));
          square_sum += ElementLayernormCompute(reference_C0.at({m, n})) * ElementLayernormCompute(reference_C0.at({m, n}));
        }
        
        ElementLayernormCompute mean = sum / ElementLayernormCompute(options.problem_size0.m());
        ElementLayernormCompute square_mean = square_sum / ElementLayernormCompute(options.problem_size0.m());
        ElementLayernormCompute variance = cutlass::constants::one<ElementLayernormCompute>() / cutlass::fast_sqrt(square_mean - mean * mean + ElementLayernormCompute(1e-6) ) ;

        mean = -mean * variance;

        reference_Mean.at({0, n}) = ElementInputScaleBias(mean);
        reference_Variance.at({0, n}) = ElementInputScaleBias(variance);
      }
    }else{
      for (int m = 0; m < options.problem_size0.m(); ++m) {
      
        ElementLayernormCompute sum = ElementLayernormCompute(0);
        ElementLayernormCompute square_sum = ElementLayernormCompute(0);
        for (int n = 0; n < options.problem_size0.n(); ++n) {
          sum += ElementLayernormCompute(reference_C0.at({m, n})) ;
          square_sum += ElementLayernormCompute(reference_C0.at({m, n})) * ElementLayernormCompute(reference_C0.at({m, n})) ;
        }

        ElementLayernormCompute mean = sum / ElementLayernormCompute(options.problem_size0.n());
        ElementLayernormCompute square_mean = square_sum / ElementLayernormCompute(options.problem_size0.n());
        ElementLayernormCompute variance = cutlass::constants::one<ElementLayernormCompute>() / cutlass::fast_sqrt(square_mean - mean * mean + ElementLayernormCompute(1e-6)) ;

        mean = -mean * variance;

        reference_Mean.at({0, m}) = ElementInputScaleBias(mean);
        reference_Variance.at({0, m}) = ElementInputScaleBias(variance);
      }
    }

    // Element-wise transform for OutputC0 using 1-pass layernorm algo
    if (kIsColumnMajorOutput) {
      for (int n = 0; n < options.problem_size0.n(); ++n) {

        ElementLayernormCompute sum = ElementLayernormCompute(0);
        for (int m = 0; m < options.problem_size0.m(); ++m) {
          sum += ElementLayernormCompute(reference_C0.at({m, n})) ;
        }

        ElementInputScaleBias mean = ElementInputScaleBias(sum / ElementLayernormCompute(options.problem_size0.m()));
        sum = ElementLayernormCompute(0);
        for (int m = 0; m < options.problem_size0.m(); ++m) {
          sum += ElementLayernormCompute(reference_C0.at({m, n}) - ElementLayernormCompute(mean)) * ElementLayernormCompute(reference_C0.at({m, n}) - ElementLayernormCompute(mean)) ;
        }

        ElementLayernormCompute square_mean = sum / ElementLayernormCompute(options.problem_size0.m());
        ElementInputScaleBias variance = ElementInputScaleBias(cutlass::constants::one<ElementLayernormCompute>() 
                            / cutlass::fast_sqrt(square_mean + ElementLayernormCompute(1e-6))) ;

        for (int m = 0; m < options.problem_size0.m(); ++m) {
          reference_C0.at({m, n}) = 
              ElementOutput( ( (ElementInputScaleBias(reference_C0.at({m, n})) - mean) * variance )
                * tensor_Gamma.at({0, m}) + tensor_Beta.at({0, m}));

        }

      }
    }else{

      for (int m = 0; m < options.problem_size0.m(); ++m) {

        float sum = float(0);
        for (int n = 0; n < options.problem_size0.n(); ++n) {
          sum += float(reference_C0.at({m, n})) ;
        }

        float mean = sum / float(options.problem_size0.n());
        sum = float(0);
        for (int n = 0; n < options.problem_size0.n(); ++n) {
          sum += float(reference_C0.at({m, n}) - mean) * float(reference_C0.at({m, n}) - mean) ;
        }

        float square_mean = sum / float(options.problem_size0.n());
        float variance = cutlass::constants::one<float>() / cutlass::fast_sqrt(square_mean + ElementLayernormCompute(1e-6)) ;

        for (int n = 0; n < options.problem_size0.n(); ++n) {
          reference_C0.at({m, n}) = 
              ElementOutput( ( (float(reference_C0.at({m, n})) - mean) * variance )
                * float(tensor_Gamma.at({0, n})) + float(tensor_Beta.at({0, n})));

        }

      }

    }


    // Sync host data with device after element-wise transform
    reference_C0.sync_device();

    // Compute 2nd GEMM
    gemm_device1(
      options.problem_size1,
      ElementCompute(options.alpha),
      kIsColumnMajorOutput ? tensor_A1.device_ref() : reference_C0.device_ref(),
      kIsColumnMajorOutput ? reference_C0.device_ref() :tensor_A1.device_ref(),
      ElementCompute(options.beta),
      reference_C1.device_ref(),
      reference_C1.device_ref()
    );

  }

  /// Emits all tensor values
  void emit_results() {
    std::cout << "tensor_C1 = \n" << tensor_C1.host_view() << "\n\n";
    std::cout << "Reference C1 = \n" << reference_C1.host_view() << "\n\n";
    std::cout << "Mean = \n" << tensor_Mean.host_view() << "\n\n";
    std::cout << "rsqrt(Variance) = \n" << tensor_Variance.host_view() << "\n\n";
    std::cout << "Reference Mean = \n" << reference_Mean.host_view() << "\n\n";
    std::cout << "Reference rsqrt(Variance) = \n" << reference_Variance.host_view() << "\n\n";
  }

  template<typename Element, typename Layout>
  bool verify_tensor(cutlass::HostTensor<Element, Layout> tensor, \
                       cutlass::HostTensor<Element, Layout> reference,
                       int leading_dim0, int leading_dim1, bool is_print = false) {
    float const kThreshold = float(options.tolerance);
    float const kAbsThreshold = 0.5f;
    float const kRelativeThreshold = 0.1f;
    // Adds a constant bias to avoid being divided by '0'
    float const kBias = 1e-5f;
    int counter = 0;
    for (int m = 0; m < leading_dim0; m++) {
      for (int n = 0; n < leading_dim1; ++n) {
        float diff = (float)(tensor.at({m, n}) - reference.at({m, n}));
        float rel_diff = fabs(diff) / fabs(reference.at({m, n}) + kBias);
        if (fabs(diff) > kAbsThreshold && rel_diff > kRelativeThreshold) {
          counter++;
        }
      }
    }

    float err_rate = float(counter) / (float(leading_dim0) * float(leading_dim1));
    return (err_rate < kThreshold);
  }

  /// Verifies the reference matches
  bool verify() {

    tensor_Variance.sync_host();
    tensor_Mean.sync_host();
    tensor_C1.sync_host();
    reference_C1.sync_host();

    // Verification checks - set any of these to 'true' to override the verification checks.
    bool verified_C1 = false;
    bool verified_Mean = false;
    bool verified_Variance = false;

    // Verify layernorm output
    if (!verified_C1) {
      verified_C1 = verify_tensor<ElementOutputC1, LayoutOutputC1>(tensor_C1, reference_C1, options.problem_size1.m(), options.problem_size1.n());
    }

    if (!verified_Variance) {
      verified_Variance = verify_tensor<ElementInputScaleBias, LayoutInputScaleBias>(tensor_Variance, reference_Variance, 1, options.problem_size0.n());
    }

    if (!verified_Mean) {
      verified_Mean = verify_tensor<ElementInputScaleBias, LayoutInputScaleBias>(tensor_Mean, reference_Mean, 1, options.problem_size0.n());
    }

    if (!verified_C1 || !verified_Mean || !verified_Variance) {

      // emit_results();

      std::cerr << "Verification check failed for tensor Layernorm" << std::endl;

      // Summarize which checks failed
      if (!verified_C1) {
        std::cerr << "Verification of O tensor failed\n";
      }

      if (!verified_Mean) {
        std::cerr << "Verification of Mean tensor failed\n";
      }

      if (!verified_Variance) {
        std::cerr << "Verification of Variance tensor failed\n";
      }

      return false;
    }

    return true;
  }

  /// Profiles
  bool profile() {

    //
    // Profile
    //

    cutlass::Status status = cutlass::Status::kSuccess;
    cudaError_t result;
    cudaEvent_t events[2];
    int const kIterations = options.iterations;

    for (cudaEvent_t &evt : events) {
      result = cudaEventCreate(&evt);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventCreate failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }
    }

    result = cudaEventRecord(events[0]);

    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    for (int iter = 0; iter < kIterations; ++iter) {

      status = execute_device_kernel();

      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Device execution failed." << std::endl;
        return false;
      }
    }

    result = cudaEventRecord(events[1]);

    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    result = cudaDeviceSynchronize();

    if (result != cudaSuccess) {
      std::cerr << "cudaDeviceSynchronize() failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    float elapsed_ms = 0;
    result = cudaEventElapsedTime(&elapsed_ms, events[0], events[1]);

    float elapsed_ms_per_iter = elapsed_ms / float(kIterations);

    if (result != cudaSuccess) {
      std::cerr << "cudaEventElapsedTime() failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    for (cudaEvent_t &evt : events) {
      result = cudaEventDestroy(evt);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventDestroy() failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }
    }

    int64_t flops = int64_t(options.problem_size0.m()) * options.problem_size0.n() * options.problem_size0.k() * 2 \
                   + int64_t(options.problem_size1.m()) * options.problem_size1.n() * options.problem_size1.k() * 2;

    double gflops_per_second = double(flops) * kIterations / double(elapsed_ms / 1000.0f) / double(1.0e9);

    std::cout << "    1st  GEMM: "
              << options.problem_size0.m() << "-by-" << options.problem_size0.n() << "-by-" << options.problem_size0.k() << "\n"
              << "    2nd  GEMM: "
              << options.problem_size1.m() << "-by-" << options.problem_size1.n() << "-by-" << options.problem_size1.k()
              << std::endl;

    std::cout << " Runtime / iteration: " << elapsed_ms_per_iter << " ms\n" << std::endl;
    std::cout << "              GFLOPs: " << gflops_per_second << "  GFLOPs" << std::endl;

    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  
  // Define final layout
  using LayoutOutput = cutlass::layout::ColumnMajor;

  // Options parsing
  Options<LayoutOutput> options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (!options.supported()) {
    return 0;
  }

  // Run
  Testbed<LayoutOutput> testbed(options);

  Disposition disposition = testbed.run();

  std::cout << std::endl;

  switch (disposition) {
    case Disposition::kPassed:
      std::cout << "Passed" << std::endl;
      break;
    case Disposition::kIncorrect:
      std::cout << "Incorrect" << std::endl;
      break;
    case Disposition::kNotVerified:
      std::cout << "Not verified" << std::endl;
      break;
  }

  return (disposition == Disposition::kPassed ? 0 : -1);
}


/////////////////////////////////////////////////////////////////////////////////////////////////
