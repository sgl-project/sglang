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

/**
NVIDIA Ampere architecture starts supporting tfloat32 (see include/cutlass/tfloat32.h)
data types in tensor cores.  One big advantage is that we can load in F32 data and convert them
implicitly to tf32 inside the SYMM kernel which means no change is needed to accelerate traditional
F32 data by using NVIDIA Ampere architecture.

We can use the tf32 mode of tensor core to emulate a fast accurate SYMM kernel which is accelerated
using Ampere Tensor Cores (see include/cutlass/gemm/warp/mma_tensor_op_fast_f32.h).

The trick is very simple
  a x b = (a_big + a_small) x (b_big + b_small) = a_big x b_big + a_big x b_small + a_small x b_big
  big = convert_to_tf32(F32)
  small = convert_to_tf32(F32 - big)

a_small x b_small is discarded because they are too small.

This example demonstrates usage of this kernel, along with accuracy measurements w.r.t. actual F32
results (SSYMM from cuBLAS) and against F64 results (DSYMM from CUTLASS)

To enable this feature, the only change needs to make is to change the default OpMultiplyAdd to
OpMultiplyAddFastF32.

Now, we have two different flavors of SSYMM in the profiler for Ampere:

  s1688symm       // Use 3xTF32 to emulate F32.  F32 in, converted in TF32-big and TF32-small internally,
                  // accumulated in F32, F32 out.
  s1688tf32symm   // Use 1xTF32.  F32 in, converted to one TF32 internally, accumulated in F32, F32 out.
*/

#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/blas3.h"
#include "cutlass/gemm/device/symm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/host/symm.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

#if CUTLASS_ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  cutlass::gemm::GemmCoord problem_size;
  float alpha;
  float beta;
  std::string rand_mode;
  int seed;

  Options():
    help(false),
    problem_size({4096, 4096, 4096}),
    seed(1),
    alpha(1),
    beta(),
    rand_mode("uniform") { }

  bool valid() {
    //
    // CUTLASS attempts to load 128b vectors of F32 elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 4 elements.
    //
    int const kAlignment = 4;

    if ((problem_size.m() % kAlignment) ||
      (problem_size.n() % kAlignment) ||
      (problem_size.k() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    // Since the kernels in this example are in Left Side Mode
    cmd.get_cmd_line_argument("m", problem_size.k());

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);

    cmd.get_cmd_line_argument("seed", seed);
    cmd.get_cmd_line_argument("rand_mode", rand_mode);

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "33_ampere_3xtf32_tensorop_symm example\n\n"
      << "  This example uses the CUTLASS Library to execute 3xTF32 tensorop SYMM computations.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   SYMM M dimension\n"
      << "  --n=<int>                   SYMM N dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --rand_mode=<string>        gauss / uniform*\n\n"
      << "  --seed=<int>                Random number seed (1*)\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/33_ampere_3xtf32_tensorop_symm/33_ampere_3xtf32_tensorop_symm --m=1024 --n=512 \\\n"
      << "     --alpha=2 --beta=1 \n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Matrix B and Matrix C (since that's what cuBLAS supports, CUTLASS supports Row Major too)
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

// Symmetric Matrix A is in Left Side mode
constexpr cutlass::SideMode SideModeA = cutlass::SideMode::kLeft;
// Symmetric Matrix A is in Lower Filled mode
constexpr cutlass::FillMode FillModeA = cutlass::FillMode::kLower;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 64, 16>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    float,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<float>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    float,                                   // <- data type of accumulator
    float>;                                  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 3;
// Alignment
constexpr int Alignment = 4;

//
// CUTLASS Symm Operators (SSYM: Symm_3xTF32, Symm_1xTF32, DSYMM: Symm_F64)
//

// Symm_3xTF32
using Symm_3xTF32 = cutlass::gemm::device::Symm<
                                              float,
                                              LayoutInputA,
                                              SideModeA,
                                              FillModeA,
                                              float,
                                              LayoutInputB,
                                              float,
                                              LayoutOutput,
                                              float,
                                              MMAOp,
                                              SmArch,
                                              ShapeMMAThreadBlock,
                                              ShapeMMAWarp,
                                              ShapeMMAOp,
                                              EpilogueOp,
                                              SwizzleThreadBlock,
                                              NumStages,
                                              1, // Symmetric matrix is always align 1
                                              Alignment,
                                              false,
                                              cutlass::arch::OpMultiplyAddFastF32>;

// Symm_1xTF32
using Symm_1xTF32 = cutlass::gemm::device::Symm<
                                              float,
                                              LayoutInputA,
                                              SideModeA,
                                              FillModeA,
                                              float,
                                              LayoutInputB,
                                              float,
                                              LayoutOutput,
                                              float,
                                              MMAOp,
                                              SmArch,
                                              ShapeMMAThreadBlock,
                                              ShapeMMAWarp,
                                              ShapeMMAOp,
                                              EpilogueOp,
                                              SwizzleThreadBlock,
                                              NumStages,
                                              1, // Symmetric matrix is always align 1
                                              Alignment,
                                              false,
                                              cutlass::arch::OpMultiplyAdd>;

// Symm_F64
using Symm_F64 = cutlass::gemm::device::Symm<
                                              double,
                                              LayoutInputA,
                                              SideModeA,
                                              FillModeA,
                                              double,
                                              LayoutInputB,
                                              double,
                                              LayoutOutput,
                                              double,
                                              cutlass::arch::OpClassTensorOp,
                                              cutlass::arch::Sm80,
                                              cutlass::gemm::GemmShape<32, 32, 16>,
                                              cutlass::gemm::GemmShape<16, 16, 16>,
                                              cutlass::gemm::GemmShape<8, 8, 4>,
                                              cutlass::epilogue::thread::LinearCombination<
                                                double,
                                                1,
                                                double,
                                                double
                                              >,
                                              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
                                              4>;

bool run(Options &options) {

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size = options.problem_size;

  ////////////////////////////////////////////////////////////////////////////////
  /// 1. Initialize F32 Precision input tensors using CUTLASS helper functions
  ////////////////////////////////////////////////////////////////////////////////
  cutlass::HostTensor<float, LayoutInputA> tensor_a_F32(problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<float, LayoutInputB> tensor_b_F32(problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<float, LayoutOutput> tensor_c_F32(problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<float, LayoutOutput> tensor_d_F32(problem_size.mn());  // <- Create matrix D with dimensions M x N

  if (options.rand_mode == "uniform") {
    const float min = -1;
    const float max =  1;
    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a_F32.host_view(),
        options.seed,
        double(max),
        double(min));      // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b_F32.host_view(),
        options.seed,
        double(max),
        double(min));      // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c_F32.host_view(),
        options.seed,
        double(max),
        double(min));      // <- Fill matrix C on host with uniform-distribution random data
  } else if (options.rand_mode == "gauss") {
    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomGaussian(
        tensor_a_F32.host_view(),
        options.seed,
        double(0),
        double(5));      // <- Fill matrix A on host with gaussian-distribution random data
    cutlass::reference::host::TensorFillRandomGaussian(
        tensor_b_F32.host_view(),
        options.seed,
        double(0),
        double(5));      // <- Fill matrix B on host with gaussian-distribution random data
    cutlass::reference::host::TensorFillRandomGaussian(
        tensor_c_F32.host_view(),
        options.seed,
        double(0),
        double(5));      // <- Fill matrix C on host with gaussian-distribution random data
  }
  cutlass::reference::host::TensorFill(
      tensor_d_F32.host_view());  // <- fill matrix D on host with zeros

  // Copy data from host to GPU
  tensor_a_F32.sync_device();
  tensor_b_F32.sync_device();
  tensor_c_F32.sync_device();
  tensor_d_F32.sync_device();

  ////////////////////////////////////////////////////////////////////////////////
  /// 2. Initialize F64 tensors, Output tensors and setup arguments
  ////////////////////////////////////////////////////////////////////////////////
  // Symm F64 input operands (A, B, C)
  cutlass::HostTensor<double, LayoutInputA> tensor_a_F64(problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<double, LayoutInputB> tensor_b_F64(problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<double, LayoutOutput> tensor_c_F64(problem_size.mn());  // <- Create matrix C with dimensions M x N

  // Symm output (D) for SYMM_3xTF32
  cutlass::HostTensor<float, LayoutOutput> tensor_d_3xTF32(problem_size.mn());  // <- Create matrix D with dimensions M x N
  // Symm output (D) for SYMM_1xTF32
  cutlass::HostTensor<float, LayoutOutput> tensor_d_1xTF32(problem_size.mn());  // <- Create matrix D with dimensions M x N
  // Symm output (D) for SYMM_F64
  cutlass::HostTensor<double, LayoutOutput> tensor_d_F64(problem_size.mn());  // <- Create matrix D with dimensions M x N
#if CUTLASS_ENABLE_CUBLAS
  // Symm output (D) for SYMM_cublasF32
  cutlass::HostTensor<float, LayoutOutput> tensor_d_cublasF32(problem_size.mn());  // <- Create matrix D with dimensions M x N
#endif

  // Copy values from the DP tensors
  cutlass::reference::host::TensorCopy(tensor_a_F64.host_view(), tensor_a_F32.host_view());
  cutlass::reference::host::TensorCopy(tensor_b_F64.host_view(), tensor_b_F32.host_view());
  cutlass::reference::host::TensorCopy(tensor_c_F64.host_view(), tensor_c_F32.host_view());
  cutlass::reference::host::TensorCopy(tensor_d_F64.host_view(), tensor_d_F32.host_view());
  cutlass::reference::host::TensorCopy(tensor_d_3xTF32.host_view(), tensor_d_F32.host_view());
  cutlass::reference::host::TensorCopy(tensor_d_1xTF32.host_view(), tensor_d_F32.host_view());
#if CUTLASS_ENABLE_CUBLAS
  cutlass::reference::host::TensorCopy(tensor_d_cublasF32.host_view(), tensor_d_F32.host_view());
#endif

  // Copy data from host to GPU
  tensor_a_F64.sync_device();
  tensor_b_F64.sync_device();
  tensor_c_F64.sync_device();
  tensor_d_F64.sync_device();
  tensor_d_3xTF32.sync_device();
  tensor_d_1xTF32.sync_device();
#if CUTLASS_ENABLE_CUBLAS
  tensor_d_cublasF32.sync_device();
#endif

  // Initialize alpha and beta for dot product computation
  float alpha = float(options.alpha);
  float beta =  float(options.beta);

  // Batch count as 1
  int batch_count = 1;

  // Batch stride for A, when matrix A is in Left Side mode
  int batch_stride_A = problem_size.m()*problem_size.m();

  ////////////////////////////////////////////////////////////////////////////////
  /// 3. Run 3xTF32 kernel
  ////////////////////////////////////////////////////////////////////////////////
  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Symm_3xTF32::Arguments arguments_3xtf32{
                                     cutlass::gemm::GemmUniversalMode::kGemm,
                                     problem_size,                  // <- problem size of matrix multiplication
                                     batch_count,                   // <- batch count
                                     {alpha, beta},                 // <- tuple of alpha and beta
                                     tensor_a_F32.device_data(),    // <- reference to matrix A on device
                                     tensor_b_F32.device_data(),    // <- reference to matrix B on device
                                     tensor_c_F32.device_data(),    // <- reference to matrix C on device
                                     tensor_d_3xTF32.device_data(), // <- reference to matrix D on device
                                     batch_stride_A,                // <- batch stride and ld for matrices
                                     problem_size.m() * problem_size.n(),
                                     problem_size.m() * problem_size.n(),
                                     problem_size.m() * problem_size.n(),
                                     tensor_a_F32.layout().stride(0),
                                     tensor_b_F32.layout().stride(0),
                                     tensor_c_F32.layout().stride(0),
                                     tensor_d_3xTF32.layout().stride(0)
                                     };

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size_3xtf32 = Symm_3xTF32::get_workspace_size(arguments_3xtf32);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace_3xtf32(workspace_size_3xtf32);

  // Instantiate CUTLASS kernel depending on templates
  Symm_3xTF32 symm_op_3xtf32;

  // Check the problem size is supported or not
  cutlass::Status status_3xtf32 = symm_op_3xtf32.can_implement(arguments_3xtf32);
  CUTLASS_CHECK(status_3xtf32);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status_3xtf32 = symm_op_3xtf32.initialize(arguments_3xtf32, workspace_3xtf32.get());
  CUTLASS_CHECK(status_3xtf32);

  // Launch initialized CUTLASS kernel
  status_3xtf32 = symm_op_3xtf32();
  CUTLASS_CHECK(status_3xtf32);

  tensor_d_3xTF32.sync_host();

  ////////////////////////////////////////////////////////////////////////////////
  /// 4. Run 1xTF32 kernel
  ////////////////////////////////////////////////////////////////////////////////
  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Symm_1xTF32::Arguments arguments_1xtf32{
                                     cutlass::gemm::GemmUniversalMode::kGemm,
                                     problem_size,                  // <- problem size of matrix multiplication
                                     batch_count,                   // <- batch count
                                     {alpha, beta},                 // <- tuple of alpha and beta
                                     tensor_a_F32.device_data(),    // <- reference to matrix A on device
                                     tensor_b_F32.device_data(),    // <- reference to matrix B on device
                                     tensor_c_F32.device_data(),    // <- reference to matrix C on device
                                     tensor_d_1xTF32.device_data(), // <- reference to matrix D on device
                                     batch_stride_A,                // <- batch stride and ld for matrices
                                     problem_size.m() * problem_size.n(),
                                     problem_size.m() * problem_size.n(),
                                     problem_size.m() * problem_size.n(),
                                     tensor_a_F32.layout().stride(0),
                                     tensor_b_F32.layout().stride(0),
                                     tensor_c_F32.layout().stride(0),
                                     tensor_d_1xTF32.layout().stride(0)
                                     };

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size_1xtf32 = Symm_1xTF32::get_workspace_size(arguments_1xtf32);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace_1xtf32(workspace_size_1xtf32);

  // Instantiate CUTLASS kernel depending on templates
  Symm_1xTF32 symm_op_1xtf32;

  // Check the problem size is supported or not
  cutlass::Status status_1xtf32 = symm_op_1xtf32.can_implement(arguments_1xtf32);
  CUTLASS_CHECK(status_1xtf32);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status_1xtf32 = symm_op_1xtf32.initialize(arguments_1xtf32, workspace_1xtf32.get());
  CUTLASS_CHECK(status_1xtf32);

  // Launch initialized CUTLASS kernel
  status_1xtf32 = symm_op_1xtf32();
  CUTLASS_CHECK(status_1xtf32);

  tensor_d_1xTF32.sync_host();

  ////////////////////////////////////////////////////////////////////////////////
  /// 5. Run F64 kernel
  ////////////////////////////////////////////////////////////////////////////////
  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Symm_F64::Arguments arguments_f64{
                                     cutlass::gemm::GemmUniversalMode::kGemm,
                                     problem_size,                  // <- problem size of matrix multiplication
                                     batch_count,                   // <- batch count
                                     {double(options.alpha), double(options.alpha)},                 // <- tuple of alpha and beta
                                     tensor_a_F64.device_data(),    // <- reference to matrix A on device
                                     tensor_b_F64.device_data(),    // <- reference to matrix B on device
                                     tensor_c_F64.device_data(),    // <- reference to matrix C on device
                                     tensor_d_F64.device_data(),    // <- reference to matrix D on device
                                     batch_stride_A,                // <- batch stride and ld for matrices
                                     problem_size.m() * problem_size.n(),
                                     problem_size.m() * problem_size.n(),
                                     problem_size.m() * problem_size.n(),
                                     tensor_a_F64.layout().stride(0),
                                     tensor_b_F64.layout().stride(0),
                                     tensor_c_F64.layout().stride(0),
                                     tensor_d_F64.layout().stride(0)
                                     };

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size_f64 = Symm_F64::get_workspace_size(arguments_f64);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace_f64(workspace_size_f64);

  // Instantiate CUTLASS kernel depending on templates
  Symm_F64 symm_op_f64;

  // Check the problem size is supported or not
  cutlass::Status status_f64 = symm_op_f64.can_implement(arguments_f64);
  CUTLASS_CHECK(status_f64);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status_f64 = symm_op_f64.initialize(arguments_f64, workspace_f64.get());
  CUTLASS_CHECK(status_f64);

  // Launch initialized CUTLASS kernel
  status_f64 = symm_op_f64();
  CUTLASS_CHECK(status_f64);

  cudaDeviceSynchronize();

  tensor_d_F64.sync_host();

  ////////////////////////////////////////////////////////////////////////////////
  /// 6. Run cuBLAS SSYMM kernel
  ////////////////////////////////////////////////////////////////////////////////

#if CUTLASS_ENABLE_CUBLAS
  cublasStatus_t cublas_status;
  cublasHandle_t handle;

  cublas_status = cublasCreate(&handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
  std::cerr << "Failed to create cuBLAS handle." << std::endl;
    return false;
  }

  cublas_status = cublasSsymm(
      handle,
      CUBLAS_SIDE_LEFT,
      CUBLAS_FILL_MODE_LOWER,
      problem_size.m(),
      problem_size.n(),
      static_cast<const float*>(&alpha),
      static_cast<const float*>(tensor_a_F32.device_data()),
      int(tensor_a_F32.layout().stride(0)),
      static_cast<const float*>(tensor_b_F32.device_data()),
      int(tensor_b_F32.layout().stride(0)),
      static_cast<const float*>(&beta),
      static_cast<float*>(tensor_d_cublasF32.device_data()),
      int(tensor_d_cublasF32.layout().stride(0))
    );

  cudaDeviceSynchronize();

  tensor_d_cublasF32.sync_host();
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// 7. Compute l2 norms
  ////////////////////////////////////////////////////////////////////////////////

#if CUTLASS_ENABLE_CUBLAS
  // l2 norm cuBLAS F32 vs F64
  cutlass::HostTensor<double, LayoutOutput> tensor_d_cublasF32_in_F64(problem_size.mn());
  cutlass::reference::host::TensorCopy(tensor_d_cublasF32_in_F64.host_view(), tensor_d_cublasF32.host_view());

  double l2_norm_cublasf32_vs_f64 = cutlass::reference::host::TensorRelativeErrorMetric(
    tensor_d_cublasF32_in_F64.host_view(), tensor_d_F64.host_view());
#endif

  // l2 norm 3xTF32 vs F64
  cutlass::HostTensor<double, LayoutOutput> tensor_d_3xTF32_in_F64(problem_size.mn());
  cutlass::reference::host::TensorCopy(tensor_d_3xTF32_in_F64.host_view(), tensor_d_3xTF32.host_view());
  double l2_norm_3xtf32_vs_f64 = cutlass::reference::host::TensorRelativeErrorMetric(
    tensor_d_3xTF32_in_F64.host_view(), tensor_d_F64.host_view());

  // l2 norm 1xTF32 vs F64
  cutlass::HostTensor<double, LayoutOutput> tensor_d_1xTF32_in_F64(problem_size.mn());
  cutlass::reference::host::TensorCopy(tensor_d_1xTF32_in_F64.host_view(), tensor_d_1xTF32.host_view());
  double l2_norm_1xtf32_vs_f64 = cutlass::reference::host::TensorRelativeErrorMetric(
    tensor_d_1xTF32_in_F64.host_view(), tensor_d_F64.host_view());

#if CUTLASS_ENABLE_CUBLAS
  // l2 norm 3xTF32 vs cuBLAS F32
  double l2_norm_3xtf32_vs_cublasf32 = cutlass::reference::host::TensorRelativeErrorMetric(
    tensor_d_3xTF32.host_view(), tensor_d_cublasF32.host_view());
#endif

  // l2 norm 3xTF32 vs 1xTF32
  double l2_norm_3xtf32_vs_1xtf32 = cutlass::reference::host::TensorRelativeErrorMetric(
    tensor_d_3xTF32.host_view(), tensor_d_1xTF32.host_view());

  ///////////////////////////////////////////////////////////////////////////////

  // Print kernel info and L2 norms
  std::cout << "Problem Size: (" << problem_size.m() << "," << problem_size.n() << "," << problem_size.k() << ") "
            << "Alpha: "  << alpha << "," << " Beta: "  << beta << std::endl;
  std::cout << std::fixed;
  std::cout << "Normalized L2 norm of" << std::endl;
  std::cout.precision(8);
  std::cout << std::scientific
#if CUTLASS_ENABLE_CUBLAS
            << " - cuBLAS F32 error with F64 reference    : " << l2_norm_cublasf32_vs_f64 << std::endl
#endif
            << " - 3xTF32 error with F64 reference        : " << l2_norm_3xtf32_vs_f64 << std::endl
            << " - 1xTF32 error with F64 reference        : " << l2_norm_1xtf32_vs_f64 << std::endl
#if CUTLASS_ENABLE_CUBLAS
            << " - 3xTF32 error with cuBLAS F32 reference : " << l2_norm_3xtf32_vs_cublasf32 << std::endl
#endif
            << " - 3xTF32 error with 1xTF32 reference     : " << l2_norm_3xtf32_vs_1xtf32 << std::endl;

  return true;
}

int main(int argc, const char **argv) {

  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
  // in CUDA 11.0.
  //
  // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  bool result = true;

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  result = run(options);

  if (!result) return -1;

  return 0;
}
