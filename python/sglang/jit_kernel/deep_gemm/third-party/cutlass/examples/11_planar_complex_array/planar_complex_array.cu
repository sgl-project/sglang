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
    \brief Planar Complex Array Example

  This example demonstrates the CUTLASS Library's exposure of planar complex GEMM kernels which
  execute a batch of matrix products, loading problem sizes and matrix base pointers from arrays
  in global memory.

  These kernels represent complex matrices by storing the real and imaginary parts of the matrix in
  disjoint regions in memory. These real-valued matrices are stored using existing cuBLAS layouts
  as either column-major or row-major layouts with a single leading dimension indicating the stride
  between columns or rows.

  The CUTLASS Library collects multiple template instantiations in a data structure and offers
  a BLAS-like dispatch API to invoke the appropriate kernel on the Volta or Turing architectures.

  CUTLASS decouples matrix layout from complex transformation, so four possible transformations
  are possible on the A and B operands:

    n:  column-major
    c:  column-major complex conjugate
    t:  row-major
    h:  row-major complex conjugate

  To build strictly the planar complex kernels needed for general application, execute the following
  CMake command in an empty build directory.

    $ cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" \
      -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_*gemm_planar_complex

  This builds all planar complex GEMM variants for Volta and Turing architectures.

  To build strictly the kernels needed for this example, an even narrower filter string may be
  specified as follows. This only builds planar complex GEMMs targeting Tensor Cores for
  the 'CN' layout configuration (conjugate A operand with both A and B as column-major).

    $ cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" \
      -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_f16_s*gemm_planar_complex_array_f16*cn

    $ make 11_planar_complex_array

    $ ./examples/11_planar_complex_array/11_planar_complex_array --m=2048 --n=1024 --k=512 --batch=10
*/

#include <iostream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor_planar_complex.h"

#include "cutlass/util/reference/device/tensor_fill.h"

#include "cutlass/util/reference/device/gemm_planar_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "cutlass/library/handle.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  cutlass::gemm::GemmCoord problem_size;
  int batch_count;
  cutlass::complex<float> alpha;
  cutlass::complex<float> beta;

  bool reference_check;
  int iterations;
  
  Options():
    help(false),
    problem_size({1024, 1024, 1024}),
    batch_count(1),
    reference_check(true),
    iterations(20),
    alpha(1),
    beta() { }

  bool valid() {
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
    cmd.get_cmd_line_argument("k", problem_size.k());
    cmd.get_cmd_line_argument("batch", batch_count);

    cmd.get_cmd_line_argument("alpha", alpha.real());
    cmd.get_cmd_line_argument("alpha_i", alpha.imag());
    cmd.get_cmd_line_argument("beta", beta.real());
    cmd.get_cmd_line_argument("beta_i", beta.imag());
    
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "11_planar_complex_array example\n\n"
      << "  This example uses the CUTLASS Library to execute Planar Complex Array GEMM computations.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --batch=<int>               Number of GEMM operations executed in one batch\n"
      << "  --alpha=<f32>               Epilogue scalar alpha (real part)\n"
      << "  --alpha_i=<f32>             Epilogue scalar alpha (imaginary part)\n"
      << "  --beta=<f32>                Epilogue scalar beta (real part)\n\n"
      << "  --beta_i=<f32>              Epilogue scalar beta (imaginary part)\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/11_planar_complex_array/11_planar_complex_array\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = problem_size.product() * batch_count * 4;
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Performance test environment for planar complex
class TestbedPlanarComplex {
public:

  // Half-precision input and output
  using Element = cutlass::half_t;

  // Configurations for layouts and internal computation
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementCompute = float;
  using ElementAccumulator = float;

  //
  // Data members
  //

  cutlass::library::Handle handle;

  cutlass::gemm::GemmCoord problem_size;
  int batch_count;
  cutlass::DeviceAllocation<Element> tensor_A;
  cutlass::DeviceAllocation<Element> tensor_B;
  cutlass::DeviceAllocation<Element> tensor_C;
  cutlass::DeviceAllocation<Element> tensor_D;
  cutlass::DeviceAllocation<Element> tensor_D_ref;

  cutlass::DeviceAllocation<void *> ptr_A_real;
  cutlass::DeviceAllocation<void *> ptr_A_imag;
  cutlass::DeviceAllocation<void *> ptr_B_real;
  cutlass::DeviceAllocation<void *> ptr_B_imag;
  cutlass::DeviceAllocation<void *> ptr_C_real;
  cutlass::DeviceAllocation<void *> ptr_C_imag;
  cutlass::DeviceAllocation<void *> ptr_D_real;
  cutlass::DeviceAllocation<void *> ptr_D_imag;

  //
  // Methods
  //

  TestbedPlanarComplex(
    Options const &options
  ): 
    problem_size(options.problem_size), batch_count(options.batch_count) {

    // Allocate device memory for batched planar complex GEMM
    tensor_A.reset(int64_t(problem_size.m()) * problem_size.k() * batch_count * 2);
    tensor_B.reset(int64_t(problem_size.k()) * problem_size.n() * batch_count * 2);
    tensor_C.reset(int64_t(problem_size.m()) * problem_size.n() * batch_count * 2);
    tensor_D.reset(int64_t(problem_size.m()) * problem_size.n() * batch_count * 2);
    tensor_D_ref.reset(int64_t(problem_size.m()) * problem_size.n() * batch_count * 2);

    ptr_A_real.reset(batch_count);
    ptr_A_imag.reset(batch_count);
    ptr_B_real.reset(batch_count);
    ptr_B_imag.reset(batch_count);
    ptr_C_real.reset(batch_count);
    ptr_C_imag.reset(batch_count);
    ptr_D_real.reset(batch_count);
    ptr_D_imag.reset(batch_count);

  }

  void initialize() {

    uint64_t seed = 1073;

    // Use small integers to simplify correctness checking
    int scope_max = 6;
    int scope_min = -6;

    cutlass::reference::device::BlockFillRandomUniform(
        tensor_A.get(), tensor_A.size(), seed, Element(scope_max), Element(scope_min), 0);

    cutlass::reference::device::BlockFillRandomUniform(
        tensor_B.get(), tensor_B.size(), seed * 2019, Element(scope_max), Element(scope_min), 0);

    cutlass::reference::device::BlockFillRandomUniform(
        tensor_C.get(), tensor_C.size(), seed * 2020, Element(scope_max), Element(scope_min), 0);
  }

  Result profile(Options const &options) {

    Result result;

    initialize();

    Element *ptr_A = tensor_A.get();
    Element *ptr_B = tensor_B.get();
    Element *ptr_C = tensor_C.get();
    Element *ptr_D = tensor_D.get();

    int64_t batch_stride_A = int64_t(problem_size.m()) * problem_size.k() * 2;
    int64_t batch_stride_B = int64_t(problem_size.k()) * problem_size.n() * 2;
    int64_t batch_stride_C = int64_t(problem_size.m()) * problem_size.n() * 2;
    int64_t batch_stride_D = int64_t(problem_size.m()) * problem_size.n() * 2;

    typename LayoutA::Stride::Index lda = LayoutA::packed({problem_size.m(), problem_size.k()}).stride(0);
    typename LayoutB::Stride::Index ldb = LayoutB::packed({problem_size.k(), problem_size.n()}).stride(0);
    typename LayoutC::Stride::Index ldc = LayoutC::packed({problem_size.m(), problem_size.n()}).stride(0);
    typename LayoutC::Stride::Index ldd = LayoutC::packed({problem_size.m(), problem_size.n()}).stride(0);


    int64_t imag_stride_A = int64_t(problem_size.m()) * problem_size.k();
    int64_t imag_stride_B = int64_t(problem_size.k()) * problem_size.n();
    int64_t imag_stride_C = int64_t(problem_size.m()) * problem_size.n();
    int64_t imag_stride_D = int64_t(problem_size.m()) * problem_size.n();
    
    //
    // Configure pointers in global memory
    //

    struct {
      Element *base;
      void **ptr_real;
      void **ptr_imag;
      int64_t batch_stride;
      int64_t imag_stride;
    } tensors[] = {
      { tensor_A.get(), ptr_A_real.get(), ptr_A_imag.get(), batch_stride_A, imag_stride_A},
      { tensor_B.get(), ptr_B_real.get(), ptr_B_imag.get(), batch_stride_B, imag_stride_B},
      { tensor_C.get(), ptr_C_real.get(), ptr_C_imag.get(), batch_stride_C, imag_stride_C},
      { tensor_D.get(), ptr_D_real.get(), ptr_D_imag.get(), batch_stride_D, imag_stride_D}
    };

    for (auto const &tensor : tensors) {
      for (int idx = 0; idx < batch_count; ++idx) {

        void *ptr_real = tensor.base + idx * tensor.batch_stride;
        void *ptr_imag = tensor.base + idx * tensor.batch_stride + tensor.imag_stride;

        cudaError_t error = cudaMemcpy(
          tensor.ptr_real + idx,
          &ptr_real,
          sizeof(void *),
          cudaMemcpyHostToDevice);

        if (error != cudaSuccess) {
          throw std::runtime_error("Failed to copy pointer to device memory");
        }

        error = cudaMemcpy(
          tensor.ptr_imag + idx,
          &ptr_imag,
          sizeof(void *),
          cudaMemcpyHostToDevice);

        if (error != cudaSuccess) {
          throw std::runtime_error("Failed to copy pointer to device memory");
        }
      }
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
      }
    }

    // Record an event at the start of a series of GEMM operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    for (int iter = 0; iter < options.iterations; ++iter) {

      //
      // Execute the planar complex array GEMM kernel via the CUTLASS Library's
      // dispatch routines.
      //
      // Note, for planar complex array GEMM kernels, all numeric type arguments 
      // specify the data type of the base real types. These are understood to
      // apply to planar complex representations of matrices in memory and to complex<T>
      // structures for scalars.
      //
      // See tools/library/include/cutlass/library/handle.h for more details.
      //

      result.status = handle.gemm_planar_complex_array(

        problem_size.m(),                                 // expected GEMM M dimension
        problem_size.n(),                                 // expected GEMM N dimension
        problem_size.k(),                                 // expected GEMM K dimension
        batch_count,                                      // Number of batched elements

        nullptr,
        nullptr,
        nullptr,

        cutlass::library::NumericTypeID::kF32,            // Base data type of complex-valued accumulation
        cutlass::library::NumericTypeID::kF32,            // Base data type of complex-valued alpha/beta scalars

        &options.alpha,                                   // Pointer to alpha scalar, of type complex<T>

        cutlass::library::NumericTypeID::kF16,            // Base data type of complex-valued A matrix
        cutlass::library::LayoutTypeID::kColumnMajor,     // Layout of A matrix
        cutlass::library::ComplexTransform::kConjugate,   // Complex transformation on A matrix operand

        ptr_A_real.get(),                                 // Pointer to array of pointers to real part of A matrix
        ptr_A_imag.get(),                                 // Pointer to array of pointers to imaginary part of A matrix

        lda,                                              // Leading dimension of real part of A matrix
        lda,                                              // Leading dimension of imaginary part of A matrix

        cutlass::library::NumericTypeID::kF16,            // Base data type of complex-valued B matrix
        cutlass::library::LayoutTypeID::kColumnMajor,     // Layout of B matrix
        cutlass::library::ComplexTransform::kNone,        // Complex transformation on B matrix operand

        ptr_B_real.get(),                                 // Pointer to array of pointers to real part of B matrix
        ptr_B_imag.get(),                                 // Pointer to array of pointers to imaginary part of B matrix

        ldb,                                              // Leading dimension of real part of B matrix
        ldb,                                              // Leading dimension of imaginary part of B matrix

        &options.beta,                                    // Pointer to beta scalar, of type complex<T>

        cutlass::library::NumericTypeID::kF16,            // Base data type of complex valued C and D matrices

        ptr_C_real.get(),                                 // Pointer to array of pointers to real part of C matrix
        ptr_C_imag.get(),                                 // Pointer to array of pointers to imaginary part of C matrix

        ldc,                                              // Leading dimension of real part of C matrix
        ldc,                                              // Leading dimension of imaginary part of C matrix

        ptr_D_real.get(),                                 // Pointer to array of pointers to real part of D matrix
        ptr_D_imag.get(),                                 // Pointer to array of pointers to imaginary part of D matrix

        ldd,                                              // Leading dimension of real part of D matrix
        ldd                                               // Leading dimension of imaginary part of D matrix
      );

      if (result.status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS internal error - configuration not supported" << std::endl;
        return result;
      }
    }
    
    //
    // Stop profiling loop
    //

    // Record an event when the GEMM operations have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    // Cleanup
    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    if (handle.get_last_operation()) {
      std::cout << "Recently executed '" << handle.get_last_operation()->description().name << "'" << std::endl;
    }

    //
    // Compute reference in device code
    //

    if (options.reference_check) {

      result.passed = true;

      for (int64_t idx = 0; result.passed && idx < int64_t(batch_count); ++idx) {
        cutlass::reference::device::GemmPlanarComplex<
          Element, LayoutA,
          Element, LayoutB,
          Element, LayoutC,
          ElementAccumulator
        >(
          problem_size,
          options.alpha,
          {tensor_A.get() + idx * batch_stride_A, lda, imag_stride_A},
          cutlass::ComplexTransform::kConjugate,
          {tensor_B.get() + idx * batch_stride_B, ldb, imag_stride_B},
          cutlass::ComplexTransform::kNone,
          options.beta,
          {tensor_C.get() + idx * batch_stride_C, ldc, imag_stride_C},
          {tensor_D_ref.get() + idx * batch_stride_D, ldd, imag_stride_D}
        );

        Element epsilon = 0.1_hf;
        Element nonzero_floor = 0.1_hf;

        result.passed = cutlass::reference::device::BlockCompareRelativelyEqual(
          tensor_D.get() + idx * batch_stride_D,
          tensor_D_ref.get() + idx * batch_stride_D,
          batch_stride_D,
          epsilon,
          nonzero_floor
        );
      }

      if (result.passed) {
        std::cout << "Reference check passed." << std::endl;
      }
      else {
        std::cerr << "Error - reference check failed." << std::endl;
      }
    }

    std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << " GFLOPs: " << result.gflops << std::endl;

    return result;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  //
  // This example uses mma.sync to directly access Tensor Cores to achieve peak performance.
  //
  // Volta Tensor Core operations are first available in CUDA 10.1 Toolkit.
  //
  // Turing Tensor Core operations are first available in CUDA 10.2 Toolkit.
  //

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major < 7) {
    std::cerr << "Tensor Core operations must be run on a machine with compute capability at least 70."
              << std::endl;

    // Returning zero so this passes on older architectures. Its actions are no-op.
    return 0;
  }
  else if (props.major == 7 && props.minor <= 2) {
    //
    // If running on the Volta architecture, at least CUDA 10.1 Toolkit is required to run this example.
    //
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
      std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;
      
      // Returning zero so this passes on older Toolkits. Its actions are no-op.
      return 0;
    }
  }
  else if (props.major == 7 && props.minor >= 5) {
    //
    // If running on the Turing architecture, at least CUDA 10.2 Toolkit is required to run this example.
    //
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
      std::cerr << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
      
      // Returning zero so this passes on older Toolkits. Its actions are no-op.
      return 0;
    }
  }
  else {
    // NVIDIA Ampere Architecture GPUs (SM80 and later) are fully supported on CUDA 11 Toolkit and beyond.
    //
    // fall through
  }

  //
  // Parse options
  //

  Options options;
  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  // Execute one problem size
  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  TestbedPlanarComplex testbed(options);

  Result result = testbed.profile(options);

  return result.passed ? 0 : -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

