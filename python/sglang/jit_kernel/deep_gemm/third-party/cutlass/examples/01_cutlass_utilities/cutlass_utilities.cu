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

/*
  This example demonstrates several CUTLASS utilities in the context of a mixed-precision
  floating-point matrix product computation.

  These utilities are intended to be useful supporting components for managing tensor and matrix
  memory allocations, initializing and comparing results, and computing reference output.

  CUTLASS utilities are defined in the directory `tools/util`, and definitions appear
  namespace `cutlass::` or an inner namespace therein. Operations in `cutlass::reference::` have
  both host-side and device-side implementations, and the choice to use device-side initialization
  and host-side verification in this example was arbitrary.


  cutlass::half_t

    This is a numeric type implementing IEEE half-precision quantities. It is functional in host
    and device code. In host-side code, CUTLASS_ENABLE_F16C optionally enables hardware-accelerated
    numeric conversion on x86-64 CPUs support F16C extensions. In device code, all available
    hardware is used to implement conversion and numeric operations.


  cutlass::HostTensor<>

    This template class simplifies the creation of tensors for all supported layouts. It simplifies
    allocation and management of host- and device- memory allocations.

    This class offers methods device_view() and host_view() to provide TensorView objects for
    device- and host-side memory allocations.


  cutlass::reference::device::TensorFillRandomGaussian()

    This template function initializes elementsof a tensor to a random Gaussian distribution. It
    uses cuRAND in device code to compute random numbers.


  cutlass::reference::host::Gemm<>

    This template function computes the general matrix product. This template supports unique
    data types for each matrix operand, the internal accumulation type, and the scalar parameters
    alpha and beta.


  cutlass::reference::host::TensorEquals()

    Compares two tensors of identical rank and returns true if values are bit equivalent.

*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>

// CUTLASS includes needed for half-precision GEMM kernel
#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"

//
// CUTLASS utility includes
//

// Defines operator<<() to write TensorView objects to std::ostream
#include "cutlass/util/tensor_view_io.h"

// Defines cutlass::HostTensor<>
#include "cutlass/util/host_tensor.h"

// Defines cutlass::half_t
#include "cutlass/numeric_types.h"

// Defines device_memory::copy_device_to_device()
#include "cutlass/util/device_memory.h"

// Defines cutlass::reference::device::TensorFillRandomGaussian()
#include "cutlass/util/reference/device/tensor_fill.h"

// Defines cutlass::reference::host::TensorEquals()
#include "cutlass/util/reference/host/tensor_compare.h"

// Defines cutlass::reference::host::Gemm()
#include "cutlass/util/reference/host/gemm.h"

#pragma warning( disable : 4503)
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t cutlass_hgemm_nn(
  int M,
  int N,
  int K,
  cutlass::half_t alpha,
  cutlass::half_t const *A,
  cutlass::layout::ColumnMajor::Stride::Index lda,
  cutlass::half_t const *B,
  cutlass::layout::ColumnMajor::Stride::Index ldb,
  cutlass::half_t beta,
  cutlass::half_t *C,
  cutlass::layout::ColumnMajor::Stride::Index ldc) {

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::ColumnMajor,              // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementOutput
    cutlass::layout::ColumnMajor               // LayoutOutput
  >;

  Gemm gemm_op;
  
  cutlass::Status status = gemm_op({
    {M, N, K},
    {A, lda},
    {B, ldb},
    {C, ldc},
    {C, ldc},
    {alpha, beta}
  });

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, cutlass::half_t alpha, cutlass::half_t beta) {
  cudaError_t result;

  //
  // Construct cutlass::HostTensor<> using the half-precision host-side type.
  //
  // cutlass::HostTensor<> allocates memory on both the host and device corresponding to rank=2
  // tensors in column-major layout. Explicit synchronization methods are offered to copy the
  // tensor to the device or to the host.
  //

  // M-by-K matrix of cutlass::half_t
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A(cutlass::MatrixCoord(M, K));

  // K-by-N matrix of cutlass::half_t
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(cutlass::MatrixCoord(K, N));

  // M-by-N matrix of cutlass::half_t
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_cutlass(cutlass::MatrixCoord(M, N));

  // M-by-N matrix of cutlass::half_t
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_reference(cutlass::MatrixCoord(M, N));

  //
  // Initialize matrices with small, random integers.
  //

  // Arbitrary RNG seed value. Hard-coded for deterministic results.
  uint64_t seed = 2080;

  // Gaussian random distribution
  cutlass::half_t mean = 0.0_hf;
  cutlass::half_t stddev = 5.0_hf;

  // Specify the number of bits right of the binary decimal that are permitted
  // to be non-zero. A value of "0" here truncates random values to integers
  int bits_less_than_one = 0;

  cutlass::reference::device::TensorFillRandomGaussian(
    A.device_view(),
    seed,
    mean,
    stddev,
    bits_less_than_one
  );
  
  cutlass::reference::device::TensorFillRandomGaussian(
    B.device_view(),
    seed * 2019,
    mean,
    stddev,
    bits_less_than_one
  );
  
  cutlass::reference::device::TensorFillRandomGaussian(
    C_cutlass.device_view(),
    seed * 1993,
    mean,
    stddev,
    bits_less_than_one
  );


  // Copy C_cutlass into C_reference so the GEMM is correct when beta != 0.
  cutlass::device_memory::copy_device_to_device(
    C_reference.device_data(), 
    C_cutlass.device_data(), 
    C_cutlass.capacity());

  // Copy the device-side view into host memory
  C_reference.sync_host();

  //
  // Launch the CUTLASS GEMM kernel
  //

  result = cutlass_hgemm_nn(
    M,
    N,
    K,
    alpha,
    A.device_data(),
    A.stride(0),
    B.device_data(),
    B.stride(0),
    beta,
    C_cutlass.device_data(),
    C_cutlass.stride(0)
  );

  if (result != cudaSuccess) {
    return result;
  }

  //
  // Verify the result using a host-side reference
  //

  // A and B were initialized using device-side procedures. The intent of this example is to
  // use the host-side reference GEMM, so we must perform a device-to-host copy.
  A.sync_host();
  B.sync_host();

  // Copy CUTLASS's GEMM results into host memory.
  C_cutlass.sync_host();

  // Compute the reference result using the host-side GEMM reference implementation.
  cutlass::reference::host::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::ColumnMajor,              // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementOutput
    cutlass::layout::ColumnMajor,              // LayoutOutput
    cutlass::half_t,
    cutlass::half_t
  > gemm_ref;

  gemm_ref(
    {M, N, K},                          // problem size (type: cutlass::gemm::GemmCoord)
    alpha,                              // alpha        (type: cutlass::half_t)
    A.host_ref(),                       // A            (type: TensorRef<half_t, ColumnMajor>)
    B.host_ref(),                       // B            (type: TensorRef<half_t, ColumnMajor>)
    beta,                               // beta         (type: cutlass::half_t)
    C_reference.host_ref()              // C            (type: TensorRef<half_t, ColumnMajor>)
  );

  // Compare reference to computed results.
  if (!cutlass::reference::host::TensorEquals(
    C_reference.host_view(), 
    C_cutlass.host_view())) {

    char const *filename = "errors_01_cutlass_utilities.csv";

    std::cerr << "Error - CUTLASS GEMM kernel differs from reference. Wrote computed and reference results to '" << filename << "'" << std::endl;

    //
    // On error, print C_cutlass and C_reference to std::cerr.
    //
    // Note, these are matrices of half-precision elements stored in host memory as
    // arrays of type cutlass::half_t.
    //

    std::ofstream file(filename);

    // Result of CUTLASS GEMM kernel
    file << "\n\nCUTLASS =\n" << C_cutlass.host_view() << std::endl;

    // Result of reference computation
    file << "\n\nReference =\n" << C_reference.host_view() << std::endl;

    // Return error code.
    return cudaErrorUnknown;
  }

  // Passed error check
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to cutlass_utilities example.
//
// usage:
//
//   01_cutlass_utilities <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

  //
  // This example uses half-precision and is only suitable for devices with compute capabitliy 5.3 or greater.
  //

  cudaDeviceProp prop;
  cudaError_t result = cudaGetDeviceProperties(&prop, 0);
  
  if (result != cudaSuccess) {
    std::cerr << "Failed to query device properties with error " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  if (!(prop.major > 5 || (prop.major == 5 && prop.minor >= 3))) {
    std::cerr << "This example uses half precision and is only suitable for devices with compute capability 5.3 or greater.\n";
    std::cerr << "You are using a CUDA device with compute capability " << prop.major << "." << prop.minor << std::endl;
    return -1;
  }

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions: <M> <N> <K>
  int problem[3] = { 128, 128, 128 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Linear scale factors in GEMM. Note, these are half-precision values stored as
  // cutlass::half_t.
  //
  // Values outside the range of IEEE FP16 will overflow to infinity or underflow to zero.
  //
  cutlass::half_t scalars[2] = { 1.0_hf, 0.0_hf };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);

    ss >> scalars[i - 4];   // lexical cast to cutlass::half_t
  }

  //
  // Run the CUTLASS GEMM test.
  //

  result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1]      // beta
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

