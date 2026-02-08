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
    \brief BLAS-like handle used to launch operations on the CUDA device.
*/

#pragma once

#include <memory>
#include "cutlass/library/library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Handle object
class Handle {
private:

  /// Host workspace
  static int const kHostWorkspaceSize = (4 << 10);

  /// Provider of operations
  Provider provider_;

  /// CUDA device properties
  cudaDeviceProp device_;

  /// CUDA stream
  cudaStream_t stream_;

  /// Device workspace
  void *workspace_;

  /// Size of device workspace in bytes
  size_t workspace_size_;

  /// Indicates whether scalars are host or device pointers
  ScalarPointerMode scalar_pointer_mode_;

  /// Pointer to the most recently executed operation
  Operation const *last_operation_;

  int device_idx_;

public:

  /// Constructor
  Handle(cudaStream_t stream = nullptr, size_t workspace_size = (4<<20));

  /// Destructor
  ~Handle();

  /// Move constructor
  Handle(Handle && handle);

  /// Move assignment operator
  Handle &operator=(Handle && handle);

  //
  // Persistent state accessors
  //

  /// Returns compute capability of the selected device
  int compute_capability() const;

  /// Sets the current CUDA stream
  void set_stream(cudaStream_t stream);

  /// Gets the current CUDA stream
  cudaStream_t get_stream() const;

  /// Gets the current provider
  Provider get_provider() const;

  /// Sets the provider of operations
  void set_provider(Provider provider);

  /// Gets the device workspace size
  size_t get_workspace_size() const;

  /// Gets a pointer to the device workspace allocation in Global Memory
  void *get_workspace() const;

  /// Sets the size of device workspace, invalidating calls to get_device_workspace()
  void set_workspace_size(size_t bytes);

  /// Gets the scalar pointer mode
  ScalarPointerMode get_scalar_pointer_mode() const;

  /// Sets the scalar pointer mode
  void set_scalar_pointer_mode(ScalarPointerMode mode);

  /// Gets the most recently executed operation
  Operation const *get_last_operation() const;

  //
  // Computations
  //

  /// Executes a GEMM computation: D <= alpha * A*B + beta * C
  Status gemm(

    int M,                                    /// GEMM M dimension
    int N,                                    /// GEMM N dimension
    int K,                                    /// GEMM K dimension

    NumericTypeID element_compute,            /// Data type of internal accumulation

    NumericTypeID element_scalar,             /// Data type of alpha/beta scalars

    void const *alpha,                        /// Pointer to alpha scalar

    NumericTypeID element_A,                  /// Data type of A matrix elements
    LayoutTypeID layout_A,                    /// Layout of A matrix
    ComplexTransform transform_A,             /// Complex transformation applied to A matrix - ignored for real-valued matrices

    void const * ptr_A,                       /// Pointer to A matrix in Global Memory
    int64_t lda,                              /// Leading dimension of A matrix

    NumericTypeID element_B,                  /// Data type of B matrix elements
    LayoutTypeID layout_B,                    /// Layout of B matrix
    ComplexTransform transform_B,             /// Complex transformation applied to B matrix - ignored for real-valued matrices

    void const * ptr_B,                       /// Pointer to B matrix in Global Memory
    int64_t ldb,                              /// Leading dimension of B matrix

    void const * beta,                        /// Pointer to beta scalar

    NumericTypeID element_C,                  /// Data type of C and D matrices

    void const * ptr_C,                       /// Pointer to C matrix
    int64_t ldc,                              /// Leading dimension of C matrix

    void * ptr_D,                             /// Pointer to D matrix
    int64_t ldd                               /// Leading dimension of D matrix
  );

  /// Executes a GEMM computation: D <= alpha * A*B + beta * C.
  //
  // Supports batched-strided, batched array or split-K serial or split-K parallel.
  //
  Status gemm_universal(

    GemmUniversalMode mode,                   /// indicates the mode in which the kUniversal GEMM is launched

    int M,                                    /// GEMM M dimension
    int N,                                    /// GEMM N dimension
    int K,                                    /// GEMM K dimension
    
    int cluster_m,                            /// cluster shape M dimension
    int cluster_n,                            /// cluster shape N dimension
    int cluster_k,                            /// cluster shape K dimension
    int cluster_m_fallback,                   /// Fallback cluster shape M dimension
    int cluster_n_fallback,                   /// Fallback cluster shape N dimension
    int cluster_k_fallback,                   /// Fallback cluster shape K dimension
    
    
    NumericTypeID element_compute,            /// Data type of internal accumulation

    NumericTypeID element_scalar,             /// Data type of alpha/beta scalars

    void const *alpha,                        /// Pointer to alpha scalar

    NumericTypeID element_A,                  /// Data type of A matrix elements
    LayoutTypeID layout_A,                    /// Layout of A matrix
    ComplexTransform transform_A,             /// Complex transformation applied to A matrix - ignored for real-valued matrices
    void const * ptr_A,                       /// Pointer to A matrix in Global Memory
    int64_t lda,                              /// Leading dimension of A matrix

    NumericTypeID element_B,                  /// Data type of B matrix elements
    LayoutTypeID layout_B,                    /// Layout of B matrix
    ComplexTransform transform_B,             /// Complex transformation applied to B matrix - ignored for real-valued matrices
    void const * ptr_B,                       /// Pointer to B matrix in Global Memory
    int64_t ldb,                              /// Leading dimension of B matrix

    void const * beta,                        /// Pointer to beta scalar

    NumericTypeID element_C,                  /// Data type of C matrix
    LayoutTypeID layout_C,                    /// Layout of D matrix
    void const * ptr_C,                       /// Pointer to C matrix
    int64_t ldc,                              /// Leading dimension of C matrix

    NumericTypeID element_D,                  /// Data type of D matrix
    LayoutTypeID layout_D,                    /// Layout of D matrix
    void * ptr_D,                             /// Pointer to D matrix
    int64_t ldd,                              /// Leading dimension of D matrix

    int batch_count = 1,                      /// Batch count or number of split-K slices

    int64_t batch_stride_A = 0,               /// Batch stride of A operand
    int64_t batch_stride_B = 0,               /// Batch stride of B operand
    int64_t batch_stride_C = 0,               /// Batch stride of C operand
    int64_t batch_stride_D = 0                /// Batch stride of D operand
  );

  /// Planar complex GEMM
  ///
  /// Note, all data types are the real-valued base types used by the planar-complex GEMM kernel.
  ///
  Status gemm_planar_complex(

    int M,                                    /// GEMM M dimension
    int N,                                    /// GEMM N dimension
    int K,                                    /// GEMM K dimension

    NumericTypeID element_compute,            /// Data type of internal accumulation

    NumericTypeID element_scalar,             /// Data type of alpha/beta scalars

    void const *alpha,                        /// Pointer to alpha scalar

    NumericTypeID element_A,                  /// Data type of A matrix elements
    LayoutTypeID layout_A,                    /// Layout of A matrix
    ComplexTransform transform_A,             /// Complex transformation applied to A matrix

    void const * ptr_A_real,                  /// Pointer to real part of A matrix
    void const * ptr_A_imag,                  /// Pointer to imaginary part of A matrix
    int64_t lda_real,                         /// Leading dimension of real part of A matrix
    int64_t lda_imag,                         /// Leading dimension of imaginary part of A matrix

    NumericTypeID element_B,                  /// Data type of B matrix elements
    LayoutTypeID layout_B,                    /// Layout of B matrix
    ComplexTransform transform_B,             /// Complex transformation applied to B matrix

    void const * ptr_B_real,                  /// Pointer to real part of B matrix
    void const * ptr_B_imag,                  /// Pointer to imaginary part of B matrix
    int64_t ldb_real,                         /// Leading dimension of real part of B matrix
    int64_t ldb_imag,                         /// Leading dimension of imaginary part of B matrix

    void const * beta,                        /// Pointer to beta scalar

    NumericTypeID element_C,                  /// Data type of C and D matrix

    void const * ptr_C_real,                  /// Pointer to real part of C matrix
    void const * ptr_C_imag,                  /// Pointer to imaginary part of C matrix
    int64_t ldc_real,                         /// Leading dimension of real part of C matrix
    int64_t ldc_imag,                         /// Leading dimension of imaginary part of C matrix

    void * ptr_D_real,                        /// Pointer to real part of D matrix
    void * ptr_D_imag,                        /// Pointer to imaginary part of D matrix
    int64_t ldd_real,                         /// Leading dimension of real part of D matrix
    int64_t ldd_imag,                         /// Leading dimension of imaginary part of D matrix

    int batch_count = 1,                      /// Number of batched GEMMs to execute

    int64_t batch_stride_A_real = 0,
    int64_t batch_stride_A_imag = 0,

    int64_t batch_stride_B_real = 0,
    int64_t batch_stride_B_imag = 0,

    int64_t batch_stride_C_real = 0,
    int64_t batch_stride_C_imag = 0,

    int64_t batch_stride_D_real = 0,
    int64_t batch_stride_D_imag = 0
  );

  /// Planar complex GEMM loading pointers from arrays in global memory
  Status gemm_planar_complex_array(

    int expected_M,                           /// Expected GEMM M dimension (used for sizing CUDA grid)
    int expected_N,                           /// Expected GEMM N dimension (used for sizing CUDA grid)
    int expected_K,                           /// Expected GEMM K dimension
    int batch_count,                          /// Number of independent GEMM computations to execute

    int const *M,                             /// Array containing the GEMM M dimension for each batch index
    int const *N,                             /// Array containing the GEMM N dimension for each batch index
    int const *K,                             /// Array containing the GEMM K dimension for each batch index

    NumericTypeID element_compute,            /// Data type of internal accumulation

    NumericTypeID element_scalar,             /// Data type of alpha/beta scalars

    void const *alpha,                        /// Pointer to alpha scalar

    NumericTypeID element_A,                  /// Data type of A matrix elements
    LayoutTypeID layout_A,                    /// Layout of A matrix
    ComplexTransform transform_A,             /// Complex transformation applied to A matrix

    void const * const * ptr_A_real,          /// Pointer to array containing pointers to real part of A matrices
    void const * const * ptr_A_imag,          /// Pointer to array containing pointers to imaginary part of A matrices

    int64_t lda_real,                         /// Leading dimension of real part of A matrix
    int64_t lda_imag,                         /// Leading dimension of imaginary part of A matrix

    NumericTypeID element_B,                  /// Data type of B matrix elements
    LayoutTypeID layout_B,                    /// Layout of B matrix
    ComplexTransform transform_B,             /// Complex transformation applied to B matrix

    void const * const * ptr_B_real,          /// Pointer to array containing pointers to real part of B matrices
    void const * const * ptr_B_imag,          /// Pointer to array containing pointers to imaginary part of B matrices

    int64_t ldb_real,                         /// Leading dimension of real part of B matrix
    int64_t ldb_imag,                         /// Leading dimension of imaginary part of B matrix

    void const * beta,                        /// Pointer to beta scalar

    NumericTypeID element_C,                  /// Data type of C and D matrix

    void const * const * ptr_C_real,          /// Pointer to array containing pointers to real part of C matrices
    void const * const * ptr_C_imag,          /// Pointer to array containing pointers to imaginary part of C matrices

    int64_t ldc_real,                         /// Leading dimension of real part of C matrix
    int64_t ldc_imag,                         /// Leading dimension of imaginary part of C matrix

    void * const * ptr_D_real,                /// Pointer to array containing pointers to real part of D matrices
    void * const * ptr_D_imag,                /// Pointer to array containing pointers to imaginary part of D matrices

    int64_t ldd_real,                         /// Leading dimension of real part of D matrix
    int64_t ldd_imag                          /// Leading dimension of imaginary part of D matrix
  );

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Unique pointer storing the handle
using HandlePtr = std::unique_ptr<Handle>;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Finds conv2d operation instances with Conv2d::ElementC = Reduction::ElementWorkspace
Operation const* find_conv_operation_for_parallel_reduction(Operation const *operation);
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Finds gemm operation instances with ElementC = Reduction::ElementWorkspace
Operation const* find_gemm_operation_for_parallel_reduction(Operation const *operation);
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

