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
/*!
  \file

  \brief CUTLASS Library is an object-oriented approach to managing operations implemented by CUTLASS.

  Generally,

    description   - compile-time constant parameters used to instantiate an operation

    configuration - runtime parameters with computationally expensive initialization

    arguments     - runtime parameters that may be passed to an initialized operation with low
                    computational overhead
*/

#ifndef CUTLASS_LIBRARY_LIBRARY_H
#define CUTLASS_LIBRARY_LIBRARY_H

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/library/types.h"
#include "cutlass/library/descriptions.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/blas3.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Mode of Universal GEMM
using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Base class for all operations
class Operation {
public:

  virtual ~Operation() { }

  virtual OperationDescription const & description() const = 0;

  virtual Status can_implement(
    void const *configuration,
    void const *arguments) const = 0;

  virtual uint64_t get_host_workspace_size(
    void const *configuration) const = 0;

  virtual uint64_t get_device_workspace_size(
    void const *configuration,
    void const *arguments = nullptr) const = 0;

  virtual Status initialize(
    void const *configuration,
    void *host_workspace,
    void *device_workspace = nullptr,
    cudaStream_t stream = nullptr) const = 0;

  // Originally designed for metadata, but should be useful for FP8/6/4 too.  
  virtual Status initialize_with_profiler_workspace(
    void const *configuration,
    void *host_workspace,
    void *device_workspace,
    uint8_t **profiler_workspace_ptrs,
    int problem_count,
    cudaStream_t stream = nullptr) {
    return Status::kErrorNotSupported;
  }

  virtual Status run(
    void const *arguments,
    void *host_workspace,
    void *device_workspace = nullptr,
    cudaStream_t stream = nullptr) const = 0;

  // Set arguments that should only be set once before verifying or profiling the kernel.
  // This should encompass any expensive operations that don't vary from run to run
  // (e.g., max_active_clusters).
  virtual Status initialize_with_arguments(void* arguments_ptr) const {
    return Status::kSuccess;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for basic GEMM operations
//
// OperationKind: Gemm
// GemmKind:      Gemm
//
struct GemmConfiguration {

  /// GEMM problem size
  gemm::GemmCoord problem_size{};

  /// Leading dimension of A matrix
  int64_t lda{0};

  /// Leading dimension of B matrix
  int64_t ldb{0};

  /// Leading dimension of C matrix
  int64_t ldc{0};

  /// Leading dimension of D matrix
  int64_t ldd{0};

  /// Number of partitions of K dimension
  int split_k_slices{0};
};

/// Arguments for GEMM
struct GemmArguments {

  /// Pointer to A matrix
  void const *A{nullptr};

  /// Pointer to B matrix
  void const *B{nullptr};

  /// Pointer to C matrix
  void const *C{nullptr};

  /// Pointer to D matrix
  void *D{nullptr};

  /// Host or device pointer to alpha scalar
  void const *alpha{nullptr};

  /// Host or device pointer to beta scalar
  void const *beta{nullptr};

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode{};
  
  /// Whether to use PDL when launching the kernel
  bool use_pdl{false};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for batched GEMM in which multiple matrix products are computed
//
// OperationKind: Gemm
// GemmKind:      Batched

struct GemmBatchedConfiguration {

  /// GEMM problem size
  gemm::GemmCoord problem_size{};

  /// Leading dimension of A matrix
  int64_t lda{0};

  /// Leading dimension of B matrix
  int64_t ldb{0};

  /// Leading dimension of C matrix
  int64_t ldc{0};

  /// Leading dimension of D matrix
  int64_t ldd{0};

  /// Stride between instances of the A matrix in memory
  int64_t batch_stride_A{0};

  /// Stride between instances of the B matrix in memory
  int64_t batch_stride_B{0};

  /// Stride between instances of the C matrix in memory
  int64_t batch_stride_C{0};

  /// Stride between instances of the D matrix in memory
  int64_t batch_stride_D{0};

  /// Number of GEMMs in batch
  int batch_count{1};
};

/// Arguments to batched GEMM
using GemmBatchedArguments = GemmArguments;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for batched GEMM in which multiple matrix products are computed
//
// OperationKind: Gemm
// GemmKind:      Array

struct GemmArrayConfiguration {

  gemm::GemmCoord problem_size{};

  /// Leading dimension of A matrix
  int64_t lda{0};

  /// Leading dimension of B matrix
  int64_t ldb{0};

  /// Leading dimension of C matrix
  int64_t ldc{0};

  /// Leading dimension of D matrix
  int64_t ldd{0};

  int batch_count{1};
};

/// Arguments for GEMM - used by all the GEMM operations
struct GemmArrayArguments {
  void const * const *A{nullptr};
  void const * const *B{nullptr};
  void const * const *C{nullptr};
  void * const *D{nullptr};
  void const *alpha{nullptr};
  void const *beta{nullptr};
  ScalarPointerMode pointer_mode{};
  bool use_pdl{false};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Universal GEMM supporting multiple split-K modes, multiple batched modes, real and complex
//
// OperationKind: Gemm
// GemmKind:      Universal

struct GemmUniversalConfiguration {

  GemmUniversalMode mode{GemmUniversalMode::kGemm};
  gemm::GemmCoord problem_size{};
  gemm::GemmCoord cluster_shape{};           
  gemm::GemmCoord cluster_shape_fallback{};  
  int batch_count{1};

  int64_t lda{0};
  int64_t ldb{0};
  int64_t ldc{0};
  int64_t ldd{0};

  int device_count{1};
};

enum class Sm90MixedInputWiderOperand {
  A = 0,
  B = 1
};

struct GemmUniversalArguments {
  // NOTE: these are replicated for 3.0 interfaces
  gemm::GemmCoord problem_size{};
  gemm::GemmCoord cluster_shape{};          
  gemm::GemmCoord cluster_shape_fallback{}; 
  int batch_count{1};

  void const *A{nullptr};
  void const *B{nullptr};
  void const *C{nullptr};
  void *D{nullptr};

  void const *alpha{nullptr};
  void const *beta{nullptr};
  ScalarPointerMode pointer_mode{};

  // NOTE: these are replicated for 3.0 interfaces
  int64_t lda{0};
  int64_t ldb{0};
  int64_t ldc{0};
  int64_t ldd{0};

  int64_t batch_stride_A{0};
  int64_t batch_stride_B{0};
  int64_t batch_stride_C{0};
  int64_t batch_stride_D{0};

  // Needed for some 3.x kernels
  int sm_count{0};
  library::RasterOrder raster_order{};
  library::RuntimeDatatype runtime_input_datatype_a{};
  library::RuntimeDatatype runtime_input_datatype_b{};
  int swizzle_size{1};
  int split_k_slices{1};

  // For SM90 mixed input dtype kernels
  bool is_sm90_mixed_dtype{false};
  Sm90MixedInputWiderOperand wider_operand{Sm90MixedInputWiderOperand::B};
  bool generate_scale_and_zero{false};
  bool generate_dequantized_AB{false};
  void *Scale{nullptr};                 // Scale tensor
  void *Zero{nullptr};                  // Zero tensor
  void *dequantized_AB{nullptr};        // Dequantized A or B tensor for verification
  void *encoded_AB{nullptr};            // Encoded A or B in int4 x fp8 or shuffle
  void *packed_Scale{nullptr};          // Packed scale for int4 * fp8

  int device_index{0};

  bool use_pdl{false};
};

/// Block Scaled GEMM
//
// OperationKind: kBlockScaledGemm
// GemmKind:      Universal

struct BlockScaledGemmArguments {
  // NOTE: these are replicated for 3.0 interfaces
  gemm::GemmCoord problem_size{};
  gemm::GemmCoord cluster_shape{};  
  gemm::GemmCoord cluster_shape_fallback{}; 
  int batch_count{1};

  void const *A{nullptr};
  void const *B{nullptr};
  void const *SFA{nullptr};
  void const *SFB{nullptr};
  void const *C{nullptr};
  void *D{nullptr};
  void *SFD{nullptr}; 

  void const *alpha{nullptr};
  void const *beta{nullptr};
  ScalarPointerMode pointer_mode{};

  // NOTE: these are replicated for 3.0 interfaces
  int64_t lda{0};
  int64_t ldb{0};
  int64_t ldc{0};
  int64_t ldd{0};

  int64_t batch_stride_A{0};
  int64_t batch_stride_B{0};
  int64_t batch_stride_C{0};
  int64_t batch_stride_D{0};

  // Needed for ScaleFactor Generation
  void const *norm_constant{nullptr};

  // Needed for some 3.x kernels
  int sm_count{0};
  library::RasterOrder raster_order{};
  int swizzle_size{1};
  int split_k_slices{1};

  library::RuntimeDatatype runtime_input_datatype_a{library::RuntimeDatatype::kStatic}; 
  library::RuntimeDatatype runtime_input_datatype_b{library::RuntimeDatatype::kStatic}; 

  bool use_pdl{false};
};

/// Blockwise GEMM
//
// OperationKind: kBlockwiseGemm
// GemmKind:      Universal

struct BlockwiseGemmArguments {
  // NOTE: these are replicated for 3.0 interfaces
  gemm::GemmCoord problem_size{};
  gemm::GemmCoord cluster_shape{};  
  gemm::GemmCoord cluster_shape_fallback{}; 
  int batch_count{1};

  void const *A{nullptr};
  void const *B{nullptr};
  void const *SFA{nullptr};
  void const *SFB{nullptr};
  void const *C{nullptr};
  void *D{nullptr};

  void const *alpha{nullptr};
  void const *beta{nullptr};
  ScalarPointerMode pointer_mode{};

  // NOTE: these are replicated for 3.0 interfaces
  int64_t lda{0};
  int64_t ldb{0};
  int64_t ldc{0};
  int64_t ldd{0};

  int64_t batch_stride_A{0};
  int64_t batch_stride_B{0};
  int64_t batch_stride_C{0};
  int64_t batch_stride_D{0};

  int sf_m_vec_size{0};
  int sf_n_vec_size{0};
  int sf_k_vec_size{0};

  // Needed for some 3.x kernels
  int sm_count{0};
  library::RasterOrder raster_order{};
  int swizzle_size{1};
  int split_k_slices{1};

  library::RuntimeDatatype runtime_input_datatype_a{library::RuntimeDatatype::kStatic}; 
  library::RuntimeDatatype runtime_input_datatype_b{library::RuntimeDatatype::kStatic}; 

  bool use_pdl{false};
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Complex valued GEMM in which real and imaginary parts are separated by a stride
//
// OperationKind: Gemm
// GemmKind:      Planar complex

struct GemmPlanarComplexConfiguration {

  GemmUniversalMode mode{GemmUniversalMode::kGemm};
  gemm::GemmCoord problem_size{};
  int batch_count{1};
  int64_t lda_real{0};
  int64_t lda_imag{0};
  int64_t ldb_real{0};
  int64_t ldb_imag{0};
  int64_t ldc_real{0};
  int64_t ldc_imag{0};
  int64_t ldd_real{0};
  int64_t ldd_imag{0};
};

/// Arguments for planar complex GEMMs
struct GemmPlanarComplexArguments {

  void const *A_real{nullptr};
  void const *A_imag{nullptr};
  void const *B_real{nullptr};
  void const *B_imag{nullptr};
  void const *C_real{nullptr};
  void const *C_imag{nullptr};
  void *D_real{nullptr};
  void *D_imag{nullptr};
  void const *alpha{nullptr};
  void const *beta{nullptr};
  ScalarPointerMode pointer_mode{};

  int64_t batch_stride_A_real{0};
  int64_t batch_stride_A_imag{0};
  int64_t batch_stride_B_real{0};
  int64_t batch_stride_B_imag{0};
  int64_t batch_stride_C_real{0};
  int64_t batch_stride_C_imag{0};
  int64_t batch_stride_D_real{0};
  int64_t batch_stride_D_imag{0};
  bool use_pdl{false};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This is a special form of planar complex which loads pointers and problem size
/// from memory.
struct GemmPlanarComplexArrayConfiguration {

  gemm::GemmCoord problem_size{};
  int batch_count{1};

  int64_t lda_real{0};
  int64_t lda_imag{0};
  int64_t ldb_real{0};
  int64_t ldb_imag{0};
  int64_t ldc_real{0};
  int64_t ldc_imag{0};
  int64_t ldd_real{0};
  int64_t ldd_imag{0};
};

/// Arguments for planar complex GEMMs
struct GemmPlanarComplexArrayArguments {

  int const *M{nullptr};
  int const *N{nullptr};
  int const *K{nullptr};

  void const * const * A_real{nullptr};
  void const * const * A_imag{nullptr};
  void const * const * B_real{nullptr};
  void const * const * B_imag{nullptr};
  void const * const * C_real{nullptr};
  void const * const * C_imag{nullptr};
  void * const * D_real{nullptr};
  void * const * D_imag{nullptr};

  void const * alpha{nullptr};
  void const * beta{nullptr};
  ScalarPointerMode pointer_mode{};
  bool use_pdl{false};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Grouped GEMM supporting
//
// OperationKind: Gemm
// GemmKind:      Grouped

struct GemmGroupedConfiguration {
  int problem_count{0};
  // GemmGroupedConfiguration is passed to initialize(), which
  // is responsible for allocating the device-side stride storage.
  int64_t* lda;
  int64_t* ldb;
  int64_t* ldc;

  cute::Shape<int, int, int>* problem_sizes_3x_host;
};

struct GemmGroupedArguments {
  int problem_count{};
  gemm::GemmCoord* problem_sizes{nullptr};

  void* ptr_A{nullptr};
  void* ptr_B{nullptr};
  void* ptr_C{nullptr};
  void* ptr_D{nullptr};

  int64_t* lda{nullptr};
  int64_t* ldb{nullptr};
  int64_t* ldc{nullptr};
  int64_t* ldd{nullptr};

  void const *alpha{nullptr};
  void const *beta{nullptr};
  ScalarPointerMode pointer_mode{};
  bool use_pdl{false};

  gemm::GemmCoord cluster_shape{};
  gemm::GemmCoord cluster_shape_fallback{};

  library::RasterOrder raster_order{};
  library::RuntimeDatatype runtime_input_datatype_a{library::RuntimeDatatype::kStatic};
  library::RuntimeDatatype runtime_input_datatype_b{library::RuntimeDatatype::kStatic};
  int swizzle_size{1};

  // these should really be in the configuration but staying consistent with GEMM
  int sm_count{0};
  int max_active_clusters{0};

  // The user is responsible for allocating storage for problem sizes.
  // Since GemmGroupedArguments is used by both the 2.x and 3.x APIs, we
  // unfortunately need to have both options in this struct, and the
  // underlying operation uses the one it needs.
  cute::Shape<int, int, int>* problem_sizes_3x;
  cute::Shape<int, int, int>* problem_sizes_3x_host;
};

struct GroupedGemmBlockScaledArguments : GemmGroupedArguments {
  void* SFA{nullptr};
  void* SFB{nullptr};
  void* SFD{nullptr};
  void* norm_constant{nullptr};
};

struct GroupedGemmBlockwiseArguments : GemmGroupedArguments {
  void* SFA{nullptr};
  void* SFB{nullptr};
};


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// OperationKind: kSparseGemm
//

/// Computes GEMM assuming one of the inputs has 2:4 structured sparsity.
struct SparseGemmConfiguration {

  GemmUniversalMode mode{GemmUniversalMode::kGemm};
  gemm::GemmCoord problem_size{};
  int batch_count{1};         /// number of sparse matrix products in batch
  int64_t lda{0};             /// leading dimension of A operand
  int64_t ldb{0};             /// leading dimension of B operand
  int64_t ldc{0};             /// leading dimension of C operand
  int64_t ldd{0};             /// leading dimension of D operand
  int64_t lde{0};             /// leading dimension of E operand (metadata matrix)
  int64_t batch_stride_A{0};  // stride between matrices
  int64_t batch_stride_B{0};  // stride between matrices
  int64_t batch_stride_C{0};  // stride between matrices
  int64_t batch_stride_D{0};  // stride between matrices
  int64_t batch_stride_E{0};  // stride between matrices
};

/// Arguments for sparse GEMMs
struct SparseGemmArguments {
  void const *A{nullptr};          /// pointer to A matrix
  void const *B{nullptr};          /// pointer to B matrix
  void const *C{nullptr};          /// pointer to C matrix
  void *D{nullptr};                  /// pointer to D matrix
  void const *E{nullptr};          /// pointer to E matrix (metadata)
  void const *alpha{nullptr};      /// pointer to alpha scalar
  void const *beta{nullptr};       /// pointer to beta scalar
  ScalarPointerMode pointer_mode{}; /// enumerant indicating whether alpha/beta pointers are host
                                    ///   or device pointers.
  bool use_pdl{false};              /// Whether to use PDL when launching the kernel
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for basic Rank K update operations
//
// OperationKind: (Syrk, Herk, Syr2k, Her2k)
// RankKKind:      Universal
//
struct RankKConfiguration {

  /// SYRK problem size
  gemm::GemmCoord problem_size{};

  /// Leading dimension of A matrix
  int64_t lda{0};

  /// Leading dimension of B matrix
  int64_t ldb{0};

  /// Leading dimension of C matrix
  int64_t ldc{0};

  /// Leading dimension of D matrix
  int64_t ldd{0};

  /// Batch Count
  int batch_count{1};
};

/// Arguments for (Syrk, Herk, Syr2k, Her2k)
struct RankKArguments {

  /// Pointer to A matrix
  void const *A{nullptr};

  /// Pointer to B matrix (used only for Syr2k and Her2k)
  void const *B{nullptr};

  /// Pointer to C matrix
  void const *C{nullptr};

  /// Pointer to D matrix
  void *D{nullptr};

  /// Host or device pointer to alpha scalar
  void const *alpha{nullptr};

  /// Host or device pointer to beta scalar
  void const *beta{nullptr};

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode{};

  int64_t batch_stride_A{0};
  int64_t batch_stride_B{0};
  int64_t batch_stride_C{0};
  int64_t batch_stride_D{0};
  bool use_pdl{false};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for basic TRMM operations
//
// OperationKind: Trmm
// TrmmKind:      Universal
//
struct TrmmConfiguration {

  /// TRMM problem size
  gemm::GemmCoord problem_size{};

  /// Leading dimension of A matrix
  int64_t lda{0};

  /// Leading dimension of B matrix
  int64_t ldb{0};

  /// Leading dimension of D matrix
  int64_t ldd{0};

  /// Batch Count
  int batch_count{1};
};

/// Arguments for TRMM
struct TrmmArguments {

  /// Pointer to A matrix
  void const *A{nullptr};

  /// Pointer to B matrix
  void const *B{nullptr};

  /// Pointer to D matrix
  void *D{nullptr};

  /// Host or device pointer to alpha scalar
  void const *alpha{nullptr};

  /// Host or device pointer to beta scalar
  void const *beta{nullptr};

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode{};

  int64_t batch_stride_A{0};
  int64_t batch_stride_B{0};
  int64_t batch_stride_D{0};
  bool use_pdl{false};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for basic SYMM/HEMM update operations
//
// OperationKind: (Symm, Hemm)
// SymmKind:      Universal
//
struct SymmConfiguration {

  /// SYMM/HEMM problem size
  gemm::GemmCoord problem_size{};

  /// Leading dimension of A matrix
  int64_t lda{0};

  /// Leading dimension of B matrix
  int64_t ldb{0};

  /// Leading dimension of C matrix
  int64_t ldc{0};

  /// Leading dimension of D matrix
  int64_t ldd{0};

  /// Batch Count
  int batch_count{1};
};

/// Arguments for (Symm, Hemm)
struct SymmArguments {

  /// Pointer to A matrix
  void const *A{nullptr};

  /// Pointer to B matrix
  void const *B{nullptr};

  /// Pointer to C matrix
  void const *C{nullptr};

  /// Pointer to D matrix
  void *D{nullptr};

  /// Host or device pointer to alpha scalar
  void const *alpha{nullptr};

  /// Host or device pointer to beta scalar
  void const *beta{nullptr};

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode{};

  int64_t batch_stride_A{0};
  int64_t batch_stride_B{0};
  int64_t batch_stride_C{0};
  int64_t batch_stride_D{0};
  bool use_pdl{false};
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Two dimensional convolution
//
// OperationKind: Conv2d
//
struct Conv2dConfiguration {

  conv::SplitKMode split_k_mode;

  /// Conv2d problem size
  //  contains strictly conv2d size (N,H,W,C,K,R,S,P,Q,padding,stride,dilation,mode)
  //  also includes (split_k_slices, groups)
  conv::Conv2dProblemSize problem_size{};

  // stride of operand A
  std::vector<int64_t> stride_a{};

  // stride of operand B
  std::vector<int64_t> stride_b{};

  // stride of operand C
  std::vector<int64_t> stride_c{};
};


/// Three dimensional convolution
//
// OperationKind: Conv3d
//
struct Conv3dConfiguration {

  conv::SplitKMode split_k_mode{};

  /// Conv2d problem size
  //  contains strictly conv2d size (N,D,H,W,C,K,T,R,S,Z,P,Q,padding,stride,dilation,mode)
  //  also includes (split_k_slices, groups)
  conv::Conv3dProblemSize problem_size{};

  /// Layout object for activations tensor
  layout::TensorNDHWC layout_activations{};

  /// Layout object for filters tensor
  layout::TensorNDHWC layout_filters{};

  /// Layout object for source tensor
  layout::TensorNDHWC layout_source{};

  /// Layout object for output tensor
  layout::TensorNDHWC layout_output{};

  //
  // Methods
  //

  // Mapping functions (A,B,C -> activation,filter,output)
  layout::TensorNDHWC layout_a(library::ConvKind const &conv_kind) const {
    switch (conv_kind) {
      case library::ConvKind::kFprop: return layout_activations;
      case library::ConvKind::kDgrad: return layout_output;
      case library::ConvKind::kWgrad: return layout_output;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

  layout::TensorNDHWC layout_b(library::ConvKind const &conv_kind) const {
    switch (conv_kind) {
      case library::ConvKind::kFprop: return layout_filters;
      case library::ConvKind::kDgrad: return layout_filters;
      case library::ConvKind::kWgrad: return layout_activations;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

  layout::TensorNDHWC layout_c(library::ConvKind const &conv_kind) const {
    switch (conv_kind) {
      case library::ConvKind::kFprop: return layout_output;
      case library::ConvKind::kDgrad: return layout_activations;
      case library::ConvKind::kWgrad: return layout_filters;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }
};

/// Arguments for CONV
struct ConvArguments {

  /////////////////////////////////////////////////////////
  /// ImplicitGemm matrices A, B, C, D
  /////////////////////////////////////////////////////////
  /// pointer to implicit gemm matrix A
  void const *A{nullptr};

  /// pointer to implicit gemm matrix B
  void const *B{nullptr};

  /// pointer to reordered matrix B
  void const *reordered_B{nullptr};

  /// pointer to implicit gemm matrix C
  void const *C{nullptr};

  /// pointer to implicit gemm destination matrix D
  void *D{nullptr};

  /// Host or device pointer to alpha scalar
  void const *alpha{nullptr};

  /// Host or device pointer to beta scalar
  void const *beta{nullptr};

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode{};
  
  /// Whether to use PDL when launching the kernel
  bool use_pdl{false};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for Reduction operations
//
// OperationKind: Reduction
//
struct ReductionConfiguration {

  /// Reduction problem size
  MatrixCoord problem_size{};

  /// Number of partitions to reduce
  int partitions{0};

  /// Number of elements between each partition
  int64_t partition_stride{0};

  /// leading dimension of 'w'orkspace operand
  int64_t ldw{0};

  /// leading dimension of 's'ource operand
  int64_t lds{0};

  /// leading dimension of 'd'estination operand
  int64_t ldd{0};
};

/// Arguments for Reduction
struct ReductionArguments {

  /// Pointer to workspace matrix
  void const *workspace{nullptr};

  /// Pointer to source matrix
  void const *source{nullptr};

  /// Pointer to destination matrix
  void *destination{nullptr};

  /// pointer to reference matrix
  void *reference{nullptr};

  /// Host or device pointer to alpha scalar
  void const *alpha{nullptr};

  /// Host or device pointer to beta scalar
  void const *beta{nullptr};

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode{};

  /// Whether to use PDL when launching the kernel
  bool use_pdl{false};
};

} // namespace library
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif
