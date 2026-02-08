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
/* \file
   \brief Defines profiling functionality for convolution

*/

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <unordered_map>

// CUTLASS Library includes
#include "cutlass/library/library.h"
#include "cutlass/library/util.h"
#include "cutlass/library/handle.h"
#include "cutlass/library/manifest.h"
#include "cutlass/library/singleton.h"

// Profiler includes
#include "options.h"
#include "device_context.h"
#include "operation_profiler.h"
#include "performance_result.h"
#include "problem_space.h"
#include "reduction_operation_profiler.h"
#if CUTLASS_ENABLE_CUDNN
#include "cudnn_helpers.h"
#endif //#if CUTLASS_ENABLE_CUDNN
#include "debug.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Abstract base class for each math function
class Conv3dOperationProfiler : public OperationProfiler {
public:

  /// Problem structure obtained from problem space
  struct Conv3dProblem {

    int64_t n, d, h, w, c, z, p, q, k, t, r, s;
    int64_t pad_d, pad_h, pad_w;
    int64_t stride_d, stride_h, stride_w;
    int64_t dilation_d, dilation_h, dilation_w;

    std::vector<uint8_t> alpha;
    std::vector<uint8_t> beta;

    library::SplitKMode split_k_mode;
    int64_t split_k_slices;

    library::ConvModeID conv_mode;

    library::Provider eq_gemm_provider;

    // convolution with parallel interleaved reduction  
    // convolution epilogue (alpha, beta) = (1.0, 0.0)
    // reduction epilogue (alpha, beta) = (Conv3dProblem::alpha, Conv3dProblem::beta)
    std::vector<uint8_t> alpha_one;
    std::vector<uint8_t> beta_zero;

    //
    // Methods
    //

    /// Total number of bytes loaded
    int64_t bytes(library::ConvDescription const &operation_desc) const;

    /// Total number of flops computed
    int64_t flops(library::ConvDescription const &operation_desc) const;

    /// Infers output size from the input size, padding, stride, and dilation
    void set_default_output_size() {
      z = ((d + pad_d - t * dilation_d) / stride_d) + 1;
      p = ((h + pad_h - r * dilation_h) / stride_h) + 1;
      q = ((w + pad_w - s * dilation_w) / stride_w) + 1;
    }

    // Returns equivalent gemm problem size for convolution
    cutlass::gemm::GemmCoord eq_gemm_size(library::ConvKind const &conv_kind) const {

      switch (conv_kind) {
        case library::ConvKind::kFprop: return cutlass::gemm::GemmCoord(int(n * z * p * q), int(k), int(t * r * s * c));
        case library::ConvKind::kDgrad: return cutlass::gemm::GemmCoord(int(n * d * h * w), int(c), int(t * r * s * k));
        case library::ConvKind::kWgrad: return cutlass::gemm::GemmCoord(int(k), int(t * r * s * c), int(n * z * p * q));
        default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
      }
    }

    // Returns extent for tensor A
    std::vector<int> extent_a(library::ConvKind const &conv_kind) const {

      switch (conv_kind) {
        case library::ConvKind::kFprop: return {int(n), int(d), int(h), int(w), int(c)};
        case library::ConvKind::kDgrad: return {int(n), int(z), int(p), int(q), int(k)};
        case library::ConvKind::kWgrad: return {int(n), int(z), int(p), int(q), int(k)};
        default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
      }
    }

    // Returns extent for tensor B
    std::vector<int> extent_b(library::ConvKind const &conv_kind) const {

      switch (conv_kind) {
        case library::ConvKind::kFprop: return {int(k), int(t), int(r), int(s), int(c)};
        case library::ConvKind::kDgrad: return {int(k), int(t), int(r), int(s), int(c)};
        case library::ConvKind::kWgrad: return {int(n), int(d), int(h), int(w), int(c)};
        default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
      }
    }

    // Returns extent for tensor C
    std::vector<int> extent_c(library::ConvKind const &conv_kind) const {
    
      switch (conv_kind) {
        case library::ConvKind::kFprop: return {int(n), int(z), int(p), int(q), int(k)};
        case library::ConvKind::kDgrad: return {int(n), int(d), int(h), int(w), int(c)};
        case library::ConvKind::kWgrad: return {int(k), int(t), int(r), int(s), int(c)};
        default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
      }
    }

    // Returns layout for equivalent gemm matrix A
    library::LayoutTypeID eq_gemm_layout_a(library::ConvKind const &conv_kind) const {

      switch (conv_kind) {
        case library::ConvKind::kFprop: return library::LayoutTypeID::kRowMajor;    // TN Gemm
        case library::ConvKind::kDgrad: return library::LayoutTypeID::kRowMajor;    // TT Gemm
        case library::ConvKind::kWgrad: return library::LayoutTypeID::kColumnMajor; // NT Gemm
        default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
      }
    }

    // Returns layout for equivalent gemm matrix B
    library::LayoutTypeID eq_gemm_layout_b(library::ConvKind const &conv_kind) const {

      switch (conv_kind) {
        case library::ConvKind::kFprop: return library::LayoutTypeID::kColumnMajor;  // TN Gemm
        case library::ConvKind::kDgrad: return library::LayoutTypeID::kRowMajor;     // TT Gemm
        case library::ConvKind::kWgrad: return library::LayoutTypeID::kRowMajor;     // NT Gemm
        default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
      }
    }

    // Returns layout for equivalent gemm matrix C
    library::LayoutTypeID eq_gemm_layout_c(library::ConvKind const &conv_kind) const {

      switch (conv_kind) {
        // Gemm operator assumes column-major output
        case library::ConvKind::kFprop:
        case library::ConvKind::kDgrad: 
        case library::ConvKind::kWgrad: return library::LayoutTypeID::kColumnMajor;
        default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
      }
    }

    // Returns leading dimension for equivalent gemm matrix A
    int64_t eq_gemm_lda(library::ConvKind const &conv_kind) const {

      switch (conv_kind) {
        case library::ConvKind::kFprop: return eq_gemm_size(conv_kind).k();
        case library::ConvKind::kDgrad: return eq_gemm_size(conv_kind).k();
        case library::ConvKind::kWgrad: return eq_gemm_size(conv_kind).m();
        default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
      }
    }

    // Returns leading dimension for equivalent gemm matrix B
    int64_t eq_gemm_ldb(library::ConvKind const &conv_kind) const {

      switch (conv_kind) {
        case library::ConvKind::kFprop: return eq_gemm_size(conv_kind).k();
        case library::ConvKind::kDgrad: return eq_gemm_size(conv_kind).n();
        case library::ConvKind::kWgrad: return eq_gemm_size(conv_kind).n();
        default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
      }
    }

    // Returns leading dimension for equivalent gemm matrix C
    int64_t eq_gemm_ldc(library::ConvKind const &conv_kind) const {

      switch (conv_kind) {
        case library::ConvKind::kFprop: 
        case library::ConvKind::kDgrad: 
        case library::ConvKind::kWgrad: return eq_gemm_size(conv_kind).m();
        default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
      }
    }
  };

  /// Workspace used 
  struct Conv2dWorkspace {

    /// Conv device allocations
    DeviceAllocation *A;
    DeviceAllocation *B;
    DeviceAllocation *C;
    DeviceAllocation *Computed;
    DeviceAllocation *Reference;
    
    /// Library configuration and arguments for convolution operator
    library::Conv3dConfiguration configuration;
    library::ConvArguments arguments;

    /// Number of copies of the problem workspace which are visited sequentially during
    /// profiling to avoid camping in the last level cache.
    int problem_count;

    /// Buffer used for the cutlass conv2d operations' host workspace
    std::vector<uint8_t> host_workspace;

    /// Buffer used for the cutlass operations' device workspace
    DeviceAllocation device_workspace;
    
    /// Library configuration and arguments for reduction operator
    library::ReductionConfiguration reduction_configuration;
    library::ReductionArguments reduction_arguments;

    /// Buffer used for the cutlass reduction operations' host workspace
    std::vector<uint8_t> reduction_host_workspace;
  
    /// Host data buffers for host reference operation
    /// host buffer for tensor 
    std::vector<uint8_t> host_tensor_a;

    /// host buffer for tensor b
    std::vector<uint8_t> host_tensor_b;

    /// host buffer for tensor c
    std::vector<uint8_t> host_tensor_c;


    //
    // Methods
    //

    Conv2dWorkspace(): 
      A(nullptr), B(nullptr), C(nullptr), Computed(nullptr), Reference(nullptr) { }

      // Returns stride vector for tensor A
      std::vector<int64_t> stride_a(library::ConvKind const &conv_kind) {
        return {        
          configuration.layout_a(conv_kind).stride()[0],
          configuration.layout_a(conv_kind).stride()[1],
          configuration.layout_a(conv_kind).stride()[2],
          configuration.layout_a(conv_kind).stride()[3]
        };
      }

      // Returns stride vector for tensor B
      std::vector<int64_t> stride_b(library::ConvKind const &conv_kind) {

        return {        
          configuration.layout_b(conv_kind).stride()[0],
          configuration.layout_b(conv_kind).stride()[1],
          configuration.layout_b(conv_kind).stride()[2],
          configuration.layout_b(conv_kind).stride()[3]
        };
      }

      // Returns stride vector for tensor C
      std::vector<int64_t> stride_c(library::ConvKind const &conv_kind) {

        return {        
          configuration.layout_c(conv_kind).stride()[0],
          configuration.layout_c(conv_kind).stride()[1],
          configuration.layout_c(conv_kind).stride()[2],
          configuration.layout_c(conv_kind).stride()[3]
        };
      }
  };

protected:

  //
  // Data members
  //

  /// CONV problem obtained from problem space
  Conv3dProblem problem_;

  /// Device memory allocations 
  Conv2dWorkspace conv_workspace_;

  /// CUTLASS parallel reduction operation to follow this* conv2d operation
  library::Operation const *reduction_op_;

public:
  //
  // Methods
  //

  /// Ctor
  Conv3dOperationProfiler(Options const &options);

  /// Destructor
  virtual ~Conv3dOperationProfiler();

  Conv3dProblem const& problem() const { return problem_; }

  /// Prints usage statement for the math function
  virtual void print_usage(std::ostream &out) const;

  /// Prints examples
  virtual void print_examples(std::ostream &out) const;

  /// Extracts the problem dimensions
  virtual Status initialize_configuration(
    Options const &options, 
    PerformanceReport &report, 
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

  /// Initializes workspace
  virtual Status initialize_workspace(
    Options const &options, 
    PerformanceReport &report, 
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

  /// Verifies CUTLASS against references
  virtual bool verify_cutlass(
    Options const &options,  
    PerformanceReport &report,
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

  /// Measures performance results
  virtual bool profile(
    Options const &options, 
    PerformanceReport &report, 
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

protected:

  /// Updates the arguments structure for the CUTLASS operator based on
  /// the problem index.
  void set_cutlass_operator_arguments_(int problem_idx = 0);

  /// Method to profile an initialized CUTLASS operation
  virtual Status profile_cutlass_(
    PerformanceResult &result,
    Options const &options,
    library::Operation const *operation,
    void *arguments,
    void *host_workspace,
    void *device_workspace);
  
  /// Initialize reduction problem dimensions and library::Operation
  bool initialize_reduction_configuration_(
    Options const &options,  
    PerformanceReport &report,
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

  /// Initializes the performance result
  void initialize_result_(
    PerformanceResult &result,
    Options const &options,  
    library::ConvDescription const &operation_desc,
    ProblemSpace const &problem_space);

  /// Verifies CUTLASS against host reference
  bool verify_with_host_reference_(
    Options const &options,  
    PerformanceReport &report,
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

  /// Verifies CUTLASS against device reference
  bool verify_with_device_reference_(
    Options const &options,  
    PerformanceReport &report,
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

#if CUTLASS_ENABLE_CUDNN

  /// Verifies CUTLASS against cudnn reference
  bool verify_with_cudnn_(
    Options const &options,  
    PerformanceReport &report,
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

#endif //#if CUTLASS_ENABLE_CUDNN

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

