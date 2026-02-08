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
   \brief Defines a math function
*/

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

// CUTLASS includes
#include "cutlass/trace.h"

// CUTLASS Library includes
#include "cutlass/library/library.h"
#include "cutlass/library/util.h"
#include "cutlass/library/manifest.h"

// Profiler includes
#include "options.h"
#include "device_context.h"
#include "performance_result.h"
#include "performance_report.h"
#include "problem_space.h"
#include "debug.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Abstract base class for each math function
class OperationProfiler {
public:


protected:
  //
  // Data members
  //

  /// Top-level operation kind
  library::OperationKind kind_;

  /// Human readable description
  std::string description_;

  /// Arguments parsed from command line
  ArgumentDescriptionVector arguments_;

  /// List of providers used to verify and compare each result
  ProviderVector verification_providers_;

  /// Model performance result initialized by the operation profiler with workload statistics
  /// and reasonable default state.
  PerformanceResult model_result_;

  /// Performance result vector constructed by profiling the operation
  PerformanceResultVector results_;

public:

  //
  // Methods
  //

  /// Ctor
  OperationProfiler();

  OperationProfiler(
    Options const &options,
    library::OperationKind kind, 
    ArgumentDescriptionVector const &arguments = ArgumentDescriptionVector(),
    ProviderVector const & verification_providers = ProviderVector());

  /// Destructor
  virtual ~OperationProfiler();

  /// Obtains the operation kind
  library::OperationKind kind() const { return kind_; }

  /// Gets the schema description
  std::string const &description() const;

  /// Returns a reference to the arguments
  ArgumentDescriptionVector const &arguments() const { return arguments_; }

public:

  //
  // Basic overrides
  //


  /// Prints usage statement for the math function
  virtual void print_usage(std::ostream &out) const;

  /// Prints examples
  virtual void print_examples(std::ostream &out) const =0;

  /// Entry point to profile all operations in the manifest
  virtual int profile_all(
    Options const &options, 
    library::Manifest const &manifest, 
    DeviceContext &device_context);

public:

  //
  // Operation-specific phases of verification and profiling
  //

  /// Extracts the problem dimensions
  virtual Status initialize_configuration(
    Options const &options, 
    PerformanceReport &report, 
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem) = 0;

  /// Initializes workspace
  virtual Status initialize_workspace(
    Options const &options, 
    PerformanceReport &report, 
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem) = 0;

  /// Verifies CUTLASS against references
  virtual bool verify_cutlass(
    Options const &options,  
    PerformanceReport &report,
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem) = 0;

  /// Measures performance results
  virtual bool profile(
    Options const &options,  
    PerformanceReport &report,
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem) = 0;

public:

  //
  // Static helpers
  //

  /// Sleep for a given duration in ms
  static void sleep(int sleep_duration);

  /// Returns true if the current operation description satisfies the problem space
  static bool satisfies(
    library::OperationDescription const &op_desc,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);
  
  /// Compares tensors for equality
  static Disposition compare_tensors(
    Options const &options,
    DeviceAllocation &experimental,
    DeviceAllocation &reference,
    int64_t count = 0);

  static void save_workspace(
    DeviceContext &device_context,
    Options const &options,
    library::OperationDescription const &desc,
    library::Provider provider,
    library::Provider verification_provider = library::Provider::kInvalid);
  
  /// Helper to set a performance result member
  static void set_argument(  
    PerformanceResult &result,
    char const *name,
    ProblemSpace const &problem_space,
    std::string const &value);

  /// Helper to set a performance result member
  static void set_argument(  
    PerformanceResult &result,
    char const *name,
    ProblemSpace const &problem_space,
    int64_t value);

protected:

  /// Sets operation description 
  static void initialize_result_(
    PerformanceResult &result,
    library::OperationDescription const &operation_desc,
    ProblemSpace const &problem_space);

  /// Method to profile an initialized CUTLASS operation
  virtual Status profile_cutlass_(
    PerformanceResult &result,
    Options const &options,
    library::Operation const *operation,
    void *arguments,
    void *host_workspace,
    void *device_workspace);

  /// Profiles the GPU kernel launched in `func` running simultaneously on all
  /// requested devices.
  Status profile_kernel_w_cuda_graphs_(
    PerformanceResult& result,
    Options const& options,
    std::function<Status(int, cudaStream_t, int)> const& func,
    std::vector<cudaStream_t> const& streams);

  Status profile_kernel_(
    PerformanceResult& result,
    Options const& options,
    std::function<Status(int, cudaStream_t, int)> const& func,
    std::vector<cudaStream_t> const& streams);

  /// Profiles the GPU kernel launched in `func` on the `stream`
  Status profile_kernel_(
    PerformanceResult& result,
    Options const& options,
    std::function<Status(cudaStream_t, int)> const& func,
    cudaStream_t stream = nullptr);

  /// Profiles the GPU kernel launched in `func` on the `stream`
  Status profile_kernel_no_cuda_graphs_(
    PerformanceResult& result,
    Options const& options,
    std::function<Status(cudaStream_t, int)> const& func,
    cudaStream_t stream = nullptr);

private:
  /// finds string matches filter_string in operation_name
  bool find_string_matches_(
    std::string const &filter_string, 
    std::string const &operation_name);
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Vector of owning operation profilers
using OperationProfilerVector = std::vector<std::unique_ptr<OperationProfiler>>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
