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
   \brief Command line options for performance test program
*/

#pragma once

#include <string>
#include <vector>
#include <map>

#include <cuda_runtime.h>

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/library/library.h"

#include "enumerated_types.h"

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Global options
class Options {
public:

  /// Cublas and cuDNN options
  struct Library {

    //
    // Data members
    //

    /// Algorithm mode
    AlgorithmMode algorithm_mode;

    /// Algorithm enumerants
    std::vector<int> algorithms;

    //
    // Methods
    //

    explicit Library(CommandLine const &cmdline);

    void print_usage(std::ostream &out) const;
    void print_options(std::ostream &out, int indent = 0) const;
  };

  /// Options related to the selected device
  struct Device {

    /// Device ID
    std::vector<int> devices;

    /// Number of total devices
    /// This is not set by the user, it is set by automatically
    int num_devices;

    /// CUDA Device properties
    std::vector<cudaDeviceProp> properties;

    /// Total memory allocation on each device
    size_t maximum_capacity;

  private:
    /// SM Count
    /// Limits the number of SMs to use on each device 
    int sm_count;

    //
    // Methods
    //
  public:
    explicit Device(CommandLine const &cmdline);

    void print_usage(std::ostream &out) const;
    void print_options(std::ostream &out, int indent = 0) const;
    void print_device_info(std::ostream &out) const;

    /// Returns the device ID from a device index
    int device_id(size_t device_index) const;

    /// Returns the sm_count if set, otherwise returns the number of SMs on the device
    int get_sm_count(int device_index) const;

    /// Returns the compute capability of the listed devices (e.g. 70, 75, 80, etc.)
    int compute_capability(int device_index) const;
  };

  /// Options related to initializing input tensors
  struct Initialization {

    /// If true, data is initialized randomly. If false, no initialization is performed after
    /// allocating tensors.
    bool enabled;

    /// If true, data distribution is set by the user and is not allowed to change
    /// If false, data distribution is allowed to change based on element_type (library::NumericTypeID)
    bool fix_data_distribution;

    /// Data distribution for input tensors
    Distribution data_distribution;

    /// Source of random tensor elements
    library::Provider provider;

    /// Random number generator seed.
    int seed;

    //
    // Methods
    //

    explicit Initialization(CommandLine const &cmdline);

    void print_usage(std::ostream &out) const;
    void print_options(std::ostream &out, int indent = 0) const;

    /// Helper to parse a Distribution object from the command line parser
    static void get_distribution(
      cutlass::CommandLine const &args,
      std::string const &arg,
      cutlass::Distribution &dist);
  };

  /// Options related to verification of the result
  struct Verification {

    //
    // Data members
    //

    /// If true, kernels are verified before they are profiled
    bool enabled;

    /// If true, causes profiler to return an error code if no reference check is run.
    /// Only valid when verification is enabled.
    bool required;

    /// Relative error threshold - zero to require bit-level consistency
    double epsilon;

    /// Values smaller than this are assumed to be zero
    double nonzero_floor;

    /// List of providers used to verify each result
    ProviderVector providers;

    /// Indicates when to save the workspace
    SaveWorkspace save_workspace;

    //
    // Methods
    //

    explicit Verification(CommandLine const &cmdline);

    void print_usage(std::ostream &out) const;
    void print_options(std::ostream &out, int indent = 0) const;

    /// Returns true if a provider is enabled
    bool provider_enabled(library::Provider provider) const;

    /// Returns the index of a provider if its enabled
    size_t index(library::Provider provider) const;
  };

  /// Options related to profiling
  struct Profiling {

    /// Number of workspaces to rotate through to avoid cache-resident working sets
    int workspace_count{0};

    /// Number of iterations to warmup each kernel prior to profiling
    int warmup_iterations{10};

    /// Number of iterations to profile each kernel - if 0, kernels are launched up to the profiling duration
    /// This will always override profiling-duration and min-iterations.
    int iterations{100};

    /// Time to spend profiling each kernel (ms)
    int duration{10};

    /// Minimum number of iterations to profile
    int min_iterations{10};

    /// If true, profiling with cuda graph enabled.
    bool use_cuda_graphs{false};

    /// If enabled, the CUTLASS profiler searches for the best-performing kernel 
    /// within the subset of kernels matching a kernel filter regex. The best 
    /// performance is determined by screening over a set of predefined M/N/K 
    /// sizes and performance-related parameters, including cluster shapes, 
    /// swizzle sizes, and rasterization orders.
    /// For now, it only supports legacy GEMM and blockscaled GEMM.
    bool enable_kernel_performance_search{false};

    /// If enabled, the CUTLASS profiler searches for the best-performing kernel 
    /// for a given M/N/K problem size by evaluating various performance-related 
    /// parameters such as cluster shapes, swizzle sizes, and rasterization orders.
    /// For now, it only supports legacy GEMM and blockscaled GEMM.
    bool enable_best_kernel_for_fixed_shape{false};

    /// Number of ms to sleep between profiling periods (ms)
    int sleep_duration{50};

    /// If true, profiling is actually conducted.
    bool enabled{true};

    /// If true, profiling returns an error code if no kernels are found to match the filters.
    bool error_on_no_match{false};

    /// If true, profiling returns an error code if no kernel are profiled
    // Sometimes the kernel matches but failed to profile (e.g. can_implement() error)
    bool error_if_nothing_is_profiled{false};

    /// List of providers of each functionality to be profiled
    ProviderVector providers;

    //
    // Methods
    //

    explicit Profiling(CommandLine const &cmdline);

    void print_usage(std::ostream &out) const;
    void print_options(std::ostream &out, int indent = 0) const;

    /// Returns true if a provider is enabled
    bool provider_enabled(library::Provider provider) const;

    /// Returns the index of a provider if its enabled
    size_t index(library::Provider provider) const;
  };

  /// Options related to reporting
  struct Report {

    /// If true, result is appended to possibly existing file
    bool append;

    /// Path to a file containing results
    std::string output_path;

    /// Path to a file containing junit xml results
    std::string junit_output_path;

    /// Sequence of tags to attach to each result
    std::vector<std::pair<std::string, std::string>> pivot_tags;

    /// If true, reports status of all kernels including those that were
    /// not run for the given arguments
    bool report_not_run;

    /// Prints human-readable text to stdout. If false, nothing is written to stdout
    bool verbose;

    /// Sort results by flops-per-byte
    bool sort_flops_per_byte;

    /// Sort results by flops-per-second
    bool sort_flops_per_sec;

    /// Prints the name of the kernel being profiled before running the kernel.
    /// This is useful for determining which kernel is causing a run of the profiler to hang
    bool print_kernel_before_running;

    //
    // Methods
    //

    explicit Report(CommandLine const &cmdline);

    void print_usage(std::ostream &out) const;
    void print_options(std::ostream &out, int indent = 0) const;
  };

  /// Options related to printing usage and version information
  struct About {

    /// If true, usage is printed and the program ends.
    bool help;

    /// Prints version string
    bool version;

    /// Print information about devices
    bool device_info;

    //
    // Methods
    //

    explicit About(CommandLine const &cmdline);

    void print_usage(std::ostream &out) const;
    void print_options(std::ostream &out, int indent = 0) const;

    static void print_version(std::ostream &out);
  };

public:

  //
  // Data members
  //

  /// Top-level execution mode
  ExecutionMode execution_mode;

  /// Name of math function to profile
  library::OperationKind operation_kind;

  /// Vector of operation name substrings
  std::vector<std::string> operation_names;

  /// Map of problems to run for each operation
  /// [operation_name] -> vector of problems, each problem specified as a vector of [argument name] -> [argument value]
  std::unordered_map<std::string, std::vector<CommandLine>> operation_problems;

  /// Vector of operation name substrings
  std::vector<std::string> excluded_operation_names;


  //
  // Detailed configuration options
  //

  /// Configuration
  CommandLine cmdline;
  Device device;
  Initialization initialization;
  Library library;
  Verification verification;
  Profiling profiling;
  Report report;
  About about;

public:

  explicit Options(CommandLine const &cmdline);

  void print_usage(std::ostream &out) const;
  void print_options(std::ostream &out) const;

  static std::string indent_str(int indent);
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass
