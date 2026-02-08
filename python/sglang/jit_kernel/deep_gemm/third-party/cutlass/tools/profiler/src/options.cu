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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <fstream>
#include <set>

#include "cutlass/cutlass.h"
#include "cutlass/version.h"

#include "cutlass/library/util.h"

#include "cutlass/profiler/options.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Newline and indent for help strings
static char const *end_of_line = "\n                                             ";

/////////////////////////////////////////////////////////////////////////////////////////////////

Options::Device::Device(cutlass::CommandLine const &cmdline) {

  // Gets the number of devices for future validation
  cudaError_t result;
  result = cudaGetDeviceCount(&num_devices);
  if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetNumDevices() failed");
  }

  // Gets the devices specified by the user
  // This preserves the user specified order and checks for duplicates
  {
    std::vector<int> temp_device_list;
    cmdline.get_cmd_line_arguments("devices", temp_device_list);
    if (temp_device_list.empty()) {
      temp_device_list.push_back(0);
    }
    {
      std::set<int> temp_device_set;
      for (int device : temp_device_list) {
        auto res = temp_device_set.insert(device);
        if (!res.second) {
          throw std::runtime_error("Duplicate device specified: " +
                                   std::to_string(device));
        } else if (device > num_devices) {
          throw std::runtime_error("Bad device ID: " +
                                   std::to_string(device));
        } else {
          devices.push_back(device);
        }
      }
    }
  }

  properties.resize(devices.size());
  // Retrieves properties for all specified devices
  for (size_t device_index = 0; device_index < devices.size(); device_index++) {
    int device = devices[device_index];

    result = cudaGetDeviceProperties(&properties[device_index], device);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed for given device");
    }

    // Check that all devices are the same
    if (device_index > 0) {
      if ((properties[device_index].major != properties[0].major) ||
          (properties[device_index].minor != properties[0].minor)) {
        throw std::runtime_error("All selected devices must have the same "
                                 "compute capability");
      }
      if (properties[device_index].l2CacheSize != properties[0].l2CacheSize) {
        throw std::runtime_error("All selected devices must have the same "
                                 "L2 cache size");
      }
      if (properties[device_index].multiProcessorCount != properties[0].multiProcessorCount) {
        throw std::runtime_error("All selected devices must have the same "
                                 "SM count");
      }
    }

    result = cudaSetDevice(device);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaSetDevice() failed for given device.");
    }

    // Permit overriding the compute capability
    if (cmdline.check_cmd_line_flag("compute-capability")) {
      int cc = compute_capability(device_index);
      cmdline.get_cmd_line_argument("compute-capability", cc, cc);
      properties[device_index].major = cc / 10;
      properties[device_index].minor = cc % 10;
    }

    // Permit overriding the L2 cache capacity
    if (cmdline.check_cmd_line_flag("llc-capacity")) {
      int llc_capacity = 0;
      cmdline.get_cmd_line_argument("llc-capacity", llc_capacity, 0);

      if (llc_capacity >= 0) {
        properties[device_index].l2CacheSize = (llc_capacity << 10);
      }
    }

    // Permit overriding the sm_count
    cmdline.get_cmd_line_argument("sm-count", sm_count, 0);
  }
}

int Options::Device::get_sm_count(int device_index) const {
  if (sm_count <= 0) {
    return properties[device_index].multiProcessorCount;
  }
  return sm_count;
}

void Options::Device::print_usage(std::ostream &out) const {

  out << "Device:\n"
    << "  --devices=<int>,<int>,...                      "
    << "    CUDA Device IDs\n\n";

  int device_count = 0;
  cudaError_t result = cudaGetDeviceCount(&device_count);

  if (result != cudaSuccess) {
    out << "      <could not query for CUDA devices>\n";
  }
  else {

    for (int idx = 0; idx < device_count; ++idx) {
      cudaDeviceProp prop;
      result = cudaGetDeviceProperties(&prop, idx);
      if (result != cudaSuccess) {
        out << "      <could not obtain device properties for device " << idx << ">" << std::endl;
        break;
      }
      else {
        int32_t clock_KHz;
        cudaDeviceGetAttribute(&clock_KHz, cudaDevAttrClockRate, 0);
        out << "    [" << idx << "] - "
          << prop.name << " - SM " << prop.major << "." << prop.minor << ", "
          << prop.multiProcessorCount << " SMs @ " << (clock_KHz / 1000.0) << " MHz, "
          << "L2 cache: " << (prop.l2CacheSize >> 20) << " MB, Global Memory: " << (prop.totalGlobalMem >> 30) << " GB"
          << std::endl;
      }
    }
    out << "\n";
  }

  out
    << "  --compute-capability=<int>                   "
    << "    Override the compute capability.\n\n"

    << "  --llc-capacity=<capacity in KiB>             "
    << "    Capacity of last-level cache in kilobytes. If this is non-zero," << end_of_line
    << "      profiling phases cycle through different input tensors to induce" << end_of_line
    << "      capacity misses in the L2.\n\n"

     << "  --sm-count=<int>                             "
     << "    Override the number of SMs. This is used to limit the number of " << end_of_line
     << "      during profiling. If this is set, profiling attempts to limit the sm_count " << end_of_line
     << "      to user-set value. This is not possible on all architectures and all kernel types. \n\n";

}

void Options::Device::print_device_info(std::ostream &out) const {
  cudaDeviceProp props;
  cudaError_t result;

  out << "Device Name,SM,CUDA Device ID,Phy Device ID" << std::endl;

  for (int device = 0; device < num_devices; device++) {
    result = cudaSetDevice(device);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaSetDevice() failed for device");
    }

    result = cudaGetDeviceProperties(&props, device);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties failed for device");
    }

    out << props.name << "," << props.major << props.minor << ","
      << device << "," << props.multiGpuBoardGroupID << std::endl;

  }
}

void Options::Device::print_options(std::ostream &out, int indent) const {

  out
    << indent_str(indent) << "devices: ";
  for (int device : devices) {
    out << device << ',';
  }
  int32_t clock_KHz;
  cudaDeviceGetAttribute(&clock_KHz, cudaDevAttrClockRate, 0);
  out
    << "\n"
    << indent_str(indent) << "clock: " << int(double(clock_KHz) / 1000.0) << "\n"
    << indent_str(indent) << "compute-capability: " << compute_capability(0) << "\n";
}

/// Returns the device ID from a device index
int Options::Device::device_id(size_t device_index) const {
  if (device_index > devices.size()) {
    throw std::runtime_error("Out of bounds device index: " +
                             std::to_string(device_index));
  }
  return devices.at(device_index);
}

/// Returns the compute capability of the listed device (e.g. 61, 60, 70, 75)
int Options::Device::compute_capability(int device_index) const {
  return properties[device_index].major * 10 + properties[device_index].minor;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Options::Initialization::Initialization(cutlass::CommandLine const &cmdline) {

  cmdline.get_cmd_line_argument("initialization-enabled", enabled, true);

  if (cmdline.check_cmd_line_flag("initialization-provider")) {
    std::string str;
    cmdline.get_cmd_line_argument("initialization-provider", str);
    provider = library::from_string<library::Provider>(str);
    if (provider == library::Provider::kInvalid) {
      enabled = false;
    }
    else if (provider != library::Provider::kReferenceHost && provider != library::Provider::kReferenceDevice) {
      throw std::runtime_error("Unsupported initialization provider specified.");
    }
  }
  else {
    provider = library::Provider::kReferenceDevice;
  }

  cmdline.get_cmd_line_argument("seed", seed, 2019);

  if (cmdline.check_cmd_line_flag("dist")) {
    // user has set the data distribution (fix data distribution once set)
    fix_data_distribution = true;
    // set user provided data distribution
    get_distribution(cmdline, "dist", data_distribution);
  }
  else {
    // profiler chosen data distribution (allowed to change based on numeric types)
    fix_data_distribution = false;
    // set uniform data distribution with range [-4, 4]
    data_distribution.set_uniform(-4, 4, 0);
  }


}

/// Gets the initial distribution
void Options::Initialization::get_distribution(
  cutlass::CommandLine const &args,
  std::string const &arg,
  cutlass::Distribution &dist) {

  struct {
    const char *label;
    cutlass::Distribution::Kind kind;
  } distribution_kinds[] = {
    {"uniform", cutlass::Distribution::Uniform},
    {"gaussian", cutlass::Distribution::Gaussian},
    {"identity", cutlass::Distribution::Identity},
    {"sequential", cutlass::Distribution::Sequential},
    {0, cutlass::Distribution::Invalid}
  };

  struct {
    char const *label;
    double *member;
  } members[] = {
    {"min", &dist.uniform.min},
    {"max", &dist.uniform.max},
    {"mean", &dist.gaussian.mean},
    {"stddev", &dist.gaussian.stddev},
    {"pnzA", &dist.gaussian.pnzA},
    {"pnzB", &dist.gaussian.pnzB},
    {"pnzC", &dist.gaussian.pnzC},
    {"start", &dist.sequential.start},
    {"delta", &dist.sequential.delta},
    {0, 0}
  };

  using KeyValueVector = std::vector<std::pair<std::string, std::string> >;

  KeyValueVector values;
  args.get_cmd_line_argument_pairs(arg.c_str(), values);

  // The parser expects the first token to be a string identifying the distribution type.
  auto it = values.begin();
  if (it != values.end()) {
    for (int i = 0; distribution_kinds[i].label; ++i) {
      if (it->first == distribution_kinds[i].label) {
        dist.kind = distribution_kinds[i].kind;
        break;
      }
    }
    ++it;
  }

  // Default initialization
  switch (dist.kind) {
    case cutlass::Distribution::Uniform:
      dist.set_uniform(-4/*min*/, 4/*max*/);
      break;
    case cutlass::Distribution::Gaussian:
      dist.set_gaussian(0/*mean*/, 4/*stddev*/);
      break;
    case cutlass::Distribution::Identity:
      dist.set_identity();
      break;
    case cutlass::Distribution::Sequential:
      dist.set_sequential(0/*start*/, 4/*delta*/);
      break;
    default:
      dist.set_uniform(-4/*min*/, 4/*max*/);
      return;
  }

  // Subsequent key-value pairs update the named field of the distribution struct.
  for (; it != values.end(); ++it) {
    // Integer scaling factor - if < 0, no integer rounding is performed.
    if ((it->first.compare("scale") == 0) && !it->second.empty()) {
      std::stringstream ss;
      ss << it->second;
      ss >> dist.int_scale;
      continue;  // next token
    }

    // Casts as integer without scaling
    if (it->first.compare("integer") == 0) {
      dist.int_scale = 0;
      continue;  // next token
    }

    // initialize other members
    for (int m = 0; members[m].label; ++m) {
      if (it->first == members[m].label && !it->second.empty()) {
        std::stringstream ss;
        ss << it->second;
        ss >> *(members[m].member);
      }
    }
  }
}

void Options::Initialization::print_usage(std::ostream &out) const {

  out << "Initialization:\n"

    << "  --initialization=<bool>                      "
    << "    Enables initialization (default: true). If false, device memory is" << end_of_line
    << "      not initialized after allocation.\n\n"

    << "  --initialization-provider=<provider>         "
    << "    Selects initialization provider {host, device*}. (default: '*')\n\n"

    << "  --dist=<distribution>                        "
    << "    Data distribution of input tensors {uniform*, gaussian, identity, sequential}"  << end_of_line
    << "       --dist=uniform,min:<double>,max:<double>,scale:<integer>"  << end_of_line
    << "       --dist=gaussian,mean:<double>,stddev:<double>,scale:<integer>,pnzA:<double>,pnzB:<double>,pnzC:<double>"  << end_of_line
    << "       --dist=sequential,start:<double>,delta:<double>,scale:<integer>"  << end_of_line
    << "       --dist=identity\n\n"

    << "  --seed=<int>                                 "
    << "    Random number generator seed. Used to enforce deterministic" << end_of_line
    << "      initialization.\n\n";

}

void Options::Initialization::print_options(std::ostream &out, int indent) const {

}

/////////////////////////////////////////////////////////////////////////////////////////////////

Options::Library::Library(cutlass::CommandLine const &cmdline) {

  algorithm_mode = AlgorithmMode::kDefault;

  if (cmdline.check_cmd_line_flag("library-algo-mode")) {
    std::string mode = "default";
    cmdline.get_cmd_line_argument("library-algo-mode", mode);
    algorithm_mode = from_string<AlgorithmMode>(mode);
  }

  if (cmdline.check_cmd_line_flag("library-algos")) {

    // If algorithms are specified, override as kBest.
    algorithm_mode = AlgorithmMode::kBest;

    std::vector<std::string> tokens;
    cmdline.get_cmd_line_arguments("library-algos", tokens);

    algorithms.reserve(tokens.size());

    for (auto const & token : tokens) {
      if (token.find(":")) {
        // TODO: tokenized range
      }
      else {
        int algo;
        std::stringstream ss;

        ss << token;
        ss >> algo;

        algorithms.push_back(algo);
      }
    }
  }
}

void Options::Library::print_usage(std::ostream &out) const {

  out << "Library:\n"

    << "  --library-algo-mode=<mode>                   "
    << "    Indicates algorithm mode used to call libraries such as cuBLAS and cuDNN.\n"
    << "                                               "
    << "    mode={default*,matching,best}\n\n"

    << "  --library-algos=<range-list>                 "
    << "    If --algorithm-mode=best, permits specifying a selection of algorithms.\n\n";

}

void Options::Library::print_options(std::ostream &out, int indent) const {

  out
    << indent_str(indent) << "library-algo-mode: " << to_string(algorithm_mode) << "\n"
    << indent_str(indent) << "library-algos: ";

  int j = 0;
  for (int x : algorithms) {
    out << (j++ ? "," : "") << x;
  }

  out << "\n\n";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Options::Profiling::Profiling(cutlass::CommandLine const &cmdline) {

  cmdline.get_cmd_line_argument("workspace-count", workspace_count, 0);
  cmdline.get_cmd_line_argument("warmup-iterations", warmup_iterations, 10);
  cmdline.get_cmd_line_argument("profiling-iterations", iterations, 100);
  cmdline.get_cmd_line_argument("sleep-duration", sleep_duration, 50);
  cmdline.get_cmd_line_argument("profiling-enabled", enabled, true);
  cmdline.get_cmd_line_argument("profiling-duration", duration, 10);
  cmdline.get_cmd_line_argument("min-iterations", min_iterations, 10);
  cmdline.get_cmd_line_argument("use-cuda-graphs", use_cuda_graphs, false);
  cmdline.get_cmd_line_argument("enable-kernel-performance-search", enable_kernel_performance_search, false);
  cmdline.get_cmd_line_argument("enable-best-kernel-for-fixed-shape", enable_best_kernel_for_fixed_shape, false);

  if (cmdline.check_cmd_line_flag("providers")) {

    std::vector<std::string> tokens;
    cmdline.get_cmd_line_arguments("providers", tokens);

    providers.clear();

    for (auto const &token : tokens) {
      providers.push_back(library::from_string<library::Provider>(token));
    }
  }
  else {
    providers.push_back(library::Provider::kCUTLASS);
    providers.push_back(library::Provider::kCUBLAS);
    providers.push_back(library::Provider::kCUDNN);
  }
}

void Options::Profiling::print_usage(std::ostream &out) const {

  out << "Profiling:\n"

    << "  --workspace-count=<workspace count>          "
    << "    Number of discrete workspaces maintained to avoid cache-resident " << end_of_line
    << "    If zero (default), the amount is chosen for each workload based on " << end_of_line
    << "    capacity of the last-level cache.\n\n"

    << "  --profiling-iterations=<iterations>          "
    << "    Number of iterations to profile each kernel. If zero, kernels" << end_of_line
    << "      are launched up to the profiling duration. If non-zero, this overrides" << end_of_line
    << "      --profiling-duration and --min-iterations.\n\n"

    << "  --profiling-duration=<duration>             "
    << "    Time to spend profiling each kernel (ms)." << end_of_line
    << "    Overriden by `profiling-iterations` when `profiling-iterations` > 0." << end_of_line
    << "    Note that `min-iterations` must also be satisfied.\n\n"

    << "  --min-iterations=<iterations>             "
    << "    Minimum number of iterations to spend profiling each kernel, even if" << end_of_line
    << "    `profiling-duration` has been met.\n\n"

    << "  --warmup-iterations=<iterations>             "
    << "    Number of iterations to execute each kernel prior to profiling.\n\n"

    << "  --sleep-duration=<duration>                  "
    << "    Number of ms to sleep between profiling periods (ms).\n\n"

    << "  --profiling-enabled=<bool>                   "
    << "    If true, profiling is actually conducted.\n\n"

    << "  --enable-best-kernel-for-fixed-shape=<bool>   "
    << "    If true, iterate through common cluster sizes, raster orders, and swizzle sizes for each kernel.\n\n"

  ;
}

void Options::Profiling::print_options(std::ostream &out, int indent) const {

  out
    << indent_str(indent) << "profiling_iterations: " << iterations << "\n"
    << indent_str(indent) << "sleep_duration: " << sleep_duration << "\n"
    << indent_str(indent) << "profiling_enabled: " << enabled << "\n"
    << indent_str(indent) << "providers: [";

  int j = 0;
  for (auto const & provider : providers) {
    out << (j++ ? ", " : "") << library::to_string(provider);
  }
  out << "]\n";
}

/// Returns true if a provider is enabled
bool Options::Profiling::provider_enabled(library::Provider provider) const {
  return std::find(providers.begin(), providers.end(), provider) != providers.end();
}

/// Returns the index of a provider if its enabled
size_t Options::Profiling::index(library::Provider provider) const {
  size_t idx = 0;
  for (auto const & x : providers) {
    if (x == provider) {
      return idx;
    }
    ++idx;
  }
  return idx;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Options::Verification::Verification(cutlass::CommandLine const &cmdline) {

  cmdline.get_cmd_line_argument("verification-enabled", enabled, true);
  if (enabled) {
    cmdline.get_cmd_line_argument("verification-required", required, false);
  }
  else {
    required = false;
  }

  cmdline.get_cmd_line_argument("epsilon", epsilon, 0.05);

  cmdline.get_cmd_line_argument("nonzero-floor", nonzero_floor, 1.0 / 256.0);

  if (cmdline.check_cmd_line_flag("save-workspace")) {
    std::string value;
    cmdline.get_cmd_line_argument("save-workspace", value);
    save_workspace = from_string<SaveWorkspace>(value);
  }
  else {
    save_workspace = SaveWorkspace::kNever;
  }

  if (cmdline.check_cmd_line_flag("verification-providers")) {

    std::vector<std::string> tokens;
    cmdline.get_cmd_line_arguments("verification-providers", tokens);

    providers.clear();

    for (auto const &token : tokens) {
      library::Provider provider = library::from_string<library::Provider>(token);
      if (provider != library::Provider::kInvalid) {
        providers.push_back(provider);
      }
    }
  }
  else {
    providers.push_back(library::Provider::kCUBLAS);
    providers.push_back(library::Provider::kReferenceDevice);
    providers.push_back(library::Provider::kCUDNN);
  }
}

void Options::Verification::print_usage(std::ostream &out) const {

  out << "Verification:\n"

    << "  --verification-enabled=<bool>                "
    << "    Whether to perform verification checks.\n\n"

    << "  --epsilon=<error>                            "
    << "    Error threshold. Setting to zero (default) requires" << end_of_line
    << "      bit-level equivalence.\n\n"

    << "  --nonzero-floor=<floor>                      "
    << "    Results whose absolute value is less than this quantity" << end_of_line
    << "      are treated as zero for comparisons.\n\n"

    << "  --save-workspace=<string>                    "
    << "    Specifies when to save the GEMM inputs and results to the filesystem." << end_of_line
    << "       --save-workspace=never      never save workspace (default)" << end_of_line
    << "       --save-workspace=incorrect  save workspace for incorrect results" << end_of_line
    << "       --save-workspace=always     always save workspace\n\n"

    << "  --verification-providers=<providers>         "
    << "    List of providers used to verify result. (default: '*')" << end_of_line
    << "      Gemm verification-providers {cublas*}" << end_of_line
    << "      Conv2d verification-providers {cudnn*, device*, host}"
    << "\n\n";
}

void Options::Verification::print_options(std::ostream &out, int indent) const {

  out
    << indent_str(indent) << "verification_enabled: " << enabled << "\n"
    << indent_str(indent) << "epsilon: " << epsilon << "\n"
    << indent_str(indent) << "save_workspace: " << to_string(save_workspace) << "\n"
    << indent_str(indent) << "verification_providers: [";

  int j = 0;
  for (auto const & provider : providers) {
    out << (j++ ? ", " : "") << library::to_string(provider);
  }
  out << "]\n";
}

/// Returns true if a provider is enabled
bool Options::Verification::provider_enabled(library::Provider provider) const {
  return std::find(providers.begin(), providers.end(), provider) != providers.end();
}

/// Returns the index of a provider if its enabled
size_t Options::Verification::index(library::Provider provider) const {
  size_t idx = 0;
  for (auto const & x : providers) {
    if (x == provider) {
      return idx;
    }
    ++idx;
  }
  return idx;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Options::Report::Report(cutlass::CommandLine const &cmdline) {

  cmdline.get_cmd_line_argument("append", append, false);
  cmdline.get_cmd_line_argument("output", output_path);
  cmdline.get_cmd_line_argument("junit-output", junit_output_path);

  if (cmdline.check_cmd_line_flag("tags")) {
    cmdline.get_cmd_line_argument_pairs("tags", pivot_tags);
  }

  cmdline.get_cmd_line_argument("report-not-run", report_not_run, false);

  cmdline.get_cmd_line_argument("verbose", verbose, true);

  cmdline.get_cmd_line_argument("sort-results-flops-per-byte", sort_flops_per_byte, false);

  cmdline.get_cmd_line_argument("sort-results-flops-per-sec", sort_flops_per_sec, false);

  cmdline.get_cmd_line_argument("print-kernel-before-running", print_kernel_before_running, false);
}

void Options::Report::print_usage(std::ostream &out) const {

  out << "Report:\n"

    << "  --append=<bool>                              "
    << "    If true, result is appended to possibly existing file. Otherwise, " << end_of_line
    << "      any existing file is overwritten.\n\n"

    << "  --output=<path>                              "
    << "    Path to output file for machine readable results. Operation kind and '.csv' is appended.\n\n"

    << "  --junit-output=<path>                        "
    << "    Path to junit output file for result reporting. Operation kind and '.junit.xml' is appended.\n\n"

    << "  --print-kernel-before-running=<bool>                "
    << "    Prints the name of the kernel being profiled before running the kernel." << end_of_line
    << "      This is useful for determining which kernel is causing a run of the profiler to hang\n\n"

    << "  --report-not-run=<bool>                      "
    << "    If true, reports the status of all kernels including those that" << end_of_line
    << "      do not satisfy the given arguments.\n\n"

    << "  --tags=<column:tag,...>                      "
    << "    Inserts leading columns in output table and uniform values for each" << end_of_line
    << "      column. Useful for generating pivot tables.\n\n"

    << "  --verbose=<bool>                             "
    << "    Prints human-readable text to stdout. If false, nothing is written to stdout.\n\n"

    << "  --sort-results=<bool>                        "
    << "    Sorts results (by flops-per-byte).\n\n";
}

void Options::Report::print_options(std::ostream &out, int indent) const {

  out
    << indent_str(indent) << "append: " << append << "\n"
    << indent_str(indent) << "output: " << output_path << "\n"
    << indent_str(indent) << "junit-output: " << junit_output_path << "\n"
    << indent_str(indent) << "print-kernel-before-running: " << print_kernel_before_running << "\n"
    << indent_str(indent) << "report-not-run: " << report_not_run << "\n"
    << indent_str(indent) << "tags:\n";

  for (auto const & tag : pivot_tags) {
    out << indent_str(indent + 1) << tag.first << ": " << tag.second << "\n";
  }

  out
    << indent_str(indent) << "verbose: " << verbose << "\n";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Options::About::About(cutlass::CommandLine const &cmdline) {
  help = cmdline.check_cmd_line_flag("help");
  version = cmdline.check_cmd_line_flag("version");
  device_info = cmdline.check_cmd_line_flag("device-info");
}

void Options::About::print_usage(std::ostream &out) const {

  out << "About:\n"
    << "  --version                                        ";

  print_version(out);

  out << "\n";
}

void Options::About::print_version(std::ostream &out) {
  out << "CUTLASS " << cutlass::getVersionString()
      << " built on " << __DATE__ << " at " << __TIME__;
  if (!cutlass::getGitRevision().empty()) out << " with commit " << cutlass::getGitRevision() << "";
}

void Options::About::print_options(std::ostream &out, int indent) const {

}

/////////////////////////////////////////////////////////////////////////////////////////////////

Options::Options(cutlass::CommandLine const &cmdline):
  cmdline(cmdline),
  device(cmdline),
  initialization(cmdline),
  library(cmdline),
  profiling(cmdline),
  verification(cmdline),
  report(cmdline),
  about(cmdline) {

  if (cmdline.check_cmd_line_flag("mode")) {
    std::string token;
    cmdline.get_cmd_line_argument("mode", token);
    execution_mode = from_string<ExecutionMode>(token);
  }
  else {
    execution_mode = ExecutionMode::kProfile;
  }

  // Enumerating kernels is equivalent to a dry run.
  if (execution_mode == ExecutionMode::kEnumerate) {
    execution_mode = ExecutionMode::kDryRun;
  }

  if (cmdline.check_cmd_line_flag("operation")) {
    std::string str;
    cmdline.get_cmd_line_argument("operation", str);
    operation_kind = library::from_string<library::OperationKind>(str);
  }
  else if (cmdline.check_cmd_line_flag("function")) {
    std::string str;
    cmdline.get_cmd_line_argument("function", str);
    operation_kind = library::from_string<library::OperationKind>(str);
  }
  else {
    operation_kind = library::OperationKind::kInvalid;
  }

  if (cmdline.check_cmd_line_flag("operation_names")) {
    cmdline.get_cmd_line_arguments("operation_names", operation_names);
  }
  else if (cmdline.check_cmd_line_flag("kernels")) {
    cmdline.get_cmd_line_arguments("kernels", operation_names);
  }

  if (cmdline.check_cmd_line_flag("kernels-file")) {
    std::string filename;
    cmdline.get_cmd_line_argument("kernels-file", filename, {});
    std::ifstream input(filename);
    if (!input.good()) {
      throw std::runtime_error("failed to open: " + filename);
    }
    for (std::string line; getline(input, line);) {
      operation_names.push_back(line);
    }
  } else if (cmdline.check_cmd_line_flag("testlist-file")) {
    // Problems file is a CSV, where the first column is the kernel name and the rest are the problem arguments
    std::string filename;
    cmdline.get_cmd_line_argument("testlist-file", filename, {});
    std::ifstream input(filename);
    if (!input.good()) {
      throw std::runtime_error("failed to open: " + filename);
    }

    std::string line;
    std::vector<std::string> col_names;
    // Read header line
    if (std::getline(input, line)) {
      std::stringstream ss(line);
      std::string header;
      while (std::getline(ss, header, ',')) {
        col_names.push_back(header);
      }
    }

    // Read content lines
    while (std::getline(input, line)) {
      std::stringstream ss(line);
      std::string item;

      size_t colIdx = 0;
      std::string operation_name;

      std::unordered_map<std::string, std::string> arguments;

      while (std::getline(ss, item, ',')) {
        if (!colIdx) {
          // First column is operation name
          if (operation_problems.find(item) == operation_problems.end()) {
            operation_names.push_back(item);
          }
          operation_name = item;
        } else {
          if (colIdx < col_names.size()) {
            arguments[col_names[colIdx]] = item;
          }
        }
        colIdx++;
      }
      operation_problems[operation_name].emplace_back(arguments);
    }
  }

  if (cmdline.check_cmd_line_flag("ignore-kernels")) {
    cmdline.get_cmd_line_arguments("ignore-kernels", excluded_operation_names);
  }

  profiling.error_on_no_match            = cmdline.check_cmd_line_flag("error-on-no-match");
  profiling.error_if_nothing_is_profiled = cmdline.check_cmd_line_flag("error-if-nothing-is-profiled");

  // Prevent launches on the device for anything other than CUTLASS operation
  // Allow verification only on host
  if (execution_mode == ExecutionMode::kTrace) {
    initialization.provider = library::Provider::kReferenceHost;
    verification.providers = {library::Provider::kReferenceHost};
    profiling.enabled = false;
  }
}

void Options::print_usage(std::ostream &out) const {

  out
    << "CUTLASS Profiler\n"
    << "usage:\n\n"
    << "    cutlass_profiler [options]\n\n"
    << "  --help\n\n"

    << "  --mode=<string>                              "
    << "    Cutlass profiler execution mode." << end_of_line
    << "       --mode=profile    regular verification and profiling (default)" << end_of_line
    << "       --mode=dry_run    no kernels are launched or workspaces allocated" << end_of_line
    << "       --mode=enumerate  lists all operation kind and operations" << end_of_line
    << "       --mode=trace      executes a single device-side computation with" << end_of_line
    << "                          no other kernel launches\n\n"

    << "  --device-info                                "
    << "    Prints information on all GPUs present in the system\n\n"

    << "  --operation=<operation_kind>                 "
    << "    CUTLASS operation to profile.\n\n"

    << "  --kernels=<string_list>                      "
    << "    Filter operations by kernel names. For example, call all kernels with" << end_of_line
    << "      (\"s1688\" and \"nt\") or (\"s844\" and \"tn\" and \"align8\") in their" << end_of_line
    << "      operation name using --kernels=\"s1688*nt, s884*tn*align8\"\n\n"

    << "  --kernels-file=<filename>                      "
    << "    Same behavior as --kernels, but kernel names are specified in a file" << end_of_line
    << "    with one kernel on each line. Set of profiled kernels is the union of kernels specified" << end_of_line
    << "    here and those specified in `kernels`.\n\n"

    << "  --ignore-kernels=<string_list>               "
    << "    Excludes kernels whose names match anything in this list.\n\n"

    << "  --testlist-file=<filename>               "
    << "    A CSV, where each row is a problem, where the first column is the kernel name and the rest are the problem arguments" << end_of_line
	<< "    The column names should match cutlass_profiler cmd line arguments. \n\n"
    ;

  //
  // Detailed options
  //

  device.print_usage(out);
  out << "\n";

  initialization.print_usage(out);
  out << "\n";

  library.print_usage(out);
  out << "\n";

  profiling.print_usage(out);
  out << "\n";

  verification.print_usage(out);
  out << "\n";

  report.print_usage(out);
  out << "\n";

  about.print_usage(out);
  out << "\n";
}

void Options::print_options(std::ostream &out) const {

  out
    << "options:\n"
    << "  help: " << about.help << "\n"
    << "  mode: " << to_string(execution_mode) << "\n";

  out
    << "  device:\n";
  device.print_options(out, 2);

  out
    << "  initialization:\n";
  initialization.print_options(out, 2);

  out
    << "  profiling:\n";
  profiling.print_options(out, 2);

  out
    << "  verification:\n";
  verification.print_options(out, 2);

  out
    << "  report:\n";
  report.print_options(out, 2);
}

std::string Options::indent_str(int indent) {
  return std::string(indent * 2, ' ');
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass
