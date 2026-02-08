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

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef __unix__
#include <unistd.h>
#elif defined(_WIN32) || defined(WIN32)
#include <windows.h>
#else
// sleep not supported
#endif

#include <cuda/atomic>

#include "cutlass/profiler/options.h"
#include "cutlass/profiler/operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"

#include "cutlass/trace.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {
///////////////////////////////////////////////////////////////////////////////////////////////////

OperationProfiler::OperationProfiler(): kind_(library::OperationKind::kInvalid) { }

/// Ctor
OperationProfiler::OperationProfiler(
  Options const &options,
  library::OperationKind kind,
  ArgumentDescriptionVector const &arguments,
  ProviderVector const & verification_providers
):
  kind_(kind), arguments_(arguments) {

  ArgumentDescriptionVector tile_description_arguments{
    {ArgumentTypeID::kEnumerated, {"op_class", "opcode-class"}, "Class of math instruction (simt, tensorop, wmmatensorop, wmma)"},
    {ArgumentTypeID::kEnumerated, {"accum", "accumulator-type"}, "Math instruction accumulator data type"},
    {ArgumentTypeID::kInteger, {"cta_m", "threadblock-shape::m"}, "Threadblock shape in the M dimension"},
    {ArgumentTypeID::kInteger, {"cta_n", "threadblock-shape::n"}, "Threadblock shape in the N dimension"},
    {ArgumentTypeID::kInteger, {"cta_k", "threadblock-shape::k"}, "Threadblock shape in the K dimension"},
    {ArgumentTypeID::kInteger, {"cluster_m", "cluster-shape::m"}, "Cluster shape in the M dimension"},
    {ArgumentTypeID::kInteger, {"cluster_n", "cluster-shape::n"}, "Cluster shape in the N dimension"},
    {ArgumentTypeID::kInteger, {"cluster_k", "cluster-shape::k"}, "Cluster shape in the K dimension"},
    
    {ArgumentTypeID::kInteger, {"cluster_m_fallback", "cluster-shape-fallback::m"}, "Fallback Cluster shape in the M dimension"},
    {ArgumentTypeID::kInteger, {"cluster_n_fallback", "cluster-shape-fallback::n"}, "Fallback Cluster shape in the N dimension"},
    {ArgumentTypeID::kInteger, {"cluster_k_fallback", "cluster-shape-fallback::k"}, "Fallback Cluster shape in the K dimension"},
    
    {ArgumentTypeID::kInteger, {"stages", "threadblock-stages"}, "Number of stages of threadblock-scoped matrix multiply"},
    {ArgumentTypeID::kInteger, {"warps_m", "warp-count::m"}, "Number of warps within threadblock along the M dimension"},
    {ArgumentTypeID::kInteger, {"warps_n", "warp-count::n"}, "Number of warps within threadblock along the N dimension"},
    {ArgumentTypeID::kInteger, {"warps_k", "warp-count::k"}, "Number of warps within threadblock along the K dimension"},
    {ArgumentTypeID::kInteger, {"inst_m", "instruction-shape::m"}, "Math instruction shape in the M dimension"},
    {ArgumentTypeID::kInteger, {"inst_n", "instruction-shape::n"}, "Math instruction shape in the N dimension"},
    {ArgumentTypeID::kInteger, {"inst_k", "instruction-shape::k"}, "Math instruction shape in the K dimension"},
    {ArgumentTypeID::kInteger, {"min_cc", "minimum-compute-capability"}, "Minimum device compute capability"},
    {ArgumentTypeID::kInteger, {"max_cc", "maximum-compute-capability"}, "Maximum device compute capability"}
  };

  arguments_.insert(arguments_.end(), tile_description_arguments.begin(), tile_description_arguments.end());

  for (auto provider : verification_providers) {
    if (std::find(
      options.verification.providers.begin(),
      options.verification.providers.end(),
      provider) != options.verification.providers.end()) {

      verification_providers_.push_back(provider);
    }
  }

}

/// Destructor
OperationProfiler::~OperationProfiler() {}

/// Gets the schema description
std::string const & OperationProfiler::description() const {
  return description_;
}

/// Prints usage statement for the math function
void OperationProfiler::print_usage(std::ostream &out) const {
  for (auto const & desc : arguments_) {

    size_t const kAliasStart = 10;

    size_t columns = 0;

    std::string type_str = to_string(desc.type);
    columns += type_str.size();

    out << "  [" << type_str << "]";

    if (columns < kAliasStart) {
      out << std::string(kAliasStart - columns, ' ');
    }

    columns = 0;

    int j = 0;
    for (auto const & alias : desc.aliases) {
      columns += alias.size() + (j ? 1 : 0) + 2;

      out << (j++ ? "," : "") << "--" << alias;
    }

    size_t const kTotalColumns = 50;

    if (columns < kTotalColumns) {
      out << std::string(kTotalColumns - columns, ' ');
    }

    out << desc.description << "\n";
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if the current operation description satisfies the problem space
bool OperationProfiler::satisfies(
  library::OperationDescription const &op_desc,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::OpcodeClassID opcode_class;
  if (arg_as_OpcodeClassID(opcode_class, "op_class", problem_space, problem)) {
    if (opcode_class != op_desc.tile_description.math_instruction.opcode_class) {
      return false;
    }
  }
  
  bool dynamic_cluster = int64_t(op_desc.tile_description.cluster_shape.m()) == 0 ||
                         int64_t(op_desc.tile_description.cluster_shape.n()) == 0 ||
                         int64_t(op_desc.tile_description.cluster_shape.k()) == 0;
  
  int64_t int_value;

  if (arg_as_int(int_value, "inst_m", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.math_instruction.instruction_shape.m()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "inst_n", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.math_instruction.instruction_shape.n()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "inst_k", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.math_instruction.instruction_shape.k()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cta_m", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.threadblock_shape.m()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cta_n", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.threadblock_shape.n()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cta_k", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.threadblock_shape.k()) != int_value) {
      return false;
    }
  }

  if (!dynamic_cluster) { 
  if (arg_as_int(int_value, "cluster_m", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.cluster_shape.m()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cluster_n", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.cluster_shape.n()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cluster_k", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.cluster_shape.k()) != int_value) {
      return false;
    }
  }

  } 
  if (arg_as_int(int_value, "stages", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.threadblock_stages) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "warps_m", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.warp_count.m()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "warps_n", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.warp_count.n()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "warps_k", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.warp_count.k()) != int_value) {
      return false;
    }
  }

  library::NumericTypeID numeric_type;
  if (arg_as_NumericTypeID(numeric_type, "accum", problem_space, problem)) {
    if (numeric_type != op_desc.tile_description.math_instruction.element_accumulator) {
      return false;
    }
  }

  return true;
}

#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)

std::ostream& operator<<(std::ostream& out, library::Provider provider) {
  if (provider == library::Provider::kNone) {
    out << "kNone";
  }
  else if (provider == library::Provider::kCUTLASS) {
    out << "kCUTLASS";
  }
  else if (provider == library::Provider::kReferenceHost) {
    out << "kReferenceHost";
  }
  else if (provider == library::Provider::kReferenceDevice) {
    out << "kReferenceDevice";
  }
  else if (provider == library::Provider::kCUBLAS) {
    out << "kCUBLAS";
  }
  else if (provider == library::Provider::kCUDNN) {
    out << "kCUDNN";
  }
  else {
    out << "kInvalid";
  }

  return out;
}

std::ostream& operator<<(std::ostream& out, library::OperationKind op_kind) {
  if (op_kind == library::OperationKind::kGemm) {
    out << "kGemm";
  }
  else if (op_kind == library::OperationKind::kBlockScaledGemm) {
    out << "kBlockScaledGemm";
  }
  else if (op_kind == library::OperationKind::kBlockwiseGemm) {
    out << "kBlockwiseGemm";
  }
  else if (op_kind == library::OperationKind::kRankK) {
    out << "kRankK";
  }
  else if (op_kind == library::OperationKind::kRank2K) {
    out << "kRank2K";
  }
  else if (op_kind == library::OperationKind::kTrmm) {
    out << "kTrmm";
  }
  else if (op_kind == library::OperationKind::kSymm) {
    out << "kSymm";
  }
  else if (op_kind == library::OperationKind::kConv2d) {
    out << "kConv2d";
  }
  else if (op_kind == library::OperationKind::kConv3d) {
    out << "kConv3d";
  }
  else if (op_kind == library::OperationKind::kEqGemm) {
    out << "kEqGemm";
  }
  else if (op_kind == library::OperationKind::kSparseGemm) {
    out << "kSparseGemm";
  }
  else if (op_kind == library::OperationKind::kReduction) {
    out << "kReduction";
  }
  else if (op_kind == library::OperationKind::kGroupedGemm) {
    out << "kGroupedGemm";
  }
  else {
    out << "kInvalid";
  }

  return out;
}

#endif // defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)

/// Entry point to profile all operations in the manifest
int OperationProfiler::profile_all(
  Options const &options,
  library::Manifest const &manifest,
  DeviceContext &device_context) {
  ProblemSpace cmdline_problem_space(arguments_, options.cmdline);

  bool do_testlist_run = !options.operation_problems.empty();

  std::vector<std::pair<std::string, std::unique_ptr<ProblemSpace>>> all_operations_and_problems;
  if (do_testlist_run) {
    for (const auto& [operation_name, cmd_vec] : options.operation_problems)  {
      for (auto& cmd_line : cmd_vec) {
        all_operations_and_problems.push_back({operation_name, std::make_unique<ProblemSpace>(arguments_, cmd_line)});
      }
    }
  }

  // 1. Construct performance report
  PerformanceReport report(options, cmdline_problem_space.argument_names(), kind_);

  //
  int retval = 0;

  size_t bound = (all_operations_and_problems.empty() ? 1 : all_operations_and_problems.size());
  for (size_t i = 0; i < bound; i++) {

    // New problem space for each operation if we are running a testlist
    ProblemSpace& problem_space = do_testlist_run ? *all_operations_and_problems[i].second : cmdline_problem_space;

    // 2. For each problem in problem space
    ProblemSpace::Iterator problem_it = problem_space.begin();
    ProblemSpace::Iterator problem_end = problem_space.end();

    bool continue_profiling = true;

    // For each problem in problem space
    for (; continue_profiling && problem_it != problem_end; ++problem_it) {
      ProblemSpace::Problem problem = problem_it.at();
      report.next_problem();

      // For each operation in manifest
      int matched_operation_count = 0;
      int profiled_operation_count = 0;
      for (auto const& operation_ptr : manifest) {

        library::Operation const *operation = operation_ptr.get();
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        std::cerr << "  Operation: " << typeid(*operation).name() << "\n"
                  << "    name: " << operation->description().name << "\n"
                  << "    kind: " << operation->description().kind << "\n"
                  << "    provider: " << operation->description().provider << "\n";
#endif // CUTLASS_DEBUG_TRACE_LEVEL

        auto min_cc = operation->description().tile_description.minimum_compute_capability;
        auto max_cc = operation->description().tile_description.maximum_compute_capability;

#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        std::cerr << "    min_cc: " << min_cc << "\n";
        std::cerr << "    max_cc: " << min_cc << "\n";
#endif

        // Clear named allocations
        device_context.free();

#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        if (operation->description().kind != kind_) {
          std::cerr << "    @ kind " << operation->description().kind
                    << " != kind_ " << kind_ << "\n";
        }
        if (operation->description().provider != library::Provider::kCUTLASS) {
          std::cerr << "    @ provider " << operation->description().provider
                    << " != library::Provider::kCUTLASS\n";
        }
        if (options.device.compute_capability(0) < min_cc) {
          std::cerr << "    @ compute_capability "
                    << options.device.compute_capability(0)
                    << " < min_cc " << min_cc << "\n";
        }
        if (options.device.compute_capability(0) > max_cc) {
          std::cerr << "    @ compute_capability "
                    << options.device.compute_capability(0)
                    << " > max_cc " << max_cc << "\n";
        }
#endif

        // Execute compatible cutlass operations if they satisfy the current device's compute capability
        if (operation->description().kind == kind_ &&
            operation->description().provider == library::Provider::kCUTLASS &&
            options.device.compute_capability(0) >= min_cc &&
            options.device.compute_capability(0) <= max_cc) {

          std::string operation_name(operation->description().name);
          // Filter kernels by name
          bool filtered_by_name = options.operation_names.empty();
          if (!filtered_by_name) {

            for (auto const & op_name : options.operation_names) {
              if (find_string_matches_(op_name, operation_name)) {
                filtered_by_name = true;
                break;
              }
            }
          }

          for (auto const & op_name : options.excluded_operation_names) {
            if (find_string_matches_(op_name, operation_name)) {
              filtered_by_name = false;
              break;
            }
          }

          // Problems list uses exact match on operation names
          if (do_testlist_run && !(all_operations_and_problems[i].first == operation_name)) {
            filtered_by_name = false;
          }

          if (!filtered_by_name || !satisfies(operation->description(), problem_space, problem)) {
            continue;
          }

          // we have found a kernel match, so increment the counter for match kernels
          ++matched_operation_count;

          // A. Initialize configuration
          Status status = this->initialize_configuration(
            options,
            report,
            device_context,
            operation,
            problem_space,
            problem);

          if (status == Status::kErrorInternal) {

            // If there was an internal error, consume the CUDA error and move to the next operation.
            (void)cudaGetLastError();

            report.append_result(model_result_);
            continue;
          }
          else if (status != Status::kSuccess) {
            // If the workspace could not be initialized for any other reason, continue to
            // the next operation.
            continue;
          }

          if (continue_profiling) {

            if (options.report.print_kernel_before_running) {
              std::cout << "Profiling kernel for JUnit test " << options.report.junit_output_path << ": "
                        << operation_name << std::endl;
            }

            status = this->initialize_workspace(
              options,
              report,
              device_context,
              operation,
              problem_space,
              problem);

            if (status == Status::kErrorInternal) {

              // If there was an internal error, consume the CUDA error and move to the next operation.
              (void)cudaGetLastError();

              report.append_results(results_);
              continue;
            }
            else if (status != Status::kSuccess) {
              // If the workspace could not be initialized for any other reason, continue to
              // the next operation.
              continue;
            }
          }

          //
          // Profile CUTLASS if it is enabled
          //

          // B. Verify CUTLASS
          if (continue_profiling && options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

            continue_profiling = this->verify_cutlass(
              options,
              report,
              device_context,
              operation,
              problem_space,
              problem);

            retval |= (not continue_profiling);
          }

          if (options.execution_mode == ExecutionMode::kDryRun) {
            report.append_results(results_);
            results_.clear();
            continue;
          }

          //
          // C. Optionally save workspace
          //

          if (options.verification.save_workspace == SaveWorkspace::kAlways) {
            save_workspace(
              device_context,
              options,
              operation->description(),
              library::Provider::kCUTLASS);
          }

          //
          // D. Profile
          //

          if (continue_profiling && options.profiling.enabled) {

            continue_profiling = this->profile(
              options,
              report,
              device_context,
              operation,
              problem_space,
              problem);

            // Count op as profiled, even it failed to profile
            profiled_operation_count++;
          }

          report.append_results(results_);
          results_.clear();
        } // if op satisfied compute capacity

        if (!continue_profiling) {
          // break out of `for op in manifest` loop and move to next problem
          // `for each problem in problem space` conditional check on not continue profiling
          break;
        }
      } // for op in manifest

      // If we did not find any kernels that match our filters and error_on_no_match was set, report an error
      if (options.profiling.error_on_no_match && matched_operation_count <= 0) {
        #if !NDEBUG
        std::cerr << "Error: No matching kernels found with kernel selection filters [--error_on_no_match]" << std::endl;
        #endif
        retval |= 1;
        // Stop profiling on error no match
        continue_profiling = false;
      }

      if (options.profiling.error_if_nothing_is_profiled && options.profiling.enabled && profiled_operation_count <= 0) {
        #if !NDEBUG
        std::cerr << "Error: No kernels profiled found with kernel selection filters [--error_if_nothing_is_profiled]" << std::endl;
        #endif
        retval |= 1;
        // Stop profiling on error no match
        continue_profiling = false;
      }

    } // for each problem in problem space
  }

  return retval;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Sleep for a given duration in ms
void OperationProfiler::sleep(int sleep_duration) {
  if (sleep_duration) {
    #ifdef __unix__
    usleep(sleep_duration * 1000);
    #elif defined(_WIN32) || defined(WIN32)
    SleepEx(sleep_duration, false);
    #else
    // sleep not supported
    #endif
  }
}


/// Compares tensors for equality
Disposition OperationProfiler::compare_tensors(
  Options const &options,
  DeviceAllocation &experimental,
  DeviceAllocation &reference,
  int64_t count) {

  if (experimental.type() != reference.type()) {
    return Disposition::kIncorrect;
  }

  bool passed = false;

  if (count == 0) {
    count = reference.capacity();
  }

  if (options.verification.epsilon == 0) {

    // bit-level equality
    passed = DeviceAllocation::block_compare_equal(
      experimental.type(),
      experimental.data(),
      reference.data(),
      count);
  }
  else {

    // relative error function
    passed = DeviceAllocation::block_compare_relatively_equal(
      experimental.type(),
      experimental.data(),
      reference.data(),
      count,
      options.verification.epsilon,
      options.verification.nonzero_floor);
  }

  return passed ? Disposition::kPassed : Disposition::kIncorrect;
}

/// Saves the workspace
void OperationProfiler::save_workspace(
  DeviceContext &device_context,
  Options const &options,
  library::OperationDescription const &desc,
  library::Provider provider,
  library::Provider verification_provider) {

  for (auto const & named_allocation : device_context) {

    DeviceAllocation *allocation = named_allocation.second;

    if (allocation->layout() == library::LayoutTypeID::kUnknown) {
      continue; // write_tensor not set up to handle DeviceAllocations initialized using
                // allocate_block()
    }

    std::stringstream filename;

    filename << desc.name << "_" << library::to_string(provider) << "_";

    if (verification_provider != library::Provider::kInvalid) {
      filename << "verified_by_" << library::to_string(verification_provider) << "_";
    }

    filename << named_allocation.first + ".mat";

    std::ofstream out(filename.str());

    allocation->write_tensor_csv(out);
    out << "\n";

    if (options.report.verbose) {
      std::cout << "wrote '" << filename.str() << "'" << std::endl;
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
extern "C" {
__global__ void delay(cuda::atomic<bool> const *release) {
  while (release->load(cuda::memory_order_acquire) != true) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    __nanosleep(100);
#endif
  }
}
}

Status predict_iters(
  int &iterations,
  Options const &options,
  const std::function<Status(cudaStream_t, int)> &func,
  cudaStream_t stream) {
  // always use profiling-iterations if requested
  if (options.profiling.iterations != 0) {
    iterations = options.profiling.iterations;
    return Status::kSuccess;
  }

  // otherwise run for as many iterations as necessary to
  // meet profiling-duration
  constexpr int CALIBRATION_ITERS = 5;
  GpuTimer timer;
  timer.start(stream);
  for (int i = 0; i < CALIBRATION_ITERS; i++) {
    Status status = func(stream, i);
    if (status != Status::kSuccess) {
      return status;
    }
  }
  timer.stop_and_wait(stream);

  double est_iters             = options.profiling.duration / std::max(timer.duration(CALIBRATION_ITERS), 1e-6);
  constexpr uint64_t MAX_ITERS = 1'000'000;
  iterations = std::min(static_cast<uint64_t>(std::ceil(est_iters)), static_cast<uint64_t>(MAX_ITERS));
  iterations = std::max(options.profiling.min_iterations, iterations);
  return Status::kSuccess;
};

} // namespace

/// This profiling method is designed to run a kernel on several GPUs to
/// measure interference (e.g. due to power throttling).
/// To encourage the kernels to start at the same time and minimize jitter,
/// a spinloop kernel blocks each stream while work is being enqueued, which is
/// later triggered from the host.
/// CUDA graphs allows you to record the launch of large numbers of kernels without
/// blocking and therefore avoids a deadlock which happens if you try to enqueue too
/// many kernels behind the spinloop kernel.
Status OperationProfiler::profile_kernel_w_cuda_graphs_(
  PerformanceResult& result,
  Options const& options,
  std::function<Status(int, cudaStream_t, int)> const& func,
  std::vector<cudaStream_t> const& streams) {

  auto dev_count = streams.size();

  cuda::atomic<bool> *release;

  if (dev_count > 1) {
    CUDA_CHECK(cudaHostAlloc(&release, sizeof(*release), cudaHostAllocPortable));
    release->store(false, cuda::memory_order_release);
  }

  std::vector<GpuTimer> timer;
  for (size_t i = 0; i < dev_count; ++i) {
    CUDA_CHECK(cudaSetDevice(options.device.device_id(i)));
    timer.emplace_back();
  }

  std::vector<cudaGraph_t> graphs;
  graphs.resize(dev_count);
  std::vector<cudaGraphExec_t> graphExecs;
  graphExecs.resize(dev_count);

  sleep(options.profiling.sleep_duration);

  // predict time by running on device 0
  int iterations;
  CUDA_CHECK(cudaSetDevice(0));
  Status status = predict_iters(
    iterations,
    options,
    [&](cudaStream_t stream, int iter) { return func(0, stream, iter); },
    streams[0]);
  if (status != Status::kSuccess) {
    return status;
  }

  for (size_t i = 0; i < dev_count; ++i) {
    CUDA_CHECK(cudaSetDevice(options.device.device_id(i)));
    CUDA_CHECK(cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeGlobal));
    if (dev_count > 1) {
      // Halt execution until all GPUs are ready to precede.
      // It allows the CPU to trigger the GPUs all start at the same time.
      delay<<<1, 1, 0, streams[i]>>>(release);
    }
    for (int iteration = 0; iteration < options.profiling.warmup_iterations; ++iteration) {
      Status status = func(i, streams[i], iteration);
      if (status != Status::kSuccess) {
        return status;
      }
    }

    timer[i].start(streams[i], cudaEventRecordExternal);

    int iteration = 0;
    for (; iteration < iterations; ++iteration) {
      Status status = func(i, streams[i], iteration + options.profiling.warmup_iterations);
      if (status != Status::kSuccess) {
        return status;
      }
    }
    timer[i].stop(streams[i], cudaEventRecordExternal);
    CUDA_CHECK(cudaStreamEndCapture(streams[i], &graphs[i]));
    CUDA_CHECK(cudaGraphInstantiate(&graphExecs[i], graphs[i], nullptr, nullptr, 0));
  }

  for (size_t i = 0; i < dev_count; ++i) {
    CUDA_CHECK(cudaSetDevice(options.device.device_id(i)));
    CUDA_CHECK(cudaGraphLaunch(graphExecs[i], streams[i]));
  }

  if (dev_count > 1) {
    // release the enqueued kernels
    release->store(true, cuda::memory_order_release);
  }

  for (size_t i = 0; i < dev_count; ++i) {
    CUDA_CHECK(cudaSetDevice(options.device.device_id(i)));
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }

  result.runtime = 0;
  for (size_t i = 0; i < dev_count; ++i) {
    CUDA_CHECK(cudaSetDevice(options.device.device_id(i)));
    result.runtime_vector[i] = timer[i].duration(iterations);
    result.runtime += result.runtime_vector[i];
  }
  result.runtime /= static_cast<double>(dev_count);

  if (dev_count > 1) {
    CUDA_CHECK(cudaFreeHost(release));
  }

  for (size_t i = 0; i < dev_count; ++i) {
    CUDA_CHECK(cudaSetDevice(options.device.device_id(i)));
    CUDA_CHECK(cudaGraphExecDestroy(graphExecs[i]));
    CUDA_CHECK(cudaGraphDestroy(graphs[i]));
  }

  for (size_t i = 0; i < dev_count; ++i) {
    CUDA_CHECK(cudaSetDevice(options.device.device_id(dev_count - i - 1)));
    timer.pop_back();
  }

  return Status::kSuccess;
}

Status OperationProfiler::profile_kernel_(
  PerformanceResult &result,
  Options const &options,
  const std::function<Status(int, cudaStream_t, int)> &func,
  const std::vector<cudaStream_t> &streams) {

  if (options.profiling.use_cuda_graphs) {
    return profile_kernel_w_cuda_graphs_(result, options, func, streams);
  }
  else if (streams.size() == 1) {
    auto single_device_func = [&](cudaStream_t stream, int iteration) {
      return func(0, stream, iteration);
    };
    return profile_kernel_no_cuda_graphs_(result, options, single_device_func, streams[0]);
  }
  return Status::kErrorNotSupported;
}

/// Method to profile GPU execution time of a kernel launched in func
Status OperationProfiler::profile_kernel_(
  PerformanceResult& result,
  Options const& options,
  std::function<Status(cudaStream_t, int)> const& func,
  cudaStream_t stream) {

  if (options.profiling.use_cuda_graphs) {
    auto graph_func = [&](int dev_id, cudaStream_t stream, int iteration) {
      return func(stream, iteration);
    };
    return profile_kernel_w_cuda_graphs_(result, options, graph_func, {stream});
  } else {
    return profile_kernel_no_cuda_graphs_(result, options, func, stream);
  }
  return Status::kSuccess;
}

/// Method to profile GPU execution time of a kernel launched in func
Status OperationProfiler::profile_kernel_no_cuda_graphs_(
  PerformanceResult& result,
  Options const& options,
  std::function<Status(cudaStream_t, int)> const& func,
  cudaStream_t stream) {

  GpuTimer timer;
  // Optional sleep to limit power consumption and thermals
  sleep(options.profiling.sleep_duration);

  Status status = Status::kSuccess;

  int iterations;
  status = predict_iters(iterations, options, func, stream);
  if (status != Status::kSuccess) {
    return status;
  }

  for (int iteration = 0; iteration < options.profiling.warmup_iterations; ++iteration) {
    status = func(stream, iteration);
    if (status != Status::kSuccess) {
      return status;
    }
  }

  timer.start(stream);

  int iteration = 0;
  for (; iteration < iterations; ++iteration) {
    status = func(stream, iteration + options.profiling.warmup_iterations);

    if (status != Status::kSuccess) {
      result.status = status;
      return status;
    }
  }

  timer.stop_and_wait(stream);

  result.runtime = timer.duration(iteration);
  result.status  = status;

  return status;
}

/// Method to profile a CUTLASS Operation
Status OperationProfiler::profile_cutlass_(
  PerformanceResult &result,
  Options const &options,
  library::Operation const *operation,
  void *arguments,
  void *host_workspace,
  void *device_workspace) {

  auto op = [=](cudaStream_t, int) { return operation->run(arguments, host_workspace, device_workspace); };
  return profile_kernel_(result, options, op);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Sets operation description
void OperationProfiler::initialize_result_(
  PerformanceResult &result,
  library::OperationDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  set_argument(result, "op_class", problem_space,
    library::to_string(operation_desc.tile_description.math_instruction.opcode_class));

  set_argument(result, "accum", problem_space,
    library::to_string(operation_desc.tile_description.math_instruction.element_accumulator));

  set_argument(result, "cta_m", problem_space, operation_desc.tile_description.threadblock_shape.m());
  set_argument(result, "cta_n", problem_space, operation_desc.tile_description.threadblock_shape.n());
  set_argument(result, "cta_k", problem_space, operation_desc.tile_description.threadblock_shape.k());
  set_argument(result, "stages", problem_space, operation_desc.tile_description.threadblock_stages);
  set_argument(result, "warps_m", problem_space, operation_desc.tile_description.warp_count.m());
  set_argument(result, "warps_n", problem_space, operation_desc.tile_description.warp_count.n());
  set_argument(result, "warps_k", problem_space, operation_desc.tile_description.warp_count.k());
  set_argument(result, "inst_m", problem_space, operation_desc.tile_description.math_instruction.instruction_shape.m());
  set_argument(result, "inst_n", problem_space, operation_desc.tile_description.math_instruction.instruction_shape.n());
  set_argument(result, "inst_k", problem_space, operation_desc.tile_description.math_instruction.instruction_shape.k());
  set_argument(result, "min_cc", problem_space, operation_desc.tile_description.minimum_compute_capability);
  set_argument(result, "max_cc", problem_space, operation_desc.tile_description.maximum_compute_capability);
}

/// Helper
void OperationProfiler::set_argument(
  PerformanceResult &result,
  char const *name,
  ProblemSpace const &problem_space,
  std::string const &value) {

  result.arguments.at(problem_space.argument_index(name)) = make_pair(std::string(name), value);
}

void OperationProfiler::set_argument(
  PerformanceResult &result,
  char const *name,
  ProblemSpace const &problem_space,
  int64_t value) {

  result.arguments.at(problem_space.argument_index(name)) = make_pair(std::string(name), library::lexical_cast(value));
}


/// finds string matches filter_string in operation_name
bool OperationProfiler::find_string_matches_(
  std::string const &filter_string,
  std::string const &operation_name) {
  // Returns true if all substrings appear in the operation_name in order

  // Split filter_string of the format "gemm*f32*nt" to tokens ["gemm", "f32", "nt"]
  std::string item;
  std::istringstream iss(filter_string);
  std::vector<std::string> filter_tokens;
  while (std::getline(iss, item, '*')) {
    filter_tokens.push_back(item);
  }

  // Search filter_tokens in operation_name in order
  size_t start = 0, idx = 0;
  for (auto & token : filter_tokens) {
    // Check if characters left to be parsed in operation_name
    if (start < operation_name.length()) {
      // Find token in operation_name[start:]
      idx = operation_name.substr(start).find(token);
      if (idx == std::string::npos) {
        return false;
      }
    }
    start += (idx + token.length());
  }

  // All tokens in filter_string found in operation_name
  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
