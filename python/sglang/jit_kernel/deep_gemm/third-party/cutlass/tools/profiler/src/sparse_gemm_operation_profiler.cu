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
   \brief Execution environment

*/

#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <ios>

#include "cutlass/profiler/cublas_helpers.h"
#include "cutlass/profiler/sparse_gemm_operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Ctor
SparseGemmOperationProfiler::SparseGemmOperationProfiler(Options const &options):
  OperationProfiler(
    options,
    library::OperationKind::kSparseGemm,
    {
      {ArgumentTypeID::kEnumerated, {"gemm_kind"}, "Variant of GEMM (e.g. sparse, ...)"},
      {ArgumentTypeID::kInteger, {"m", "problem-size::m"}, "M dimension of the GEMM problem space"},
      {ArgumentTypeID::kInteger, {"n", "problem-size::n"}, "N dimension of the GEMM problem space"},
      {ArgumentTypeID::kInteger, {"k", "problem-size::k"}, "K dimension of the GEMM problem space"},
      {ArgumentTypeID::kTensor, {"A"}, "Tensor storing the A operand"},
      {ArgumentTypeID::kTensor, {"B"}, "Tensor storing the B operand"},
      {ArgumentTypeID::kTensor, {"C"}, "Tensor storing the C operand"},
      {ArgumentTypeID::kTensor, {"E"}, "Tensor storing the E operand"},
      {ArgumentTypeID::kScalar, {"alpha", "epilogue::alpha"}, "Epilogue scalar alpha"},
      {ArgumentTypeID::kScalar, {"beta", "epilogue::beta"}, "Epilogue scalar beta"},
      {ArgumentTypeID::kInteger, {"split_k_slices"}, "Number of partitions of K dimension"},
      {ArgumentTypeID::kInteger, {"batch_count"}, "Number of GEMMs computed in one batch"},
    }
  ) {

  description_ = "      Structured sparse GEMM. D = alpha * A*B + beta * C";
}

/// Destructor
SparseGemmOperationProfiler::~SparseGemmOperationProfiler() {

}

/// Prints usage statement for the math function
void SparseGemmOperationProfiler::print_usage(std::ostream &out) const {
  out << "Sparse GEMM" << "\n\n";

  OperationProfiler::print_usage(out);
}

/// Prints examples
void SparseGemmOperationProfiler::print_examples(std::ostream &out) const {

  out << "\nExamples:\n\n"
    << "Profile a particular problem size:\n"
    << "  $ cutlass_profiler --operation=SparseGemm --m=1024 --n=1024 --k=128\n\n"

    << "Schmoo over problem size and beta:\n"
    << "  $ cutlass_profiler --operation=SparseGemm --m=1024:4096:256 --n=1024:4096:256 --k=128:8192:128 --beta=0,1,2.5\n\n"

    << "Schmoo over accumulator types:\n"
    << "  $ cutlass_profiler --operation=SparseGemm --accumulator-type=f16,f32\n\n"

    << "Run when A is f16 with column-major and B is any datatype with row-major (For column major, use column, col, or n. For row major use, row or t):\n"
    << "  $ cutlass_profiler --operation=SparseGemm --A=f16:column --B=*:row\n\n"

    << "Using various input value distribution:\n"
    << "  $ cutlass_profiler --operation=SparseGemm --dist=uniform,min:0,max:3\n"
    << "  $ cutlass_profiler --operation=SparseGemm --dist=gaussian,mean:0,stddev:3\n"
    << "  $ cutlass_profiler --operation=SparseGemm --dist=sequential,start:0,delta:1\n\n"

    << "Run a kernel with cta tile size of 256x128x32 and save workspace if results are incorrect (note that --cta-tile::k=32 is default cta-tile size):\n"
    << " $ cutlass_profiler --operation=SparseGemm --cta_m=256 --cta_n=128  --cta_k=32 --save-workspace=incorrect\n\n"

    << "Test your changes to gemm kernels with a quick functional test and save results in functional-test.csv:\n"
    << " $ cutlass_profiler  --operation=SparseGemm \\ \n"
    << "   --m=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \\ \n"
    << "   --n=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \\ \n"
    << "   --k=8,16,32,64,128,256,288,384,504,512,520 \\ \n"
    << "   --beta=0,1,2 --profiling-iterations=1 \\ \n"
    << "   --providers=cutlass --output=functional-test.csv\n\n";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Status SparseGemmOperationProfiler::SparseGemmProblem::parse(
  library::SparseGemmDescription const &operation_desc,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  if (!arg_as_int(this->m, "m", problem_space, problem)) {
    // default value
    this->m = 1024;
  }

  if (!arg_as_int(this->n, "n", problem_space, problem)) {
    // default value
    this->n = 1024;
  }

  if (!arg_as_int(this->k, "k", problem_space, problem)) {
    // default value
    this->k = 1024;
  }

  if (!arg_as_int(this->split_k_slices, "split_k_slices", problem_space, problem)) {
    // default value
    this->split_k_slices = 1;
  }

  if (!arg_as_int(this->batch_count, "batch_count", problem_space, problem)) {
    // default value
    this->batch_count = 1;
  }

  if (!tensor_description_satisfies(operation_desc.A, "A", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!tensor_description_satisfies(operation_desc.B, "B", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!tensor_description_satisfies(operation_desc.C, "C", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!tensor_description_satisfies(operation_desc.E, "E", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!arg_as_scalar(
    this->alpha,
    operation_desc.element_epilogue,
    "alpha",
    problem_space,
    problem)) {

    if (!cast_from_double(this->alpha, operation_desc.element_epilogue, 1)) {
      return Status::kErrorInternal;
    }
  }

  if (!arg_as_scalar(
    this->beta,
    operation_desc.element_epilogue,
    "beta",
    problem_space,
    problem)) {

    if (!cast_from_double(this->beta, operation_desc.element_epilogue, 0)) {
      return Status::kErrorInternal;
    }
  }

  this->elements_per_128b =
      128 / library::sizeof_bits(operation_desc.A.element);

  this->lda = DeviceAllocation::get_packed_layout(
                  operation_desc.A.layout,
                  {int(this->m), int(this->k) / int(this->sparse)})
                  .front();

  this->ldb = DeviceAllocation::get_packed_layout(
    operation_desc.B.layout, {int(this->k), int(this->n)}).front();

  this->ldc = DeviceAllocation::get_packed_layout(
    operation_desc.C.layout, {int(this->m), int(this->n)}).front();

  this->lde =
      DeviceAllocation::get_packed_layout(
          operation_desc.E.layout,
          {int(this->m), int(this->k / this->sparse / this->elements_per_128b)})
          .front();

  return Status::kSuccess;
}

/// Initializes a performance result
void SparseGemmOperationProfiler::SparseGemmProblem::initialize_result(
  PerformanceResult &result,
  library::SparseGemmDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  result.arguments.resize(problem_space.rank());

  set_argument(result, "gemm_kind", problem_space, library::to_string(operation_desc.gemm_kind));

  set_argument(result, "A", problem_space,
    std::string(library::to_string(operation_desc.A.element)) + ":" + library::to_string(operation_desc.A.layout));

  set_argument(result, "B", problem_space,
    std::string(library::to_string(operation_desc.B.element)) + ":" + library::to_string(operation_desc.B.layout));

  set_argument(result, "C", problem_space,
    std::string(library::to_string(operation_desc.C.element)) + ":" + library::to_string(operation_desc.C.layout));

  set_argument(result, "E", problem_space,
    std::string(library::to_string(operation_desc.E.element)) + ":" + library::to_string(operation_desc.E.layout));

  set_argument(result, "m", problem_space, m);
  set_argument(result, "n", problem_space, n);
  set_argument(result, "k", problem_space, k);

  set_argument(result, "cluster_m", problem_space, operation_desc.tile_description.cluster_shape.m());
  set_argument(result, "cluster_n", problem_space, operation_desc.tile_description.cluster_shape.n());
  set_argument(result, "cluster_k", problem_space, operation_desc.tile_description.cluster_shape.k());

  set_argument(result, "split_k_slices", problem_space, split_k_slices);
  set_argument(result, "batch_count", problem_space, batch_count);

  set_argument(result, "alpha", problem_space,
    library::lexical_cast(alpha, operation_desc.element_epilogue));

  set_argument(result, "beta", problem_space,
    library::lexical_cast(beta, operation_desc.element_epilogue));
}

/// Extracts the problem dimensions
Status SparseGemmOperationProfiler::initialize_configuration(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::SparseGemmDescription const &operation_desc =
    static_cast<library::SparseGemmDescription const &>(operation->description());

  if (operation_desc.gemm_kind != library::GemmKind::kSparse) {
    return Status::kErrorInvalidProblem;
  }

  Status status = problem_.parse(operation_desc, problem_space, problem);

  if (status != Status::kSuccess) {
    return status;
  }

  gemm_workspace_.configuration.problem_size.m() = int(problem_.m);
  gemm_workspace_.configuration.problem_size.n() = int(problem_.n);
  gemm_workspace_.configuration.problem_size.k() = int(problem_.k);
  gemm_workspace_.configuration.lda = problem_.lda;
  gemm_workspace_.configuration.ldb = problem_.ldb;
  gemm_workspace_.configuration.ldc = problem_.ldc;
  gemm_workspace_.configuration.ldd = problem_.ldc;
  gemm_workspace_.configuration.lde = problem_.lde;

  gemm_workspace_.arguments.A = nullptr;
  gemm_workspace_.arguments.B = nullptr;
  gemm_workspace_.arguments.C = nullptr;
  gemm_workspace_.arguments.D = nullptr;
  gemm_workspace_.arguments.E = nullptr;
  gemm_workspace_.arguments.alpha = problem_.alpha.data();
  gemm_workspace_.arguments.beta = problem_.beta.data();
  gemm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  initialize_result_(this->model_result_, options, operation_desc, problem_space);

  return operation->can_implement(&gemm_workspace_.configuration, &gemm_workspace_.arguments);
}

/// Initializes the performance result
void SparseGemmOperationProfiler::initialize_result_(
  PerformanceResult &result,
  Options const &options,
  library::SparseGemmDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  result.provider = library::Provider::kCUTLASS;
  result.disposition = Disposition::kNotRun;
  result.status = Status::kSuccess;
  result.operation_name = operation_desc.name;

  problem_.initialize_result(result, operation_desc, problem_space);

  OperationProfiler::initialize_result_(result, operation_desc, problem_space);

  // Input bytes read and Output bytes written for the gemm problem
  result.bytes =
      int64_t(library::sizeof_bits(operation_desc.A.element) * problem_.m / 8) *
          problem_.k / problem_.sparse +
      int64_t(library::sizeof_bits(operation_desc.B.element) * problem_.n / 8) *
          problem_.k +
      int64_t(library::sizeof_bits(operation_desc.C.element) * problem_.m / 8) *
          problem_.n +
      int64_t(library::sizeof_bits(operation_desc.E.element) * problem_.m / 8) *
          problem_.k / problem_.sparse / problem_.elements_per_128b;

  // Set is_beta_zero true if beta is zero
  bool is_beta_zero = std::all_of(problem_.beta.begin(), problem_.beta.end(), [](uint8_t i) { return i==0; });

  // Output bytes read for the gemm problem for non-zero beta values
  if (!is_beta_zero) {
    result.bytes += int64_t(library::sizeof_bits(operation_desc.C.element) * problem_.m / 8) * problem_.n;
  }

  result.flops = 2 * (problem_.m * problem_.n * problem_.k + problem_.m * problem_.n);
  result.runtime = 0;

}

/// Initializes workspace
Status SparseGemmOperationProfiler::initialize_workspace(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  if (options.device.devices.size() != 1) {
    throw std::runtime_error("This operation profiler only supports a single "
                             "device.");
  }

  cudaError_t result;
  result = cudaSetDevice(options.device.device_id(0));
  if (result != cudaSuccess) {
    throw std::runtime_error("cudaSetDevice() failed.");
  }

  library::SparseGemmDescription const &operation_desc =
    static_cast<library::SparseGemmDescription const &>(operation->description());

  if (options.execution_mode != ExecutionMode::kDryRun) {
    int seed_shift = 0;
    gemm_workspace_.A = device_context.allocate_and_initialize_tensor(
      options,
      "A",
      operation_desc.A.element,
      operation_desc.A.layout,
      {int(problem_.m), int(problem_.k) / int(problem_.sparse)},
      {int(problem_.lda)},
      1, // batch_count
      seed_shift++,
      0 // device_index
    );

    gemm_workspace_.B = device_context.allocate_and_initialize_tensor(
      options,
      "B",
      operation_desc.B.element,
      operation_desc.B.layout,
      {int(problem_.k), int(problem_.n)},
      {int(problem_.ldb)},
      1, // batch_count
      seed_shift++,
      0 // device_index
    );

    gemm_workspace_.C = device_context.allocate_and_initialize_tensor(
      options,
      "C",
      operation_desc.C.element,
      operation_desc.C.layout,
      {int(problem_.m), int(problem_.n)},
      {int(problem_.ldc)},
      1, // batch_count
      seed_shift++,
      0 // device_index
    );

    gemm_workspace_.Computed = device_context.allocate_tensor(
      options,
      "D",
      operation_desc.C.element,
      operation_desc.C.layout,
      {int(problem_.m), int(problem_.n)},
      {int(problem_.ldc)},
      1, // batch_count
      0 // device_index
    );

    gemm_workspace_.E = device_context.allocate_and_initialize_sparsemeta_tensor(
      options,
      "E",
      operation_desc.E.element,
      operation_desc.E.layout,
      operation_desc.A.element,
      {int(problem_.m), int(problem_.k) / int(problem_.sparse) / int(problem_.elements_per_128b)},
      {int(problem_.lde)},
      1, // batch_count
      seed_shift++,
      0 // device_index
    );

    gemm_workspace_.Reference = device_context.allocate_tensor(
      options,
      "Reference",
      operation_desc.C.element,
      operation_desc.C.layout,
      {int(problem_.m), int(problem_.n)},
      {int(problem_.ldc)},
      1, // batch_count
      0 // device_index
    );

    gemm_workspace_.Reference->copy_from_device(gemm_workspace_.C->data());
  }

  //
  // Initialize the CUTLASS operation
  //

  Status status = Status::kSuccess;

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    if (options.execution_mode != ExecutionMode::kDryRun) {

      uint64_t workspace_size = operation->get_host_workspace_size(&gemm_workspace_.configuration);
      gemm_workspace_.host_workspace.resize(workspace_size, 0);

      workspace_size = operation->get_device_workspace_size(&gemm_workspace_.configuration);
      gemm_workspace_.device_workspace.reset(library::NumericTypeID::kU8, workspace_size);

      status = operation->initialize(
        &gemm_workspace_.configuration,
        gemm_workspace_.host_workspace.data(),
        gemm_workspace_.device_workspace.data());
    }

    //
    // If CUTLASS is enabled, generate a result for it
    //

    results_.push_back(model_result_);
    results_.back().provider = library::Provider::kCUTLASS;
    results_.back().op_kind = library::OperationKind::kSparseGemm;
    results_.back().disposition = Disposition::kNotRun;

    for(auto &verification_provider : options.verification.providers) {
      results_.back().verification_map[verification_provider] = Disposition::kNotRun;
    }
  }

  return status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Verifies CUTLASS against references
bool SparseGemmOperationProfiler::verify_cutlass(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  if (!options.profiling.provider_enabled(library::Provider::kCUTLASS)) {
    return true;
  }

  if (options.execution_mode == ExecutionMode::kDryRun) {
    return true;
  }

  // Initialize structure containing GEMM arguments
  gemm_workspace_.arguments.A = gemm_workspace_.A->data();
  gemm_workspace_.arguments.B = gemm_workspace_.B->data();
  gemm_workspace_.arguments.C = gemm_workspace_.C->data();
  gemm_workspace_.arguments.D = gemm_workspace_.Computed->data();
  gemm_workspace_.arguments.E = gemm_workspace_.E->data();
  gemm_workspace_.arguments.alpha = problem_.alpha.data();
  gemm_workspace_.arguments.beta = problem_.beta.data();
  gemm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  //
  // Run the CUTLASS operation
  //

  results_.back().status = operation->run(
    &gemm_workspace_.arguments,
    gemm_workspace_.host_workspace.data(),
    gemm_workspace_.device_workspace.data());

  if (results_.back().status != Status::kSuccess) {
    results_.back().disposition = Disposition::kFailed;
    return false;
  }

  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    results_.back().disposition = Disposition::kFailed;
    return false;
  }

  // CUTLASS op ran the but not yet verified against any verification provider
  results_.back().disposition = Disposition::kNotVerified;

  //
  // Run verification providers
  //

  if (options.verification.enabled) {

    // Update disposition to worst case verification outcome among all
    // verification providers which are supported
    bool is_any_verification_run_passed = false;

    for(auto &m : results_.back().verification_map) {
      if(m.second == Disposition::kFailed || m.second == Disposition::kIncorrect) {
        results_.back().disposition = m.second;
        return true;
      }
      if(!is_any_verification_run_passed && m.second == Disposition::kPassed) {
        is_any_verification_run_passed = true;
      }
    }

    if(is_any_verification_run_passed) {
      results_.back().disposition = Disposition::kPassed;
    }
  }

  // Return true means continue profiling
  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Measures performance results
bool SparseGemmOperationProfiler::profile(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    // Initialize structure containing GEMM arguments
    gemm_workspace_.arguments.A = gemm_workspace_.A->data();
    gemm_workspace_.arguments.B = gemm_workspace_.B->data();
    gemm_workspace_.arguments.C = gemm_workspace_.C->data();
    gemm_workspace_.arguments.D = gemm_workspace_.Computed->data();
    gemm_workspace_.arguments.E = gemm_workspace_.E->data();
    gemm_workspace_.arguments.alpha = problem_.alpha.data();
    gemm_workspace_.arguments.beta = problem_.beta.data();
    gemm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

    results_.back().status = profile_cutlass_(
      results_.back(),
      options,
      operation,
      &gemm_workspace_.arguments,
      gemm_workspace_.host_workspace.data(),
      gemm_workspace_.device_workspace.data()
    );
  }

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
