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

#include "cutlass/core_io.h"

#include "cutlass/profiler/cublas_helpers.h"
#include "cutlass/profiler/rank_2k_operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Ctor
Rank2KOperationProfiler::Rank2KOperationProfiler(Options const &options):
  OperationProfiler(
    options,
    library::OperationKind::kRank2K,
    {
      {ArgumentTypeID::kEnumerated, {"rank_k_kind"}, "Variant of RankK (universal)"},
      {ArgumentTypeID::kInteger, {"n", "problem-size::n"}, "N dimension of the RankK problem space"},
      {ArgumentTypeID::kInteger, {"k", "problem-size::k"}, "K dimension of the RankK problem space"},
      {ArgumentTypeID::kTensor, {"A"}, "Tensor storing the A operand"},
      {ArgumentTypeID::kTensor, {"B"}, "Tensor storing the B operand"},
      {ArgumentTypeID::kTensor, {"C"}, "Tensor storing the C operand"},
      {ArgumentTypeID::kEnumerated, {"fill_mode"}, "Fill Mode for RankK kernel (lower or upper)"},
      {ArgumentTypeID::kEnumerated, {"blas_mode"}, "Blas Mode for RankK kernel (symmetric or hermitian)"},
      {ArgumentTypeID::kScalar, {"alpha", "epilogue::alpha"}, "Epilogue scalar alpha"},
      {ArgumentTypeID::kScalar, {"beta", "epilogue::beta"}, "Epilogue scalar beta"},
      {ArgumentTypeID::kInteger, {"split_k_slices", "split-k-slices"}, "Number of partitions of K dimension"},
      {ArgumentTypeID::kInteger, {"batch_count", "batch-count"}, "Number of RankK computed in one batch"},
    },
    { library::Provider::kCUBLAS}
  ) {
  description_ = "      Rank 2k Update. D = alpha * (A*B^T + B*A^T) + beta * C (symmetric) or D = alpha * (A*B^H+B*A^H) + beta * C (hermitian)";
}

/// Destructor
Rank2KOperationProfiler::~Rank2KOperationProfiler() {

}

/// Prints usage statement for the math function
void Rank2KOperationProfiler::print_usage(std::ostream &out) const {
  out << "RankK" << "\n\n";

  OperationProfiler::print_usage(out);
}

/// Prints examples
void Rank2KOperationProfiler::print_examples(std::ostream &out) const {

  out << "\nExamples:\n\n"
    << "Profile a particular problem size Syrk kernel:\n"
    << "  $ cutlass_profiler --operation=rank_2k --blas_mode=symmetric --n=1024 --k=128\n\n"

    << "Profile a particular problem size Herk kernel:\n"
    << "  $ cutlass_profiler --operation=rank_2k --blas_mode=hermitian --n=1024 --k=128\n\n"

    << "Schmoo over problem size and beta:\n"
    << "  $ cutlass_profiler --operation=rank_2k --n=1024:4096:256 --k=128:8192:128 --beta=0,1,2.5\n\n"

    << "Schmoo over accumulator types:\n"
    << "  $ cutlass_profiler --operation=rank_2k --accumulator-type=f16,f32\n\n"

    << "Schmoo over fill modees:\n"
    << "  $ cutlass_profiler --operation=rank_2k --fill_mode=lower/upper\n\n"

    << "Run when A is f16 with column-major or A is any datatype with row-major (For column major, use column, col, or n. For row major use, row or t):\n"
    << "  $ cutlass_profiler --operation=rank_2k --A=f16:column or --A=*:row\n\n"

    << "Using various input value distribution:\n"
    << "  $ cutlass_profiler --operation=rank_2k --dist=uniform,min:0,max:3\n"
    << "  $ cutlass_profiler --operation=rank_2k --dist=gaussian,mean:0,stddev:3\n"
    << "  $ cutlass_profiler --operation=rank_2k --dist=sequential,start:0,delta:1\n\n"

    << "Run a kernel with cta tile size of 256x128x32 and save workspace if results are incorrect (note that --cta-tile::k=32 is default cta-tile size):\n"
    << " $ cutlass_profiler --operation=rank_2k --cta_m=256 --cta_n=128  --cta_k=32 --save-workspace=incorrect\n\n"

    << "Test your changes to rank_2k kernels with a quick functional test and save results in functional-test.csv:\n"
    << " $ cutlass_profiler  --operation=rank_2k \\ \n"
    << "   --n=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \\ \n"
    << "   --k=8,16,32,64,128,256,288,384,504,512,520 \\ \n"
    << "   --beta=0,1,2 --profiling-iterations=1 \\ \n"
    << "   --providers=cutlass --output=functional-test.csv\n\n";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
// used this for debugging
static std::string byte_string(std::vector<uint8_t> const &bytes) {
  std::stringstream ss;

  ss << "0x";

  for (size_t idx = bytes.size(); idx > 0; --idx) {
    ss << std::hex << std::setw(2) << std::setfill('0') << uint32_t(bytes.at(idx - 1));
  }

  return ss.str();
}
#endif

Status Rank2KOperationProfiler::RankKProblem::parse(
  library::RankKDescription const &operation_desc,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

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

  if (this->split_k_slices > 1 && this->batch_count > 1) {
    // At least one of these must be one
    return Status::kErrorInvalidProblem;
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

  this->lda = DeviceAllocation::get_packed_layout(
    operation_desc.A.layout, {int(this->n), int(this->k)}).front();

  this->ldb = DeviceAllocation::get_packed_layout(
    operation_desc.B.layout, {int(this->n), int(this->k)}).front();

  this->ldc = DeviceAllocation::get_packed_layout(
    operation_desc.C.layout, {int(this->n), int(this->n)}).front();

  return Status::kSuccess;
}

/// Total number of bytes loaded
int64_t Rank2KOperationProfiler::RankKProblem::bytes(library::RankKDescription const &operation_desc) const {
  // Input bytes read and Output bytes written for the gemm problem
  int64_t bytes =
    2 * int64_t(library::sizeof_bits(operation_desc.A.element) * n / 8) * k +
    2 * int64_t(library::sizeof_bits(operation_desc.B.element) * n / 8) * k +
    // Half matrix including the diagonal will have (N*(N+1))/2 elements
    int64_t(library::sizeof_bits(operation_desc.C.element) * n / 8) * (n+1) / 2;

  // Set is_beta_zero true if beta is zero
  bool is_beta_zero = std::all_of(beta.begin(), beta.end(), [](uint8_t i) { return i==0; });

  // Output bytes read for the gemm problem for non-zero beta values
  if (!is_beta_zero) {
    bytes += int64_t(library::sizeof_bits(operation_desc.C.element) * n / 8) * (n+1) / 2;
  }

  bytes *= batch_count;

  return bytes;
}

/// Total number of flops computed
int64_t Rank2KOperationProfiler::RankKProblem::flops(library::RankKDescription const &operation_desc) const {

  // FLOPs = 2 * n(n+1)k/2 [mma1] + 2 * n(n+1)k/2 [mma2] + 2 * n(n+1)/2 [epilogue]
  // FLOPs = n(n+1)(2k + 1)
  int64_t flops_ = n * (n + 1) * (2*k + 1);

  // complex-valued support
  switch (operation_desc.tile_description.math_instruction.math_operation) {
  case library::MathOperationID::kMultiplyAddComplex:
    flops_ *= 4;
    break;

  case library::MathOperationID::kMultiplyAddComplexFastF32:
    flops_ *= 4;
    break;

  case library::MathOperationID::kMultiplyAddGaussianComplex:
    flops_ *= 3;
    break;

  default: break;
  }

  return flops_;
}

/// Initializes a performance result
void Rank2KOperationProfiler::RankKProblem::initialize_result(
  PerformanceResult &result,
  library::RankKDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  result.arguments.resize(problem_space.rank());

  set_argument(result, "rank_k_kind", problem_space, library::to_string(operation_desc.rank_k_kind));

  set_argument(result, "A", problem_space,
    std::string(library::to_string(operation_desc.A.element)) + ":" + library::to_string(operation_desc.A.layout));

  set_argument(result, "B", problem_space,
    std::string(library::to_string(operation_desc.B.element)) + ":" + library::to_string(operation_desc.B.layout));

  set_argument(result, "C", problem_space,
    std::string(library::to_string(operation_desc.C.element)) + ":" + library::to_string(operation_desc.C.layout));

  set_argument(result, "fill_mode", problem_space, library::to_string(operation_desc.fill_mode));

  set_argument(result, "blas_mode", problem_space, library::to_string(operation_desc.blas_mode));

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

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Extracts the problem dimensions
Status Rank2KOperationProfiler::initialize_configuration(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::RankKDescription const &operation_desc =
    static_cast<library::RankKDescription const &>(operation->description());

  if (operation_desc.rank_k_kind != library::RankKKind::kUniversal) {
    return Status::kErrorInvalidProblem;
  }

  Status status = problem_.parse(operation_desc, problem_space, problem);

  if (status != Status::kSuccess) {
    return status;
  }

  rank_k_workspace_.configuration.problem_size.m() = int(problem_.n);
  rank_k_workspace_.configuration.problem_size.n() = int(problem_.n);
  rank_k_workspace_.configuration.problem_size.k() = int(problem_.k);
  rank_k_workspace_.configuration.lda = problem_.lda;
  rank_k_workspace_.configuration.ldb = problem_.ldb;
  rank_k_workspace_.configuration.ldc = problem_.ldc;
  rank_k_workspace_.configuration.ldd = problem_.ldc;
  //rank_k_workspace_.configuration.split_k_slices = int(problem_.split_k_slices);
  rank_k_workspace_.configuration.batch_count = int(problem_.split_k_slices);

  rank_k_workspace_.arguments.A = nullptr;
  rank_k_workspace_.arguments.B = nullptr;
  rank_k_workspace_.arguments.C = nullptr;
  rank_k_workspace_.arguments.D = nullptr;
  rank_k_workspace_.arguments.alpha = problem_.alpha.data();
  rank_k_workspace_.arguments.beta = problem_.beta.data();
  rank_k_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  initialize_result_(this->model_result_, options, operation_desc, problem_space);

  return operation->can_implement(&rank_k_workspace_.configuration, &rank_k_workspace_.arguments);
}

/// Initializes the performance result
void Rank2KOperationProfiler::initialize_result_(
  PerformanceResult &result,
  Options const &options,
  library::RankKDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  result.provider = library::Provider::kCUTLASS;
  result.disposition = Disposition::kNotRun;
  result.status = Status::kSuccess;
  result.operation_name = operation_desc.name;

  problem_.initialize_result(result, operation_desc, problem_space);

  OperationProfiler::initialize_result_(result, operation_desc, problem_space);


  result.bytes = problem_.bytes(operation_desc);
  result.flops = problem_.flops(operation_desc);
  result.runtime = 0;


}

/// Initializes workspace
Status Rank2KOperationProfiler::initialize_workspace(
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

  library::RankKDescription const &operation_desc =
    static_cast<library::RankKDescription const &>(operation->description());

  if (options.execution_mode != ExecutionMode::kDryRun) {
    int seed_shift = 0;
    rank_k_workspace_.A = device_context.allocate_and_initialize_tensor(
      options,
      "A",
      operation_desc.A.element,
      operation_desc.A.layout,
      {int(problem_.n), int(problem_.k)},
      {int(problem_.lda)},
      1, // batch_count
      seed_shift++,
      0 // device_index
    );

    rank_k_workspace_.B = device_context.allocate_and_initialize_tensor(
      options,
      "B",
      operation_desc.B.element,
      operation_desc.B.layout,
      {int(problem_.n), int(problem_.k)},
      {int(problem_.ldb)},
      1, // batch_count
      seed_shift++,
      0 // device_index
    );

    rank_k_workspace_.C = device_context.allocate_and_initialize_tensor(
      options,
      "C",
      operation_desc.C.element,
      operation_desc.C.layout,
      {int(problem_.n), int(problem_.n)},
      {int(problem_.ldc)},
      1, // batch_count
      seed_shift++,
      0 // device_index
    );

    rank_k_workspace_.Computed = device_context.allocate_tensor(
      options,
      "D",
      operation_desc.C.element,
      operation_desc.C.layout,
      {int(problem_.n), int(problem_.n)},
      {int(problem_.ldc)},
      1, // batch_count
      0 // device_index
    );

    rank_k_workspace_.Reference = device_context.allocate_tensor(
      options,
      "Reference",
      operation_desc.C.element,
      operation_desc.C.layout,
      {int(problem_.n), int(problem_.n)},
      {int(problem_.ldc)},
      1, // batch_count
      0 // device_index
    );

    rank_k_workspace_.Computed->copy_from_device(rank_k_workspace_.C->data());
    rank_k_workspace_.Reference->copy_from_device(rank_k_workspace_.C->data());
  }


  //
  // Initialize the CUTLASS operation
  //
  Status status = Status::kSuccess;

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    if (options.execution_mode != ExecutionMode::kDryRun) {

      uint64_t workspace_size = operation->get_host_workspace_size(&rank_k_workspace_.configuration);
      rank_k_workspace_.host_workspace.resize(workspace_size, 0);

      workspace_size = operation->get_device_workspace_size(&rank_k_workspace_.configuration);
      rank_k_workspace_.device_workspace.reset(library::NumericTypeID::kU8, workspace_size);

      status = operation->initialize(
        &rank_k_workspace_.configuration,
        rank_k_workspace_.host_workspace.data(),
        rank_k_workspace_.device_workspace.data());
    }

    //
    // If CUTLASS is enabled, generate a result for it
    //
    results_.push_back(model_result_);
    results_.back().provider = library::Provider::kCUTLASS;
    results_.back().op_kind = library::OperationKind::kRank2K;
    results_.back().disposition = Disposition::kNotRun;

    for(auto provider : verification_providers_) {
      results_.back().verification_map[provider] = Disposition::kNotRun;
    }
  }

  return status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Verifies CUTLASS against references
bool Rank2KOperationProfiler::verify_cutlass(
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

  // Initialize structure containing RankK arguments
  rank_k_workspace_.arguments.A = rank_k_workspace_.A->data();
  rank_k_workspace_.arguments.B = rank_k_workspace_.B->data();
  rank_k_workspace_.arguments.C = rank_k_workspace_.C->data();
  rank_k_workspace_.arguments.D = rank_k_workspace_.Computed->data();
  rank_k_workspace_.arguments.alpha = problem_.alpha.data();
  rank_k_workspace_.arguments.beta = problem_.beta.data();
  rank_k_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  //
  // Run the CUTLASS operation
  //

  results_.back().status = operation->run(
    &rank_k_workspace_.arguments,
    rank_k_workspace_.host_workspace.data(),
    rank_k_workspace_.device_workspace.data());

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

#if CUTLASS_ENABLE_CUBLAS
    if (options.verification.provider_enabled(library::Provider::kCUBLAS)) {

      // Guard against unsupported cases
      auto const & rank_k_desc = static_cast<library::RankKDescription const &>(operation->description());

      if (cublas_satisfies(rank_k_desc) == Status::kSuccess) {

        // call cublas verification if supported
        verify_with_cublas_(
          options,
          report,
          device_context,
          operation,
          problem_space,
          problem);
        }

      else {
        // set verification map for cublas to not supported
        results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kNotSupported;
      }
    }
#endif // #if CUTLASS_ENABLE_CUBLAS

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

/// Verifies CUTLASS against references
bool Rank2KOperationProfiler::verify_with_cublas_(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {


#if CUTLASS_ENABLE_CUBLAS

  library::RankKDescription const &rank_k_desc =
    static_cast<library::RankKDescription const &>(operation->description());

  //
  // Construct cuBLAS operators
  //

  CublasCreate handle;
  cublasStatus_t status = handle.get_cublas_create_status();

  if (status != CUBLAS_STATUS_SUCCESS) {

    results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kFailed;
    return true;
  }

  //
  // Initialize state
  //

  try {

    //
    // Construct dispatcher to cublas<t>Syr2k()
    //

    // Initialize structure containing RankK arguments
    rank_k_workspace_.arguments.A = rank_k_workspace_.A->data();
    rank_k_workspace_.arguments.B = rank_k_workspace_.B->data();
    rank_k_workspace_.arguments.C = rank_k_workspace_.Reference->data();
    rank_k_workspace_.arguments.D = rank_k_workspace_.Reference->data();
    rank_k_workspace_.arguments.alpha = problem_.alpha.data();
    rank_k_workspace_.arguments.beta = problem_.beta.data();
    rank_k_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

    detail::cublasRankKDispatcher rank_k_op(
      rank_k_desc,
      rank_k_workspace_.configuration,
      rank_k_workspace_.arguments
    );

    if (rank_k_op.status != Status::kSuccess) {
      results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kNotRun;
      return true;
    }

    results_.back().status = Status::kSuccess;

    status = rank_k_op(handle);

    // Handle errors
    if (status != CUBLAS_STATUS_SUCCESS) {

      results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kFailed;
      return true;
    }

    //
    // Verify results
    //

    results_.back().verification_map[library::Provider::kCUBLAS] = compare_tensors(
      options,
      *rank_k_workspace_.Computed,
      *rank_k_workspace_.Reference
    );

    // Save workspace if incorrect
    if (options.verification.save_workspace == SaveWorkspace::kIncorrect &&
      results_.back().verification_map[library::Provider::kCUBLAS] == Disposition::kIncorrect) {

      save_workspace(
        device_context,
        options,
        rank_k_desc,
        library::Provider::kCUTLASS,
        library::Provider::kCUBLAS);
    }
  }
  catch (...) {
    results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kFailed;
  }

#endif

  // Return true means continue profiling
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Measures performance results
bool Rank2KOperationProfiler::profile(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    // Initialize structure containing RankK arguments
    rank_k_workspace_.arguments.A = rank_k_workspace_.A->data();
    rank_k_workspace_.arguments.B = rank_k_workspace_.B->data();
    rank_k_workspace_.arguments.C = rank_k_workspace_.C->data();
    rank_k_workspace_.arguments.D = rank_k_workspace_.Computed->data();
    rank_k_workspace_.arguments.alpha = problem_.alpha.data();
    rank_k_workspace_.arguments.beta = problem_.beta.data();
    rank_k_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

    results_.back().status = profile_cutlass_(
      results_.back(),
      options,
      operation,
      &rank_k_workspace_.arguments,
      rank_k_workspace_.host_workspace.data(),
      rank_k_workspace_.device_workspace.data()
    );
  }
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
