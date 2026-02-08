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
#include "cutlass/profiler/trmm_operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Ctor
TrmmOperationProfiler::TrmmOperationProfiler(Options const &options):
  OperationProfiler(
    options,
    library::OperationKind::kTrmm,
    {
      {ArgumentTypeID::kEnumerated, {"trmm_kind"}, "Variant of TRMM (universal)"},
      {ArgumentTypeID::kInteger, {"m", "problem-size::m"}, "M dimension of the TRMM problem space"},
      {ArgumentTypeID::kInteger, {"n", "problem-size::n"}, "N dimension of the TRMM problem space"},
      {ArgumentTypeID::kTensor, {"A"}, "Tensor storing the A operand"},
      {ArgumentTypeID::kEnumerated, {"side_mode"}, "Side Mode for TRMM (left, right)"},
      {ArgumentTypeID::kEnumerated, {"fill_mode"}, "Fill Mode for TRMM (lower, upper)"},
      {ArgumentTypeID::kEnumerated, {"diag_type"}, "Diag Type for TRMM (nonunit, unit)"},
      {ArgumentTypeID::kTensor, {"B"}, "Tensor storing the B operand"},
      {ArgumentTypeID::kTensor, {"D"}, "Tensor storing the D operand"},
      {ArgumentTypeID::kScalar, {"alpha", "epilogue::alpha"}, "Epilogue scalar alpha"},
      {ArgumentTypeID::kScalar, {"beta", "epilogue::beta"}, "Epilogue scalar beta"},
      {ArgumentTypeID::kInteger, {"split_k_slices", "split-k-slices"}, "Number of partitions of K dimension"},
      {ArgumentTypeID::kInteger, {"batch_count", "batch-count"}, "Number of TRMMs computed in one batch"},
    },
    { library::Provider::kCUBLAS}
  ) {
  description_ = "      Triangular Matrix-Multiplication. D = alpha * A * B or alpha * B * A";
}

/// Destructor
TrmmOperationProfiler::~TrmmOperationProfiler() {

}

/// Prints usage statement for the math function
void TrmmOperationProfiler::print_usage(std::ostream &out) const {
  out << "TRMM" << "\n\n";

  OperationProfiler::print_usage(out);
}

/// Prints examples
void TrmmOperationProfiler::print_examples(std::ostream &out) const {

  out << "\nExamples:\n\n"
    << "Profile a particular problem size:\n"
    << "  $ cutlass_profiler --operation=Trmm --n=1024 --m=128\n\n"

    << "Schmoo over problem size and beta:\n"
    << "  $ cutlass_profiler --operation=Trmm --n=1024:4096:256 --m=128:8192:128 --beta=0,1,2.5\n\n"

    << "Schmoo over accumulator types:\n"
    << "  $ cutlass_profiler --operation=Trmm --accumulator-type=f16,f32\n\n"

    << "Run when A is f16 with column-major or A is any datatype with row-major (For column major, use column, col, or n. For row major use, row or t):\n"
    << "  $ cutlass_profiler --operation=Trmm --A=f16:column or --A=*:row\n\n"

    << "Using various input value distribution:\n"
    << "  $ cutlass_profiler --operation=Trmm --dist=uniform,min:0,max:3\n"
    << "  $ cutlass_profiler --operation=Trmm --dist=gaussian,mean:0,stddev:3\n"
    << "  $ cutlass_profiler --operation=Trmm --dist=sequential,start:0,delta:1\n\n"

    << "Run a kernel with cta tile size of 256x128x32 and save workspace if results are incorrect (note that --cta-tile::k=32 is default cta-tile size):\n"
    << " $ cutlass_profiler --operation=Trmm --cta_m=256 --cta_n=128  --cta_k=32 --save-workspace=incorrect\n\n"

    << "Test your changes to trmm kernels with a quick functional test and save results in functional-test.csv:\n"
    << " $ cutlass_profiler  --operation=Trmm \\ \n"
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

Status TrmmOperationProfiler::TrmmProblem::parse(
  library::TrmmDescription const &operation_desc,
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

  if (!tensor_description_satisfies(operation_desc.D, "D", problem_space, problem)) {
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

  if (operation_desc.side_mode == SideMode::kLeft) {
    this->lda = DeviceAllocation::get_packed_layout(
      operation_desc.A.layout, {int(this->m), int(this->m)}).front();
  }
  else if (operation_desc.side_mode == SideMode::kRight) {
    this->lda = DeviceAllocation::get_packed_layout(
      operation_desc.A.layout, {int(this->n), int(this->n)}).front();
  }

  this->ldb = DeviceAllocation::get_packed_layout(
    operation_desc.B.layout, {int(this->m), int(this->n)}).front();

  this->ldd = DeviceAllocation::get_packed_layout(
    operation_desc.D.layout, {int(this->m), int(this->n)}).front();

  return Status::kSuccess;
}

/// Initializes a performance result
void TrmmOperationProfiler::TrmmProblem::initialize_result(
  PerformanceResult &result,
  library::TrmmDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  result.arguments.resize(problem_space.rank());

  set_argument(result, "trmm_kind", problem_space, library::to_string(operation_desc.trmm_kind));

  set_argument(result, "A", problem_space,
    std::string(library::to_string(operation_desc.A.element)) + ":" + library::to_string(operation_desc.A.layout));

  set_argument(result, "side_mode", problem_space, library::to_string(operation_desc.side_mode));

  set_argument(result, "fill_mode", problem_space, library::to_string(operation_desc.fill_mode));

  set_argument(result, "diag_type", problem_space, library::to_string(operation_desc.diag_type));

  set_argument(result, "B", problem_space,
    std::string(library::to_string(operation_desc.B.element)) + ":" + library::to_string(operation_desc.B.layout));

  set_argument(result, "D", problem_space,
    std::string(library::to_string(operation_desc.D.element)) + ":" + library::to_string(operation_desc.D.layout));

  set_argument(result, "m", problem_space, m);
  set_argument(result, "n", problem_space, n);

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
Status TrmmOperationProfiler::initialize_configuration(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::TrmmDescription const &operation_desc =
    static_cast<library::TrmmDescription const &>(operation->description());

  if (operation_desc.trmm_kind != library::TrmmKind::kUniversal) {
    return Status::kErrorInvalidProblem;
  }

  Status status = problem_.parse(operation_desc, problem_space, problem);

  if (status != Status::kSuccess) {
    return status;
  }

  trmm_workspace_.configuration.problem_size.m() = int(problem_.m);
  trmm_workspace_.configuration.problem_size.n() = int(problem_.n);
  trmm_workspace_.configuration.problem_size.k() = (operation_desc.side_mode == SideMode::kLeft)
                                                    ? int(problem_.m) : int(problem_.n);
  trmm_workspace_.configuration.lda = problem_.lda;
  trmm_workspace_.configuration.ldb = problem_.ldb;
  trmm_workspace_.configuration.ldd = problem_.ldd;
  //trmm_workspace_.configuration.split_k_slices = int(problem_.split_k_slices);
  trmm_workspace_.configuration.batch_count = int(problem_.split_k_slices);

  trmm_workspace_.arguments.A = nullptr;
  trmm_workspace_.arguments.B = nullptr;
  trmm_workspace_.arguments.D = nullptr;
  trmm_workspace_.arguments.alpha = problem_.alpha.data();
  trmm_workspace_.arguments.beta = problem_.beta.data();
  trmm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  initialize_result_(this->model_result_, options, operation_desc, problem_space);

  return operation->can_implement(&trmm_workspace_.configuration, &trmm_workspace_.arguments);
}

/// Initializes the performance result
void TrmmOperationProfiler::initialize_result_(
  PerformanceResult &result,
  Options const &options,
  library::TrmmDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  result.provider = library::Provider::kCUTLASS;
  result.disposition = Disposition::kNotRun;
  result.status = Status::kSuccess;
  result.operation_name = operation_desc.name;

  problem_.initialize_result(result, operation_desc, problem_space);

  OperationProfiler::initialize_result_(result, operation_desc, problem_space);

  if (operation_desc.side_mode == SideMode::kLeft) {
    // Input bytes read and Output bytes written for the trmm problem
    result.bytes =
      // Half matrix including the diagonal will have (M*(M+1))/2 elements
      int64_t(library::sizeof_bits(operation_desc.A.element) * problem_.m / 8) * (problem_.m + 1) / 2 +
      int64_t(library::sizeof_bits(operation_desc.B.element) * problem_.m / 8) * problem_.n +
      int64_t(library::sizeof_bits(operation_desc.D.element) * problem_.m / 8) * problem_.n;
  } else if (operation_desc.side_mode == SideMode::kRight) {
    // Input bytes read and Output bytes written for the trmm problem
    result.bytes =
      // Half matrix including the diagonal will have (N*(N+1))/2 elements
      int64_t(library::sizeof_bits(operation_desc.A.element) * problem_.n / 8) * (problem_.n + 1) / 2 +
      int64_t(library::sizeof_bits(operation_desc.B.element) * problem_.m / 8) * problem_.n +
      int64_t(library::sizeof_bits(operation_desc.D.element) * problem_.m / 8) * problem_.n;
  }

  // FLOPs = 2 * [ ( M * (M+1)/2 * N ) ] // Beta is zero
  result.flops = problem_.m * (problem_.m + 1) * problem_.n;

   result.runtime = 0;

  // complex-valued support
  switch (operation_desc.tile_description.math_instruction.math_operation) {
  case library::MathOperationID::kMultiplyAddComplex:
    result.flops *= 4;
    break;

  case library::MathOperationID::kMultiplyAddComplexFastF32:
    result.flops *= 4;
    break;

  default: break;
  }

}

/// Initializes workspace
Status TrmmOperationProfiler::initialize_workspace(
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

  library::TrmmDescription const &operation_desc =
    static_cast<library::TrmmDescription const &>(operation->description());

  if (options.execution_mode != ExecutionMode::kDryRun) {
    int seed_shift = 0;
    if (operation_desc.side_mode == SideMode::kLeft) {
      trmm_workspace_.A = device_context.allocate_and_initialize_tensor(
        options,
        "A",
        operation_desc.A.element,
        operation_desc.A.layout,
        {int(problem_.m), int(problem_.m)},
        {int(problem_.lda)},
        1, // batch_count
        seed_shift++,
        0 // device_index
      );
    } else if (operation_desc.side_mode == SideMode::kRight) {
      trmm_workspace_.A = device_context.allocate_and_initialize_tensor(
        options,
        "A",
        operation_desc.A.element,
        operation_desc.A.layout,
        {int(problem_.n), int(problem_.n)},
        {int(problem_.lda)},
        1, // batch_count
        seed_shift++,
        0 // device_index
      );
    }

    trmm_workspace_.B = device_context.allocate_and_initialize_tensor(
      options,
      "B",
      operation_desc.B.element,
      operation_desc.B.layout,
      {int(problem_.m), int(problem_.n)},
      {int(problem_.ldb)},
      1, // batch_count
      seed_shift++,
      0 // device_index
    );

    trmm_workspace_.Computed = device_context.allocate_tensor(
      options,
      "D",
      operation_desc.D.element,
      operation_desc.D.layout,
      {int(problem_.m), int(problem_.n)},
      {int(problem_.ldd)},
      1, // batch_count
      0 // device_index
    );

    trmm_workspace_.Reference = device_context.allocate_tensor(
      options,
      "Reference",
      operation_desc.D.element,
      operation_desc.D.layout,
      {int(problem_.m), int(problem_.n)},
      {int(problem_.ldd)},
      1, // batch_count
      0 // device_index
    );

  }

  //
  // Initialize the CUTLASS operation
  //
  Status status = Status::kSuccess;

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    if (options.execution_mode != ExecutionMode::kDryRun) {

      uint64_t workspace_size = operation->get_host_workspace_size(&trmm_workspace_.configuration);
      trmm_workspace_.host_workspace.resize(workspace_size, 0);

      workspace_size = operation->get_device_workspace_size(&trmm_workspace_.configuration);
      trmm_workspace_.device_workspace.reset(library::NumericTypeID::kU8, workspace_size);

      status = operation->initialize(
        &trmm_workspace_.configuration,
        trmm_workspace_.host_workspace.data(),
        trmm_workspace_.device_workspace.data());
    }

    //
    // If CUTLASS is enabled, generate a result for it
    //
    results_.push_back(model_result_);
    results_.back().provider = library::Provider::kCUTLASS;
    results_.back().op_kind = library::OperationKind::kTrmm;
    results_.back().disposition = Disposition::kNotRun;

    for(auto provider : verification_providers_) {
      results_.back().verification_map[provider] = Disposition::kNotRun;
    }
  }

  return status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Verifies CUTLASS against references
bool TrmmOperationProfiler::verify_cutlass(
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

  // Initialize structure containing TRMM arguments
  trmm_workspace_.arguments.A = trmm_workspace_.A->data();
  trmm_workspace_.arguments.B = trmm_workspace_.B->data();
  trmm_workspace_.arguments.D = trmm_workspace_.Computed->data();
  trmm_workspace_.arguments.alpha = problem_.alpha.data();
  trmm_workspace_.arguments.beta = problem_.beta.data();
  trmm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  //
  // Run the CUTLASS operation
  //

  results_.back().status = operation->run(
    &trmm_workspace_.arguments,
    trmm_workspace_.host_workspace.data(),
    trmm_workspace_.device_workspace.data());

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
      auto const & trmm_desc = static_cast<library::TrmmDescription const &>(operation->description());

      if (cublas_satisfies(trmm_desc) == Status::kSuccess) {

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
bool TrmmOperationProfiler::verify_with_cublas_(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {


#if CUTLASS_ENABLE_CUBLAS

  library::TrmmDescription const &trmm_desc =
    static_cast<library::TrmmDescription const &>(operation->description());

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
    // Construct dispatcher to cublas<t>Trmm()
    //

    // Initialize structure containing TRMM arguments
    trmm_workspace_.arguments.A = trmm_workspace_.A->data();
    trmm_workspace_.arguments.B = trmm_workspace_.B->data();
    trmm_workspace_.arguments.D = trmm_workspace_.Reference->data();
    trmm_workspace_.arguments.alpha = problem_.alpha.data();
    trmm_workspace_.arguments.beta = problem_.beta.data();
    trmm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

    detail::cublasTrmmDispatcher trmm_op(
      trmm_desc,
      trmm_workspace_.configuration,
      trmm_workspace_.arguments
    );

    if (trmm_op.status != Status::kSuccess) {
      results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kNotRun;
      return true;
    }

    results_.back().status = Status::kSuccess;

    status = trmm_op(handle);

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
      *trmm_workspace_.Computed,
      *trmm_workspace_.Reference
    );

    // Save workspace if incorrect
    if (options.verification.save_workspace == SaveWorkspace::kIncorrect &&
      results_.back().verification_map[library::Provider::kCUBLAS] == Disposition::kIncorrect) {

      save_workspace(
        device_context,
        options,
        trmm_desc,
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
bool TrmmOperationProfiler::profile(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    // Initialize structure containing TRMM arguments
    trmm_workspace_.arguments.A = trmm_workspace_.A->data();
    trmm_workspace_.arguments.B = trmm_workspace_.B->data();
    trmm_workspace_.arguments.D = trmm_workspace_.Computed->data();
    trmm_workspace_.arguments.alpha = problem_.alpha.data();
    trmm_workspace_.arguments.beta = problem_.beta.data();
    trmm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

    results_.back().status = profile_cutlass_(
      results_.back(),
      options,
      operation,
      &trmm_workspace_.arguments,
      trmm_workspace_.host_workspace.data(),
      trmm_workspace_.device_workspace.data()
    );
  }
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
