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
#include "cutlass/profiler/symm_operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Ctor
SymmOperationProfiler::SymmOperationProfiler(Options const &options):
  OperationProfiler(
    options,
    library::OperationKind::kSymm,
    {
      {ArgumentTypeID::kEnumerated, {"symm_kind"}, "Variant of Symm (universal)"},
      {ArgumentTypeID::kInteger, {"m", "problem-size::m"}, "M dimension of the Symm problem space"},
      {ArgumentTypeID::kInteger, {"n", "problem-size::n"}, "N dimension of the Symm problem space"},
      {ArgumentTypeID::kTensor, {"A"}, "Tensor storing the A operand"},
      {ArgumentTypeID::kTensor, {"B"}, "Tensor storing the B operand"},
      {ArgumentTypeID::kTensor, {"C"}, "Tensor storing the C operand"},
      {ArgumentTypeID::kEnumerated, {"side_mode"}, "Side Mode for Symm kernel (left or right)"},
      {ArgumentTypeID::kEnumerated, {"fill_mode"}, "Fill Mode for Symm kernel (lower or upper)"},
      {ArgumentTypeID::kEnumerated, {"blas_mode"}, "Blas Mode for Symm kernel (symmetric or hermitian)"},
      {ArgumentTypeID::kScalar, {"alpha", "epilogue::alpha"}, "Epilogue scalar alpha"},
      {ArgumentTypeID::kScalar, {"beta", "epilogue::beta"}, "Epilogue scalar beta"},
      {ArgumentTypeID::kInteger, {"split_k_slices", "split-k-slices"}, "Number of partitions of K dimension"},
      {ArgumentTypeID::kInteger, {"batch_count", "batch-count"}, "Number of Symm computed in one batch"},
    },
    { library::Provider::kCUBLAS }
  ) {
  description_ = "      Symmetric Matrix-Matrix Multiplication. D = alpha * A * B OR alpha * B * A + beta * C (where A is symmetric/hermitian)";
}

/// Destructor
SymmOperationProfiler::~SymmOperationProfiler() {

}

/// Prints usage statement for the math function
void SymmOperationProfiler::print_usage(std::ostream &out) const {
  out << "Symm" << "\n\n";

  OperationProfiler::print_usage(out);
}

/// Prints examples
void SymmOperationProfiler::print_examples(std::ostream &out) const {

  out << "\nExamples:\n\n"
    << "Profile a particular problem size SYMM kernel:\n"
    << "  $ cutlass_profiler --operation=Symm --blas_mode=symmetric --m=1024 --n=128\n\n"

    << "Profile a particular problem size HEMM kernel:\n"
    << "  $ cutlass_profiler --operation=Symm --blas_mode=hermitian --m=1024 --n=128\n\n"

    << "Schmoo over problem size and beta:\n"
    << "  $ cutlass_profiler --operation=Symm --m=1024:4096:256 --n=128:8192:128 --beta=0,1,2.5\n\n"

    << "Schmoo over accumulator types:\n"
    << "  $ cutlass_profiler --operation=Symm --accumulator-type=f16,f32\n\n"

    << "Schmoo over side modees:\n"
    << "  $ cutlass_profiler --operation=Symm --side_mode=left/right\n\n"

    << "Schmoo over fill modees:\n"
    << "  $ cutlass_profiler --operation=Symm --fill_mode=lower/upper\n\n"

    << "Run when A is f16 with column-major or A is any datatype with row-major (For column major, use column, col, or n. For row major use, row or t):\n"
    << "  $ cutlass_profiler --operation=Symm --A=f16:column or --A=*:row\n\n"

    << "Using various input value distribution:\n"
    << "  $ cutlass_profiler --operation=Symm --dist=uniform,min:0,max:3\n"
    << "  $ cutlass_profiler --operation=Symm --dist=gaussian,mean:0,stddev:3\n"
    << "  $ cutlass_profiler --operation=Symm --dist=sequential,start:0,delta:1\n\n"

    << "Run a kernel with cta tile size of 256x128x32 and save workspace if results are incorrect (note that --cta-tile::k=32 is default cta-tile size):\n"
    << " $ cutlass_profiler --operation=Symm --cta_m=256 --cta_n=128  --cta_k=32 --save-workspace=incorrect\n\n"

    << "Test your changes to symm kernels with a quick functional test and save results in functional-test.csv:\n"
    << " $ cutlass_profiler  --operation=Symm \\ \n"
    << "   --m=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \\ \n"
    << "   --n=8,16,32,64,128,256,288,384,504,512,520 \\ \n"
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

Status SymmOperationProfiler::SymmProblem::parse(
  library::SymmDescription const &operation_desc,
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

  this->ldc = DeviceAllocation::get_packed_layout(
    operation_desc.C.layout, {int(this->m), int(this->n)}).front();

  return Status::kSuccess;
}

/// Total number of bytes loaded
int64_t SymmOperationProfiler::SymmProblem::bytes(library::SymmDescription const &operation_desc) const {
  int64_t bytes = 0;
  // Input bytes read and Output bytes written for the gemm problem
  // Half matrix including the diagonal will have (X*(X+1))/2 elements
  if (operation_desc.side_mode == SideMode::kLeft) {
    bytes =
      int64_t(library::sizeof_bits(operation_desc.A.element) * m / 8) * (m + 1) / 2 +
      int64_t(library::sizeof_bits(operation_desc.B.element) * m / 8) * n +
      int64_t(library::sizeof_bits(operation_desc.C.element) * m / 8) * n;
  } else if (operation_desc.side_mode == SideMode::kRight) {
    bytes =
      int64_t(library::sizeof_bits(operation_desc.A.element) * n / 8) * (n + 1) / 2 +
      int64_t(library::sizeof_bits(operation_desc.B.element) * m / 8) * n +
      int64_t(library::sizeof_bits(operation_desc.C.element) * m / 8) * n;
  }
  // Set is_beta_zero true if beta is zero
  bool is_beta_zero = std::all_of(beta.begin(), beta.end(), [](uint8_t i) { return i==0; });

  // Output bytes read for the gemm problem for non-zero beta values
  if (!is_beta_zero) {
    bytes += int64_t(library::sizeof_bits(operation_desc.C.element) * m / 8) * n;
  }

  bytes *= batch_count;

  return bytes;
}

/// Total number of flops computed
int64_t SymmOperationProfiler::SymmProblem::flops(library::SymmDescription const &operation_desc) const {

  // FLOPs for first TRMM kernel (with diagonal) = 2 * [ ( M * (M+1)/2 * N ) ] // Beta is zero
  // FLOPs for second TRMM kernel (with diagonal) = 2 * [ ( M * (M-1)/2 * N ) ] // Beta is zero
  // FLOPs = m*(m+1)*n [mma1] + m*(m-1)*n [mma2] + 2*m*n [epilogue]
  // FLOPs = 2*m*n(m+1) for left side mode
  // FLOPs can also be calculated to be same as GEMM with correct value for 'k' as below.
  int64_t k = (operation_desc.side_mode == SideMode::kLeft) ? int64_t(m) : int64_t(n);
  int64_t flops_ = (int64_t(m) * n * k + m * n) * 2;

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
void SymmOperationProfiler::SymmProblem::initialize_result(
  PerformanceResult &result,
  library::SymmDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  result.arguments.resize(problem_space.rank());

  set_argument(result, "symm_kind", problem_space, library::to_string(operation_desc.symm_kind));

  set_argument(result, "A", problem_space,
    std::string(library::to_string(operation_desc.A.element)) + ":" + library::to_string(operation_desc.A.layout));

  set_argument(result, "B", problem_space,
    std::string(library::to_string(operation_desc.B.element)) + ":" + library::to_string(operation_desc.B.layout));

  set_argument(result, "C", problem_space,
    std::string(library::to_string(operation_desc.C.element)) + ":" + library::to_string(operation_desc.C.layout));

  set_argument(result, "side_mode", problem_space, library::to_string(operation_desc.side_mode));

  set_argument(result, "fill_mode", problem_space, library::to_string(operation_desc.fill_mode));

  set_argument(result, "blas_mode", problem_space, library::to_string(operation_desc.blas_mode));

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
Status SymmOperationProfiler::initialize_configuration(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::SymmDescription const &operation_desc =
    static_cast<library::SymmDescription const &>(operation->description());

  if (operation_desc.symm_kind != library::SymmKind::kUniversal) {
    return Status::kErrorInvalidProblem;
  }

  Status status = problem_.parse(operation_desc, problem_space, problem);

  if (status != Status::kSuccess) {
    return status;
  }

  symm_workspace_.configuration.problem_size.m() = int(problem_.m);
  symm_workspace_.configuration.problem_size.n() = int(problem_.n);
  symm_workspace_.configuration.problem_size.k() = (operation_desc.side_mode == SideMode::kLeft)
                                                    ? int(problem_.m) : int(problem_.n);
  symm_workspace_.configuration.lda = problem_.lda;
  symm_workspace_.configuration.ldb = problem_.ldb;
  symm_workspace_.configuration.ldc = problem_.ldc;
  symm_workspace_.configuration.ldd = problem_.ldc;
  //symm_workspace_.configuration.split_k_slices = int(problem_.split_k_slices);
  symm_workspace_.configuration.batch_count = int(problem_.split_k_slices);

  symm_workspace_.arguments.A = nullptr;
  symm_workspace_.arguments.B = nullptr;
  symm_workspace_.arguments.C = nullptr;
  symm_workspace_.arguments.D = nullptr;
  symm_workspace_.arguments.alpha = problem_.alpha.data();
  symm_workspace_.arguments.beta = problem_.beta.data();
  symm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  initialize_result_(this->model_result_, options, operation_desc, problem_space);

  return operation->can_implement(&symm_workspace_.configuration, &symm_workspace_.arguments);
}

/// Initializes the performance result
void SymmOperationProfiler::initialize_result_(
  PerformanceResult &result,
  Options const &options,
  library::SymmDescription const &operation_desc,
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
Status SymmOperationProfiler::initialize_workspace(
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

  library::SymmDescription const &operation_desc =
    static_cast<library::SymmDescription const &>(operation->description());

  if (options.execution_mode != ExecutionMode::kDryRun) {
    int seed_shift = 0;
    if (operation_desc.side_mode == SideMode::kLeft) {
      symm_workspace_.A = device_context.allocate_and_initialize_tensor(
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
      symm_workspace_.A = device_context.allocate_and_initialize_tensor(
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

    symm_workspace_.B = device_context.allocate_and_initialize_tensor(
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

    symm_workspace_.C = device_context.allocate_and_initialize_tensor(
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

    symm_workspace_.Computed = device_context.allocate_tensor(
      options,
      "D",
      operation_desc.C.element,
      operation_desc.C.layout,
      {int(problem_.m), int(problem_.n)},
      {int(problem_.ldc)},
      1, // batch_count
      0 // device_index
    );

    symm_workspace_.Reference = device_context.allocate_tensor(
      options,
      "Reference",
      operation_desc.C.element,
      operation_desc.C.layout,
      {int(problem_.m), int(problem_.n)},
      {int(problem_.ldc)},
      1, // batch_count
      0 // device_index
    );

    symm_workspace_.Computed->copy_from_device(symm_workspace_.C->data());
    symm_workspace_.Reference->copy_from_device(symm_workspace_.C->data());
  }


  //
  // Initialize the CUTLASS operation
  //
  Status status = Status::kSuccess;

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    if (options.execution_mode != ExecutionMode::kDryRun) {

      uint64_t workspace_size = operation->get_host_workspace_size(&symm_workspace_.configuration);
      symm_workspace_.host_workspace.resize(workspace_size, 0);

      workspace_size = operation->get_device_workspace_size(&symm_workspace_.configuration);
      symm_workspace_.device_workspace.reset(library::NumericTypeID::kU8, workspace_size);

      status = operation->initialize(
        &symm_workspace_.configuration,
        symm_workspace_.host_workspace.data(),
        symm_workspace_.device_workspace.data());
    }

    //
    // If CUTLASS is enabled, generate a result for it
    //
    results_.push_back(model_result_);
    results_.back().provider = library::Provider::kCUTLASS;
    results_.back().op_kind = library::OperationKind::kSymm;
    results_.back().disposition = Disposition::kNotRun;

    for(auto provider : verification_providers_) {
      results_.back().verification_map[provider] = Disposition::kNotRun;
    }
  }

  return status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Verifies CUTLASS against references
bool SymmOperationProfiler::verify_cutlass(
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

  // Initialize structure containing Symm arguments
  symm_workspace_.arguments.A = symm_workspace_.A->data();
  symm_workspace_.arguments.B = symm_workspace_.B->data();
  symm_workspace_.arguments.C = symm_workspace_.C->data();
  symm_workspace_.arguments.D = symm_workspace_.Computed->data();
  symm_workspace_.arguments.alpha = problem_.alpha.data();
  symm_workspace_.arguments.beta = problem_.beta.data();
  symm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  //
  // Run the CUTLASS operation
  //

  results_.back().status = operation->run(
    &symm_workspace_.arguments,
    symm_workspace_.host_workspace.data(),
    symm_workspace_.device_workspace.data());

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
      auto const & symm_desc = static_cast<library::SymmDescription const &>(operation->description());

      if (cublas_satisfies(symm_desc) == Status::kSuccess) {

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
bool SymmOperationProfiler::verify_with_cublas_(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {


#if CUTLASS_ENABLE_CUBLAS

  library::SymmDescription const &symm_desc =
    static_cast<library::SymmDescription const &>(operation->description());

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
    // Construct dispatcher to cublas<t>Symm()
    //

    // Initialize structure containing Symm arguments
    symm_workspace_.arguments.A = symm_workspace_.A->data();
    symm_workspace_.arguments.B = symm_workspace_.B->data();
    symm_workspace_.arguments.C = symm_workspace_.Reference->data();
    symm_workspace_.arguments.D = symm_workspace_.Reference->data();
    symm_workspace_.arguments.alpha = problem_.alpha.data();
    symm_workspace_.arguments.beta = problem_.beta.data();
    symm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

    detail::cublasSymmDispatcher symm_op(
      symm_desc,
      symm_workspace_.configuration,
      symm_workspace_.arguments
    );

    if (symm_op.status != Status::kSuccess) {
      results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kNotRun;
      return true;
    }

    results_.back().status = Status::kSuccess;

    status = symm_op(handle);

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
      *symm_workspace_.Computed,
      *symm_workspace_.Reference
    );

    // Save workspace if incorrect
    if (options.verification.save_workspace == SaveWorkspace::kIncorrect &&
      results_.back().verification_map[library::Provider::kCUBLAS] == Disposition::kIncorrect) {

      save_workspace(
        device_context,
        options,
        symm_desc,
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
bool SymmOperationProfiler::profile(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    // Initialize structure containing Symm arguments
    symm_workspace_.arguments.A = symm_workspace_.A->data();
    symm_workspace_.arguments.B = symm_workspace_.B->data();
    symm_workspace_.arguments.C = symm_workspace_.C->data();
    symm_workspace_.arguments.D = symm_workspace_.Computed->data();
    symm_workspace_.arguments.alpha = problem_.alpha.data();
    symm_workspace_.arguments.beta = problem_.beta.data();
    symm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

    results_.back().status = profile_cutlass_(
      results_.back(),
      options,
      operation,
      &symm_workspace_.arguments,
      symm_workspace_.host_workspace.data(),
      symm_workspace_.device_workspace.data()
    );
  }
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
