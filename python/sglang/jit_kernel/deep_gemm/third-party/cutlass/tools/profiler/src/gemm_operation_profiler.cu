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
#include <vector>

#include "cutlass/core_io.h"
#include <cuda_runtime_api.h>
#include <cuda/atomic>

#include "cutlass/profiler/cublas_helpers.h"
#include "cutlass/profiler/gemm_operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"
#include "cutlass/library/singleton.h"
#include "cutlass/library/library.h"
#include "cutlass/library/handle.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Ctor
GemmOperationProfiler::GemmOperationProfiler(Options const &options):
  OperationProfiler(
    options,
    library::OperationKind::kGemm,
    {
      {ArgumentTypeID::kEnumerated, {"gemm_kind"}, "Variant of GEMM (universal, gemm, planar_complex, planar_complex_array)"},
      {ArgumentTypeID::kInteger, {"m", "problem-size::m"}, "M dimension of the GEMM problem space"},
      {ArgumentTypeID::kInteger, {"n", "problem-size::n"}, "N dimension of the GEMM problem space"},
      {ArgumentTypeID::kInteger, {"k", "problem-size::k"}, "K dimension of the GEMM problem space"},
      {ArgumentTypeID::kTensor, {"A"}, "Tensor storing the A operand"},
      {ArgumentTypeID::kTensor, {"B"}, "Tensor storing the B operand"},
      {ArgumentTypeID::kTensor, {"C"}, "Tensor storing the C operand"},
      {ArgumentTypeID::kTensor, {"D"}, "Tensor storing the D output"},
      {ArgumentTypeID::kScalar, {"alpha", "epilogue::alpha"}, "Epilogue scalar alpha"},
      {ArgumentTypeID::kScalar, {"beta", "epilogue::beta"}, "Epilogue scalar beta"},
      {ArgumentTypeID::kEnumerated, {"split_k_mode", "split-k-mode"}, "Variant of split K mode(serial, parallel)"},
      {ArgumentTypeID::kInteger, {"split_k_slices", "split-k-slices"}, "Number of partitions of K dimension"},
      {ArgumentTypeID::kInteger, {"batch_count", "batch-count"}, "Number of GEMMs computed in one batch"},
      {ArgumentTypeID::kEnumerated, {"raster_order", "raster-order"}, "Raster order (heuristic, along_n, along_m)"},
      {ArgumentTypeID::kEnumerated, {"runtime_input_datatype_a", "runtime-input-datatype::a"}, "Runtime datatype (e4m3, e5m2, e3m2, e2m3, e2m1)"}, 
      {ArgumentTypeID::kEnumerated, {"runtime_input_datatype_b", "runtime-input-datatype::b"}, "Runtime datatype (e4m3, e5m2, e3m2, e2m3, e2m1)"}, 
      {ArgumentTypeID::kInteger, {"use_pdl", "use-pdl"}, "Use PDL (true, false)"}, 
      {ArgumentTypeID::kEnumerated, {"enable_sm90_mixed_dtype_shuffle_test", "enable-sm90-mixed-dtype-shuffle-test"}, "Enable SM90 mixed input data type kernel shuffle layout test (true, false)"},
      {ArgumentTypeID::kInteger, {"swizzle_size", "swizzle-size"}, "Size to swizzle"},
    },
    { library::Provider::kCUBLAS}
  ) {

  description_ = "      General matrix-matrix product. D = alpha * A*B + beta * C";
}

/// Destructor
GemmOperationProfiler::~GemmOperationProfiler() {

}

/// Prints usage statement for the math function
void GemmOperationProfiler::print_usage(std::ostream &out) const {
  out << "GEMM" << "\n\n";

  OperationProfiler::print_usage(out);
}

/// Prints examples
void GemmOperationProfiler::print_examples(std::ostream &out) const {

  out << "\nExamples:\n\n"
    << "Profile a particular problem size:\n"
    << "  $ cutlass_profiler --operation=Gemm --m=1024 --n=1024 --k=128\n\n"

    << "Schmoo over problem size and beta:\n"
    << "  $ cutlass_profiler --operation=Gemm --m=1024:4096:256 --n=1024:4096:256 --k=128:8192:128 --beta=0,1,2.5\n\n"

    << "Schmoo over accumulator types:\n"
    << "  $ cutlass_profiler --operation=Gemm --accumulator-type=f16,f32\n\n"

    << "Run when A is f16 with column-major and B is any datatype with row-major (For column major, use column, col, or n. For row major use, row or t):\n"
    << "  $ cutlass_profiler --operation=Gemm --A=f16:column --B=*:row\n\n"

    << "Profile a particular problem size with split K and parallel reduction:\n"
    << "  $ cutlass_profiler --operation=Gemm --split_k_mode=parallel --split_k_slices=2 --m=1024 --n=1024 --k=128\n\n"

    << "Using various input value distribution:\n"
    << "  $ cutlass_profiler --operation=Gemm --dist=uniform,min:0,max:3\n"
    << "  $ cutlass_profiler --operation=Gemm --dist=gaussian,mean:0,stddev:3\n"
    << "  $ cutlass_profiler --operation=Gemm --dist=sequential,start:0,delta:1\n\n"

    << "Run a kernel with cta tile size of 256x128x32 and save workspace if results are incorrect (note that --cta-tile::k=32 is default cta-tile size):\n"
    << " $ cutlass_profiler --operation=Gemm --cta_m=256 --cta_n=128  --cta_k=32 --save-workspace=incorrect\n\n"

    << "Test your changes to gemm kernels with a quick functional test and save results in functional-test.csv:\n"
    << " $ cutlass_profiler  --operation=Gemm \\ \n"
    << "   --m=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \\ \n"
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

Status GemmOperationProfiler::GemmProblem::parse(
  library::GemmDescription const &operation_desc,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  this->mode = library::GemmUniversalMode::kGemm;

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

  if (!arg_as_int(this->cluster_m, "cluster_m", problem_space, problem)) {
    // default value
    this->cluster_m = std::string(operation_desc.name).find("_2sm") != std::string::npos ? 2 : 1;
  }

  if (!arg_as_int(this->cluster_n, "cluster_n", problem_space, problem)) {
    // default value
    this->cluster_n = 1;
  }

  if (!arg_as_int(this->cluster_k, "cluster_k", problem_space, problem)) {
    // default value
    this->cluster_k = 1;
  }

  if (!arg_as_int(this->cluster_m_fallback, "cluster_m_fallback", problem_space, problem)) {
    // default value
    this->cluster_m_fallback = (this->cluster_m % 2 == 0) ? 2 : 1;
  }

  if (!arg_as_int(this->cluster_n_fallback, "cluster_n_fallback", problem_space, problem)) {
    // default value
    this->cluster_n_fallback = 1;
  }

  if (!arg_as_int(this->cluster_k_fallback, "cluster_k_fallback", problem_space, problem)) {
    // default value
    this->cluster_k_fallback = 1;
  }

  if (!arg_as_bool(this->use_pdl, "use_pdl", problem_space, problem)) {
    // default value
    this->use_pdl = false;
  }

  if (!arg_as_bool(this->enable_sm90_mixed_dtype_shuffle_test, "enable_sm90_mixed_dtype_shuffle_test", problem_space, problem)) {
    // default value
    this->enable_sm90_mixed_dtype_shuffle_test = false;
  }

  if (!arg_as_SplitKModeID(this->split_k_mode, "split_k_mode", problem_space, problem)) {
    // default value
    this->split_k_mode = library::SplitKMode::kSerial;
  }

  this->mode = library::GemmUniversalMode::kGemm;
  if (this->split_k_mode == library::SplitKMode::kParallel) {
    this->mode = library::GemmUniversalMode::kGemmSplitKParallel;
  }

  if (!arg_as_int(this->split_k_slices, "split_k_slices", problem_space, problem)) {
    // default value
    this->split_k_slices = 1;
  }

  
  if (!arg_as_RuntimeDatatype(this->runtime_input_datatype_a, "runtime_input_datatype_a", problem_space, problem)) {
    // default value
    this->runtime_input_datatype_a = cutlass::library::RuntimeDatatype::kStatic;
  }

  if (!arg_as_RuntimeDatatype(this->runtime_input_datatype_b, "runtime_input_datatype_b", problem_space, problem)) {
    // default value
    this->runtime_input_datatype_b = cutlass::library::RuntimeDatatype::kStatic;
  }
  

  if (!arg_as_int(this->batch_count, "batch_count", problem_space, problem)) {
    // default value
    this->batch_count = 1;
  } else if (this->batch_count > 1) {
    this->mode = library::GemmUniversalMode::kBatched;
  }

  if (!arg_as_int(this->swizzle_size, "swizzle_size", problem_space, problem)) {
    // default value
    this->swizzle_size = 1;
  }

  if (!arg_as_RasterOrder(this->raster_order, "raster_order", problem_space, problem)) {
    // default value
    this->raster_order = library::RasterOrder::kHeuristic;
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

  this->lda = DeviceAllocation::get_packed_layout(
    operation_desc.A.layout, {int(this->m), int(this->k)}).front();

  this->ldb = DeviceAllocation::get_packed_layout(
    operation_desc.B.layout, {int(this->k), int(this->n)}).front();

  this->ldc = DeviceAllocation::get_packed_layout(
    operation_desc.C.layout, {int(this->m), int(this->n)}).front();

  // instantiation
  int num_sizes = 8;
  this->problem_sizes.resize(num_sizes);
  this->leading_dims.resize(num_sizes, {0, 0, 0});
    
  int m0 = 1024;
  int n0 = 1024;
  int k0 = 1024;
  for (int i = 0; i < num_sizes; i++) {
    auto m = m0 * (i + 1);
    auto n = n0 * (i + 1);
    auto k = k0 * (i + 1);
    this->problem_sizes[i] = {m, n, k};
    this->leading_dims[i] = {
      DeviceAllocation::get_packed_layout(operation_desc.A.layout, {int(m), int(k)}).front(),
      DeviceAllocation::get_packed_layout(operation_desc.B.layout, {int(k), int(n)}).front(),
      DeviceAllocation::get_packed_layout(operation_desc.C.layout, {int(m), int(n)}).front()
    };

  }

  this->raster_orders = {
    cutlass::library::RasterOrder::kAlongN,
    cutlass::library::RasterOrder::kAlongM
  };

  this->swizzle_sizes = {1, 2, 4, 8};

  this->preferred_clusters = {
    {1, 1, 1}, {2, 1, 1}, {2, 2, 1}, {4, 1, 1}, {4, 2, 1}, {4, 4, 1}, {8, 2, 1}
  };

  this->fallback_clusters = {
    {1, 1, 1}, {2, 1, 1}, {2, 2, 1}
  };

  return Status::kSuccess;
}

int64_t GemmOperationProfiler::GemmProblem::bytes_with_problem_shape(
  library::GemmDescription const &operation_desc,
  gemm::GemmCoord const &problem_shape) const {

  // Input bytes read and Output bytes written for the gemm problem
  int64_t bytes =
    int64_t(library::sizeof_bits(operation_desc.A.element) * problem_shape.m() / 8) * problem_shape.k() +
    int64_t(library::sizeof_bits(operation_desc.B.element) * problem_shape.n() / 8) * problem_shape.k() +
    int64_t(library::sizeof_bits(operation_desc.C.element) * problem_shape.m() / 8) * problem_shape.n();

  // Set is_beta_zero true if beta is zero
  bool is_beta_zero = std::all_of(beta.begin(), beta.end(), [](uint8_t i) { return i==0; });

  // Output bytes read for the gemm problem for non-zero beta values
  if (!is_beta_zero) {
    bytes += int64_t(library::sizeof_bits(operation_desc.C.element) * problem_shape.m() / 8) * problem_shape.n();
  }

  bytes *= batch_count;

  return bytes;

}

/// Total number of bytes loaded
int64_t GemmOperationProfiler::GemmProblem::bytes(library::GemmDescription const &operation_desc) const {
  gemm::GemmCoord problem_shape({int(m), int(n), int(k)});
  return bytes_with_problem_shape(operation_desc, problem_shape);
}

/// Total number of flops computed
int64_t GemmOperationProfiler::GemmProblem::flops_with_problem_shape(
  library::GemmDescription const &operation_desc,
  gemm::GemmCoord const &problem_shape) const {
  int64_t flops_ = (int64_t(problem_shape.m()) * problem_shape.n() * problem_shape.k() + problem_shape.m() * problem_shape.n()) * 2 * batch_count;

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

/// Total number of flops computed
int64_t GemmOperationProfiler::GemmProblem::flops(library::GemmDescription const &operation_desc) const {
  gemm::GemmCoord problem_shape({int(m), int(n), int(k)});
  return flops_with_problem_shape(operation_desc, problem_shape);
}


/// Initializes a performance result
void GemmOperationProfiler::GemmProblem::initialize_result(
  PerformanceResult &result,
  library::GemmDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  result.arguments.resize(problem_space.rank());

  set_argument(result, "gemm_kind", problem_space, library::to_string(operation_desc.gemm_kind));

  set_argument(result, "A", problem_space,
    std::string(library::to_string(operation_desc.A.element)) + ":" + library::to_string(operation_desc.A.layout));

  set_argument(result, "B", problem_space,
    std::string(library::to_string(operation_desc.B.element)) + ":" + library::to_string(operation_desc.B.layout));

  set_argument(result, "C", problem_space,
    std::string(library::to_string(operation_desc.C.element)) + ":" + library::to_string(operation_desc.C.layout));

  set_argument(result, "D", problem_space,
    std::string(library::to_string(operation_desc.D.element)) + ":" + library::to_string(operation_desc.D.layout));

  set_argument(result, "m", problem_space, m);
  set_argument(result, "n", problem_space, n);
  set_argument(result, "k", problem_space, k);

  auto cluster_shape = operation_desc.tile_description.cluster_shape;
  auto is_dynamic = cluster_shape.m() == 0 || cluster_shape.n() == 0 || cluster_shape.k() == 0;
  set_argument(result, "cluster_m", problem_space, is_dynamic ? this->cluster_m : cluster_shape.m());
  set_argument(result, "cluster_n", problem_space, is_dynamic ? this->cluster_n : cluster_shape.n());
  set_argument(result, "cluster_k", problem_space, is_dynamic ? this->cluster_k : cluster_shape.k());
  set_argument(result, "cluster_m_fallback", problem_space, cluster_m_fallback);
  set_argument(result, "cluster_n_fallback", problem_space, cluster_n_fallback);
  set_argument(result, "cluster_k_fallback", problem_space, cluster_k_fallback);
  

  set_argument(result, "split_k_mode", problem_space, library::to_string(split_k_mode));
  set_argument(result, "split_k_slices", problem_space, split_k_slices);
  set_argument(result, "batch_count", problem_space, batch_count);
  set_argument(result, "raster_order", problem_space, library::to_string(raster_order));
  set_argument(result, "swizzle_size", problem_space, swizzle_size);
  set_argument(result, "use_pdl", problem_space, library::to_string(use_pdl));
  set_argument(result, "enable_sm90_mixed_dtype_shuffle_test", problem_space, library::to_string(enable_sm90_mixed_dtype_shuffle_test));

  
  set_argument(result, "runtime_input_datatype_a", problem_space, library::to_string(runtime_input_datatype_a));
  set_argument(result, "runtime_input_datatype_b", problem_space, library::to_string(runtime_input_datatype_b));
  

  set_argument(result, "alpha", problem_space,
    library::lexical_cast(alpha, operation_desc.element_epilogue));

  set_argument(result, "beta", problem_space,
    library::lexical_cast(beta, operation_desc.element_epilogue));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Extracts the problem dimensions
Status GemmOperationProfiler::initialize_configuration(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::GemmDescription const &operation_desc =
    static_cast<library::GemmDescription const &>(operation->description());

  if (operation_desc.gemm_kind != library::GemmKind::kUniversal) {
    return Status::kErrorInvalidProblem;
  }

  Status status = problem_.parse(operation_desc, problem_space, problem);

  // Note: this is a temporary workaround
  bool is_sm90_operation = (strstr(operation_desc.name, "_sm90") != NULL);
  bool is_sm90_mixed_dtype_shuffle_operation = (strstr(operation_desc.name, "_shfl") != NULL);
  if (is_sm90_mixed_dtype_shuffle_operation && (problem_.enable_sm90_mixed_dtype_shuffle_test == false)) {
    return Status::kErrorInvalidProblem;
  }

  if (status != Status::kSuccess) {
    return status;
  }

  auto const device_count = options.device.devices.size();

  gemm_workspace_.clear();

  library::NumericTypeID a_elem = library::get_real_type(operation_desc.A.element);
  library::NumericTypeID b_elem = library::get_real_type(operation_desc.B.element);
  int a_elem_bits = library::sizeof_bits(a_elem);
  int b_elem_bits = library::sizeof_bits(b_elem);
  bool is_sm90_mixed_dtype_operation = is_sm90_operation && (a_elem_bits != b_elem_bits);

  for (size_t i = 0; i < device_count; ++i) {
    cudaSetDevice(options.device.device_id(i));
    gemm_workspace_.emplace_back();
    cudaStreamCreateWithFlags(&gemm_workspace_[i].stream, cudaStreamNonBlocking);
    gemm_workspace_[i].configuration.mode = problem_.mode;
    gemm_workspace_[i].configuration.problem_size.m() = int(problem_.m);
    gemm_workspace_[i].configuration.problem_size.n() = int(problem_.n);
    gemm_workspace_[i].configuration.problem_size.k() = int(problem_.k);
    
    gemm_workspace_[i].configuration.cluster_shape.m() = int(problem_.cluster_m);
    gemm_workspace_[i].configuration.cluster_shape.n() = int(problem_.cluster_n);
    gemm_workspace_[i].configuration.cluster_shape.k() = int(problem_.cluster_k);
    gemm_workspace_[i].configuration.cluster_shape_fallback.m() = int(problem_.cluster_m_fallback);
    gemm_workspace_[i].configuration.cluster_shape_fallback.n() = int(problem_.cluster_n_fallback);
    gemm_workspace_[i].configuration.cluster_shape_fallback.k() = int(problem_.cluster_k_fallback);
    gemm_workspace_[i].configuration.lda = problem_.lda;
    gemm_workspace_[i].configuration.ldb = problem_.ldb;
    gemm_workspace_[i].configuration.ldc = problem_.ldc;
    gemm_workspace_[i].configuration.ldd = problem_.ldc;

    gemm_workspace_[i].configuration.device_count = static_cast<int>(device_count);
    gemm_workspace_[i].arguments.device_index = static_cast<int>(i);
    gemm_workspace_[i].arguments.use_pdl = problem_.use_pdl;

    if (problem_.mode == library::GemmUniversalMode::kBatched) {
      gemm_workspace_[i].configuration.batch_count = problem_.batch_count;
    }
    else {
      gemm_workspace_[i].configuration.batch_count = problem_.split_k_slices;
    }

    gemm_workspace_[i].arguments.problem_size.m() = int(problem_.m);
    gemm_workspace_[i].arguments.problem_size.n() = int(problem_.n);
    gemm_workspace_[i].arguments.problem_size.k() = int(problem_.k);
    if (problem_.mode == library::GemmUniversalMode::kBatched) {
      gemm_workspace_[i].arguments.batch_count = problem_.batch_count;
    }
    else {
      gemm_workspace_[i].arguments.batch_count = problem_.split_k_slices;
    }

    gemm_workspace_[i].arguments.A = nullptr;
    gemm_workspace_[i].arguments.B = nullptr;
    gemm_workspace_[i].arguments.C = nullptr;
    gemm_workspace_[i].arguments.D = nullptr;
    gemm_workspace_[i].arguments.alpha = problem_.alpha.data();
    gemm_workspace_[i].arguments.beta = problem_.beta.data();
    gemm_workspace_[i].arguments.pointer_mode = library::ScalarPointerMode::kHost;
    gemm_workspace_[i].arguments.swizzle_size = problem_.swizzle_size;
    gemm_workspace_[i].arguments.raster_order = problem_.raster_order;
    gemm_workspace_[i].arguments.cluster_shape = {int(problem_.cluster_m), int(problem_.cluster_n), int(problem_.cluster_k)}; 
    gemm_workspace_[i].arguments.cluster_shape_fallback = {int(problem_.cluster_m_fallback), int(problem_.cluster_n_fallback), int(problem_.cluster_k_fallback)}; 
    gemm_workspace_[i].arguments.split_k_slices = problem_.split_k_slices;

    
    gemm_workspace_[i].arguments.runtime_input_datatype_a = problem_.runtime_input_datatype_a;
    gemm_workspace_[i].arguments.runtime_input_datatype_b = problem_.runtime_input_datatype_b;
    

    initialize_result_(this->model_result_, options, operation_desc, problem_space);
    if (is_sm90_mixed_dtype_operation)
    {
      const int options_g = problem_.k;
      const int options_l = problem_.batch_count;
      const int scale_k = (problem_.k + options_g - 1) / options_g;
      // We cannot get the mainloop's ElementScale and ElementZero here,
      // use the wide type to allocate a large enough workspace for S and Z.
      library::NumericTypeID wide_dtype;
      size_t SZ_mat_size = 0;
      if (a_elem_bits > b_elem_bits) {
        wide_dtype = a_elem;
        SZ_mat_size = static_cast<size_t>(problem_.n * scale_k);
      }
      else {
        wide_dtype = b_elem;
        SZ_mat_size = static_cast<size_t>(problem_.m * scale_k);
      }

      gemm_workspace_[i].Scale = device_context.allocate_tensor(
        options,
        "Scale",
        wide_dtype,
        library::LayoutTypeID::kRowMajor,
        {int(SZ_mat_size), int(options_l)},
        {int(options_l)},
        problem_.batch_count * gemm_workspace_[i].problem_count,
        i // device_index
      );
      gemm_workspace_[i].Zero = device_context.allocate_tensor(
        options,
        "Zero",
        wide_dtype,
        library::LayoutTypeID::kRowMajor,
        {int(SZ_mat_size), int(options_l)},
        {int(options_l)},
        problem_.batch_count * gemm_workspace_[i].problem_count,
        i // device_index
      );

      // Packed scale is for int4 * fp8, where the original scale is fp8, and
      // each scale element will be packed into an Array<fp8, 8> which is 64-bit
      gemm_workspace_[i].packed_Scale = device_context.allocate_tensor(
        options,
        "packed-Scale",
        library::NumericTypeID::kU64,
        library::LayoutTypeID::kRowMajor,
        {int(SZ_mat_size), int(options_l)},
        {int(options_l)},
        problem_.batch_count * gemm_workspace_[i].problem_count,
        i // device_index
      );

      gemm_workspace_[i].arguments.problem_size = {int(problem_.m), int(problem_.n), int(problem_.k)};
      gemm_workspace_[i].arguments.batch_count = problem_.batch_count;

      // Here is the first touch of the arguments, mark the mixed dtype,
      // populate the scale and zero tensors in the following can_implement() call later.
      // A and B are not populated at this moment, so do not update the dequantized A or B
      gemm_workspace_[i].arguments.is_sm90_mixed_dtype = true;
      gemm_workspace_[i].arguments.wider_operand = (a_elem_bits > b_elem_bits) ? cutlass::library::Sm90MixedInputWiderOperand::A : cutlass::library::Sm90MixedInputWiderOperand::B;
      gemm_workspace_[i].arguments.generate_scale_and_zero = true;
      gemm_workspace_[i].arguments.generate_dequantized_AB = false;
      gemm_workspace_[i].arguments.Scale = gemm_workspace_[i].Scale->data();
      gemm_workspace_[i].arguments.Zero = gemm_workspace_[i].Zero->data();
      gemm_workspace_[i].arguments.packed_Scale = gemm_workspace_[i].packed_Scale->data();
    }  // End of "if (is_sm90_mixed_dtype_operation)"

    const auto can_implement = operation->can_implement(&gemm_workspace_[i].configuration, &gemm_workspace_[i].arguments);
    if (can_implement != Status::kSuccess) {
      return can_implement;
    }
  }

  // initialize reduction operation for parallel splitKMode
  if (problem_.split_k_mode == library::SplitKMode::kParallel) {
    if (!initialize_reduction_configuration_(operation, problem)) {
      return Status::kErrorInternal;
    }
  }

  return status;
}

void GemmOperationProfiler::update_workspace_(
  GemmWorkspace &gemm_workspace,
  gemm::GemmCoord const &problem_shape,
  std::array<int64_t, 3> const &leading_dim,
  std::array<int64_t, 3> const &preferred_cluster,
  std::array<int64_t, 3> const &fallback_cluster,
  cutlass::library::RasterOrder const &raster_order,
  int swizzle_size,
  bool is_dynamic_cluster_enabled
) {

  gemm_workspace.arguments.problem_size.m() = problem_shape.m();
  gemm_workspace.arguments.problem_size.n() = problem_shape.n();
  gemm_workspace.arguments.problem_size.k() = problem_shape.k();

  gemm_workspace.arguments.lda = leading_dim[0];
  gemm_workspace.arguments.ldb = leading_dim[1];
  gemm_workspace.arguments.ldc = leading_dim[2];

  gemm_workspace.arguments.swizzle_size = swizzle_size;
  gemm_workspace.arguments.raster_order = raster_order;

  if (is_dynamic_cluster_enabled) {
    gemm_workspace.arguments.cluster_shape = {int(preferred_cluster[0]), int(preferred_cluster[1]), int(preferred_cluster[2])};
    gemm_workspace.arguments.cluster_shape_fallback = {int(fallback_cluster[0]), int(fallback_cluster[1]), int(fallback_cluster[2])};
    gemm_workspace.configuration.cluster_shape = {int(preferred_cluster[0]), int(preferred_cluster[1]), int(preferred_cluster[2])};
    gemm_workspace.configuration.cluster_shape_fallback = {int(fallback_cluster[0]), int(fallback_cluster[1]), int(fallback_cluster[2])};
  }

  gemm_workspace.configuration.problem_size.m() = problem_shape.m();
  gemm_workspace.configuration.problem_size.n() = problem_shape.n();
  gemm_workspace.configuration.problem_size.k() = problem_shape.k();

  gemm_workspace.configuration.lda = leading_dim[0];
  gemm_workspace.configuration.ldb = leading_dim[1];
  gemm_workspace.configuration.ldc = leading_dim[2];

}

void GemmOperationProfiler::update_result_(
  PerformanceResult &result,
  library::GemmDescription const &operation_desc,
  ProblemSpace const &problem_space,
  gemm::GemmCoord const &problem_shape,
  cutlass::library::RasterOrder const &raster_order,
  std::array<int64_t, 3> const &preferred_cluster,
  std::array<int64_t, 3> const &fallback_cluster,
  int swizzle_size,
  bool is_dynamic_cluster_enabled
) {
  result.bytes = problem_.bytes_with_problem_shape(operation_desc, problem_shape);
  result.flops = problem_.flops_with_problem_shape(operation_desc, problem_shape);

  set_argument(result, "m", problem_space, problem_shape.m());
  set_argument(result, "n", problem_space, problem_shape.n());
  set_argument(result, "k", problem_space, problem_shape.k());

  set_argument(result, "raster_order", problem_space, library::to_string(raster_order));
  set_argument(result, "swizzle_size", problem_space, swizzle_size);

  if (is_dynamic_cluster_enabled) {
    set_argument(result, "cluster_m", problem_space, preferred_cluster[0]);
    set_argument(result, "cluster_n", problem_space, preferred_cluster[1]);
    set_argument(result, "cluster_k", problem_space, preferred_cluster[2]);
    set_argument(result, "cluster_m_fallback", problem_space, fallback_cluster[0]);
    set_argument(result, "cluster_n_fallback", problem_space, fallback_cluster[1]);
    set_argument(result, "cluster_k_fallback", problem_space, fallback_cluster[2]);
  }

}

/// Initializes the performance result
void GemmOperationProfiler::initialize_result_(
  PerformanceResult &result,
  Options const &options,
  library::GemmDescription const &operation_desc,
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
  result.runtime_vector.resize(options.device.devices.size(), 0);

}

/// Initialize reduction problem dimensions and library::Operation
bool GemmOperationProfiler::initialize_reduction_configuration_(
  library::Operation const *operation,
  ProblemSpace::Problem const &problem) {

  library::GemmDescription const &gemm_desc =
    static_cast<library::GemmDescription const&>(operation->description());

  if (!cast_from_double(problem_.alpha_one, gemm_desc.element_epilogue, 1)) {
    return false;
  }

  if (!cast_from_double(problem_.beta_zero, gemm_desc.element_epilogue, 0)) {
    return false;
  }

  /// initialize library::ReductionConfiguration
  for (auto &gemm_workspace : gemm_workspace_) {
    gemm_workspace.reduction_configuration.problem_size      = gemm::GemmCoord(int(problem_.n), int(problem_.m), int(problem_.k)).mn();
    gemm_workspace.reduction_configuration.partitions        = int(problem_.split_k_slices);
    gemm_workspace.reduction_configuration.partition_stride  = gemm::GemmCoord(int(problem_.n), int(problem_.m), int(problem_.k)).mn().product();
    gemm_workspace.reduction_configuration.ldw               = problem_.ldc;
    gemm_workspace.reduction_configuration.lds               = problem_.ldc;
    gemm_workspace.reduction_configuration.ldd               = problem_.ldc;
  }

  // find reduction operation
  library::ReductionFunctionalKey reduction_key(
    library::Provider::kCUTLASS,
    gemm_desc.tile_description.math_instruction.element_accumulator,    // element workspace
    gemm_desc.tile_description.math_instruction.element_accumulator,    // element accumulator
    gemm_desc.D.element,                                                // element output
    gemm_desc.element_epilogue                                          // element compute
  );

  auto reduction_it = library::Singleton::get().operation_table.reduction_operations.find(reduction_key);

  if (reduction_it == library::Singleton::get().operation_table.reduction_operations.end()) {
    return false;
  }

  // initialize reduction operation required for parallel split-k operator
  reduction_op_ = reduction_it->second;

  // reduction operation found and initialized
  return true;
}

/// Initializes workspace
Status GemmOperationProfiler::initialize_workspace(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  cudaError_t result;
  result = cudaSetDevice(options.device.device_id(0));
  if (result != cudaSuccess) {
    throw std::runtime_error("cudaSetDevice() failed.");
  }

  library::Operation const* underlying_operation = operation;

  if (problem_.split_k_mode == library::SplitKMode::kParallel) {
    if (!(underlying_operation = library::find_gemm_operation_for_parallel_reduction(operation))) {
      return Status::kErrorNotSupported;
    }
  }

  library::GemmDescription const &operation_desc =
    static_cast<library::GemmDescription const &>(operation->description());

  bool is_sparse = operation_desc.tile_description.math_instruction.opcode_class == cutlass::library::OpcodeClassID::kSparseTensorOp;

  for (size_t i = 0; i < gemm_workspace_.size(); ++i) {
    cudaSetDevice(options.device.device_id(i));

    // Compute the number of copies of the problem to avoid L2 camping.
    if (!options.profiling.workspace_count) {
      int64_t bytes = problem_.bytes(operation_desc);
      if (bytes < 3 * int64_t(options.device.properties[0].l2CacheSize)) {
        gemm_workspace_[i].problem_count =
          1 + int((3 * int64_t(options.device.properties[0].l2CacheSize)) / bytes);
      }
      else {
        gemm_workspace_[i].problem_count = 1;
      }
    }
    else {
      gemm_workspace_[i].problem_count = options.profiling.workspace_count;
    }

    bool allocate_device_tensors = options.execution_mode != ExecutionMode::kDryRun;
    if (allocate_device_tensors) {
      bool enable_deep_profiling = options.profiling.enable_kernel_performance_search;
      int seed_shift = 0;

      // When exhaustive performance search (deep profiling) option is enabled, device buffers are initialized to the largest problem shape
      // so that later performance search can re-use those buffers.
      int init_m = enable_deep_profiling ? std::max(int(problem_.m),  problem_.problem_sizes.back().m()) : int(problem_.m);
      int init_n = enable_deep_profiling ? std::max(int(problem_.n),  problem_.problem_sizes.back().n()) : int(problem_.n);
      int init_k = enable_deep_profiling ? std::max(int(problem_.k),  problem_.problem_sizes.back().k()) : int(problem_.k);
      int init_lda = enable_deep_profiling ? int(std::max(problem_.lda,  problem_.leading_dims.back()[0])) : int(problem_.lda);
      int init_ldb = enable_deep_profiling ? int(std::max(problem_.ldb,  problem_.leading_dims.back()[1])) : int(problem_.ldb);
      int init_ldc = enable_deep_profiling ? int(std::max(problem_.ldc,  problem_.leading_dims.back()[2])) : int(problem_.ldc);

      gemm_workspace_[i].A = device_context.allocate_and_initialize_tensor(
        options,
        "A",
        operation_desc.A.element,
        operation_desc.A.layout,
        {init_m, init_k},
        {init_lda},
        problem_.batch_count * gemm_workspace_[i].problem_count,
        seed_shift++,
        i // device_index
      );

      gemm_workspace_[i].B = device_context.allocate_and_initialize_tensor(
        options,
        "B",
        operation_desc.B.element,
        operation_desc.B.layout,
        {init_k, init_n},
        {init_ldb},
        problem_.batch_count * gemm_workspace_[i].problem_count,
        seed_shift++,
        i // device_index
      );

      gemm_workspace_[i].C = device_context.allocate_and_initialize_tensor(
        options,
        "C",
        operation_desc.C.element,
        operation_desc.C.layout,
        {init_m, init_n},
        {init_ldc},
        problem_.batch_count * gemm_workspace_[i].problem_count,
        seed_shift++,
        i // device_index
      );

      gemm_workspace_[i].Computed = device_context.allocate_tensor(
        options,
        "D",
        operation_desc.D.element,
        operation_desc.D.layout,
        {init_m, init_n},
        {init_ldc},
        problem_.batch_count * gemm_workspace_[i].problem_count,
        i // device_index
      );

      gemm_workspace_[i].Reference = device_context.allocate_tensor(
        options,
        "Reference",
        operation_desc.D.element,
        operation_desc.D.layout,
        {init_m, init_n},
        {init_ldc},
        problem_.batch_count * gemm_workspace_[i].problem_count,
        i // device_index
      );

      if (gemm_workspace_[i].arguments.is_sm90_mixed_dtype) {
        // Dequantized tensor has the same shape of the narrow data type tensor,
        // and the same data type as the wide data type tensor
        // Encoded tensor has the same shape and data type of the narrow data type tensor
        if (gemm_workspace_[i].arguments.wider_operand == cutlass::library::Sm90MixedInputWiderOperand::A) {
          gemm_workspace_[i].dequantized_AB = device_context.allocate_tensor(
            options,
            "dequantized-B",
            operation_desc.A.element,
            operation_desc.B.layout,
            {int(problem_.k), int(problem_.n)},
            {int(problem_.ldb)},
            problem_.batch_count * gemm_workspace_[i].problem_count,
            i // device_index
          );
          gemm_workspace_[i].encoded_AB = device_context.allocate_tensor(
            options,
            "encoded-B",
            operation_desc.B.element,
            operation_desc.B.layout,
            {int(problem_.k), int(problem_.n)},
            {int(problem_.ldb)},
            problem_.batch_count * gemm_workspace_[i].problem_count,
            i // device_index
          );
        }
        else {
          gemm_workspace_[i].dequantized_AB = device_context.allocate_tensor(
            options,
            "dequantized-A",
            operation_desc.B.element,
            operation_desc.A.layout,
            {int(problem_.m), int(problem_.k)},
            {int(problem_.lda)},
            problem_.batch_count * gemm_workspace_[i].problem_count,
            i // device_index
          );
          gemm_workspace_[i].encoded_AB = device_context.allocate_tensor(
            options,
            "encoded-A",
            operation_desc.A.element,
            operation_desc.A.layout,
            {int(problem_.m), int(problem_.k)},
            {int(problem_.lda)},
            problem_.batch_count * gemm_workspace_[i].problem_count,
            i // device_index
          );
        }
      }  // End of "if (gemm_workspace_[i].arguments.is_sm90_mixed_dtype)"
    }

    if (options.execution_mode != ExecutionMode::kDryRun) {
      // NOTE: the leading non-batch strides are duplicated here for 3.0 API kernels
      gemm_workspace_[i].arguments.problem_size = {int(problem_.m), int(problem_.n), int(problem_.k)};
      gemm_workspace_[i].arguments.cluster_shape = {int(problem_.cluster_m), int(problem_.cluster_n), int(problem_.cluster_k)}; 
      gemm_workspace_[i].arguments.cluster_shape_fallback = {int(problem_.cluster_m_fallback), int(problem_.cluster_n_fallback), int(problem_.cluster_k_fallback)};
      gemm_workspace_[i].arguments.split_k_slices = problem_.split_k_slices;
      gemm_workspace_[i].arguments.batch_count = problem_.batch_count;
      gemm_workspace_[i].arguments.lda = problem_.lda;
      gemm_workspace_[i].arguments.ldb = problem_.ldb;
      gemm_workspace_[i].arguments.ldc = problem_.ldc;
      gemm_workspace_[i].arguments.ldd = problem_.ldc;
      gemm_workspace_[i].arguments.batch_stride_A = gemm_workspace_[i].A->batch_stride();
      gemm_workspace_[i].arguments.batch_stride_B = gemm_workspace_[i].B->batch_stride();
      gemm_workspace_[i].arguments.batch_stride_C = gemm_workspace_[i].C->batch_stride();
      gemm_workspace_[i].arguments.batch_stride_D = gemm_workspace_[i].Computed->batch_stride();

      /* Query device SM count to pass onto the kernel as an argument, where needed */
      gemm_workspace_[i].arguments.sm_count = options.device.get_sm_count(i);
      gemm_workspace_[i].arguments.device_index = static_cast<int>(i);
    }
  }

  //
  // Initialize the CUTLASS operation
  //
  Status status = Status::kSuccess;

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    if (options.execution_mode != ExecutionMode::kDryRun) {
      for (size_t i = 0; i < gemm_workspace_.size(); ++i) {
        cudaSetDevice(options.device.device_id(i));
        uint64_t workspace_size = underlying_operation->get_host_workspace_size(&gemm_workspace_[i].configuration);
        gemm_workspace_[i].host_workspace.resize(workspace_size, 0);

        workspace_size = underlying_operation->get_device_workspace_size(&gemm_workspace_[i].configuration,
                                                              &gemm_workspace_[i].arguments);
        if (is_sparse) {
          // sparse gemm get_device_workspace_size() only return device workspace size per iteration
          // Needs to multiply it w/ number of iteration
          workspace_size *= gemm_workspace_[i].problem_count;
        }
        gemm_workspace_[i].device_workspace.reset(library::NumericTypeID::kU8, workspace_size);

        // Convert to structure sparse contents here.
        if (is_sparse) {
          uint8_t* profiler_workspaces[1];
          profiler_workspaces[0] = reinterpret_cast<uint8_t*>(gemm_workspace_[i].A->data());
          // Sparse operations have a different initialize interface.
          // initialize_with_profiler_workspace converts mxk tensorA to compressed mxk/sp tensorA and the tensorE
          auto modifiable_underlying_op = const_cast<library::Operation*>(underlying_operation);
          status = modifiable_underlying_op->initialize_with_profiler_workspace(
            &gemm_workspace_[i].configuration,
            gemm_workspace_[i].host_workspace.data(),
            gemm_workspace_[i].device_workspace.data(),
            profiler_workspaces,
            gemm_workspace_[i].problem_count,
            gemm_workspace_[i].stream);
        }
        else {
          status = underlying_operation->initialize(
            &gemm_workspace_[i].configuration,
            gemm_workspace_[i].host_workspace.data(),
            gemm_workspace_[i].device_workspace.data(),
            gemm_workspace_[i].stream);
        }

        if (status != Status::kSuccess) {
          return status;
        }

        if (problem_.split_k_mode == library::SplitKMode::kParallel) {
          workspace_size = reduction_op_->get_host_workspace_size(&gemm_workspace_[i].reduction_configuration);
          gemm_workspace_[i].reduction_host_workspace.resize(workspace_size, 0);

          status = reduction_op_->initialize(
            &gemm_workspace_[i].reduction_configuration,
            gemm_workspace_[i].reduction_host_workspace.data(),
            nullptr,
            gemm_workspace_[i].stream);

          if (status != Status::kSuccess) {
            return status;
          }
        }
      }
    }

    for (size_t i = 0; i < gemm_workspace_.size(); ++i) {
      cudaSetDevice(options.device.device_id(i));
      cudaDeviceSynchronize();
    }

    //
    // If CUTLASS is enabled, generate a result for it
    //
    results_.push_back(model_result_);
    results_.back().provider = library::Provider::kCUTLASS;
    results_.back().op_kind = library::OperationKind::kGemm;
    results_.back().disposition = Disposition::kNotRun;

    for (auto provider : verification_providers_) {
      results_.back().verification_map[provider] = Disposition::kNotRun;
    }
  }
  return status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Verifies CUTLASS against references
bool GemmOperationProfiler::verify_cutlass(
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
  for (size_t i = 0; i < gemm_workspace_.size(); ++i) {
    gemm_workspace_[i].arguments.A = gemm_workspace_[i].A->data();
    gemm_workspace_[i].arguments.B = gemm_workspace_[i].B->data();
    gemm_workspace_[i].arguments.C = gemm_workspace_[i].C->data();
    gemm_workspace_[i].arguments.D = gemm_workspace_[i].Computed->data();
    gemm_workspace_[i].arguments.alpha = problem_.alpha.data();
    gemm_workspace_[i].arguments.beta = problem_.beta.data();
    gemm_workspace_[i].arguments.pointer_mode = library::ScalarPointerMode::kHost;
    gemm_workspace_[i].arguments.batch_stride_A = gemm_workspace_[i].A->batch_stride();
    gemm_workspace_[i].arguments.batch_stride_B = gemm_workspace_[i].B->batch_stride();
    gemm_workspace_[i].arguments.batch_stride_C = gemm_workspace_[i].C->batch_stride();
    gemm_workspace_[i].arguments.batch_stride_D = gemm_workspace_[i].Computed->batch_stride();

    if (gemm_workspace_[i].arguments.is_sm90_mixed_dtype) {
      // Scale and zero already generated in initialize_configuration(),
      // A and B already generated in initialize_workspace(), signal
      // GemmUniversal3xOperation::update_arguments_() (trigger by underlying_operation->run())
      // to generate the dequantized matrix for verification
      gemm_workspace_[i].arguments.generate_scale_and_zero = false;
      gemm_workspace_[i].arguments.generate_dequantized_AB = true;
      gemm_workspace_[i].arguments.dequantized_AB = gemm_workspace_[i].dequantized_AB->data();
      gemm_workspace_[i].arguments.encoded_AB = gemm_workspace_[i].encoded_AB->data();
    }

    if (problem_.split_k_mode == library::SplitKMode::kParallel) {
      gemm_workspace_[i].arguments.D                       = gemm_workspace_[i].device_workspace.data();
      gemm_workspace_[i].arguments.alpha                   = problem_.alpha_one.data();
      gemm_workspace_[i].arguments.beta                    = problem_.beta_zero.data();

      gemm_workspace_[i].reduction_arguments.workspace     = gemm_workspace_[i].device_workspace.data();
      gemm_workspace_[i].reduction_arguments.source        = gemm_workspace_[i].C->data();
      gemm_workspace_[i].reduction_arguments.destination   = gemm_workspace_[i].Computed->data();
      gemm_workspace_[i].reduction_arguments.alpha         = problem_.alpha.data();
      gemm_workspace_[i].reduction_arguments.beta          = problem_.beta.data();
      gemm_workspace_[i].reduction_arguments.pointer_mode  = library::ScalarPointerMode::kHost;
    }
  }

  //
  // Run the CUTLASS operation
  //

 // initialize gemm underlying operation to handle parallel reduction
  library::Operation const * underlying_operation = operation;

  if (problem_.split_k_mode == library::SplitKMode::kParallel) {
    if (!(underlying_operation = library::find_gemm_operation_for_parallel_reduction(operation))) {
      results_.back().disposition = Disposition::kFailed;
      return false;
    }
  }

  for (size_t i = 0; i < gemm_workspace_.size(); ++i) {
    cudaSetDevice(options.device.device_id(i));

    results_.back().status = underlying_operation->run(
     &gemm_workspace_[i].arguments,
     gemm_workspace_[i].host_workspace.data(),
     gemm_workspace_[i].device_workspace.data(),
     gemm_workspace_[i].stream);

    if (results_.back().status != Status::kSuccess) {
      results_.back().disposition = Disposition::kFailed;
      return false;
    }

    // Run parallel reduction kernel for parallel split_k_mode
    if (problem_.split_k_mode == library::SplitKMode::kParallel) {
      results_.back().status = reduction_op_->run(
        &gemm_workspace_[i].reduction_arguments,
        gemm_workspace_[i].reduction_host_workspace.data(),
        nullptr,
        gemm_workspace_[i].stream);

      if (results_.back().status != Status::kSuccess) {
        results_.back().disposition = Disposition::kFailed;
        return false;
      }
    }
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
      auto const & gemm_desc = static_cast<library::GemmDescription const &>(operation->description());

      if (cublas_satisfies(gemm_desc) == Status::kSuccess) {

        // call cublas verification if supported
        for (size_t i = 0; i < gemm_workspace_.size(); ++i) {
          cudaSetDevice(options.device.device_id(i));
          verify_with_cublas_(
           options,
           report,
           device_context,
           operation,
           problem_space,
           problem,
           gemm_workspace_[i]);
        }
        }

      else {
        // set verification map for cublas to not supported
        results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kNotSupported;
      }
    }
#endif // #if CUTLASS_ENABLE_CUBLAS

    
    cutlass::library::RuntimeDatatype runtime_datatype_a = gemm_workspace_.front().arguments.runtime_input_datatype_a;
    cutlass::library::RuntimeDatatype runtime_datatype_b = gemm_workspace_.front().arguments.runtime_input_datatype_b;

    bool is_runtime_datatype_a = runtime_datatype_a != cutlass::library::RuntimeDatatype::kStatic;
    bool is_runtime_datatype_b = runtime_datatype_b != cutlass::library::RuntimeDatatype::kStatic;

    assert(is_runtime_datatype_a == is_runtime_datatype_b && "runtime datatype should be both dynamic or static.");
    

    library::GemmDescription const &gemm_desc =
      static_cast<library::GemmDescription const &>(operation->description());


    cutlass::library::NumericTypeID element_A = gemm_desc.A.element;
    cutlass::library::NumericTypeID element_B = gemm_desc.B.element;
    
    if (is_runtime_datatype_a) {
      element_A = cutlass::library::dynamic_datatype_to_id(runtime_datatype_a);
    }

    if (is_runtime_datatype_b) {
      element_B = cutlass::library::dynamic_datatype_to_id(runtime_datatype_b);
    }
    

    bool verification_status = verify_with_reference_(options, report, device_context, operation, problem_space, problem, element_A, element_B);

    // Update disposition to worst case verification outcome among all
    // verification providers which are supported
    bool is_any_verification_run_passed = false;
    for (auto &m : results_.back().verification_map) {
      if (m.second == Disposition::kFailed || m.second == Disposition::kIncorrect) {
        results_.back().disposition = m.second;
        return true;
      }
      if (!is_any_verification_run_passed && m.second == Disposition::kPassed) {
        is_any_verification_run_passed = true;
      }
    }

    if (is_any_verification_run_passed) {
      results_.back().disposition = Disposition::kPassed;
    }
  }

  // if verification.required is set, then return success iff at least one ref-check was run
  if (options.verification.required) {
    bool did_any_verification_run = false;
    for (auto provider : options.verification.providers) {
      did_any_verification_run |= (Disposition::kNotRun != results_.back().verification_map[provider]);
    }

    if (not did_any_verification_run) {
      results_.back().status = Status::kErrorNotSupported;
      return false;
    }
  }

  // Return true means continue profiling
  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Verifies CUTLASS against references
bool GemmOperationProfiler::verify_with_cublas_(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem,
  GemmWorkspace &gemm_workspace_) {

#if CUTLASS_ENABLE_CUBLAS

  library::GemmDescription const &gemm_desc =
    static_cast<library::GemmDescription const &>(operation->description());

  //
  // Construct cuBLAS operators
  //

  CublasLtCreate handle;
  cublasStatus_t status = handle.get_cublaslt_create_status();

  if (status != CUBLAS_STATUS_SUCCESS) {
    results_.back().verification_map[library::Provider::kCUBLAS] = get_cutlass_disposition(status);
    return true;
  }


  //
  // Initialize state
  //

  try {

    //
    // Construct dispatcher to cublasGemmEx()
    //

    // Initialize structure containing GEMM arguments
    gemm_workspace_.arguments.A = gemm_workspace_.A->data();
    gemm_workspace_.arguments.batch_stride_A = gemm_workspace_.A->batch_stride();
    gemm_workspace_.arguments.B = gemm_workspace_.B->data();
    gemm_workspace_.arguments.batch_stride_B = gemm_workspace_.B->batch_stride();
    gemm_workspace_.arguments.C = gemm_workspace_.Reference->data();
    gemm_workspace_.arguments.batch_stride_C = gemm_workspace_.Reference->batch_stride();
    gemm_workspace_.arguments.D = gemm_workspace_.Reference->data();
    gemm_workspace_.arguments.batch_stride_D = gemm_workspace_.Reference->batch_stride();
    gemm_workspace_.arguments.alpha = problem_.alpha.data();
    gemm_workspace_.arguments.beta = problem_.beta.data();
    gemm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

    detail::cublasLtGemmExDispatcher gemm_op(
      gemm_desc,
      gemm_workspace_.configuration,
      gemm_workspace_.arguments
    );

    gemm_op.initialize_cublaslt();

    if(!gemm_op.get_cublaslt_algo(handle, AlgorithmMode::kDefault)){
      return true;
    }

    if (gemm_op.status != Status::kSuccess) {
      results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kNotRun;
      return true;
    }

    status = gemm_op(handle);

    // Handle errors
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "cublasLt Verification run failed with status : " << cublasLtGetStatusName(status) << "\n";
      results_.back().verification_map[library::Provider::kCUBLAS] = get_cutlass_disposition(status);
      return true;
    }

    results_.back().status = Status::kSuccess;

    //
    // Verify results
    //

    results_.back().verification_map[library::Provider::kCUBLAS] = compare_tensors(
      options,
      *gemm_workspace_.Computed,
      *gemm_workspace_.Reference,
      gemm_workspace_.Computed->batch_stride()
    );

    // Save workspace if incorrect
    if (options.verification.save_workspace == SaveWorkspace::kIncorrect &&
      results_.back().verification_map[library::Provider::kCUBLAS] == Disposition::kIncorrect) {

      save_workspace(
        device_context,
        options,
        gemm_desc,
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

/// Verifies CUTLASS against host and device references
bool GemmOperationProfiler::verify_with_reference_(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem,
  cutlass::library::NumericTypeID element_A,
  cutlass::library::NumericTypeID element_B)
{
  library::GemmDescription const &gemm_desc =
    static_cast<library::GemmDescription const &>(operation->description());

  //
  // Initialize state
  //
  for (auto provider : options.verification.providers) {

    // Skip providers that are not enabled
    if (!options.verification.provider_enabled(provider)) {
      continue;
    }

    for (size_t i = 0; i < gemm_workspace_.size(); ++i) {
      cudaSetDevice(options.device.device_id(i));

      void *ptr_A = gemm_workspace_[i].A->data();
      void *ptr_B = gemm_workspace_[i].B->data();
      void *ptr_C = gemm_workspace_[i].C->data();
      void *ptr_D = gemm_workspace_[i].Reference->data();

      cutlass::library::NumericTypeID element_A_for_reference = element_A;
      cutlass::library::NumericTypeID element_B_for_reference = element_B;
      if (gemm_workspace_[i].arguments.is_sm90_mixed_dtype) {
        // Dequantized tensor has the same shape of the narrow data type tensor,
        // and the same data type as the wide data type tensor
        if (gemm_workspace_[i].arguments.wider_operand == cutlass::library::Sm90MixedInputWiderOperand::A) {
          ptr_B = gemm_workspace_[i].dequantized_AB->data();
          element_B_for_reference = element_A;
        }
        else {
          ptr_A = gemm_workspace_[i].dequantized_AB->data();
          element_A_for_reference = element_B;
        }
      }

      // To support the host-side reference, conditionally allocate and
      // copy tensors to host memory.
      std::vector<uint8_t> host_data_A;
      std::vector<uint8_t> host_data_B;
      std::vector<uint8_t> host_data_C;
      std::vector<uint8_t> host_data_D;

      if (provider == library::Provider::kReferenceHost) {

        host_data_A.resize(gemm_workspace_[i].A->bytes());
        ptr_A = host_data_A.data();
        gemm_workspace_[i].A->copy_to_host(ptr_A);

        host_data_B.resize(gemm_workspace_[i].B->bytes());
        ptr_B = host_data_B.data();
        gemm_workspace_[i].B->copy_to_host(ptr_B);

        host_data_C.resize(gemm_workspace_[i].C->bytes());
        ptr_C = host_data_C.data();
        gemm_workspace_[i].C->copy_to_host(ptr_C);

        host_data_D.resize(gemm_workspace_[i].Reference->bytes());
        ptr_D = host_data_D.data();
      }

      //
      // Launch
      //

      library::Handle handle;

      handle.set_provider(provider);

      Status status = handle.gemm_universal(
        problem_.mode,
        gemm_workspace_[i].configuration.problem_size.m(),
        gemm_workspace_[i].configuration.problem_size.n(),
        gemm_workspace_[i].configuration.problem_size.k(),
        
        gemm_workspace_[i].configuration.cluster_shape.m(),
        gemm_workspace_[i].configuration.cluster_shape.n(),
        gemm_workspace_[i].configuration.cluster_shape.k(),
        gemm_workspace_[i].configuration.cluster_shape_fallback.m(),
        gemm_workspace_[i].configuration.cluster_shape_fallback.n(),
        gemm_workspace_[i].configuration.cluster_shape_fallback.k(),
        
        gemm_desc.tile_description.math_instruction.element_accumulator,
        gemm_desc.element_epilogue,

        problem_.alpha.data(),

        element_A_for_reference,
        gemm_desc.A.layout,
        gemm_desc.transform_A,
        ptr_A,
        int(gemm_workspace_[i].configuration.lda),

        element_B_for_reference,
        gemm_desc.B.layout,
        gemm_desc.transform_B,
        ptr_B,
        int(gemm_workspace_[i].configuration.ldb),

        problem_.beta.data(),

        gemm_desc.C.element,
        gemm_desc.C.layout,
        ptr_C,
        int(gemm_workspace_[i].configuration.ldc),

        gemm_desc.D.element,
        gemm_desc.D.layout,
        ptr_D,
        int(gemm_workspace_[i].configuration.ldd),

        gemm_workspace_[i].configuration.batch_count,
        gemm_workspace_[i].A->batch_stride(),
        gemm_workspace_[i].B->batch_stride(),
        gemm_workspace_[i].C->batch_stride(),
        gemm_workspace_[i].Reference->batch_stride());

      if (status != Status::kSuccess) {
        results_.back().verification_map[provider] = Disposition::kNotRun;
        continue;
      }
      results_.back().status = status;

      if (provider == library::Provider::kReferenceHost) {
        gemm_workspace_[i].Reference->copy_from_host(ptr_D);
      }

      //
      // Verify results
      //

      results_.back().verification_map[provider] = compare_tensors(
        options,
        *gemm_workspace_[i].Computed,
        *gemm_workspace_[i].Reference,
        gemm_workspace_[i].Computed->batch_stride()
      );

      // Save workspace if incorrect
      if (options.verification.save_workspace == SaveWorkspace::kIncorrect &&
        results_.back().verification_map[provider] == Disposition::kIncorrect) {

        save_workspace(
          device_context,
          options,
          gemm_desc,
          library::Provider::kCUTLASS,
          provider);
        }
    }
  }

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Measures performance results
bool GemmOperationProfiler::profile(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    // Case when we either screen the best performance number of kernels with or without a fixed problem shape fed in.
    if (options.profiling.enable_kernel_performance_search || options.profiling.enable_best_kernel_for_fixed_shape) {
      library::GemmDescription const &operation_desc =
        static_cast<library::GemmDescription const &>(operation->description());

      auto cluster_shape = operation_desc.tile_description.cluster_shape;
      bool is_dynamic_cluster_enabled = cluster_shape.m() == 0 || cluster_shape.n() == 0 || cluster_shape.k() == 0;

      // Helper function wrapping up performance test with flexible parameters.
      auto initialize_and_profile = [&](
        PerformanceResult const &result,
        gemm::GemmCoord const &problem_shape,
        std::array<int64_t, 3> const &leading_dim,
        std::array<int64_t, 3> const &preferred_cluster,
        std::array<int64_t, 3> const &fallback_cluster,
        cutlass::library::RasterOrder const &raster_order,
        int swizzle_size) -> std::optional<PerformanceResult> {

        for (size_t i = 0; i < gemm_workspace_.size(); ++i) {
          // Initialize structure containing GEMM arguments
          auto& workspace = gemm_workspace_[i];
          workspace.arguments.A = workspace.A->data();
          workspace.arguments.B = workspace.B->data();
          workspace.arguments.C = workspace.C->data();
          workspace.arguments.D = workspace.Computed->data();
          workspace.arguments.alpha = problem_.alpha.data();
          workspace.arguments.beta = problem_.beta.data();
          workspace.arguments.pointer_mode = library::ScalarPointerMode::kHost;
          workspace.arguments.batch_stride_A = workspace.A->batch_stride();
          workspace.arguments.batch_stride_B = workspace.B->batch_stride();
          workspace.arguments.batch_stride_C = workspace.C->batch_stride();
          workspace.arguments.batch_stride_D = workspace.Computed->batch_stride();

          if (problem_.split_k_mode == library::SplitKMode::kParallel) {
            workspace.arguments.D = workspace.device_workspace.data();
            workspace.arguments.alpha = problem_.alpha_one.data();
            workspace.arguments.beta = problem_.beta_zero.data();

            workspace.reduction_arguments.workspace = workspace.device_workspace.data();
            workspace.reduction_arguments.source = workspace.C->data();
            workspace.reduction_arguments.destination = workspace.Computed->data();
            workspace.reduction_arguments.alpha = problem_.alpha.data();
            workspace.reduction_arguments.beta = problem_.beta.data();
            workspace.reduction_arguments.pointer_mode = library::ScalarPointerMode::kHost;
          }

          update_workspace_(workspace, problem_shape, leading_dim, preferred_cluster, fallback_cluster, raster_order, swizzle_size, is_dynamic_cluster_enabled);

          const auto can_implement = operation->can_implement(&workspace.configuration, &workspace.arguments);
          if (can_implement != Status::kSuccess) {
            return std::nullopt;  // Return nullopt to indicate failure
          }
          library::Operation const* underlying_operation = operation;
          cudaSetDevice(options.device.device_id(i));
          uint64_t workspace_size = underlying_operation->get_host_workspace_size(&workspace.configuration);
          workspace.host_workspace.resize(workspace_size, 0);

          workspace_size = underlying_operation->get_device_workspace_size(&workspace.configuration,
                                                                &workspace.arguments);
          
          bool is_sparse = operation_desc.tile_description.math_instruction.opcode_class == cutlass::library::OpcodeClassID::kSparseTensorOp;
          if (is_sparse) {
            // sparse gemm get_device_workspace_size() only return device workspace size per iteration
            // Needs to multiply it w/ number of iteration
            workspace_size *= workspace.problem_count;
          }

          workspace.device_workspace.reset(library::NumericTypeID::kU8, workspace_size);

          Status status = Status::kSuccess;

          if (is_sparse) {
            uint8_t* profiler_workspaces[1];
            profiler_workspaces[0] = reinterpret_cast<uint8_t*>(workspace.A->data());
            // Sparse operations have a different initialize interface.
            // initialize_with_profiler_workspace converts mxk tensorA to compressed mxk/sp tensorA and the tensorE
            auto modifiable_underlying_op = const_cast<library::Operation*>(underlying_operation);
            status = modifiable_underlying_op->initialize_with_profiler_workspace(
              &workspace.configuration,
              workspace.host_workspace.data(),
              workspace.device_workspace.data(),
              profiler_workspaces,
              workspace.problem_count,
              workspace.stream);
          }
          else {
            status = underlying_operation->initialize(
              &workspace.configuration,
              workspace.host_workspace.data(),
              workspace.device_workspace.data(),
              workspace.stream);
          }

          if (status != Status::kSuccess) {
            return std::nullopt;  // Return nullopt to indicate failure
          }

        }

        PerformanceResult curr_result(result);
        update_result_(curr_result, operation_desc, problem_space, problem_shape, raster_order, preferred_cluster, fallback_cluster, swizzle_size, is_dynamic_cluster_enabled);

        curr_result.status = profile_cutlass_(
          curr_result,
          options,
          operation,
          nullptr,
          nullptr,
          nullptr
        );

        return curr_result;
      };

      // Helper function to test validity of fallback cluster shapes and preferred cluster shapes.
      auto is_valid_dynamic_cluster_shape = [](const std::array<int64_t, 3>& preferred_cluster, const std::array<int64_t, 3>& fallback_cluster) {
        for (size_t i = 0; i < 3; ++i) {
          if (preferred_cluster[i] % fallback_cluster[i] != 0) {
            return false;
          }
        }
        return true;
      };

      // Helper function to select the best performance number among a list.
      auto select_best_candidate = [&](std::vector<PerformanceResult> &candidates) {
        assert(!candidates.empty() && "Candidates vector should not be empty");
        auto best_iter = std::max_element(
          candidates.begin(), candidates.end(),
          [](PerformanceResult const &a, PerformanceResult const &b) {
            return a.gflops_per_sec() < b.gflops_per_sec();
          }
        );
        assert(best_iter != candidates.end() && "No candidate found despite non-empty candidates vector");
        results_.push_back(std::move(*best_iter));
      };

      std::vector<PerformanceResult> candidates;
      PerformanceResult result_base = results_.back();
      results_.pop_back();

      std::vector<std::array<int64_t, 3>> preferred_clusters;
      std::vector<std::array<int64_t, 3>> fallback_clusters;

      // Only loop over built-in cluster shape lists for dynamic cluster kernels
      // and for kernels that can leverage the dynamic cluster feature.
      if (is_dynamic_cluster_enabled) {
        preferred_clusters = this->problem_.preferred_clusters;
        fallback_clusters = this->problem_.fallback_clusters;
      } 
      else {
        preferred_clusters = {{int(problem_.cluster_m), int(problem_.cluster_n), int(problem_.cluster_k)}};
        fallback_clusters = {{int(problem_.cluster_m_fallback), int(problem_.cluster_n_fallback), int(problem_.cluster_k_fallback)}};
      }

      for (auto preferred_cluster : preferred_clusters) {
        for (auto fallback_cluster : fallback_clusters) {
          if (is_dynamic_cluster_enabled && !is_valid_dynamic_cluster_shape(preferred_cluster, fallback_cluster)) {
            continue;
          }
          for (auto swizzle_size : this->problem_.swizzle_sizes) {
            for (auto raster_order : this->problem_.raster_orders) {
              // With the fixed shape option turned on, only a specific problem shape is tested.
              if (options.profiling.enable_best_kernel_for_fixed_shape) {
                this->problem_.problem_sizes = {{int(this->problem_.m), int(this->problem_.n), int(this->problem_.k)}};
                this->problem_.leading_dims = {{this->problem_.lda, this->problem_.ldb, this->problem_.ldc}};
              }

              for (int i = 0; i < int(this->problem_.problem_sizes.size()); i++) {
                gemm::GemmCoord problem_shape = problem_.problem_sizes[i];
                std::array<int64_t, 3> leading_dim = problem_.leading_dims[i];
                auto result_opt = initialize_and_profile(result_base, problem_shape, leading_dim, preferred_cluster, fallback_cluster, raster_order, swizzle_size);
                  
                if (result_opt) {  // Only add valid results
                  candidates.push_back(*result_opt);
                }

              }

            }// for raster_order
          }// for swizzle_size
        }// for fallback_cluster
      }// for swizzle_size

      if (candidates.empty()) {
        return false;
      }

      select_best_candidate(candidates);
    }
    // Basic case where we benchmark input parameters only.
    else {
      results_.back().status = profile_cutlass_(
        results_.back(),
        options,
        operation,
        nullptr,
        nullptr,
        nullptr
      );
    }

  }
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Method to profile a CUTLASS Operation
Status GemmOperationProfiler::profile_cutlass_(
  PerformanceResult &result,
  Options const &options,
  library::Operation const *operation,
  void *,
  void *,
  void *) {

  // initialize gemm underlying operation to handle parallel reduction
  library::Operation const * underlying_operation = operation;

  if (problem_.split_k_mode == library::SplitKMode::kParallel) {
    if (!(underlying_operation = library::find_gemm_operation_for_parallel_reduction(operation))) {
      return Status::kErrorNotSupported;
    }
  }

  auto launch_gemm = [&](int dev_id, cudaStream_t stream, int iteration) {
    int problem_idx = (iteration % gemm_workspace_[dev_id].problem_count) * problem_.batch_count;

    gemm_workspace_[dev_id].arguments.A = gemm_workspace_[dev_id].A->batch_data(problem_idx);
    gemm_workspace_[dev_id].arguments.B = gemm_workspace_[dev_id].B->batch_data(problem_idx);
    gemm_workspace_[dev_id].arguments.C = gemm_workspace_[dev_id].C->batch_data(problem_idx);
    gemm_workspace_[dev_id].arguments.D = gemm_workspace_[dev_id].Computed->batch_data(problem_idx);

      if (gemm_workspace_[dev_id].arguments.is_sm90_mixed_dtype) {
        // Scale, zero, and dequantized tensors are already generated in
        // verify_cutlass(), no need to re-generate them in profiling
        gemm_workspace_[dev_id].arguments.generate_scale_and_zero = false;
        gemm_workspace_[dev_id].arguments.generate_dequantized_AB = false;
      }

    if (problem_.split_k_mode == library::SplitKMode::kParallel) {
      gemm_workspace_[dev_id].arguments.D                     = gemm_workspace_[dev_id].device_workspace.data();

      gemm_workspace_[dev_id].reduction_arguments.workspace   = gemm_workspace_[dev_id].device_workspace.data();
      gemm_workspace_[dev_id].reduction_arguments.source      = gemm_workspace_[dev_id].C->batch_data(problem_idx);
      gemm_workspace_[dev_id].reduction_arguments.destination = gemm_workspace_[dev_id].Computed->batch_data(problem_idx);
    }

    // Execute the CUTLASS operation
    Status status = underlying_operation->run(
      &gemm_workspace_[dev_id].arguments,
      gemm_workspace_[dev_id].host_workspace.data(),
      gemm_workspace_[dev_id].device_workspace.data(),
      stream);

    if (status != Status::kSuccess) {
      return status;
    }

    // Run parallel reduction kernel for parallel split_k_mode
    if (problem_.split_k_mode == library::SplitKMode::kParallel) {
      status = reduction_op_->run(
        &gemm_workspace_[dev_id].reduction_arguments,
        gemm_workspace_[dev_id].reduction_host_workspace.data(),
        nullptr,
        gemm_workspace_[dev_id].stream);

      if (status != Status::kSuccess) {
        return status;
      }
    }
    return Status::kSuccess;
  };

  std::vector<cudaStream_t> streams(gemm_workspace_.size());
  for (size_t i = 0; i < streams.size(); i++) {
    streams[i] = gemm_workspace_[i].stream;
  }
  return profile_kernel_(result, options, launch_gemm, streams);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
