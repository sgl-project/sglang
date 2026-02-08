/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <bitset>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <regex>

#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"
#include "cutlass/profiler/grouped_gemm_operation_profiler.h"
#include "cutlass/library/handle.h"
#include "cutlass/library/library.h"
#include "cutlass/library/operation_table.h"
#include "cutlass/library/singleton.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
std::vector<std::vector<int>> parseProblemSizes(std::string const& input) {
  // input must be of the form:
  //  `[m0xn0xk0][m1xn1xk1]` where 0, 1 are the group indexes
  std::stringstream ss(input);
  std::string token;
  std::vector<std::vector<int>> result;
  while (std::getline(ss, token, ']')) {
    std::stringstream ss(token);
    std::string token;
    ss.get(); // discard '['
    std::getline(ss, token, 'x');
    auto m = std::stoi(token);
    std::getline(ss, token, 'x');
    auto n = std::stoi(token);
    std::getline(ss, token);
    auto k = std::stoi(token);
    result.push_back({m, n, k});
  }
  return result;
}
} // namespace

namespace cutlass {
namespace profiler {

GroupedGemmOperationProfiler::GroupedGemmOperationProfiler(Options const& options)
    : OperationProfiler(
        options,
        library::OperationKind::kGroupedGemm,
        {{ArgumentTypeID::kEnumerated,
          {"gemm_kind"},
          "Variant of GEMM (universal, gemm, planar_complex, planar_complex_array)"},
         {ArgumentTypeID::kInteger,
          {"m", "problem-size::m"},
          "M dimension of the GEMM problem space (for all groups)"},
         {ArgumentTypeID::kInteger,
          {"n", "problem-size::n"},
          "N dimension of the GEMM problem space (for all groups)"},
         {ArgumentTypeID::kInteger,
          {"k", "problem-size::k"},
          "K dimension of the GEMM problem space (for all groups)"},
         {ArgumentTypeID::kInteger,
          {"num_groups"},
          "If m,n,k are specified, run a grouped GEMM with this number of groups, where each GEMM "
          "uses the same m,n,k values."},
         {ArgumentTypeID::kTensor, {"A"}, "Tensor storing the A operand"},
         {ArgumentTypeID::kTensor, {"B"}, "Tensor storing the B operand"},
         {ArgumentTypeID::kTensor, {"C"}, "Tensor storing the C operand"},
         {ArgumentTypeID::kTensor, {"D"}, "Tensor storing the D output"},
         {ArgumentTypeID::kScalar,
          {"alpha", "epilogue::alpha"},
          "Epilogue scalar alpha (applied to all GEMMs in group)."},
         {ArgumentTypeID::kScalar,
          {"beta", "epilogue::beta"},
          "Epilogue scalar beta (applied to all GEMMs in group)."},
         {ArgumentTypeID::kEnumerated, {"runtime_input_datatype_a", "runtime-input-datatype::a"},
          "Runtime datatype (e4m3, e5m2, e3m2, e2m3, e2m1)"}, 
         {ArgumentTypeID::kEnumerated, {"runtime_input_datatype_b", "runtime-input-datatype::b"},
          "Runtime datatype (e4m3, e5m2, e3m2, e2m3, e2m1)"}, 
         {ArgumentTypeID::kEnumerated, {"raster_order", "raster-order"},
          "Raster order (heuristic, along_n, along_m)"},
         {ArgumentTypeID::kInteger, {"swizzle_size", "swizzle-size"}, "Size to swizzle"},
         {ArgumentTypeID::kEnumerated, {"use_pdl", "use_pdl"}, "Use PDL (true, false)"},
         {ArgumentTypeID::kScalar,
          {"problem-sizes"},
          "MxNxK Problem sizes for the grouped GEMM, where a group is enclosed by `[]`. E.g. "
          "--problem-sizes='[m1xn1xk1][m2xn2xk2]'"},
         {ArgumentTypeID::kScalar,
          {"problem-sizes-file"},
          "File containing grouped GEMM problem sizes, where each line represents a group whose "
          "GEMM dimensions are 'mxnxk'."}},
        {library::Provider::kReferenceDevice}) {

  description_ = "      Grouped matrix-matrix product. D[g] = alpha[g] * A[g] * B[g] + beta[g] * "
                 "C[g] for g in [0, num_groups)";
}

GroupedGemmOperationProfiler::~GroupedGemmOperationProfiler() {}

void GroupedGemmOperationProfiler::print_usage(std::ostream& out) const {
  OperationProfiler::print_usage(out);
}

void GroupedGemmOperationProfiler::print_examples(std::ostream& out) const {

  out
    << "\nExamples:\n\n"
    << "Profile a particular problem size (explicit shapes):\n"
    << "  $ cutlass_profiler --operation=GroupedGemm --problem-sizes='[1024x1024x128][16x8x8]'\n\n"

    << "Profile a particular problem size (same M, N, K for all groups):\n"
    << "  $ cutlass_profiler --operation=GroupedGemm --m=16 --n=32 --k=64 --num_groups=8'\n\n"

    << "Profile a particular problem size from a file:\n"
    << "  $ cutlass_profiler --operation=GroupedGemm --problem-sizes-file=shapes.txt\n\n"

    << "Schmoo over problem size and beta:\n"
    << "  $ cutlass_profiler --operation=GroupedGemm --problem-sizes='[8x8x8],[16x8x16][32x32x32]' "
       "--beta=0,1,2.5\n\n"

    << "Schmoo over accumulator types:\n"
    << "  $ cutlass_profiler --operation=GroupedGemm --accumulator-type=f16,f32\n\n"

    << "Run when A is f16 with column-major and B is any datatype with row-major (For column "
       "major, use column, col, or n. For row major use, row or t):\n"
    << "  $ cutlass_profiler --operation=GroupedGemm --A=f16:column --B=*:row\n\n"

    << "Using various input value distribution:\n"
    << "  $ cutlass_profiler --operation=GroupedGemm --dist=uniform,min:0,max:3\n"
    << "  $ cutlass_profiler --operation=GroupedGemm --dist=gaussian,mean:0,stddev:3\n"
    << "  $ cutlass_profiler --operation=GroupedGemm --dist=sequential,start:0,delta:1\n\n"

    << "Test your changes to gemm kernels with a quick functional test and save results in "
       "functional-test.csv:\n"
    << " $ cutlass_profiler  --operation=Gemm \\ \n"
    << "   --problem-sizes='[8x8x8][5x10x5],[16x8x16][32x32x32]' \\ \n"
    << "   --beta=0,1,2 --profiling-iterations=1 \\ \n"
    << "   --providers=cutlass --output=functional-test.csv\n\n";
}

Status GroupedGemmOperationProfiler::GroupedGemmProblem::parse(
  library::GroupedGemmDescription const& operation_desc,
  ProblemSpace const& problem_space,
  ProblemSpace::Problem const& problem) {

  this->mode = library::GemmUniversalMode::kGrouped;

  std::bitset<3> args_exist;
  std::string problem_sizes_str;
  args_exist[0] = arg_as_string(problem_sizes_str, "problem-sizes", problem_space, problem);
  int m, n, k;
  args_exist[1] = arg_as_int(m, "m", problem_space, problem) &&
                  arg_as_int(n, "n", problem_space, problem) &&
                  arg_as_int(k, "k", problem_space, problem);
  std::string problem_file;
  args_exist[2] = arg_as_string(problem_file, "problem-sizes-file", problem_space, problem);

  if (args_exist.count() == 0) {
    int num_groups = 8;
    problem_sizes.resize(num_groups);
    problem_sizes_3x.resize(num_groups);
    int m0 = 16;
    int n0 = 32;
    int k0 = 64;
    for (int i = 0; i < num_groups; i++) {
      auto m = m0 * (i + 1);
      auto n = n0 * (i + 1);
      auto k = k0 * (i + 1);
      problem_sizes[i] = {m, n, k};
      problem_sizes_3x[i] = {m, n, k};
    }
  }
  else if (args_exist.count() > 1) {
    std::cerr
      << "Exactly one of --problem-sizes, --problem-sizes-file, or --m --n --k may be specified.\n";
    return Status::kErrorInvalidProblem;
  }
  // --problem-sizes path
  else if (args_exist[0]) {
    auto problems = parseProblemSizes(problem_sizes_str);
    auto num_groups = problems.size();
    problem_sizes.resize(num_groups);
    problem_sizes_3x.resize(num_groups);
    for (size_t i = 0; i < num_groups; i++) {
      auto m = problems[i][0];
      auto n = problems[i][1];
      auto k = problems[i][2];
      problem_sizes[i] = {m, n, k};
      problem_sizes_3x[i] = {m, n, k};
    }
  }
  // m, n, k path
  else if (args_exist[1]) {
    int num_groups;
    if (!arg_as_int(num_groups, "num_groups", problem_space, problem)) {
      std::cerr << "num_groups must be specified if --m --n and --k are set.\n";
      return Status::kErrorInvalidProblem;
    }
    problem_sizes.resize(num_groups);
    problem_sizes_3x.resize(num_groups);
    for (int i = 0; i < num_groups; i++) {
      problem_sizes[i] = {m, n, k};
      problem_sizes_3x[i] = {m, n, k};
    }
  }
  // --problem-sizes-file path
  else if (args_exist[2]) {
    std::ifstream file(problem_file);
    if (!file.good()) {
      throw std::runtime_error("Failed to open file: " + problem_file);
    }
    // clear the problem sizes and 3x problem sizes from previous operation
    problem_sizes.clear();
    problem_sizes_3x.clear();

    for (std::string line; std::getline(file, line);) {
      std::istringstream iss(line);

      int m, n, k;
      char sep1, sep2;
      std::string remaining;

      if (iss >> m >> sep1 >> n >> sep2 >> k && sep1 == 'x' && sep2 == 'x' && !(iss >> remaining)) {
        problem_sizes.emplace_back(m, n, k);
        problem_sizes_3x.emplace_back(m, n, k);
      }
      else {
        throw std::runtime_error(
          "Invalid format in line: " + line + ". Each line in file expected to be 'mxnxk'.");
      }
    }
  }

  if (!arg_as_int(this->cluster_m, "cluster_m", problem_space, problem)) {
    // default value
    this->cluster_m = std::string(operation_desc.gemm.name).find("_2sm") != std::string::npos ? 2 : 1;
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

  this->mode = library::GemmUniversalMode::kGrouped;

  if (!tensor_description_satisfies(operation_desc.gemm.A, "A", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!tensor_description_satisfies(operation_desc.gemm.B, "B", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!tensor_description_satisfies(operation_desc.gemm.C, "C", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!tensor_description_satisfies(operation_desc.gemm.D, "D", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!arg_as_bool(this->use_pdl, "use_pdl", problem_space, problem)) {
    // default value
    this->use_pdl = false;
  }
  
  if (!arg_as_RuntimeDatatype(this->runtime_input_datatype_a, "runtime_input_datatype_a", problem_space, problem)) {
    // default value
    this->runtime_input_datatype_a = cutlass::library::RuntimeDatatype::kStatic;
  }

  if (!arg_as_RuntimeDatatype(this->runtime_input_datatype_b, "runtime_input_datatype_b", problem_space, problem)) {
    // default value
    this->runtime_input_datatype_b = cutlass::library::RuntimeDatatype::kStatic;
  }

  if (!arg_as_int(this->swizzle_size, "swizzle_size", problem_space, problem)) {
    // default value
    this->swizzle_size = 1;
  }

  if (!arg_as_RasterOrder(this->raster_order, "raster_order", problem_space, problem)) {
    // default value
    this->raster_order = library::RasterOrder::kHeuristic;
  }

  if (!arg_as_scalar(
        this->alpha,
        operation_desc.gemm.element_epilogue,
        "alpha",
        problem_space,
        problem)) {

    if (!cast_from_double(this->alpha, operation_desc.gemm.element_epilogue, 1)) {
      return Status::kErrorInternal;
    }
  }

  if (!arg_as_scalar(
        this->beta,
        operation_desc.gemm.element_epilogue,
        "beta",
        problem_space,
        problem)) {

    if (!cast_from_double(this->beta, operation_desc.gemm.element_epilogue, 0)) {
      return Status::kErrorInternal;
    }
  }

  auto num_groups = problem_sizes.size();
  this->lda.resize(num_groups);
  this->ldb.resize(num_groups);
  this->ldc.resize(num_groups);
  for (size_t group_idx = 0; group_idx < num_groups; group_idx++) {
    this->lda[group_idx] = DeviceAllocation::get_packed_layout(
                             operation_desc.gemm.A.layout,
                             {int(this->m(group_idx)), int(this->k(group_idx))})
                             .front();

    this->ldb[group_idx] = DeviceAllocation::get_packed_layout(
                             operation_desc.gemm.B.layout,
                             {int(this->k(group_idx)), int(this->n(group_idx))})
                             .front();

    this->ldc[group_idx] = DeviceAllocation::get_packed_layout(
                             operation_desc.gemm.C.layout,
                             {int(this->m(group_idx)), int(this->n(group_idx))})
                             .front();
  }

  // instantiation for exploration profiling
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

/// Total number of bytes loaded
int64_t GroupedGemmOperationProfiler::GroupedGemmProblem::bytes(
  library::GroupedGemmDescription const& operation_desc) const {
  // Input bytes read and Output bytes written for the gemm problem
  int64_t bytes = 0;
  for (size_t group_idx = 0, num_groups = problem_sizes.size(); group_idx < num_groups;
       group_idx++) {

    // If M = 0 or N = 0, no tiles are scheduled and no bytes are loaded for the group
    if (m(group_idx) * n(group_idx) == 0) {
      continue;
    }

    bytes +=
      int64_t(library::sizeof_bits(operation_desc.gemm.A.element) * m(group_idx) / 8) * k(group_idx) +
      int64_t(library::sizeof_bits(operation_desc.gemm.B.element) * n(group_idx) / 8) * k(group_idx) +
      int64_t(library::sizeof_bits(operation_desc.gemm.C.element) * m(group_idx) / 8) * n(group_idx);

    // Set is_beta_zero true if beta is zero
    bool is_beta_zero = std::all_of(beta.begin(), beta.end(), [](uint8_t i) { return i == 0; });
    // Output bytes read for the gemm problem for non-zero beta values
    if (!is_beta_zero) {
      bytes +=
        int64_t(library::sizeof_bits(operation_desc.gemm.C.element) * m(group_idx) / 8) * n(group_idx);
    }
  }

  return bytes;
}

/// Total number of flops computed
int64_t GroupedGemmOperationProfiler::GroupedGemmProblem::flops(
  library::GroupedGemmDescription const& operation_desc) const {
  int64_t flops_ = 0;
  for (size_t group_idx = 0, num_groups = problem_sizes.size(); group_idx < num_groups;
       group_idx++) {
    flops_ +=
      (int64_t(m(group_idx)) * n(group_idx) * k(group_idx) + m(group_idx) * n(group_idx)) * 2;
  }

  // complex-valued support
  switch (operation_desc.gemm.tile_description.math_instruction.math_operation) {
  case library::MathOperationID::kMultiplyAddComplex:
  case library::MathOperationID::kMultiplyAddComplexFastF32:
    flops_ *= 4;
    break;
  case library::MathOperationID::kMultiplyAddGaussianComplex:
    flops_ *= 3;
    break;

  default:
    break;
  }

  return flops_;
}

/// Initializes a performance result
void GroupedGemmOperationProfiler::GroupedGemmProblem::initialize_result(
  PerformanceResult& result,
  library::GroupedGemmDescription const& operation_desc,
  ProblemSpace const& problem_space) {

  result.arguments.resize(problem_space.rank());

  set_argument(
    result,
    "gemm_kind",
    problem_space,
    library::to_string(operation_desc.gemm.gemm_kind));

  set_argument(
    result,
    "A",
    problem_space,
    std::string(library::to_string(operation_desc.gemm.A.element)) + ":" +
      library::to_string(operation_desc.gemm.A.layout));

  set_argument(
    result,
    "B",
    problem_space,
    std::string(library::to_string(operation_desc.gemm.B.element)) + ":" +
      library::to_string(operation_desc.gemm.B.layout));

  set_argument(
    result,
    "C",
    problem_space,
    std::string(library::to_string(operation_desc.gemm.C.element)) + ":" +
      library::to_string(operation_desc.gemm.C.layout));

  set_argument(
    result,
    "D",
    problem_space,
    std::string(library::to_string(operation_desc.gemm.D.element)) + ":" +
      library::to_string(operation_desc.gemm.D.layout));

  {
    std::stringstream ss;
    ss << "'";
    for (auto const& problem_size : problem_sizes) {
      ss << "[";
      auto m = problem_size[0];
      auto n = problem_size[1];
      auto k = problem_size[2];
      ss << m << "x" << n << "x" << k;
      ss << "]";
    }
    ss << "'";
    set_argument(result, "problem-sizes", problem_space, ss.str());
  }

  auto cluster_shape = operation_desc.gemm.tile_description.cluster_shape;
  auto is_dynamic = cluster_shape.m() == 0 || cluster_shape.n() == 0 || cluster_shape.k() == 0;
  set_argument(result, "cluster_m", problem_space, is_dynamic ? this->cluster_m : cluster_shape.m());
  set_argument(result, "cluster_n", problem_space, is_dynamic ? this->cluster_n : cluster_shape.n());
  set_argument(result, "cluster_k", problem_space, is_dynamic ? this->cluster_k : cluster_shape.k());
  set_argument(result, "cluster_m_fallback", problem_space, cluster_m_fallback);
  set_argument(result, "cluster_n_fallback", problem_space, cluster_n_fallback);
  set_argument(result, "cluster_k_fallback", problem_space, cluster_k_fallback);

  set_argument(result, "raster_order", problem_space, library::to_string(raster_order));
  set_argument(result, "swizzle_size", problem_space, swizzle_size);
  set_argument(result, "use_pdl", problem_space, library::to_string(use_pdl));
  
  set_argument(result, "runtime_input_datatype_a", problem_space, library::to_string(runtime_input_datatype_a));
  set_argument(result, "runtime_input_datatype_b", problem_space, library::to_string(runtime_input_datatype_b));

  set_argument(
    result,
    "alpha",
    problem_space,
    library::lexical_cast(alpha, operation_desc.gemm.element_epilogue));

  set_argument(
    result,
    "beta",
    problem_space,
    library::lexical_cast(beta, operation_desc.gemm.element_epilogue));
}

void GroupedGemmOperationProfiler::update_workspace_and_result_(
  GroupedGemmWorkspace &gemm_workspace,
  PerformanceResult &result,
  ProblemSpace const &problem_space,
  cutlass::library::RasterOrder const &raster_order,
  std::array<int64_t, 3> const &preferred_cluster,
  std::array<int64_t, 3> const &fallback_cluster,
  int swizzle_size,
  bool is_dynamic_cluster_enabled
) {

  gemm_workspace.arguments.swizzle_size = swizzle_size;
  gemm_workspace.arguments.raster_order = raster_order;

  set_argument(result, "raster_order", problem_space, library::to_string(raster_order));
  set_argument(result, "swizzle_size", problem_space, swizzle_size);

  if (is_dynamic_cluster_enabled) {
    gemm_workspace.arguments.cluster_shape = {int(preferred_cluster[0]), int(preferred_cluster[1]), int(preferred_cluster[2])};
    gemm_workspace.arguments.cluster_shape_fallback = {int(fallback_cluster[0]), int(fallback_cluster[1]), int(fallback_cluster[2])};
    set_argument(result, "cluster_m", problem_space, preferred_cluster[0]);
    set_argument(result, "cluster_n", problem_space, preferred_cluster[1]);
    set_argument(result, "cluster_k", problem_space, preferred_cluster[2]);
    set_argument(result, "cluster_m_fallback", problem_space, fallback_cluster[0]);
    set_argument(result, "cluster_n_fallback", problem_space, fallback_cluster[1]);
    set_argument(result, "cluster_k_fallback", problem_space, fallback_cluster[2]);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Extracts the problem dimensions
Status GroupedGemmOperationProfiler::initialize_configuration(
  Options const& options,
  PerformanceReport& report,
  DeviceContext& device_context,
  library::Operation const* operation,
  ProblemSpace const& problem_space,
  ProblemSpace::Problem const& problem) {

  library::GroupedGemmDescription const& operation_desc =
    static_cast<library::GroupedGemmDescription const&>(operation->description());

  // We want to share the same operation profiler for any grouped gemm operation.
  // We distinguish between block scaled and non-block scaled operations by looking at the kernel
  // name, which tells us what reference kernel to use, which arguments to pass to the operation
  // etc. This avoids creating yet another OperationProfiler with a lot of boilerplate in it.

  std::string sf_tuple = "\\d+x\\d+";
  std::string datatypes_regex = "\\w?f\\d+|e\\dm\\d"; // bf16 | f16 | f32 | e4m3 | ...
  std::string blockwise_regex_string = sf_tuple + "(" +  datatypes_regex + ")x(" + 
                                       datatypes_regex + ")_" + sf_tuple + "(" + 
                                       datatypes_regex + ")x(" + datatypes_regex + ")";

  if (std::string(operation_desc.gemm.name).find("bstensor") != std::string::npos) {
    is_block_scaled = true;
    gemm_workspace_.block_scales = BlockScalingWorkspace{};
  }
  else if (std::regex_search(operation_desc.gemm.name, std::regex(blockwise_regex_string))) {
    is_blockwise = true;
    gemm_workspace_.block_scales = BlockScalingWorkspace{};
  }
  else {
    is_block_scaled = false;
    gemm_workspace_.block_scales = std::nullopt;
  }

  if (operation_desc.gemm.gemm_kind != library::GemmKind::kGrouped) {
    return Status::kErrorInvalidProblem;
  }

  Status status = problem_.parse(operation_desc, problem_space, problem);
  if (status != Status::kSuccess) {
    return status;
  }

  auto num_groups = problem_.problem_sizes.size();
  auto& config = gemm_workspace_.configuration;
  config.problem_count = num_groups;
  config.lda = problem_.lda.data();
  config.ldb = problem_.ldb.data();
  config.ldc = problem_.ldc.data();
  config.problem_sizes_3x_host = problem_.problem_sizes_3x.data();

  gemm_workspace_.arguments.swizzle_size = problem_.swizzle_size;
  gemm_workspace_.arguments.raster_order = problem_.raster_order;
  
  gemm_workspace_.arguments.runtime_input_datatype_a = problem_.runtime_input_datatype_a;
  gemm_workspace_.arguments.runtime_input_datatype_b = problem_.runtime_input_datatype_b;

  gemm_workspace_.arguments.use_pdl = problem_.use_pdl;

  cudaStreamCreateWithFlags(&gemm_workspace_.stream, cudaStreamNonBlocking);

  initialize_result_(this->model_result_, options, operation_desc, problem_space);

  return status;
}

/// Initializes the performance result
void GroupedGemmOperationProfiler::initialize_result_(
  PerformanceResult& result,
  Options const& options,
  library::GroupedGemmDescription const& operation_desc,
  ProblemSpace const& problem_space) {

  result.provider = library::Provider::kCUTLASS;
  result.disposition = Disposition::kNotRun;
  result.status = Status::kSuccess;
  result.operation_name = operation_desc.gemm.name;

  problem_.initialize_result(result, operation_desc, problem_space);

  OperationProfiler::initialize_result_(result, operation_desc, problem_space);

  result.bytes = problem_.bytes(operation_desc);
  result.flops = problem_.flops(operation_desc);
  result.runtime = 0;
  result.runtime_vector.resize(options.device.devices.size(), 0);

}

/// Initializes workspace
Status GroupedGemmOperationProfiler::initialize_workspace(
  Options const& options,
  PerformanceReport& report,
  DeviceContext& device_context,
  library::Operation const* operation,
  ProblemSpace const& problem_space,
  ProblemSpace::Problem const& problem) {

  if (options.device.devices.size() != 1) {
    throw std::runtime_error("This operation profiler only supports a single "
                             "device.");
  }

  cudaError_t result;
  result = cudaSetDevice(options.device.device_id(0));
  if (result != cudaSuccess) {
    throw std::runtime_error("cudaSetDevice() failed.");
  }

  library::Operation const* underlying_operation = operation;
  library::GroupedGemmDescription const& operation_desc =
    static_cast<library::GroupedGemmDescription const&>(operation->description());

  // Compute the number of copies of the problem to avoid L2 camping.
  if (!options.profiling.workspace_count) {
    int64_t bytes = problem_.bytes(operation_desc);
    if (bytes < 3 * int64_t(options.device.properties[0].l2CacheSize)) {
      gemm_workspace_.problem_count =
        1 + int((3 * int64_t(options.device.properties[0].l2CacheSize)) / bytes);
    }
    else {
      gemm_workspace_.problem_count = 1;
    }
  }
  else {
    gemm_workspace_.problem_count = options.profiling.workspace_count;
  }

  bool allocate_device_tensors = options.execution_mode != ExecutionMode::kDryRun;
  if (allocate_device_tensors) {
    size_t num_groups = problem_.problem_sizes.size();
    // input data
    gemm_workspace_.A_ptr_array_host.resize(num_groups);
    gemm_workspace_.B_ptr_array_host.resize(num_groups);
    gemm_workspace_.C_ptr_array_host.resize(num_groups);
    gemm_workspace_.D_ptr_array_host.resize(num_groups);
    if (is_block_scaled) {
      auto& block_scaling_ws = gemm_workspace_.block_scales.value();
      block_scaling_ws.SFA_ptr_array_host.resize(num_groups);
      block_scaling_ws.SFB_ptr_array_host.resize(num_groups);
      block_scaling_ws.SFC_ptr_array_host.resize(num_groups);
      block_scaling_ws.SFD_ptr_array_host.resize(num_groups);
      block_scaling_ws.SFD_reference_ptr_array_host.resize(num_groups);
    }
    else if (is_blockwise) {
      auto& block_scaling_ws = gemm_workspace_.block_scales.value();
      block_scaling_ws.SFA_ptr_array_host.resize(num_groups);
      block_scaling_ws.SFB_ptr_array_host.resize(num_groups);
      block_scaling_ws.SFC_ptr_array_host.resize(num_groups);
    }
    static_assert(sizeof(void*) == 8); // allocating blocks for pointers, so verify pointer size
    // ldx
    gemm_workspace_.lda_array_device =
      device_context
        .allocate_block(options, "lda_array", library::NumericTypeID::kS64, num_groups, 0);
    gemm_workspace_.ldb_array_device =
      device_context
        .allocate_block(options, "ldb_array", library::NumericTypeID::kS64, num_groups, 0);
    gemm_workspace_.ldc_array_device =
      device_context
        .allocate_block(options, "ldc_array", library::NumericTypeID::kS64, num_groups, 0);
    gemm_workspace_.lda_array_device->copy_from_host(problem_.lda.data());
    gemm_workspace_.ldb_array_device->copy_from_host(problem_.ldb.data());
    gemm_workspace_.ldc_array_device->copy_from_host(problem_.ldc.data());
    // problem sizes
    gemm_workspace_.problem_sizes_array_device = device_context.allocate_block(
      options,
      "problem_sizes_array",
      library::NumericTypeID::kU8,
      num_groups * sizeof(gemm::GemmCoord),
      0);
    gemm_workspace_.problem_sizes_array_device->copy_from_host(problem_.problem_sizes.data());

    gemm_workspace_.problem_sizes_3x_array_device = device_context.allocate_block(
      options,
      "problem_sizes_array_3x",
      library::NumericTypeID::kU8,
      num_groups * sizeof(cute::Shape<int, int, int>),
      0);
    gemm_workspace_.problem_sizes_3x_array_device->copy_from_host(problem_.problem_sizes_3x.data());

    // reference
    gemm_workspace_.reference_ptr_array_host.resize(num_groups);

    int seed_shift = 0;
    for (size_t group_idx = 0; group_idx < num_groups; group_idx++) {
      auto group_str = std::to_string(group_idx);
      gemm_workspace_.A_ptr_array_host[group_idx] = device_context.allocate_and_initialize_tensor(
        options,
        "A_" + group_str,
        operation_desc.gemm.A.element,
        operation_desc.gemm.A.layout,
        {int(problem_.m(group_idx)), int(problem_.k(group_idx))},
        {int(problem_.lda[group_idx])},
        gemm_workspace_.problem_count,
        seed_shift++,
        0);
      gemm_workspace_.B_ptr_array_host[group_idx] = device_context.allocate_and_initialize_tensor(
        options,
        "B_" + group_str,
        operation_desc.gemm.B.element,
        operation_desc.gemm.B.layout,
        {int(problem_.k(group_idx)), int(problem_.n(group_idx))},
        {int(problem_.ldb[group_idx])},
        gemm_workspace_.problem_count,
        seed_shift++,
        0);
      gemm_workspace_.C_ptr_array_host[group_idx] = device_context.allocate_and_initialize_tensor(
        options,
        "C_" + group_str,
        operation_desc.gemm.C.element,
        operation_desc.gemm.C.layout,
        {int(problem_.m(group_idx)), int(problem_.n(group_idx))},
        {int(problem_.ldc[group_idx])},
        gemm_workspace_.problem_count,
        seed_shift++,
        0);
      gemm_workspace_.D_ptr_array_host[group_idx] = device_context.allocate_tensor(
        options,
        "D_" + group_str,
        operation_desc.gemm.D.element,
        operation_desc.gemm.D.layout,
        {int(problem_.m(group_idx)), int(problem_.n(group_idx))},
        {int(problem_.ldc[group_idx])},
        gemm_workspace_.problem_count,
        0);

      gemm_workspace_.reference_ptr_array_host[group_idx] = device_context.allocate_tensor(
        options,
        "Reference_" + group_str,
        operation_desc.gemm.D.element,
        operation_desc.gemm.D.layout,
        {int(problem_.m(group_idx)), int(problem_.n(group_idx))},
        {int(problem_.ldc[group_idx])},
        1,
        0);

      if (is_block_scaled) {
        auto const block_scale_desc = operation_desc.block_scales.value();
        auto& block_scale_ws = gemm_workspace_.block_scales.value();
        int sfa_m = round_up(int(problem_.m(group_idx)), 128);
        int sfb_n = round_up(int(problem_.n(group_idx)), 128);
        int sfa_sfb_k =
          round_up(ceil_div(int(problem_.k(group_idx)), block_scale_desc.SFKVecSize), 4);

        int sfd_m =
          block_scale_desc.SFD.layout == cutlass::library::LayoutTypeID::kRowMajor
            ? sfa_m
            : round_up(ceil_div(int(problem_.m(group_idx)), block_scale_desc.EpilogueSFVecSize), 4);
        int sfd_n =
          block_scale_desc.SFD.layout == cutlass::library::LayoutTypeID::kRowMajor
            ? round_up(ceil_div(int(problem_.n(group_idx)), block_scale_desc.EpilogueSFVecSize), 4)
            : sfb_n;

        block_scale_ws.SFA_ptr_array_host[group_idx] =
          device_context.allocate_and_initialize_tensor(
            options,
            "SFA",
            block_scale_desc.SFA.element,
            block_scale_desc.SFA.layout,
            {sfa_m, sfa_sfb_k},
            {sfa_sfb_k},
            gemm_workspace_.problem_count,
            seed_shift++,
            0);

        block_scale_ws.SFB_ptr_array_host[group_idx] =
          device_context.allocate_and_initialize_tensor(
            options,
            "SFB",
            block_scale_desc.SFB.element,
            block_scale_desc.SFB.layout,
            {sfb_n, sfa_sfb_k},
            {sfa_sfb_k},
            gemm_workspace_.problem_count,
            seed_shift++,
            0);

        block_scale_ws.SFD_ptr_array_host[group_idx] = device_context.allocate_tensor(
          options,
          "SFD",
          block_scale_desc.SFD.element,
          block_scale_desc.SFD.layout,
          {sfd_m, sfd_n},
          {sfd_n},
          gemm_workspace_.problem_count,
          0);

        block_scale_ws.SFD_reference_ptr_array_host[group_idx] = device_context.allocate_tensor(
          options,
          "Reference_SFD",
          block_scale_desc.SFD.element,
          block_scale_desc.SFD.layout,
          {sfd_m, sfd_n},
          {sfd_n},
          gemm_workspace_.problem_count,
          0);

        // ScaleFactor tensor results may have some holes and will not be touched by the kernel.
        // If we randomly fill the two tensors, these holes may encounter refcheck errors.
        if (block_scale_ws.SFD_ptr_array_host[group_idx]->type() != library::NumericTypeID::kVoid) {
          block_scale_ws.SFD_reference_ptr_array_host[group_idx]->fill_device(0);
          block_scale_ws.SFD_ptr_array_host[group_idx]->fill_device(0);
        }
      }
      else if (is_blockwise) {
        auto const block_scale_desc = operation_desc.block_scales.value();
        auto& block_scale_ws = gemm_workspace_.block_scales.value();
        int sfa_m     = ceil_div(int(problem_.m(group_idx)), block_scale_desc.SFMVecSize);
        int sfb_n     = ceil_div(int(problem_.n(group_idx)), block_scale_desc.SFNVecSize);
        int sfa_sfb_k = ceil_div(int(problem_.k(group_idx)), block_scale_desc.SFKVecSize);

        block_scale_ws.SFA_ptr_array_host[group_idx] =
          device_context.allocate_and_initialize_tensor(
            options,
            "SFA_" + std::to_string(group_idx),
            block_scale_desc.SFA.element,
            block_scale_desc.SFA.layout,
            {sfa_m, sfa_sfb_k},
            {sfa_m},
            gemm_workspace_.problem_count,
            seed_shift++,
            0);

        block_scale_ws.SFB_ptr_array_host[group_idx] =
          device_context.allocate_and_initialize_tensor(
            options,
            "SFB_" + std::to_string(group_idx),
            block_scale_desc.SFB.element,
            block_scale_desc.SFB.layout,
            {sfa_sfb_k, sfb_n},
            {sfb_n},
            gemm_workspace_.problem_count,
            seed_shift++,
            0);
      }
    }

    // takes the allocated tensors and initializes an array of pointers per problem in the workspace
    auto create_dev_ptr_array_all_workspace = [&](
                                                std::vector<DeviceAllocation*>& dev_ptr_arrays,
                                                std::vector<DeviceAllocation*> const& input,
                                                std::string const& id) {
      auto num_workspaces = gemm_workspace_.problem_count;
      dev_ptr_arrays.resize(num_workspaces);
      // note "problem_count" here refers to input/output count for L2 cycling
      for (int i = 0; i < gemm_workspace_.problem_count; i++) {
        std::string name = id + "_ptr_array_workspace" + std::to_string(i);
        dev_ptr_arrays[i] =
          device_context.allocate_block(options, name, library::NumericTypeID::kU64, num_groups, 0);
        std::vector<void*> group_ptrs(num_groups);
        for (size_t group_idx = 0; group_idx < num_groups; group_idx++) {
          group_ptrs[group_idx] = input[group_idx]->batch_data(i);
        }
        dev_ptr_arrays[i]->copy_from_host(group_ptrs.data());
      }
    };
    create_dev_ptr_array_all_workspace(
      gemm_workspace_.A_ptr_array_device,
      gemm_workspace_.A_ptr_array_host,
      "A");
    create_dev_ptr_array_all_workspace(
      gemm_workspace_.B_ptr_array_device,
      gemm_workspace_.B_ptr_array_host,
      "B");
    create_dev_ptr_array_all_workspace(
      gemm_workspace_.C_ptr_array_device,
      gemm_workspace_.C_ptr_array_host,
      "C");
    create_dev_ptr_array_all_workspace(
      gemm_workspace_.D_ptr_array_device,
      gemm_workspace_.D_ptr_array_host,
      "D");

    if (is_block_scaled) {
      auto& block_scale_ws = gemm_workspace_.block_scales.value();
      create_dev_ptr_array_all_workspace(
        block_scale_ws.SFA_ptr_array_device,
        block_scale_ws.SFA_ptr_array_host,
        "SFA");
      create_dev_ptr_array_all_workspace(
        block_scale_ws.SFB_ptr_array_device,
        block_scale_ws.SFB_ptr_array_host,
        "SFB");
      create_dev_ptr_array_all_workspace(
        block_scale_ws.SFD_ptr_array_device,
        block_scale_ws.SFD_ptr_array_host,
        "SFD");

      block_scale_ws.norm_constant = device_context.allocate_and_initialize_tensor(
        options,
        "norm_constant",
        operation_desc.gemm.element_epilogue,
        operation_desc.gemm.A.layout, // copied, but should this be D layout?
        {1, 1},
        {1},
        1,
        seed_shift++,
        0 // device_index
      );
    }
    else if (is_blockwise) {
      auto& block_scale_ws = gemm_workspace_.block_scales.value();
      create_dev_ptr_array_all_workspace(
        block_scale_ws.SFA_ptr_array_device,
        block_scale_ws.SFA_ptr_array_host,
        "SFA");
      create_dev_ptr_array_all_workspace(
        block_scale_ws.SFB_ptr_array_device,
        block_scale_ws.SFB_ptr_array_host,
        "SFB");
    }

    init_arguments(options);
  }

  //
  // Initialize the CUTLASS operation
  //
  Status status = Status::kSuccess;
  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {
    if (options.execution_mode != ExecutionMode::kDryRun) {
      uint64_t workspace_size =
        underlying_operation->get_host_workspace_size(&gemm_workspace_.configuration);
      gemm_workspace_.host_workspace.resize(workspace_size, 0);

      workspace_size = underlying_operation->get_device_workspace_size(
        &gemm_workspace_.configuration,
        &gemm_workspace_.arguments);
      gemm_workspace_.device_workspace.reset(library::NumericTypeID::kU8, workspace_size);

      status = underlying_operation->initialize(
        &gemm_workspace_.configuration,
        gemm_workspace_.host_workspace.data(),
        gemm_workspace_.device_workspace.data());
      if (status != Status::kSuccess) {
        return status;
      }

      status = underlying_operation->can_implement(
        &gemm_workspace_.configuration,
        &gemm_workspace_.arguments);
      if (status != Status::kSuccess) {
        return status;
      }
    }

    //
    // If CUTLASS is enabled, generate a result for it
    //
    results_.push_back(model_result_);
    results_.back().provider = library::Provider::kCUTLASS;
    results_.back().op_kind = library::OperationKind::kGroupedGemm;
    results_.back().disposition = Disposition::kNotRun;

    for (auto provider : verification_providers_) {
      results_.back().verification_map[provider] = Disposition::kNotRun;
    }
  }
  return status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Verifies CUTLASS against references
bool GroupedGemmOperationProfiler::verify_cutlass(
  Options const& options,
  PerformanceReport& report,
  DeviceContext& device_context,
  library::Operation const* operation,
  ProblemSpace const& problem_space,
  ProblemSpace::Problem const& problem) {

  if (!options.profiling.provider_enabled(library::Provider::kCUTLASS)) {
    return true;
  }

  if (options.execution_mode == ExecutionMode::kDryRun) {
    return true;
  }

  init_arguments(options);

  library::Operation const* underlying_operation = operation;
  results_.back().status = underlying_operation->initialize_with_arguments(&gemm_workspace_.arguments);
  if (results_.back().status != Status::kSuccess) {
    return false;
  }

  results_.back().status = underlying_operation->run(
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

#if CUTLASS_ENABLE_CUBLAS
    if (options.verification.provider_enabled(library::Provider::kCUBLAS)) {
      // set verification map for cublas to not supported
      results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kNotSupported;
    }
#endif // #if CUTLASS_ENABLE_CUBLAS

    auto const& desc =
      static_cast<library::GroupedGemmDescription const&>(operation->description());

    cutlass::library::RuntimeDatatype runtime_datatype_a = gemm_workspace_.arguments.runtime_input_datatype_a;
    cutlass::library::RuntimeDatatype runtime_datatype_b = gemm_workspace_.arguments.runtime_input_datatype_b;

    bool is_runtime_datatype_a = runtime_datatype_a != cutlass::library::RuntimeDatatype::kStatic;
    bool is_runtime_datatype_b = runtime_datatype_b != cutlass::library::RuntimeDatatype::kStatic;

    assert(is_runtime_datatype_a == is_runtime_datatype_b && "runtime datatype should be both dynamic or static.");
    
    cutlass::library::NumericTypeID element_A = desc.gemm.A.element;
    cutlass::library::NumericTypeID element_B = desc.gemm.B.element;
    
    if (is_runtime_datatype_a) {
      element_A = cutlass::library::dynamic_datatype_to_id(runtime_datatype_a);
    }

    if (is_runtime_datatype_b) {
      element_B = cutlass::library::dynamic_datatype_to_id(runtime_datatype_b);
    }

    bool verification_status = verify_with_reference_(
      options,
      report,
      device_context,
      operation,
      problem_space,
      problem,
      element_A,
      element_B);

    // Update disposition to worst case verification outcome among all
    // verification providers which are supported
    bool is_any_verification_run_passed = false;
    for (auto& m : results_.back().verification_map) {
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
      did_any_verification_run |=
        (Disposition::kNotRun != results_.back().verification_map[provider]);
    }

    if (not did_any_verification_run) {
      results_.back().status = Status::kErrorNotSupported;
      return false;
    }
  }

  // Return true means continue profiling
  return true;
}

/// Verifies CUTLASS against host and device references
bool GroupedGemmOperationProfiler::verify_with_reference_(
  Options const& options,
  PerformanceReport& report,
  DeviceContext& device_context,
  library::Operation const* operation,
  ProblemSpace const& problem_space,
  ProblemSpace::Problem const& problem,
  cutlass::library::NumericTypeID element_A,
  cutlass::library::NumericTypeID element_B) {
  library::GroupedGemmDescription const& desc =
    static_cast<library::GroupedGemmDescription const&>(operation->description());

  for (auto provider : options.verification.providers) {

    // Skip providers that are not enabled
    if (!options.verification.provider_enabled(provider)) {
      continue;
    }

    // we only have a block scaled reference kernel implemented on the host
    if ((is_block_scaled || is_blockwise) && provider != library::Provider::kReferenceHost) {
      continue;
    }

    auto status = Status::kSuccess;
    auto disposition = Disposition::kFailed;
    // we don't have grouped GEMM reference kernels so we loop over the groups and perform
    // a regular GEMM for each group
    for (size_t group_idx = 0, num_groups = problem_.problem_sizes.size(); group_idx < num_groups;
         group_idx++) {
      void* ptr_A = gemm_workspace_.A_ptr_array_host[group_idx]->data();
      void* ptr_B = gemm_workspace_.B_ptr_array_host[group_idx]->data();
      void* ptr_C = gemm_workspace_.C_ptr_array_host[group_idx]->data();
      void* ptr_D = gemm_workspace_.reference_ptr_array_host[group_idx]->data();

      // To support the host-side reference, conditionally allocate and
      // copy tensors to host memory.
      std::vector<uint8_t> host_data_A;
      std::vector<uint8_t> host_data_B;
      std::vector<uint8_t> host_data_C;
      std::vector<uint8_t> host_data_D;
      std::vector<uint8_t> host_data_SFA;
      std::vector<uint8_t> host_data_SFB;
      std::vector<uint8_t> host_data_SFC;
      std::vector<uint8_t> host_data_SFD;
      std::vector<uint8_t> host_data_norm_constant;

      void* ptr_SFA{nullptr};
      void* ptr_SFB{nullptr};
      void* ptr_SFD{nullptr};
      void* ptr_norm_constant{nullptr};

      if (provider == library::Provider::kReferenceHost) {
        host_data_A.resize(gemm_workspace_.A_ptr_array_host[group_idx]->bytes());
        ptr_A = host_data_A.data();
        gemm_workspace_.A_ptr_array_host[group_idx]->copy_to_host(
          ptr_A); // this is copying all the data for L2 busting as well

        host_data_B.resize(gemm_workspace_.B_ptr_array_host[group_idx]->bytes());
        ptr_B = host_data_B.data();
        gemm_workspace_.B_ptr_array_host[group_idx]->copy_to_host(ptr_B);

        host_data_C.resize(gemm_workspace_.C_ptr_array_host[group_idx]->bytes());
        ptr_C = host_data_C.data();
        gemm_workspace_.C_ptr_array_host[group_idx]->copy_to_host(ptr_C);

        host_data_D.resize(gemm_workspace_.reference_ptr_array_host[group_idx]->bytes());
        ptr_D = host_data_D.data();

        if (is_block_scaled) {
          auto const& ws = gemm_workspace_.block_scales.value();

          host_data_SFA.resize(ws.SFA_ptr_array_host[group_idx]->bytes());
          ptr_SFA = host_data_SFA.data();
          ws.SFA_ptr_array_host[group_idx]->copy_to_host(ptr_SFA);
          host_data_SFB.resize(ws.SFB_ptr_array_host[group_idx]->bytes());
          ptr_SFB = host_data_SFB.data();
          ws.SFB_ptr_array_host[group_idx]->copy_to_host(ptr_SFB);

          host_data_SFD.resize(ws.SFD_reference_ptr_array_host[group_idx]->bytes());
          ptr_SFD = host_data_SFD.data();

          host_data_norm_constant.resize(ws.norm_constant->bytes());
          ptr_norm_constant = host_data_norm_constant.data();
          ws.norm_constant->copy_to_host(ptr_norm_constant);
        }
        else if (is_blockwise) {
          auto const& ws = gemm_workspace_.block_scales.value();

          host_data_SFA.resize(ws.SFA_ptr_array_host[group_idx]->bytes());
          ptr_SFA = host_data_SFA.data();
          ws.SFA_ptr_array_host[group_idx]->copy_to_host(ptr_SFA);
          host_data_SFB.resize(ws.SFB_ptr_array_host[group_idx]->bytes());
          ptr_SFB = host_data_SFB.data();
          ws.SFB_ptr_array_host[group_idx]->copy_to_host(ptr_SFB);
        }
      }

      const auto &desc = static_cast<library::GroupedGemmDescription const &>(operation->description());
      const auto& gemm_desc = desc.gemm;

      if (!is_block_scaled and !is_blockwise) {
        library::Handle handle;
        handle.set_provider(provider);

        status = handle.gemm_universal(
          library::GemmUniversalMode::kGemm,
          problem_.m(group_idx),
          problem_.n(group_idx),
          problem_.k(group_idx),
          problem_.cluster_m,
          problem_.cluster_n,
          problem_.cluster_k,
          problem_.cluster_m_fallback,
          problem_.cluster_n_fallback,
          problem_.cluster_k_fallback,
          desc.gemm.tile_description.math_instruction.element_accumulator,
          desc.gemm.element_epilogue,
          problem_.alpha.data(),
          element_A,
          desc.gemm.A.layout,
          desc.gemm.transform_A,
          ptr_A,
          int(problem_.lda[group_idx]),
          element_B,
          desc.gemm.B.layout,
          desc.gemm.transform_B,
          ptr_B,
          int(problem_.ldb[group_idx]),
          problem_.beta.data(),
          desc.gemm.C.element,
          desc.gemm.C.layout,
          ptr_C,
          int(problem_.ldc[group_idx]),
          desc.gemm.D.element,
          desc.gemm.D.layout,
          ptr_D,
          int(problem_.ldc[group_idx]),
          1,
          gemm_workspace_.A_ptr_array_host[group_idx]->batch_stride(),
          gemm_workspace_.B_ptr_array_host[group_idx]->batch_stride(),
          gemm_workspace_.C_ptr_array_host[group_idx]->batch_stride(),
          gemm_workspace_.reference_ptr_array_host[group_idx]->batch_stride());
      }
      else if (is_block_scaled) {
        auto const& block_scale_desc = desc.block_scales.value();
        auto& block_scale_ws = gemm_workspace_.block_scales.value();

        library::BlockScaledGemmFunctionalKey blockScaledGemm_key(
          library::Provider::kReferenceHost,
          library::GemmKind::kUniversal,
          library::OperationKind::kBlockScaledGemm,
          gemm_desc.tile_description.math_instruction.element_accumulator,
          gemm_desc.element_epilogue,
          element_A,
          gemm_desc.A.layout,
          block_scale_desc.SFA.element,
          element_B,
          gemm_desc.B.layout,
          block_scale_desc.SFB.element,
          gemm_desc.C.element,
          gemm_desc.C.layout,
          gemm_desc.D.element,
          gemm_desc.D.layout,
          block_scale_desc.SFD.element,
          block_scale_desc.SFD.layout,
          block_scale_desc.SFKVecSize,
          block_scale_desc.EpilogueSFVecSize);

        auto operators_it =
          library::Singleton::get().operation_table.block_scaled_gemm_operations.find(
            blockScaledGemm_key);
        if (
          operators_it ==
          library::Singleton::get().operation_table.block_scaled_gemm_operations.end()) {
          disposition = Disposition::kNotSupported;
          break;
        }

        if (operators_it->second.empty()) {
          disposition = Disposition::kNotSupported;
          break;
        }

        auto cc_it = operators_it->second.begin();
        if (cc_it == operators_it->second.end()) {
          disposition = Disposition::kNotSupported;
          break;
        }

        // host reference has only one instances in BlockScaledOperationVectorMap
        library::Operation const* reference_op = cc_it->second[0];
        library::BlockScaledGemmArguments arguments{
          {int(problem_.m(group_idx)), int(problem_.n(group_idx)), int(problem_.k(group_idx))},
          {int(problem_.cluster_m), int(problem_.cluster_n), int(problem_.cluster_k)},
          {int(problem_.cluster_m_fallback), int(problem_.cluster_n_fallback), int(problem_.cluster_k_fallback)},
          1, // batch count
          ptr_A,
          ptr_B,
          ptr_SFA,
          ptr_SFB,
          ptr_C,
          ptr_D,
          ptr_SFD,
          problem_.alpha.data(),
          problem_.beta.data(),
          library::ScalarPointerMode::kHost,
          problem_.lda[group_idx],
          problem_.ldb[group_idx],
          problem_.ldc[group_idx],
          problem_.ldc[group_idx],
          gemm_workspace_.A_ptr_array_host[group_idx]->batch_stride(),
          gemm_workspace_.B_ptr_array_host[group_idx]->batch_stride(),
          gemm_workspace_.C_ptr_array_host[group_idx]->batch_stride(),
          gemm_workspace_.reference_ptr_array_host[group_idx]->batch_stride(),
          ptr_norm_constant};

        library::GemmUniversalConfiguration configuration{
          library::GemmUniversalMode::kGemm,
          problem_.problem_sizes[group_idx],
          {problem_.cluster_m, problem_.cluster_n, problem_.cluster_k},
          {problem_.cluster_m_fallback, problem_.cluster_n_fallback, problem_.cluster_k_fallback},
          1,
          problem_.lda[group_idx],
          problem_.ldb[group_idx],
          problem_.ldc[group_idx],
          problem_.ldc[group_idx],
          1,
        };
        uint64_t host_workspace_size_needed = reference_op->get_host_workspace_size(&gemm_workspace_.configuration);
        std::vector<char> host_workspace(host_workspace_size_needed);
        status = reference_op->initialize(&configuration, host_workspace.data());
        if (status != Status::kSuccess) {
          break;
        }

        status = reference_op->run(&arguments, host_workspace.data());

        block_scale_ws.SFD_reference_ptr_array_host[group_idx]->copy_from_host(ptr_SFD);
      }
      else {
        // Blockwise
        auto const& block_scale_desc = desc.block_scales.value();
        auto& block_scale_ws = gemm_workspace_.block_scales.value();

        library::BlockwiseGemmFunctionalKey blockwiseGemm_key(
          library::Provider::kReferenceHost,
          library::GemmKind::kUniversal,
          library::OperationKind::kBlockwiseGemm,
          gemm_desc.tile_description.math_instruction.element_accumulator,
          gemm_desc.element_epilogue,
          element_A,
          gemm_desc.A.layout,
          block_scale_desc.SFA.element,
          element_B,
          gemm_desc.B.layout,
          block_scale_desc.SFB.element,
          gemm_desc.C.element,
          gemm_desc.C.layout,
          gemm_desc.D.element,
          gemm_desc.D.layout,
          block_scale_desc.SFMVecSize,
          block_scale_desc.SFNVecSize,
          block_scale_desc.SFKVecSize
        );

        auto operators_it = library::Singleton::get().operation_table.blockwise_gemm_operations.find(blockwiseGemm_key);
        if (
          operators_it ==
          library::Singleton::get().operation_table.blockwise_gemm_operations.end()) {
          disposition = Disposition::kNotSupported;
          break;
        }

        if (operators_it->second.empty()) {
          disposition = Disposition::kNotSupported;
          break;
        }

        auto cc_it = operators_it->second.begin();
        if (cc_it == operators_it->second.end()) {
          disposition = Disposition::kNotSupported;
          break;
        }

        // host reference has only one instances in BlockScaledOperationVectorMap
        library::Operation const* reference_op = cc_it->second[0];

        library::BlockwiseGemmArguments arguments {
          {int(problem_.m(group_idx)), int(problem_.n(group_idx)), int(problem_.k(group_idx))},
          {int(problem_.cluster_m), int(problem_.cluster_n), int(problem_.cluster_k)},
          {int(problem_.cluster_m_fallback), int(problem_.cluster_n_fallback), int(problem_.cluster_k_fallback)},
          1, // batch_count
          ptr_A,
          ptr_B,
          ptr_SFA,
          ptr_SFB,
          ptr_C,
          ptr_D,
          problem_.alpha.data(),
          problem_.beta.data(),
          library::ScalarPointerMode::kHost,
          problem_.lda[group_idx],
          problem_.ldb[group_idx],
          problem_.ldc[group_idx],
          problem_.ldc[group_idx],
          gemm_workspace_.A_ptr_array_host[group_idx]->batch_stride(),
          gemm_workspace_.B_ptr_array_host[group_idx]->batch_stride(),
          gemm_workspace_.C_ptr_array_host[group_idx]->batch_stride(),
          gemm_workspace_.reference_ptr_array_host[group_idx]->batch_stride(),
        };

        library::GemmUniversalConfiguration configuration{
          library::GemmUniversalMode::kGemm,
          problem_.problem_sizes[group_idx],
          {problem_.cluster_m, problem_.cluster_n, problem_.cluster_k},
          {problem_.cluster_m_fallback, problem_.cluster_n_fallback, problem_.cluster_k_fallback},
          1,
          problem_.lda[group_idx],
          problem_.ldb[group_idx],
          problem_.ldc[group_idx],
          problem_.ldc[group_idx],
          1,
        };
        uint64_t host_workspace_size_needed = reference_op->get_host_workspace_size(&gemm_workspace_.configuration);
        std::vector<char> host_workspace(host_workspace_size_needed);
        status = reference_op->initialize(&configuration, host_workspace.data());
        if (status != Status::kSuccess) {
          break;
        }

        status = reference_op->run(&arguments, host_workspace.data());
      }

      if (status != Status::kSuccess) {
        break;
      }

      if (provider == library::Provider::kReferenceHost) {
        gemm_workspace_.reference_ptr_array_host[group_idx]->copy_from_host(ptr_D);
      }

      disposition = compare_tensors(
        options,
        *gemm_workspace_.D_ptr_array_host[group_idx],
        *gemm_workspace_.reference_ptr_array_host[group_idx],
        gemm_workspace_.D_ptr_array_host[group_idx]->batch_stride());
      if (disposition != Disposition::kPassed) {
        break;
      }

      if (is_block_scaled) {
        auto& ws = gemm_workspace_.block_scales.value();
        auto const& block_scale_desc = desc.block_scales.value();
        if (block_scale_desc.SFD.element != library::NumericTypeID::kVoid) {
          disposition = compare_tensors(
            options,
            *ws.SFD_ptr_array_host[group_idx],
            *ws.SFD_reference_ptr_array_host[group_idx],
            ws.SFD_ptr_array_host[group_idx]->batch_stride());
          if (disposition != Disposition::kPassed) {
            break;
          }
        }
      }
    }
    if (status != Status::kSuccess) {
      results_.back().verification_map[provider] = Disposition::kNotVerified;
      continue;
    }
    results_.back().status = status;
    results_.back().verification_map[provider] = disposition;

    if (
      options.verification.save_workspace == SaveWorkspace::kIncorrect &&
      results_.back().verification_map[provider] == Disposition::kIncorrect) {
      save_workspace(device_context, options, desc, library::Provider::kCUTLASS, provider);
    }
  }

  return true; // continue profiling
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Measures performance results
bool GroupedGemmOperationProfiler::profile(
  Options const& options,
  PerformanceReport& report,
  DeviceContext& device_context,
  library::Operation const* operation,
  ProblemSpace const& problem_space,
  ProblemSpace::Problem const& problem) {

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {
    if (options.profiling.enable_kernel_performance_search) {
      std::cerr << "Exhaustive performance search is not available for Grouped GEMMs. " 
                << "Please use --enable-best-kernel-for-fixed-shape to profile a specific problem size "
                << "with --problem-sizes or --problem-sizes-file.\n";
    }
    else if (options.profiling.enable_best_kernel_for_fixed_shape) {
      return profile_cutlass_for_fixed_shape_(options, operation, problem_space);
    }
    else {
      results_.back().status = profile_cutlass_(
        results_.back(),
        options,
        operation,
        &gemm_workspace_.arguments,
        gemm_workspace_.host_workspace.data(),
        gemm_workspace_.device_workspace.data());
    }
  }
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Method to profile a CUTLASS Operation
Status GroupedGemmOperationProfiler::profile_cutlass_(
  PerformanceResult& result,
  Options const& options,
  library::Operation const* operation,
  void* arguments,
  void* host_workspace,
  void* device_workspace) {
  library::Operation const* underlying_operation = operation;
  result.status = underlying_operation->initialize_with_arguments(&gemm_workspace_.arguments);
  if (result.status != Status::kSuccess) {
    return result.status;
  }

  auto func = [&](cudaStream_t stream, int iteration) {
    // Iterate over copies of the problem in memory
    int workspace_idx = options.profiling.warmup_iterations + iteration;
    int problem_idx = (workspace_idx % gemm_workspace_.problem_count);

    gemm_workspace_.arguments.ptr_A = gemm_workspace_.A_ptr_array_device[problem_idx]->data();
    gemm_workspace_.arguments.ptr_B = gemm_workspace_.B_ptr_array_device[problem_idx]->data();
    gemm_workspace_.arguments.ptr_C = gemm_workspace_.C_ptr_array_device[problem_idx]->data();
    gemm_workspace_.arguments.ptr_D = gemm_workspace_.D_ptr_array_device[problem_idx]->data();

    return underlying_operation->run(arguments, host_workspace, device_workspace, stream);
  };
  return profile_kernel_(result, options, func, gemm_workspace_.stream);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Method to profile a CUTLASS Operation for the best configuration for a fixed shape
bool GroupedGemmOperationProfiler::profile_cutlass_for_fixed_shape_(
  Options const& options,
  library::Operation const* operation,
  ProblemSpace const& problem_space) {
  library::GroupedGemmDescription const &operation_desc =
    static_cast<library::GroupedGemmDescription const &>(operation->description());

  auto cluster_shape = operation_desc.tile_description.cluster_shape;
  bool is_dynamic_cluster_enabled = cluster_shape.m() == 0 || cluster_shape.n() == 0 || cluster_shape.k() == 0;

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
          PerformanceResult curr_result(result_base);
          update_workspace_and_result_(gemm_workspace_, curr_result, problem_space, raster_order, preferred_cluster, fallback_cluster, swizzle_size, is_dynamic_cluster_enabled);
          curr_result.status  = profile_cutlass_(
            curr_result,
            options,
            operation,
            &gemm_workspace_.arguments,
            gemm_workspace_.host_workspace.data(),
            gemm_workspace_.device_workspace.data()
          );
          if (curr_result.status == Status::kSuccess) {  // Only add valid results
            candidates.push_back(curr_result);
          }
        }// for raster_order
      }// for swizzle_size
    }// for fallback_cluster
  }// for preferred_clusters

  if (candidates.empty()) {
    return false;
  }
  select_best_candidate(candidates);
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
