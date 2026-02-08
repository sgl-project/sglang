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
   \brief GroupedGemm Profiler
*/

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// CUTLASS Library includes
#include "cutlass/library/library.h"

// Profiler includes
#include "device_context.h"
#include "operation_profiler.h"
#include "options.h"
#include "performance_result.h"
#include "problem_space.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Abstract base class for each math function
class GroupedGemmOperationProfiler : public OperationProfiler {
public:
  /// Problem structure obtained from problem space
  struct GroupedGemmProblem {

    cutlass::library::GemmUniversalMode mode{library::GemmUniversalMode::kGrouped};

    std::vector<gemm::GemmCoord> problem_sizes;
    std::vector<cute::Shape<int, int, int>> problem_sizes_3x;

    /// For exploration purposes
    std::vector<std::array<int64_t, 3>> preferred_clusters;
    std::vector<std::array<int64_t, 3>> fallback_clusters;
    std::vector<cutlass::library::RasterOrder> raster_orders;
    std::vector<int> swizzle_sizes;

    int cluster_m{1};
    int cluster_n{1};
    int cluster_k{1};
    int cluster_m_fallback{1};
    int cluster_n_fallback{1};
    int cluster_k_fallback{1};

    std::vector<int64_t> lda{0};
    std::vector<int64_t> ldb{0};
    std::vector<int64_t> ldc{0};

    std::vector<uint8_t> alpha;
    std::vector<uint8_t> beta;

    cutlass::library::RasterOrder raster_order{cutlass::library::RasterOrder::kHeuristic};
    int swizzle_size{1};

    cutlass::library::RuntimeDatatype runtime_input_datatype_a{};
    cutlass::library::RuntimeDatatype runtime_input_datatype_b{};

    bool use_pdl{false};

    /// Parses the problem
    Status parse(
      library::GroupedGemmDescription const& operation_desc,
      ProblemSpace const& problem_space,
      ProblemSpace::Problem const& problem);

    int64_t m(int group_idx) const { return problem_sizes[group_idx].m(); };
    int64_t n(int group_idx) const { return problem_sizes[group_idx].n(); };
    int64_t k(int group_idx) const { return problem_sizes[group_idx].k(); };

    /// Total number of bytes loaded
    int64_t bytes(library::GroupedGemmDescription const& operation_desc) const;

    /// Total number of flops computed
    int64_t flops(library::GroupedGemmDescription const& operation_desc) const;

    /// Initializes a performance result
    void initialize_result(
      PerformanceResult& result,
      library::GroupedGemmDescription const& operation_desc,
      ProblemSpace const& problem_space);
  };

  struct BlockScalingWorkspace {
    // host vector (per L2 workspace) of device vectors (per group) of device pointers
    std::vector<DeviceAllocation*> SFA_ptr_array_device;
    std::vector<DeviceAllocation*> SFB_ptr_array_device;
    std::vector<DeviceAllocation*> SFC_ptr_array_device;
    std::vector<DeviceAllocation*> SFD_ptr_array_device;

    // host vector (per group) of device tensors
    // (where each batch of device allocation is for a L2 workspace)
    std::vector<DeviceAllocation*> SFA_ptr_array_host;
    std::vector<DeviceAllocation*> SFB_ptr_array_host;
    std::vector<DeviceAllocation*> SFC_ptr_array_host;
    std::vector<DeviceAllocation*> SFD_ptr_array_host;
    std::vector<DeviceAllocation*> SFD_reference_ptr_array_host;

    // matrix wide constant, not per-batch or per-group
    DeviceAllocation* norm_constant;
  };

  // workspace contains the allocated blocks, arguments just contain the raw
  // pointers
  struct GroupedGemmWorkspace {

    // host vector (per L2 workspace) of device vectors (per group) of device pointers
    std::vector<DeviceAllocation*> A_ptr_array_device;
    std::vector<DeviceAllocation*> B_ptr_array_device;
    std::vector<DeviceAllocation*> C_ptr_array_device;
    std::vector<DeviceAllocation*> D_ptr_array_device;
    std::vector<DeviceAllocation*> reference_ptr_array_host;

    // host vector (per group) of device tensors
    // (where each batch of device allocation is for a L2 workspace)
    std::vector<DeviceAllocation*> A_ptr_array_host;
    std::vector<DeviceAllocation*> B_ptr_array_host;
    std::vector<DeviceAllocation*> C_ptr_array_host;
    std::vector<DeviceAllocation*> D_ptr_array_host;

    /// Number of copies of the problem workspace which are visited sequentially during
    /// profiling to avoid camping in the last level cache.
    /// *NOT* the number of groups in the grouped GEMM (we use `num_groups` in the profiler)
    int problem_count{1};

    DeviceAllocation* problem_sizes_array_device{nullptr};
    DeviceAllocation* problem_sizes_3x_array_device{nullptr};
    DeviceAllocation* lda_array_device{nullptr};
    DeviceAllocation* ldb_array_device{nullptr};
    DeviceAllocation* ldc_array_device{nullptr};
    DeviceAllocation* ldd_array_device{nullptr};

    std::optional<BlockScalingWorkspace> block_scales;

    library::GemmGroupedConfiguration configuration;
    library::GroupedGemmBlockScaledArguments arguments;

    std::vector<uint8_t> host_workspace;
    DeviceAllocation device_workspace;

    cudaStream_t stream;
  };

private:
  void init_arguments(Options const& options) {
    auto& arguments = gemm_workspace_.arguments;
    // these get updated in each profiler run to ensure L2 cycling
    arguments.ptr_A = gemm_workspace_.A_ptr_array_device[0]->data();
    arguments.ptr_B = gemm_workspace_.B_ptr_array_device[0]->data();
    arguments.ptr_C = gemm_workspace_.C_ptr_array_device[0]->data();
    arguments.ptr_D = gemm_workspace_.D_ptr_array_device[0]->data();

    arguments.alpha = problem_.alpha.data();
    arguments.beta = problem_.beta.data();
    arguments.pointer_mode = library::ScalarPointerMode::kHost;
    arguments.lda = static_cast<int64_t*>(gemm_workspace_.lda_array_device->data());
    arguments.ldb = static_cast<int64_t*>(gemm_workspace_.ldb_array_device->data());
    arguments.ldc = static_cast<int64_t*>(gemm_workspace_.ldc_array_device->data());
    arguments.ldd = static_cast<int64_t*>(gemm_workspace_.ldc_array_device->data());
    arguments.problem_sizes =
      static_cast<gemm::GemmCoord*>(gemm_workspace_.problem_sizes_array_device->data());
    arguments.problem_sizes_3x = static_cast<cute::Shape<int, int, int>*>(
      gemm_workspace_.problem_sizes_3x_array_device->data());
    gemm_workspace_.arguments.problem_sizes_3x_host = problem_.problem_sizes_3x.data();
    gemm_workspace_.arguments.problem_count = problem_.problem_sizes.size();
    gemm_workspace_.arguments.cluster_shape = {int(problem_.cluster_m), int(problem_.cluster_n), int(problem_.cluster_k)};
    gemm_workspace_.arguments.cluster_shape_fallback = {int(problem_.cluster_m_fallback), int(problem_.cluster_n_fallback), int(problem_.cluster_k_fallback)};

    /* Query device SM count to pass onto the kernel as an argument, where needed */
    arguments.sm_count = options.device.get_sm_count(0);
    if (is_block_scaled) {
      auto& block_scaled_ws = gemm_workspace_.block_scales.value();
      arguments.SFA = block_scaled_ws.SFA_ptr_array_device[0]->data();
      arguments.SFB = block_scaled_ws.SFB_ptr_array_device[0]->data();
      arguments.SFD = block_scaled_ws.SFD_ptr_array_device[0]->data();
      arguments.norm_constant = block_scaled_ws.norm_constant->data();
    }
    else if (is_blockwise) {
      auto& block_scaled_ws = gemm_workspace_.block_scales.value();
      arguments.SFA = block_scaled_ws.SFA_ptr_array_device[0]->data();
      arguments.SFB = block_scaled_ws.SFB_ptr_array_device[0]->data();
    }
  }

protected:
  /// GEMM problem obtained from problem space
  GroupedGemmProblem problem_;

  /// Device memory allocations
  GroupedGemmWorkspace gemm_workspace_;

  bool is_block_scaled{false};
  bool is_blockwise{false};

public:
  GroupedGemmOperationProfiler(Options const& options);

  virtual ~GroupedGemmOperationProfiler();

  GroupedGemmProblem const& problem() const { return problem_; }

  /// Prints usage statement for the math function
  virtual void print_usage(std::ostream& out) const;

  /// Prints examples
  virtual void print_examples(std::ostream& out) const;

  /// Extracts the problem dimensions
  virtual Status initialize_configuration(
    Options const& options,
    PerformanceReport& report,
    DeviceContext& device_context,
    library::Operation const* operation,
    ProblemSpace const& problem_space,
    ProblemSpace::Problem const& problem);

  /// Initializes workspace
  virtual Status initialize_workspace(
    Options const& options,
    PerformanceReport& report,
    DeviceContext& device_context,
    library::Operation const* operation,
    ProblemSpace const& problem_space,
    ProblemSpace::Problem const& problem);

  /// Verifies CUTLASS against references
  virtual bool verify_cutlass(
    Options const& options,
    PerformanceReport& report,
    DeviceContext& device_context,
    library::Operation const* operation,
    ProblemSpace const& problem_space,
    ProblemSpace::Problem const& problem);

  /// Measures performance results
  virtual bool profile(
    Options const& options,
    PerformanceReport& report,
    DeviceContext& device_context,
    library::Operation const* operation,
    ProblemSpace const& problem_space,
    ProblemSpace::Problem const& problem);

protected:
  /// Initializes the performance result
  void initialize_result_(
    PerformanceResult& result,
    Options const& options,
    library::GroupedGemmDescription const& operation_desc,
    ProblemSpace const& problem_space);

  /// Update workspace configuration according to flexible user setups
  void update_workspace_(
    GroupedGemmWorkspace &gemm_workspace,
    std::array<int64_t, 3> const &preferred_cluster,
    std::array<int64_t, 3> const &fallback_cluster,
    cutlass::library::RasterOrder const &raster_order,
    int swizzle_size,
    bool is_dynamic_cluster_enabled);

  /// Update performance result configuration for exploration parameters
  void update_workspace_and_result_(
    GroupedGemmWorkspace &gemm_workspace,
    PerformanceResult &result,
    ProblemSpace const &problem_space,
    cutlass::library::RasterOrder const &raster_order,
    std::array<int64_t, 3> const &preferred_cluster,
    std::array<int64_t, 3> const &fallback_cluster,
    int swizzle_size,
    bool is_dynamic_cluster_enabled);

  /// Verifies CUTLASS against host and device references
  bool verify_with_reference_(
    Options const& options,
    PerformanceReport& report,
    DeviceContext& device_context,
    library::Operation const* operation,
    ProblemSpace const& problem_space,
    ProblemSpace::Problem const& problem,
    cutlass::library::NumericTypeID element_A,
    cutlass::library::NumericTypeID element_B);

  /// Method to profile a CUTLASS Operation
  Status profile_cutlass_(
    PerformanceResult& result,
    Options const& options,
    library::Operation const* operation,
    void* arguments,
    void* host_workspace,
    void* device_workspace) override;

  /// Method to profile a CUTLASS Operation for the best configuration for a fixed shape
  bool profile_cutlass_for_fixed_shape_(
    Options const& options,
    library::Operation const* operation,
    ProblemSpace const& problem_space);

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
