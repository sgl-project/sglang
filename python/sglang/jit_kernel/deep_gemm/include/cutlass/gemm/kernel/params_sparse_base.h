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

/*! \file
    \brief Base functionality for common types of sparse GEMM kernel parameters
*/

#pragma once

#include "cutlass/cutlass.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure
template <
  typename ThreadblockSwizzle,
  typename ParamsA,
  typename TensorRefA,
  typename ParamsB,
  typename TensorRefB,
  typename ParamsE,
  typename TensorRefE>
struct SparseParamsBase
{
  //
  // Data members
  //

  cutlass::gemm::GemmCoord problem_size{};
  cutlass::gemm::GemmCoord grid_tiled_shape{};
  int swizzle_log_tile;
  ParamsA params_A{};
  TensorRefA ref_A{};
  ParamsB params_B{};
  TensorRefB ref_B{};
  ParamsE params_E{};
  TensorRefE ref_E{};
  int gemm_k_iterations{0};
  int gemm_k_size{0};

  //
  // Host dispatch API
  //

  /// Default constructor
  SparseParamsBase() = default;

  /// Constructor
  CUTLASS_HOST_DEVICE
  SparseParamsBase(
    cutlass::gemm::GemmCoord const & problem_size,
    cutlass::gemm::GemmCoord const & grid_tiled_shape,
    TensorRefA ref_A,
    TensorRefB ref_B,
    TensorRefE ref_E,
    int const mma_shape_k)
  :
    problem_size(problem_size),
    grid_tiled_shape(grid_tiled_shape),
    swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
    params_A(ref_A.layout()),
    ref_A(ref_A),
    params_B(ref_B.layout()),
    ref_B(ref_B),
    params_E(ref_E.layout()),
    ref_E(ref_E)
  {
    int total_gemm_k_iterations = (problem_size.k() + mma_shape_k - 1) / mma_shape_k;
    int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();

    gemm_k_size = gemm_k_iterations * mma_shape_k;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
