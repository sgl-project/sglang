/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Distributed GEMM barrier kernel.

    The kernel resets the per-stage arrival flags, performs a full barrier (any-to-any),
    and also atomically resets the local barrier arrival count.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/grid_dependency_control.h"

#include "cutlass/experimental/distributed/kernel/detail.hpp"

namespace cutlass::distributed::kernel {

template <int NP, typename IntType, int Iterations, typename FlagType>
__global__ void full_barrier_kernel(
    cutlass::Array<IntType*, NP> device_arrival_ptrs,
    cutlass::Array<FlagType*, Iterations> iteration_flag_ptrs,
    IntType device_idx) {

  arch::launch_dependent_grids();
  arch::wait_on_dependent_grids();

  CUTLASS_PRAGMA_UNROLL
  for (FlagType i = 0; i < Iterations; ++i) {
    iteration_flag_ptrs[i][0] = static_cast<FlagType>(0);
  }

  IntType val = 1;
  IntType max_val = static_cast<IntType>(NP - 1);

  CUTLASS_PRAGMA_UNROLL
  for (IntType d = 0; d < NP; ++d) {
    if (d != device_idx) {
      atomicAdd(device_arrival_ptrs[d], val);
    }
  }

  IntType curr_val = 0;
  detail::ld_without_cache(curr_val, device_arrival_ptrs[device_idx]);
  while (curr_val < max_val) {
    __nanosleep(40);
    detail::ld_without_cache(curr_val, device_arrival_ptrs[device_idx]);
  }

  atomicSub(device_arrival_ptrs[device_idx], max_val);
}

} // namespace cutlass::distributed::kernel

