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
    \brief Device layer interface for Distributed GEMM barrier kernel.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/experimental/distributed/kernel/full_barrier.hpp"

namespace cutlass::distributed::device {

template <int NP, typename IntType, int Iterations, typename FlagType>
void launch_full_barrier(
    cutlass::Array<IntType*, NP> device_arrival_ptrs,
    cutlass::Array<FlagType*, Iterations> iteration_flag_ptrs,
    IntType device_idx,
    cudaStream_t stream,
    bool launch_with_pdl) {

#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 6))
  // Legacy (kernel) launch with PDL
  cudaLaunchAttribute attributes[1];
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t launch_config;
  launch_config.gridDim = 1;
  launch_config.blockDim = 1;
  launch_config.dynamicSmemBytes = 0;
  launch_config.stream = stream;
  launch_config.attrs = attributes;
  launch_config.numAttrs = launch_with_pdl ? 1 : 0;

  cudaLaunchKernelEx(
      &launch_config,
      cutlass::distributed::kernel::full_barrier_kernel<NP, IntType, Iterations, FlagType>,
      device_arrival_ptrs,
      iteration_flag_ptrs,
      device_idx);
#endif
}

} // namespace cutlass::distributed::device

