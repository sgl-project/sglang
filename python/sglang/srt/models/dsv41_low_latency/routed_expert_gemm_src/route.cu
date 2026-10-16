// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Route kernel of the DeepSeek-V4.1 low-batch routed-expert GEMMs
// (routed_expert_gemm.py): builds the routing record (route.cuh) from
// precomputed top-k ids.
//
// PDL: dependents may launch at once (the trigger is the first instruction), so
// GEMM1's set-up overlaps this kernel; they read its outputs only after their
// own grid-dependency wait, which returns once this grid has completed. Every
// global access here follows this kernel's own grid-dependency wait.

#include "route.cuh"

namespace {
constexpr int kThreads = 64; // >= rows * top_k (at most 8 * 8 slots)
} // namespace

// experts: a multiple of 32, at most 384; rows <= 8; rows * top_k <= 64.
extern "C" __global__
__launch_bounds__(kThreads) void dsv41_routed_expert_route(
    const int *ids, int *__restrict__ permuted, int *__restrict__ record,
    unsigned int *barrier, int rows, int top_k, int experts) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __shared__ route::Scratch scratch;
  const int thread = static_cast<int>(threadIdx.x);
  route::Clear(scratch, thread, kThreads);
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  route::Zero(record, rows, top_k, experts, thread, kThreads);
  route::Build(scratch, ids, permuted, record, barrier, rows, top_k, experts,
               thread);
}
