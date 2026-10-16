// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Top-k finalize of the DeepSeek-V4.1 low-batch routed-expert GEMMs
// (routed_expert_gemm.py), for the layers whose MoE output is not handed to the
// fused all-reduce + mHC (their finalize is not deferred):
//
//   out[t, h] = BF16( sum_k  w[t, k] * rows[permuted[t, k], h] )
//
// with the FP32 sum accumulated in slot order k = 0..5 by fused multiply-adds
// and slots whose permuted row is -1 skipped. This is the arithmetic of
// FlashInfer's TRT-LLM-gen finalize kernel for small batches (one thread per
// output value, grid hidden / 256 x tokens), so the output matches the
// do_finalize=True path given the same GEMM2 rows.
//
// PDL: waits for GEMM2's grid before reading; triggers its dependents only on
// completion (the next kernel may not wait).

#include <cuda_bf16.h>

namespace {
constexpr int kHidden = 5120;
constexpr int kThreads = 256;
static_assert(kHidden % kThreads == 0, "one thread per output value");
} // namespace

// rows: BF16 [*, 5120] GEMM2 rows; permuted, weights: [tokens, top_k];
// out: BF16 [tokens, 5120]. Grid (5120 / 256, tokens).
extern "C" __global__
__launch_bounds__(kThreads) void dsv41_routed_expert_finalize(
    const __nv_bfloat16 *__restrict__ rows, const int *__restrict__ permuted,
    const float *__restrict__ weights, __nv_bfloat16 *__restrict__ out,
    int top_k) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int token = static_cast<int>(blockIdx.y);
  const int hidden = static_cast<int>(threadIdx.x + blockDim.x * blockIdx.x);
  float data = 0.0F;
  for (int k = 0; k < top_k; ++k) {
    const int slot = token * top_k + k;
    const int row = permuted[slot];
    if (row == -1)
      continue;
    data = fmaf(weights[slot], __bfloat162float(rows[row * kHidden + hidden]),
                data);
  }
  out[token * kHidden + hidden] = __float2bfloat16(data);
}
