// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright (c) 2024 by FlashInfer team.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// Adapted from FlashInfer v0.7.2 include/flashinfer/sampling.cuh (Apache-2.0, Copyright (c) 2024 by FlashInfer team):
// verify_softmax_stats_kernel follows its OnlineSoftmaxReduceKernel (via verify_softmax_norm_kernel in verify_softmax_wide.cuh).
// FlashInfer NOTICE: FlashInfer / Copyright 2025-2026 NVIDIA / Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
// Modified in 2026 by NVIDIA CORPORATION & AFFILIATES: writes the row statistics instead of normalizing every column.
#pragma once

// DFLASH sampled verify without the probability tensor: verify_softmax_map_kernel's slice partials, then one CTA per
// row that merges them exactly as verify_softmax_norm_kernel does and writes the row's (max, 1 / denominator,
// 1 / temperature) instead of normalizing every column. A reader that evaluates
//   p = __expf(__fmul_rn(logit, inv_temp) - row_max) * inv_denominator
// with the same instructions (fast-math: mul.rn.ftz, sub.ftz, mul.ftz by log2(e), ex2.approx.ftz, mul.ftz) gets
// verify_softmax_norm_kernel's p bit for bit, for the columns it needs only.
// Synchronization: the map -> stats edge is the map -> norm PDL edge of verify_softmax_wide.cuh (griddepcontrol.wait
// before the first read of partials and temperatures, the trigger after the last store); readers of the stats are
// ordinary stream-ordered launches.

#include "verify_softmax_wide.cuh"

namespace sglang {

struct VswRowStats {
  float row_max;
  float inv_denominator;
  float inv_temp;
  float pad;
};

// verify_softmax_norm_kernel up to its (final_max, inv_denominator), line for line; grid (rows).
template <bool kUsePDL>
__global__ __launch_bounds__(kVswThreads) void verify_softmax_stats_kernel(
    VswRowStats* stats,
    const VswPartial* partial_results,
    const float* temperature_arr,
    uint32_t rows_per_request,
    uint32_t num_slices) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;

  __shared__ VswTempStorage temp_storage;

  const VswFloat2SoftmaxReduceOp reduce_op;

  float2 thread_aggregate = make_float2(-cuda::std::numeric_limits<float>::infinity(), 0.0f);

  device::PDLWaitPrimary<kUsePDL>();
  const float inv_temp = vsw_inv_temp(temperature_arr, bx, rows_per_request);

  for (uint32_t i = tx; i < num_slices; i += kVswThreads) {
    VswPartial partial = partial_results[bx * num_slices + i];
    float2 partial_pair = make_float2(partial.max_val, partial.denominator);
    thread_aggregate = reduce_op(thread_aggregate, partial_pair);
  }

  float2 block_result =
      cub::BlockReduce<float2, kVswThreads>(temp_storage.block_prim.reduce_pair).Reduce(thread_aggregate, reduce_op);

  if (tx == 0) {
    temp_storage.shared_state.max_val = block_result.x;
    temp_storage.shared_state.denominator = block_result.y;
  }
  __syncthreads();

  const float final_max = temp_storage.shared_state.max_val;
  const float inv_denominator = 1.0f / temp_storage.shared_state.denominator;

  if (tx == 0) {
    stats[bx] = {final_max, inv_denominator, inv_temp, 0.0f};
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct VerifySoftmaxStatsKernel {
  // logits fp32 [rows, d] contiguous; stats fp32 [rows, 4]; partials fp32 [rows, num_slices, 2];
  // temperatures fp32 [rows / rows_per_request] contiguous.
  static void
  run(tvm::ffi::TensorView logits,
      tvm::ffi::TensorView stats,
      tvm::ffi::TensorView partials,
      tvm::ffi::TensorView temperatures,
      int64_t rows_per_request) {
    using namespace host;
    const int64_t rows = logits.size(0);
    const int64_t d = logits.size(1);
    RuntimeCheck(rows >= 1 && rows <= 128, "verify_softmax_stats: rows must be in [1, 128], got ", rows);
    RuntimeCheck(d >= 24576 && d % kVswVec == 0, "verify_softmax_stats: d must be >= 24576 and a multiple of 4, got ", d);
    RuntimeCheck(rows_per_request >= 1 && rows % rows_per_request == 0, "verify_softmax_stats: bad rows_per_request");
    RuntimeCheck(temperatures.size(0) * rows_per_request == rows, "verify_softmax_stats: temperatures do not match rows");
    const int64_t num_slices = (d + kVswSlice - 1) / kVswSlice;
    RuntimeCheck(partials.size(0) == rows && partials.size(1) == num_slices, "verify_softmax_stats: bad partials shape");
    RuntimeCheck(stats.size(0) == rows && stats.size(1) == 4, "verify_softmax_stats: bad stats shape");
    const auto device = logits.device();
    const auto* lg = static_cast<const float*>(logits.data_ptr());
    auto* st = static_cast<VswRowStats*>(stats.data_ptr());
    auto* part = static_cast<VswPartial*>(partials.data_ptr());
    const auto* temp = static_cast<const float*>(temperatures.data_ptr());
    const auto rpr = static_cast<uint32_t>(rows_per_request);
    const auto dd = static_cast<uint32_t>(d);
    const auto ns = static_cast<uint32_t>(num_slices);
    LaunchKernel(dim3(static_cast<uint32_t>(rows), ns), kVswThreads, device)  //
        .enable_pdl(kUsePDL)(verify_softmax_map_kernel<kUsePDL>, lg, part, temp, rpr, dd, ns);
    LaunchKernel(dim3(static_cast<uint32_t>(rows)), kVswThreads, device)  //
        .enable_pdl(kUsePDL)(verify_softmax_stats_kernel<kUsePDL>, st, part, temp, rpr, ns);
  }
};

}  // namespace sglang
