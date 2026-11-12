// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

// DFLASH sampled verify row statistics at any row count (SGLANG_OPT_DFLASH_VERIFY_ROW_STATS). One read of fp32 logits
// [rows, d] gives each row the VswRowStats {row_max, inv_denominator, inv_temp, 0} that the selector logit accept reads,
// with inv_temp = 1 / temperatures[r / rows_per_request] (0 for temperature 0, as flashinfer computes it):
//   row_max         = max over c of __fmul_rn(logit[r, c], inv_temp)                     (exact: order does not matter)
//   inv_denominator = 1 / (sum over c of __expf(__fmul_rn(logit[r, c], inv_temp) - row_max))   (fp32, this kernel's order)
// row_max and inv_temp carry the served softmax's bits. The denominator is the same fp32 sum taken in another order, so p
// evaluated from these stats differs from the served p by the denominator's relative difference plus one rounding.
// Layout: rs_partial_kernel runs one 256-thread CTA per (row, 4096-column slice). Each thread issues its four float4 loads
// before using any, keeps its 16 values in registers (its max first, then one __expf per value), and the CTA merges
// (max, sum) pairs with warp shuffles and one shared-memory step. rs_combine_kernel merges a row's slice pairs in one warp.
// Synchronization: PDL edges only. Each kernel executes griddepcontrol.wait before its first read of producer data
// (logits, temperatures, partials) and triggers its dependents after its last store; no flags, atomics or polls. No
// __restrict__ on the pointers: their loads stay coherent ld.global (no ld.global.nc on data a predecessor grid writes).

#include "verify_softmax_stats.cuh"

namespace sglang {

constexpr uint32_t kRsThreads = 256;
constexpr uint32_t kRsLoads = 4;                                // float4 loads per thread per slice
constexpr uint32_t kRsSlice = kRsThreads * kRsLoads * kVswVec;  // 4096 columns per partial CTA
constexpr uint32_t kRsRowsPerCombineCta = kRsThreads / 32;      // one warp per row

struct RsPair {
  float max_val;
  float sum;
};

// (max, sum of exp(x - max)) of the union of two sets; a side whose max is -inf (empty, or every value -inf) adds nothing.
SGL_DEVICE RsPair rs_merge(const RsPair a, const RsPair b) {
  if (a.max_val == -cuda::std::numeric_limits<float>::infinity()) return b;
  if (b.max_val == -cuda::std::numeric_limits<float>::infinity()) return a;
  const float m = fmaxf(a.max_val, b.max_val);
  return {m, a.sum * __expf(a.max_val - m) + b.sum * __expf(b.max_val - m)};
}

SGL_DEVICE RsPair rs_warp_merge(RsPair v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const RsPair other{__shfl_xor_sync(0xffffffffu, v.max_val, offset), __shfl_xor_sync(0xffffffffu, v.sum, offset)};
    v = rs_merge(v, other);
  }
  return v;
}

// grid (rows, num_slices): partials[row * num_slices + slice] = the slice's (max, sum).
template <bool kUsePDL>
__global__ __launch_bounds__(kRsThreads) void rs_partial_kernel(
    RsPair* partials,
    const float* logits,
    const float* temperature_arr,
    uint32_t rows_per_request,
    uint32_t d,
    uint32_t num_slices) {
  const uint32_t row = blockIdx.x;
  const uint32_t slice = blockIdx.y;
  const uint32_t tx = threadIdx.x;
  __shared__ RsPair warp_pairs[kRsThreads / 32];
  constexpr float kNegInf = -cuda::std::numeric_limits<float>::infinity();

  device::PDLWaitPrimary<kUsePDL>();
  const float slice_inv_temp = vsw_inv_temp(temperature_arr, row, rows_per_request);
  const float* base = logits + static_cast<uint64_t>(row) * d + static_cast<uint64_t>(slice) * kRsSlice;
  const uint32_t cols = min(kRsSlice, d - slice * kRsSlice);  // a multiple of kVswVec

  float v[kRsLoads][kVswVec];
#pragma unroll
  for (uint32_t k = 0; k < kRsLoads; ++k) {
    const uint32_t c = (k * kRsThreads + tx) * kVswVec;
    if (c < cols) {
      vsw_load4(v[k], base + c);
#pragma unroll
      for (uint32_t j = 0; j < kVswVec; ++j) v[k][j] = __fmul_rn(v[k][j], slice_inv_temp);
    } else {
#pragma unroll
      for (uint32_t j = 0; j < kVswVec; ++j) v[k][j] = kNegInf;
    }
  }
  float thread_max = kNegInf;
#pragma unroll
  for (uint32_t k = 0; k < kRsLoads; ++k)
#pragma unroll
    for (uint32_t j = 0; j < kVswVec; ++j) thread_max = fmaxf(thread_max, v[k][j]);
  float thread_sum = 0.0f;
  if (thread_max != kNegInf) {
#pragma unroll
    for (uint32_t k = 0; k < kRsLoads; ++k)
#pragma unroll
      for (uint32_t j = 0; j < kVswVec; ++j) thread_sum += __expf(v[k][j] - thread_max);
  }

  const RsPair warp_pair = rs_warp_merge({thread_max, thread_sum});
  if ((tx & 31) == 0) warp_pairs[tx >> 5] = warp_pair;
  __syncthreads();
  if (tx < 32) {
    const RsPair cta_pair = rs_warp_merge(tx < kRsThreads / 32 ? warp_pairs[tx] : RsPair{kNegInf, 0.0f});
    if (tx == 0) partials[static_cast<uint64_t>(row) * num_slices + slice] = cta_pair;
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// grid (ceil(rows / 8)), one warp per row: stats[row] = (max, 1 / sum, inv_temp, 0) over the row's slice pairs.
template <bool kUsePDL>
__global__ __launch_bounds__(kRsThreads) void rs_combine_kernel(
    VswRowStats* stats,
    const RsPair* partials,
    const float* temperature_arr,
    uint32_t rows_per_request,
    uint32_t rows,
    uint32_t num_slices) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t row = blockIdx.x * kRsRowsPerCombineCta + (threadIdx.x >> 5);

  device::PDLWaitPrimary<kUsePDL>();
  if (row < rows) {  // the whole warp takes the same branch
    RsPair acc{-cuda::std::numeric_limits<float>::infinity(), 0.0f};
    for (uint32_t i = lane; i < num_slices; i += 32) acc = rs_merge(acc, partials[static_cast<uint64_t>(row) * num_slices + i]);
    acc = rs_warp_merge(acc);
    if (lane == 0) {
      const float row_inv_temp = vsw_inv_temp(temperature_arr, row, rows_per_request);
      stats[row] = {acc.max_val, 1.0f / acc.sum, row_inv_temp, 0.0f};
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct VerifyRowStatsKernel {
  // logits fp32 [rows, d] contiguous, 16-byte aligned; stats fp32 [rows, 4]; partials fp32 [rows, num_slices, 2];
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
    RuntimeCheck(rows >= 1 && rows <= 0x7fffffff, "verify_row_stats: bad rows ", rows);
    RuntimeCheck(d >= kVswVec && d % kVswVec == 0 && d <= 0x7fffffff, "verify_row_stats: d must be a multiple of 4, got ", d);
    RuntimeCheck(reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0, "verify_row_stats: logits must be 16-byte aligned");
    RuntimeCheck(rows_per_request >= 1 && rows % rows_per_request == 0, "verify_row_stats: bad rows_per_request");
    RuntimeCheck(temperatures.size(0) * rows_per_request == rows, "verify_row_stats: temperatures do not match rows");
    const int64_t num_slices = (d + kRsSlice - 1) / kRsSlice;
    RuntimeCheck(num_slices <= 65535, "verify_row_stats: d too large");
    RuntimeCheck(
        partials.size(0) == rows && partials.size(1) == num_slices && partials.size(2) == 2,
        "verify_row_stats: bad partials shape");
    RuntimeCheck(stats.size(0) == rows && stats.size(1) == 4, "verify_row_stats: bad stats shape");
    const auto device = logits.device();
    const auto* lg = static_cast<const float*>(logits.data_ptr());
    auto* st = static_cast<VswRowStats*>(stats.data_ptr());
    auto* part = static_cast<RsPair*>(partials.data_ptr());
    const auto* temp = static_cast<const float*>(temperatures.data_ptr());
    const auto rpr = static_cast<uint32_t>(rows_per_request);
    const auto r32 = static_cast<uint32_t>(rows);
    const auto dd = static_cast<uint32_t>(d);
    const auto ns = static_cast<uint32_t>(num_slices);
    LaunchKernel(dim3(r32, ns), kRsThreads, device)  //
        .enable_pdl(kUsePDL)(rs_partial_kernel<kUsePDL>, part, lg, temp, rpr, dd, ns);
    LaunchKernel(dim3((r32 + kRsRowsPerCombineCta - 1) / kRsRowsPerCombineCta), kRsThreads, device)  //
        .enable_pdl(kUsePDL)(rs_combine_kernel<kUsePDL>, st, part, temp, rpr, r32, ns);
  }
};

}  // namespace sglang
