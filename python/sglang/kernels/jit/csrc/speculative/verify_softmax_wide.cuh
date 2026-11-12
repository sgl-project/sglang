// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright (c) 2024 by FlashInfer team.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// Adapted from FlashInfer v0.7.2 include/flashinfer/sampling.cuh (Apache-2.0, Copyright (c) 2024 by FlashInfer team):
// verify_softmax_map_kernel / verify_softmax_norm_kernel follow its OnlineSoftmaxMapKernel / OnlineSoftmaxReduceKernel.
// FlashInfer NOTICE: FlashInfer / Copyright 2025-2026 NVIDIA / Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
// Modified in 2026 by NVIDIA CORPORATION & AFFILIATES: the two changes listed below.
#pragma once

// DFLASH sampled verify: the temperature-scaled target softmax over [rows, d] fp32 logits, bitwise equal to
// flashinfer 0.7.2's OnlineSoftmax "Path A" (batch <= 128, d >= 24576): OnlineSoftmaxMapKernel then
// OnlineSoftmaxReduceKernel with BLOCK_THREADS 1024, VEC_SIZE 4 and 8192-wide slices, compiled with
// -use_fast_math as flashinfer's sampling module is. The arithmetic below is flashinfer's, line for line.
// Two changes, neither touches a value:
//   1. row r reads its temperature at temperatures[r / rows_per_request] (flashinfer reads a per-row copy that
//      torch.repeat_interleave makes first; the float is the same);
//   2. the normalization pass runs on a (row, 4096-column tile) grid instead of one 1024-thread CTA per row.
//      Every tile CTA repeats the row's 1024-thread cub merge of the same slice partials, so every CTA holds the
//      same (max, denominator) bits, and p is elementwise in (logit, max, 1 / denominator).
// Synchronization: PDL edges only. Each kernel executes griddepcontrol.wait before its first read of producer
// data (logits, temperatures, partials) and triggers its dependents after its last store. No __restrict__ on the
// pointers: their loads must stay coherent ld.global (no ld.global.nc on data a predecessor grid writes).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cuda/std/limits>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

constexpr uint32_t kVswThreads = 1024;                 // flashinfer DISPATCH_COMPUTE_CAP_NUM_THREADS (cc >= 8)
constexpr uint32_t kVswVec = 4;                        // gcd(16 / sizeof(float), d) for d % 4 == 0
constexpr uint32_t kVswSlice = 8192;                   // flashinfer DEFAULT_SLICE_SIZE
constexpr uint32_t kVswTile = kVswThreads * kVswVec;   // columns per normalization CTA (one flashinfer pass)

struct VswPartial {  // flashinfer PartialSoftmaxResult
  float max_val;
  float denominator;
};

struct VswFloat2SoftmaxReduceOp {  // flashinfer Float2SoftmaxReduceOp
  __device__ __forceinline__ float2 operator()(const float2& a, const float2& b) const {
    if (isinf(a.x)) return b;
    if (isinf(b.x)) return a;

    float new_max = max(a.x, b.x);
    float new_denom = a.y * __expf(a.x - new_max) + b.y * __expf(b.x - new_max);
    return make_float2(new_max, new_denom);
  }
};

struct VswTempStorage {  // flashinfer OnlineSoftmaxTempStorage<1024>
  union {
    typename cub::BlockReduce<float, kVswThreads>::TempStorage reduce;
    typename cub::BlockReduce<float2, kVswThreads>::TempStorage reduce_pair;
  } block_prim;

  struct {
    float max_val;
    float denominator;
  } shared_state;
};

SGL_DEVICE void vsw_load4(float (&v)[kVswVec], const float* ptr) {
  const float4 x = *reinterpret_cast<const float4*>(ptr);
  v[0] = x.x;
  v[1] = x.y;
  v[2] = x.z;
  v[3] = x.w;
}

SGL_DEVICE void vsw_store4(float* ptr, const float (&v)[kVswVec]) {
  *reinterpret_cast<float4*>(ptr) = make_float4(v[0], v[1], v[2], v[3]);
}

// The request's temperature, as flashinfer computes inv_temp from its per-row copy.
SGL_DEVICE float vsw_inv_temp(const float* temperature_arr, uint32_t row, uint32_t rows_per_request) {
  float temperature = temperature_arr[row / rows_per_request];
  return (temperature == 0.f) ? 0.f : 1.f / temperature;
}

// flashinfer OnlineSoftmaxMapKernel<1024, 4, float>; grid (rows, num_slices).
template <bool kUsePDL>
__global__ __launch_bounds__(kVswThreads) void verify_softmax_map_kernel(
    const float* logits,
    VswPartial* partial_results,
    const float* temperature_arr,
    uint32_t rows_per_request,
    uint32_t d,
    uint32_t num_slices) {
  const uint32_t bx = blockIdx.x;
  const uint32_t by = blockIdx.y;  // slice index
  const uint32_t tx = threadIdx.x;

  const uint32_t slice_stride = (((d + num_slices - 1) / num_slices) + kVswVec - 1) / kVswVec * kVswVec;
  const uint32_t slice_start = by * slice_stride;
  const uint32_t slice_size = min((by + 1) * slice_stride, d) - slice_start;

  if (slice_start >= d) return;

  __shared__ VswTempStorage temp_storage;

  float logits_vec[kVswVec];
  float running_max = -cuda::std::numeric_limits<float>::infinity();
  float running_denominator = 0.0f;
  float threadlocal_running_denominator = 0.0f;

  device::PDLWaitPrimary<kUsePDL>();
  const float inv_temp = vsw_inv_temp(temperature_arr, bx, rows_per_request);

#pragma unroll 2
  for (uint32_t i = 0; i < (slice_size + kVswThreads * kVswVec - 1) / (kVswThreads * kVswVec); ++i) {
#pragma unroll
    for (uint32_t j = 0; j < kVswVec; ++j) {
      logits_vec[j] = -cuda::std::numeric_limits<float>::infinity();
    }

    if ((i * kVswThreads + tx) * kVswVec < slice_size) {
      vsw_load4(logits_vec, logits + static_cast<uint64_t>(bx) * d + slice_start + (i * kVswThreads + tx) * kVswVec);
    }

    float thread_max = -cuda::std::numeric_limits<float>::infinity();
#pragma unroll
    for (uint32_t j = 0; j < kVswVec; ++j) {
      // __fmul_rn, not *: it must not be contracted into an FMA (flashinfer, PR #5088).
      logits_vec[j] = __fmul_rn(static_cast<float>(logits_vec[j]), inv_temp);
      thread_max = max(thread_max, logits_vec[j]);
    }

    float block_max =
        cub::BlockReduce<float, kVswThreads>(temp_storage.block_prim.reduce).Reduce(thread_max, cuda::maximum<>{});

    if (tx == 0) {
      temp_storage.shared_state.max_val = block_max;
    }
    __syncthreads();
    block_max = temp_storage.shared_state.max_val;

    // if block_max is -inf, then this block contains all -inf values, so we can skip updating
    if (!isinf(block_max)) {
      float threadlocal_sum = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < kVswVec; ++j) {
        threadlocal_sum += __expf(logits_vec[j] - block_max);
      }
      float new_max = max(running_max, block_max);
      threadlocal_running_denominator = threadlocal_running_denominator * __expf(running_max - new_max) +
                                        threadlocal_sum * __expf(block_max - new_max);
      running_max = new_max;
    }
  }

  running_denominator =
      cub::BlockReduce<float, kVswThreads>(temp_storage.block_prim.reduce).Sum(threadlocal_running_denominator);
  if (tx == 0) {
    temp_storage.shared_state.denominator = running_denominator;
  }
  __syncthreads();
  running_denominator = temp_storage.shared_state.denominator;

  if (tx == 0) {
    partial_results[bx * num_slices + by] = {running_max, running_denominator};
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// flashinfer OnlineSoftmaxReduceKernel<1024, 4, float>, split over column tiles; grid (rows, ceil(d / 4096)).
template <bool kUsePDL>
__global__ __launch_bounds__(kVswThreads) void verify_softmax_norm_kernel(
    const float* logits,
    float* output,
    const VswPartial* partial_results,
    const float* temperature_arr,
    uint32_t rows_per_request,
    uint32_t d,
    uint32_t num_slices) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tile = blockIdx.y;
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

  // One flashinfer normalization pass (i = tile): columns [tile * 4096, tile * 4096 + 4096).
  const uint32_t col = (tile * kVswThreads + tx) * kVswVec;
  float logits_vec[kVswVec];
  float prob_vec[kVswVec];
#pragma unroll
  for (uint32_t j = 0; j < kVswVec; ++j) {
    logits_vec[j] = -cuda::std::numeric_limits<float>::infinity();
  }
  if (col < d) {
    vsw_load4(logits_vec, logits + static_cast<uint64_t>(bx) * d + col);
  }
#pragma unroll
  for (uint32_t j = 0; j < kVswVec; ++j) {
    // __fmul_rn, not *: it must not be contracted into an FMA (flashinfer, PR #5088).
    logits_vec[j] = __fmul_rn(static_cast<float>(logits_vec[j]), inv_temp);
    float p = __expf(static_cast<float>(logits_vec[j]) - final_max) * inv_denominator;
    prob_vec[j] = static_cast<float>(p);
  }
  if (col < d) {
    vsw_store4(output + static_cast<uint64_t>(bx) * d + col, prob_vec);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct VerifySoftmaxWideKernel {
  // logits fp32 [rows, d] contiguous; output fp32 [rows, d] contiguous; partials fp32 [rows, num_slices, 2];
  // temperatures fp32 [rows / rows_per_request] contiguous.
  static void
  run(tvm::ffi::TensorView logits,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView partials,
      tvm::ffi::TensorView temperatures,
      int64_t rows_per_request) {
    using namespace host;
    const int64_t rows = logits.size(0);
    const int64_t d = logits.size(1);
    RuntimeCheck(rows >= 1 && rows <= 128, "verify_softmax_wide: rows must be in [1, 128], got ", rows);
    RuntimeCheck(d >= 24576 && d % kVswVec == 0, "verify_softmax_wide: d must be >= 24576 and a multiple of 4, got ", d);
    RuntimeCheck(rows_per_request >= 1 && rows % rows_per_request == 0, "verify_softmax_wide: bad rows_per_request");
    RuntimeCheck(temperatures.size(0) * rows_per_request == rows, "verify_softmax_wide: temperatures do not match rows");
    const int64_t num_slices = (d + kVswSlice - 1) / kVswSlice;
    RuntimeCheck(partials.size(0) == rows && partials.size(1) == num_slices, "verify_softmax_wide: bad partials shape");
    RuntimeCheck(output.size(0) == rows && output.size(1) == d, "verify_softmax_wide: bad output shape");
    const auto device = logits.device();
    const auto* lg = static_cast<const float*>(logits.data_ptr());
    auto* out = static_cast<float*>(output.data_ptr());
    auto* part = static_cast<VswPartial*>(partials.data_ptr());
    const auto* temp = static_cast<const float*>(temperatures.data_ptr());
    const auto rpr = static_cast<uint32_t>(rows_per_request);
    const auto dd = static_cast<uint32_t>(d);
    const auto ns = static_cast<uint32_t>(num_slices);
    LaunchKernel(dim3(static_cast<uint32_t>(rows), ns), kVswThreads, device)  //
        .enable_pdl(kUsePDL)(verify_softmax_map_kernel<kUsePDL>, lg, part, temp, rpr, dd, ns);
    const auto tiles = static_cast<uint32_t>((d + kVswTile - 1) / kVswTile);
    LaunchKernel(dim3(static_cast<uint32_t>(rows), tiles), kVswThreads, device)  //
        .enable_pdl(kUsePDL)(verify_softmax_norm_kernel<kUsePDL>, lg, out, part, temp, rpr, dd, ns);
  }
};

}  // namespace sglang
