// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Framework-agnostic host launcher of the single-launch fused FP8 MQA-logits +
// top-k prefill kernel (SM100 family). Callers (a libtorch-stable binding, a
// TVM-FFI entry) validate their tensors and fill FusedMqaTopkParams with raw
// device pointers; nothing here allocates or synchronizes, so a launch can be
// captured in a CUDA graph. The kernel is persistent_kernel.cuh (one CTA per
// SM; sampled gate, candidate emission and selection; exact rescan of
// out-of-range blocks), for fused_persist::kMinKeys to kMaxKeys keys.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <climits>
#include <cstdint>

#include "persistent_kernel.cuh"

namespace fused_mqa_topk {

constexpr int kHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kCandidates = 16384;  // per-row candidate slab
constexpr int kRetained = 4096;     // per-row recovery slab
constexpr int64_t kMaxKeys = int64_t(1) << 20;

// q [rows,32,128] fp8e4m3, k [keys,128] fp8e4m3, scales [keys] fp32,
// weights [rows,32] fp32, starts/ends [rows] int32 (visible key range
// [start,end) of each row), out [rows,topk] int32 with topk 2048 (GLM-5.2).
// out receives absolute key indices (columns of the logits), or with relative
// indices relative to the row's start (key - start); -1 padded when fewer than
// topk keys are visible. With out_offsets (nullable, [rows] int32, requires
// relative) every valid index is shifted by out_offsets[row] (key - start +
// out_offsets[row]). With page_table (nullable, int32 rows page_table_stride
// elements apart, requires relative and row_batch [rows] int32, excludes
// out_offsets) every valid index rel of row r becomes
// page_table[row_batch[r] * page_table_stride + rel]: the KV slot of key rel
// of the row's request through a page_size = 1 table. Both are applied inside
// the kernel.
// Workspace (caller-owned, reused across calls on ONE stream), with
// B = ceil(rows / 8) and slots >= 2 per CTA of the persistent grid:
//   counter [2] int32 (zero between calls; the kernel leaves it zeroed),
//   values [8*slots,16384] fp16, indices [8*slots,16384] int32,
//   retained [8*slots,4096] fp32, retained_indices [8*slots,4096] int32,
//   counts [>=8B] int32, defer [>=B] int32.
struct FusedMqaTopkParams {
  const void* q;
  const void* k;
  const float* scales;
  const float* weights;
  const int32_t* starts;
  const int32_t* ends;
  const int32_t* out_offsets;
  const int32_t* page_table;
  int64_t page_table_stride;
  const int32_t* row_batch;
  int32_t* out;
  int64_t rows;
  int64_t keys;
  int64_t topk;
  bool relative;
  int32_t* counter;
  uint16_t* values;
  int32_t* indices;
  float* retained;
  int32_t* retained_indices;
  int32_t* counts;
  int32_t* defer;
  int64_t slots;  // rows of values / 8
  int num_sms;
  cudaStream_t stream;
};

namespace detail {

inline void* driver_handle() {
  static void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  return h;
}

inline CUresult enc_tiled(CUtensorMap* tm, CUtensorMapDataType dt,
                          cuuint32_t rank, void* addr, const cuuint64_t* dims,
                          const cuuint64_t* strides, const cuuint32_t* box,
                          const cuuint32_t* estrides, CUtensorMapInterleave il,
                          CUtensorMapSwizzle sw, CUtensorMapL2promotion l2,
                          CUtensorMapFloatOOBfill oob) {
  using FT =
      CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                   const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                   const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  static FT f = driver_handle()
                    ? reinterpret_cast<FT>(
                          dlsym(driver_handle(), "cuTensorMapEncodeTiled"))
                    : nullptr;
  if (!f) return CUDA_ERROR_NOT_FOUND;
  return f(tm, dt, rank, addr, dims, strides, box, estrides, il, sw, l2, oob);
}

inline CUresult make_2d(CUtensorMap* tm, const void* ptr,
                        CUtensorMapDataType dt, int elem_size, int gmem_inner,
                        int gmem_outer, int smem_inner, int smem_outer,
                        long gmem_outer_stride, int swizzle_mode) {
  if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
  const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
  const cuuint32_t sdims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
  const cuuint64_t gstrides[1] = {(cuuint64_t)(gmem_outer_stride * elem_size)};
  const cuuint32_t estrides[2] = {1, 1};
  const CUtensorMapSwizzle swizzle =
      swizzle_mode == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
      : swizzle_mode == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
      : swizzle_mode == 32 ? CU_TENSOR_MAP_SWIZZLE_32B
                           : CU_TENSOR_MAP_SWIZZLE_NONE;
  return enc_tiled(tm, dt, 2, const_cast<void*>(ptr), gdims, gstrides, sdims,
                   estrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// KV [keys,128] bytes viewed as {128, groups, stride} with strides
// {128*stride, 128}: box {128, 256, 1} at (0, 0, o) loads keys o, o+stride, ...
// with 128B swizzle.
inline CUresult make_strided_kv(CUtensorMap* tm, const void* ptr,
                                unsigned groups, unsigned stride) {
  const cuuint64_t gdims[3] = {128, groups, stride};
  const cuuint64_t gstrides[2] = {128ull * stride, 128};
  const cuuint32_t box[3] = {128, 256, 1};
  const cuuint32_t estrides[3] = {1, 1, 1};
  return enc_tiled(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(ptr),
                   gdims, gstrides, box, estrides,
                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

}  // namespace detail

// Shape/alignment preconditions the kernels rely on; the caller has already
// checked dtypes, devices and contiguity.
inline bool params_supported(const FusedMqaTopkParams& p) {
  const auto aligned = [](const void* ptr, uintptr_t a) {
    return reinterpret_cast<uintptr_t>(ptr) % a == 0;
  };
  // rows * 32 is the int32 outer dimension / coordinate of the Q tensor map.
  return p.rows > 0 && p.rows * kHeads <= INT32_MAX &&
         p.keys >= fused_persist::kMinKeys && p.keys <= kMaxKeys &&
         p.topk == fused_persist::kTopK &&
         (p.out_offsets == nullptr || p.relative) &&
         (p.page_table == nullptr ||
          (p.relative && p.out_offsets == nullptr && p.row_batch != nullptr &&
           p.page_table_stride >= 0)) &&
         aligned(p.q, 16) && aligned(p.k, 16) && aligned(p.scales, 16) &&
         aligned(p.weights, 16) && aligned(p.out, 16);
}

// Returns cudaErrorInvalidValue for unsupported parameters (nothing launched)
// and the launch error otherwise.
inline cudaError_t launch(const FusedMqaTopkParams& p) {
  if (!params_supported(p)) return cudaErrorInvalidValue;
  const unsigned urows = static_cast<unsigned>(p.rows),
                 ukeys = static_cast<unsigned>(p.keys),
                 ublocks = static_cast<unsigned>((p.rows + 7) / 8);
  const auto* starts_p = reinterpret_cast<const unsigned*>(p.starts);
  const auto* ends_p = reinterpret_cast<const unsigned*>(p.ends);
  const PageTable paged{p.page_table, p.page_table_stride, p.row_batch};
  CUtensorMap tq, tk, ts, tw;
  if (detail::make_2d(&tq, p.q, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, 128,
                      int(p.rows) * 32, 128, 256, 128, 128) != CUDA_SUCCESS ||
      detail::make_2d(&tk, p.k, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, 128,
                      int(p.keys), 128, 256, 128, 128) != CUDA_SUCCESS ||
      detail::make_2d(&ts, p.scales, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
                      int(p.keys), 1, 256, 1, 0, 0) != CUDA_SUCCESS ||
      detail::make_2d(&tw, p.weights, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, 32,
                      int(p.rows), 32, 8, 32, 0) != CUDA_SUCCESS)
    return cudaErrorInvalidValue;

  if (p.slots < 2 || p.slots % 2 != 0) return cudaErrorInvalidValue;
  // One persistent CTA per SM (capped by the blocks and the workspace
  // slots); CTA c owns candidate slots 2c and 2c + 1.
  const unsigned grid = static_cast<unsigned>(
      std::min<int64_t>({int64_t(p.num_sms), int64_t(ublocks), p.slots / 2}));
  fused_persist::LaunchConfig c;
  if (const cudaError_t e = fused_persist::launch_config(ukeys, c);
      e != cudaSuccess)
    return e;
  CUtensorMap tks;
  if (detail::make_strided_kv(&tks, p.k, c.sample_groups, c.sample_stride) !=
      CUDA_SUCCESS)
    return cudaErrorInvalidValue;
  const FallbackBuffers fallback{p.retained, p.retained_indices, p.out};
  c.kernel<<<grid, 384, c.smem, p.stream>>>(
      urows, ukeys, starts_p, ends_p, p.values, p.indices, p.counts, fallback,
      c.phase_bytes, c.sample_target, c.sample_stride, c.sample_groups,
      c.sample_tiles, p.scales, tks, tq, ts, tk, tw, p.relative,
      p.out_offsets, ublocks, reinterpret_cast<unsigned*>(p.counter), p.defer,
      paged);
  return cudaGetLastError();
}

}  // namespace fused_mqa_topk
