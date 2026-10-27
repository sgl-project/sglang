// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Host launch of the persistent kernel: the TMA tensor maps, the gate sample
// and shared-memory configuration for the call's key count, and the launch.
// Nothing here allocates or synchronizes, so a launch can be captured in a
// CUDA graph.
#pragma once

#include "constants.h"
#include "persistent_kernel.cuh"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <mutex>

namespace sglang::fused_mqa_topk {

/**
 * \brief Device pointers and sizes of one launch; the host entry
 * (dsa/fused_mqa_topk.cuh) documents the tensor contract.
 *
 * out receives key indices relative to the row's start (`relative`), shifted
 * by out_offsets[row] when given, or mapped through the page_size = 1 table
 * (page_table[row_batch[r] * page_table_stride + rel]) when given. The
 * workspace (counter .. defer) is reused across calls on one stream; slots is
 * the number of kBlockRows-row slots of the candidate / retained slabs.
 */
struct LaunchParams {
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
  int64_t slots;
  int32_t num_sms;
  cudaStream_t stream;
};

namespace details {

using EncodeTiled = decltype(&cuTensorMapEncodeTiled);

// cuTensorMapEncodeTiled from the driver, resolved once through the runtime
// (no link-time libcuda dependency).
inline EncodeTiled encode_tiled() {
  static const EncodeTiled fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    const cudaError_t err =
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &status);
    return err == cudaSuccess && status == cudaDriverEntryPointSuccess ? reinterpret_cast<EncodeTiled>(ptr) : nullptr;
  }();
  return fn;
}

// Adapted from DeepGEMM's make_tma_2d_desc: a gmem_outer x gmem_inner tensor
// loaded in smem_outer x smem_inner boxes (swizzle_mode bytes per swizzled row).
inline CUresult make_2d(
    CUtensorMap* tm,
    const void* ptr,
    CUtensorMapDataType dt,
    int32_t elem_size,
    int32_t gmem_inner,
    int32_t gmem_outer,
    int32_t smem_inner,
    int32_t smem_outer,
    int64_t gmem_outer_stride,
    int32_t swizzle_mode) {
  const auto encode = encode_tiled();
  if (!encode) return CUDA_ERROR_NOT_FOUND;
  if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
  const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
  const cuuint32_t sdims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
  const cuuint64_t gstrides[1] = {(cuuint64_t)(gmem_outer_stride * elem_size)};
  const cuuint32_t estrides[2] = {1, 1};
  const CUtensorMapSwizzle swizzle = swizzle_mode == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
                                     : swizzle_mode == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                                     : swizzle_mode == 32 ? CU_TENSOR_MAP_SWIZZLE_32B
                                                          : CU_TENSOR_MAP_SWIZZLE_NONE;
  return encode(
      tm,
      dt,
      2,
      const_cast<void*>(ptr),
      gdims,
      gstrides,
      sdims,
      estrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// KV [keys, 128] bytes viewed as {128, groups, stride} with strides
// {128 * stride, 128}: box {128, 256, 1} at (0, 0, o) loads keys o, o + stride,
// ... with 128B swizzle. The view spans groups * stride keys, more than `keys`
// by < stride / 2; the kernel only loads offset o = stride / 2, whose last key
// is < keys (launch_config sizes `groups` for that).
inline CUresult make_strided_kv(CUtensorMap* tm, const void* ptr, uint32_t groups, uint32_t stride) {
  const auto encode = encode_tiled();
  if (!encode) return CUDA_ERROR_NOT_FOUND;
  const cuuint64_t gdims[3] = {kHeadDim, groups, stride};
  const cuuint64_t gstrides[2] = {uint64_t(kHeadDim) * stride, kHeadDim};
  const cuuint32_t box[3] = {kHeadDim, kTileKeys, 1};
  const cuuint32_t estrides[3] = {1, 1, 1};
  return encode(
      tm,
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      3,
      const_cast<void*>(ptr),
      gdims,
      gstrides,
      box,
      estrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// Dynamic shared memory of the kernel on `device`: everything past its static
// shared memory up to the device's opt-in limit (the selection scratch uses
// it), at least the exact rescan's layout. Sets the opt-in attribute; both are
// computed once per device.
inline cudaError_t kernel_smem(int32_t device, int32_t& smem) {
  // The exact rescan reuses the same dynamic shared memory with its own layout.
  constexpr int32_t kRescanSmem = int32_t(sizeof(dense::SharedStorage) + sizeof(rescan::RescanShared));
  constexpr int32_t kScanSmem = int32_t(sizeof(KernelSmem) + std::max(sizeof(StageLists), sizeof(SampleShared)));
  constexpr int32_t kSelectSmem = int32_t(kSelBase) + int32_t(kMathWarps * row_select::kStreamWords * 4);
  const void* kernel = reinterpret_cast<const void*>(fused_mqa_topk_prefill_kernel);
  static std::mutex mutex;
  static std::map<int32_t, int32_t> cache;
  const std::lock_guard<std::mutex> lock(mutex);
  if (const auto it = cache.find(device); it != cache.end()) {
    smem = it->second;
    return cudaSuccess;
  }
  int32_t max_smem = 0;
  if (const cudaError_t e = cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      e != cudaSuccess)
    return e;
  cudaFuncAttributes attr;
  if (const cudaError_t e = cudaFuncGetAttributes(&attr, kernel); e != cudaSuccess) return e;
  const int32_t bytes = std::max(kRescanSmem, max_smem - int32_t(attr.sharedSizeBytes));
  if (bytes < std::max(kScanSmem, kSelectSmem)) return cudaErrorInvalidConfiguration;
  if (const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
      e != cudaSuccess)
    return e;
  cache.emplace(device, bytes);
  smem = bytes;
  return cudaSuccess;
}

// Gate sample: sample_tiles tiles of kTileKeys keys on a uniform grid (stride
// sample_stride, sample_groups groups in view).
struct SampleConfig {
  uint32_t target, stride, groups, tiles;
};

// The gate sample for `keys` keys. The gate aims for `target` candidates per
// row and sizes the sample so its quantile rank is `rank` (count error
// ~1/sqrt(rank)); a row whose count falls below the top-k is rescanned
// exactly. Above 128K keys the sample is clamped to 16-48 tiles (rank ~48 at
// 1M keys): a sample tile costs about two scan tiles. A rescan runs as the
// last work of its CTA (one block at 1M keys costs ~1/3 of the call), so the
// gates were chosen on B200 for no rescan over GLM-5.2 layers 0, 1, 2, 6, 10
// and 14 (real dumps, 8192 rows, and layer-0 captures, 16384 rows causal /
// full) at the smallest worst-layer cost.
inline SampleConfig sample_config(uint32_t keys) {
  struct Gate {
    uint32_t target, rank;
  };
  constexpr Gate kGates[4] = {
      // <= 32K     <= 64K      <= 128K     > 128K
      {2816, 128},
      {3328, 64},
      {3584, 64},
      {4096, 64}};
  const bool long_ctx = keys > 131072;
  const Gate gate = kGates[keys <= 32768 ? 0 : keys <= 65536 ? 1 : !long_ctx ? 2 : 3];
  SampleConfig c;
  c.target = gate.target;
  const uint32_t rank_tiles = std::max(
      1u,
      uint32_t((uint64_t(gate.rank) * keys + uint64_t(kTileKeys) * c.target - 1) / (uint64_t(kTileKeys) * c.target)));
  constexpr uint32_t kMaxTiles = 48;
  c.tiles = long_ctx ? std::min(kMaxTiles, std::max(16u, rank_tiles)) : rank_tiles;
  // 16-bit sample histogram counters: at most 255 * 256 samples per row.
  c.tiles = std::min(c.tiles, 255u);
  c.stride = std::max(1u, (keys + kTileKeys * c.tiles - 1) / (kTileKeys * c.tiles));
  c.groups = (keys - c.stride / 2 + c.stride - 1) / c.stride;
  return c;
}

}  // namespace details

/// \brief Whether the kernel covers the launch (the caller checked dtypes, devices and contiguity).
inline bool params_supported(const LaunchParams& p) {
  const auto aligned = [](const void* ptr, uintptr_t a) { return reinterpret_cast<uintptr_t>(ptr) % a == 0; };
  // rows * 32 is the int32 outer dimension / coordinate of the Q tensor map.
  return p.rows > 0 && p.rows * kNumHeads <= INT32_MAX && p.keys >= kMinKeys && p.keys <= kMaxKeys && p.topk == kTopK &&
         (p.out_offsets == nullptr || p.relative) &&
         (p.page_table == nullptr ||
          (p.relative && p.out_offsets == nullptr && p.row_batch != nullptr && p.page_table_stride >= 0)) &&
         aligned(p.q, 16) && aligned(p.k, 16) && aligned(p.scales, 16) && aligned(p.weights, 16) && aligned(p.out, 16);
}

/// \brief Launches the kernel on p.stream. Returns cudaErrorInvalidValue for
/// unsupported parameters (nothing launched) and the launch error otherwise.
inline cudaError_t launch(const LaunchParams& p) {
  using namespace details;
  if (!params_supported(p)) return cudaErrorInvalidValue;
  const uint32_t urows = static_cast<uint32_t>(p.rows), ukeys = static_cast<uint32_t>(p.keys),
                 ublocks = static_cast<uint32_t>((p.rows + kBlockRows - 1) / kBlockRows);
  const auto* starts_p = reinterpret_cast<const uint32_t*>(p.starts);
  const auto* ends_p = reinterpret_cast<const uint32_t*>(p.ends);
  const PageTable paged{p.page_table, p.page_table_stride, p.row_batch};
  CUtensorMap tq, tk, ts, tw;
  if (make_2d(
          &tq,
          p.q,
          CU_TENSOR_MAP_DATA_TYPE_UINT8,
          1,
          kHeadDim,
          int32_t(p.rows) * kNumHeads,
          kHeadDim,
          kBlockRows * kNumHeads,
          kHeadDim,
          128) != CUDA_SUCCESS ||
      make_2d(
          &tk, p.k, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHeadDim, int32_t(p.keys), kHeadDim, kTileKeys, kHeadDim, 128) !=
          CUDA_SUCCESS ||
      make_2d(&ts, p.scales, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, int32_t(p.keys), 1, kTileKeys, 1, 0, 0) !=
          CUDA_SUCCESS ||
      make_2d(
          &tw,
          p.weights,
          CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
          4,
          kNumHeads,
          int32_t(p.rows),
          kNumHeads,
          kBlockRows,
          kNumHeads,
          0) != CUDA_SUCCESS)
    return cudaErrorInvalidValue;

  if (p.slots < kSlotsPerCta || p.slots % kSlotsPerCta != 0) return cudaErrorInvalidValue;
  // One persistent CTA per SM (capped by the blocks and the workspace slots);
  // CTA c owns candidate slots 2c and 2c + 1.
  const uint32_t grid =
      static_cast<uint32_t>(std::min<int64_t>({int64_t(p.num_sms), int64_t(ublocks), p.slots / kSlotsPerCta}));
  int32_t device = 0;
  if (const cudaError_t e = cudaGetDevice(&device); e != cudaSuccess) return e;
  int32_t smem = 0;
  if (const cudaError_t e = kernel_smem(device, smem); e != cudaSuccess) return e;
  const SampleConfig sample = sample_config(ukeys);
  CUtensorMap tks;
  if (make_strided_kv(&tks, p.k, sample.groups, sample.stride) != CUDA_SUCCESS) return cudaErrorInvalidValue;
  const rescan::RescanBuffers fallback{p.retained, p.retained_indices, p.out};
  fused_mqa_topk_prefill_kernel<<<grid, kNumThreads, smem, p.stream>>>(
      urows,
      ukeys,
      starts_p,
      ends_p,
      p.values,
      p.indices,
      p.counts,
      fallback,
      /*phase_bytes=*/0u,
      sample.target,
      sample.stride,
      sample.groups,
      sample.tiles,
      p.scales,
      tks,
      tq,
      ts,
      tk,
      tw,
      p.relative,
      p.out_offsets,
      ublocks,
      reinterpret_cast<uint32_t*>(p.counter),
      p.defer,
      paged);
  return cudaGetLastError();
}

}  // namespace sglang::fused_mqa_topk
