// SPDX-License-Identifier: Apache-2.0
// SM70 DeepSeek-V4.1 indexer logits (decode): packed FP4 K, fp16 Q, fp32 acc.
// score[t, j] = sum_h relu(q[t,h] . k[j]) * w[t,h], masked by (pos+1)//ratio.
#include "sm70_dsv41_fp4.cuh"

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_dsv41 {

/**
 * \brief One warp per (query, key). Decode 68-byte index-K in registers, fp32 dots.
 *
 * Grid is (n_keys, n_queries). Volta has no m16n8k8; fp32 FMA matches the oracle
 * einsum reduction closer than m8n8k4 fragment layout for this skinny GEMV.
 */
__global__ void index_logits_kernel(float* __restrict__ logits,
                                    const fp16_t* __restrict__ q,
                                    const fp16_t* __restrict__ weights,
                                    const uint8_t* __restrict__ index_rows,
                                    const int32_t* __restrict__ query_pos,
                                    int32_t ratio,
                                    int32_t n_heads,
                                    int32_t n_keys,
                                    int32_t n_queries) {
  const int key = static_cast<int>(blockIdx.x);
  const int query = static_cast<int>(blockIdx.y);
  if (key >= n_keys || query >= n_queries) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x);
  const int vis = (query_pos[query] + 1) / ratio;
  if (key >= vis) {
    if (lane == 0) {
      logits[static_cast<int64_t>(query) * n_keys + key] = -INFINITY;
    }
    return;
  }

  const uint8_t* row = index_rows + static_cast<int64_t>(key) * kIndexRowBytes;
  const uint8_t* payload = row;
  const uint8_t* exps = row + kIndexPayloadBytes;

  // 128 dims / 32 lanes = 4 elements per lane.
  float k_reg[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane * 4 + i;
    const uint8_t packed = payload[d >> 1];
    const uint8_t code = (d & 1) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0x0Fu);
    const float scale = ue8m0_to_float(exps[d / kIndexFp4Block]);
    k_reg[i] = decode_e2m1(code) * scale;
  }

  const fp16_t* q_row = q + (static_cast<int64_t>(query) * n_heads) * kIndexDim;
  const fp16_t* w_row = weights + static_cast<int64_t>(query) * n_heads;
  float acc_heads = 0.0f;
  for (int h = 0; h < n_heads; ++h) {
    const fp16_t* qh = q_row + static_cast<int64_t>(h) * kIndexDim;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane * 4 + i;
      dot = fmaf(static_cast<float>(qh[d]), k_reg[i], dot);
    }
    dot = device::warp::reduce_sum(dot);
    if (lane == 0) {
      const float w = static_cast<float>(w_row[h]);
      acc_heads = fmaf(fmaxf(dot, 0.0f), w, acc_heads);
    }
  }
  if (lane == 0) {
    logits[static_cast<int64_t>(query) * n_keys + key] = acc_heads;
  }
}

/**
 * \brief Indexer logits: q [T, H, 128] fp16, weights [T, H] fp16,
 *        index_rows [N, 68] uint8, query_pos [T] int32 -> logits [T, N] fp32.
 */
inline void index_logits(tvm::ffi::TensorView logits,
                         tvm::ffi::TensorView q,
                         tvm::ffi::TensorView weights,
                         tvm::ffi::TensorView index_rows,
                         tvm::ffi::TensorView query_pos,
                         int64_t ratio) {
  using namespace host;
  SymbolicSize n_queries = {"num_queries"};
  SymbolicSize n_heads = {"num_heads"};
  SymbolicSize n_keys = {"num_keys"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_queries, n_heads, kIndexDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(q);
  TensorMatcher({n_queries, n_heads})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(weights);
  TensorMatcher({n_keys, kIndexRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(index_rows);
  TensorMatcher({n_queries})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(query_pos);
  TensorMatcher({n_queries, n_keys})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(logits);

  CHECK_HOST(ratio >= 1) << "sm70_dsv41 index_logits: ratio must be >= 1, got " << ratio;
  const int32_t t = static_cast<int32_t>(n_queries.unwrap());
  const int32_t h = static_cast<int32_t>(n_heads.unwrap());
  const int32_t n = static_cast<int32_t>(n_keys.unwrap());
  if (t == 0 || n == 0) {
    return;
  }
  const DLDevice device = device_.unwrap();
  const dim3 grid(static_cast<uint32_t>(n), static_cast<uint32_t>(t));
  LaunchKernel(grid, 32, device)(
      index_logits_kernel,
      static_cast<float*>(logits.data_ptr()),
      static_cast<const fp16_t*>(q.data_ptr()),
      static_cast<const fp16_t*>(weights.data_ptr()),
      static_cast<const uint8_t*>(index_rows.data_ptr()),
      static_cast<const int32_t*>(query_pos.data_ptr()),
      static_cast<int32_t>(ratio),
      h,
      n,
      t);
}

// ---------------------------------------------------------------------------
// Decode (T = 1) variants over a static-capacity index buffer.
//
// The key count is read from the device (query_pos) so the launch shape is
// fixed and the kernel can live inside a CUDA graph. One warp per key with a
// grid-stride loop; the per-key math is the same as index_logits_kernel.
//
//   dense : logits[j] = score(row j) for j < vis, -inf for vis <= j < pad(vis)
//   gather: logits[j] = score(row key_ids[j]) or -inf if key_ids[j] < 0 / >= vis
// ---------------------------------------------------------------------------
inline constexpr int kDecodeLogitsThreads = 256;
inline constexpr int kDecodeLogitsBlocks = 1024;

__device__ __forceinline__ float index_logit_one_key(const fp16_t* __restrict__ q_row,
                                                     const fp16_t* __restrict__ w_row,
                                                     const uint8_t* __restrict__ row,
                                                     int32_t n_heads,
                                                     int lane) {
  const uint8_t* payload = row;
  const uint8_t* exps = row + kIndexPayloadBytes;
  float k_reg[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane * 4 + i;
    const uint8_t packed = payload[d >> 1];
    const uint8_t code = (d & 1) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0x0Fu);
    const float scale = ue8m0_to_float(exps[d / kIndexFp4Block]);
    k_reg[i] = decode_e2m1(code) * scale;
  }
  float acc_heads = 0.0f;
  for (int h = 0; h < n_heads; ++h) {
    const fp16_t* qh = q_row + static_cast<int64_t>(h) * kIndexDim;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane * 4 + i;
      dot = fmaf(static_cast<float>(qh[d]), k_reg[i], dot);
    }
    dot = device::warp::reduce_sum(dot);
    if (lane == 0) {
      const float w = static_cast<float>(w_row[h]);
      acc_heads = fmaf(fmaxf(dot, 0.0f), w, acc_heads);
    }
  }
  return acc_heads;
}

template <bool kGather>
__global__ void index_logits_decode_kernel(float* __restrict__ logits,
                                           const fp16_t* __restrict__ q,
                                           const fp16_t* __restrict__ weights,
                                           const uint8_t* __restrict__ index_rows,
                                           const int32_t* __restrict__ query_pos,
                                           const int32_t* __restrict__ key_ids,
                                           int32_t ratio,
                                           int32_t n_heads,
                                           int32_t n_cap,
                                           int32_t n_out,
                                           int32_t pad) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int vis = (query_pos[0] + 1) / ratio;
  int limit = n_out;
  if (!kGather) {
    const int padded = ((vis + pad - 1) / pad) * pad;
    limit = padded < n_out ? padded : n_out;
  }
  for (int j = static_cast<int>(blockIdx.x) * warps_per_block + warp; j < limit;
       j += static_cast<int>(gridDim.x) * warps_per_block) {
    const int key = kGather ? key_ids[j] : j;
    if (key < 0 || key >= vis || key >= n_cap) {
      if (lane == 0) {
        logits[j] = -INFINITY;
      }
      continue;
    }
    const uint8_t* row = index_rows + static_cast<int64_t>(key) * kIndexRowBytes;
    const float acc = index_logit_one_key(q, weights, row, n_heads, lane);
    if (lane == 0) {
      logits[j] = acc;
    }
  }
}

inline void index_logits_decode_impl(tvm::ffi::TensorView logits,
                                     tvm::ffi::TensorView q,
                                     tvm::ffi::TensorView weights,
                                     tvm::ffi::TensorView index_rows,
                                     tvm::ffi::TensorView query_pos,
                                     const int32_t* key_ids,
                                     int64_t ratio,
                                     int64_t pad) {
  using namespace host;
  SymbolicSize n_heads = {"num_heads"};
  SymbolicSize n_cap = {"num_keys_cap"};
  SymbolicSize n_out = {"num_out"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({1, n_heads, kIndexDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(q);
  TensorMatcher({1, n_heads})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(weights);
  TensorMatcher({n_cap, kIndexRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(index_rows);
  TensorMatcher({1})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(query_pos);
  TensorMatcher({1, n_out})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(logits);

  CHECK_HOST(ratio >= 1) << "sm70_dsv41 index_logits_decode: ratio must be >= 1, got " << ratio;
  CHECK_HOST(pad >= 1) << "sm70_dsv41 index_logits_decode: pad must be >= 1, got " << pad;
  const int32_t h = static_cast<int32_t>(n_heads.unwrap());
  const int32_t cap = static_cast<int32_t>(n_cap.unwrap());
  const int32_t out = static_cast<int32_t>(n_out.unwrap());
  if (out == 0) {
    return;
  }
  const DLDevice device = device_.unwrap();
  const int warps_per_block = kDecodeLogitsThreads / 32;
  const int need_blocks = (out + warps_per_block - 1) / warps_per_block;
  const dim3 grid(static_cast<uint32_t>(need_blocks < kDecodeLogitsBlocks ? need_blocks : kDecodeLogitsBlocks));
  if (key_ids == nullptr) {
    LaunchKernel(grid, kDecodeLogitsThreads, device)(
        index_logits_decode_kernel<false>,
        static_cast<float*>(logits.data_ptr()),
        static_cast<const fp16_t*>(q.data_ptr()),
        static_cast<const fp16_t*>(weights.data_ptr()),
        static_cast<const uint8_t*>(index_rows.data_ptr()),
        static_cast<const int32_t*>(query_pos.data_ptr()),
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t>(ratio),
        h,
        cap,
        out,
        static_cast<int32_t>(pad));
  } else {
    LaunchKernel(grid, kDecodeLogitsThreads, device)(
        index_logits_decode_kernel<true>,
        static_cast<float*>(logits.data_ptr()),
        static_cast<const fp16_t*>(q.data_ptr()),
        static_cast<const fp16_t*>(weights.data_ptr()),
        static_cast<const uint8_t*>(index_rows.data_ptr()),
        static_cast<const int32_t*>(query_pos.data_ptr()),
        key_ids,
        static_cast<int32_t>(ratio),
        h,
        cap,
        out,
        static_cast<int32_t>(pad));
  }
}

/**
 * \brief Decode logits over a static index buffer: logits [1, n_out] fp32 gets
 *        score(row j) for j < vis = (pos+1)/ratio and -inf up to pad(vis).
 *        Entries beyond pad(vis) are left untouched.
 */
inline void index_logits_decode_dense(tvm::ffi::TensorView logits,
                                      tvm::ffi::TensorView q,
                                      tvm::ffi::TensorView weights,
                                      tvm::ffi::TensorView index_rows,
                                      tvm::ffi::TensorView query_pos,
                                      int64_t ratio,
                                      int64_t pad) {
  index_logits_decode_impl(logits, q, weights, index_rows, query_pos, nullptr, ratio, pad);
}

/**
 * \brief Decode logits for an explicit key list: logits[1, n_out] gets
 *        score(row key_ids[j]) or -inf when key_ids[j] < 0 or unreachable.
 */
inline void index_logits_decode_gather(tvm::ffi::TensorView logits,
                                       tvm::ffi::TensorView q,
                                       tvm::ffi::TensorView weights,
                                       tvm::ffi::TensorView index_rows,
                                       tvm::ffi::TensorView query_pos,
                                       tvm::ffi::TensorView key_ids,
                                       int64_t ratio) {
  using namespace host;
  SymbolicSize n_out = {"num_out"};
  TensorMatcher({1, n_out})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>()
      .verify(key_ids);
  CHECK_HOST(n_out.unwrap() == logits.shape()[1])
      << "sm70_dsv41 index_logits_decode_gather: key_ids/logits width mismatch";
  index_logits_decode_impl(logits, q, weights, index_rows, query_pos,
                           static_cast<const int32_t*>(key_ids.data_ptr()), ratio, 1);
}

}  // namespace sglang::sm70_dsv41
