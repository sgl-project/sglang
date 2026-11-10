// SPDX-License-Identifier: Apache-2.0
// SM70 DeepSeek-V4.1 sparse decode: 128 SWA FP8 rows + selected FP4 KV rows,
//  one query, online softmax with sink, P rounded to fp16 before PV.
#include "sm70_dsv41_fp4.cuh"

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <mma.h>

namespace sglang::sm70_dsv41 {

inline constexpr int kSparseThreads = 256;
inline constexpr int kElemsPerThread = kHeadDim / kSparseThreads;  // 2

SGL_DEVICE void dequant_swa_row(const uint8_t* row, fp16_t* dst, int tid) {
  const uint8_t* payload = row;
  const uint8_t* exps = row + kSwaPayloadBytes;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    const int d = tid * kElemsPerThread + i;
    const float scale = ue8m0_to_float(exps[d / kSwaFp8Block]);
    dst[d] = static_cast<fp16_t>(e4m3fn_to_float(payload[d]) * scale);
  }
}

SGL_DEVICE void dequant_kv_row(const uint8_t* row, fp16_t* dst, int tid) {
  const uint8_t* payload = row;
  const uint8_t* scales = row + kKvPayloadBytes;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    const int d = tid * kElemsPerThread + i;
    const uint8_t packed = payload[d >> 1];
    const uint8_t code = (d & 1) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0x0Fu);
    const float scale = e4m3fn_to_float(scales[d / kKvFp4Block]);
    dst[d] = static_cast<fp16_t>(decode_e2m1(code) * scale);
  }
}

SGL_DEVICE float cta_sum(float v, float* smem) {
  v = device::warp::reduce_sum(v);
  const uint32_t warp = threadIdx.x / 32u;
  const uint32_t lane = threadIdx.x % 32u;
  if (lane == 0) {
    smem[warp] = v;
  }
  __syncthreads();
  if (warp == 0) {
    const float partial = (lane < (kSparseThreads / 32)) ? smem[lane] : 0.0f;
    const float total = device::warp::reduce_sum(partial);
    if (lane == 0) {
      smem[0] = total;
    }
  }
  __syncthreads();
  const float out = smem[0];
  __syncthreads();
  return out;
}

/**
 * \brief One CTA per head. Streams SWA then compressed keys.
 *
 * Denominator uses fp32 p = exp(s-m); PV uses p rounded to fp16 (oracle).
 * An empty valid set writes zeros. All threads must take the same valid branch.
 */
__global__ __launch_bounds__(kSparseThreads, 1) void sparse_decode_kernel(
    fp16_t* __restrict__ out,
    const fp16_t* __restrict__ q,
    const uint8_t* __restrict__ swa_rows,
    const uint8_t* __restrict__ kv_rows,
    const uint8_t* __restrict__ swa_valid,
    const uint8_t* __restrict__ kv_valid,
    const float* __restrict__ sink,
    float softmax_scale,
    int32_t n_heads,
    int32_t n_swa,
    int32_t n_kv) {
  const int head = static_cast<int>(blockIdx.x);
  if (head >= n_heads) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  __shared__ fp16_t ks[kHeadDim];
  __shared__ fp16_t qs[kHeadDim];
  __shared__ float red[kSparseThreads / 32];

  const fp16_t* qh = q + static_cast<int64_t>(head) * kHeadDim;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    const int d = tid * kElemsPerThread + i;
    qs[d] = qh[d];
  }
  __syncthreads();

  float q_reg[kElemsPerThread];
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    q_reg[i] = static_cast<float>(qs[tid * kElemsPerThread + i]);
  }

  float m = -INFINITY;
  float l = 0.0f;
  float o_reg[kElemsPerThread];
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    o_reg[i] = 0.0f;
  }
  int n_valid = 0;

  const int n_total = n_swa + n_kv;
  for (int t = 0; t < n_total; ++t) {
    const int is_swa = static_cast<int>(t < n_swa);
    const int local = is_swa ? t : (t - n_swa);
    const int valid = is_swa ? (swa_valid[local] != 0) : (kv_valid[local] != 0);
    if (valid) {
      const uint8_t* row = is_swa ? (swa_rows + static_cast<int64_t>(local) * kSwaRowBytes)
                                  : (kv_rows + static_cast<int64_t>(local) * kKvRowBytes);
      if (is_swa) {
        dequant_swa_row(row, ks, tid);
      } else {
        dequant_kv_row(row, ks, tid);
      }
    }
    __syncthreads();
    if (valid) {
      float partial = 0.0f;
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        const int d = tid * kElemsPerThread + i;
        partial = fmaf(q_reg[i], static_cast<float>(ks[d]), partial);
      }
      const float s = cta_sum(partial, red) * softmax_scale;
      const float m_new = fmaxf(m, s);
      const float alpha = (m > -1.0e20f) ? expf(m - m_new) : 0.0f;
      const float p = expf(s - m_new);
      const float p16 = static_cast<float>(static_cast<fp16_t>(p));
      l = l * alpha + p;
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        const int d = tid * kElemsPerThread + i;
        o_reg[i] = o_reg[i] * alpha + p16 * static_cast<float>(ks[d]);
      }
      m = m_new;
    }
    n_valid += valid;
    __syncthreads();
  }

  fp16_t* oh = out + static_cast<int64_t>(head) * kHeadDim;
  if (n_valid == 0) {
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      oh[tid * kElemsPerThread + i] = static_cast<fp16_t>(0.0f);
    }
    return;
  }
  const float den = l + expf(sink[head] - m);
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    oh[tid * kElemsPerThread + i] = static_cast<fp16_t>(o_reg[i] / den);
  }
}

/**
 * \brief Sparse decode for one query: out/q [H, 512] fp16,
 *        swa_rows [W, 528], kv_rows [K, 288] uint8,
 *        valid masks uint8, sink [H] fp32.
 */
inline void sparse_decode(tvm::ffi::TensorView out,
                          tvm::ffi::TensorView q,
                          tvm::ffi::TensorView swa_rows,
                          tvm::ffi::TensorView kv_rows,
                          tvm::ffi::TensorView swa_valid,
                          tvm::ffi::TensorView kv_valid,
                          tvm::ffi::TensorView sink,
                          double softmax_scale) {
  using namespace host;
  SymbolicSize n_heads = {"num_heads"};
  SymbolicSize n_swa = {"num_swa"};
  SymbolicSize n_kv = {"num_kv"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_heads, kHeadDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(q)
      .verify(out);
  TensorMatcher({n_swa, kSwaRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(swa_rows);
  TensorMatcher({n_kv, kKvRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(kv_rows);
  TensorMatcher({n_swa})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(swa_valid);
  TensorMatcher({n_kv})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(kv_valid);
  TensorMatcher({n_heads})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(sink);

  const int32_t h = static_cast<int32_t>(n_heads.unwrap());
  const int32_t w = static_cast<int32_t>(n_swa.unwrap());
  const int32_t k = static_cast<int32_t>(n_kv.unwrap());
  CHECK_HOST(h > 0) << "sm70_dsv41 sparse_decode: n_heads must be > 0";
  const DLDevice device = device_.unwrap();
  LaunchKernel(static_cast<uint32_t>(h), kSparseThreads, device)(
      sparse_decode_kernel,
      static_cast<fp16_t*>(out.data_ptr()),
      static_cast<const fp16_t*>(q.data_ptr()),
      static_cast<const uint8_t*>(swa_rows.data_ptr()),
      static_cast<const uint8_t*>(kv_rows.data_ptr()),
      static_cast<const uint8_t*>(swa_valid.data_ptr()),
      static_cast<const uint8_t*>(kv_valid.data_ptr()),
      static_cast<const float*>(sink.data_ptr()),
      static_cast<float>(softmax_scale),
      h,
      w,
      k);
}

/**
 * \brief Python remainder for ``a % m`` with ``m > 0``.
 */
SGL_DEVICE int64_t remainder_pos(int64_t a, int32_t m) {
  const int64_t mm = static_cast<int64_t>(m);
  int64_t r = a % mm;
  if (r < 0) {
    r += mm;
  }
  return r;
}

/**
 * \brief Decode sparse attn that reads the SWA ring and a KV table in place.
 *
 * SWA slot ``s`` is valid iff ``remainder(pos - s, n_swa) <= pos``.
 * KV row ``j`` is ``kv_table[kv_idx[j]]`` when ``kv_idx[j] >= 0``.
 */
__global__ __launch_bounds__(kSparseThreads, 1) void sparse_decode_indexed_kernel(
    fp16_t* __restrict__ out,
    const fp16_t* __restrict__ q,
    const uint8_t* __restrict__ swa_rows,
    const uint8_t* __restrict__ kv_table,
    const int32_t* __restrict__ kv_idx,
    const int64_t* __restrict__ positions,
    const float* __restrict__ sink,
    float softmax_scale,
    int32_t n_heads,
    int32_t n_swa,
    int32_t n_kv) {
  const int head = static_cast<int>(blockIdx.x);
  if (head >= n_heads) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  __shared__ fp16_t ks[kHeadDim];
  __shared__ fp16_t qs[kHeadDim];
  __shared__ float red[kSparseThreads / 32];

  const fp16_t* qh = q + static_cast<int64_t>(head) * kHeadDim;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    const int d = tid * kElemsPerThread + i;
    qs[d] = qh[d];
  }
  __syncthreads();

  float q_reg[kElemsPerThread];
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    q_reg[i] = static_cast<float>(qs[tid * kElemsPerThread + i]);
  }

  float m = -INFINITY;
  float l = 0.0f;
  float o_reg[kElemsPerThread];
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    o_reg[i] = 0.0f;
  }
  int n_valid = 0;
  const int64_t pos = positions[0];

  const int n_total = n_swa + n_kv;
  for (int t = 0; t < n_total; ++t) {
    const int is_swa = static_cast<int>(t < n_swa);
    const int local = is_swa ? t : (t - n_swa);
    int valid;
    const uint8_t* row = nullptr;
    if (is_swa) {
      valid = static_cast<int>(remainder_pos(pos - local, n_swa) <= pos);
      row = swa_rows + static_cast<int64_t>(local) * kSwaRowBytes;
    } else {
      const int32_t idx = kv_idx[local];
      valid = static_cast<int>(idx >= 0);
      row = kv_table + static_cast<int64_t>(idx) * kKvRowBytes;
    }
    if (valid) {
      if (is_swa) {
        dequant_swa_row(row, ks, tid);
      } else {
        dequant_kv_row(row, ks, tid);
      }
    }
    __syncthreads();
    if (valid) {
      float partial = 0.0f;
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        const int d = tid * kElemsPerThread + i;
        partial = fmaf(q_reg[i], static_cast<float>(ks[d]), partial);
      }
      const float s = cta_sum(partial, red) * softmax_scale;
      const float m_new = fmaxf(m, s);
      const float alpha = (m > -1.0e20f) ? expf(m - m_new) : 0.0f;
      const float p = expf(s - m_new);
      const float p16 = static_cast<float>(static_cast<fp16_t>(p));
      l = l * alpha + p;
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        const int d = tid * kElemsPerThread + i;
        o_reg[i] = o_reg[i] * alpha + p16 * static_cast<float>(ks[d]);
      }
      m = m_new;
    }
    n_valid += valid;
    __syncthreads();
  }

  fp16_t* oh = out + static_cast<int64_t>(head) * kHeadDim;
  if (n_valid == 0) {
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      oh[tid * kElemsPerThread + i] = static_cast<fp16_t>(0.0f);
    }
    return;
  }
  const float den = l + expf(sink[head] - m);
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    oh[tid * kElemsPerThread + i] = static_cast<fp16_t>(o_reg[i] / den);
  }
}

/**
 * \brief Sparse decode over a static SWA ring and an indexed KV table.
 *
 * q/out [H, 512] fp16, swa_rows [W, 528], kv_table [cap, 288],
 * kv_idx [K] int32, positions [1] int64, sink [H] fp32.
 */
inline void sparse_decode_indexed(tvm::ffi::TensorView out,
                                  tvm::ffi::TensorView q,
                                  tvm::ffi::TensorView swa_rows,
                                  tvm::ffi::TensorView kv_table,
                                  tvm::ffi::TensorView kv_idx,
                                  tvm::ffi::TensorView positions,
                                  tvm::ffi::TensorView sink,
                                  double softmax_scale) {
  using namespace host;
  SymbolicSize n_heads = {"num_heads"};
  SymbolicSize n_swa = {"num_swa"};
  SymbolicSize n_kv = {"num_kv"};
  SymbolicSize n_table = {"num_kv_table"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_heads, kHeadDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(q)
      .verify(out);
  TensorMatcher({n_swa, kSwaRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(swa_rows);
  TensorMatcher({n_table, kKvRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(kv_table);
  TensorMatcher({n_kv})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(kv_idx);
  TensorMatcher({1})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(positions);
  TensorMatcher({n_heads})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(sink);

  const int32_t h = static_cast<int32_t>(n_heads.unwrap());
  const int32_t w = static_cast<int32_t>(n_swa.unwrap());
  const int32_t k = static_cast<int32_t>(n_kv.unwrap());
  CHECK_HOST(h > 0) << "sm70_dsv41 sparse_decode_indexed: n_heads must be > 0";
  CHECK_HOST(w > 0) << "sm70_dsv41 sparse_decode_indexed: n_swa must be > 0";
  const DLDevice device = device_.unwrap();
  LaunchKernel(static_cast<uint32_t>(h), kSparseThreads, device)(
      sparse_decode_indexed_kernel,
      static_cast<fp16_t*>(out.data_ptr()),
      static_cast<const fp16_t*>(q.data_ptr()),
      static_cast<const uint8_t*>(swa_rows.data_ptr()),
      static_cast<const uint8_t*>(kv_table.data_ptr()),
      static_cast<const int32_t*>(kv_idx.data_ptr()),
      static_cast<const int64_t*>(positions.data_ptr()),
      static_cast<const float*>(sink.data_ptr()),
      static_cast<float>(softmax_scale),
      h,
      w,
      k);
}

// Prefill sparse: Flash-style tiles + Volta WMMA 16x16x16 (not Ampere m16n8k16).
// One CTA per (query, 16-head tile). 8 warps split the 512-d axis (64 each).
inline constexpr int kPrefillThreads = 256;
inline constexpr int kPrefillWarps = 8;
inline constexpr int kHeadTile = 16;
inline constexpr int kKeyTile = 16;
inline constexpr int kDimPerWarp = kHeadDim / kPrefillWarps;  // 64

struct PrefillSmem {
  alignas(16) half K[kKeyTile * kHeadDim];
  alignas(16) float Spartial[kPrefillWarps * kHeadTile * kKeyTile];
  alignas(16) float S[kHeadTile * kKeyTile];
  alignas(16) half P[kHeadTile * kKeyTile];
  float m[kHeadTile];
  float l[kHeadTile];
  float alpha[kHeadTile];
  uint8_t kvld[kKeyTile];
  int n_valid_all;
};

inline constexpr int kPrefillSmemBytes = static_cast<int>((sizeof(PrefillSmem) + 255) & ~255);

// Volta wmma 16x16x16 f32 accumulator: 8 values / thread.
// Dumped against nvcuda::wmma::store_matrix_sync(..., 16, mem_row_major).
SGL_DEVICE int wmma_accum_row(int lane, int i) {
  return (lane & 1) + (i & 2) + ((lane & 16) >> 2) + ((lane & 4) << 1);
}

SGL_DEVICE int wmma_accum_col(int lane, int i) {
  return (i & 1) + (lane & 2) + (i & 4) + (lane & 8);
}

SGL_DEVICE void dequant_prefill_row(half* dst, const uint8_t* row, int is_swa, int lane) {
  const int d0 = lane * 16;
  if (is_swa) {
    const uint4 raw = *reinterpret_cast<const uint4*>(row + d0);
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
    const float sc = ue8m0_to_float(row[kSwaPayloadBytes + d0 / kSwaFp8Block]);
    half tmp[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      tmp[i] = __float2half_rn(e4m3fn_to_float(b[i]) * sc);
    }
    *reinterpret_cast<uint4*>(dst + d0) = *reinterpret_cast<const uint4*>(tmp);
    *reinterpret_cast<uint4*>(dst + d0 + 8) = *reinterpret_cast<const uint4*>(tmp + 8);
  } else {
    const uint2 raw = *reinterpret_cast<const uint2*>(row + (d0 >> 1));
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
    const float sc = e4m3fn_to_float(row[kKvPayloadBytes + d0 / kKvFp4Block]);
    half tmp[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint8_t packed = b[i];
      tmp[2 * i] = __float2half_rn(decode_e2m1(static_cast<uint8_t>(packed & 0x0Fu)) * sc);
      tmp[2 * i + 1] = __float2half_rn(decode_e2m1(static_cast<uint8_t>(packed >> 4)) * sc);
    }
    *reinterpret_cast<uint4*>(dst + d0) = *reinterpret_cast<const uint4*>(tmp);
    *reinterpret_cast<uint4*>(dst + d0 + 8) = *reinterpret_cast<const uint4*>(tmp + 8);
  }
}

/**
 * \brief T-wide packed sparse. Volta WMMA QK + PV, online softmax, fp16 P.
 *
 * q/out [T, H, 512], swa [T, W, 528], kv [T, K, 288]. Grid is (T, ceil(H/16)).
 */
__global__ __launch_bounds__(kPrefillThreads, 3) void sparse_prefill_wmma_kernel(
    fp16_t* __restrict__ out,
    const fp16_t* __restrict__ q,
    const uint8_t* __restrict__ swa_rows,
    const uint8_t* __restrict__ kv_rows,
    const uint8_t* __restrict__ swa_valid,
    const uint8_t* __restrict__ kv_valid,
    const float* __restrict__ sink,
    float softmax_scale,
    int32_t n_heads,
    int32_t n_swa,
    int32_t n_kv) {
  extern __shared__ char smem_raw[];
  PrefillSmem* sm = reinterpret_cast<PrefillSmem*>(smem_raw);

  const int query = static_cast<int>(blockIdx.x);
  const int h0 = static_cast<int>(blockIdx.y) * kHeadTile;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int h_tile = n_heads - h0;
  const int h_ok = h_tile < kHeadTile ? h_tile : kHeadTile;
  if (h_ok <= 0) {
    return;
  }

  const fp16_t* q_tile = q + (static_cast<int64_t>(query) * n_heads + h0) * kHeadDim;
  if (tid < kHeadTile) {
    sm->m[tid] = -INFINITY;
    sm->l[tid] = 0.0f;
  }
  if (tid == 0) {
    sm->n_valid_all = 0;
  }
  using namespace nvcuda;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[kDimPerWarp / 16];
#pragma unroll
  for (int dd = 0; dd < kDimPerWarp / 16; ++dd) {
    wmma::fill_fragment(o_frag[dd], 0.0f);
  }
  __syncthreads();

  const int n_total = n_swa + n_kv;
  const int64_t swa_base = static_cast<int64_t>(query) * n_swa;
  const int64_t kv_base = static_cast<int64_t>(query) * n_kv;

  for (int k0 = 0; k0 < n_total; k0 += kKeyTile) {
    if (tid < kKeyTile) {
      const int idx = k0 + tid;
      int valid = 0;
      if (idx < n_total) {
        valid = (idx < n_swa) ? static_cast<int>(swa_valid[swa_base + idx] != 0)
                              : static_cast<int>(kv_valid[kv_base + (idx - n_swa)] != 0);
      }
      sm->kvld[tid] = static_cast<uint8_t>(valid);
    }
    __syncthreads();

#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const int n = warp + pass * kPrefillWarps;
      half* dst = sm->K + n * kHeadDim;
      if (sm->kvld[n] == 0) {
        *reinterpret_cast<uint4*>(dst + lane * 16) = uint4{0, 0, 0, 0};
        *reinterpret_cast<uint4*>(dst + lane * 16 + 8) = uint4{0, 0, 0, 0};
      } else {
        const int idx = k0 + n;
        const int is_swa = static_cast<int>(idx < n_swa);
        const uint8_t* row = is_swa ? (swa_rows + (swa_base + idx) * kSwaRowBytes)
                                    : (kv_rows + (kv_base + (idx - n_swa)) * kKvRowBytes);
        dequant_prefill_row(dst, row, is_swa, lane);
      }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
    wmma::fill_fragment(s_frag, 0.0f);
    const int k_lo = warp * kDimPerWarp;
#pragma unroll
    for (int kk = 0; kk < kDimPerWarp; kk += 16) {
      wmma::load_matrix_sync(a_frag, q_tile + k_lo + kk, kHeadDim);
      wmma::load_matrix_sync(b_frag, sm->K + k_lo + kk, kHeadDim);
      wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
    }
    wmma::store_matrix_sync(
        sm->Spartial + warp * (kHeadTile * kKeyTile), s_frag, kKeyTile, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < kHeadTile * kKeyTile; i += kPrefillThreads) {
      float s = 0.0f;
#pragma unroll
      for (int w = 0; w < kPrefillWarps; ++w) {
        s += sm->Spartial[w * (kHeadTile * kKeyTile) + i];
      }
      const int n = i - (i / kKeyTile) * kKeyTile;
      s *= softmax_scale;
      if (sm->kvld[n] == 0) {
        s = -INFINITY;
      }
      sm->S[i] = s;
    }
    __syncthreads();

    if (tid < kHeadTile) {
      const int hh = tid;
      float mx = -INFINITY;
#pragma unroll
      for (int n = 0; n < kKeyTile; ++n) {
        mx = fmaxf(mx, sm->S[hh * kKeyTile + n]);
      }
      if (mx <= -1.0e20f) {
        sm->alpha[hh] = 1.0f;
#pragma unroll
        for (int n = 0; n < kKeyTile; ++n) {
          sm->P[hh * kKeyTile + n] = static_cast<half>(0.0f);
        }
      } else {
        const float m0 = sm->m[hh];
        const float m1 = fmaxf(m0, mx);
        const float alpha = (m0 > -1.0e20f) ? expf(m0 - m1) : 0.0f;
        sm->alpha[hh] = alpha;
        float lsum = 0.0f;
#pragma unroll
        for (int n = 0; n < kKeyTile; ++n) {
          const float p = expf(sm->S[hh * kKeyTile + n] - m1);
          lsum += p;
          sm->P[hh * kKeyTile + n] = static_cast<half>(p);
        }
        sm->l[hh] = sm->l[hh] * alpha + lsum;
        sm->m[hh] = m1;
      }
    }
    if (tid == 0) {
      int add = 0;
#pragma unroll
      for (int n = 0; n < kKeyTile; ++n) {
        add += static_cast<int>(sm->kvld[n]);
      }
      sm->n_valid_all += add;
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
    const int d0 = warp * kDimPerWarp;
#pragma unroll
    for (int dd = 0; dd < kDimPerWarp / 16; ++dd) {
#pragma unroll
      for (int i = 0; i < o_frag[dd].num_elements; ++i) {
        o_frag[dd].x[i] *= sm->alpha[wmma_accum_row(lane, i)];
      }
      wmma::load_matrix_sync(p_frag, sm->P, kKeyTile);
      wmma::load_matrix_sync(v_frag, sm->K + d0 + dd * 16, kHeadDim);
      wmma::mma_sync(o_frag[dd], p_frag, v_frag, o_frag[dd]);
    }
  }

  if (sm->n_valid_all == 0) {
    for (int hh = tid; hh < h_ok; hh += kPrefillThreads) {
      fp16_t* oh = out + (static_cast<int64_t>(query) * n_heads + h0 + hh) * kHeadDim;
      for (int d = 0; d < kHeadDim; ++d) {
        oh[d] = static_cast<fp16_t>(0.0f);
      }
    }
    return;
  }
  const int r0 = (lane & 1) + ((lane & 16) >> 2) + ((lane & 4) << 1);
  const int r1 = r0 + 2;
  const float den0 =
      (r0 < h_ok) ? (sm->l[r0] + expf(sink[h0 + r0] - sm->m[r0])) : 1.0f;
  const float den1 =
      (r1 < h_ok) ? (sm->l[r1] + expf(sink[h0 + r1] - sm->m[r1])) : 1.0f;
  const int d0 = warp * kDimPerWarp;
#pragma unroll
  for (int dd = 0; dd < kDimPerWarp / 16; ++dd) {
#pragma unroll
    for (int i = 0; i < o_frag[dd].num_elements; ++i) {
      const int hh = wmma_accum_row(lane, i);
      const int c = wmma_accum_col(lane, i);
      if (hh < h_ok) {
        const float den = (hh == r0) ? den0 : den1;
        out[(static_cast<int64_t>(query) * n_heads + h0 + hh) * kHeadDim + d0 + dd * 16 + c] =
            static_cast<fp16_t>(o_frag[dd].x[i] / den);
      }
    }
  }
}

/**
 * \brief Packed prefill sparse: out/q [T, H, 512] fp16,
 *        swa_rows [T, W, 528], kv_rows [T, K, 288] uint8,
 *        valid [T, W] / [T, K] uint8, sink [H] fp32.
 */
inline void sparse_prefill(tvm::ffi::TensorView out,
                           tvm::ffi::TensorView q,
                           tvm::ffi::TensorView swa_rows,
                           tvm::ffi::TensorView kv_rows,
                           tvm::ffi::TensorView swa_valid,
                           tvm::ffi::TensorView kv_valid,
                           tvm::ffi::TensorView sink,
                           double softmax_scale) {
  using namespace host;
  SymbolicSize n_q = {"num_queries"};
  SymbolicSize n_heads = {"num_heads"};
  SymbolicSize n_swa = {"num_swa"};
  SymbolicSize n_kv = {"num_kv"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_q, n_heads, kHeadDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(q)
      .verify(out);
  TensorMatcher({n_q, n_swa, kSwaRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(swa_rows);
  TensorMatcher({n_q, n_kv, kKvRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(kv_rows);
  TensorMatcher({n_q, n_swa})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(swa_valid);
  TensorMatcher({n_q, n_kv})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(kv_valid);
  TensorMatcher({n_heads})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(sink);

  const int32_t t = static_cast<int32_t>(n_q.unwrap());
  const int32_t h = static_cast<int32_t>(n_heads.unwrap());
  const int32_t w = static_cast<int32_t>(n_swa.unwrap());
  const int32_t k = static_cast<int32_t>(n_kv.unwrap());
  CHECK_HOST(h > 0) << "sm70_dsv41 sparse_prefill: n_heads must be > 0";
  CHECK_HOST(t > 0) << "sm70_dsv41 sparse_prefill: n_queries must be > 0";
  const DLDevice device = device_.unwrap();
  const uint32_t head_tiles = static_cast<uint32_t>((h + kHeadTile - 1) / kHeadTile);
  static bool smem_ready = false;
  if (!smem_ready) {
    CHECK_CUDA(cudaFuncSetAttribute(
        sparse_prefill_wmma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kPrefillSmemBytes));
    smem_ready = true;
  }
  LaunchKernel(
      dim3(static_cast<uint32_t>(t), head_tiles),
      kPrefillThreads,
      device,
      kPrefillSmemBytes)(
      sparse_prefill_wmma_kernel,
      static_cast<fp16_t*>(out.data_ptr()),
      static_cast<const fp16_t*>(q.data_ptr()),
      static_cast<const uint8_t*>(swa_rows.data_ptr()),
      static_cast<const uint8_t*>(kv_rows.data_ptr()),
      static_cast<const uint8_t*>(swa_valid.data_ptr()),
      static_cast<const uint8_t*>(kv_valid.data_ptr()),
      static_cast<const float*>(sink.data_ptr()),
      static_cast<float>(softmax_scale),
      h,
      w,
      k);
}

}  // namespace sglang::sm70_dsv41
