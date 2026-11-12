// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

// DFlash MLA draft: the RoPE, FP8 store and prefix-valid scatter of the target-context K rows of every draft
// layer in one launch, replacing per layer the in-place k_pe RoPE (fused_rope_kernel), torch's two BF16 -> FP8
// casts and set_mla_kv_buffer with the same arithmetic (rope_rotate_head itself); slot 0 is skipped as there.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include "../elementwise/rope.cuh"

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_fp8.h>

#include <cstdint>

namespace sglang {

constexpr uint32_t kDFlashCtxMaxLayers = 8;

struct DFlashCtxKVStoreParams {
  const void* knope[kDFlashCtxMaxLayers];    // bf16 [rows, 512], the kv_a RMSNorm output
  void* latent[kDFlashCtxMaxLayers];         // bf16 [rows, >= 576]; k_pe at column 512, rotated in place
  void* kv_buffer[kDFlashCtxMaxLayers];      // fp8 bytes [slots, 576]
  const void* cos_sin[kDFlashCtxMaxLayers];  // fp32 [positions, 64]: cos | sin halves
  const int64_t* positions;                  // [rows]
  const int64_t* loc;                        // [rows] (or [bs, width] with commit_lens)
  const int32_t* commit_lens;                // [bs], or nullptr: loc is used as given
  int64_t knope_stride_bytes;
  int64_t latent_stride_bytes;
  int64_t buffer_stride_bytes;
  uint32_t num_rows;
  uint32_t row_width;
};

// torch's BF16 -> float8_e4m3fn cast (ConvertToFloat8E4M3fnOp): round to nearest even, saturate to +-448,
// NaN -> 0x7f; checked against torch on all 65536 BF16 patterns.
SGL_DEVICE uint8_t fp8e4m3fn_from_fp32_torch(float f) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3));
}

// 8 BF16 (16 bytes) -> 8 FP8 bytes.
SGL_DEVICE uint2 bf16x8_to_fp8x8(uint4 v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  uint32_t out[2] = {0u, 0u};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t h = (w[i >> 1] >> ((i & 1) * 16)) & 0xFFFFu;
    const uint32_t b = fp8e4m3fn_from_fp32_torch(__uint_as_float(h << 16));
    out[i >> 2] |= b << ((i & 3) * 8);
  }
  return make_uint2(out[0], out[1]);
}

// grid (rows, layers), 64 threads: thread t owns nope columns [8t, 8t + 8); threads 0-7 also rotate and
// own rope columns [8t, 8t + 8) (the fused_rope_kernel split: 8 lanes x 4 pairs).
template <bool kUsePDL>
__global__ void dflash_ctx_kv_store_kernel(const __grid_constant__ DFlashCtxKVStoreParams p) {
  using namespace device;
  constexpr int64_t kNope = 512, kRope = 64;
  const uint32_t row = blockIdx.x, layer = blockIdx.y, t = threadIdx.x;

  PDLWaitPrimary<kUsePDL>();
  void* rope = pointer::offset(p.latent[layer], row * p.latent_stride_bytes + kNope * 2);
  if (t < 8) {
    const int64_t pos = p.positions[row];
    const void* cos_ptr = pointer::offset(p.cos_sin[layer], pos * kRope * 4);
    const void* sin_ptr = pointer::offset(cos_ptr, kRope / 2 * 4);
    rope_rotate_head<bf16_t, false, kRope, 4>(rope, cos_ptr, sin_ptr, t);
  }
  int64_t loc;
  if (p.commit_lens != nullptr) {
    const uint32_t b = row / p.row_width, j = row % p.row_width;
    loc = static_cast<int64_t>(j) < static_cast<int64_t>(p.commit_lens[b]) ? p.loc[row] : 0;
  } else {
    loc = p.loc[row];
  }
  __syncwarp();
  if (loc != 0) {
    uint8_t* dst = static_cast<uint8_t*>(p.kv_buffer[layer]) + loc * p.buffer_stride_bytes;
    const uint4 nope = *reinterpret_cast<const uint4*>(
        pointer::offset(p.knope[layer], row * p.knope_stride_bytes + t * 16));
    *reinterpret_cast<uint2*>(dst + t * 8) = bf16x8_to_fp8x8(nope);
    if (t < 8) {
      const uint4 r = *reinterpret_cast<const uint4*>(pointer::offset(rope, t * 16));
      *reinterpret_cast<uint2*>(dst + kNope + t * 8) = bf16x8_to_fp8x8(r);
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct DFlashCtxKVStoreKernel {
  // ptrs: int64 CPU [4, layers] = (knope, latent, kv_buffer, cos_sin) data pointers.
  static void
  run(tvm::ffi::TensorView ptrs,
      tvm::ffi::TensorView positions,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView commit_lens,
      int64_t num_rows,
      int64_t row_width,
      int64_t knope_stride_bytes,
      int64_t latent_stride_bytes,
      int64_t buffer_stride_bytes,
      int64_t use_commit_lens) {
    using namespace host;
    const int64_t layers = ptrs.size(1);
    CHECK_HOST(ptrs.size(0) == 4 && layers >= 1 && layers <= kDFlashCtxMaxLayers);
    CHECK_HOST(num_rows >= 0 && row_width >= 1);
    CHECK_HOST(knope_stride_bytes % 16 == 0 && latent_stride_bytes % 16 == 0 && buffer_stride_bytes % 8 == 0);
    if (num_rows == 0) return;
    DFlashCtxKVStoreParams p{};
    const auto* table = static_cast<const int64_t*>(ptrs.data_ptr());
    for (int64_t l = 0; l < layers; ++l) {
      p.knope[l] = reinterpret_cast<const void*>(table[l]);
      p.latent[l] = reinterpret_cast<void*>(table[layers + l]);
      p.kv_buffer[l] = reinterpret_cast<void*>(table[2 * layers + l]);
      p.cos_sin[l] = reinterpret_cast<const void*>(table[3 * layers + l]);
    }
    p.positions = static_cast<const int64_t*>(positions.data_ptr());
    p.loc = static_cast<const int64_t*>(loc.data_ptr());
    p.commit_lens = use_commit_lens ? static_cast<const int32_t*>(commit_lens.data_ptr()) : nullptr;
    p.knope_stride_bytes = knope_stride_bytes;
    p.latent_stride_bytes = latent_stride_bytes;
    p.buffer_stride_bytes = buffer_stride_bytes;
    p.num_rows = static_cast<uint32_t>(num_rows);
    p.row_width = static_cast<uint32_t>(row_width);
    const auto device = positions.device();
    LaunchKernel(dim3(static_cast<uint32_t>(num_rows), static_cast<uint32_t>(layers)), 64, device)  //
        .enable_pdl(kUsePDL)(dflash_ctx_kv_store_kernel<kUsePDL>, p);
  }
};


// ---- the kv_a RMSNorm folded into the store (SGLANG_OPT_DFLASH_CTX_KV_STORE_NORM) ----------------------
// Replaces, per layer, flashinfer's RMSNorm kernel on latent[:, :512] (the served kv_a_layernorm) plus the unfused store's
// read of its output. It reproduces flashinfer's CuTe RMSNormKernel (norm/kernels/rmsnorm.py, served image) at H = 512 BF16:
// cluster 1, 32 threads per row, thread t owns columns [8t, 8t + 8) then [256 + 8t, 256 + 8t + 8); its sum of squares is one
// FMA chain from 0 in that order (PTX fma.rn.f32.bf16 x, x, acc), then a butterfly over lane xor 1, 2, 4, 8, 16 (add.f32),
// mean = div.rn(sum, 512), rstd = rsqrt.approx.ftz(eps + mean), y = (x * rstd) * (w + 0), cvt.rn to BF16. The FP8 bytes are
// torch's cast of that BF16 value, as the unfused store writes them. Accepted at run time only after a bitwise probe against
// the served module (dflash_context_fusion.py).

struct DFlashCtxKVStoreNormParams {
  void* latent[kDFlashCtxMaxLayers];         // bf16 [rows, >= 576]: [0, 512) kv_a latent before the norm; k_pe at 512
  void* kv_buffer[kDFlashCtxMaxLayers];      // fp8 bytes [slots, 576]
  const void* cos_sin[kDFlashCtxMaxLayers];  // fp32 [positions, 64]: cos | sin halves
  const void* norm_w[kDFlashCtxMaxLayers];   // bf16 [512]: kv_a_layernorm weight
  float eps[kDFlashCtxMaxLayers];
  void* dbg_out;                             // test only: bf16 [layers, rows, 512] normalized rows, or nullptr
  const int64_t* positions;
  const int64_t* loc;
  const int32_t* commit_lens;
  int64_t latent_stride_bytes;
  int64_t buffer_stride_bytes;
  uint32_t num_rows;
  uint32_t row_width;
};

SGL_DEVICE float dflash_rsqrt_approx_ftz(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE void dflash_unpack_bf16x8(uint4 v, float* out) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = __uint_as_float(((w[i >> 1] >> ((i & 1) * 16)) & 0xFFFFu) << 16);
}

template <bool kUsePDL>
__global__ void dflash_ctx_kv_store_norm_kernel(const __grid_constant__ DFlashCtxKVStoreNormParams p) {
  using namespace device;
  constexpr int64_t kNope = 512, kRope = 64;
  const uint32_t row = blockIdx.x, layer = blockIdx.y, t = threadIdx.x;  // one warp per (row, layer)

  PDLWaitPrimary<kUsePDL>();
  void* base = pointer::offset(p.latent[layer], row * p.latent_stride_bytes);
  void* rope = pointer::offset(base, kNope * 2);
  if (t < 8) {
    const int64_t pos = p.positions[row];
    const void* cos_ptr = pointer::offset(p.cos_sin[layer], pos * kRope * 4);
    const void* sin_ptr = pointer::offset(cos_ptr, kRope / 2 * 4);
    rope_rotate_head<bf16_t, false, kRope, 4>(rope, cos_ptr, sin_ptr, t);
  }
  int64_t loc;
  if (p.commit_lens != nullptr) {
    const uint32_t b = row / p.row_width, j = row % p.row_width;
    loc = static_cast<int64_t>(j) < static_cast<int64_t>(p.commit_lens[b]) ? p.loc[row] : 0;
  } else {
    loc = p.loc[row];
  }
  __syncwarp();
  if (loc != 0 || p.dbg_out != nullptr) {
    float x[16], w[16];
    dflash_unpack_bf16x8(*reinterpret_cast<const uint4*>(pointer::offset(base, t * 16)), x);
    dflash_unpack_bf16x8(*reinterpret_cast<const uint4*>(pointer::offset(base, 512 + t * 16)), x + 8);
    dflash_unpack_bf16x8(*reinterpret_cast<const uint4*>(pointer::offset(p.norm_w[layer], t * 16)), w);
    dflash_unpack_bf16x8(*reinterpret_cast<const uint4*>(pointer::offset(p.norm_w[layer], 512 + t * 16)), w + 8);
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) acc = __fmaf_rn(x[i], x[i], acc);
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) acc = __fadd_rn(__shfl_xor_sync(0xFFFFFFFFu, acc, off), acc);
    float rstd = dflash_rsqrt_approx_ftz(__fadd_rn(p.eps[layer], __fdiv_rn(acc, 512.0f)));
    uint32_t y[8];
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
      float o[2];
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const float wk = __fadd_rn(w[i + k], 0.0f);
        o[k] = __fmul_rn(__fmul_rn(x[i + k], rstd), wk);
      }
      y[i >> 1] = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(o[0]))) |
                  (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(o[1]))) << 16);
    }
    const uint4 ya = make_uint4(y[0], y[1], y[2], y[3]), yb = make_uint4(y[4], y[5], y[6], y[7]);
    if (p.dbg_out != nullptr) {
      void* d = pointer::offset(p.dbg_out, (static_cast<int64_t>(layer) * p.num_rows + row) * kNope * 2);
      *reinterpret_cast<uint4*>(pointer::offset(d, t * 16)) = ya;
      *reinterpret_cast<uint4*>(pointer::offset(d, 512 + t * 16)) = yb;
    }
    if (loc != 0) {
      uint8_t* dst = static_cast<uint8_t*>(p.kv_buffer[layer]) + loc * p.buffer_stride_bytes;
      *reinterpret_cast<uint2*>(dst + t * 8) = bf16x8_to_fp8x8(ya);
      *reinterpret_cast<uint2*>(dst + 256 + t * 8) = bf16x8_to_fp8x8(yb);
      if (t < 8) {
        const uint4 r = *reinterpret_cast<const uint4*>(pointer::offset(rope, t * 16));
        *reinterpret_cast<uint2*>(dst + kNope + t * 8) = bf16x8_to_fp8x8(r);
      }
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct DFlashCtxKVStoreNormKernel {
  // ptrs: int64 CPU [4, layers] = (latent, kv_buffer, cos_sin, norm_w) data pointers; eps: float32 CPU [layers];
  // dbg: bf16 [layers, rows, 512] when use_dbg (test only).
  static void
  run(tvm::ffi::TensorView ptrs,
      tvm::ffi::TensorView eps,
      tvm::ffi::TensorView positions,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView commit_lens,
      tvm::ffi::TensorView dbg,
      int64_t num_rows,
      int64_t row_width,
      int64_t latent_stride_bytes,
      int64_t buffer_stride_bytes,
      int64_t use_commit_lens,
      int64_t use_dbg) {
    using namespace host;
    const int64_t layers = ptrs.size(1);
    CHECK_HOST(ptrs.size(0) == 4 && layers >= 1 && layers <= kDFlashCtxMaxLayers && eps.size(0) == layers);
    CHECK_HOST(num_rows >= 0 && row_width >= 1);
    CHECK_HOST(latent_stride_bytes % 16 == 0 && buffer_stride_bytes % 8 == 0);
    if (num_rows == 0) return;
    DFlashCtxKVStoreNormParams p{};
    const auto* table = static_cast<const int64_t*>(ptrs.data_ptr());
    const auto* eps_host = static_cast<const float*>(eps.data_ptr());
    for (int64_t l = 0; l < layers; ++l) {
      p.latent[l] = reinterpret_cast<void*>(table[l]);
      p.kv_buffer[l] = reinterpret_cast<void*>(table[layers + l]);
      p.cos_sin[l] = reinterpret_cast<const void*>(table[2 * layers + l]);
      p.norm_w[l] = reinterpret_cast<const void*>(table[3 * layers + l]);
      p.eps[l] = eps_host[l];
    }
    p.dbg_out = use_dbg ? dbg.data_ptr() : nullptr;
    p.positions = static_cast<const int64_t*>(positions.data_ptr());
    p.loc = static_cast<const int64_t*>(loc.data_ptr());
    p.commit_lens = use_commit_lens ? static_cast<const int32_t*>(commit_lens.data_ptr()) : nullptr;
    p.latent_stride_bytes = latent_stride_bytes;
    p.buffer_stride_bytes = buffer_stride_bytes;
    p.num_rows = static_cast<uint32_t>(num_rows);
    p.row_width = static_cast<uint32_t>(row_width);
    const auto device = positions.device();
    LaunchKernel(dim3(static_cast<uint32_t>(num_rows), static_cast<uint32_t>(layers)), 32, device)  //
        .enable_pdl(kUsePDL)(dflash_ctx_kv_store_norm_kernel<kUsePDL>, p);
  }
};

}  // namespace sglang
