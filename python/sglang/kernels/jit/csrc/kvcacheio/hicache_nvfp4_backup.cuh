// HiCache backup of FP8 device KV rows into an NVFP4 host copy, one pass per token
// (sglang.srt.mem_cache.pool_host.mha_fp8_nvfp4, SGLANG_HICACHE_FP4_HOST_FUSED_BACKUP).
//
// Host layout (page_first, per K and per V): packed E2M1 rows [tokens][layers][kRow / 2] (element 2j in
// the low nibble of byte j) and E4M3 block scales [tokens][layers][kRow / 16], one per 16 values. So one
// token's rows of every layer are kRow / 2 * layers contiguous bytes, its scales kRow / 16 * layers.
//
// One warp quantizes one (token, K|V): two layer rows per step, each lane 32 FP8 values (two blocks),
// then stores the token's whole packed region as full contiguous lines (32 lanes x 16 B per step) and its
// scales, staged per warp in shared memory, as full lines after the last layer. The same warp copies the
// token's FP8 MTP draft rows, which the pool keeps byte for byte in their own host span. The reference backup
// (mha_fp8_nvfp4._backup_tile) writes each (token, layer) row and its 32 B of scales as their own
// stores; partial-line writes to host memory are what bounded it.
//
// The arithmetic is the reference's instruction for instruction, as inline PTX so no compiler flag can
// change it: E4M3 -> F32 through cvt.rn.f16x2.e4m3x2 and cvt.f32.f16, block amax by max.f32 (NaN
// ignored), scale = cvt.rn.satfinite.e4m3x2.f32(amax * f32(1/6) * (1/g)), inv = 0 for a zero scale else
// div.full.f32(1, scale * g), and cvt.rn.satfinite.e2m1x2.f32 of each value times inv. The caller checks
// it against the reference on every (block amax, value) pair at init.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>

#include <cstdint>
#include <cstring>

namespace sglang {

namespace device::nvfp4_backup {

SGL_DEVICE float f32_of_e4m3_lo(uint32_t pair16, float& hi) {
  uint32_t h2;
  asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(h2) : "r"(pair16));
  float lo;
  asm("{ .reg .b16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h; }" : "=f"(lo), "=f"(hi) : "r"(h2));
  return lo;
}

SGL_DEVICE float f32_abs(float x) {
  float y;
  asm("abs.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE float f32_max(float a, float b) {  // NaN-ignoring, as the reference's maximum and max
  float y;
  asm("max.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

SGL_DEVICE float f32_mul(float a, float b) {
  float y;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

SGL_DEVICE float f32_div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

SGL_DEVICE uint32_t e4m3_of_f32(float x) {  // the low byte of cvt.rn.satfinite.e4m3x2.f32 (0, x)
  uint32_t r;
  asm("{ .reg .b16 t; .reg .f32 z; mov.f32 z, 0f00000000; cvt.rn.satfinite.e4m3x2.f32 t, z, %1; cvt.u32.u16 %0, t; }"
      : "=r"(r)
      : "f"(x));
  return r & 0xFFu;
}

SGL_DEVICE uint32_t e2m1x2_of_f32(float high, float low) {  // one packed byte: low in bits 0-3
  uint32_t r;
  asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }" : "=r"(r) : "f"(high), "f"(low));
  return r & 0xFFu;
}

// One 16-value block: four words of FP8 bytes in element order -> two words of packed E2M1 and the scale.
template <uint32_t kInvGlobalScale>
SGL_DEVICE uint2 quantize_block(const uint32_t (&w)[4], uint32_t& scale_byte) {
  float v[16];
#pragma unroll
  for (uint32_t k = 0; k < 4; ++k) {
    float hi0, hi1;
    v[4 * k + 0] = f32_of_e4m3_lo(w[k] & 0xFFFFu, hi0);
    v[4 * k + 1] = hi0;
    v[4 * k + 2] = f32_of_e4m3_lo(w[k] >> 16, hi1);
    v[4 * k + 3] = hi1;
  }
  float amax = f32_max(f32_abs(v[0]), f32_abs(v[1]));
#pragma unroll
  for (uint32_t k = 1; k < 8; ++k) amax = f32_max(amax, f32_max(f32_abs(v[2 * k]), f32_abs(v[2 * k + 1])));
  // amax * f32(1/6) * (1/g), as the reference writes it (two roundings; the second is exact).
  scale_byte = e4m3_of_f32(f32_mul(f32_mul(amax, 1.0f / 6.0f), static_cast<float>(kInvGlobalScale)));
  float unused;
  const float scale32 = f32_of_e4m3_lo(scale_byte, unused);  // the pair (scale, 0)
  const float inv = scale32 == 0.0f ? 0.0f : f32_div_full(1.0f, f32_mul(scale32, 1.0f / kInvGlobalScale));
  uint32_t out[2] = {0, 0};
#pragma unroll
  for (uint32_t j = 0; j < 8; ++j) {
    out[j / 4] |= e2m1x2_of_f32(f32_mul(v[2 * j + 1], inv), f32_mul(v[2 * j], inv)) << (8 * (j % 4));
  }
  return make_uint2(out[0], out[1]);
}

}  // namespace device::nvfp4_backup

struct Nvfp4BackupParams {
  const uint64_t* src_ptrs[2];  // device tables [layers] of FP8 layer bases, K and V
  void* dst_data[2];            // host packed rows, K and V
  void* dst_scale[2];           // host scales, K and V
  const uint64_t* draft_ptrs[2];  // device tables [draft_layers] of draft layer bases, K and V
  void* draft_dst[2];             // host draft rows [tokens][draft_layers][draft_row_bytes], K and V
  const int64_t* host_idx;
  const int64_t* dev_idx;
  int64_t src_stride;        // bytes between device rows
  int64_t draft_src_stride;  // bytes between device draft rows
  uint32_t length;
  uint32_t layers;
  uint32_t draft_layers;     // 0: no draft rows
  uint32_t draft_row_bytes;  // a multiple of 16
};

// kRow FP8 values per (token, layer) row; kInvGlobalScale = 1 / g, a power of two; kWarps warps per CTA;
// kGroup steps of layer rows loaded before any is quantized.
template <uint32_t kRow, uint32_t kInvGlobalScale, uint32_t kWarps, uint32_t kGroup>
__global__ void __launch_bounds__(kWarps * device::kWarpThreads) hicache_nvfp4_backup_kernel(
    const __grid_constant__ Nvfp4BackupParams p) {
  using namespace device;
  using namespace device::nvfp4_backup;
  constexpr uint32_t kLanesPerRow = kRow / 32;  // each lane: 32 values = two blocks
  constexpr uint32_t kRowsPerStep = kWarpThreads / kLanesPerRow;
  constexpr uint32_t kPacked = kRow / 2;
  constexpr uint32_t kScales = kRow / 16;
  static_assert(kWarpThreads % kLanesPerRow == 0, "a warp covers whole rows");
  extern __shared__ __align__(16) uint8_t smem_raw[];

  const uint32_t warp = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  const uint32_t layers = p.layers;
  uint8_t* scales = smem_raw + warp * layers * kScales;
  const uint32_t lrow = lane / kLanesPerRow;  // which of the step's rows
  const uint32_t col = lane % kLanesPerRow;   // which 32 values of it

  const uint32_t items = 2u * p.length;  // (token, K|V), K and V of a token adjacent
  for (uint32_t item = blockIdx.x * kWarps + warp; item < items; item += gridDim.x * kWarps) {
    const uint32_t t = item >> 1, kv = item & 1u;
    const int64_t dev = p.dev_idx[t];
    const int64_t host = p.host_idx[t];
    const uint64_t* table = p.src_ptrs[kv];
    uint8_t* data_out = static_cast<uint8_t*>(p.dst_data[kv]) + host * layers * kPacked;

    for (uint32_t l0 = 0; l0 < layers; l0 += kRowsPerStep * kGroup) {
      uint4 x[kGroup][2];
#pragma unroll
      for (uint32_t g = 0; g < kGroup; ++g) {
        const uint32_t l = l0 + g * kRowsPerStep + lrow;
        if (l < layers) {
          const uint4* src = reinterpret_cast<const uint4*>(
              reinterpret_cast<const uint8_t*>(table[l]) + dev * p.src_stride + col * 32);
          x[g][0] = __ldg(src);
          x[g][1] = __ldg(src + 1);
        }
      }
#pragma unroll
      for (uint32_t g = 0; g < kGroup; ++g) {
        const uint32_t l = l0 + g * kRowsPerStep + lrow;
        if (l < layers) {
          uint32_t s0, s1;
          const uint32_t w0[4] = {x[g][0].x, x[g][0].y, x[g][0].z, x[g][0].w};
          const uint32_t w1[4] = {x[g][1].x, x[g][1].y, x[g][1].z, x[g][1].w};
          const uint2 q0 = quantize_block<kInvGlobalScale>(w0, s0);
          const uint2 q1 = quantize_block<kInvGlobalScale>(w1, s1);
          // Lanes of a step write consecutive 16 B: rows l, l + 1 are adjacent in the token's region.
          *reinterpret_cast<uint4*>(data_out + l * kPacked + col * 16) = make_uint4(q0.x, q0.y, q1.x, q1.y);
          *reinterpret_cast<uint16_t*>(scales + l * kScales + col * 2) = static_cast<uint16_t>(s0 | (s1 << 8));
        }
      }
    }
    __syncwarp();
    uint8_t* scale_out = static_cast<uint8_t*>(p.dst_scale[kv]) + host * layers * kScales;
    for (uint32_t o = lane * 16; o < layers * kScales; o += kWarpThreads * 16) {
      *reinterpret_cast<uint4*>(scale_out + o) = *reinterpret_cast<const uint4*>(scales + o);
    }
    if (p.draft_layers) {
      const uint32_t row_units = p.draft_row_bytes / 16;
      const uint64_t* draft_table = p.draft_ptrs[kv];
      uint4* draft_out = reinterpret_cast<uint4*>(
          static_cast<uint8_t*>(p.draft_dst[kv]) + host * p.draft_layers * p.draft_row_bytes);
      for (uint32_t u = lane; u < p.draft_layers * row_units; u += kWarpThreads) {
        const uint32_t l = u / row_units;
        draft_out[u] = __ldg(reinterpret_cast<const uint4*>(
            reinterpret_cast<const uint8_t*>(draft_table[l]) + dev * p.draft_src_stride) + (u - l * row_units));
      }
    }
    __syncwarp();
  }
}

template <uint32_t kRow, uint32_t kInvGlobalScale, uint32_t kWarps, uint32_t kGroup>
struct HiCacheNvfp4BackupKernel {
  static_assert(kRow % 512 == 0 || kRow == 256, "rows of 256 or a multiple of 512 FP8 values");
  static_assert(kInvGlobalScale > 0 && (kInvGlobalScale & (kInvGlobalScale - 1)) == 0, "1 / g a power of two");
  static constexpr uint32_t kThreads = kWarps * device::kWarpThreads;
  static constexpr auto kernel = hicache_nvfp4_backup_kernel<kRow, kInvGlobalScale, kWarps, kGroup>;

  static int64_t smem_bytes(int64_t layers) {
    return static_cast<int64_t>(kWarps) * layers * (kRow / 16);
  }

  // One backup op: K and V rows of `length` tokens (and their draft rows: empty draft tables for none) on
  // `stream`, at most `block_quota` CTAs.
  static void launch(
      const tvm::ffi::TensorView k_ptrs,
      const tvm::ffi::TensorView v_ptrs,
      const tvm::ffi::TensorView draft_k_ptrs,
      const tvm::ffi::TensorView draft_v_ptrs,
      const tvm::ffi::TensorView host_idx,
      const tvm::ffi::TensorView dev_idx,
      const int64_t k_data,
      const int64_t v_data,
      const int64_t k_scale,
      const int64_t v_scale,
      const int64_t draft_k_data,
      const int64_t draft_v_data,
      const int64_t src_stride,
      const int64_t draft_src_stride,
      const int64_t draft_row_bytes,
      const int64_t block_quota,
      const int64_t stream) {
    using namespace host;
    auto N = SymbolicSize{"layers"};
    auto D = SymbolicSize{"draft layers"};
    auto L = SymbolicSize{"indices length"};
    auto device_ = SymbolicDevice{};
    TensorMatcher({N}).with_dtype<int64_t, uint64_t>().with_device<kDLGPU>(device_).verify(k_ptrs).verify(v_ptrs);
    TensorMatcher({D}).with_dtype<int64_t, uint64_t>().with_device<kDLGPU>(device_).verify(draft_k_ptrs).verify(draft_v_ptrs);
    const int64_t draft_layers = D.unwrap();
    if (draft_layers) {
      RuntimeCheck(draft_row_bytes > 0 && draft_row_bytes % 16 == 0, "NVFP4 backup: draft rows must be 16 B multiples");
      RuntimeCheck(draft_src_stride % 16 == 0 && draft_k_data % 16 == 0 && draft_v_data % 16 == 0,
                   "NVFP4 backup: draft buffers must be 16 B aligned");
    }
    TensorMatcher({L}).with_dtype<int64_t>().with_device<kDLGPU>(device_).verify(host_idx).verify(dev_idx);
    RuntimeCheck(block_quota > 0, "NVFP4 backup: block quota must be positive");
    RuntimeCheck(src_stride >= kRow && src_stride % 16 == 0, "NVFP4 backup: device rows must be 16 B aligned");
    for (const int64_t ptr : {k_data, v_data, k_scale, v_scale})
      RuntimeCheck(ptr % 16 == 0, "NVFP4 backup: host buffers must be 16 B aligned");
    const int64_t length = L.unwrap();
    if (length == 0) return;
    RuntimeCheck(length < (1ll << 31), "NVFP4 backup: too many tokens");
    const int64_t layers = N.unwrap();
    const auto smem = static_cast<std::size_t>(smem_bytes(layers));
    RuntimeCheck(smem <= 48 * 1024, "NVFP4 backup: too many layers for the per-warp scale staging");
    const Nvfp4BackupParams params{
        .src_ptrs = {static_cast<const uint64_t*>(k_ptrs.data_ptr()), static_cast<const uint64_t*>(v_ptrs.data_ptr())},
        .dst_data = {reinterpret_cast<void*>(k_data), reinterpret_cast<void*>(v_data)},
        .dst_scale = {reinterpret_cast<void*>(k_scale), reinterpret_cast<void*>(v_scale)},
        .draft_ptrs = {static_cast<const uint64_t*>(draft_k_ptrs.data_ptr()),
                       static_cast<const uint64_t*>(draft_v_ptrs.data_ptr())},
        .draft_dst = {reinterpret_cast<void*>(draft_k_data), reinterpret_cast<void*>(draft_v_data)},
        .host_idx = static_cast<const int64_t*>(host_idx.data_ptr()),
        .dev_idx = static_cast<const int64_t*>(dev_idx.data_ptr()),
        .src_stride = src_stride,
        .draft_src_stride = draft_src_stride,
        .length = static_cast<uint32_t>(length),
        .layers = static_cast<uint32_t>(layers),
        .draft_layers = static_cast<uint32_t>(draft_layers),
        .draft_row_bytes = static_cast<uint32_t>(draft_layers ? draft_row_bytes : 0),
    };
    const uint32_t warps_needed = static_cast<uint32_t>(div_ceil(2 * length, static_cast<int64_t>(kWarps)));
    const uint32_t grid = std::min<uint32_t>(warps_needed, static_cast<uint32_t>(block_quota));
    LaunchKernel(grid, kThreads, reinterpret_cast<cudaStream_t>(stream), smem)(kernel, params);
  }
};

}  // namespace sglang
