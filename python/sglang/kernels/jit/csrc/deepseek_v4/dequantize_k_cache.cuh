#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace sglang {

// ============================================================
// DSv4 paged FP8 K-cache dequantize kernel (paged variant).
//
// Mirrors `_dequantize_k_cache_paged_kernel` in
// `python/sglang/srt/layers/attention/dsv4/dequant_k_cache.py`.
//
// Per-token layout (576 bytes data + 8 bytes scales = 584 B/token):
//   [  448 B FP8 nope  | 128 B BF16 rope ] data
//   [  7 UE8M0 scales  | 1 B pad        ] scales
// Per-page layout (padded to 576-byte multiples):
//   page_size * 576 bytes data, then page_size * 8 bytes scales.
//
// Output dim per token = 448 dequantized BF16 (FP8 -> BF16 with
// per-quant-block UE8M0 scale) + 64 directly-copied BF16 = 512.
//
// Unlike DequantizeGatherKernel, this kernel takes a flat list of
// token IDs (page_table) and dequantizes arbitrary tokens from
// possibly different pages — one output row per token ID.
// ============================================================

namespace dequant_k_cache {

constexpr uint32_t kBlockSize = 128;      // threads per CTA
constexpr uint32_t kFP8Dim = 448;         // FP8 nope width
constexpr uint32_t kBF16Dim = 64;         // BF16 rope width
constexpr uint32_t kScaleDim = 8;         // 7 scales + 1 pad per token
constexpr uint32_t kQuantBlock = 64;      // FP8 quant block length
constexpr uint32_t kTokenDataSize = 576;  // 448 + 128
constexpr uint32_t kOutputDim = 512;      // kFP8Dim + kBF16Dim
constexpr uint32_t kNumQuantBlocks = 7;   // kFP8Dim / kQuantBlock
constexpr uint32_t kMaxCTAs = 256;        // CTA cap for grid-stride loop

}  // namespace dequant_k_cache

struct DequantKCacheParams {
  const uint8_t* __restrict__ k_cache;     // raw byte buffer [num_pages, bytes_per_page]
  const int32_t* __restrict__ page_table;  // [num_tokens] token IDs (flat)
  __nv_bfloat16* __restrict__ out;         // [num_tokens, 1, 512] bf16
  int64_t out_stride0;                     // bf16 elements per token row
  int64_t bytes_per_page;                  // bytes per physical page
  int32_t page_size;                       // tokens per page
  int32_t num_tokens;                      // total output tokens
  int32_t use_fp8_native;                  // 1: __nv_cvt_fp8_to_halfraw; 0: fused IEEE754 path
};

// FP8 E4M3 (FN, no Inf) -> float32 via the standard CUDA intrinsic.
__device__ __forceinline__ float kcache_fp8e4m3_to_float(uint8_t bits) {
  const __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(bits);
  const __half h = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
  return __half2float(h);
}

// UE8M0: encoded scale `s` represents `2^(s - 127)`.
__device__ __forceinline__ float kcache_ue8m0_to_scale(uint8_t encoded) {
  return exp2f(static_cast<float>(static_cast<int32_t>(encoded) - 127));
}

// Fused FP8 E4M3 + UE8M0 -> float32 via IEEE754 bit construction.
// Used on hardware without native FP8 support (e.g. SM80/A100). Produces
// bit-identical results to `kcache_fp8e4m3_to_float(x) * kcache_ue8m0_to_scale(s)`
// for all FP8 normal values. Subnormals (exp == 0) are flushed to zero (FTZ).
__device__ __forceinline__ float kcache_fused_fp8_ue8m0_to_float(uint8_t fp8_bits, uint8_t ue8m0) {
  const int32_t x = static_cast<int32_t>(fp8_bits);
  const int32_t sign = x >> 7;
  const int32_t exp = (x >> 3) & 0xF;
  const int32_t mant = x & 0x7;
  const int32_t ue32 = static_cast<int32_t>(ue8m0);
  // Construct float32: sign | (fp8_exp + ue8m0 - 7) << 23 | mant << 20
  const int32_t f32_bits = (sign << 31) | ((exp + ue32 - 7) << 23) | (mant << 20);
  // FTZ for FP8 subnormals (exp == 0) — matches the Triton fused path.
  if (exp == 0) return 0.0f;
  return __int_as_float(f32_bits);
}

__global__ __launch_bounds__(dequant_k_cache::kBlockSize, 1) void dequantize_k_cache_paged_kernel(
    const __grid_constant__ DequantKCacheParams params) {
  using namespace dequant_k_cache;

  const uint32_t global_tid = blockIdx.x * kBlockSize + threadIdx.x;
  const uint32_t total_threads = gridDim.x * kBlockSize;

  constexpr int64_t kNopeRopeBytes = static_cast<int64_t>(kTokenDataSize);   // 576
  constexpr int64_t kPaddedScalePerToken = static_cast<int64_t>(kScaleDim);  // 8
  const int64_t s_offset_bytes = static_cast<int64_t>(params.page_size) * kNopeRopeBytes;

  const bool use_fp8_native = params.use_fp8_native != 0;

  for (int32_t token_id = static_cast<int32_t>(global_tid); token_id < params.num_tokens;
       token_id += static_cast<int32_t>(total_threads)) {
    // Resolve page and in-page offset from the flat page table.
    const int32_t loc = params.page_table[token_id];
    const int32_t page_idx = loc / params.page_size;
    const int32_t in_page = loc % params.page_size;

    const int64_t page_byte_base = static_cast<int64_t>(page_idx) * params.bytes_per_page;
    const int64_t token_data_base = page_byte_base + static_cast<int64_t>(in_page) * kNopeRopeBytes;
    const int64_t token_scale_base =
        page_byte_base + s_offset_bytes + static_cast<int64_t>(in_page) * kPaddedScalePerToken;

    // Output row for this token (bf16 elements).
    __nv_bfloat16* out_row = params.out + static_cast<int64_t>(token_id) * params.out_stride0;

    // ------------------------------------------------------------
    // Stage 1: FP8 dequantization (7 blocks * 64 = 448 elements).
    //
    // One thread per token, mirroring the Triton program-per-token
    // model and the gather kernel's per-token processing.
    // ------------------------------------------------------------
#pragma unroll
    for (uint32_t qb = 0; qb < kNumQuantBlocks; ++qb) {
      const uint8_t raw_scale = params.k_cache[token_scale_base + qb];
      const float scale = use_fp8_native ? kcache_ue8m0_to_scale(raw_scale) : 0.0f;
      const uint32_t base = qb * kQuantBlock;

#pragma unroll
      for (uint32_t k = 0; k < kQuantBlock; ++k) {
        const uint32_t elem_off = base + k;
        const uint8_t fp8_byte = params.k_cache[token_data_base + elem_off];

        float f;
        if (use_fp8_native) {
          f = kcache_fp8e4m3_to_float(fp8_byte) * scale;
        } else {
          f = kcache_fused_fp8_ue8m0_to_float(fp8_byte, raw_scale);
        }
        out_row[elem_off] = __float2bfloat16(f);
      }
    }

    // ------------------------------------------------------------
    // Stage 2: BF16 direct copy of the rope tail (64 bf16 = 128 B).
    // ------------------------------------------------------------
    const __nv_bfloat16* rope_src = reinterpret_cast<const __nv_bfloat16*>(params.k_cache + token_data_base + kFP8Dim);
#pragma unroll
    for (uint32_t j = 0; j < kBF16Dim; ++j) {
      out_row[kFP8Dim + j] = rope_src[j];
    }
  }
}

struct DequantizeKCacheKernel {
  static void
  run(const tvm::ffi::TensorView out,         // [N, 1, 512] bf16
      const tvm::ffi::TensorView k_cache,     // [num_pages, page_bytes] uint8
      const tvm::ffi::TensorView page_table,  // [N] int32
      const int64_t page_size,
      const int64_t use_fp8_native) {
    using namespace host;
    using namespace dequant_k_cache;

    auto N = SymbolicSize{"num_tokens"};
    auto BS = SymbolicSize{"bytes_per_page"};
    auto OS0 = SymbolicSize{"out_stride0"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    // out: [N, 1, 512] bf16 with free stride on dim 0.
    TensorMatcher({N, 1, static_cast<int64_t>(kOutputDim)})
        .with_strides({OS0, kOutputDim, 1})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(out);

    // k_cache: [num_pages, page_bytes] uint8.
    TensorMatcher({-1, -1}).with_strides({BS, 1}).with_dtype<uint8_t>().with_device(device_).verify(k_cache);

    // page_table: [N] int32.
    TensorMatcher({N}).with_dtype<int32_t>().with_device(device_).verify(page_table);

    RuntimeCheck(page_size > 0, "page_size must be positive");

    const int32_t num_tokens = static_cast<int32_t>(N.unwrap());
    if (num_tokens == 0) return;

    const DequantKCacheParams params = {
        .k_cache = static_cast<const uint8_t*>(k_cache.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .out = static_cast<__nv_bfloat16*>(out.data_ptr()),
        .out_stride0 = OS0.unwrap(),
        .bytes_per_page = BS.unwrap(),
        .page_size = static_cast<int32_t>(page_size),
        .num_tokens = num_tokens,
        .use_fp8_native = static_cast<int32_t>(use_fp8_native),
    };

    const uint32_t desired_ctas = static_cast<uint32_t>((num_tokens + kBlockSize - 1) / kBlockSize);
    const uint32_t grid_size = std::max(1u, std::min(desired_ctas, kMaxCTAs));

    LaunchKernel(grid_size, kBlockSize, device_.unwrap(), 0)(dequantize_k_cache_paged_kernel, params);
  }
};

}  // namespace sglang
