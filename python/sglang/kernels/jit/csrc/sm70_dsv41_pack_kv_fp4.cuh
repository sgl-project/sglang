// SPDX-License-Identifier: Apache-2.0
// SM70 DeepSeek-V4.1 compressed-KV pack: fp16 row -> 288 B
// (256 E2M1 nibble bytes, even-low, + 32 E4M3 scale bytes).
// One warp (32 lanes) per row: coalesced 16 B fp16 loads, local per-block
// amax (or a 2/4-lane max reduction -- max is order-independent).
#include "sm70_dsv41_fp4.cuh"

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cstdint>

namespace sglang::sm70_dsv41 {

inline constexpr uint32_t kPackWarpsPerBlock = 8;
inline constexpr uint32_t kPackBlockThreads = kPackWarpsPerBlock * device::kWarpThreads;

/**
 * \brief Convert 8 contiguous fp16 values to fp32. ``vec_offset`` is in
 * units of 8-element (16 B) vectors from ``ptr``.
 */
SGL_DEVICE void load_fp16x8_as_fp32(float out[8], const fp16_t* ptr, int vec_offset) {
  device::AlignedVector<fp16_t, 8> v;
  v.load(ptr, vec_offset);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = static_cast<float>(v[i]);
  }
}

/**
 * \brief RoPE + fp16-round the pairs of ``vals`` whose source index is in
 * the tail ``[dim - rope_dim, dim)``. Same expressions as ``rope_tail_fp32_dim``.
 */
SGL_DEVICE void rope_fp16_round_chunk8(
    float vals[8], const float* freqs, int e0, int dim, int rope_dim) {
  const int head = dim - rope_dim;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const int e = e0 + i;
    if (e >= head) {
      const int j = (e - head) >> 1;
      const float re = vals[i];
      const float im = vals[i + 1];
      const float fr = freqs[2 * j];
      const float fi = freqs[2 * j + 1];
      vals[i] = re * fr - im * fi;
      vals[i + 1] = re * fi + im * fr;
      vals[i] = static_cast<float>(static_cast<fp16_t>(vals[i]));
      vals[i + 1] = static_cast<float>(static_cast<fp16_t>(vals[i + 1]));
    }
  }
}

SGL_DEVICE float amax8(const float vals[8]) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    amax = fmaxf(amax, fabsf(vals[i]));
  }
  return amax;
}

/**
 * \brief Pack one fp16 head-dim row to the 288-byte E2M1/E4M3 layout.
 *
 * Mapping: 1 warp / row. Two coalesced 16 B waves cover the 512-fp16 row
 * (8 elements / lane / wave). Each 16-element FP4 block is a pair of
 * consecutive lanes; amax uses ``fmaxf`` + ``reduce_max<2>`` (max is
 * associative / NaN-ignoring, same as a sequential ``fmaxf`` from 0).
 *
 * Rounding chain: optional RoPE in fp32, round the tail to fp16, then per-16
 * amax, e4m3(amax/6) clamped to [2^-9, 448], RNE E2M1. The fp16 rounding after
 * RoPE is not fused away.
 */
template <bool kApplyRope, bool kScatterAt>
__global__ void pack_kv_fp4_kernel(uint8_t* __restrict__ dst,
                                   const fp16_t* __restrict__ src,
                                   const float* __restrict__ freqs,
                                   const int64_t* __restrict__ positions,
                                   uint32_t n_rows,
                                   uint32_t rope_dim,
                                   int32_t row_div,
                                   int32_t freq_mul,
                                   int32_t dst_mod) {
  const uint32_t lane = threadIdx.x & (device::kWarpThreads - 1u);
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= n_rows) {
    return;
  }
  int32_t dst_row;
  int64_t freq_row;
  pack_scatter_rows<kScatterAt>(row, positions, row_div, freq_mul, dst_mod, &dst_row, &freq_row);
  const fp16_t* in = src + static_cast<int64_t>(row) * kHeadDim;
  uint8_t* out = dst + static_cast<int64_t>(dst_row) * kKvRowBytes;
  uint8_t* payload = out;
  uint8_t* scales = out + kKvPayloadBytes;
  const float* row_freqs = nullptr;
  if constexpr (kApplyRope) {
    row_freqs = freqs + freq_row * rope_dim;
  }
  const int rope_head = kHeadDim - static_cast<int>(rope_dim);

  // Wave r covers elements [256*r, 256*r + 256): lane L holds 8 contiguous
  // fp16. FP4 block = 16 elems = 2 lanes; block id = 16*r + L/2.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int e0 = r * 256 + static_cast<int>(lane) * 8;
    float vals[8];
    load_fp16x8_as_fp32(vals, in, r * 32 + static_cast<int>(lane));
    if constexpr (kApplyRope) {
      if (e0 + 7 >= rope_head) {
        rope_fp16_round_chunk8(vals, row_freqs, e0, kHeadDim, static_cast<int>(rope_dim));
      }
    }

    float amax = amax8(vals);
    amax = device::warp::reduce_max<2>(amax);

    float scale = fminf(fmaxf(amax * (1.0f / kFp4Max), kE4m3MinNormalScale), kFp8Max);
    const uint8_t scale_bits = float_to_e4m3fn(scale);
    scale = e4m3fn_to_float(scale_bits);
    const int block = r * 16 + static_cast<int>(lane >> 1);
    if ((lane & 1u) == 0u) {
      scales[block] = scale_bits;
    }

    device::AlignedVector<uint8_t, 4> packed;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const float s0 = fminf(fmaxf(vals[i] / scale, -kFp4Max), kFp4Max);
      const float s1 = fminf(fmaxf(vals[i + 1] / scale, -kFp4Max), kFp4Max);
      const uint8_t c0 = e2m1_code(round_e2m1(s0));
      const uint8_t c1 = e2m1_code(round_e2m1(s1));
      packed[i >> 1] = static_cast<uint8_t>((c0 & 0x0Fu) | ((c1 & 0x0Fu) << 4));
    }
    packed.store(payload, static_cast<int64_t>(block) * 2 + static_cast<int>(lane & 1u));
  }
}

/**
 * \brief Pack one fp16 head-dim row to the 528-byte SWA FP8 layout
 * (512 E4M3 bytes + 16 UE8M0 exponent bytes). HF act_quant per 32.
 *
 * Mapping: 1 warp / row, same two 16 B load waves as ``pack_kv_fp4_kernel``.
 * Each 32-element FP8 block is 4 consecutive lanes; amax uses
 * ``reduce_max<4>``.
 */
template <bool kApplyRope, bool kScatterAt>
__global__ void pack_swa_fp8_kernel(uint8_t* __restrict__ dst,
                                    const fp16_t* __restrict__ src,
                                    const float* __restrict__ freqs,
                                    const int64_t* __restrict__ positions,
                                    uint32_t n_rows,
                                    uint32_t rope_dim,
                                    int32_t row_div,
                                    int32_t freq_mul,
                                    int32_t dst_mod) {
  const uint32_t lane = threadIdx.x & (device::kWarpThreads - 1u);
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= n_rows) {
    return;
  }
  int32_t dst_row;
  int64_t freq_row;
  pack_scatter_rows<kScatterAt>(row, positions, row_div, freq_mul, dst_mod, &dst_row, &freq_row);
  const fp16_t* in = src + static_cast<int64_t>(row) * kHeadDim;
  uint8_t* out = dst + static_cast<int64_t>(dst_row) * kSwaRowBytes;
  uint8_t* payload = out;
  uint8_t* exps = out + kSwaPayloadBytes;
  const float* row_freqs = nullptr;
  if constexpr (kApplyRope) {
    row_freqs = freqs + freq_row * rope_dim;
  }
  const int rope_head = kHeadDim - static_cast<int>(rope_dim);

  // Wave r covers elements [256*r, 256*r + 256). FP8 block = 32 elems = 4
  // lanes; block id = 8*r + L/4.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int e0 = r * 256 + static_cast<int>(lane) * 8;
    float vals[8];
    load_fp16x8_as_fp32(vals, in, r * 32 + static_cast<int>(lane));
    if constexpr (kApplyRope) {
      if (e0 + 7 >= rope_head) {
        rope_fp16_round_chunk8(vals, row_freqs, e0, kHeadDim, static_cast<int>(rope_dim));
      }
    }

    float amax = amax8(vals);
    amax = device::warp::reduce_max<4>(amax);
    amax = fmaxf(amax, kFp8ActAmaxFloor);
    const float scale = ceil_pow2(amax * (1.0f / kFp8Max));
    const uint8_t exp = static_cast<uint8_t>((__float_as_uint(scale) >> 23) & 0xFFu);
    const int block = r * 8 + static_cast<int>(lane >> 2);
    if ((lane & 3u) == 0u) {
      exps[block] = exp;
    }

    device::AlignedVector<uint8_t, 8> packed;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float q = fminf(fmaxf(vals[i] / scale, -kFp8Max), kFp8Max);
      packed[i] = float_to_e4m3fn(q);
    }
    packed.store(payload, static_cast<int64_t>(block) * 4 + static_cast<int>(lane & 3u));
  }
}

inline void pack_kv_fp4_impl(tvm::ffi::TensorView dst,
                             tvm::ffi::TensorView src,
                             const float* freqs,
                             int64_t rope_dim,
                             bool apply_rope) {
  using namespace host;
  SymbolicSize n_rows = {"num_rows"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_rows, kHeadDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(src);
  TensorMatcher({n_rows, kKvRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(dst);

  const uint32_t n = static_cast<uint32_t>(n_rows.unwrap());
  if (n == 0) {
    return;
  }
  CHECK_HOST(!apply_rope || (rope_dim > 0 && rope_dim <= kHeadDim && (rope_dim % 2 == 0)))
      << "sm70_dsv41 pack_kv_fp4: invalid rope_dim " << rope_dim;

  const uint32_t grid = div_ceil(n, kPackWarpsPerBlock);
  const DLDevice device = device_.unwrap();
  if (apply_rope) {
    LaunchKernel(grid, kPackBlockThreads, device)(
        pack_kv_fp4_kernel<true, false>,
        static_cast<uint8_t*>(dst.data_ptr()),
        static_cast<const fp16_t*>(src.data_ptr()),
        freqs,
        static_cast<const int64_t*>(nullptr),
        n,
        static_cast<uint32_t>(rope_dim),
        1,
        1,
        0);
  } else {
    LaunchKernel(grid, kPackBlockThreads, device)(
        pack_kv_fp4_kernel<false, false>,
        static_cast<uint8_t*>(dst.data_ptr()),
        static_cast<const fp16_t*>(src.data_ptr()),
        static_cast<const float*>(nullptr),
        static_cast<const int64_t*>(nullptr),
        n,
        0u,
        1,
        1,
        0);
  }
}

inline void pack_swa_fp8_impl(tvm::ffi::TensorView dst,
                              tvm::ffi::TensorView src,
                              const float* freqs,
                              int64_t rope_dim,
                              bool apply_rope) {
  using namespace host;
  SymbolicSize n_rows = {"num_rows"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_rows, kHeadDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(src);
  TensorMatcher({n_rows, kSwaRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(dst);

  const uint32_t n = static_cast<uint32_t>(n_rows.unwrap());
  if (n == 0) {
    return;
  }
  CHECK_HOST(!apply_rope || (rope_dim > 0 && rope_dim <= kHeadDim && (rope_dim % 2 == 0)))
      << "sm70_dsv41 pack_swa_fp8: invalid rope_dim " << rope_dim;

  const uint32_t grid = div_ceil(n, kPackWarpsPerBlock);
  const DLDevice device = device_.unwrap();
  if (apply_rope) {
    LaunchKernel(grid, kPackBlockThreads, device)(
        pack_swa_fp8_kernel<true, false>,
        static_cast<uint8_t*>(dst.data_ptr()),
        static_cast<const fp16_t*>(src.data_ptr()),
        freqs,
        static_cast<const int64_t*>(nullptr),
        n,
        static_cast<uint32_t>(rope_dim),
        1,
        1,
        0);
  } else {
    LaunchKernel(grid, kPackBlockThreads, device)(
        pack_swa_fp8_kernel<false, false>,
        static_cast<uint8_t*>(dst.data_ptr()),
        static_cast<const fp16_t*>(src.data_ptr()),
        static_cast<const float*>(nullptr),
        static_cast<const int64_t*>(nullptr),
        n,
        0u,
        1,
        1,
        0);
  }
}

/**
 * \brief Pack fp16 [N, 512] compressed latents to uint8 [N, 288].
 */
inline void pack_kv_fp4(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  pack_kv_fp4_impl(dst, src, nullptr, 0, false);
}

/**
 * \brief RoPE (fp32) then fp16 round then pack to [N, 288].
 *
 * \param freqs Interleaved real/imag fp32 [N, rope_dim].
 */
inline void pack_kv_fp4_rope(tvm::ffi::TensorView dst,
                             tvm::ffi::TensorView src,
                             tvm::ffi::TensorView freqs,
                             int64_t rope_dim) {
  using namespace host;
  SymbolicSize n_rows = {"num_rows"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({n_rows, rope_dim})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(freqs);
  pack_kv_fp4_impl(dst, src, static_cast<const float*>(freqs.data_ptr()), rope_dim, true);
}

/**
 * \brief Pack fp16 [N, 512] window K to uint8 [N, 528].
 */
inline void pack_swa_fp8(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  pack_swa_fp8_impl(dst, src, nullptr, 0, false);
}

/**
 * \brief RoPE then fp16 round then pack window K to [N, 528].
 */
inline void pack_swa_fp8_rope(tvm::ffi::TensorView dst,
                              tvm::ffi::TensorView src,
                              tvm::ffi::TensorView freqs,
                              int64_t rope_dim) {
  using namespace host;
  SymbolicSize n_rows = {"num_rows"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({n_rows, rope_dim})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(freqs);
  pack_swa_fp8_impl(dst, src, static_cast<const float*>(freqs.data_ptr()), rope_dim, true);
}

/**
 * \brief Pack src rows into dst[(pos / row_div) % dst_mod] with optional table RoPE.
 *
 * ``freqs`` is the full interleaved table [max_pos, rope_dim], indexed at
 * ``(pos / row_div) * freq_mul``. ``dst_mod == 0`` skips the modulo.
 */
inline void pack_kv_fp4_at(tvm::ffi::TensorView dst,
                           tvm::ffi::TensorView src,
                           tvm::ffi::Optional<tvm::ffi::TensorView> freqs,
                           int64_t rope_dim,
                           tvm::ffi::TensorView positions,
                           int64_t row_div,
                           int64_t freq_mul,
                           int64_t dst_mod) {
  using namespace host;
  SymbolicSize n_rows = {"num_rows"};
  SymbolicSize n_dst = {"dst_rows"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_rows, kHeadDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(src);
  TensorMatcher({n_dst, kKvRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(dst);
  TensorMatcher({n_rows})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(positions);

  const uint32_t n = static_cast<uint32_t>(n_rows.unwrap());
  if (n == 0) {
    return;
  }
  CHECK_HOST(row_div >= 1) << "sm70_dsv41 pack_kv_fp4_at: row_div must be >= 1";
  CHECK_HOST(freq_mul >= 0 && dst_mod >= 0)
      << "sm70_dsv41 pack_kv_fp4_at: freq_mul/dst_mod must be >= 0";
  const bool apply_rope = freqs.has_value();
  const float* freq_ptr = nullptr;
  if (apply_rope) {
    SymbolicSize n_freq = {"num_freq"};
    TensorMatcher({n_freq, rope_dim})  //
        .with_dtype<fp32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(freqs.value());
    CHECK_HOST(rope_dim > 0 && rope_dim <= kHeadDim && (rope_dim % 2 == 0))
        << "sm70_dsv41 pack_kv_fp4_at: invalid rope_dim " << rope_dim;
    freq_ptr = static_cast<const float*>(freqs.value().data_ptr());
  }

  const uint32_t grid = div_ceil(n, kPackWarpsPerBlock);
  const DLDevice device = device_.unwrap();
  const auto* pos = static_cast<const int64_t*>(positions.data_ptr());
  const int32_t rd = static_cast<int32_t>(row_div);
  const int32_t fm = static_cast<int32_t>(freq_mul);
  const int32_t dm = static_cast<int32_t>(dst_mod);
  if (apply_rope) {
    LaunchKernel(grid, kPackBlockThreads, device)(
        pack_kv_fp4_kernel<true, true>,
        static_cast<uint8_t*>(dst.data_ptr()),
        static_cast<const fp16_t*>(src.data_ptr()),
        freq_ptr,
        pos,
        n,
        static_cast<uint32_t>(rope_dim),
        rd,
        fm,
        dm);
  } else {
    LaunchKernel(grid, kPackBlockThreads, device)(
        pack_kv_fp4_kernel<false, true>,
        static_cast<uint8_t*>(dst.data_ptr()),
        static_cast<const fp16_t*>(src.data_ptr()),
        static_cast<const float*>(nullptr),
        pos,
        n,
        0u,
        rd,
        fm,
        dm);
  }
}

/// \brief Same addressing as ``pack_kv_fp4_at`` for the 528-byte SWA layout.
inline void pack_swa_fp8_at(tvm::ffi::TensorView dst,
                            tvm::ffi::TensorView src,
                            tvm::ffi::Optional<tvm::ffi::TensorView> freqs,
                            int64_t rope_dim,
                            tvm::ffi::TensorView positions,
                            int64_t row_div,
                            int64_t freq_mul,
                            int64_t dst_mod) {
  using namespace host;
  SymbolicSize n_rows = {"num_rows"};
  SymbolicSize n_dst = {"dst_rows"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_rows, kHeadDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(src);
  TensorMatcher({n_dst, kSwaRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(dst);
  TensorMatcher({n_rows})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(positions);

  const uint32_t n = static_cast<uint32_t>(n_rows.unwrap());
  if (n == 0) {
    return;
  }
  CHECK_HOST(row_div >= 1) << "sm70_dsv41 pack_swa_fp8_at: row_div must be >= 1";
  CHECK_HOST(freq_mul >= 0 && dst_mod >= 0)
      << "sm70_dsv41 pack_swa_fp8_at: freq_mul/dst_mod must be >= 0";
  const bool apply_rope = freqs.has_value();
  const float* freq_ptr = nullptr;
  if (apply_rope) {
    SymbolicSize n_freq = {"num_freq"};
    TensorMatcher({n_freq, rope_dim})  //
        .with_dtype<fp32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(freqs.value());
    CHECK_HOST(rope_dim > 0 && rope_dim <= kHeadDim && (rope_dim % 2 == 0))
        << "sm70_dsv41 pack_swa_fp8_at: invalid rope_dim " << rope_dim;
    freq_ptr = static_cast<const float*>(freqs.value().data_ptr());
  }

  const uint32_t grid = div_ceil(n, kPackWarpsPerBlock);
  const DLDevice device = device_.unwrap();
  const auto* pos = static_cast<const int64_t*>(positions.data_ptr());
  const int32_t rd = static_cast<int32_t>(row_div);
  const int32_t fm = static_cast<int32_t>(freq_mul);
  const int32_t dm = static_cast<int32_t>(dst_mod);
  if (apply_rope) {
    LaunchKernel(grid, kPackBlockThreads, device)(
        pack_swa_fp8_kernel<true, true>,
        static_cast<uint8_t*>(dst.data_ptr()),
        static_cast<const fp16_t*>(src.data_ptr()),
        freq_ptr,
        pos,
        n,
        static_cast<uint32_t>(rope_dim),
        rd,
        fm,
        dm);
  } else {
    LaunchKernel(grid, kPackBlockThreads, device)(
        pack_swa_fp8_kernel<false, true>,
        static_cast<uint8_t*>(dst.data_ptr()),
        static_cast<const fp16_t*>(src.data_ptr()),
        static_cast<const float*>(nullptr),
        pos,
        n,
        0u,
        rd,
        fm,
        dm);
  }
}

}  // namespace sglang::sm70_dsv41
