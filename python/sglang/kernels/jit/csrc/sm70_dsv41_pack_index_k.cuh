// SPDX-License-Identifier: Apache-2.0
// SM70 DeepSeek-V4.1 index-K pack: fp16 row -> 68 B
// (64 E2M1 nibble bytes, even-low, + 4 UE8M0 exponent bytes).
// One warp (32 lanes) per row: coalesced 8 B fp16 loads, 8-lane max per
// 32-element FP4 block (max is order-independent).
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

inline constexpr uint32_t kIndexPackWarpsPerBlock = 8;
inline constexpr uint32_t kIndexPackBlockThreads = kIndexPackWarpsPerBlock * device::kWarpThreads;

/**
 * \brief Pack one fp16 index-head row to the 68-byte UE8M0/E2M1 layout.
 *
 * Mapping: 1 warp / row. Each lane holds 4 consecutive fp16 (8 B coalesced
 * load). Each 32-element FP4 block is 8 consecutive lanes; amax uses
 * ``fmaxf`` + ``reduce_max<8>``.
 *
 * Same rounding chain as the CSA2 oracle: optional RoPE in fp32, round the
 * tail to fp16, per-32 amax floored at 6*2^-126, scale = ceil_pow2(amax/6),
 * RNE E2M1. Scale byte is the biased IEEE exponent of that power-of-two.
 */
template <bool kApplyRope, bool kScatterAt>
__global__ void pack_index_k_kernel(uint8_t* __restrict__ dst,
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
  const fp16_t* in = src + static_cast<int64_t>(row) * kIndexDim;
  const int e0 = static_cast<int>(lane) * 4;

  device::AlignedVector<fp16_t, 4> in_v;
  in_v.load(in, static_cast<int>(lane));
  float vals[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    vals[i] = static_cast<float>(in_v[i]);
  }

  if constexpr (kApplyRope) {
    const float* f = freqs + freq_row * rope_dim;
    const int head = kIndexDim - static_cast<int>(rope_dim);
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
      const int e = e0 + i;
      if (e >= head) {
        const int j = (e - head) >> 1;
        const float re = vals[i];
        const float im = vals[i + 1];
        const float fr = f[2 * j];
        const float fi = f[2 * j + 1];
        vals[i] = re * fr - im * fi;
        vals[i + 1] = re * fi + im * fr;
        vals[i] = static_cast<float>(static_cast<fp16_t>(vals[i]));
        vals[i + 1] = static_cast<float>(static_cast<fp16_t>(vals[i + 1]));
      }
    }
  }

  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    amax = fmaxf(amax, fabsf(vals[i]));
  }
  amax = device::warp::reduce_max<8>(amax);
  amax = fmaxf(amax, kFp4Ue8m0AmaxFloor);
  const float scale = ceil_pow2(amax * (1.0f / kFp4Max));

  uint8_t* out = dst + static_cast<int64_t>(dst_row) * kIndexRowBytes;
  uint8_t* payload = out;
  uint8_t* exps = out + kIndexPayloadBytes;
  const int block = static_cast<int>(lane >> 3);
  if ((lane & 7u) == 0u) {
    exps[block] = static_cast<uint8_t>((__float_as_uint(scale) >> 23) & 0xFFu);
  }

  // 4 e2m1 values -> 2 bytes. Row stride is 68 B (4-byte aligned, not 8),
  // so stores are 2 B, not a wider aligned vector.
#pragma unroll
  for (int i = 0; i < 4; i += 2) {
    const float s0 = fminf(fmaxf(vals[i] / scale, -kFp4Max), kFp4Max);
    const float s1 = fminf(fmaxf(vals[i + 1] / scale, -kFp4Max), kFp4Max);
    const uint8_t c0 = e2m1_code(round_e2m1(s0));
    const uint8_t c1 = e2m1_code(round_e2m1(s1));
    payload[(block * kIndexFp4Block + static_cast<int>(lane & 7u) * 4 + i) >> 1] =
        static_cast<uint8_t>((c0 & 0x0Fu) | ((c1 & 0x0Fu) << 4));
  }
}

inline void pack_index_k_impl(tvm::ffi::TensorView dst,
                              tvm::ffi::TensorView src,
                              const float* freqs,
                              int64_t rope_dim,
                              bool apply_rope) {
  using namespace host;
  SymbolicSize n_rows = {"num_rows"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_rows, kIndexDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(src);
  TensorMatcher({n_rows, kIndexRowBytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(dst);

  const uint32_t n = static_cast<uint32_t>(n_rows.unwrap());
  if (n == 0) {
    return;
  }
  CHECK_HOST(!apply_rope || (rope_dim > 0 && rope_dim <= kIndexDim && (rope_dim % 2 == 0)))
      << "sm70_dsv41 pack_index_k: invalid rope_dim " << rope_dim;

  const uint32_t grid = div_ceil(n, kIndexPackWarpsPerBlock);
  const DLDevice device = device_.unwrap();
  if (apply_rope) {
    LaunchKernel(grid, kIndexPackBlockThreads, device)(
        pack_index_k_kernel<true, false>,
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
    LaunchKernel(grid, kIndexPackBlockThreads, device)(
        pack_index_k_kernel<false, false>,
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
 * \brief Pack fp16 [N, 128] index keys to uint8 [N, 68].
 */
inline void pack_index_k(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  pack_index_k_impl(dst, src, nullptr, 0, false);
}

/**
 * \brief RoPE (fp32) then fp16 round then pack to [N, 68].
 */
inline void pack_index_k_rope(tvm::ffi::TensorView dst,
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
  pack_index_k_impl(dst, src, static_cast<const float*>(freqs.data_ptr()), rope_dim, true);
}

/// \brief Pack index-K rows into dst[(pos / row_div) % dst_mod] with table RoPE.
inline void pack_index_k_at(tvm::ffi::TensorView dst,
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

  TensorMatcher({n_rows, kIndexDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(src);
  TensorMatcher({n_dst, kIndexRowBytes})  //
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
  CHECK_HOST(row_div >= 1) << "sm70_dsv41 pack_index_k_at: row_div must be >= 1";
  CHECK_HOST(freq_mul >= 0 && dst_mod >= 0)
      << "sm70_dsv41 pack_index_k_at: freq_mul/dst_mod must be >= 0";
  const bool apply_rope = freqs.has_value();
  const float* freq_ptr = nullptr;
  if (apply_rope) {
    SymbolicSize n_freq = {"num_freq"};
    TensorMatcher({n_freq, rope_dim})  //
        .with_dtype<fp32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(freqs.value());
    CHECK_HOST(rope_dim > 0 && rope_dim <= kIndexDim && (rope_dim % 2 == 0))
        << "sm70_dsv41 pack_index_k_at: invalid rope_dim " << rope_dim;
    freq_ptr = static_cast<const float*>(freqs.value().data_ptr());
  }

  const uint32_t grid = div_ceil(n, kIndexPackWarpsPerBlock);
  const DLDevice device = device_.unwrap();
  const auto* pos = static_cast<const int64_t*>(positions.data_ptr());
  const int32_t rd = static_cast<int32_t>(row_div);
  const int32_t fm = static_cast<int32_t>(freq_mul);
  const int32_t dm = static_cast<int32_t>(dst_mod);
  if (apply_rope) {
    LaunchKernel(grid, kIndexPackBlockThreads, device)(
        pack_index_k_kernel<true, true>,
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
    LaunchKernel(grid, kIndexPackBlockThreads, device)(
        pack_index_k_kernel<false, true>,
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
