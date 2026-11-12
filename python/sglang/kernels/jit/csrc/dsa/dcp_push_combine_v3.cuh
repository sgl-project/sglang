// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// K31HT_DCP_COMBINE=v3 form of the DCP push + combine
// (side-by-side with the served dsa/dcp_push_combine.cuh, which is NOT changed; the knob OFF loads the served module).
// Same protocol, slot layout, phase counter and PDL points as the served kernel; only the work split changes:
// a warp owns a PAIR of rows (unit u -> rows 2u, 2u + 1) and each HALF-warp (16 lanes = one 256 B row) runs one row in
// both stages, so a warp keeps two rows' loads / polls in flight where the served kernel keeps one with 16 lanes idle.
//   stage 1 per half: N-1 payload loads, then N-1 stores, then lanes 0..N-2 store the N-1 LSE words.
//   stage 2 per half: every partial's first load, then lanes 0..N-1 poll / read the N LSE words, half-warp LSE max (lanes >= N
//   hold -inf, as in the served 32-lane max), weights, re-poll only the unarrived payload words, zero restores, accumulation in
//   the served order (i = 0..N-1) with the served arithmetic -> output bitwise equal to the served kernel by construction.
// Row -> block: block(row) = ((row / 2) mod num_gwarps) / kWarps in BOTH stages, and the grid is a function of the row count
// alone (identical on every rank), so the producer and the consumer of a row use the same block and phase counter, as the
// served kernel requires. 512 threads (1 block / SM): world 8 compiles to 80 registers, no spill (1024 spills at world 8).
// Bitwise knob; default OFF = served.
// Sync: no new or changed sync edge. Every 16 B word is self-validating (per-word lamport,
// data-is-flag, the served form = proven in production); only which lanes issue which independent relaxed accesses changes;
// the zero restore of a word still follows its own read in the same thread; re-arm chain unchanged (counter flip after
// __syncthreads, kernel completion, PDL wait).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/extra/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sglang {

using device::distributed::Counter;
using host::distributed::CommunicatorRef;

namespace dcp_push_combine_v3 {

constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kRowBytes = kHeadDim * 2;   // bf16 payload
constexpr uint32_t kRowVecs = kRowBytes / 16;  // 16 = one half-warp
constexpr uint32_t kBlockSize = 512;
constexpr uint32_t kWarps = kBlockSize / 32;
constexpr uint32_t kNegZeroWord = 0x80000000u;

struct alignas(16) Vec16 {
  uint32_t w[4];
};

template <uint32_t kWorldSize>
struct Params {
  const uint8_t* v;    // [B, H, kHeadDim] bf16, last dim contiguous
  int64_t v_stride_b;  // bytes
  int64_t v_stride_h;  // bytes
  const float* lse;    // [B, H]
  int64_t lse_stride_b;
  int64_t lse_stride_h;
  uint8_t* out;  // [B, H / N * kHeadDim] bf16, contiguous
  uint8_t* workspaces[kWorldSize];
  Counter* counter;
  uint32_t slot_bytes;
  uint32_t lse_offset;
  uint32_t rank;
  uint32_t num_rows;  // B * H / N
  uint32_t heads_per_rank;
  bool lse_base_e;
};

SGL_DEVICE float bf16_lo(uint32_t w) {
  return __uint_as_float(w << 16);
}

SGL_DEVICE float bf16_hi(uint32_t w) {
  return __uint_as_float(w & 0xffff0000u);
}

SGL_DEVICE uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&p);
}

// A payload word that is +0.0 in both halves would read as "not arrived".
SGL_DEVICE void mark_payload(Vec16& x) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (x.w[i] == 0) x.w[i] = 0x00008000u;  // low half -0.0
  }
}

SGL_DEVICE bool arrived(const Vec16& x) {
  return x.w[0] != 0 && x.w[1] != 0 && x.w[2] != 0 && x.w[3] != 0;
}

// The DCP combine treats a NaN or +inf LSE (no local key) as -inf.
SGL_DEVICE float sanitize_lse(float lse) {
  return (lse != lse || lse == INFINITY) ? -INFINITY : lse;
}

template <uint32_t kWorldSize, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize, 1) void push_combine_v3_kernel(const __grid_constant__ Params<kWorldSize> p) {
  static_assert(kWorldSize >= 2 && kWorldSize <= 16, "v3 keeps one LSE lane per peer inside a half-warp");
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t half = lane >> 4;
  const uint32_t hl = lane & 15u;
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t bx = blockIdx.x;
  const uint32_t gwarp = bx * kWarps + warp;
  const uint32_t num_gwarps = gridDim.x * kWarps;
  const uint32_t num_units = (p.num_rows + 1) / 2;
  const uint32_t r = p.rank;
  const uint32_t hpr = p.heads_per_rank;

  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t phase = p.counter[bx].get() & 1;
  const auto slot = [&](uint32_t dst, uint32_t src) {
    return p.workspaces[dst] + (phase * kWorldSize + src) * p.slot_bytes;
  };

  // stage 1: push the partials of every peer's heads into that peer's slot (no shuffles: divergence is harmless)
  for (uint32_t u = gwarp; u < num_units; u += num_gwarps) {
    const uint32_t row = 2 * u + half;
    if (row >= p.num_rows) continue;
    const uint32_t b = row / hpr;
    const uint32_t j = row - b * hpr;
    Vec16 x[kWorldSize - 1];
#pragma unroll
    for (uint32_t k = 1; k < kWorldSize; ++k) {
      const uint32_t h = ((r + k) % kWorldSize) * hpr + j;
      device::ptx::ld_global_16B(x[k - 1], p.v + b * p.v_stride_b + h * p.v_stride_h, hl);
    }
#pragma unroll
    for (uint32_t k = 1; k < kWorldSize; ++k) {
      const uint32_t dst = (r + k) % kWorldSize;
      mark_payload(x[k - 1]);
      device::ptx::st_relaxed_16B(x[k - 1], slot(dst, r) + row * kRowBytes, hl);
    }
    if (hl < kWorldSize - 1) {
      const uint32_t k = hl + 1;
      const uint32_t dst = (r + k) % kWorldSize;
      const uint32_t h = dst * hpr + j;
      const float lse = p.lse[b * p.lse_stride_b + h * p.lse_stride_h];
      uint32_t bits = __float_as_uint(lse);
      Vec16 y{{bits == 0 ? kNegZeroWord : bits, kNegZeroWord, kNegZeroWord, kNegZeroWord}};
      device::ptx::st_relaxed_16B(y, slot(dst, r) + p.lse_offset + row * 16, 0);
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();

  // stage 2: combine the N partials of each local head by LSE weight (warp-uniform trip count: shuffles stay convergent)
  const Vec16 zero{{0, 0, 0, 0}};
  for (uint32_t u = gwarp; u < num_units; u += num_gwarps) {
    const uint32_t row = 2 * u + half;
    const bool valid = row < p.num_rows;
    const uint32_t b = valid ? row / hpr : 0;
    const uint32_t j = valid ? row - b * hpr : 0;
    const uint32_t h_local = r * hpr + j;

    Vec16 xs[kWorldSize];
    if (valid) {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i == r) {
          device::ptx::ld_global_16B(xs[i], p.v + b * p.v_stride_b + h_local * p.v_stride_h, hl);
        } else {
          device::ptx::ld_relaxed_16B(xs[i], slot(r, i) + row * kRowBytes, hl);
        }
      }
    }
    float lse = -INFINITY;
    if (valid && hl < kWorldSize) {
      if (hl == r) {
        lse = p.lse[b * p.lse_stride_b + h_local * p.lse_stride_h];
      } else {
        uint8_t* const src_lse = slot(r, hl) + p.lse_offset + row * 16;
        Vec16 x;
        do {
          device::ptx::ld_relaxed_16B(x, src_lse, 0);
        } while (!arrived(x));
        device::ptx::st_global_16B(zero, src_lse, 0);
        lse = __uint_as_float(x.w[0] == kNegZeroWord ? 0u : x.w[0]);
      }
      lse = sanitize_lse(lse);
    }
    float lse_max = lse;  // max over this half's 16 lanes
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
      lse_max = fmaxf(lse_max, __shfl_xor_sync(0xffffffffu, lse_max, off));
    }
    if (lse_max == -INFINITY) lse_max = 0.f;
    float w = 0.f;
    if (hl < kWorldSize) {
      w = p.lse_base_e ? expf(lse - lse_max) : exp2f(lse - lse_max);
    }
    float ws[kWorldSize];
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      const uint32_t src_lane = (half << 4) + i;
      ws[i] = __shfl_sync(0xffffffffu, w, src_lane);
    }
    if (valid) {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) {
          uint8_t* const src = slot(r, i) + row * kRowBytes;
          while (!arrived(xs[i])) device::ptx::ld_relaxed_16B(xs[i], src, hl);
        }
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) device::ptx::st_global_16B(zero, slot(r, i) + row * kRowBytes, hl);
      }
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float weight_sum = 0.f;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {  // served order and ops
        weight_sum += ws[i];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          acc[2 * q] += bf16_lo(xs[i].w[q]) * ws[i];
          acc[2 * q + 1] += bf16_hi(xs[i].w[q]) * ws[i];
        }
      }
      const float inv = 1.f / fmaxf(weight_sum, 1.f);
      Vec16 o;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        o.w[q] = pack_bf16x2(acc[2 * q] * inv, acc[2 * q + 1] * inv);
      }
      device::ptx::st_global_16B(o, p.out + row * kRowBytes, hl);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) p.counter[bx].set(phase ^ 1);
}

}  // namespace dcp_push_combine_v3

template <uint32_t kWorldSize, bool kUsePDL>
struct DcpPushCombineV3Kernel {
  static void
  run(CommunicatorRef ref,
      tvm::ffi::TensorView v,
      tvm::ffi::TensorView lse,
      tvm::ffi::TensorView out,
      bool lse_base_e) {
    using namespace host;
    using namespace dcp_push_combine_v3;
    const auto& push = ref.get()->get_push_obj();
    CHECK_HOST(push.world_size == kWorldSize) << push.world_size;
    SymbolicSize B = {"batch"}, H = {"num_heads"}, HO = {"local_heads_x_head_dim"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({B, H, kHeadDim}).with_strides({-1, -1, 1}).with_dtype<bf16_t>().with_device(device).verify(v);
    TensorMatcher({B, H}).with_strides({-1, -1}).with_dtype<float>().with_device(device).verify(lse);
    TensorMatcher({B, HO}).with_dtype<bf16_t>().with_device(device).verify(out);
    const int64_t num_heads = H.unwrap();
    CHECK_HOST(num_heads % kWorldSize == 0) << num_heads;
    const int64_t hpr = num_heads / kWorldSize;
    CHECK_HOST(HO.unwrap() == hpr * kHeadDim) << HO.unwrap();
    CHECK_HOST(out.stride(0) == hpr * kHeadDim && out.stride(1) == 1);
    const int64_t batch = B.unwrap();
    if (batch == 0) return;
    const int64_t num_rows = batch * hpr;
    const int64_t need = num_rows * (kRowBytes + 16);
    CHECK_HOST(need <= push.slot_bytes) << "DCP push-combine v3 needs " << need << " B per slot, plane has "
                                        << push.slot_bytes;
    CHECK_HOST(reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0 && v.stride(0) % 8 == 0 && v.stride(1) % 8 == 0);
    Params<kWorldSize> p{};
    p.v = static_cast<const uint8_t*>(v.data_ptr());
    p.v_stride_b = v.stride(0) * 2;
    p.v_stride_h = v.stride(1) * 2;
    p.lse = static_cast<const float*>(lse.data_ptr());
    p.lse_stride_b = lse.stride(0);
    p.lse_stride_h = lse.stride(1);
    p.out = static_cast<uint8_t*>(out.data_ptr());
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      p.workspaces[i] = push.workspaces[i];
    }
    p.counter = push.counter;
    p.slot_bytes = static_cast<uint32_t>(push.slot_bytes);
    p.lse_offset = static_cast<uint32_t>(num_rows * kRowBytes);
    p.rank = push.rank;
    p.num_rows = static_cast<uint32_t>(num_rows);
    p.heads_per_rank = static_cast<uint32_t>(hpr);
    p.lse_base_e = lse_base_e;
    // A half-warp per row: the grid is a function of the row count alone, so every rank maps a row to the same block
    // and that block's phase counter (blocks past it keep their phase), as in the served kernel.
    const int64_t units = div_ceil(num_rows, int64_t{2});
    const auto num_blocks = std::min<int64_t>(push.num_blocks, div_ceil(units, int64_t{kWarps}));
    LaunchKernel(static_cast<uint32_t>(num_blocks), kBlockSize, v.device())
        .enable_pdl(kUsePDL)(push_combine_v3_kernel<kWorldSize, kUsePDL>, p);
  }
};

}  // namespace sglang
