// SPDX-License-Identifier: Apache-2.0

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <cmath>
#include <cstdint>
#include <limits>

#if !defined(USE_ROCM) && defined(SGL_CUDA_ARCH) && (SGL_CUDA_ARCH < 1200 || SGL_CUDA_ARCH >= 1300)
#error "moe_route_lowm_sm120.cuh requires SM12x (compute capability 12.x)"
#endif

namespace sglang::inkling2_moe_route_sm120 {

namespace {

constexpr int32_t kNumExperts = 128;
constexpr int32_t kTopK = 8;

SGL_DEVICE uint32_t ord_f32(float f) {
  uint32_t const b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

SGL_DEVICE uint32_t same_lanes(int32_t expert) {
  uint32_t same = 0xFFFFFFFFu;
#pragma unroll
  for (int32_t bit = 0; bit < 7; ++bit) {
    bool const set = ((expert >> bit) & 1) != 0;
    uint32_t const lanes = __ballot_sync(0xFFFFFFFFu, set);
    same &= set ? lanes : ~lanes;
  }
  bool const oob = (static_cast<uint32_t>(expert) >= static_cast<uint32_t>(kNumExperts));
  uint32_t const oob_lanes = __ballot_sync(0xFFFFFFFFu, oob);
  same &= oob ? oob_lanes : ~oob_lanes;
  return same;
}

inline void verify_scalar_fp32(tvm::ffi::TensorView tensor, host::SymbolicDevice const& device) {
  DLDevice const expected = device.unwrap();
  DLDevice const actual = tensor.device();
  CHECK_HOST(actual.device_type == expected.device_type && actual.device_id == expected.device_id)
      << "scalar tensor device mismatch";
  CHECK_HOST(host::is_type<fp32_t>(tensor.dtype())) << "scalar tensor must be float32";
  CHECK_HOST(tensor.is_contiguous() && tensor.numel() == 1) << "scalar tensor must have 1 contiguous element";
}

}  // namespace

template <int32_t kBlockThreads, int32_t kMaxTokens>
__global__ void __launch_bounds__(kBlockThreads) route_topk_meta_lowm_kernel(
    float const* __restrict__ partials,
    float const* __restrict__ bias,
    float const* __restrict__ global_scale,
    int32_t* __restrict__ ids,
    float* __restrict__ weights,
    int32_t* __restrict__ src2dst,
    int32_t* __restrict__ dst2src,
    float* __restrict__ gammas,
    int32_t* __restrict__ offsets,
    float route_scale,
    int32_t groups,
    int32_t T) {
  constexpr int32_t kMaxN = kMaxTokens * kTopK;
  constexpr int32_t kMaxChunks = (kMaxN + 31) >> 5;

  __shared__ int32_t s_ids[kMaxN];
  __shared__ float s_w[kMaxN];
  __shared__ alignas(16) uint16_t s_chunk_counts[kMaxChunks][kNumExperts];
  __shared__ int32_t s_expert_start[kNumExperts + 1];
  __shared__ int32_t s_warp_sums[4];

  int32_t const warp = static_cast<int32_t>(threadIdx.x >> 5);
  int32_t const lane = static_cast<int32_t>(device::get_lane_id());
  int32_t const n = T * kTopK;
  int32_t const chunks = (n + 31) >> 5;

  int32_t const zero_u4 = chunks * (kNumExperts / 8);
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < zero_u4; i += kBlockThreads) {
    reinterpret_cast<uint4*>(&s_chunk_counts[0][0])[i] = make_uint4(0, 0, 0, 0);
  }
  device::PDLWaitPrimary<true>();

  float b4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (bias != nullptr) {
    float4 const bv = *reinterpret_cast<float4 const*>(bias + lane * 4);
    b4[0] = bv.x;
    b4[1] = bv.y;
    b4[2] = bv.z;
    b4[3] = bv.w;
  }

  if (warp < T) {
    int32_t const t = warp;
    float4 const lv0 = *reinterpret_cast<float4 const*>(partials + static_cast<int64_t>(t) * kNumExperts + lane * 4);
    float l4[4] = {lv0.x, lv0.y, lv0.z, lv0.w};
    for (int32_t g = 1; g < groups; ++g) {
      float4 const lvg =
          *reinterpret_cast<float4 const*>(partials + (static_cast<int64_t>(g) * T + t) * kNumExperts + lane * 4);
      l4[0] += lvg.x;
      l4[1] += lvg.y;
      l4[2] += lvg.z;
      l4[3] += lvg.w;
    }

    uint64_t keys[4];
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) {
      int32_t const e = lane * 4 + j;
      float const sig = __fdiv_rn(1.0f, 1.0f + expf(-l4[j]));
      float const sel = sig + b4[j];
      keys[j] = (static_cast<uint64_t>(ord_f32(sel)) << 16) | static_cast<uint64_t>(kNumExperts - e);
    }

    int32_t pick_e = 0;
    float pick_z = 0.0f;
#pragma unroll
    for (int32_t k = 0; k < kTopK; ++k) {
      uint64_t m = max(max(keys[0], keys[1]), max(keys[2], keys[3]));
#pragma unroll
      for (int32_t s = 16; s >= 1; s >>= 1) {
        uint64_t const peer = __shfl_xor_sync(0xFFFFFFFFu, m, s);
        if (peer > m) m = peer;
      }
      int32_t my_e = -1;
      float my_z = 0.0f;
#pragma unroll
      for (int32_t j = 0; j < 4; ++j) {
        if (keys[j] == m) {
          my_e = lane * 4 + j;
          my_z = l4[j];
          keys[j] = 0ULL;
        }
      }
      int32_t const winner_lane = __ffs(__ballot_sync(0xFFFFFFFFu, my_e >= 0)) - 1;
      int32_t const chosen_e = __shfl_sync(0xFFFFFFFFu, my_e, winner_lane);
      float const chosen_z = __shfl_sync(0xFFFFFFFFu, my_z, winner_lane);
      if (lane == k) {
        pick_e = chosen_e;
        pick_z = chosen_z;
      }
    }

    float log_p = -INFINITY;
    if (lane < kTopK) {
      log_p = fminf(pick_z, 0.0f) - log1pf(expf(-fabsf(pick_z)));
    }
    float max_lp = log_p;
    max_lp = fmaxf(max_lp, __shfl_xor_sync(0xFFFFFFFFu, max_lp, 1));
    max_lp = fmaxf(max_lp, __shfl_xor_sync(0xFFFFFFFFu, max_lp, 2));
    max_lp = fmaxf(max_lp, __shfl_xor_sync(0xFFFFFFFFu, max_lp, 4));

    float s = (lane < kTopK) ? expf(log_p - max_lp) : 0.0f;
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 2);
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 1);
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 4);
    float const total = __shfl_sync(0xFFFFFFFFu, s, 0);

    if (lane < kTopK) {
      float const logsumexp = max_lp + logf(total);
      float const gs = (global_scale != nullptr) ? *global_scale : 1.0f;
      float const w = expf(log_p - logsumexp) * route_scale * gs;
      int32_t const p = t * kTopK + lane;
      ids[p] = pick_e;
      weights[p] = w;
      s_ids[p] = pick_e;
      s_w[p] = w;
    }
  }
  __syncthreads();

  int32_t const p = static_cast<int32_t>(threadIdx.x);
  int32_t const e = (p < n) ? s_ids[p] : kNumExperts;
  float const w = (p < n) ? s_w[p] : 0.0f;
  int32_t rank = 0;
  if (warp < chunks) {
    uint32_t const same = same_lanes(e);
    rank = __popc(same & ((1u << lane) - 1u));
    if (e < kNumExperts && lane == __ffs(same) - 1) {
      s_chunk_counts[warp][e] = static_cast<uint16_t>(__popc(same));
    }
  }
  __syncthreads();

  if (threadIdx.x < static_cast<uint32_t>(kNumExperts)) {
    int32_t const ex = static_cast<int32_t>(threadIdx.x);
    int32_t sum = 0;
    for (int32_t c = 0; c < chunks; ++c) {
      int32_t const cnt = s_chunk_counts[c][ex];
      s_chunk_counts[c][ex] = static_cast<uint16_t>(sum);
      sum += cnt;
    }
    int32_t x = sum;
#pragma unroll
    for (int32_t s = 1; s < 32; s *= 2) {
      int32_t const y = __shfl_up_sync(0xFFFFFFFFu, x, s);
      if (lane >= s) x += y;
    }
    if (lane == 31) s_warp_sums[warp] = x;
    s_expert_start[ex] = x - sum;
  }
  __syncthreads();

  if (warp == 0 && lane < 4) {
    int32_t ws = s_warp_sums[lane];
#pragma unroll
    for (int32_t s = 1; s < 4; s *= 2) {
      int32_t const y = __shfl_up_sync(0xFu, ws, s);
      if (lane >= s) ws += y;
    }
    s_warp_sums[lane] = ws;
  }
  __syncthreads();

  if (threadIdx.x < static_cast<uint32_t>(kNumExperts)) {
    int32_t const base = s_expert_start[threadIdx.x] + (warp ? s_warp_sums[warp - 1] : 0);
    s_expert_start[threadIdx.x] = base;
    offsets[threadIdx.x] = base;
    if (threadIdx.x == 0) {
      s_expert_start[kNumExperts] = s_warp_sums[3];
      offsets[kNumExperts] = s_warp_sums[3];
    }
  }
  __syncthreads();

  if (p < n) {
    int32_t const dst = s_expert_start[e] + s_chunk_counts[warp][e] + rank;
    dst2src[dst] = p;
    gammas[dst] = w;
    src2dst[p] = dst;
  }
  device::PDLTriggerSecondary<true>();
}

template <int32_t kBlockThreads, int32_t kMaxChunks>
__global__ void __launch_bounds__(kBlockThreads) route_metadata_lowm_kernel(
    int32_t const* __restrict__ ids,
    float const* __restrict__ weights,
    int32_t* __restrict__ src2dst,
    int32_t* __restrict__ dst2src,
    float* __restrict__ gammas,
    int32_t* __restrict__ offsets,
    int32_t n) {
  __shared__ alignas(16) uint16_t s_chunk_counts[kMaxChunks][kNumExperts];
  __shared__ int32_t s_expert_start[kNumExperts + 1];
  __shared__ int32_t s_warp_sums[4];

  int32_t const warp = static_cast<int32_t>(threadIdx.x >> 5);
  int32_t const lane = static_cast<int32_t>(device::get_lane_id());
  int32_t const chunks = (n + 31) >> 5;

  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < chunks * (kNumExperts / 8); i += kBlockThreads) {
    reinterpret_cast<uint4*>(&s_chunk_counts[0][0])[i] = make_uint4(0, 0, 0, 0);
  }
  device::PDLWaitPrimary<true>();

  int32_t const p = static_cast<int32_t>(threadIdx.x);
  int32_t const e = (p < n) ? ids[p] : kNumExperts;
  bool const valid_e = (p < n) && (static_cast<uint32_t>(e) < static_cast<uint32_t>(kNumExperts));
  float const w = (p < n) ? weights[p] : 0.0f;
  __syncthreads();

  int32_t rank = 0;
  if (warp < chunks) {
    uint32_t const same = same_lanes(e);
    rank = __popc(same & ((1u << lane) - 1u));
    if (valid_e && lane == __ffs(same) - 1) {
      s_chunk_counts[warp][e] = static_cast<uint16_t>(__popc(same));
    }
  }
  __syncthreads();

  if (threadIdx.x < static_cast<uint32_t>(kNumExperts)) {
    int32_t const ex = static_cast<int32_t>(threadIdx.x);
    int32_t sum = 0;
    for (int32_t c = 0; c < chunks; ++c) {
      int32_t const cnt = s_chunk_counts[c][ex];
      s_chunk_counts[c][ex] = static_cast<uint16_t>(sum);
      sum += cnt;
    }
    int32_t x = sum;
#pragma unroll
    for (int32_t s = 1; s < 32; s *= 2) {
      int32_t const y = __shfl_up_sync(0xFFFFFFFFu, x, s);
      if (lane >= s) x += y;
    }
    if (lane == 31) s_warp_sums[warp] = x;
    s_expert_start[ex] = x - sum;
  }
  __syncthreads();

  if (warp == 0 && lane < 4) {
    int32_t ws = s_warp_sums[lane];
#pragma unroll
    for (int32_t s = 1; s < 4; s *= 2) {
      int32_t const y = __shfl_up_sync(0xFu, ws, s);
      if (lane >= s) ws += y;
    }
    s_warp_sums[lane] = ws;
  }
  __syncthreads();

  if (threadIdx.x < static_cast<uint32_t>(kNumExperts)) {
    int32_t const base = s_expert_start[threadIdx.x] + (warp ? s_warp_sums[warp - 1] : 0);
    s_expert_start[threadIdx.x] = base;
    offsets[threadIdx.x] = base;
    if (threadIdx.x == 0) {
      s_expert_start[kNumExperts] = s_warp_sums[3];
      offsets[kNumExperts] = s_warp_sums[3];
    }
  }
  __syncthreads();

  if (p < n) {
    if (valid_e) {
      int32_t const dst = s_expert_start[e] + s_chunk_counts[warp][e] + rank;
      dst2src[dst] = p;
      gammas[dst] = w;
      src2dst[p] = dst;
    } else {
      src2dst[p] = -1;
    }
  }
  device::PDLTriggerSecondary<true>();
}

/**
 * \brief Fused router reduction, top-8 selection, renormalization, and expert-order routing metadata.
 * \param partials Contiguous [G, T, 128] FP32 router split-K partials.
 * \param bias Optional contiguous [128] FP32 expert selection bias.
 * \param global_scale Optional scalar (0-D or [1]) FP32 global route multiplier.
 * \param ids Output [T, 8] int32 selected expert IDs.
 * \param weights Output [T, 8] FP32 routing weights.
 * \param src2dst Output [T * 8] int32 source-to-sorted mapping.
 * \param dst2src Output [T * 8] int32 sorted-to-source mapping.
 * \param gammas Output [T * 8] FP32 sorted routing weights.
 * \param offsets Output [129] int32 CSR expert row offsets.
 * \param route_scale Route scale multiplier.
 */
inline void route_topk_meta_lowm(
    tvm::ffi::TensorView partials,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    tvm::ffi::Optional<tvm::ffi::TensorView> global_scale,
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView src2dst,
    tvm::ffi::TensorView dst2src,
    tvm::ffi::TensorView gammas,
    tvm::ffi::TensorView offsets,
    double route_scale) {
  auto G = host::SymbolicSize{"groups"};
  auto T = host::SymbolicSize{"tokens"};
  auto N = host::SymbolicSize{"routed_rows"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();
  host::TensorMatcher({G, T, kNumExperts})  //
      .with_dtype<fp32_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(partials);
  if (bias.has_value()) {
    host::TensorMatcher({kNumExperts})  //
        .with_dtype<fp32_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(bias.value());
  }
  if (global_scale.has_value()) {
    verify_scalar_fp32(global_scale.value(), device);
  }
  host::TensorMatcher({T, kTopK}).with_dtype<int32_t>().with_device(device).verify(ids);
  host::TensorMatcher({T, kTopK}).with_dtype<fp32_t>().with_device(device).verify(weights);
  host::TensorMatcher({N}).with_dtype<int32_t>().with_device(device).verify(src2dst).verify(dst2src);
  host::TensorMatcher({N}).with_dtype<fp32_t>().with_device(device).verify(gammas);
  host::TensorMatcher({kNumExperts + 1}).with_dtype<int32_t>().with_device(device).verify(offsets);

  int64_t const groups = G.unwrap();
  int64_t const tokens = T.unwrap();
  CHECK_HOST(groups >= 1 && groups <= std::numeric_limits<int32_t>::max())
      << "route_topk_meta_lowm: invalid groups " << groups;
  CHECK_HOST(tokens >= 0 && tokens <= 16) << "route_topk_meta_lowm: requires 0 <= T <= 16, got " << tokens;
  CHECK_HOST(N.unwrap() == tokens * kTopK) << "route_topk_meta_lowm: routed row count mismatch";
  if (tokens == 0) return;

  float const* bias_ptr = bias.has_value() ? static_cast<float const*>(bias.value().data_ptr()) : nullptr;
  float const* gs_ptr = global_scale.has_value() ? static_cast<float const*>(global_scale.value().data_ptr()) : nullptr;
  if (tokens <= 8) {
    host::LaunchKernel(1, 256, device.unwrap())
        .enable_pdl(true)(
            route_topk_meta_lowm_kernel<256, 8>,
            static_cast<float const*>(partials.data_ptr()),
            bias_ptr,
            gs_ptr,
            static_cast<int32_t*>(ids.data_ptr()),
            static_cast<float*>(weights.data_ptr()),
            static_cast<int32_t*>(src2dst.data_ptr()),
            static_cast<int32_t*>(dst2src.data_ptr()),
            static_cast<float*>(gammas.data_ptr()),
            static_cast<int32_t*>(offsets.data_ptr()),
            static_cast<float>(route_scale),
            static_cast<int32_t>(groups),
            static_cast<int32_t>(tokens));
  } else {
    host::LaunchKernel(1, 512, device.unwrap())
        .enable_pdl(true)(
            route_topk_meta_lowm_kernel<512, 16>,
            static_cast<float const*>(partials.data_ptr()),
            bias_ptr,
            gs_ptr,
            static_cast<int32_t*>(ids.data_ptr()),
            static_cast<float*>(weights.data_ptr()),
            static_cast<int32_t*>(src2dst.data_ptr()),
            static_cast<int32_t*>(dst2src.data_ptr()),
            static_cast<float*>(gammas.data_ptr()),
            static_cast<int32_t*>(offsets.data_ptr()),
            static_cast<float>(route_scale),
            static_cast<int32_t>(groups),
            static_cast<int32_t>(tokens));
  }
}

/**
 * \brief Single-CTA stable counting sort producing routing metadata for 128 experts.
 * \param ids Flat [N] int32 expert IDs in [0, 128).
 * \param weights Flat [N] FP32 routing weights.
 * \param src2dst Output [N] int32 source-to-sorted mapping.
 * \param dst2src Output [N] int32 sorted-to-source mapping.
 * \param gammas Output [N] FP32 sorted routing weights.
 * \param offsets Output [129] int32 CSR expert row offsets.
 */
inline void route_metadata_lowm(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView src2dst,
    tvm::ffi::TensorView dst2src,
    tvm::ffi::TensorView gammas,
    tvm::ffi::TensorView offsets) {
  auto N = host::SymbolicSize{"routed_rows"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();
  host::TensorMatcher({N}).with_dtype<int32_t>().with_device(device).verify(ids).verify(src2dst).verify(dst2src);
  host::TensorMatcher({N}).with_dtype<fp32_t>().with_device(device).verify(weights).verify(gammas);
  host::TensorMatcher({kNumExperts + 1}).with_dtype<int32_t>().with_device(device).verify(offsets);

  int64_t const n = N.unwrap();
  CHECK_HOST(n >= 0 && n <= 512) << "route_metadata_lowm: requires 0 <= n <= 512, got " << n;
  if (n == 0) return;

  if (n <= 256) {
    host::LaunchKernel(1, 256, device.unwrap())
        .enable_pdl(true)(
            route_metadata_lowm_kernel<256, 8>,
            static_cast<int32_t const*>(ids.data_ptr()),
            static_cast<float const*>(weights.data_ptr()),
            static_cast<int32_t*>(src2dst.data_ptr()),
            static_cast<int32_t*>(dst2src.data_ptr()),
            static_cast<float*>(gammas.data_ptr()),
            static_cast<int32_t*>(offsets.data_ptr()),
            static_cast<int32_t>(n));
  } else {
    host::LaunchKernel(1, 512, device.unwrap())
        .enable_pdl(true)(
            route_metadata_lowm_kernel<512, 16>,
            static_cast<int32_t const*>(ids.data_ptr()),
            static_cast<float const*>(weights.data_ptr()),
            static_cast<int32_t*>(src2dst.data_ptr()),
            static_cast<int32_t*>(dst2src.data_ptr()),
            static_cast<float*>(gammas.data_ptr()),
            static_cast<int32_t*>(offsets.data_ptr()),
            static_cast<int32_t>(n));
  }
}

}  // namespace sglang::inkling2_moe_route_sm120
