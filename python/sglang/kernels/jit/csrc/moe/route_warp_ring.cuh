// The ring pick needs the whole batch, so the batch is one CTA or one cluster of <= 8 CTAs.

#pragma once

#include "muse_ring_pick.h"
#include "route_warp.cuh"
#include <cooperative_groups.h>

namespace sglang {

namespace moe::route_warp_ring {

namespace rw = ::sglang::moe::route_warp;

struct Params {
  const float* __restrict__ logits;
  const float* __restrict__ bias;
  float* __restrict__ out_w;
  int32_t* __restrict__ out_i;
  int32_t* __restrict__ out_packed;
  const int32_t* __restrict__ num_non_padded;
  int64_t logits_stride;
  uint32_t num_tokens;
  float scale;
  float renorm_eps;
  float logit_scale;
};

SGL_DEVICE void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release;" : : : "memory");
}

SGL_DEVICE void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire;" : : : "memory");
}

template <uint32_t E, uint32_t K, uint32_t kRanks, uint32_t kChunksPerHome, uint32_t kMaxWarps, bool kUsePDL>
__global__ __launch_bounds__(kMaxWarps * 32) void route_warp_ring_kernel(const __grid_constant__ Params p) {
  using namespace device;
  using Pick = muse_ring::RingPick<static_cast<int>(kRanks), static_cast<int>(kChunksPerHome)>;
  constexpr uint32_t L = E / 32;
  constexpr uint32_t M = std::min(L, K);
  constexpr uint32_t V = rw::kVecSize<float, L>;
  constexpr uint32_t kPitch = rw::smem_pitch(L);
  static_assert(E % 32 == 0 && L <= 16 && K >= 1 && K <= 32);
  static_assert(Pick::kLogical == static_cast<int>(E), "ring R * L must equal the routed experts");

#if SGL_ARCH_HOPPER_OR_GREATER
  __shared__ float s_val[kMaxWarps][32 * kPitch];
  __shared__ uint32_t s_key[kMaxWarps][32 * kPitch];
  __shared__ int32_t s_flags[E];

  const uint32_t lane_id = threadIdx.x % 32;
  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t token = blockIdx.x * (blockDim.x / 32) + warp_id;
  const uint32_t base = lane_id * L;
  const bool clustered = gridDim.x > 1;  // the grid is exactly one cluster

  for (uint32_t i = threadIdx.x; i < E; i += blockDim.x) {
    s_flags[i] = 0;
  }
  // Publishes the zeroing before any peer's remote store (waited on below).
  if (clustered) cluster_arrive_release();

  PDLWaitPrimary<kUsePDL>();

  const bool in_range = token < p.num_tokens;
  const bool real = in_range && (p.num_non_padded == nullptr || token < static_cast<uint32_t>(*p.num_non_padded));

  uint32_t expert = 0;
  float w = 0.0f;
  if (real) {  // warp-uniform
    // route_warp_kernel's kMuseSpark path; logical ids and weights must stay bit-identical, keep in sync.
    const auto row = p.logits + static_cast<int64_t>(token) * p.logits_stride + base;
    AlignedVector<float, V> xv[L / V];
#pragma unroll
    for (uint32_t i = 0; i < L / V; ++i) {
      xv[i].load(row, i);
    }
    AlignedVector<float, V> bv[L / V];
#pragma unroll
    for (uint32_t i = 0; i < L / V; ++i) {
      bv[i].load(p.bias + base, i);
    }

    float val[L];
    float biased[L];
    uint32_t key[L];
#pragma unroll
    for (uint32_t j = 0; j < L; ++j) {
      auto x = xv[j / V][j % V];
      x *= p.logit_scale;
      val[j] = activation::sigmoid(x);
      biased[j] = val[j] + bv[j / V][j % V];
      biased[j] = (biased[j] == biased[j]) ? biased[j] + 0.0f : -1e30f;
      key[j] = rw::order_key(biased[j]);
      s_val[warp_id][lane_id * kPitch + j] = val[j];
    }

#pragma unroll
    for (uint32_t j = 0; j < L; ++j) {
      s_key[warp_id][lane_id * kPitch + j] = key[j];
    }
    merge_sort(key);
    uint32_t my_key = 0, my_lane = 0, my_run = 0;
    uint32_t prev_key = 0, prev_lane = 32, run = 0;
#pragma unroll
    for (uint32_t k = 0; k < K; ++k) {
      const auto m = warp::reduce_max(key[0]);
      const auto winner = warp::reduce_min(key[0] == m ? lane_id : 32u);
      const bool pop = lane_id == winner;
#pragma unroll
      for (uint32_t i = 0; i + 1 < M; ++i) {
        key[i] = pop ? key[i + 1] : key[i];
      }
      key[M - 1] = pop ? rw::kKeyTaken : key[M - 1];
      run = (m == prev_key && winner == prev_lane) ? run + 1 : 0;
      prev_key = m;
      prev_lane = winner;
      my_key = (lane_id == k) ? m : my_key;
      my_lane = (lane_id == k) ? winner : my_lane;
      my_run = (lane_id == k) ? run : my_run;
    }

    __syncwarp();
    if (lane_id < K) {
      uint32_t hits = 0;
#pragma unroll
      for (uint32_t j = 0; j < L; ++j) {
        hits |= (s_key[warp_id][my_lane * kPitch + j] == my_key ? 1u : 0u) << j;
      }
      for (uint32_t r = 0; r < my_run; ++r) {
        hits &= hits - 1;
      }
      const uint32_t j = __ffs(hits) - 1;
      expert = my_lane * L + j;
      w = s_val[warp_id][my_lane * kPitch + j];
    }

    float sum = __shfl_sync(0xffffffffu, w, 0);
#pragma unroll
    for (uint32_t k = 1; k < K; ++k) {
      sum += __shfl_sync(0xffffffffu, w, k);
    }
    w *= rw::ptx::div_full(p.scale, sum + p.renorm_eps);

    if (lane_id < K) {
      p.out_w[static_cast<int64_t>(token) * K + lane_id] = w;
      p.out_i[static_cast<int64_t>(token) * K + lane_id] = static_cast<int32_t>(expert);
    }
  } else if (in_range && lane_id < K) {
    p.out_i[static_cast<int64_t>(token) * K + lane_id] = -1;
    p.out_packed[static_cast<int64_t>(token) * K + lane_id] = static_cast<int32_t>(0xffff0000u);
  }

  // Same-value stores, so no atomics.
  const bool flag = real && lane_id < K;
  if (clustered) {
    cluster_wait_acquire();
    if (flag) {
      const auto cluster = cooperative_groups::this_cluster();
      for (uint32_t dst = 0; dst < gridDim.x; ++dst) {
        *cluster.map_shared_rank(&s_flags[expert], dst) = 1;
      }
    }
    cluster_arrive_release();
    cluster_wait_acquire();
  } else {
    __syncthreads();
    if (flag) s_flags[expert] = 1;
    __syncthreads();
  }
  if (!real) return;  // no peer reads this CTA's smem past the last barrier

  // Every warp computes the same result from the same flags, so no barrier.
  uint32_t m[Pick::kChunks];
#pragma unroll
  for (int c = 0; c < Pick::kChunks; ++c) {
    m[c] = __ballot_sync(0xffffffffu, s_flags[c * 32 + lane_id] != 0);
  }
  int k[kRanks];
  Pick::counts(m, k);
  int total = 0;
#pragma unroll
  for (uint32_t h = 0; h < kRanks; ++h) {
    total += k[h];
  }
  int best = Pick::kNoKey;
#pragma unroll
  for (int i = 0; i <= static_cast<int>(kChunksPerHome); ++i) {
    const int t = static_cast<int>(lane_id) + 32 * i;
    if (t <= Pick::kHome) best = muse_ring::imin(best, Pick::candidate_key(k, total, t));
  }
  best = __reduce_min_sync(0xffffffffu, best);  // T = kHome is always feasible
  const int t_best = best / Pick::kKeyBase;
  int y[kRanks];
  Pick::flows(k, t_best, best - t_best * Pick::kKeyBase, y);

  if (lane_id < K) {
    const auto phys = static_cast<uint32_t>(Pick::physical(m, k, y, static_cast<int>(expert)));
    const auto w_bits = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(w)));
    p.out_packed[static_cast<int64_t>(token) * K + lane_id] = static_cast<int32_t>((phys << 16) | w_bits);
  }
#else
  __trap();  // the host side refuses pre-sm_90 builds
#endif
}

template <uint32_t E, uint32_t K, uint32_t kRanks, uint32_t kChunksPerHome, uint32_t kMaxWarps, bool kUsePDL>
struct RouteWarpRingKernel {
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      const tvm::ffi::TensorView out_packed,
      const tvm::ffi::Optional<tvm::ffi::TensorView> num_non_padded,
      const float scale,
      const float renorm_eps,
      const float logit_scale,
      const int64_t warps_per_cta,
      const int64_t num_ctas) {
    using namespace host;
    constexpr uint32_t L = E / 32;
    constexpr int64_t kMaxCluster = 8;  // portable cluster size
    auto N_ = SymbolicSize{"num_tokens"};
    auto device_sym = SymbolicDevice{};
    device_sym.set_options<kDLCUDA>();
    TensorMatcher({N_, E})  //
        .with_strides({-1, 1})
        .with_dtype<float>()
        .with_device(device_sym)
        .ensure_alignment(rw::kVecSize<float, L> * sizeof(float))
        .verify(logits);
    TensorMatcher({E})  //
        .with_dtype<float>()
        .with_device(device_sym)
        .ensure_alignment(rw::kVecSize<float, L> * sizeof(float))
        .verify(bias);
    TensorMatcher({N_, K})  //
        .with_strides({static_cast<int64_t>(K), 1})
        .with_dtype<float>()
        .with_device(device_sym)
        .verify(out_w);
    TensorMatcher({N_, K})  //
        .with_strides({static_cast<int64_t>(K), 1})
        .with_dtype<int32_t>()
        .with_device(device_sym)
        .verify(out_i);
    TensorMatcher({N_, K})  //
        .with_strides({static_cast<int64_t>(K), 1})
        .with_dtype<int32_t>()
        .with_device(device_sym)
        .verify(out_packed);
    CHECK_HOST(SGL_CUDA_ARCH >= 900) << "route_warp_ring: thread-block clusters need sm_90+";

    auto p = Params{};
    if (num_non_padded.has_value()) {
      const auto& n = num_non_padded.value();
      CHECK_HOST(is_type<int32_t>(n.dtype()) && n.numel() == 1 && n.device().device_type == kDLCUDA);
      p.num_non_padded = static_cast<const int32_t*>(n.data_ptr());
    }
    p.logits = static_cast<const float*>(logits.data_ptr());
    p.bias = static_cast<const float*>(bias.data_ptr());
    p.out_w = static_cast<float*>(out_w.data_ptr());
    p.out_i = static_cast<int32_t*>(out_i.data_ptr());
    p.out_packed = static_cast<int32_t*>(out_packed.data_ptr());
    p.logits_stride = logits.stride(0);
    p.num_tokens = static_cast<uint32_t>(logits.size(0));
    p.scale = scale;
    p.renorm_eps = renorm_eps;
    p.logit_scale = logit_scale;
    if (p.num_tokens == 0) return;
    const auto T = static_cast<int64_t>(p.num_tokens);
    CHECK_HOST(warps_per_cta >= 1 && warps_per_cta <= kMaxWarps && num_ctas >= 1 && num_ctas <= kMaxCluster)
        << "route_warp_ring: launch shape " << num_ctas << "x" << warps_per_cta;
    // Every CTA owns at least one in-range row (launch_shape guarantees it).
    CHECK_HOST(T <= warps_per_cta * num_ctas && T > warps_per_cta * (num_ctas - 1))
        << "route_warp_ring: " << num_ctas << "x" << warps_per_cta << " warps do not cover " << T << " tokens";
    const auto device = device_sym.unwrap();
    const auto kernel = route_warp_ring_kernel<E, K, kRanks, kChunksPerHome, kMaxWarps, kUsePDL>;
    const auto ctas = static_cast<uint32_t>(num_ctas);
    LaunchKernel launch(dim3(ctas), dim3(static_cast<uint32_t>(warps_per_cta) * 32), device);
    launch.enable_pdl(kUsePDL);
    if (ctas > 1) launch.enable_cluster(dim3(ctas, 1, 1));
    launch(kernel, p);
  }
};

}  // namespace moe::route_warp_ring

}  // namespace sglang
