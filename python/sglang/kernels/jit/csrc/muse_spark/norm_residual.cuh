/**
 * \brief Fused post-norm + gated residual + pre-norm for the MuseSpark v1.2 layer boundary.
 *
 * branch: [M, H] bf16, the sublayer output.
 * stream: [M, H] fp32, the residual stream.
 * output: [M, H] bf16, pre-norm of the new stream (next sublayer input).
 * new_stream: [M, H] fp32, gate_a * stream + gate_b * post_norm(branch).
 * router_out: [M, H] fp32, fp32 copy of output; null when unused.
 * post_weight / pre_weight: [H] bf16, null when absent.
 * alpha / beta: [H] or [1] fp32 residual gates, null when ungated.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cooperative_groups.h>
#include <cstdint>

namespace sglang {

struct NormResidualParams {
  const void* __restrict__ branch;
  const void* __restrict__ stream;
  void* __restrict__ output;
  void* __restrict__ new_stream;
  void* __restrict__ router_out;
  const void* __restrict__ post_weight;
  const void* __restrict__ pre_weight;
  const void* __restrict__ alpha;
  const void* __restrict__ beta;
  float post_eps;
  float pre_eps;
  int num_rows;
  bool gate_channel_wise;
  bool has_pre_weight;
  bool pre_norm_bf16;
};

namespace device::norm_residual {

// 16B per vectorized bf16 access; the fp32 tensors are covered by 256-bit
// accesses over the same element range (two 128-bit halves pre-Blackwell) so
// both dtypes share one thread map. All row data lives in registers as
// fp32x2_t pairs, so conversions and the bf16 round-trip use the paired
// cvt forms instead of per-element scalars.
inline constexpr int kVec = 8;
inline constexpr int kFVec = kMaxVecBytes >= 32 ? kVec : kVec / 2;
inline constexpr int kFParts = kVec / kFVec;
inline constexpr int kPairs = kVec / 2;
inline constexpr int kFPairs = kFVec / 2;

// Triton's `.to(tl.bfloat16).to(tl.float32)` round-trip; load-bearing for parity.
SGL_DEVICE fp32x2_t round_bf16(fp32x2_t v) {
  return cast<fp32x2_t>(cast<bf16x2_t>(v));
}

}  // namespace device::norm_residual

// One CTA per row; the row stays in registers across both RMS reductions.
// `kPrefetchWeights` additionally issues the gate / pre-norm weight loads before
// the first reduction, so a small-M launch costs one DRAM round trip instead of
// three; at large M the register cost of that outweighs the latency it hides.
template <
    int64_t kHidden,
    int kNumThreads,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kHasRouter,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    muse_spark_norm_residual_kernel(const __grid_constant__ NormResidualParams params) {
  using namespace device;
  using namespace device::norm_residual;

  constexpr int kPacks = kHidden / (kNumThreads * kVec);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr int kPreSlots = kPrefetchWeights ? kPacks : 1;
  static_assert(kHidden % (kNumThreads * kVec) == 0);

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  const int64_t base = static_cast<int64_t>(blockIdx.x) * kHidden;
  const auto branch = static_cast<const bf16_t*>(params.branch) + base;
  const auto stream = static_cast<const fp32_t*>(params.stream) + base;
  const auto out = static_cast<bf16_t*>(params.output) + base;
  const auto new_stream = static_cast<fp32_t*>(params.new_stream) + base;
  const auto router = static_cast<fp32_t*>(params.router_out) + (kHasRouter ? base : 0);
  const auto post_weight = static_cast<const bf16_t*>(params.post_weight);
  const auto pre_weight = static_cast<const bf16_t*>(params.pre_weight);
  const auto alpha_ptr = static_cast<const fp32_t*>(params.alpha);
  const auto beta_ptr = static_cast<const fp32_t*>(params.beta);

  // [0, kNumWarps) and [kNumWarps, 2*kNumWarps) are the two reduction scratch
  // areas; separate areas keep the second reduction free of a WAR barrier.
  __shared__ float smem[kNumWarps == 1 ? 1 : 2 * kNumWarps];

  const bool channel_wise = kGate && params.gate_channel_wise;
  const bool has_pre_weight = params.has_pre_weight;
  const bool pre_bf16 = params.pre_norm_bf16;

  fp32x2_t vals[kPacks][kPairs];
  fp32x2_t post_w[kPostWeight ? kPacks : 1][kPairs];
  F32Vec stream_v[kPacks][kFParts];
  F32Vec alpha_pf[kPreSlots][kFParts], beta_pf[kPreSlots][kFParts];
  BF16Vec pre_w_pf[kPreSlots];

  PDLWaitPrimary<kUsePDL>();

  float sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    BF16Vec branch_v;
    branch_v.load(branch, idx);
#pragma unroll
    for (int j = 0; j < kFParts; ++j)
      stream_v[p][j].load(stream, idx * kFParts + j);
    if constexpr (kPrefetchWeights) {
      if (has_pre_weight) pre_w_pf[p].load(pre_weight, idx);
      if constexpr (kGate) {
        if (channel_wise) {
#pragma unroll
          for (int j = 0; j < kFParts; ++j) {
            alpha_pf[p][j].load(alpha_ptr, idx * kFParts + j);
            beta_pf[p][j].load(beta_ptr, idx * kFParts + j);
          }
        }
      }
    }
    if constexpr (kPostWeight) {
      BF16Vec weight_v;
      weight_v.load(post_weight, idx);
#pragma unroll
      for (int q = 0; q < kPairs; ++q)
        post_w[p][q] = cast<fp32x2_t>(weight_v[q]);
    }
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      auto v = cast<fp32x2_t>(branch_v[q]);
      if constexpr (kPostNormIsInputScale && kPostWeight) {
        v.x *= post_w[p][q].x;
        v.y *= post_w[p][q].y;
        v = round_bf16(v);
      }
      vals[p][q] = v;
      sum_sq = fmaf(v.x, v.x, sum_sq);
      sum_sq = fmaf(v.y, v.y, sum_sq);
    }
  }

  sum_sq = cta::reduce_sum<kNumThreads>(sum_sq, smem);
  const float post_rstd = math::rsqrt(sum_sq / kHidden + params.post_eps);

  fp32x2_t gate_a = {1.0f, 1.0f};
  fp32x2_t gate_b = {1.0f, 1.0f};
  if constexpr (kGate) {
    if (!channel_wise) {
      gate_a.x = gate_a.y = *alpha_ptr;
      gate_b.x = gate_b.y = *beta_ptr;
    }
  }

  float pre_sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    F32Vec alpha_v[kFParts], beta_v[kFParts];
    if constexpr (kGate) {
      if (channel_wise) {
#pragma unroll
        for (int j = 0; j < kFParts; ++j) {
          if constexpr (kPrefetchWeights) {
            alpha_v[j] = alpha_pf[p][j];
            beta_v[j] = beta_pf[p][j];
          } else {
            alpha_v[j].load(alpha_ptr, idx * kFParts + j);
            beta_v[j].load(beta_ptr, idx * kFParts + j);
          }
        }
      }
    }
    F32Vec new_stream_v[kFParts];
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      const int h = q / kFPairs;
      const int l = q % kFPairs;
      fp32x2_t normed_b = vals[p][q];
      normed_b.x *= post_rstd;
      normed_b.y *= post_rstd;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        normed_b.x *= post_w[p][q].x;
        normed_b.y *= post_w[p][q].y;
      }
      normed_b = round_bf16(normed_b);
      const fp32x2_t s = stream_v[p][h][l];
      fp32x2_t new_s;
      if constexpr (kGate) {
        const fp32x2_t ga = channel_wise ? alpha_v[h][l] : gate_a;
        const fp32x2_t gb = channel_wise ? beta_v[h][l] : gate_b;
        new_s.x = ga.x * s.x + gb.x * normed_b.x;
        new_s.y = ga.y * s.y + gb.y * normed_b.y;
      } else {
        new_s.x = s.x + normed_b.x;
        new_s.y = s.y + normed_b.y;
      }
      new_stream_v[h][l] = new_s;
      const fp32x2_t pre_v = pre_bf16 ? round_bf16(new_s) : new_s;
      vals[p][q] = pre_v;
      pre_sum_sq = fmaf(pre_v.x, pre_v.x, pre_sum_sq);
      pre_sum_sq = fmaf(pre_v.y, pre_v.y, pre_sum_sq);
    }
#pragma unroll
    for (int j = 0; j < kFParts; ++j)
      new_stream_v[j].store(new_stream, idx * kFParts + j);
  }

  pre_sum_sq = cta::reduce_sum<kNumThreads>(pre_sum_sq, kNumWarps == 1 ? smem : smem + kNumWarps);
  const float pre_rstd = math::rsqrt(pre_sum_sq / kHidden + params.pre_eps);

#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    BF16Vec weight_v;
    if (has_pre_weight) {
      if constexpr (kPrefetchWeights) {
        weight_v = pre_w_pf[p];
      } else {
        weight_v.load(pre_weight, idx);
      }
    }
    BF16Vec out_v;
    F32Vec router_v[kFParts];
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      fp32x2_t v = vals[p][q];
      v.x *= pre_rstd;
      v.y *= pre_rstd;
      if (has_pre_weight) {
        const auto wf = cast<fp32x2_t>(weight_v[q]);
        v.x *= wf.x;
        v.y *= wf.y;
      }
      if (pre_bf16) v = round_bf16(v);
      out_v[q] = cast<bf16x2_t>(v);
      if constexpr (kHasRouter) {
        router_v[q / kFPairs][q % kFPairs] = v;
      }
    }
    out_v.store(out, idx);
    if constexpr (kHasRouter) {
#pragma unroll
      for (int j = 0; j < kFParts; ++j)
        router_v[j].store(router, idx * kFParts + j);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

// Multi-CTA path partitions each row and exchanges FP32 partials through DSM.
// Kept separate to leave the single-CTA throughput path unchanged.

namespace device::norm_residual_cluster {

template <int kClusterSize, int kNumWarps>
struct ReductionStorage4 {
  float warp_sums[4][kNumWarps];
  alignas(16) float partials[kClusterSize][4];
  alignas(8) uint64_t barrier;
};

}  // namespace device::norm_residual_cluster

namespace device::ptx {

SGL_DEVICE void publish_cluster_barriers() {
  asm volatile("fence.mbarrier_init.release.cluster;" : :);
}

SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" : : : "memory");
}

SGL_DEVICE void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" : : : "memory");
}

SGL_DEVICE void mbar_wait_parity_acquire_cluster(uint64_t* barrier) {
  const uint32_t address = to_shared(barrier);
  uint32_t ready = 0;
  while (!ready) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
        "p, [%1], 0;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(ready)
        : "r"(address)
        : "memory");
  }
}

SGL_DEVICE void st_async_remote4(
    float* destination, float value0, float value1, float value2, float value3, uint64_t* barrier, int peer) {
  uint32_t remote_destination;
  uint32_t remote_barrier;
  const uint32_t destination_address = to_shared(destination);
  const uint32_t barrier_address = to_shared(barrier);
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_destination) : "r"(destination_address), "r"(peer));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_barrier) : "r"(barrier_address), "r"(peer));
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
      "[%0], {%1, %2, %3, %4}, [%5];"
      :
      : "r"(remote_destination), "f"(value0), "f"(value1), "f"(value2), "f"(value3), "r"(remote_barrier)
      : "memory");
}

}  // namespace device::ptx

namespace device::norm_residual_cluster {

template <int kClusterSize, int kNumWarps>
SGL_DEVICE void local_reduce(
    float& value0, float& value1, float& value2, float& value3, ReductionStorage4<kClusterSize, kNumWarps>& storage) {
  using namespace device;

  const int lane = threadIdx.x % kWarpThreads;
  const int warp_index = threadIdx.x / kWarpThreads;
  value0 = warp::reduce_sum(value0);
  value1 = warp::reduce_sum(value1);
  value2 = warp::reduce_sum(value2);
  value3 = warp::reduce_sum(value3);
  if (lane == 0) {
    storage.warp_sums[0][warp_index] = value0;
    storage.warp_sums[1][warp_index] = value1;
    storage.warp_sums[2][warp_index] = value2;
    storage.warp_sums[3][warp_index] = value3;
  }
  // Reduce four moments together so they share one CTA barrier. Calling the
  // generic scalar cta::reduce_sum helper four times would require four.
  __syncthreads();

  value0 = warp::reduce_sum(lane < kNumWarps ? storage.warp_sums[0][lane] : 0.0f);
  value1 = warp::reduce_sum(lane < kNumWarps ? storage.warp_sums[1][lane] : 0.0f);
  value2 = warp::reduce_sum(lane < kNumWarps ? storage.warp_sums[2][lane] : 0.0f);
  value3 = warp::reduce_sum(lane < kNumWarps ? storage.warp_sums[3][lane] : 0.0f);
}

template <int kClusterSize, int kNumWarps>
SGL_DEVICE void dsmem_reduce(
    float& value0,
    float& value1,
    float& value2,
    float& value3,
    ReductionStorage4<kClusterSize, kNumWarps>& storage,
    int cluster_rank) {
  if (threadIdx.x < kClusterSize && threadIdx.x != cluster_rank) {
    ptx::st_async_remote4(
        &storage.partials[cluster_rank][0], value0, value1, value2, value3, &storage.barrier, threadIdx.x);
  }

  ptx::mbar_wait_parity_acquire_cluster(&storage.barrier);
  const int slot = threadIdx.x & (kClusterSize - 1);
  float total0 = 0.0f;
  float total1 = 0.0f;
  float total2 = 0.0f;
  float total3 = 0.0f;
  if (slot != cluster_rank) {
    const auto remote = reinterpret_cast<const float4*>(storage.partials)[slot];
    total0 = remote.x;
    total1 = remote.y;
    total2 = remote.z;
    total3 = remote.w;
  }
  total0 = warp::reduce_sum<kClusterSize>(total0);
  total1 = warp::reduce_sum<kClusterSize>(total1);
  total2 = warp::reduce_sum<kClusterSize>(total2);
  total3 = warp::reduce_sum<kClusterSize>(total3);
  value0 += total0;
  value1 += total1;
  value2 += total2;
  value3 += total3;
}

}  // namespace device::norm_residual_cluster

// Multiple CTAs per row exchange partial RMS statistics through DSM.
template <
    int64_t kHidden,
    int kNumThreads,
    int kClusterSize,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kHasRouter,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    muse_spark_norm_residual_cluster(const __grid_constant__ NormResidualParams params) {
  using namespace device;
  using namespace device::norm_residual;
  using namespace device::norm_residual_cluster;

  constexpr int kTotalThreads = kNumThreads * kClusterSize;
  constexpr int kPacks = kHidden / (kTotalThreads * kVec);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr int kPreSlots = kPrefetchWeights ? kPacks : 1;
  constexpr float kInvHidden = 1.0f / static_cast<float>(kHidden);
  static_assert(kClusterSize == 2 || kClusterSize == 4 || kClusterSize == 8 || kClusterSize == 16);
  static_assert(kHidden % (kTotalThreads * kVec) == 0);

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  const int cluster_rank = cooperative_groups::this_cluster().block_rank();
  const int row = static_cast<int>(blockIdx.x) / kClusterSize;
  const int64_t base = static_cast<int64_t>(row) * kHidden;
  const auto branch = static_cast<const bf16_t*>(params.branch) + base;
  const auto stream = static_cast<const fp32_t*>(params.stream) + base;
  const auto out = static_cast<bf16_t*>(params.output) + base;
  const auto new_stream = static_cast<fp32_t*>(params.new_stream) + base;
  const auto router = static_cast<fp32_t*>(params.router_out) + (kHasRouter ? base : 0);
  const auto post_weight = static_cast<const bf16_t*>(params.post_weight);
  const auto pre_weight = static_cast<const bf16_t*>(params.pre_weight);
  const auto alpha_ptr = static_cast<const fp32_t*>(params.alpha);
  const auto beta_ptr = static_cast<const fp32_t*>(params.beta);
  const bool channel_wise = kGate && params.gate_channel_wise;
  const bool has_pre_weight = params.has_pre_weight;

  __shared__ ReductionStorage4<kClusterSize, kNumWarps> reduction;
  if (threadIdx.x == 0) {
    ptx::mbar_init(&reduction.barrier, 1);
    ptx::mbar_arrive_expect_tx(&reduction.barrier, (kClusterSize - 1) * sizeof(float4));
    ptx::publish_cluster_barriers();
  }
  ptx::cluster_arrive_relaxed();

  fp32x2_t values[kPacks][kPairs];
  fp32x2_t post_w[kPostWeight ? kPacks : 1][kPairs];
  fp32x2_t residual_terms[kPacks][kPairs];
  fp32x2_t beta_values[kPacks][kPairs];
  BF16Vec pre_w_pf[kPreSlots];

  PDLWaitPrimary<kUsePDL>();

  float branch_sum = 0.0f;
  float residual_sum = 0.0f;
  float cross_sum = 0.0f;
  float scaled_branch_sum = 0.0f;
  fp32x2_t gate_a = {1.0f, 1.0f};
  fp32x2_t gate_b = {1.0f, 1.0f};
  if constexpr (kGate) {
    if (!channel_wise) {
      gate_a.x = gate_a.y = *alpha_ptr;
      gate_b.x = gate_b.y = *beta_ptr;
    }
  }
#pragma unroll
  for (int pack = 0; pack < kPacks; ++pack) {
    const int index = cluster_rank * kNumThreads + threadIdx.x + pack * kTotalThreads;
    BF16Vec branch_values;
    BF16Vec post_weights;
    F32Vec stream_values[kFParts];
    F32Vec alpha_values[kFParts];
    F32Vec beta_vector[kFParts];
    branch_values.load(branch, index);
    if constexpr (kPostWeight) post_weights.load(post_weight, index);
    if constexpr (kPrefetchWeights) {
      if (has_pre_weight) pre_w_pf[pack].load(pre_weight, index);
    }
#pragma unroll
    for (int part = 0; part < kFParts; ++part) {
      const int float_index = index * kFParts + part;
      stream_values[part].load(stream, float_index);
      if constexpr (kGate) {
        if (channel_wise) {
          alpha_values[part].load(alpha_ptr, float_index);
          beta_vector[part].load(beta_ptr, float_index);
        }
      }
    }

#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      const int part = pair / kFPairs;
      const int lane = pair % kFPairs;
      auto value = cast<fp32x2_t>(branch_values[pair]);
      if constexpr (kPostWeight) {
        const auto weight = cast<fp32x2_t>(post_weights[pair]);
        post_w[pack][pair] = weight;
        if constexpr (kPostNormIsInputScale) {
          value.x *= weight.x;
          value.y *= weight.y;
          value = round_bf16(value);
        }
      }
      auto moment_value = value;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        const auto weight = post_w[pack][pair];
        moment_value.x *= weight.x;
        moment_value.y *= weight.y;
      }
      const auto stream_value = stream_values[part][lane];
      auto alpha_value = gate_a;
      auto beta_value = gate_b;
      if constexpr (kGate) {
        if (channel_wise) {
          alpha_value = alpha_values[part][lane];
          beta_value = beta_vector[part][lane];
        }
      }
      const fp32x2_t residual = {
          alpha_value.x * stream_value.x,
          alpha_value.y * stream_value.y,
      };
      const fp32x2_t scaled_branch = {
          beta_value.x * moment_value.x,
          beta_value.y * moment_value.y,
      };
      values[pack][pair] = value;
      residual_terms[pack][pair] = residual;
      beta_values[pack][pair] = beta_value;
      branch_sum = fmaf(value.x, value.x, branch_sum);
      branch_sum = fmaf(value.y, value.y, branch_sum);
      residual_sum = fmaf(residual.x, residual.x, residual_sum);
      residual_sum = fmaf(residual.y, residual.y, residual_sum);
      cross_sum = fmaf(residual.x, scaled_branch.x, cross_sum);
      cross_sum = fmaf(residual.y, scaled_branch.y, cross_sum);
      scaled_branch_sum = fmaf(scaled_branch.x, scaled_branch.x, scaled_branch_sum);
      scaled_branch_sum = fmaf(scaled_branch.y, scaled_branch.y, scaled_branch_sum);
    }
  }
  // ---- PDL: Trigger next kernel here to increase overlap
  PDLTriggerSecondary<kUsePDL>();

  local_reduce(branch_sum, residual_sum, cross_sum, scaled_branch_sum, reduction);
  ptx::cluster_wait();
  dsmem_reduce(branch_sum, residual_sum, cross_sum, scaled_branch_sum, reduction, cluster_rank);
  const float post_rstd = math::rsqrt(branch_sum * kInvHidden + params.post_eps);
  const float pre_sum = fmaf(post_rstd, fmaf(post_rstd, scaled_branch_sum, 2.0f * cross_sum), residual_sum);
  const float pre_rstd = math::rsqrt(fmaxf(pre_sum, 0.0f) * kInvHidden + params.pre_eps);

#pragma unroll
  for (int pack = 0; pack < kPacks; ++pack) {
    const int index = cluster_rank * kNumThreads + threadIdx.x + pack * kTotalThreads;
    BF16Vec output_values;
    BF16Vec weight_v;
    F32Vec new_stream_values[kFParts];
    F32Vec router_values[kFParts];
    if (has_pre_weight) {
      if constexpr (kPrefetchWeights)
        weight_v = pre_w_pf[pack];
      else
        weight_v.load(pre_weight, index);
    }
#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      const int part = pair / kFPairs;
      const int lane = pair % kFPairs;
      auto normed = values[pack][pair];
      normed.x *= post_rstd;
      normed.y *= post_rstd;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        const auto weight = post_w[pack][pair];
        normed.x *= weight.x;
        normed.y *= weight.y;
      }
      normed = round_bf16(normed);
      const auto beta_value = beta_values[pack][pair];
      const auto residual = residual_terms[pack][pair];
      const fp32x2_t updated = {
          fmaf(beta_value.x, normed.x, residual.x),
          fmaf(beta_value.y, normed.y, residual.y),
      };
      new_stream_values[part][lane] = updated;
      fp32x2_t normalized = {updated.x * pre_rstd, updated.y * pre_rstd};
      if (has_pre_weight) {
        const auto weight = cast<fp32x2_t>(weight_v[pair]);
        normalized.x *= weight.x;
        normalized.y *= weight.y;
      }
      output_values[pair] = cast<bf16x2_t>(normalized);
      if constexpr (kHasRouter) router_values[part][lane] = normalized;
    }
    output_values.store(out, index);
#pragma unroll
    for (int part = 0; part < kFParts; ++part) {
      const int float_index = index * kFParts + part;
      new_stream_values[part].store(new_stream, float_index);
      if constexpr (kHasRouter) router_values[part].store(router, float_index);
    }
  }
}

template <
    int64_t kHidden,
    int kNumThreads,
    bool kUsePDL,
    bool kPrefetchWeights,
    int kClusterSize,
    bool kPostNormIsInputScale,
    bool kHasRouter>
struct MuseSparkNormResidual {
  static_assert(kClusterSize == 1 || kClusterSize == 2 || kClusterSize == 4 || kClusterSize == 8 || kClusterSize == 16);

  static void
  run(const tvm::ffi::TensorView branch,
      const tvm::ffi::TensorView stream,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView new_stream,
      const tvm::ffi::Optional<tvm::ffi::TensorView> router_out,
      const tvm::ffi::Optional<tvm::ffi::TensorView> post_weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> pre_weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> alpha,
      const tvm::ffi::Optional<tvm::ffi::TensorView> beta,
      double post_eps,
      double pre_eps,
      bool pre_norm_bf16) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto G = SymbolicSize{"gate_dim"};
    auto device_ = SymbolicDevice{};
    H.set_value(kHidden);
    device_.set_options<kDLCUDA>();

    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(branch);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(stream);
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(output);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(new_stream);
    RuntimeCheck(router_out.has_value() == kHasRouter, "router_out presence must match the kHasRouter instantiation");
    if constexpr (kHasRouter) {
      TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(router_out.value());
    }
    if (post_weight.has_value()) {
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(post_weight.value());
    }
    if (pre_weight.has_value()) {
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(pre_weight.value());
    }

    const bool gate = alpha.has_value();
    RuntimeCheck(gate == beta.has_value(), "alpha and beta must both be present or both absent");
    bool gate_channel_wise = false;
    if (gate) {
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(alpha.value());
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(beta.value());
      const auto n = G.unwrap();
      RuntimeCheck(n == 1 || n == kHidden, "alpha/beta must have 1 or hidden_size elements");
      gate_channel_wise = (n == kHidden);
    }

    const auto num_rows = static_cast<int>(M.unwrap());
    if (num_rows == 0) return;

    const auto params = NormResidualParams{
        .branch = branch.data_ptr(),
        .stream = stream.data_ptr(),
        .output = output.data_ptr(),
        .new_stream = new_stream.data_ptr(),
        .router_out = kHasRouter ? router_out.value().data_ptr() : nullptr,
        .post_weight = post_weight.has_value() ? post_weight.value().data_ptr() : nullptr,
        .pre_weight = pre_weight.has_value() ? pre_weight.value().data_ptr() : nullptr,
        .alpha = gate ? alpha.value().data_ptr() : nullptr,
        .beta = gate ? beta.value().data_ptr() : nullptr,
        .post_eps = static_cast<float>(post_eps),
        .pre_eps = static_cast<float>(pre_eps),
        .num_rows = num_rows,
        .gate_channel_wise = gate_channel_wise,
        .has_pre_weight = pre_weight.has_value(),
        .pre_norm_bf16 = pre_norm_bf16,
    };
    const auto device = device_.unwrap();

    if constexpr (kClusterSize > 1) RuntimeCheck(!pre_norm_bf16, "Cluster path requires FP32 pre-norm input");

    const auto launch = [&]<bool kHasPostWeight, bool kHasGate>() {
      if constexpr (kClusterSize == 1) {
        LaunchKernel(num_rows, kNumThreads, device)
            .config({.use_pdl = kUsePDL})(
                muse_spark_norm_residual_kernel<
                    kHidden,
                    kNumThreads,
                    kUsePDL,
                    kPrefetchWeights,
                    kPostNormIsInputScale,
                    kHasRouter,
                    kHasPostWeight,
                    kHasGate>,
                params);
      } else {
        constexpr auto kernel = muse_spark_norm_residual_cluster<
            kHidden,
            kNumThreads,
            kClusterSize,
            kUsePDL,
            kPrefetchWeights,
            kPostNormIsInputScale,
            kHasRouter,
            kHasPostWeight,
            kHasGate>;
        [[maybe_unused]] static const bool cluster_configured = [kernel] {
          if constexpr (kClusterSize > 8) {
            RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
          }
          cudaLaunchConfig_t occupancy_config{};
          occupancy_config.gridDim = dim3(kClusterSize);
          occupancy_config.blockDim = dim3(kNumThreads);
          int max_cluster_size = 0;
          RuntimeDeviceCheck(cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, kernel, &occupancy_config));
          RuntimeCheck(
              max_cluster_size >= kClusterSize,
              "Requested cluster size ",
              kClusterSize,
              " exceeds this kernel/device limit ",
              max_cluster_size);
          return true;
        }();
        LaunchKernel(num_rows * kClusterSize, kNumThreads, device)
            .config({
                .use_pdl = kUsePDL,
                .cluster_dim = dim3(kClusterSize, 1, 1),
            })(kernel, params);
      }
    };

    if (post_weight.has_value()) {
      if (gate)
        launch.template operator()<true, true>();
      else
        launch.template operator()<true, false>();
    } else {
      if (gate)
        launch.template operator()<false, true>();
      else
        launch.template operator()<false, false>();
    }
  }
};

}  // namespace sglang
