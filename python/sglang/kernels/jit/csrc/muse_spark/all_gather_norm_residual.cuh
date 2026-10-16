// Fused bf16 all-gather + post-norm + gated residual + pre-norm (MuseSpark v1.2 FFN boundary).
#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "norm_residual.cuh"
#include <algorithm>
#include <cstdint>

namespace sglang {

namespace muse_ag_norm {

using device::distributed::Counter;

struct Params {
  uint8_t* ws_local;      // LOCAL VA of the push workspace base (poll + reset)
  uint8_t* ws_mc;         // multicast VA of the push workspace base (push)
  Counter* counter;       // per-block phase counters
  uint32_t num_counters;  // PushPlane::num_blocks
  uint32_t half_bytes;    // world_size * slot_bytes
  uint32_t rank;
  uint32_t num_rows;
  const bf16_t* local;  // [M, H / world]
  NormResidualParams norm;
};

inline constexpr uint32_t kVec = 8;  // bf16 per 16B

SGL_DEVICE fp32x2_t mul_pair(fp32x2_t a, fp32x2_t b) {
  return {a.x * b.x, a.y * b.y};
}

SGL_DEVICE fp32x2_t fma_pair(fp32x2_t a, fp32x2_t b, fp32x2_t c) {
  return {fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y)};
}

// Reads this block's phase and advances the counters no block of this grid owns.
template <int kNumThreads>
SGL_DEVICE uint32_t begin_phase(const Params& params) {
  const uint32_t phase = params.counter[blockIdx.x].get() & 1;
  if (blockIdx.x == 0) {
    for (uint32_t i = gridDim.x + threadIdx.x; i < params.num_counters; i += kNumThreads) {
      params.counter[i].set(phase ^ 1);
    }
  }
  return phase;
}

template <uint32_t kNLocal, int kNumThreads>
SGL_DEVICE void push_local_slice(const Params& params, uint32_t half_off) {
  const uint32_t num_local = params.num_rows * kNLocal;
  const uint32_t worker = blockIdx.x * kNumThreads + threadIdx.x;
  const uint32_t stride = gridDim.x * kNumThreads * kVec;
  for (uint32_t elem = worker * kVec; elem < num_local; elem += stride) {
    uint4 raw = *reinterpret_cast<const uint4*>(params.local + elem);
    uint32_t* w = reinterpret_cast<uint32_t*>(&raw);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      if ((w[j] & 0xffffu) == 0) w[j] |= 0x8000u;
      if ((w[j] >> 16) == 0) w[j] |= 0x80000000u;
    }
    const uint32_t byte_off = half_off + (params.rank * num_local + elem) * sizeof(bf16_t);
    device::ptx::st_multimem_16B(raw, params.ws_mc + byte_off, 0);
  }
}

SGL_DEVICE bool vector_pending(uint4 raw) {
  return (raw.x & 0xffffu) == 0 || (raw.x >> 16) == 0 || (raw.y & 0xffffu) == 0 || (raw.y >> 16) == 0 ||
         (raw.z & 0xffffu) == 0 || (raw.z >> 16) == 0 || (raw.w & 0xffffu) == 0 || (raw.w >> 16) == 0;
}

// Spins until every rank's copy of the vector at (row, col) arrived, then clears the slot.
template <uint32_t kNLocal>
SGL_DEVICE uint4 take_gathered(bf16_t* gathered, uint32_t num_rows, uint32_t row, uint32_t col) {
  bf16_t* src = gathered + ((col / kNLocal) * num_rows + row) * kNLocal + col % kNLocal;
  uint4 raw;
  bool pending;
  do {
    device::ptx::ld_relaxed_16B(raw, src, 0);
    pending = vector_pending(raw);
  } while (pending);
  device::AlignedVector<uint32_t, 4> zero;
  zero.fill(0);
  zero.store(src);
  return raw;
}

template <int kPacks, int kVectorStride, uint32_t kNLocal>
SGL_DEVICE uint4
receive_pack(bf16_t* gathered, uint32_t num_rows, uint32_t row, uint32_t col, int pack, uint4 (&cache)[2]) {
  if (pack % 2 == 0) {
    if (pack + 1 < kPacks) {
      const uint32_t next_col = col + kVectorStride * kVec;
      bf16_t* src0 = gathered + ((col / kNLocal) * num_rows + row) * kNLocal + col % kNLocal;
      bf16_t* src1 = gathered + ((next_col / kNLocal) * num_rows + row) * kNLocal + next_col % kNLocal;
      bool pending;
      do {
        device::ptx::ld_relaxed_16B(cache[0], src0, 0);
        device::ptx::ld_relaxed_16B(cache[1], src1, 0);
        pending = vector_pending(cache[0]) | vector_pending(cache[1]);
      } while (pending);
      device::AlignedVector<uint32_t, 4> zero;
      zero.fill(0);
      zero.store(src0);
      zero.store(src1);
    } else {
      cache[0] = take_gathered<kNLocal>(gathered, num_rows, row, col);
    }
  }
  return cache[pack % 2];
}

}  // namespace muse_ag_norm

// C > 1: one cluster per row; muse_spark_norm_residual_cluster's arithmetic.
template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    int kClusterSize,
    bool kUsePDL,
    bool kPostNormIsInputScale,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    muse_spark_all_gather_norm_residual_cluster_kernel(const __grid_constant__ muse_ag_norm::Params params) {
  using namespace device;
  using namespace device::norm_residual;
  using namespace device::norm_residual_cluster;

  constexpr int kTotalThreads = kNumThreads * kClusterSize;
  constexpr int kPacks = kHidden / (kTotalThreads * kVec);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr float kInvHidden = 1.0f / static_cast<float>(kHidden);
  constexpr uint32_t kNLocal = static_cast<uint32_t>(kHidden) / kWorld;
  static_assert(kClusterSize == 2 || kClusterSize == 4 || kClusterSize == 8 || kClusterSize == 16);
  static_assert(kHidden % (kTotalThreads * kVec) == 0);
  static_assert(kNLocal % kVec == 0, "a vector must never cross a rank block");

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  const auto& norm = params.norm;
  const int cluster_rank = cooperative_groups::this_cluster().block_rank();
  const int row = static_cast<int>(blockIdx.x) / kClusterSize;
  const int64_t base = static_cast<int64_t>(row) * kHidden;
  const auto stream = static_cast<const fp32_t*>(norm.stream) + base;
  const auto out = static_cast<bf16_t*>(norm.output) + base;
  const auto new_stream = static_cast<fp32_t*>(norm.new_stream) + base;
  const auto post_weight = static_cast<const bf16_t*>(norm.post_weight);
  const auto pre_weight = static_cast<const bf16_t*>(norm.pre_weight);
  const auto alpha_ptr = static_cast<const fp32_t*>(norm.alpha);
  const auto beta_ptr = static_cast<const fp32_t*>(norm.beta);
  const bool channel_wise = kGate && norm.gate_channel_wise;
  const bool has_pre_weight = norm.has_pre_weight;

  __shared__ ReductionStorage4<kClusterSize, kNumWarps> reduction;
  if (threadIdx.x == 0) {
    ptx::mbar_init(&reduction.barrier, 1);
    ptx::mbar_arrive_expect_tx(&reduction.barrier, (kClusterSize - 1) * sizeof(float4));
    ptx::publish_cluster_barriers();
  }
  ptx::cluster_arrive_relaxed();

  PDLWaitPrimary<kUsePDL>();
  const uint32_t phase = muse_ag_norm::begin_phase<kNumThreads>(params);
  const uint32_t half_off = phase * params.half_bytes;
  muse_ag_norm::push_local_slice<kNLocal, kNumThreads>(params, half_off);
  PDLTriggerSecondary<kUsePDL>();

  const auto gathered = reinterpret_cast<bf16_t*>(params.ws_local + half_off);
  fp32x2_t values[kPacks][kPairs];
  fp32x2_t post_w[kPostWeight ? kPacks : 1][kPairs];
  fp32x2_t residual_terms[kPacks][kPairs];
  fp32x2_t beta_values[kPacks][kPairs];
  BF16Vec pre_w_pf[kPacks];

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
    BF16Vec post_weights;
    F32Vec stream_values[kFParts];
    F32Vec alpha_values[kFParts];
    F32Vec beta_vector[kFParts];
    // Local operands first so their latency overlaps the wait for peer data.
    if constexpr (kPostWeight) post_weights.load(post_weight, index);
    if (has_pre_weight) pre_w_pf[pack].load(pre_weight, index);
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
    const uint4 raw = muse_ag_norm::take_gathered<kNLocal>(
        gathered, params.num_rows, static_cast<uint32_t>(row), static_cast<uint32_t>(index) * kVec);
    const auto& branch_values = *reinterpret_cast<const BF16Vec*>(&raw);

#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      const int part = pair / kFPairs;
      const int lane = pair % kFPairs;
      auto value = cast<fp32x2_t>(branch_values[pair]);
      if constexpr (kPostWeight) {
        const auto weight = cast<fp32x2_t>(post_weights[pair]);
        post_w[pack][pair] = weight;
        if constexpr (kPostNormIsInputScale) {
          value = muse_ag_norm::mul_pair(value, weight);
          value = round_bf16(value);
        }
      }
      auto moment_value = value;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        const auto weight = post_w[pack][pair];
        moment_value = muse_ag_norm::mul_pair(moment_value, weight);
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
      const fp32x2_t residual = muse_ag_norm::mul_pair(alpha_value, stream_value);
      const fp32x2_t scaled_branch = muse_ag_norm::mul_pair(beta_value, moment_value);
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

  local_reduce(branch_sum, residual_sum, cross_sum, scaled_branch_sum, reduction);
  ptx::cluster_wait();
  dsmem_reduce(branch_sum, residual_sum, cross_sum, scaled_branch_sum, reduction, cluster_rank);
  const float post_rstd = math::rsqrt(branch_sum * kInvHidden + norm.post_eps);
  const float pre_sum = fmaf(post_rstd, fmaf(post_rstd, scaled_branch_sum, 2.0f * cross_sum), residual_sum);
  const float pre_rstd = math::rsqrt(fmaxf(pre_sum, 0.0f) * kInvHidden + norm.pre_eps);

#pragma unroll
  for (int pack = 0; pack < kPacks; ++pack) {
    const int index = cluster_rank * kNumThreads + threadIdx.x + pack * kTotalThreads;
    BF16Vec output_values;
    F32Vec new_stream_values[kFParts];
#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      const int part = pair / kFPairs;
      const int lane = pair % kFPairs;
      auto normed = values[pack][pair];
      normed = muse_ag_norm::mul_pair(normed, {post_rstd, post_rstd});
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        const auto weight = post_w[pack][pair];
        normed = muse_ag_norm::mul_pair(normed, weight);
      }
      normed = round_bf16(normed);
      const auto beta_value = beta_values[pack][pair];
      const auto residual = residual_terms[pack][pair];
      const fp32x2_t updated = muse_ag_norm::fma_pair(beta_value, normed, residual);
      new_stream_values[part][lane] = updated;
      fp32x2_t normalized = muse_ag_norm::mul_pair(updated, {pre_rstd, pre_rstd});
      if (has_pre_weight) {
        const auto weight = cast<fp32x2_t>(pre_w_pf[pack][pair]);
        normalized = muse_ag_norm::mul_pair(normalized, weight);
      }
      output_values[pair] = cast<bf16x2_t>(normalized);
    }
    output_values.store(out, index);
#pragma unroll
    for (int part = 0; part < kFParts; ++part) {
      new_stream_values[part].store(new_stream, index * kFParts + part);
    }
  }

  // Every slot this block consumed was reset above; publish the next phase.
  __syncthreads();
  if (threadIdx.x == 0) params.counter[blockIdx.x].set(phase ^ 1);
}

// C == 1: persistent kernels, each owning rows bx, bx + grid, ...
template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    muse_spark_all_gather_norm_residual_row_kernel(const __grid_constant__ muse_ag_norm::Params params) {
  using namespace device;
  using namespace device::norm_residual;

  constexpr int kPacks = kHidden / (kNumThreads * kVec);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr int kPreSlots = kPrefetchWeights ? kPacks : 1;
  constexpr uint32_t kNLocal = static_cast<uint32_t>(kHidden) / kWorld;
  static_assert(kHidden % (kNumThreads * kVec) == 0);
  static_assert(kNLocal % kVec == 0, "a vector must never cross a rank block");

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  const auto& norm = params.norm;
  const auto post_weight = static_cast<const bf16_t*>(norm.post_weight);
  const auto pre_weight = static_cast<const bf16_t*>(norm.pre_weight);
  const auto alpha_ptr = static_cast<const fp32_t*>(norm.alpha);
  const auto beta_ptr = static_cast<const fp32_t*>(norm.beta);

  __shared__ float smem[kNumWarps == 1 ? 1 : 2 * kNumWarps];

  const bool channel_wise = kGate && norm.gate_channel_wise;
  const bool has_pre_weight = norm.has_pre_weight;
  const bool pre_bf16 = norm.pre_norm_bf16;

  PDLWaitPrimary<kUsePDL>();
  const uint32_t phase = muse_ag_norm::begin_phase<kNumThreads>(params);
  const uint32_t half_off = phase * params.half_bytes;
  muse_ag_norm::push_local_slice<kNLocal, kNumThreads>(params, half_off);
  PDLTriggerSecondary<kUsePDL>();

  const auto gathered = reinterpret_cast<bf16_t*>(params.ws_local + half_off);
  for (uint32_t row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    const int64_t base = static_cast<int64_t>(row) * kHidden;
    const auto stream = static_cast<const fp32_t*>(norm.stream) + base;
    const auto out = static_cast<bf16_t*>(norm.output) + base;
    const auto new_stream = static_cast<fp32_t*>(norm.new_stream) + base;

    fp32x2_t vals[kPacks][kPairs];
    fp32x2_t post_w[kPostWeight ? kPacks : 1][kPairs];
    F32Vec stream_v[kPacks][kFParts];
    F32Vec alpha_pf[kPreSlots][kFParts], beta_pf[kPreSlots][kFParts];
    BF16Vec pre_w_pf[kPreSlots];
    uint4 receive_cache[2];

    float sum_sq = 0.0f;
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = threadIdx.x + p * kNumThreads;
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
      const uint4 raw = muse_ag_norm::receive_pack<kPacks, kNumThreads, kNLocal>(
          gathered, params.num_rows, row, idx * kVec, p, receive_cache);
      const auto& branch_v = *reinterpret_cast<const BF16Vec*>(&raw);
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        auto v = cast<fp32x2_t>(branch_v[q]);
        if constexpr (kPostNormIsInputScale && kPostWeight) {
          v = muse_ag_norm::mul_pair(v, post_w[p][q]);
          v = round_bf16(v);
        }
        vals[p][q] = v;
        sum_sq = fmaf(v.x, v.x, sum_sq);
        sum_sq = fmaf(v.y, v.y, sum_sq);
      }
    }

    sum_sq = cta::reduce_sum<kNumThreads>(sum_sq, smem);
    const float post_rstd = math::rsqrt(sum_sq / kHidden + norm.post_eps);

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
        normed_b = muse_ag_norm::mul_pair(normed_b, {post_rstd, post_rstd});
        if constexpr (!kPostNormIsInputScale && kPostWeight) {
          normed_b = muse_ag_norm::mul_pair(normed_b, post_w[p][q]);
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
    const float pre_rstd = math::rsqrt(pre_sum_sq / kHidden + norm.pre_eps);

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
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        fp32x2_t v = vals[p][q];
        v = muse_ag_norm::mul_pair(v, {pre_rstd, pre_rstd});
        if (has_pre_weight) {
          const auto wf = cast<fp32x2_t>(weight_v[q]);
          v = muse_ag_norm::mul_pair(v, wf);
        }
        if (pre_bf16) v = round_bf16(v);
        out_v[q] = cast<bf16x2_t>(v);
      }
      out_v.store(out, idx);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) params.counter[blockIdx.x].set(phase ^ 1);
}

using host::distributed::CommunicatorRef;

template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    int kClusterSize,
    bool kUsePDL,
    bool kPostNormIsInputScale,
    bool kPrefetchWeights,
    bool kHasPostWeight,
    bool kHasGate>
struct MuseSparkAllGatherNormResidual {
  static_assert(kClusterSize == 1 || kClusterSize == 2 || kClusterSize == 4 || kClusterSize == 8 || kClusterSize == 16);
  static constexpr int64_t kNLocal = kHidden / kWorld;

  static void
  run(CommunicatorRef ref,
      const tvm::ffi::TensorView local,
      const tvm::ffi::TensorView stream,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView new_stream,
      const tvm::ffi::Optional<tvm::ffi::TensorView> post_weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> pre_weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> alpha,
      const tvm::ffi::Optional<tvm::ffi::TensorView> beta,
      double post_eps,
      double pre_eps) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();

    auto M = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto G = SymbolicSize{"gate_dim"};
    auto device_ = SymbolicDevice{};
    H.set_value(kHidden);
    device_.set_options<kDLCUDA>();

    TensorMatcher({M, kNLocal}).with_dtype<bf16_t>().with_device(device_).verify(local);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(stream);
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(output);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(new_stream);
    if (post_weight.has_value()) {
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(post_weight.value());
    }
    if (pre_weight.has_value()) {
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(pre_weight.value());
    }
    const bool gate = alpha.has_value();
    RuntimeCheck(gate == beta.has_value(), "alpha and beta must both be present or both absent");
    // One instantiation per module keeps JIT builds cheap; the caller picks it.
    RuntimeCheck(post_weight.has_value() == kHasPostWeight, "post_weight presence must match kHasPostWeight");
    RuntimeCheck(gate == kHasGate, "alpha/beta presence must match kHasGate");
    bool gate_channel_wise = false;
    if (gate) {
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(alpha.value());
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(beta.value());
      const auto n = G.unwrap();
      RuntimeCheck(n == 1 || n == kHidden, "alpha/beta must have 1 or hidden_size elements");
      gate_channel_wise = (n == kHidden);
    }

    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    RuntimeCheck(num_rows >= 1, "fused all-gather norm requires M >= 1");
    RuntimeCheck(push.world_size == kWorld, "the kernel is compiled for TP", kWorld);
    RuntimeCheck(push.mc_workspace != nullptr, "requires a multicast-capable push plane");
    RuntimeCheck(
        int64_t(num_rows) * kHidden * 2 <= int64_t(push.slot_bytes), "gathered rows exceed the push slot size");
    uint32_t grid = num_rows * kClusterSize;
    if constexpr (kClusterSize == 1) {
      grid = std::min<uint32_t>(num_rows, push.num_blocks);
    } else {
      RuntimeCheck(
          int64_t(num_rows) * kClusterSize <= int64_t(push.num_blocks),
          "M * cluster_size exceeds the push plane's block counters");
    }

    const auto params = muse_ag_norm::Params{
        .ws_local = push.workspaces[push.rank],
        .ws_mc = push.mc_workspace,
        .counter = push.counter,
        .num_counters = push.num_blocks,
        .half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size),
        .rank = push.rank,
        .num_rows = num_rows,
        .local = static_cast<const bf16_t*>(local.data_ptr()),
        .norm =
            NormResidualParams{
                .branch = nullptr,
                .stream = stream.data_ptr(),
                .output = output.data_ptr(),
                .new_stream = new_stream.data_ptr(),
                .router_out = nullptr,
                .post_weight = post_weight.has_value() ? post_weight.value().data_ptr() : nullptr,
                .pre_weight = pre_weight.has_value() ? pre_weight.value().data_ptr() : nullptr,
                .alpha = gate ? alpha.value().data_ptr() : nullptr,
                .beta = gate ? beta.value().data_ptr() : nullptr,
                .post_eps = static_cast<float>(post_eps),
                .pre_eps = static_cast<float>(pre_eps),
                .num_rows = static_cast<int>(num_rows),
                .gate_channel_wise = gate_channel_wise,
                .has_pre_weight = pre_weight.has_value(),
                .pre_norm_bf16 = false,
            },
    };
    const auto device = device_.unwrap();
    if constexpr (kClusterSize == 1) {
      LaunchKernel(grid, kNumThreads, device)
          .config({.use_pdl = kUsePDL})(
              muse_spark_all_gather_norm_residual_row_kernel<
                  kHidden,
                  kWorld,
                  kNumThreads,
                  kUsePDL,
                  kPrefetchWeights,
                  kPostNormIsInputScale,
                  kHasPostWeight,
                  kHasGate>,
              params);
    } else {
      constexpr auto kernel = muse_spark_all_gather_norm_residual_cluster_kernel<
          kHidden,
          kWorld,
          kNumThreads,
          kClusterSize,
          kUsePDL,
          kPostNormIsInputScale,
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
      LaunchKernel(grid, kNumThreads, device)
          .config({
              .use_pdl = kUsePDL,
              .cluster_dim = dim3(kClusterSize, 1, 1),
          })(kernel, params);
    }
  }
};

}  // namespace sglang
