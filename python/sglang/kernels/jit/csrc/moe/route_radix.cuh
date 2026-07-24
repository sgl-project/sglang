#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

inline constexpr uint32_t kNumExperts_ = 896;
inline constexpr uint32_t kTopK_ = 16;

struct LargeRouterRadixTrait {
  static constexpr uint32_t kNumExperts = kNumExperts_;
  static constexpr uint32_t kTopK = kTopK_;
  static constexpr uint32_t kVecSize = 4;

  static constexpr uint32_t kRadixBits = 8;
  static constexpr uint32_t kRadixSize = 1 << kRadixBits;
  static constexpr uint32_t kRadixRounds = 32 / kRadixBits;
  static constexpr uint32_t kBlockSize = kNumExperts / kVecSize;  // 224 = 7 warps
  static constexpr uint32_t kNumWarps = kBlockSize / 32;
  struct alignas(16) MatchBin {
    uint32_t bin;
    uint32_t above_count;  // active elements in bins strictly above `bin`
    uint32_t equal_count;  // active elements in bin `bin`
  };
  struct Smem {
    uint32_t warp_sum[3][kNumWarps];  // cross-warp scan workspace
    MatchBin match[kRadixRounds];
    uint32_t histogram[kRadixSize];
    // winner staging (compaction order = expert-id ascending)
    int32_t wid[kTopK];
    uint32_t wkey[kTopK];
    fp32_t wact[kTopK];
    // sorted staging ((key desc, id asc) order), only used when sorted != 0
    int32_t sid[kTopK];
    fp32_t sact[kTopK];
    fp32_t norm;
  };
};

struct RouteRadixParams {
  const bf16_t* __restrict__ scores;
  const fp32_t* __restrict__ bias;
  fp32_t* __restrict__ out_w;
  int32_t* __restrict__ out_i;
  int M;
  long long scores_stride;
  long long out_w_stride;
  long long out_i_stride;
  float routed_scaling_factor;
  int renormalize;
  int apply_scale;
  int sorted;
};

inline constexpr float kNanFloor_ = -1e30f;

// Monotone unsigned key: larger biased -> larger key. Caller must have floored
// biased-NaN. Canonicalizes -0.0 -> +0.0 so equal values get equal keys.
SGL_DEVICE uint32_t biased_to_key(float biased) {
  if (biased == 0.0f) biased = 0.0f;
  uint32_t u = __float_as_uint(biased);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// tl.sigmoid(x) = 1/(1+exp(-x)). Must stay instruction-identical to v1's
// sigmoid_match so both kernels rank (and weight) identically.
SGL_DEVICE float sigmoid_match(float x) {
  return __fdividef(1.0f, 1.0f + __expf(-x));
}

SGL_DEVICE float nan_floor(float x) {
  return (x == x) ? x : kNanFloor_;
}

SGL_DEVICE void bar_sync(uint32_t id, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
}

SGL_DEVICE uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

// Exclusive prefix (block-wide, thread-rank order) of `cnt`. Uses
// smem_warp_sum[kNumWarps]; syncs on entry (so the workspace can be reused
// across calls) and before the cross-warp read.
SGL_DEVICE uint32_t block_exclusive_sum(uint32_t cnt, uint32_t lane_id, uint32_t warp_id, uint32_t* smem_warp_sum) {
  const uint32_t inc = warp_inclusive_sum(lane_id, cnt);
  if (lane_id == 31) smem_warp_sum[warp_id] = inc;
  __syncthreads();
  // TODO: replace `__reduce_add_sync` with `warp::reduce_sum` after rebase
  const auto base = __reduce_add_sync(0xFFFFFFFF, lane_id < warp_id ? smem_warp_sum[lane_id] : 0u);
  return base + inc - cnt;
}

template <bool kUsePDL>
__global__ __launch_bounds__(LargeRouterRadixTrait::kBlockSize)  //
    void route_radix_kernel(const __grid_constant__ RouteRadixParams params) {
  using namespace device;
  using T = LargeRouterRadixTrait;
  constexpr uint32_t kVecSize = T::kVecSize;
  constexpr uint32_t kRadixLanes = T::kRadixSize / 2;  // 128: 2 bins per thread
  enum { BAR_RESERVED = 0, BAR_SUM = 1 };
  __shared__ typename T::Smem smem;

  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  // grid.x == M exactly; no row guard (an early return would deadlock the
  // block-wide barriers below).

  // ---- Load + key transform: thread tx owns experts [4*tx, 4*tx+4) ----
  uint32_t keys[kVecSize];
  float act[kVecSize];  // raw sigmoid (weight source) — never NaN-sanitized
  {
    const auto scores = &params.scores[bx * params.scores_stride];
    AlignedVector<fp32x2_t, kVecSize / 2> bias_vec;
    AlignedVector<bf16x2_t, kVecSize / 2> scores_vec;

    // prefetch bias (frozen weight) before the PDL wait
    bias_vec.load(params.bias, tx);
    PDLWaitPrimary<kUsePDL>();
    scores_vec.load(scores, tx);

#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      const auto [x, y] = cast<fp32x2_t>(scores_vec[i]);
      const auto sx = sigmoid_match(x), sy = sigmoid_match(y);
      keys[2 * i + 0] = biased_to_key(nan_floor(sx + bias_vec[i].x));
      keys[2 * i + 1] = biased_to_key(nan_floor(sy + bias_vec[i].y));
      act[2 * i + 0] = sx;
      act[2 * i + 1] = sy;
    }
  }

  // ---- Radix narrowing, MSB -> LSB ----
  // Invariants entering round r:
  //   active[i]      <=> key's top 8r bits == threshold's top 8r bits
  //   total_active    = size of the active set
  //   topk            = winners still to take from the active set (1..total_active)
  bool active[kVecSize];
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    active[i] = true;
  }

  uint32_t total_active = T::kNumExperts;
  uint32_t topk = T::kTopK;
  uint32_t threshold = 0;      // assembled split-key prefix (unexamined low bits zero)
  uint32_t examined_mask = 0;  // bits of `threshold` that have been fixed
  bool take_all_equals = false;

  {
    AlignedVector<uint32_t, 2> zero;
    zero.fill(0);
    if (tx < kRadixLanes) zero.store(smem.histogram, tx);

#pragma unroll
    for (uint32_t round = 0; round < T::kRadixRounds; ++round) {
      __syncthreads();  // histogram zeroed & previous match consumed
      const uint32_t shift = 24 - round * 8;
      uint32_t bin[kVecSize];
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        bin[i] = (keys[i] >> shift) & 0xff;
      }
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        if (active[i]) atomicAdd(&smem.histogram[bin[i]], 1);
      }
      __syncthreads();

      // Split-bin search on 128 threads: thread t owns bins {2t, 2t+1}.
      // The split bin b is the unique bin with above(b) < topk <= above(b) + hist[b].
      if (tx < kRadixLanes) {
        AlignedVector<uint32_t, 2> hist;
        hist.load(smem.histogram, tx);
        const auto local_val = hist[0] + hist[1];
        const auto warp_inc = warp_inclusive_sum(lane_id, local_val);
        if (lane_id == kWarpThreads - 1) smem.warp_sum[0][warp_id] = warp_inc;
        bar_sync(BAR_SUM, kRadixLanes);
        const auto inter = __reduce_add_sync(0xFFFFFFFF, lane_id < warp_id ? smem.warp_sum[0][lane_id] : 0u);
        const auto prefix = inter + warp_inc;        // active elements in bins [0, 2t+1]
        const auto above_r = total_active - prefix;  // in bins > 2t+1
        const auto above_m = above_r + hist[1];      // in bins > 2t
        const auto above_l = above_m + hist[0];      // in bins >= 2t
        if (above_r < topk && above_m >= topk) {
          smem.match[round] = {tx * 2 + 1, above_r, hist[1]};
        } else if (above_m < topk && above_l >= topk) {
          smem.match[round] = {tx * 2 + 0, above_m, hist[0]};
        }
      }
      __syncthreads();

      const auto [threshold_bin, above_count, equal_count] = smem.match[round];
      threshold |= threshold_bin << shift;
      examined_mask |= 0xffu << shift;
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        active[i] &= (bin[i] == threshold_bin);
      }
      total_active = equal_count;
      topk -= above_count;  // split condition guarantees 1 <= topk <= equal_count
      if (topk == equal_count) {
        // The remaining quota exactly covers the equal set: every active
        // element wins, no deeper narrowing or tie-break needed. At the last
        // round this is the no-full-key-tie case (the typical one).
        take_all_equals = true;
        break;
      }
      // Re-zero for the next round (synced by the loop-top barrier). Reaching
      // round 3 with topk < equal_count means a full-key tie: resolved below
      // by the smallest-id rank among `active`.
      if (round + 1 < T::kRadixRounds && tx < kRadixLanes) zero.store(smem.histogram, tx);
    }
  }

  // ---- Epilogue: collect the K winners ----
  // Strict winners: examined bits compare above the split prefix (these were
  // peeled off `active` in earlier rounds). Equal set (== `active`): take all
  // (take_all_equals) or the `topk` smallest ids (full-key tie-break).
  bool selected[kVecSize];
  if (take_all_equals) {
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      selected[i] = active[i] || (keys[i] & examined_mask) > threshold;
    }
  } else {  // deterministic tie-break
    uint32_t cnt = 0;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      cnt += active[i] ? 1 : 0;
    }
    uint32_t rank = block_exclusive_sum(cnt, lane_id, warp_id, smem.warp_sum[1]);
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      const bool eq_win = active[i] && rank < topk;
      if (active[i]) ++rank;
      selected[i] = eq_win || (keys[i] & examined_mask) > threshold;
    }
  }

  // Compaction slots in expert-id order (deterministic).
  uint32_t selected_cnt = 0;
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    selected_cnt += selected[i] ? 1 : 0;
  }
  uint32_t slot = block_exclusive_sum(selected_cnt, lane_id, warp_id, smem.warp_sum[2]);
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    if (selected[i] && slot < T::kTopK) {
      smem.wid[slot] = (int32_t)(tx * kVecSize + i);
      smem.wkey[slot] = keys[i];
      smem.wact[slot] = act[i];
      ++slot;
    }
  }
  __syncthreads();

  static_assert(T::kTopK <= kWarpThreads);
  if (tx < T::kTopK) {
    uint32_t rank = tx;
    auto w = smem.wact[tx];
    const auto id = smem.wid[tx];
    if (params.sorted) {
      const uint32_t ka = smem.wkey[tx];
      const int32_t ia = id;
      rank = 0;
#pragma unroll
      for (uint32_t b = 0; b < T::kTopK; ++b) {
        if (smem.wkey[b] > ka || (smem.wkey[b] == ka && smem.wid[b] < ia)) ++rank;
      }
    }
    PDLTriggerSecondary<kUsePDL>();
    float sum = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < T::kTopK; ++i) {
      sum += smem.wact[i];
    }
    const auto norm = (sum > 0.0f) ? sum : 1.0f;
    if (params.renormalize) w = w / norm;
    if (params.apply_scale) w = w * params.routed_scaling_factor;
    params.out_w[bx * params.out_w_stride + rank] = w;
    params.out_i[bx * params.out_i_stride + rank] = id;
  }
}

}  // namespace sglang

template <bool kUsePDL>
struct RouteRadixKernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize,
      bool apply_scale,
      bool sorted) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto N_ = SymbolicSize{"num_experts"};
    auto K_ = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M_, N_}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(scores);
    TensorMatcher({N_}).with_dtype<fp32_t>().with_device(device).verify(bias);
    TensorMatcher({M_, K_}).with_dtype<fp32_t>().with_device(device).verify(out_w);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_device(device).verify(out_i);

    RuntimeCheck(
        N_.unwrap() == sglang::kNumExperts_ && K_.unwrap() == sglang::kTopK_ && topk == sglang::kTopK_,
        "route_radix is specialized for N=896, K=16");
    // 8-byte vectorized row loads need 8B-aligned row starts.
    RuntimeCheck(scores.stride(0) % 4 == 0, "route_radix: scores row stride must be a multiple of 4");

    const auto M = static_cast<uint32_t>(M_.unwrap());
    if (M == 0) return;

    const auto params = sglang::RouteRadixParams{
        static_cast<const bf16_t*>(scores.data_ptr()),
        static_cast<const fp32_t*>(bias.data_ptr()),
        static_cast<fp32_t*>(out_w.data_ptr()),
        static_cast<int32_t*>(out_i.data_ptr()),
        static_cast<int>(M),
        static_cast<long long>(scores.stride(0)),
        static_cast<long long>(out_w.stride(0)),
        static_cast<long long>(out_i.stride(0)),
        static_cast<float>(routed_scaling_factor),
        renormalize ? 1 : 0,
        apply_scale ? 1 : 0,
        sorted ? 1 : 0};

    LaunchKernel(M, sglang::LargeRouterRadixTrait::kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(sglang::route_radix_kernel<kUsePDL>, params);
  }
};
