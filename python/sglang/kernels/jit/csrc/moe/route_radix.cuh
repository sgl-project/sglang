// MoE routing by radix select: the standalone route kernel and the fused-gate
// front end, over one copy of the radix primitives (previously
// moe/radix_select_common.cuh, folded in once its two consumers became one file).

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/type.cuh>   // For dtype_trait, bf16_t, fp32_t, cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, PDL helpers
#include <sgl_kernel/vec.cuh>    // For AlignedVector
#include <sgl_kernel/warp.cuh>   // For warp::copy_bytes, elect_one_lane, inclusive_sum

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace moe::radix {

struct RadixSelectBase {
  static constexpr uint32_t kRadixBits = 8;
  static constexpr uint32_t kRadixSize = 1 << kRadixBits;
  static constexpr uint32_t kRadixRounds = 32 / kRadixBits;

  struct alignas(16) MatchBin {
    uint32_t bin;
    uint32_t above_count;  // active elements in bins strictly above `bin`
    uint32_t equal_count;  // active elements in bin `bin`
  };
};

inline constexpr float kNanFloor = -1e30f;

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
  return (x == x) ? x : kNanFloor;
}

SGL_DEVICE void bar_sync(uint32_t id, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
}

// Exclusive prefix (block-wide, thread-rank order) of `cnt`. Uses
// smem_warp_sum[kNumWarps]; syncs on entry (so the workspace can be reused
// across calls) and before the cross-warp read.
SGL_DEVICE uint32_t block_exclusive_sum(uint32_t cnt, uint32_t lane_id, uint32_t warp_id, uint32_t* smem_warp_sum) {
  const uint32_t inc = device::warp::inclusive_sum(lane_id, cnt);
  if (lane_id == 31) smem_warp_sum[warp_id] = inc;
  __syncthreads();
  // TODO: replace `__reduce_add_sync` with `warp::reduce_sum`
  const auto base = __reduce_add_sync(0xFFFFFFFF, lane_id < warp_id ? smem_warp_sum[lane_id] : 0u);
  return base + inc - cnt;
}

}  // namespace moe::radix

inline constexpr uint32_t kNumExperts_ = 896;
inline constexpr uint32_t kTopK_ = 16;

struct LargeRouterRadixTrait : moe::radix::RadixSelectBase {
  static constexpr uint32_t kNumExperts = kNumExperts_;
  static constexpr uint32_t kTopK = kTopK_;
  static constexpr uint32_t kVecSize = 4;

  static constexpr uint32_t kBlockSize = kNumExperts / kVecSize;  // 224 = 7 warps
  static constexpr uint32_t kNumWarps = kBlockSize / 32;
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
  const void* __restrict__ scores;  // bf16 or fp32, typed by the kernel template
  const fp32_t* __restrict__ bias;
  fp32_t* __restrict__ out_w;
  int32_t* __restrict__ out_i;
  // Optional trtllm-gen routed-MoE packing: (id << 16) | bf16(weight) bits,
  // bit-identical to the standalone triton pack. nullptr skips the store.
  int32_t* __restrict__ out_packed;
  int M;
  long long scores_stride;
  long long out_w_stride;
  long long out_i_stride;
  long long out_packed_stride;
  float routed_scaling_factor;
  int renormalize;
  int apply_scale;
  int sorted;
};

// Whole-CTA routing body, callable from other kernels (the fused
// route+quant launch runs it on its first M CTAs). Routes row `blockIdx.x`;
// every thread of the 224-wide CTA must enter (block barriers inside).
template <bool kUsePDL, typename TScore>
SGL_DEVICE void route_radix_block(const RouteRadixParams& params, typename LargeRouterRadixTrait::Smem& smem) {
  using namespace device;
  using T = LargeRouterRadixTrait;
  constexpr uint32_t kVecSize = T::kVecSize;
  constexpr uint32_t kRadixLanes = T::kRadixSize / 2;  // 128: 2 bins per thread
  enum { BAR_RESERVED = 0, BAR_SUM = 1 };

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
    const auto scores = static_cast<const TScore*>(params.scores) + bx * params.scores_stride;
    AlignedVector<fp32x2_t, kVecSize / 2> bias_vec;
    // bf16: 2x bf16x2 (8B row loads); fp32: 2x fp32x2 (16B row loads). The
    // radix math below is fp32 either way — only the load width differs.
    AlignedVector<packed_t<TScore>, kVecSize / 2> scores_vec;

    // prefetch bias (frozen weight) before the PDL wait
    bias_vec.load(params.bias, tx);
    PDLWaitPrimary<kUsePDL>();
    scores_vec.load(scores, tx);

#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      fp32x2_t xy;
      if constexpr (std::is_same_v<TScore, fp32_t>) {
        xy = scores_vec[i];
      } else {
        xy = cast<fp32x2_t>(scores_vec[i]);
      }
      const auto [x, y] = xy;
      const auto sx = moe::radix::sigmoid_match(x), sy = moe::radix::sigmoid_match(y);
      keys[2 * i + 0] = moe::radix::biased_to_key(moe::radix::nan_floor(sx + bias_vec[i].x));
      keys[2 * i + 1] = moe::radix::biased_to_key(moe::radix::nan_floor(sy + bias_vec[i].y));
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
        const auto warp_inc = device::warp::inclusive_sum(lane_id, local_val);
        if (lane_id == kWarpThreads - 1) smem.warp_sum[0][warp_id] = warp_inc;
        moe::radix::bar_sync(BAR_SUM, kRadixLanes);
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
    uint32_t rank = moe::radix::block_exclusive_sum(cnt, lane_id, warp_id, smem.warp_sum[1]);
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
  uint32_t slot = moe::radix::block_exclusive_sum(selected_cnt, lane_id, warp_id, smem.warp_sum[2]);
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
    if (params.out_packed != nullptr) {
      // (id << 16) | bf16(w) bits — RN float->bf16 matches the triton pack.
      const auto bits = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(w)));
      params.out_packed[bx * params.out_packed_stride + rank] =
          static_cast<int32_t>((static_cast<uint32_t>(id) << 16) | bits);
    }
  }
}

template <bool kUsePDL, typename TScore>
__global__ __launch_bounds__(LargeRouterRadixTrait::kBlockSize)  //
    void route_radix_kernel(const __grid_constant__ RouteRadixParams params) {
  __shared__ typename LargeRouterRadixTrait::Smem smem;
  route_radix_block<kUsePDL, TScore>(params, smem);
}

// ---------------------------------------------------------------------------
// fused-gate front end. Same radix primitives above; a separate kernel because it
// folds the gate and the quant epilogue into one launch. Only the module that
// instantiates it pays for it -- an uninstantiated template costs parse time, not
// codegen, which is what lets both share this translation unit.
// ---------------------------------------------------------------------------
inline constexpr uint32_t kFGTNumExperts = 896;
inline constexpr uint32_t kFGTTopK = 16;
// 7 warps: 896 experts / 224 threads = 4 experts per thread in the epilogue, and
// one expert per warp per pass in phase 1.  Keeping the block at route_radix's
// shape lets the epilogue reuse its radix-select verbatim.
/// Block size is a tunable: it sets how many experts each thread owns in the
/// radix select (kNumExperts / kBlockSize).  It must be at least kRadixSize/2 =
/// 128 threads (the split-bin search puts 2 bins per thread) and must divide the
/// expert count into an even per-thread count (the loads are fp32x2 pairs), so
/// 224 (4 experts/thread) and 448 (2 experts/thread) are the legal choices for
/// 896 experts.
template <uint32_t kBlockSize_>
struct MoEFrontTrait : moe::radix::RadixSelectBase {
  static constexpr uint32_t kNumExperts = kFGTNumExperts;
  static constexpr uint32_t kTopK = kFGTTopK;
  static constexpr uint32_t kBlockSize = kBlockSize_;
  static constexpr uint32_t kVecSize = kNumExperts / kBlockSize;  // experts per thread
  static constexpr uint32_t kNumWarps = kBlockSize / 32;

  static_assert(kNumExperts % kBlockSize == 0, "block size must divide the expert count");
  static_assert(kVecSize % 2 == 0, "experts per thread must be even (fp32x2 loads)");
  static_assert(kBlockSize >= kRadixSize / 2, "block must cover the split-bin search lanes");

  struct Smem {
    uint32_t warp_sum[3][kNumWarps];
    MatchBin match[kRadixRounds];
    uint32_t histogram[kRadixSize];
    int32_t wid[kTopK];
    uint32_t wkey[kTopK];
    fp32_t wact[kTopK];
  };
};

struct MoEFrontParams {
  const fp32_t* __restrict__ bias;    // [E] fp32 correction bias
  const fp32_t* __restrict__ logits;  // [M, logits_stride] fp32, gate slice first
  fp32_t* __restrict__ out_w;         // [M, topk] fp32
  int32_t* __restrict__ out_i;        // [M, topk] int32
  int M;
  int logits_stride;  // E for the router-only entry, E + latent for the front
  long long out_w_stride;
  long long out_i_stride;
  float routed_scaling_factor;
  int renormalize;
  int apply_scale;
  // Merged-front entry only: the [M, latent] bf16 routed_input to emit.
  bf16_t* __restrict__ routed_out;
  int latent;
  long long routed_stride;
};

/// Radix-select top-k over one token's fp32 logits.  Lifted from
/// route_radix.cuh; the only change is the input dtype (fp32 in place of bf16).
template <bool kUsePDL, typename T>
SGL_DEVICE void fgt_select_topk(
    const MoEFrontParams& params, typename T::Smem& smem, int m, uint32_t tx, uint32_t warp_id, uint32_t lane_id) {
  constexpr uint32_t kVecSize = T::kVecSize;
  constexpr uint32_t kRadixLanes = T::kRadixSize / 2;
  enum { BAR_SUM = 1 };

  uint32_t keys[kVecSize];
  float act[kVecSize];
  {
    // 4 experts per thread as two fp32x2 (16B) loads, matching route_radix.
    device::AlignedVector<fp32x2_t, kVecSize / 2> bias_vec;
    bias_vec.load(params.bias, tx);
    device::AlignedVector<fp32x2_t, kVecSize / 2> lv;
    lv.load(params.logits + (long long)m * params.logits_stride, tx);
    float logit[kVecSize];
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      logit[2 * i + 0] = lv[i].x;
      logit[2 * i + 1] = lv[i].y;
    }
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      const float sx = moe::radix::sigmoid_match(logit[2 * i + 0]);
      const float sy = moe::radix::sigmoid_match(logit[2 * i + 1]);
      act[2 * i + 0] = sx;
      act[2 * i + 1] = sy;
      keys[2 * i + 0] = moe::radix::biased_to_key(moe::radix::nan_floor(sx + bias_vec[i].x));
      keys[2 * i + 1] = moe::radix::biased_to_key(moe::radix::nan_floor(sy + bias_vec[i].y));
    }
  }

  bool active[kVecSize];
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    active[i] = true;
  }

  uint32_t total_active = T::kNumExperts;
  uint32_t topk = T::kTopK;
  uint32_t threshold = 0;
  uint32_t examined_mask = 0;
  bool take_all_equals = false;

  {
    device::AlignedVector<uint32_t, 2> zero;
    zero.fill(0);
    if (tx < kRadixLanes) zero.store(smem.histogram, tx);

#pragma unroll
    for (uint32_t round = 0; round < T::kRadixRounds; ++round) {
      __syncthreads();
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

      if (tx < kRadixLanes) {
        device::AlignedVector<uint32_t, 2> hist;
        hist.load(smem.histogram, tx);
        const auto local_val = hist[0] + hist[1];
        const auto warp_inc = device::warp::inclusive_sum(lane_id, local_val);
        if (lane_id == 31) smem.warp_sum[0][warp_id] = warp_inc;
        moe::radix::bar_sync(BAR_SUM, kRadixLanes);
        const auto inter = __reduce_add_sync(0xFFFFFFFF, lane_id < warp_id ? smem.warp_sum[0][lane_id] : 0u);
        const auto prefix = inter + warp_inc;
        const auto above_r = total_active - prefix;
        const auto above_m = above_r + hist[1];
        const auto above_l = above_m + hist[0];
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
      topk -= above_count;
      if (topk == equal_count) {
        take_all_equals = true;
        break;
      }
      if (round + 1 < T::kRadixRounds && tx < kRadixLanes) zero.store(smem.histogram, tx);
    }
  }

  bool selected[kVecSize];
  if (take_all_equals) {
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      selected[i] = active[i] || (keys[i] & examined_mask) > threshold;
    }
  } else {
    uint32_t cnt = 0;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      cnt += active[i] ? 1 : 0;
    }
    uint32_t rank = moe::radix::block_exclusive_sum(cnt, lane_id, warp_id, smem.warp_sum[1]);
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      const bool eq_win = active[i] && rank < topk;
      if (active[i]) ++rank;
      selected[i] = eq_win || (keys[i] & examined_mask) > threshold;
    }
  }

  uint32_t selected_cnt = 0;
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    selected_cnt += selected[i] ? 1 : 0;
  }
  uint32_t slot = moe::radix::block_exclusive_sum(selected_cnt, lane_id, warp_id, smem.warp_sum[2]);
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    if (selected[i] && slot < T::kTopK) {
      smem.wid[slot] = (int32_t)(tx * kVecSize + i);
      smem.wact[slot] = act[i];
      ++slot;
    }
  }
  __syncthreads();

  static_assert(T::kTopK <= 32);
  if (tx < T::kTopK) {
    auto wv = smem.wact[tx];
    const auto id = smem.wid[tx];
    float sum = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < T::kTopK; ++i) {
      sum += smem.wact[i];
    }
    const auto norm = (sum > 0.0f) ? sum : 1.0f;
    if (params.renormalize) wv = wv / norm;
    if (params.apply_scale) wv = wv * params.routed_scaling_factor;
    params.out_w[m * params.out_w_stride + tx] = wv;
    params.out_i[m * params.out_i_stride + tx] = id;
  }
  __syncthreads();  // smem reuse across the token loop
}

/// Tunables: `kBlockSize` sets the experts-per-thread of the radix select,
/// `kCastVec` the fp32 elements each thread converts per step (the cast moves
/// [T, 3584] fp32 in and bf16 out, which dominates the epilogue at large T), and
/// `kCastFirst` whether the cast is issued before the select (loads in flight
/// during the radix rounds) or after it.
template <bool kUsePDL, uint32_t kBlockSize, uint32_t kCastVec, bool kCastFirst>
__global__ __launch_bounds__(kBlockSize)  //
    void fused_front_epilogue_kernel(const __grid_constant__ MoEFrontParams params) {
  using namespace device;
  using T = MoEFrontTrait<kBlockSize>;
  __shared__ typename T::Smem smem;
  const uint32_t tx = threadIdx.x;
  const int m = (int)blockIdx.x;

  PDLWaitPrimary<kUsePDL>();

  // Cast the latent slice: [E, E + latent) fp32 -> [0, latent) bf16.
  auto cast_latent = [&]() {
    const fp32_t* src = params.logits + (long long)m * params.logits_stride + T::kNumExperts;
    bf16_t* dst = params.routed_out + (long long)m * params.routed_stride;
    for (int i = (int)tx * kCastVec; i < params.latent; i += (int)kBlockSize * kCastVec) {
      AlignedVector<fp32x2_t, kCastVec / 2> v;
      v.load(src, i / kCastVec);
      AlignedVector<bf16_t, kCastVec> o;
#pragma unroll
      for (uint32_t j = 0; j < kCastVec / 2; ++j) {
        o[2 * j + 0] = cast<bf16_t>(v[j].x);
        o[2 * j + 1] = cast<bf16_t>(v[j].y);
      }
      o.store(dst, i / kCastVec);
    }
  };

  if (kCastFirst) cast_latent();
  fgt_select_topk<kUsePDL, T>(params, smem, m, tx, tx / 32, tx % 32);
  if (!kCastFirst) cast_latent();
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

    auto score_dtype = SymbolicDType{};
    TensorMatcher({M_, N_})
        .with_dtype<bf16_t, fp32_t>(score_dtype)
        .with_device(device)
        .with_strides({-1, 1})
        .verify(scores);
    TensorMatcher({N_}).with_dtype<fp32_t>().with_device(device).verify(bias);
    TensorMatcher({M_, K_}).with_dtype<fp32_t>().with_device(device).verify(out_w);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_device(device).verify(out_i);

    RuntimeCheck(
        N_.unwrap() == sglang::kNumExperts_ && K_.unwrap() == sglang::kTopK_ && topk == sglang::kTopK_,
        "route_radix is specialized for N=896, K=16");
    // Vectorized row loads (8B for bf16, 16B for fp32) need aligned row
    // starts; stride % 4 elements covers both (4 x 2B = 8B / 4 x 4B = 16B).
    RuntimeCheck(scores.stride(0) % 4 == 0, "route_radix: scores row stride must be a multiple of 4");

    const auto M = static_cast<uint32_t>(M_.unwrap());
    if (M == 0) return;

    const auto params = sglang::RouteRadixParams{
        scores.data_ptr(),
        static_cast<const fp32_t*>(bias.data_ptr()),
        static_cast<fp32_t*>(out_w.data_ptr()),
        static_cast<int32_t*>(out_i.data_ptr()),
        /*out_packed=*/nullptr,
        static_cast<int>(M),
        static_cast<long long>(scores.stride(0)),
        static_cast<long long>(out_w.stride(0)),
        static_cast<long long>(out_i.stride(0)),
        /*out_packed_stride=*/0,
        static_cast<float>(routed_scaling_factor),
        renormalize ? 1 : 0,
        apply_scale ? 1 : 0,
        sorted ? 1 : 0};

    if (score_dtype.is_type<fp32_t>()) {
      LaunchKernel(M, sglang::LargeRouterRadixTrait::kBlockSize, device.unwrap())
          .enable_pdl(kUsePDL)(sglang::route_radix_kernel<kUsePDL, fp32_t>, params);
    } else {
      LaunchKernel(M, sglang::LargeRouterRadixTrait::kBlockSize, device.unwrap())
          .enable_pdl(kUsePDL)(sglang::route_radix_kernel<kUsePDL, bf16_t>, params);
    }
  }
};

template <bool kUsePDL>
struct FusedFrontEpilogueKernel {
  static void
  run(const tvm::ffi::TensorView merged,  // [M, E + latent] fp32, row-dense
      const tvm::ffi::TensorView bias,    // [E] fp32
      const tvm::ffi::TensorView out_w,   // [M, topk] fp32
      const tvm::ffi::TensorView out_i,   // [M, topk] int32
      const tvm::ffi::TensorView routed,  // [M, latent] bf16
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize,
      bool apply_scale,
      int64_t block_size,
      int64_t cast_vec,
      bool cast_first) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto W_ = SymbolicSize{"merged_width"};
    auto L_ = SymbolicSize{"latent"};
    auto E_ = SymbolicSize{"num_experts"};
    auto K_ = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M_, W_}).with_dtype<fp32_t>().with_device(device).with_strides({-1, 1}).verify(merged);
    TensorMatcher({E_}).with_dtype<fp32_t>().with_device(device).verify(bias);
    TensorMatcher({M_, K_}).with_dtype<fp32_t>().with_device(device).verify(out_w);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_device(device).verify(out_i);
    TensorMatcher({M_, L_}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(routed);

    const auto M = static_cast<int>(M_.unwrap());
    const auto latent = static_cast<int>(L_.unwrap());
    RuntimeCheck(
        E_.unwrap() == sglang::kFGTNumExperts && K_.unwrap() == sglang::kFGTTopK && topk == sglang::kFGTTopK,
        "fused_front_epilogue is specialized for E=896, topk=16");
    RuntimeCheck(
        static_cast<int>(W_.unwrap()) == static_cast<int>(sglang::kFGTNumExperts) + latent,
        "fused_front_epilogue: merged width must be num_experts + latent");
    // 16B vectorized reads of the fp32 rows and 8B writes of the bf16 rows.
    RuntimeCheck(latent % 4 == 0, "fused_front_epilogue: latent must be a multiple of 4");
    RuntimeCheck(
        merged.stride(0) % 4 == 0 && routed.stride(0) % 4 == 0,
        "fused_front_epilogue: row strides must be a multiple of 4");
    if (M == 0) return;

    auto params = sglang::MoEFrontParams{};
    params.bias = static_cast<const fp32_t*>(bias.data_ptr());
    params.logits = static_cast<fp32_t*>(merged.data_ptr());
    params.out_w = static_cast<fp32_t*>(out_w.data_ptr());
    params.out_i = static_cast<int32_t*>(out_i.data_ptr());
    params.M = M;
    params.out_w_stride = static_cast<long long>(out_w.stride(0));
    params.out_i_stride = static_cast<long long>(out_i.stride(0));
    params.routed_scaling_factor = static_cast<float>(routed_scaling_factor);
    params.renormalize = renormalize ? 1 : 0;
    params.apply_scale = apply_scale ? 1 : 0;
    params.logits_stride = static_cast<int>(merged.stride(0));
    params.routed_out = static_cast<bf16_t*>(routed.data_ptr());
    params.latent = latent;
    params.routed_stride = static_cast<long long>(routed.stride(0));

    // Tunables come from the JSON config table; see kernels/ops/moe/moe_front.py.
    // cast_vec * 4 bytes per thread must stay inside the 32B vector-load limit.
    RuntimeCheck(cast_vec == 2 || cast_vec == 4 || cast_vec == 8, "fused_front_epilogue: cast_vec must be 2, 4 or 8");
    RuntimeCheck(latent % cast_vec == 0, "fused_front_epilogue: cast_vec must divide latent");

#define SGL_FRONT_LAUNCH(BS, CV, CF)   \
  LaunchKernel(M, BS, device.unwrap()) \
      .enable_pdl(kUsePDL)(sglang::fused_front_epilogue_kernel<kUsePDL, BS, CV, CF>, params)
#define SGL_FRONT_DISPATCH_CV(BS, CF) \
  do {                                \
    if (cast_vec == 2) {              \
      SGL_FRONT_LAUNCH(BS, 2, CF);    \
    } else if (cast_vec == 4) {       \
      SGL_FRONT_LAUNCH(BS, 4, CF);    \
    } else {                          \
      SGL_FRONT_LAUNCH(BS, 8, CF);    \
    }                                 \
  } while (0)
#define SGL_FRONT_DISPATCH_BS(CF)     \
  do {                                \
    if (block_size == 448) {          \
      SGL_FRONT_DISPATCH_CV(448, CF); \
    } else {                          \
      SGL_FRONT_DISPATCH_CV(224, CF); \
    }                                 \
  } while (0)

    RuntimeCheck(block_size == 224 || block_size == 448, "fused_front_epilogue: block_size must be 224 or 448");
    if (cast_first) {
      SGL_FRONT_DISPATCH_BS(true);
    } else {
      SGL_FRONT_DISPATCH_BS(false);
    }
#undef SGL_FRONT_DISPATCH_BS
#undef SGL_FRONT_DISPATCH_CV
#undef SGL_FRONT_LAUNCH
  }
};
