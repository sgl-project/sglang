// Kimi-K3 router top-k of one token on one wave (ROCm, wave64), shared by the fused route sort
// (route_sort.cuh) and the column-parallel MoE front's gather (comm/front_gather_hip.cuh).
//
// Scores follow aiter's topk_reg_kernel: sig = rcp(1 + 2^(-log2e * x)), the selection key is
// sig + bias, the weight is the pre-bias sig, renormalized in descending order; exact score ties
// go to the lower expert id.

#pragma once

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <type_traits>

namespace sglang {

inline constexpr int kRouteTopkWave = 64;
inline constexpr int kRouteTopkMaxCand = 128;

// Intra-wave LDS exchange: the wave runs in lockstep and its LDS ops retire in
// order, so a compiler barrier is enough (as aiter's topk_reg_kernel does).
SGL_DEVICE void wave_lds_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// Larger key = larger score; equal scores order the lower expert id first.
SGL_DEVICE uint64_t route_key(float score, int expert) {
  uint32_t u = __float_as_uint(score);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return (static_cast<uint64_t>(u) << 32) | static_cast<uint32_t>(0xffff - expert);
}

SGL_DEVICE uint32_t route_key_u32(float score) {
  const uint32_t u = __float_as_uint(score);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

SGL_DEVICE float route_key_u32_to_float(uint32_t key) {
  return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}

SGL_DEVICE int route_key_expert(uint64_t key) {
  return 0xffff - static_cast<int>(key & 0xffffu);
}

// The calling wave selects one token's routes. row: the token's logits as bf16 pairs (global or
// LDS); bias: [kExperts] TBias; ckey / csig: this wave's kRouteTopkMaxCand-entry LDS scratch;
// ids / tsig: kTopK-entry LDS. Lane l < kTopK receives the l-th route's expert and weight.
template <int kExperts, int kTopK, typename TBias>
SGL_DEVICE void route_topk_wave(
    const bf16x2_t* row,
    const void* bias,
    float routed_scaling_factor,
    bool renormalize,
    uint64_t* ckey,
    float* csig,
    int16_t* ids,
    float* tsig,
    int lane,
    int& out_id,
    float& out_w) {
  static_assert(kExperts % (2 * kRouteTopkWave) == 0 && kTopK <= kRouteTopkWave);
  constexpr int kPairs = kExperts / (2 * kRouteTopkWave);
  constexpr float kLog2e = 1.4426950408889634f;
  const uint64_t lanemask_lt = (1ull << lane) - 1ull;

  // lane owns expert pairs (lane + 64 j) * 2 + {0, 1}
  float score[kPairs][2];
  float sig[kPairs][2];
#pragma unroll
  for (int j = 0; j < kPairs; ++j) {
    const int e2 = lane + j * kRouteTopkWave;
    const auto [x0, x1] = device::cast<fp32x2_t>(row[e2]);
    float b0, b1;
    if constexpr (std::is_same_v<TBias, float>) {
      b0 = static_cast<const float*>(bias)[2 * e2];
      b1 = static_cast<const float*>(bias)[2 * e2 + 1];
    } else {
      const auto [c0, c1] = device::cast<fp32x2_t>(static_cast<const bf16x2_t*>(bias)[e2]);
      b0 = c0;
      b1 = c1;
    }
    sig[j][0] = __builtin_amdgcn_rcpf(1.0f + exp2f(-kLog2e * x0));
    sig[j][1] = __builtin_amdgcn_rcpf(1.0f + exp2f(-kLog2e * x1));
    score[j][0] = sig[j][0] + b0;
    score[j][1] = sig[j][1] + b1;
  }

  // Pivot = k-th largest lane max: every top-k score is >= it, so the
  // candidates at or above it are few (typically < 2k) and are ranked by
  // counting. A radix select over ballot counts finds it in 32 SALU-cheap
  // steps with no LDS round trip; NaN maxima order above every float and
  // send the token to the argmax fallback below.
  float lmax = -INFINITY;
#pragma unroll
  for (int j = 0; j < kPairs; ++j)
    lmax = fmaxf(lmax, fmaxf(score[j][0], score[j][1]));
  const uint32_t lkey = route_key_u32(lmax);
  uint32_t pkey = 0;
#pragma unroll
  for (int bit = 31; bit >= 0; --bit) {
    const uint32_t probe = pkey | (1u << bit);
    if (__popcll(__ballot(lkey >= probe)) >= kTopK) pkey = probe;
  }
  const float pivot = route_key_u32_to_float(pkey);

  int n_cand = 0;
#pragma unroll
  for (int j = 0; j < kPairs; ++j) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const bool is_cand = score[j][i] >= pivot;
      const uint64_t ballot = __ballot(is_cand);
      const int c = n_cand + __popcll(ballot & lanemask_lt);
      if (is_cand && c < kRouteTopkMaxCand) {
        ckey[c] = route_key(score[j][i], (lane + j * kRouteTopkWave) * 2 + i);
        csig[c] = sig[j][i];
      }
      n_cand += __popcll(ballot);
    }
  }
  wave_lds_sync();

  if (n_cand >= kTopK && n_cand <= kRouteTopkMaxCand) {
    for (int c = lane; c < n_cand; c += kRouteTopkWave) {
      const uint64_t key = ckey[c];
      int rank = 0;
      int d = 0;
      for (; d + 2 <= n_cand; d += 2) {
        const ulonglong2 kk = reinterpret_cast<const ulonglong2*>(ckey)[d / 2];
        rank += (kk.x > key ? 1 : 0) + (kk.y > key ? 1 : 0);
      }
      if (d < n_cand) rank += ckey[d] > key ? 1 : 0;
      if (rank < kTopK) {
        ids[rank] = static_cast<int16_t>(route_key_expert(key));
        tsig[rank] = csig[c];
      }
    }
  } else {
    // Plateau wider than the candidate buffer (or NaN scores): k rounds of
    // wave argmax over the untaken experts. Keys order every float, so each
    // round yields a valid, distinct expert id.
    bool taken[kPairs][2] = {};
    for (int k = 0; k < kTopK; ++k) {
      uint64_t best = 0;
      float best_sig = 0.0f;
#pragma unroll
      for (int j = 0; j < kPairs; ++j) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const uint64_t key = route_key(score[j][i], (lane + j * kRouteTopkWave) * 2 + i);
          if (!taken[j][i] && key > best) {
            best = key;
            best_sig = sig[j][i];
          }
        }
      }
#pragma unroll
      for (int offset = kRouteTopkWave / 2; offset > 0; offset >>= 1) {
        const uint64_t o_best = __shfl_xor(best, offset, kRouteTopkWave);
        const float o_sig = __shfl_xor(best_sig, offset, kRouteTopkWave);
        if (o_best > best) {
          best = o_best;
          best_sig = o_sig;
        }
      }
      const int best_id = route_key_expert(best);
#pragma unroll
      for (int j = 0; j < kPairs; ++j) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          if ((lane + j * kRouteTopkWave) * 2 + i == best_id) taken[j][i] = true;
        }
      }
      if (lane == k) {
        ids[k] = static_cast<int16_t>(best_id);
        tsig[k] = best_sig;
      }
    }
  }
  wave_lds_sync();

  if (lane < kTopK) {
    // Serial descending-order sum, as topk_reg_kernel renormalizes.
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
      total += tsig[k];
    float scale = routed_scaling_factor;
    if (renormalize) scale = total > 0.0f ? routed_scaling_factor / total : 0.0f;
    out_id = ids[lane];
    out_w = tsig[lane] * scale;
  }
}

}  // namespace sglang
