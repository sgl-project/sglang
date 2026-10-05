/// Four-bit radix-select router for K3 routing on CDNA (gfx942/gfx950).
///
/// One block routes one token. Each thread keeps its slice of the 896 scores in
/// registers, and the block fixes four key bits of the pivot at a time: a round
/// tallies the still-live keys into a 16-bin histogram, accumulates from the top
/// bin down, and keeps the bin the k-th key falls in. The round count therefore
/// follows the key width rather than topk.
///
/// A round's cost is dominated by the vector->scalar->vector unit trip that
/// broadcasting a cross-lane count requires, and that trip is the same price
/// whether the round resolves one bit or four, hence 16 bins. What is left is
/// the per-round work that does not shrink as the experts are spread over more
/// waves; transposing the bin totals onto lanes turns the 16-step walk over
/// them into a 4-step DPP prefix sum plus a ballot.
///
/// Selection contract: experts rank by sigmoid(score) + bias, and a NaN ranking
/// value always ranks below every number, so it can never displace one. The
/// sigmoid itself uses aiter's approximate exp2f + rcpf combination, because
/// expf plus a divide would still disagree with aiter's result at the ULP
/// level. Experts whose key is exactly identical (a strict tie) are separated
/// by kAiterTieLaneRank below, which is exactly the order aiter's wave64
/// traversal reaches them;
///
/// Winners are emitted highest key first, equal keys in the same tie order. So
/// regardless of whether the input ties, this row matches what aiter would
/// produce, expert for expert and column for column -- this kernel and aiter
/// both serve K3, just split by batch size, and a routing that changed with
/// batch size would be the cost of the two disagreeing. The write race during
/// compaction does not affect the final output: it fills the staged row in
/// whatever order wins the race, and the epilogue re-ranks that row afterward.

#pragma once

#ifndef USE_ROCM
#error "route_radix4_hip.cuh targets CDNA; it uses amdgcn DPP and wave64 ballots"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

inline constexpr uint32_t kRadix4NumExperts = 896;
inline constexpr uint32_t kRadix4TopK = 16;
inline constexpr uint32_t kRadix4Block = 256;
inline constexpr uint32_t kRadix4Wave = 64;

/// log2(e): aiter computes exp(-x) as exp2f(-kAiterSigmoidLog2E * x) rather than
/// expf(-x). Matched bit for bit with topk_softmax_kernels_group.cu's C_LOG2E.
inline constexpr float kAiterSigmoidLog2E = 1.44269504088896340736f;

struct RouteRadix4Params {
  const void* __restrict__ scores;
  const void* __restrict__ bias;
  fp32_t* __restrict__ out_w;
  int32_t* __restrict__ out_i;
  uint32_t stride_scores;
  uint32_t stride_out;
  fp32_t routed_scaling_factor;
  bool renormalize;
};

namespace radix4 {

/// Rank of a wave64 lane in aiter router's traversal order. Its tie positions
/// come from a cumulative sum over that traversal order, so two experts whose
/// ranking value is bit-for-bit equal are separated by their position in it.
/// This table was obtained by comparing item-by-item against aiter, and
/// test_moe_route_radix4 pins it back against aiter, so a change on their side
/// surfaces as a test failure rather than a silent divergence.
static __device__ __constant__ uint8_t kAiterTieLaneRank[64] = {
    56, 57, 58, 59, 63, 62, 61, 60, 52, 53, 54, 55, 51, 50, 49, 48, 40, 41, 42, 43, 47, 46,
    45, 44, 36, 37, 38, 39, 35, 34, 33, 32, 24, 25, 26, 27, 31, 30, 29, 28, 20, 21, 22, 23,
    19, 18, 17, 16, 8,  9,  10, 11, 15, 14, 13, 12, 4,  5,  6,  7,  3,  2,  1,  0,
};

/// Where an expert falls in that traversal order: a bijection onto [0, EXPERTS).
/// Experts come in groups of four, and a group lands on one lane; 224 groups
/// are cyclically assigned across 64 lanes, with lanes 0-31 each getting 4
/// groups (banks 0-3) and lanes 32-63 each getting 3 groups (banks 0-2). The
/// rank < 32 branch below takes *3 (for the 3-bank lanes), otherwise *4 (for
/// the 4-bank lanes) -- this relies on the structural fact that
/// kAiterTieLaneRank happens to map lanes 0-31 to rank>=32 and lanes 32-63 to
/// rank<32, rather than branching on the bank count directly.
SGL_DEVICE uint32_t tie_priority(int expert) {
  const int group = expert >> 2;
  const int lane = group & 63;
  const int bank = group >> 6;
  const int rank = static_cast<int>(kAiterTieLaneRank[lane]);
  assert((rank < 32) == (lane >= 32));
  const int group_rank = (rank < 32) ? (rank * 3 + bank) : (96 + (rank - 32) * 4 + bank);
  return static_cast<uint32_t>((group_rank << 2) + (expert & 3));
}

/// How many of the bitmap's members come before p. Every thread runs the same
/// straight line over the same broadcast words, so the walk costs the block no
/// divergence, only NWORDS popcounts.
template <int NWORDS>
SGL_DEVICE int tie_rank(const uint64_t* bits, uint32_t p) {
  const uint32_t w = p >> 6;
  int n = 0;
#pragma unroll
  for (uint32_t j = 0; j < NWORDS; ++j) {
    const uint64_t below = (j < w) ? ~0ull : ((j == w) ? ((1ull << (p & 63)) - 1ull) : 0ull);
    n += __popcll(bits[j] & below);
  }
  return n;
}

SGL_DEVICE float load_score(const bf16_t* p, int i) {
  return __uint_as_float(static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(p)[i]) << 16);
}

SGL_DEVICE float load_score(const fp32_t* p, int i) {
  return p[i];
}

/// Monotonic float -> uint32 map, so unsigned compares order the floats. The map
/// never returns 0: a negative f gives ~u, which is 0 only for the all-ones NaN,
/// and a positive one has the sign bit set. Key 0 is therefore free to mean "NaN,
/// ranks below everything", -inf included.
SGL_DEVICE uint32_t sortable(float f) {
  uint32_t u = __float_as_uint(f);
  // Map -0.0 and +0.0 to the same value.
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// The ranking key: 0 for a NaN, so the expert it belongs to can never displace
/// one whose ranking value is a number.
SGL_DEVICE uint32_t rank_key(float x) {
  return (x == x) ? sortable(x) : 0u;
}

template <int CTRL, int RM, int BM, int N>
SGL_DEVICE void dpp_add_stage(uint32_t (&x)[N]) {
#pragma unroll
  for (int j = 0; j < N; ++j)
    x[j] += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x[j]), CTRL, RM, BM, false));
}

/// Wave-wide inclusive prefix sum over N uint32 at a time: afterwards lane L is
/// the sum of lanes 0..L and lane 63 is the total. The narrower bank masks on the
/// row_shr:4 and row_shr:8 stages switch off exactly the lanes whose source lane
/// falls outside the row, which would have added zero, so masking them or not
/// makes no difference.
template <int N>
SGL_DEVICE void wave_sum_dpp(uint32_t (&x)[N]) {
  dpp_add_stage<0x111, 0xf, 0xf>(x);  // row_shr:1
  dpp_add_stage<0x112, 0xf, 0xf>(x);  // row_shr:2
  dpp_add_stage<0x114, 0xf, 0xe>(x);  // row_shr:4
  dpp_add_stage<0x118, 0xf, 0xc>(x);  // row_shr:8
  dpp_add_stage<0x142, 0xa, 0xf>(x);  // row_bcast:15
  dpp_add_stage<0x143, 0xc, 0xf>(x);  // row_bcast:31
}

template <int CTRL, int RM, int BM>
SGL_DEVICE float dpp_fadd_stage(float x) {
  const int moved = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), CTRL, RM, BM, false);
  return x + __builtin_bit_cast(float, moved);
}

/// Sums v within the wave and leaves the total in out[wid]. The ladder fixes the
/// order the addition happens in, so the same values give the same float on two
/// runs. Not __shfl_xor: that turns into six ds_bpermute round trips through LDS,
/// measured slower.
SGL_DEVICE void stage_wave_sum(float v, int lane, int wid, float* out) {
  v = dpp_fadd_stage<0x111, 0xf, 0xf>(v);  // row_shr:1
  v = dpp_fadd_stage<0x112, 0xf, 0xf>(v);  // row_shr:2
  v = dpp_fadd_stage<0x114, 0xf, 0xe>(v);  // row_shr:4
  v = dpp_fadd_stage<0x118, 0xf, 0xc>(v);  // row_shr:8
  v = dpp_fadd_stage<0x142, 0xa, 0xf>(v);  // row_bcast:15
  v = dpp_fadd_stage<0x143, 0xc, 0xf>(v);  // row_bcast:31
  if (lane == static_cast<int>(kRadix4Wave) - 1) out[wid] = v;
}

}  // namespace radix4

template <typename T, int EXPERTS, int TOPK, int BLOCK>
__device__ void route_radix4_body(const RouteRadix4Params params) {
  constexpr int WAVE = static_cast<int>(kRadix4Wave);
  constexpr int NWAVE = BLOCK / WAVE;
  constexpr int VPT = (EXPERTS + BLOCK - 1) / BLOCK;
  // When the histogram below is tallied, each thread counts the bins of its VPT
  // (values per thread, how many experts one thread handles) experts into one
  // 64-bit register: 16 bins of 4 bits each, and an expert adds 1 to the 4 bits
  // it belongs to.
  constexpr int CHUNK = 15;
  constexpr int NACC = (VPT + CHUNK - 1) / CHUNK;
  static_assert(BLOCK % WAVE == 0, "block must be whole waves");
  static_assert(TOPK <= WAVE, "topk must fit in one lane-indexed row");
  static_assert(VPT <= 32, "alive mask is 32 bits");

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % WAVE;
  const int wid = tid / WAVE;
  const auto* srow = static_cast<const T*>(params.scores) + static_cast<size_t>(token) * params.stride_scores;
  const auto* sbias = static_cast<const T*>(params.bias);

  float sig[VPT];
  uint32_t key[VPT];
  // 896 experts, four contiguous per thread, so threads 0..223 cover the row
  // exactly and a thread's four bf16 values are one 8-byte load. Threads
  // 224..255 stay in the block so the DPP reductions and barriers still see
  // whole waves; they contribute nothing.
  uint32_t valid = (tid < (EXPERTS / VPT)) ? ((1u << VPT) - 1u) : 0u;

  uint32_t or_all = 0u, and_all = 0xffffffffu;
  const bool vec8 = ((reinterpret_cast<uintptr_t>(srow) | reinterpret_cast<uintptr_t>(sbias)) & 7u) == 0u;
  if (tid < EXPERTS / VPT) {
    auto consume = [&](int i, float x, float b) {
      const float g = __builtin_amdgcn_rcpf(1.0f + exp2f(-kAiterSigmoidLog2E * x));
      sig[i] = g;
      key[i] = radix4::rank_key(g + b);
      or_all |= key[i];
      and_all &= key[i];
    };
    if constexpr (std::is_same_v<T, bf16_t>) {
      if (vec8) {
        const uint2 sv = *reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(srow) + tid * VPT);
        const uint2 bv = *reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(sbias) + tid * VPT);
        const uint32_t sp[2] = {sv.x, sv.y};
        const uint32_t bp[2] = {bv.x, bv.y};
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          consume(2 * p, __uint_as_float((sp[p] & 0xffffu) << 16), __uint_as_float((bp[p] & 0xffffu) << 16));
          consume(2 * p + 1, __uint_as_float(sp[p] & 0xffff0000u), __uint_as_float(bp[p] & 0xffff0000u));
        }
      } else {
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
          const int e = tid * VPT + i;
          consume(i, radix4::load_score(srow, e), radix4::load_score(sbias, e));
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < VPT; ++i) {
        const int e = tid * VPT + i;
        consume(i, radix4::load_score(srow, e), radix4::load_score(sbias, e));
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      sig[i] = 0.0f;
      key[i] = 0u;
    }
  }
  uint32_t alive = valid;

  // One bit per expert, indexed by tie priority rather than by id.
  constexpr int TIE_WORDS = (EXPERTS + 63) / 64;

  __shared__ uint32_t s_hist[2][NWAVE][8];
  __shared__ uint32_t s_pre[2][NWAVE];
  __shared__ uint64_t s_tie[TIE_WORDS];
  __shared__ float s_w[TOPK];
  __shared__ float s_wsum[NWAVE];
  __shared__ int s_id[TOPK];
  __shared__ uint32_t s_key[TOPK];
  __shared__ int s_cnt;

#pragma unroll
  for (int s = 32; s > 0; s >>= 1) {
    or_all |= __shfl_xor(or_all, s, WAVE);
    and_all &= __shfl_xor(and_all, s, WAVE);
  }
  if (lane == 0) {
    s_pre[0][wid] = or_all;
    s_pre[1][wid] = and_all;
  }
  __syncthreads();
#pragma unroll
  for (int w = 0; w < NWAVE; ++w) {
    or_all |= s_pre[0][w];
    and_all &= s_pre[1][w];
  }

  // The XOR marks the bits the keys disagree on (a disagreeing bit is 1 in the
  // OR and 0 in the AND); the highest one is where the search starts. Above it
  // every key is the same, and that value comes out of and_all as the initial
  // pivot.
  const uint32_t diff = or_all ^ and_all;
  const int start = diff ? (31 - __clz(diff)) : -1;
  uint32_t pivot = (start >= 31) ? 0u : (and_all & ~((1u << (start + 1)) - 1u));

  int need = TOPK;  // how many still have to be picked out of the live set
  // lowest bit the pivot is resolved down to
  int bend = 0;
  bool capped = (start < 0);

  // Main loop: fixes the TOPK-th largest key (the pivot) without sorting. A round
  // takes 4 bits as the bin index, tallies the histogram and accumulates from the
  // top bin down; the bin the need-th key falls in fixes those 4 pivot bits, the
  // higher bins are in for good and come off need, and alive narrows to that bin.
  // Survivors exactly equal to need exit early; still more than need at the
  // lowest bit sets capped.
#pragma unroll 1
  for (int b = (start < 0) ? -4 : (start >> 2) << 2; b >= 0; b -= 4) {
    // Two buffers used alternately, so a round's writes only have to wait for the
    // reads of the round before last.
    const int buf = (b >> 2) & 1;

    uint64_t h[NACC];
#pragma unroll
    for (int a = 0; a < NACC; ++a)
      h[a] = 0ull;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const uint32_t bin = (key[i] >> b) & 15u;
      h[i / CHUNK] += static_cast<uint64_t>((alive >> i) & 1u) << (4 * bin);
    }

    // Spread the 16 four-bit counts over 8 uint32, two bins each.
    uint32_t p[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t lo = 0u, hi = 0u;
#pragma unroll
      for (int a = 0; a < NACC; ++a) {
        lo += static_cast<uint32_t>((h[a] >> (8 * j)) & 0xfull);
        hi += static_cast<uint32_t>((h[a] >> (8 * j + 4)) & 0xfull);
      }
      p[j] = lo | (hi << 16);
    }
    radix4::wave_sum_dpp(p);
    if (lane == WAVE - 1) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        s_hist[buf][wid][j] = p[j];
    }
    __syncthreads();

    // One LDS read puts each of the 16 bins on a lane of its own, which turns the
    // walk over the bins into a prefix sum across lanes. Lane L takes bin 15 - L:
    // with the bins reversed, "how many keys are in this bin or a higher one" is
    // exactly the direction row_shr adds in.
    int c = 0;
    if (lane < 16) {
      const int d = 15 - lane;
#pragma unroll
      for (int w = 0; w < NWAVE; ++w)
        c += static_cast<int>((s_hist[buf][w][d >> 1] >> ((d & 1) * 16)) & 0xffffu);
    }

    // A prefix sum of c across lanes, so cum is the number of keys in this lane's
    // bin and in every higher bin.
    int cum = c;
    cum += __builtin_amdgcn_update_dpp(0, cum, 0x111, 0xf, 0xf, false);  // row_shr:1
    cum += __builtin_amdgcn_update_dpp(0, cum, 0x112, 0xf, 0xf, false);  // row_shr:2
    cum += __builtin_amdgcn_update_dpp(0, cum, 0x114, 0xf, 0xf, false);  // row_shr:4
    cum += __builtin_amdgcn_update_dpp(0, cum, 0x118, 0xf, 0xf, false);  // row_shr:8

    // cum rises monotonically with L, so the predicate turns on once and stays
    // on; its lowest lane is the bin holding the k-th key.
    const unsigned long long mk = __ballot(lane < 16 && cum >= need);
    const int L0 = __ffsll(mk) - 1;
    const int sel = 15 - L0;
    const int nsel = __builtin_amdgcn_readlane(c, L0);
    const int above = __builtin_amdgcn_readlane(cum, L0) - nsel;

    need -= above;
    pivot |= static_cast<uint32_t>(sel) << b;
    bend = b;
#pragma unroll
    for (int i = 0; i < VPT; ++i)
      if (((key[i] >> b) & 15u) != static_cast<uint32_t>(sel)) alive &= ~(1u << i);

    // Every survivor is a winner: the remaining bits cannot change the set. The
    // test is uniform across the block, since every thread scanned the same LDS
    // totals.
    if (nsel == need) break;
    if (b == 0) capped = true;
  }

  const uint32_t pmask = ~((1u << bend) - 1u);
  // The renorm divisor is accumulated while the winners are picked instead of
  // read back off the staged row: the order that row gets filled in can differ
  // from run to run, and the order of the float additions with it, whereas
  // reducing across threads always adds in lane order.
  float wsum = 0.0f;
  if (!capped) {
    // The survivors exactly fill the quota, so a key wins as soon as it reaches
    // the pivot prefix. Winners span waves, so ballot cannot number them; an LDS
    // bump counter can.
    if (tid == 0) s_cnt = 0;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (((valid >> i) & 1u) && (key[i] & pmask) >= pivot) {
        const int pos = atomicAdd(&s_cnt, 1);
        if (pos < TOPK) {
          s_w[pos] = sig[i];
          s_id[pos] = tid * VPT + i;
          s_key[pos] = key[i];
          wsum += sig[i];
        }
      }
    }
    radix4::stage_wave_sum(wsum, lane, wid, s_wsum);
    __syncthreads();
  } else {
    // Every pivot bit is fixed and the survivors still outnumber the quota, so
    // what is left are keys equal bit for bit and only the tie rule separates
    // them: `need` of them get in, the ones aiter's traversal reaches first.
    // Marking the survivors in a bitmap indexed by that traversal turns "how
    // many come before me" into a popcount, which costs the block no
    // divergence and no scan.
    for (int j = tid; j < TIE_WORDS; j += BLOCK)
      s_tie[j] = 0ull;
    if (tid == 0) s_cnt = 0;
    __syncthreads();

    uint32_t eqm = 0u;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (((valid >> i) & 1u) && (key[i] & pmask) == pivot) {
        eqm |= 1u << i;
        const uint32_t p = radix4::tie_priority(tid * VPT + i);
        atomicOr(&s_tie[p >> 6], 1ull << (p & 63));
      }
    }
    __syncthreads();

    // The outright winners are TOPK - need of them, so the ties owe exactly the
    // `need` the loop stopped short of and the row comes out full.
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (!((valid >> i) & 1u)) continue;
      const int e = tid * VPT + i;
      const bool tied = ((eqm >> i) & 1u) != 0u;
      const bool taken =
          (key[i] & pmask) > pivot || (tied && radix4::tie_rank<TIE_WORDS>(s_tie, radix4::tie_priority(e)) < need);
      if (taken) {
        const int pos = atomicAdd(&s_cnt, 1);
        if (pos < TOPK) {
          s_w[pos] = sig[i];
          s_id[pos] = e;
          s_key[pos] = key[i];
          wsum += sig[i];
        }
      }
    }
    radix4::stage_wave_sum(wsum, lane, wid, s_wsum);
    __syncthreads();
  }

  if (tid < TOPK) {
    // Where the compaction put a winner is a race; where it is emitted is not.
    // Tie priorities are distinct, so (key desc, priority asc) is a strict total
    // order and counting the winners that outrank this one lands each of them on
    // a position of its own. The whole staged row sits in lanes 0..TOPK-1 of this
    // wave, so the walk over it reads the other lanes' registers instead of LDS:
    // readlane is a scalar op and the lane index is a constant of the unrolled
    // loop, which leaves the vector unit with just the compares.
    const uint32_t k = s_key[tid];
    const int id = s_id[tid];
    const auto p = static_cast<int>(radix4::tie_priority(id));
    int rank = 0;
#pragma unroll
    for (int q = 0; q < TOPK; ++q) {
      const auto kq = static_cast<uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(k), q));
      const int pq = __builtin_amdgcn_readlane(p, q);
      rank += (kq > k || (kq == k && pq < p)) ? 1 : 0;
    }

    float scale = params.routed_scaling_factor;
    if (params.renormalize) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < NWAVE; ++w)
        sum += s_wsum[w];
      // Every sigmoid underflows to zero on a row of saturated scores, and a row
      // of NaN sums to NaN; neither may turn a finite weight into an inf.
      scale /= (sum > 0.0f) ? sum : 1.0f;
    }
    const size_t o = static_cast<size_t>(token) * params.stride_out + rank;
    params.out_w[o] = s_w[tid] * scale;
    params.out_i[o] = id;
  }
}

template <typename T, int EXPERTS, int TOPK, int BLOCK>
__global__ __launch_bounds__(BLOCK) void route_radix4_kernel(__grid_constant__ const RouteRadix4Params params) {
  route_radix4_body<T, EXPERTS, TOPK, BLOCK>(params);
}

// One MXFP4 pass over the token row. It does not read the route outputs, so it
// occupies the other block of the same launch and the launch takes as long as
// the slower block. Group 32, ue8m0 round-up of amax/6, e2m1 with the low
// nibble first. 3584 / 32 = 112 groups.
inline constexpr int kMxfp4Hidden = 3584;
inline constexpr int kMxfp4Group = 32;
inline constexpr int kMxfp4Groups = kMxfp4Hidden / kMxfp4Group;

__device__ float mxfp4_e8m0_scale(float amax) {
  const uint32_t u32 = __float_as_uint(amax * (1.0f / 6.0f));
  uint32_t exponent = (u32 >> 23) & 0xffu;
  if (exponent < 0xffu && (u32 & 0x7fffffu)) exponent += 1u;
  return __uint_as_float(exponent << 23);
}

__device__ __forceinline__ uint32_t bf16_pair_absmax(uint32_t x) {
  return max(x & 0x7fffu, (x >> 16) & 0x7fffu);
}

__device__ __forceinline__ uint32_t cvt_fp4_word(uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3, float dq) {
#if defined(__gfx950__)
  // The scale operand is the dequant scale. The instruction divides by it.
  auto as_bf16x2 = [](uint32_t bits) {
    __hip_bfloat162 v;
    __builtin_memcpy(&v, &bits, sizeof(v));
    return v;
  };
  uint32_t pk = 0;
  pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(pk, as_bf16x2(p0), dq, 0);
  pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(pk, as_bf16x2(p1), dq, 1);
  pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(pk, as_bf16x2(p2), dq, 2);
  pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(pk, as_bf16x2(p3), dq, 3);
  return pk;
#else
  // gfx942 has no packed fp4 convert. Round onto the e2m1 grid here so this
  // kernel does not include the DeepSeek-V4 fp4 header.
  auto e2m1 = [](float x) -> uint32_t {
    constexpr float kMax = 6.0f;
    const float mag = fminf(fabsf(x), kMax);
    const float step = mag < 2.0f ? 0.5f : (mag < 4.0f ? 1.0f : 2.0f);
    const float q = rintf(mag / step) * step;
    const uint32_t idx = q < 2.0f ? static_cast<uint32_t>(q * 2.0f)
                                  : (q < 4.0f ? static_cast<uint32_t>(q) + 2u : static_cast<uint32_t>(q * 0.5f) + 4u);
    return idx | (__float_as_uint(x) >> 31 << 3);
  };
  auto nib = [&](uint32_t pair) {
    const float a = __uint_as_float((pair & 0xffffu) << 16) * __builtin_amdgcn_rcpf(dq);
    const float b = __uint_as_float(pair & 0xffff0000u) * __builtin_amdgcn_rcpf(dq);
    return static_cast<uint8_t>(e2m1(a) | (e2m1(b) << 4));
  };
  return static_cast<uint32_t>(nib(p0)) | (static_cast<uint32_t>(nib(p1)) << 8) |
         (static_cast<uint32_t>(nib(p2)) << 16) | (static_cast<uint32_t>(nib(p3)) << 24);
#endif
}

__device__ void quant_mxfp4_token(const bf16_t* row, uint8_t* dst, uint8_t* scale) {
  const int g = static_cast<int>(threadIdx.x);
  if (g >= kMxfp4Groups) return;
  const uint32_t* v = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(row) + g * kMxfp4Group);
  uint32_t p0 = v[0], p1 = v[1], p2 = v[2], p3 = v[3];
  uint32_t p4 = v[4], p5 = v[5], p6 = v[6], p7 = v[7];
  uint32_t p8 = v[8], p9 = v[9], p10 = v[10], p11 = v[11];
  uint32_t p12 = v[12], p13 = v[13], p14 = v[14], p15 = v[15];
  uint32_t am = bf16_pair_absmax(p0);
  am = max(am, bf16_pair_absmax(p1));
  am = max(am, bf16_pair_absmax(p2));
  am = max(am, bf16_pair_absmax(p3));
  am = max(am, bf16_pair_absmax(p4));
  am = max(am, bf16_pair_absmax(p5));
  am = max(am, bf16_pair_absmax(p6));
  am = max(am, bf16_pair_absmax(p7));
  am = max(am, bf16_pair_absmax(p8));
  am = max(am, bf16_pair_absmax(p9));
  am = max(am, bf16_pair_absmax(p10));
  am = max(am, bf16_pair_absmax(p11));
  am = max(am, bf16_pair_absmax(p12));
  am = max(am, bf16_pair_absmax(p13));
  am = max(am, bf16_pair_absmax(p14));
  am = max(am, bf16_pair_absmax(p15));
  // aiter's MXFP4 kernel floors a group amax at 1e-10, including an all-zero
  // group, then applies the same round-up. A special scale of 127 would not
  // match that byte.
  const float amax = fmaxf(__uint_as_float(am << 16), 1e-10f);
  const float dq = mxfp4_e8m0_scale(amax);
  scale[g] = static_cast<uint8_t>(__float_as_uint(dq) >> 23);
  uint32_t* out = reinterpret_cast<uint32_t*>(dst + g * (kMxfp4Group / 2));
  out[0] = cvt_fp4_word(p0, p1, p2, p3, dq);
  out[1] = cvt_fp4_word(p4, p5, p6, p7, dq);
  out[2] = cvt_fp4_word(p8, p9, p10, p11, dq);
  out[3] = cvt_fp4_word(p12, p13, p14, p15, dq);
}

struct RouteQuantHipParams {
  RouteRadix4Params route;
  const bf16_t* x;
  uint32_t x_stride;
  uint8_t* out_fp4;
  uint8_t* out_scale;
  uint32_t num_tokens;
};

template <int BLOCK>
__global__
__launch_bounds__(BLOCK) void route_quant_fused_hip_kernel(__grid_constant__ const RouteQuantHipParams params) {
  if (blockIdx.x < params.num_tokens) {
    route_radix4_body<bf16_t, 896, 16, BLOCK>(params.route);
  } else {
    const uint32_t token = blockIdx.x - params.num_tokens;
    quant_mxfp4_token(
        params.x + static_cast<size_t>(token) * params.x_stride,
        params.out_fp4 + static_cast<size_t>(token) * (kMxfp4Hidden / 2),
        params.out_scale + static_cast<size_t>(token) * kMxfp4Groups);
  }
}

struct RouteRadix4Kernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto N_ = SymbolicSize{"num_experts"};
    auto K_ = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    auto score_dtype = SymbolicDType{};
    TensorMatcher({M_, N_})
        .with_dtype<bf16_t, fp32_t>(score_dtype)
        .with_device(device_)
        .with_strides({-1, 1})
        .verify(scores);
    // Rebinding the same symbolic dtype makes the bias track the scores.
    TensorMatcher({N_}).with_dtype<bf16_t, fp32_t>(score_dtype).with_device(device_).verify(bias);
    TensorMatcher({M_, K_}).with_dtype<fp32_t>().with_device(device_).verify(out_w);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_device(device_).verify(out_i);

    RuntimeCheck(
        N_.unwrap() == kRadix4NumExperts && K_.unwrap() == kRadix4TopK && topk == kRadix4TopK,
        "route_radix4 is specialized for N=896, K=16");

    const auto M = static_cast<uint32_t>(M_.unwrap());
    if (M == 0) return;

    const auto params = RouteRadix4Params{
        .scores = scores.data_ptr(),
        .bias = bias.data_ptr(),
        .out_w = static_cast<fp32_t*>(out_w.data_ptr()),
        .out_i = static_cast<int32_t*>(out_i.data_ptr()),
        .stride_scores = static_cast<uint32_t>(scores.stride(0)),
        .stride_out = static_cast<uint32_t>(out_w.stride(0)),
        .routed_scaling_factor = static_cast<fp32_t>(routed_scaling_factor),
        .renormalize = renormalize,
    };

    constexpr auto kExperts = static_cast<int>(kRadix4NumExperts);
    constexpr auto kTopK = static_cast<int>(kRadix4TopK);
    constexpr auto kBlock = static_cast<int>(kRadix4Block);
    const auto device = device_.unwrap();
    if (score_dtype.is_type<bf16_t>()) {
      LaunchKernel(M, kBlock, device)(route_radix4_kernel<bf16_t, kExperts, kTopK, kBlock>, params);
    } else {
      LaunchKernel(M, kBlock, device)(route_radix4_kernel<fp32_t, kExperts, kTopK, kBlock>, params);
    }
  }
};

struct RouteQuantFusedHipKernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      const tvm::ffi::TensorView out_fp4,
      const tvm::ffi::TensorView out_scale,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({M_, kRadix4NumExperts})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, 1})
        .verify(scores);
    TensorMatcher({kRadix4NumExperts}).with_dtype<bf16_t>().with_device(device_).verify(bias);
    TensorMatcher({M_, kMxfp4Hidden}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(x);
    TensorMatcher({M_, kRadix4TopK}).with_dtype<fp32_t>().with_device(device_).verify(out_w);
    TensorMatcher({M_, kRadix4TopK}).with_dtype<int32_t>().with_device(device_).verify(out_i);
    TensorMatcher({M_, kMxfp4Hidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(out_fp4);
    TensorMatcher({M_, kMxfp4Groups}).with_dtype<uint8_t>().with_device(device_).verify(out_scale);

    const auto M = static_cast<uint32_t>(M_.unwrap());
    RuntimeCheck(topk == kRadix4TopK && M > 0 && M <= 1024, "fused route covers K3 only");
    const auto stride_x = static_cast<uint32_t>(x.stride(0));
    RuntimeCheck((stride_x % 2u) == 0u, "MXFP4 row must be 4-byte aligned");

    RouteQuantHipParams params{};
    params.route.scores = static_cast<const bf16_t*>(scores.data_ptr());
    params.route.bias = static_cast<const bf16_t*>(bias.data_ptr());
    params.route.out_w = static_cast<float*>(out_w.data_ptr());
    params.route.out_i = static_cast<int32_t*>(out_i.data_ptr());
    params.route.stride_scores = static_cast<uint32_t>(scores.stride(0));
    params.route.stride_out = static_cast<uint32_t>(out_w.stride(0));
    params.route.renormalize = renormalize;
    params.route.routed_scaling_factor = static_cast<float>(routed_scaling_factor);
    params.x = static_cast<const bf16_t*>(x.data_ptr());
    params.x_stride = stride_x;
    params.out_fp4 = static_cast<uint8_t*>(out_fp4.data_ptr());
    params.out_scale = static_cast<uint8_t*>(out_scale.data_ptr());
    params.num_tokens = M;

    constexpr auto kBlock = static_cast<int>(kRadix4Block);
    LaunchKernel(2 * M, kBlock, device_.unwrap())(route_quant_fused_hip_kernel<kBlock>, params);
  }
};

}  // namespace sglang
