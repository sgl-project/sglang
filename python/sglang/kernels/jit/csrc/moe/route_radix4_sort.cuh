/// BM=16 MoE sort used by the fused radix4+sort kernel.
///
/// Mirrors aiter's `moe_sort_quant.cuh` sort-only path (count, padded
/// prefix, place, pad) so GEMM1/GEMM2 metadata matches the two-kernel
/// decode path bit for bit. Kept in this TU so the SGLang JIT does not
/// include AITER headers.
///
/// The phases are split by what they depend on rather than by aiter's order.
/// Padding a slot does not depend on the routing result: the pad token, the pad
/// weight and the buffer bound are all known before a single expert is picked.
/// So every routing CTA pads a slice of the buffer before it reports in, and
/// the sort tail is left with count, prefix and placement only. Placement
/// then overwrites exactly the slots a winner lands on, which is why aiter's
/// per-expert gap fill has no counterpart here -- the gaps were never written.
///
/// Two tails share that layout. Up to a few routes per thread, the last CTA to
/// arrive sorts everything (sort_subkernel). Past that one CTA's placement
/// grows with M, so sort_tail_stable spreads it over the last ceil(M / 16)
/// arrivals instead, each placing one token chunk.

#pragma once

#include <cstdint>

namespace sglang::radix4_sort {

/// Most tokens one sort_tail_stable CTA places: one s_mask bit each.
inline constexpr int kTailMaxChunkTokens = 32;

SGL_DEVICE int round_up_mpb(int x, int mpb) {
  return (x + mpb - 1) / mpb * mpb;
}

/// One DPP move: a lane reads the lane CTRL selects, or `identity` when the
/// row/bank masks switch it off or its source falls outside the row.
template <int CTRL, int ROW_MASK, int BANK_MASK>
SGL_DEVICE uint32_t dpp_move(uint32_t x, uint32_t identity) {
  return static_cast<uint32_t>(
      __builtin_amdgcn_update_dpp(static_cast<int>(identity), static_cast<int>(x), CTRL, ROW_MASK, BANK_MASK, false));
}

/// Inclusive wave64 scan under `op`, whose identity is `identity`; lane 63 ends
/// with the whole-wave result. Unlike __shfl_xor it makes no LDS round trips.
/// The narrower bank masks on row_shr:4 and row_shr:8 only switch off lanes
/// whose source is outside the row, which would have read the identity anyway.
template <typename Op>
SGL_DEVICE uint32_t wave_scan_dpp(uint32_t x, uint32_t identity, Op op) {
  x = op(x, dpp_move<0x111, 0xf, 0xf>(x, identity));  // row_shr:1
  x = op(x, dpp_move<0x112, 0xf, 0xf>(x, identity));  // row_shr:2
  x = op(x, dpp_move<0x114, 0xf, 0xe>(x, identity));  // row_shr:4
  x = op(x, dpp_move<0x118, 0xf, 0xc>(x, identity));  // row_shr:8
  x = op(x, dpp_move<0x142, 0xa, 0xf>(x, identity));  // row_bcast:15
  x = op(x, dpp_move<0x143, 0xc, 0xf>(x, identity));  // row_bcast:31
  return x;
}

/// Pad one CTA's slice of the sort output. `capacity` is the length the host
/// reserved, which bounds the padded length from above (see _sorted_capacity), so
/// padding all of it covers every slot the GEMM can reach and leaves nothing
/// uninitialized behind the padded length either. A whole number of BM blocks
/// fits in the capacity, hence a whole number of int4, so the slice needs no
/// scalar tail.
///
/// kWriteThrough stores at the coherent point (agent scope, 8 bytes at a time).
/// The distributed tail needs it: a tail CTA on another XCD may place a route
/// over any of these slots, and a pad left dirty in this XCD's L2 could be
/// written back over that route later.
template <int THREADS_PER_CTA, bool kWriteThrough = false>
SGL_DEVICE void pad_sorted_slice(
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ m_indices,
    float* __restrict__ sorted_weights,
    int capacity,
    int pad_val,
    int part,
    int nparts) {
  if constexpr (kWriteThrough) {
    const int nvec = capacity >> 1;
    const int per_part = (nvec + nparts - 1) / nparts;
    const int begin = part * per_part;
    const int end = (begin + per_part < nvec) ? (begin + per_part) : nvec;
    const auto pad = static_cast<uint32_t>(pad_val);
    const uint64_t pad2 = (static_cast<uint64_t>(pad) << 32) | pad;
    auto* tok2 = reinterpret_cast<uint64_t*>(sorted_token_ids);
    auto* idx2 = reinterpret_cast<uint64_t*>(m_indices);
    auto* w2 = reinterpret_cast<uint64_t*>(sorted_weights);
    for (int i = begin + static_cast<int>(threadIdx.x); i < end; i += THREADS_PER_CTA) {
      __hip_atomic_store(&tok2[i], pad2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&idx2[i], pad2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&w2[i], uint64_t{0}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  } else {
    const int nvec = capacity >> 2;
    const int per_part = (nvec + nparts - 1) / nparts;
    const int begin = part * per_part;
    const int end = (begin + per_part < nvec) ? (begin + per_part) : nvec;
    const int4 pad4 = make_int4(pad_val, pad_val, pad_val, pad_val);
    const int4 zero4 = make_int4(0, 0, 0, 0);
    auto* tok4 = reinterpret_cast<int4*>(sorted_token_ids);
    auto* idx4 = reinterpret_cast<int4*>(m_indices);
    auto* w4 = reinterpret_cast<int4*>(sorted_weights);
    for (int i = begin + static_cast<int>(threadIdx.x); i < end; i += THREADS_PER_CTA) {
      tok4[i] = pad4;
      idx4[i] = pad4;
      w4[i] = zero4;
    }
  }
}

/// Every CTA clears its histogram before the arrival barrier, so the leader
/// does not pay for it in the serial tail.
template <int NUM_EXPERTS, int THREADS_PER_CTA>
SGL_DEVICE void reset_counts(int* __restrict__ count) {
  // Contiguous per thread rather than strided, so the four zeros are one LDS
  // write, and so the thread clears the same experts it will later scan.
  constexpr int ITEMS_PER_THREAD = (NUM_EXPERTS + THREADS_PER_CTA - 1) / THREADS_PER_CTA;
  const int base = static_cast<int>(threadIdx.x) * ITEMS_PER_THREAD;
  if (base + ITEMS_PER_THREAD <= NUM_EXPERTS) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
      count[base + i] = 0;
  } else {
    for (int e = base; e < NUM_EXPERTS; ++e)
      count[e] = 0;
  }
}

template <int NUM_EXPERTS, int THREADS_PER_CTA>
inline constexpr int kExpertsPerThread = (NUM_EXPERTS + THREADS_PER_CTA - 1) / THREADS_PER_CTA;

/// Padded expert prefix, with thread t holding the counts of experts
/// [t * ITEMS, t * ITEMS + ITEMS). Leaves cumsum[e] = first slot of expert e and,
/// when sorted_expert_ids is given, writes the expert of every block.
template <int NUM_EXPERTS, int THREADS_PER_CTA, int SORT_MPB>
SGL_DEVICE void parallel_cumsum(
    const int (&count)[kExpertsPerThread<NUM_EXPERTS, THREADS_PER_CTA>],
    int* __restrict__ cumsum,
    int32_t* __restrict__ sorted_expert_ids) {
  // Each of the 256 lanes scans a contiguous expert group, then the block scans
  // group totals; contiguous groups keep the expert-order prefix GEMM consumes.
  constexpr int WAVE = 64;
  constexpr int N_WAVES = THREADS_PER_CTA / WAVE;
  constexpr int ITEMS_PER_THREAD = kExpertsPerThread<NUM_EXPERTS, THREADS_PER_CTA>;
  static_assert(THREADS_PER_CTA % WAVE == 0, "sort CTA must contain whole waves");
  static_assert(N_WAVES <= WAVE, "wave 0 must be able to scan wave totals");

  const int tid = threadIdx.x;
  const int lane = tid & (WAVE - 1);
  const int wave = tid / WAVE;

  int local[ITEMS_PER_THREAD];
  int thread_total = 0;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    const int e = tid * ITEMS_PER_THREAD + i;
    const int v = (e < NUM_EXPERTS) ? round_up_mpb(count[i], SORT_MPB) : 0;
    thread_total += v;
    local[i] = thread_total;
  }

  const int wave_inclusive = static_cast<int>(
      wave_scan_dpp(static_cast<uint32_t>(thread_total), 0u, [](uint32_t a, uint32_t b) { return a + b; }));

  __shared__ int wave_totals[N_WAVES];
  if (lane == WAVE - 1) wave_totals[wave] = wave_inclusive;
  __syncthreads();

  // Only N_WAVES totals: a serial sum is cheaper than a second scan + barrier.
  int wave_prefix = 0;
#pragma unroll
  for (int w = 0; w < N_WAVES; ++w)
    wave_prefix += (w < wave) ? wave_totals[w] : 0;
  const int thread_prefix = wave_prefix + wave_inclusive - thread_total;
  // The thread owns its experts' padded BM blocks, so it emits their expert ids
  // straight from the scan registers instead of re-reading the prefix from LDS.
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    const int e = tid * ITEMS_PER_THREAD + i;
    if (e < NUM_EXPERTS) {
      const int begin = thread_prefix + ((i > 0) ? local[i - 1] : 0);
      const int end = thread_prefix + local[i];
      cumsum[e + 1] = end;
      if (sorted_expert_ids != nullptr)
        for (unsigned b = static_cast<unsigned>(begin) / SORT_MPB; b < static_cast<unsigned>(end) / SORT_MPB; ++b)
          sorted_expert_ids[b] = e;
    }
  }
  if (tid == 0) cumsum[0] = 0;
  __syncthreads();
}

/// parallel_cumsum over per-expert counts held in LDS.
template <int NUM_EXPERTS, int THREADS_PER_CTA, int SORT_MPB>
SGL_DEVICE void
parallel_cumsum_lds(const int* __restrict__ count, int* __restrict__ cumsum, int32_t* __restrict__ sorted_expert_ids) {
  constexpr int ITEMS_PER_THREAD = kExpertsPerThread<NUM_EXPERTS, THREADS_PER_CTA>;
  int v[ITEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    const int e = static_cast<int>(threadIdx.x) * ITEMS_PER_THREAD + i;
    v[i] = (e < NUM_EXPERTS) ? count[e] : 0;
  }
  parallel_cumsum<NUM_EXPERTS, THREADS_PER_CTA, SORT_MPB>(v, cumsum, sorted_expert_ids);
}

/// Write one winner into its expert's next free slot. `cursor` is the padded
/// prefix, already published, bumped in place; unreached slots keep their pad.
template <int TOPK, int SLOT_BITS>
SGL_DEVICE void place_pair(
    int* __restrict__ cursor,
    int32_t* __restrict__ sorted_token_ids,
    float* __restrict__ sorted_weights,
    int32_t* __restrict__ reverse_sorted,
    int32_t* __restrict__ m_indices,
    int eid,
    float weight,
    int pair) {
  const int sp = atomicAdd(&cursor[eid], 1);
  const int token_id = (pair >> SLOT_BITS) & 0x00FFFFFF;
  const int topk_id = pair & (TOPK - 1);
  sorted_token_ids[sp] = token_id | (topk_id << 24);
  m_indices[sp] = token_id;
  sorted_weights[sp] = weight;
  reverse_sorted[pair] = sp;
}

/// The whole sort with the pairs held in registers, so counting and placement
/// share one (post-invalidate) L2 read. Only for batches one round covers.
template <int NUM_EXPERTS, int TOPK, int M_PER_BLOCK, int THREADS_PER_CTA, int PAIRS_PER_ROUND, int SLOT_BITS>
SGL_DEVICE void sort_resident(
    int* __restrict__ count,
    int* __restrict__ cumsum,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weight,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ sorted_expert_ids,
    float* __restrict__ sorted_weights,
    int32_t* __restrict__ cumsum_tensor,
    int32_t* __restrict__ reverse_sorted,
    int32_t* __restrict__ m_indices,
    int total_pairs,
    int M) {
  const int tid = threadIdx.x;
  int eid[PAIRS_PER_ROUND];
  float w[PAIRS_PER_ROUND];
#pragma unroll
  for (int j = 0; j < PAIRS_PER_ROUND; ++j) {
    const int i = tid + j * THREADS_PER_CTA;
    const bool live = i < total_pairs;
    eid[j] = live ? topk_ids[i] : -1;
    w[j] = live ? topk_weight[i] : 0.0f;
  }
#pragma unroll
  for (int j = 0; j < PAIRS_PER_ROUND; ++j) {
    if ((unsigned)eid[j] < NUM_EXPERTS) atomicAdd(&count[eid[j]], 1);
  }
  __syncthreads();

  parallel_cumsum_lds<NUM_EXPERTS, THREADS_PER_CTA, M_PER_BLOCK>(count, cumsum, sorted_expert_ids);
  // Published before placement consumes the prefix as its cursor.
  if (tid == 0) {
    cumsum_tensor[0] = cumsum[NUM_EXPERTS];
    cumsum_tensor[1] = M;
  }

#pragma unroll
  for (int j = 0; j < PAIRS_PER_ROUND; ++j) {
    if ((unsigned)eid[j] < NUM_EXPERTS)
      place_pair<TOPK, SLOT_BITS>(
          cumsum, sorted_token_ids, sorted_weights, reverse_sorted, m_indices, eid[j], w[j], tid + j * THREADS_PER_CTA);
  }
}

template <int NUM_EXPERTS, int TOPK, int M_PER_BLOCK, int THREADS_PER_CTA>
SGL_DEVICE void sort_subkernel(
    int* count,
    int* cumsum,
    const int32_t* topk_ids,
    const float* topk_weight,
    int32_t* sorted_token_ids,
    int32_t* sorted_expert_ids,
    float* sorted_weights,
    int32_t* cumsum_tensor,
    int32_t* reverse_sorted,
    int32_t* m_indices,
    int M) {
  static_assert((TOPK & (TOPK - 1)) == 0, "a power-of-two topk splits the pair index by shift and mask");
  constexpr int kSlotBits = (TOPK == 1)    ? 0
                            : (TOPK == 2)  ? 1
                            : (TOPK == 4)  ? 2
                            : (TOPK == 8)  ? 3
                            : (TOPK == 16) ? 4
                                           : 5;
  static_assert(1 << kSlotBits == TOPK, "kSlotBits must be log2(TOPK)");
  constexpr int kMaxResident = 4;

  const int total_pairs = M * TOPK;
  const int per_thread = (total_pairs + THREADS_PER_CTA - 1) / THREADS_PER_CTA;

#define SGL_RADIX4_SORT_RESIDENT(PPR)                                             \
  sort_resident<NUM_EXPERTS, TOPK, M_PER_BLOCK, THREADS_PER_CTA, PPR, kSlotBits>( \
      count,                                                                      \
      cumsum,                                                                     \
      topk_ids,                                                                   \
      topk_weight,                                                                \
      sorted_token_ids,                                                           \
      sorted_expert_ids,                                                          \
      sorted_weights,                                                             \
      cumsum_tensor,                                                              \
      reverse_sorted,                                                             \
      m_indices,                                                                  \
      total_pairs,                                                                \
      M)
  // The host sends wider batches to sort_tail_stable (kRadix4LeaderSortMaxTokens).
  if (per_thread <= 1) {
    SGL_RADIX4_SORT_RESIDENT(1);
  } else {
    SGL_RADIX4_SORT_RESIDENT(kMaxResident);
  }
#undef SGL_RADIX4_SORT_RESIDENT
}

/// The sort tail for batches past one CTA, shared by the last `num_helpers` CTAs
/// to arrive. Helper h places the routes of token chunk h, tokens
/// [h * T, h * T + T) with T = ceil(M / num_helpers) <= kTailMaxChunkTokens.
///
/// A routed token picks an expert at most once, so a route's rank within its
/// expert is the number of earlier tokens that picked it. Each helper tallies
/// every route into LDS -- tokens before its chunk into s_before, the rest into
/// s_rest -- and sets one bit per chunk token in s_mask[expert], so a route's
/// slot is cumsum[e] + s_before[e] + popcount(the mask bits of earlier chunk
/// tokens). That puts every expert's routes in token order, the layout AITER's
/// own sort writes, with no global atomics and the same result on every run.
/// Every helper rebuilds the full prefix itself, which costs less than handing
/// one around.
///
/// CUDA-graph padding rows (tokens >= n_real) send all TOPK routes to expert 0.
/// They are left out of the tally and placed after expert 0's routed tokens in
/// (token, slot) order, every route in a slot of its own as the leader sort does.
///
/// s_rest, s_before and s_mask arrive zeroed; the route ids and weights were
/// written through to the coherent point and this CTA has since acquired.
template <int NUM_EXPERTS, int TOPK, int SORT_MPB, int THREADS_PER_CTA>
SGL_DEVICE void sort_tail_stable(
    int* __restrict__ s_rest,
    int* __restrict__ s_before,
    uint32_t* __restrict__ s_mask,
    int* __restrict__ cumsum,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weight,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ sorted_expert_ids,
    float* __restrict__ sorted_weights,
    int32_t* __restrict__ cumsum_tensor,
    int32_t* __restrict__ reverse_sorted,
    int32_t* __restrict__ m_indices,
    const int32_t* __restrict__ num_non_padded,
    int helper,
    int num_helpers,
    int M,
    bool writes_layout) {
  static_assert(TOPK == 16, "tallies a token's routes as four int4");
  constexpr int kVecPerToken = TOPK / 4;
  constexpr int kPairsPerThread = (kTailMaxChunkTokens * TOPK + THREADS_PER_CTA - 1) / THREADS_PER_CTA;
  // Loads issued ahead of their tally, so a round costs one trip to memory.
  constexpr int kVecPerRound = 4;
  const int tid = threadIdx.x;
  const int chunk = (M + num_helpers - 1) / num_helpers;
  const int t_begin = helper * chunk;
  const int t_end = (t_begin + chunk < M) ? (t_begin + chunk) : M;
  int n_real = M;
  if (num_non_padded != nullptr) n_real = (*num_non_padded < M) ? *num_non_padded : M;

  int my_e[kPairsPerThread];
  float my_w[kPairsPerThread];
#pragma unroll
  for (int j = 0; j < kPairsPerThread; ++j) {
    const int q = t_begin * TOPK + tid + j * THREADS_PER_CTA;
    const bool live = q < t_end * TOPK;
    my_e[j] = live ? topk_ids[q] : -1;
    my_w[j] = live ? topk_weight[q] : 0.0f;
  }

  const auto* ids4 = reinterpret_cast<const int4*>(topk_ids);
  const int nvec = M * kVecPerToken;
  for (int base = 0; base < nvec; base += kVecPerRound * THREADS_PER_CTA) {
    int4 ids[kVecPerRound];
#pragma unroll
    for (int r = 0; r < kVecPerRound; ++r) {
      const int v = base + tid + r * THREADS_PER_CTA;
      ids[r] = (v < nvec) ? ids4[v] : make_int4(-1, -1, -1, -1);
    }
#pragma unroll
    for (int r = 0; r < kVecPerRound; ++r) {
      const int t = (base + tid + r * THREADS_PER_CTA) / kVecPerToken;
      const int e4[4] = {ids[r].x, ids[r].y, ids[r].z, ids[r].w};
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int e = e4[c];
        if ((unsigned)e >= NUM_EXPERTS || t >= n_real) continue;
        if (t < t_begin) {
          atomicAdd(&s_before[e], 1);
        } else {
          atomicAdd(&s_rest[e], 1);
          if (t < t_end) atomicOr(&s_mask[e], 1u << (t - t_begin));
        }
      }
    }
  }
  __syncthreads();

  constexpr int ITEMS = kExpertsPerThread<NUM_EXPERTS, THREADS_PER_CTA>;
  int count[ITEMS];
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int e = tid * ITEMS + i;
    count[i] = (e < NUM_EXPERTS) ? s_rest[e] + s_before[e] : 0;
  }
  if (tid == 0) count[0] += (M - n_real) * TOPK;
  parallel_cumsum<NUM_EXPERTS, THREADS_PER_CTA, SORT_MPB>(count, cumsum, writes_layout ? sorted_expert_ids : nullptr);
  if (writes_layout && tid == 0) {
    cumsum_tensor[0] = cumsum[NUM_EXPERTS];
    cumsum_tensor[1] = M;
  }

#pragma unroll
  for (int j = 0; j < kPairsPerThread; ++j) {
    const int e = my_e[j];
    if ((unsigned)e < NUM_EXPERTS) {
      const int q = t_begin * TOPK + tid + j * THREADS_PER_CTA;
      const int t = q / TOPK;
      const int rank = (t < n_real) ? s_before[e] + __popc(s_mask[e] & ((1u << (t - t_begin)) - 1u))
                                    : s_before[0] + s_rest[0] + (q - n_real * TOPK);
      const int sp = cumsum[e] + rank;
      sorted_token_ids[sp] = t | ((q & (TOPK - 1)) << 24);
      m_indices[sp] = t;
      sorted_weights[sp] = my_w[j];
      reverse_sorted[q] = sp;
    }
  }
}

template <int THREADS_PER_CTA>
SGL_DEVICE void zero_bf16_row(void* out, int row, int dim) {
  if (out == nullptr || dim <= 0) return;
  using vec_t = int4;
  constexpr int kElemsPerVec = sizeof(vec_t) / 2;
  const int tid = threadIdx.x;
  vec_t* out_v = reinterpret_cast<vec_t*>(out) + static_cast<long long>(row) * dim / kElemsPerVec;
  const int total_vecs = dim / kElemsPerVec;
  const vec_t zero = {0, 0, 0, 0};
  for (int i = tid; i < total_vecs; i += THREADS_PER_CTA)
    out_v[i] = zero;
}

}  // namespace sglang::radix4_sort
