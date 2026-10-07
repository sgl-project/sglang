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
/// the last arriver is left with count, prefix and placement only. Placement
/// then overwrites exactly the slots a winner lands on, which is why aiter's
/// per-expert gap fill has no counterpart here -- the gaps were never written.

#pragma once

#include <cstdint>

namespace sglang::radix4_sort {

SGL_DEVICE int round_up_mpb(int x, int mpb) {
  return (x + mpb - 1) / mpb * mpb;
}

/// Pad one CTA's slice of the sort output. `capacity` is the length the host
/// reserved, which bounds the padded length from above (see _max_sorted), so
/// padding all of it covers every slot the GEMM can reach and leaves nothing
/// uninitialized behind the padded length either. A whole number of BM blocks
/// fits in the capacity, hence a whole number of int4, so the slice needs no
/// scalar tail.
template <int THREADS_PER_CTA>
SGL_DEVICE void pad_sorted_slice(
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ m_indices,
    float* __restrict__ sorted_weights,
    int capacity,
    int pad_val,
    int part,
    int nparts) {
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

/// Clear the histogram the last arriver will tally into. Every CTA clears its
/// own copy before the arrival barrier, so whichever one turns out to be the
/// leader finds it already zeroed instead of paying for it in the serial tail.
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

/// Tally the (token, slot) pairs per expert. `count` arrives zeroed from
/// reset_counts, which the arrival barrier already separates from this tally.
template <int NUM_EXPERTS, int THREADS_PER_CTA>
SGL_DEVICE void
count_tokens_per_expert(int* __restrict__ count, const int32_t* __restrict__ topk_ids, int total_pairs) {
  const int tid = threadIdx.x;
  const int4* topk_vec = reinterpret_cast<const int4*>(topk_ids);
  const int total_aligned = total_pairs & ~3;
  for (int i = tid * 4; i < total_aligned; i += THREADS_PER_CTA * 4) {
    int4 ids = topk_vec[i >> 2];
    if ((unsigned)ids.x < NUM_EXPERTS) atomicAdd(&count[ids.x], 1);
    if ((unsigned)ids.y < NUM_EXPERTS) atomicAdd(&count[ids.y], 1);
    if ((unsigned)ids.z < NUM_EXPERTS) atomicAdd(&count[ids.z], 1);
    if ((unsigned)ids.w < NUM_EXPERTS) atomicAdd(&count[ids.w], 1);
  }
  for (int i = total_aligned + tid; i < total_pairs; i += THREADS_PER_CTA) {
    int eid = topk_ids[i];
    if ((unsigned)eid < NUM_EXPERTS) atomicAdd(&count[eid], 1);
  }
  __syncthreads();
}

template <int NUM_EXPERTS, int THREADS_PER_CTA, int SORT_MPB>
SGL_DEVICE void
parallel_cumsum(const int* __restrict__ count, int* __restrict__ cumsum, int32_t* __restrict__ sorted_expert_ids) {
  // The standalone AITER sorter has 1024 threads and assigns one expert to a
  // lane. Radix4 has 256 threads, so each lane scans a contiguous group of
  // experts locally, then the block scans those group totals. Keeping expert
  // groups contiguous is important: a strided assignment would not produce
  // the expert-order prefix that GEMM dispatch consumes.
  constexpr int WAVE = 64;
  constexpr int N_WAVES = THREADS_PER_CTA / WAVE;
  constexpr int ITEMS_PER_THREAD = (NUM_EXPERTS + THREADS_PER_CTA - 1) / THREADS_PER_CTA;
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
    const int v = (e < NUM_EXPERTS) ? round_up_mpb(count[e], SORT_MPB) : 0;
    thread_total += v;
    local[i] = thread_total;
  }

  int wave_inclusive = thread_total;
  wave_inclusive += __builtin_amdgcn_update_dpp(0, wave_inclusive, 0x111, 0xf, 0xf, false);
  wave_inclusive += __builtin_amdgcn_update_dpp(0, wave_inclusive, 0x112, 0xf, 0xf, false);
  wave_inclusive += __builtin_amdgcn_update_dpp(0, wave_inclusive, 0x114, 0xf, 0xe, false);
  wave_inclusive += __builtin_amdgcn_update_dpp(0, wave_inclusive, 0x118, 0xf, 0xc, false);
  wave_inclusive += __builtin_amdgcn_update_dpp(0, wave_inclusive, 0x142, 0xa, 0xf, false);
  wave_inclusive += __builtin_amdgcn_update_dpp(0, wave_inclusive, 0x143, 0xc, 0xf, false);

  __shared__ int wave_totals[N_WAVES];
  if (lane == WAVE - 1) wave_totals[wave] = wave_inclusive;
  __syncthreads();

  // There are only N_WAVES totals, so every thread adding up the ones below its
  // own wave is cheaper than a second scan followed by a second barrier -- the
  // same trade the router makes for its block-wide OR and AND.
  int wave_prefix = 0;
#pragma unroll
  for (int w = 0; w < N_WAVES; ++w)
    wave_prefix += (w < wave) ? wave_totals[w] : 0;
  const int thread_prefix = wave_prefix + wave_inclusive - thread_total;
  // The thread owns experts [tid * ITEMS, tid * ITEMS + ITEMS), so it also owns
  // the BM blocks their padded ranges cover: the block ids it writes here are
  // consecutive and its neighbours continue where it stops. Emitting them from
  // the scan registers saves reading the prefix back out of LDS, and costs
  // nothing for an expert no token picked, whose range is empty.
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    const int e = tid * ITEMS_PER_THREAD + i;
    if (e < NUM_EXPERTS) {
      const int begin = thread_prefix + ((i > 0) ? local[i - 1] : 0);
      const int end = thread_prefix + local[i];
      cumsum[e + 1] = end;
      for (unsigned b = static_cast<unsigned>(begin) / SORT_MPB; b < static_cast<unsigned>(end) / SORT_MPB; ++b)
        sorted_expert_ids[b] = e;
    }
  }
  if (tid == 0) cumsum[0] = 0;
  __syncthreads();
}

/// Write one winner into its expert's next free slot. `cursor` is the padded
/// prefix itself, bumped in place: the prefix has already been published to
/// `cumsum_tensor` and consumed for the expert ids, so an expert's next slot is
/// exactly its running cursor and placement needs one LDS atomic rather than an
/// atomic plus a read. Every slot no winner reaches already holds the pad token
/// from pad_sorted_slice, which is why nothing fills the gaps afterwards.
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

/// Placement for a batch too wide to hold in registers, streamed a round at a
/// time. A pair's expert has to come back from L2 before its slot can be
/// claimed, so reading one pair, claiming, and going back for the next would put
/// one round trip per pair end to end; the round's pairs are read up front
/// instead, which leaves one round trip for the round.
template <int NUM_EXPERTS, int TOPK, int THREADS_PER_CTA, int PAIRS_PER_ROUND, int SLOT_BITS>
SGL_DEVICE void place_streamed(
    int* __restrict__ cursor,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weight,
    int32_t* __restrict__ sorted_token_ids,
    float* __restrict__ sorted_weights,
    int32_t* __restrict__ reverse_sorted,
    int32_t* __restrict__ m_indices,
    int total_pairs) {
  const int tid = threadIdx.x;
  for (int base = 0; base < total_pairs; base += PAIRS_PER_ROUND * THREADS_PER_CTA) {
    int eid[PAIRS_PER_ROUND];
    float w[PAIRS_PER_ROUND];
#pragma unroll
    for (int j = 0; j < PAIRS_PER_ROUND; ++j) {
      const int i = base + tid + j * THREADS_PER_CTA;
      const bool live = i < total_pairs;
      eid[j] = live ? topk_ids[i] : -1;
      w[j] = live ? topk_weight[i] : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < PAIRS_PER_ROUND; ++j) {
      if ((unsigned)eid[j] < NUM_EXPERTS)
        place_pair<TOPK, SLOT_BITS>(
            cursor,
            sorted_token_ids,
            sorted_weights,
            reverse_sorted,
            m_indices,
            eid[j],
            w[j],
            base + tid + j * THREADS_PER_CTA);
    }
  }
}

/// The whole sort with the pairs held in registers across it. Counting and
/// placement read the same pairs, and the arrival release invalidated this CU's
/// L2 just before, so reading them twice means paying for the miss twice and
/// paying for the second one where nothing can cover it -- the prefix scan
/// stands between the two reads and would have covered it. Both weights and
/// experts are fetched up front for the same reason. Only usable when one round
/// covers the batch, which is what PAIRS_PER_ROUND is chosen for.
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

  parallel_cumsum<NUM_EXPERTS, THREADS_PER_CTA, M_PER_BLOCK>(count, cumsum, sorted_expert_ids);
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

  const int tid = threadIdx.x;
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
  if (per_thread <= 1) {
    SGL_RADIX4_SORT_RESIDENT(1);
  } else if (per_thread <= kMaxResident) {
    SGL_RADIX4_SORT_RESIDENT(kMaxResident);
  } else {
    // Wider than the decode batches the fusion is for: stream the pairs.
    count_tokens_per_expert<NUM_EXPERTS, THREADS_PER_CTA>(count, topk_ids, total_pairs);
    parallel_cumsum<NUM_EXPERTS, THREADS_PER_CTA, M_PER_BLOCK>(count, cumsum, sorted_expert_ids);
    if (tid == 0) {
      cumsum_tensor[0] = cumsum[NUM_EXPERTS];
      cumsum_tensor[1] = M;
    }
    place_streamed<NUM_EXPERTS, TOPK, THREADS_PER_CTA, kMaxResident, kSlotBits>(
        cumsum, topk_ids, topk_weight, sorted_token_ids, sorted_weights, reverse_sorted, m_indices, total_pairs);
  }
#undef SGL_RADIX4_SORT_RESIDENT
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
