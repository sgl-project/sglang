// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Routing record of the DeepSeek-V4.1 low-batch routed-expert GEMMs
// (routed_expert_gemm.py), built by the route kernels (route.cu,
// route_wide.cu). The top-k expert ids of up to eight rows become the
// expert-major, tile-padded row permutation and the per-CTA tables the two
// batched GEMMs read. The result is that of FlashInfer's TRT-LLM-gen routing
// kernel with precomputed ids and an 8-token tile:
//
//   * local experts (ids in [0, experts)) in ascending id order, each one's
//   rows in
//     ascending token order, every expert's segment padded to a multiple of the
//     tile (8 rows);
//   * permuted[t, k] = the slot's row in that order, -1 for an id outside
//     [0, experts);
//   * route (int32, one buffer per call):
//       [0]                       8 * CTAs (padded rows)
//       [1]                       CTAs
//       [2, 4)                    zero
//       [4, 4 + C)                CTA -> local expert
//       [4 + C, 4 + 2C)           CTA -> row limit min((c + 1) * 8, first +
//       count) [4 + 2C, 4 + 2C + 8 M)    padded row -> token (GEMM1 only)
//     where M = MaxCtas (the launch's CTA count in the token dimension) and C
//     is M rounded up to a multiple of four.
//
// The ids come from a top-k selection, so the valid ids of one row are
// distinct: an expert is chosen by at most `rows` <= 8 slots and owns exactly
// one 8-row CTA. The expert's CTA index is then the number of distinct selected
// experts below it (a presence bitmap), and a slot's rank the number of earlier
// rows that chose the expert (a row mask per expert): a few popcounts, no
// scans.
//
// `barrier` (may be null): bit 31 flips once per record; the GEMMs here do not
// read it.

namespace route {

constexpr int kTile = 8;
constexpr int kMaxExperts = 384;
constexpr int kWords = kMaxExperts / 32;

__device__ __forceinline__ int MaxCtas(int rows, int top_k, int experts) {
  const int slots = rows * top_k;
  const int filled = experts < slots ? experts : slots;
  return filled + (slots - filled) / kTile;
}

__device__ __forceinline__ int CtaWords(int max_ctas) {
  return (max_ctas + 3) / 4 * 4;
}

struct Scratch {
  unsigned int present[kWords];
  unsigned int rows_of[kMaxExperts];
};

// Zero the scratch; a __syncthreads() must separate it from Build.
__device__ __forceinline__ void Clear(Scratch &scratch, int thread,
                                      int threads) {
  for (int e = thread; e < kMaxExperts; e += threads) {
    scratch.rows_of[e] = 0U;
    if (e < kWords)
      scratch.present[e] = 0U;
  }
}

// Zero the record (`thread` of `threads` participating threads); Build's block
// synchronization orders it before Build's stores.
__device__ __forceinline__ void Zero(int *__restrict__ record, int rows,
                                     int top_k, int experts, int thread,
                                     int threads) {
  const int max_ctas = MaxCtas(rows, top_k, experts);
  const int record_words = 4 + 2 * CtaWords(max_ctas) + kTile * max_ctas;
  for (int word = thread; word < record_words; word += threads)
    record[word] = 0;
}

// Build the record from `ids` (global or shared, visible to every thread of the
// block) into a record Zero has cleared (before this call, by any threads of
// the block). Every thread of the block calls it (it synchronizes the block
// once); threads >= rows * top_k return early after that. Needs threads >= rows
// * top_k. experts: a multiple of 32, at most kMaxExperts; rows <= 8.
__device__ __forceinline__ void Build(Scratch &scratch, const int *ids,
                                      int *__restrict__ permuted,
                                      int *__restrict__ record,
                                      unsigned int *barrier, int rows,
                                      int top_k, int experts, int thread) {
  const int slots = rows * top_k;
  const int max_ctas = MaxCtas(rows, top_k, experts);
  const int cta_words = CtaWords(max_ctas);
  const int limit_offset = 4 + cta_words;
  const int token_offset = 4 + 2 * cta_words;
  if (thread == 0 && barrier != nullptr)
    atomicXor(barrier, 0x80000000U);
  const int row = thread / top_k;
  int expert = -1;
  if (thread < slots) {
    expert = ids[thread];
    if (expert < 0 || expert >= experts)
      expert = -1;
  }
  if (expert >= 0) {
    atomicOr(&scratch.present[expert >> 5], 1U << (expert & 31));
    atomicOr(&scratch.rows_of[expert], 1U << row);
  }
  __syncthreads();
  if (thread >= slots)
    return;
  int total_ctas = 0;
  int base_cta = 0;
#pragma unroll
  for (int word = 0; word < kWords; ++word) {
    const int bits = __popc(scratch.present[word]);
    total_ctas += bits;
    if (expert >= 0 && word < (expert >> 5))
      base_cta += bits;
  }
  if (thread == 0) {
    record[0] = total_ctas * kTile;
    record[1] = total_ctas;
  }
  if (expert < 0) {
    permuted[thread] = -1;
    return;
  }
  base_cta +=
      __popc(scratch.present[expert >> 5] & ((1U << (expert & 31)) - 1U));
  const unsigned int chosen_by = scratch.rows_of[expert];
  const int count = __popc(chosen_by);
  const int rank = __popc(chosen_by & ((1U << row) - 1U));
  const int base_row = base_cta * kTile;
  permuted[thread] = base_row + rank;
  record[token_offset + base_row + rank] = row;
  if (rank == 0) {
    record[4 + base_cta] = expert;
    record[limit_offset + base_cta] =
        min((base_cta + 1) * kTile, base_cta * kTile + count);
  }
}

} // namespace route
