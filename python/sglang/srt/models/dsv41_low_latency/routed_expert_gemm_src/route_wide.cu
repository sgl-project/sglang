// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Route kernel of the DeepSeek-V4.1 low-batch routed-expert GEMMs for 9-64
// rows. Same record as route.cuh, but an expert chosen by more than 8 rows owns
// ceil(count / 8) consecutive 8-row CTAs, so the unchanged GEMM pair (one
// expert and one row limit per CTA) serves it in several tiles. An expert's
// first CTA is the exclusive prefix sum of the tiles of the experts below it
// (one block scan over the expert ids); a slot's rank is the number of earlier
// rows that chose the expert (a 64-bit row mask per expert).
//
// PDL as route.cu: dependents launch at once and read the record after their
// own wait.

namespace {
constexpr int kTile = 8;
constexpr int kMaxExperts = 384;
constexpr int kThreads = 512; // >= rows * top_k (64 * 8) and >= kMaxExperts
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ int MaxCtas(int rows, int top_k, int experts) {
  const int slots = rows * top_k;
  const int filled = experts < slots ? experts : slots;
  return filled + (slots - filled) / kTile;
}
} // namespace

// experts <= 384; rows <= 64; rows * top_k <= 512. barrier: unused.
extern "C" __global__
__launch_bounds__(kThreads) void dsv41_routed_expert_route_wide(
    const int *ids, int *__restrict__ permuted, int *__restrict__ record,
    int rows, int top_k, int experts) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __shared__ unsigned long long rows_of[kMaxExperts];
  __shared__ int first_cta[kMaxExperts];
  __shared__ int warp_total[kWarps];
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  for (int e = thread; e < kMaxExperts; e += kThreads)
    rows_of[e] = 0ULL;
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");

  const int slots = rows * top_k;
  const int max_ctas = MaxCtas(rows, top_k, experts);
  const int cta_words = (max_ctas + 3) / 4 * 4;
  const int limit_offset = 4 + cta_words;
  const int token_offset = 4 + 2 * cta_words;
  const int record_words = token_offset + kTile * max_ctas;
  for (int word = thread; word < record_words; word += kThreads)
    record[word] = 0;

  const int row = thread / top_k;
  int expert = -1;
  if (thread < slots) {
    expert = ids[thread];
    if (expert < 0 || expert >= experts)
      expert = -1;
  }
  if (expert >= 0)
    atomicOr(&rows_of[expert], 1ULL << row);
  __syncthreads();

  // Exclusive scan of tiles(e) = ceil(count(e) / 8) over e = thread.
  const int tiles =
      thread < experts ? (__popcll(rows_of[thread]) + kTile - 1) / kTile : 0;
  int inclusive = tiles;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int up = __shfl_up_sync(0xffffffffU, inclusive, offset);
    if (lane >= offset)
      inclusive += up;
  }
  if (lane == 31)
    warp_total[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    int total = lane < kWarps ? warp_total[lane] : 0;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int up = __shfl_up_sync(0xffffffffU, total, offset);
      if (lane >= offset)
        total += up;
    }
    if (lane < kWarps)
      warp_total[lane] = total;
  }
  __syncthreads();
  if (thread < experts)
    first_cta[thread] =
        inclusive - tiles + (warp > 0 ? warp_total[warp - 1] : 0);
  if (thread == 0) {
    record[0] = warp_total[kWarps - 1] * kTile;
    record[1] = warp_total[kWarps - 1];
  }
  __syncthreads();

  if (thread >= slots)
    return;
  if (expert < 0) {
    permuted[thread] = -1;
    return;
  }
  const unsigned long long chosen_by = rows_of[expert];
  const int count = __popcll(chosen_by);
  const int rank = __popcll(chosen_by & ((1ULL << row) - 1ULL));
  const int base_row = first_cta[expert] * kTile;
  permuted[thread] = base_row + rank;
  record[token_offset + base_row + rank] = row;
  if (rank % kTile == 0) {
    const int cta = first_cta[expert] + rank / kTile;
    record[4 + cta] = expert;
    record[limit_offset + cta] = min((cta + 1) * kTile, base_row + count);
  }
}
