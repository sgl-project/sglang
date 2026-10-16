// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Developed at NVIDIA; reformatted for SGLang. The host side is
// sharded_markov.py.
//
// DSpark's sequential Markov sampling for a small batch (up to 8 request rows),
// with the Markov head sharded by vocabulary across the tensor-parallel ranks
// (sharded_markov.py explains the design). Compiled at runtime with NVRTC; the
// host defines
//   STEPS    number of draft positions (1..8)
//   BASE_T   element type of the base logits (__nv_bfloat16 or float)
//   CACHE_T  element type of the draft-logits cache (__nv_bfloat16 or float)
//
// For request r, step i needs r's previous token p (its anchor for i = 0):
//     logits_ri = cache_round(base_ri + bf16(markov_head . markov_embed[p]))
//     d_r,i+1   = argmax(logits_ri)                     (temperature 0)
//               = argmax(logits_ri / T_r + gumbel_ri)   (otherwise)
// Each rank owns rows [rank * V/world, (rank + 1) * V/world) of the head. Each
// warp keeps one 16-row tile of that shard in registers for all steps, so the
// head is read once per launch; the m16n8k16 MMA's eight columns carry the
// eight request rows' embeddings, so every row's bias comes from the same MMAs.
// A step is those MMAs, a CTA reduction and a two-level exchange of 64-bit
// candidate keys per request row: the CTAs of a rank fold their best key into a
// rank-local maximum, the last CTA to arrive stores that key, tagged with the
// launch epoch, into every rank's mailbox, and every CTA waits for the `world`
// records of each row in its own mailbox and takes their maximum. The exchange
// is the grid barrier and the cross-rank argmax at once, so the grid must be
// co-resident (the host launches it cooperatively).
//
// The base logits may hold the full vocabulary or only this rank's shard (the
// draft LM head's output before its all-gather): `base_column` is the column of
// this rank's first vocabulary entry in them, rank * V/world or 0.
//
// Rows whose request-state index is negative (CUDA-graph padding) are inert, as
// in the fused path: their tokens are drawn greedily and their logits are not
// cached.
//
// The draft-logits cache must hold the full vocabulary on every rank. Each rank
// writes its shard into its own cache and into a staging buffer of every peer;
// after the last exchange each rank copies the peers' shards from its staging
// buffer into its cache. Staging and mailbox records alternate between two
// parities so that a launch never overwrites what a peer may still be reading
// from the previous launch.

#include <cuda_bf16.h>

// No <cstdint>: NVRTC need not provide it.
using u32 = unsigned int;
using u64 = unsigned long long;
using i32 = int;
using i64 = long long;

#ifndef STEPS
#error "STEPS must be defined"
#endif

namespace {

constexpr int kRankDim = 256;
constexpr int kChunks = kRankDim / 32; // 8
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;
constexpr int kMaxSteps = 8;
constexpr int kMaxRanks = 8;
constexpr int kMaxRows = 8; // MMA columns
// Peer buffer layout (bytes): records u64 [parity 2][step 8][row 8][source rank
// 8] (8 KiB), then the staging logits CACHE_T [parity
// 2][max_rows][STEPS][vocab] at kStagingOffset.
constexpr int kRecordWords = 2 * kMaxSteps * kMaxRows * kMaxRanks;
constexpr i64 kStagingOffset = 16384;
// Local control words (uint32 view): combine maxima u64 [step 8][row 8] at
// words 0..127, arrivals u32 [step 8] at 128..135, launches at 136, end
// arrivals at 137.
constexpr int kArrivalWord = 128;
constexpr int kLaunchWord = 136;
constexpr int kEndWord = 137;

constexpr u32 kTagBits = 15;
constexpr u32 kTagMask = (1u << kTagBits) - 1;
constexpr u32 kIndexBits = 17;
constexpr u32 kIndexMask = (1u << kIndexBits) - 1;

static_assert(STEPS >= 1 && STEPS <= kMaxSteps, "unsupported STEPS");
static_assert(kRecordWords * 8 <= kStagingOffset, "mailbox overlaps staging");

// A 16-row tile of logits is this many 16-byte vectors of this many elements.
constexpr int kVectorElements = 16 / static_cast<int>(sizeof(CACHE_T));
constexpr int kTileVectors = 16 / kVectorElements;

// Pull one embedding row (512 bytes) into L2.
__device__ __forceinline__ void PrefetchRowL2(const __nv_bfloat16 *row) {
#pragma unroll
  for (int line = 0; line < kRankDim * 2 / 128; ++line) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(
        reinterpret_cast<const char *>(row) + line * 128));
  }
}

__device__ __forceinline__ float ToFloat(float v) { return v; }
__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
__device__ __forceinline__ void FromFloat(float v, float &out) { out = v; }
__device__ __forceinline__ void FromFloat(float v, __nv_bfloat16 &out) {
  out = __float2bfloat16_rn(v);
}

// Triton lowers an fp32 `/` to div.full.f32; the stock sampler divides that
// way.
__device__ __forceinline__ float DivFull(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// fp32 add without contraction into a neighbouring multiply.
__device__ __forceinline__ float AddRn(float a, float b) {
  return __fadd_rn(a, b);
}

__device__ __forceinline__ u32 OrderedBits(float value) {
  const u32 bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// [48:17] order-preserving score bits, [16:0] inverted index (ties prefer the
// smaller index, as torch.argmax does). Larger is better; 0 means "no key".
__device__ __forceinline__ u64 CandidateKey(float score, i32 index) {
  return (static_cast<u64>(OrderedBits(score)) << kIndexBits) |
         static_cast<u64>(kIndexMask - static_cast<u32>(index));
}

__device__ __forceinline__ i32 KeyIndex(u64 key) {
  return static_cast<i32>(kIndexMask - static_cast<u32>(key & kIndexMask));
}

// Never zero; distinct for the two launches that share a parity slot.
__device__ __forceinline__ u32 EpochTag(u32 count) {
  return 1u + count % 32767u;
}

__device__ __forceinline__ u64 WarpMaxU64(u64 value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const u64 other = __shfl_xor_sync(0xffffffffu, value, offset);
    value = other > value ? other : value;
  }
  return value;
}

__device__ __forceinline__ void StoreRecord(u64 *address, u64 value) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(address), "l"(value)
               : "memory");
}

__device__ __forceinline__ u64 LoadRecord(const u64 *address) {
  u64 value;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ u64 GlobalTimer() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Kept out of line so the report's arguments cost the hot path no registers.
__device__ __noinline__ void ReportTimeout(int step, i64 timeout_ns, int source,
                                           u32 tag, u32 seen) {
  printf("sharded Markov sampler: step %d waited %lld ns for rank %d (tag %u, "
         "saw %u)\n",
         step, static_cast<long long>(timeout_ns), source, tag, seen);
  __trap();
}

// Spin until the record at `address` carries `tag`; return its key. With a
// positive `timeout_ns` a wait that exceeds it reports and traps instead of
// hanging (debug aid for ranks that disagree on the launch sequence).
__device__ __forceinline__ u64 AwaitRecord(const u64 *address, u32 tag,
                                           i64 timeout_ns, int step,
                                           int source) {
  u64 value = LoadRecord(address);
  if ((value & kTagMask) == tag)
    return value >> kTagBits;
  const u64 start = timeout_ns > 0 ? GlobalTimer() : 0;
  u32 spins = 0;
  while ((value & kTagMask) != tag) {
    value = LoadRecord(address);
    if (timeout_ns > 0 && (++spins & 1023u) == 0 &&
        GlobalTimer() - start > static_cast<u64>(timeout_ns)) {
      ReportTimeout(step, timeout_ns, source, tag,
                    static_cast<u32>(value & kTagMask));
    }
  }
  return value >> kTagBits;
}

__device__ __forceinline__ char *PeerBase(const u64 *peers, int rank) {
  return reinterpret_cast<char *>(static_cast<unsigned long long>(peers[rank]));
}

// mma.m16n8k16 sums over k, so any bijection between fragment columns and
// physical k positions gives the same dot products when A and B share it. Lane
// (g = lane / 4, t = lane % 4) loads 16 contiguous bytes of weight rows g and
// g + 8 at k = base + 8 t and the matching 8 embedding values; values 0-3 feed
// the first MMA and 4-7 the second.
__device__ __forceinline__ void Mma16816(float (&c)[4], u32 a0, u32 a1, u32 a2,
                                         u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void MmaChunk(float (&c)[4], const uint4 &lo,
                                         const uint4 &hi, const uint4 &x) {
  Mma16816(c, lo.x, hi.x, lo.y, hi.y, x.x, x.y);
  Mma16816(c, lo.z, hi.z, lo.w, hi.w, x.z, x.w);
}

__device__ __forceinline__ void LoadTile(uint4 (&lo)[kChunks],
                                         uint4 (&hi)[kChunks],
                                         const __nv_bfloat16 *head, int tile,
                                         int lane) {
  const __nv_bfloat16 *row_lo =
      head + static_cast<i64>(tile * 16 + (lane >> 2)) * kRankDim +
      (lane & 3) * 8;
  const __nv_bfloat16 *row_hi = row_lo + 8 * kRankDim;
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    lo[chunk] = *reinterpret_cast<const uint4 *>(row_lo + chunk * 32);
    hi[chunk] = *reinterpret_cast<const uint4 *>(row_hi + chunk * 32);
  }
}

} // namespace

// Inputs written by earlier graph nodes are deliberately not const
// __restrict__, so their loads stay coherent across replays.
extern "C" __global__ void
__launch_bounds__(kThreads, 1) dspark_sharded_markov_sample(
    const BASE_T *base_logits, // [rows, STEPS, vocab or vocab_local]
    i64 base_row_stride, i64 base_step_stride, // elements
    i64 base_column,                           // column of this rank's shard
    const __nv_bfloat16 *__restrict__ embed_table, // [vocab, 256]
    const __nv_bfloat16 *__restrict__ head, // this rank's [vocab_local, 256]
    const i32 *input_ids, // row r's anchor at [r * query_per_row]
    int query_per_row,
    const i32 *idx_mapping,   // [rows * STEPS] request-state rows
    const float *temperature, // [max_num_reqs]
    const float *noise,       // [rows * STEPS, vocab_local]
    CACHE_T *logits_cache,    // [max_num_reqs, steps, vocab]
    i64 cache_stride0, i64 cache_stride1, i64 *draft_tokens,
    i64 draft_stride,              // [rows, STEPS] (row stride)
    const u64 *__restrict__ peers, // [world] peer buffer bases
    u32 *control,                  // local control words
    int rank, int world, int vocab_local, int rows, int max_rows,
    i64 timeout_ns) {
  __shared__ __align__(16) __nv_bfloat16 embedding[kMaxRows][kRankDim];
  __shared__ unsigned long long warp_best[kWarps][kMaxRows];
  __shared__ unsigned long long winner[kMaxRows];
  __shared__ u32 s_count;
  __shared__ __align__(16) CACHE_T tile_logits[kWarps][kMaxRows][16];

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int tiles = vocab_local / 16;
  const int stride = static_cast<int>(gridDim.x) * kWarps;
  const int first_tile = static_cast<int>(blockIdx.x) * kWarps + warp;
  const int ctas = static_cast<int>(gridDim.x);
  const int vocab_offset = rank * vocab_local;
  const i64 vocab = static_cast<i64>(vocab_local) * world;
  // MMA column 2t + col of this lane's accumulators is request row 2t + col.
  const bool active[2] = {2 * t < rows, 2 * t + 1 < rows};

  uint4 lo[kChunks], hi[kChunks];
  if (first_tile < tiles)
    LoadTile(lo, hi, head, first_tile, lane);

  // Base logits and noise of this lane's resident tile rows and request rows
  // for the current step; the next step's are loaded while the exchange is in
  // flight.
  const bool resident = first_tile < tiles;
  float base[2][2] = {{0.0F, 0.0F}, {0.0F, 0.0F}};
  float gumbel[2][2] = {{0.0F, 0.0F}, {0.0F, 0.0F}};
  auto load_inputs = [&](int step) {
    if (!resident)
      return;
#pragma unroll
    for (int col = 0; col < 2; ++col) {
      if (!active[col])
        continue;
      const int r = 2 * t + col;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = first_tile * 16 + g + 8 * half;
        base[col][half] =
            ToFloat(base_logits[r * base_row_stride + step * base_step_stride +
                                base_column + row]);
        gumbel[col][half] =
            noise[static_cast<i64>(r * STEPS + step) * vocab_local + row];
      }
    }
  };
  load_inputs(0);

  if (threadIdx.x == 0)
    s_count = *reinterpret_cast<volatile u32 *>(&control[kLaunchWord]);
  if (warp < kMaxRows) {
    uint4 value = make_uint4(0u, 0u, 0u, 0u);
    if (warp < rows) {
      const int anchor = input_ids[warp * query_per_row];
      value = reinterpret_cast<const uint4 *>(
          embed_table + static_cast<i64>(anchor) * kRankDim)[lane];
    }
    reinterpret_cast<uint4 *>(embedding[warp])[lane] = value;
  }
  __syncthreads();
  const u32 tag = EpochTag(s_count);
  const int parity = static_cast<int>(s_count & 1u);
  u64 *const records = reinterpret_cast<u64 *>(PeerBase(peers, rank));
  unsigned long long *const combine =
      reinterpret_cast<unsigned long long *>(control);
  u32 *const arrivals = control + kArrivalWord;
  // Staging elements of this launch's parity: [max_rows][STEPS][vocab].
  const i64 staging_parity =
      kStagingOffset / static_cast<i64>(sizeof(CACHE_T)) +
      static_cast<i64>(parity) * max_rows * STEPS * vocab;

#pragma unroll 1
  for (int step = 0; step < STEPS; ++step) {
    const bool last_step = step + 1 == STEPS;
    float temp[2];
#pragma unroll
    for (int col = 0; col < 2; ++col) {
      const int req_state =
          active[col] ? idx_mapping[(2 * t + col) * STEPS + step] : -1;
      temp[col] = req_state >= 0 ? temperature[req_state] : 0.0F;
    }
    u64 best[2] = {0ull, 0ull};
    for (int tile = first_tile; tile < tiles; tile += stride) {
      float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
      if (tile == first_tile) {
#pragma unroll
        for (int chunk = 0; chunk < kChunks; ++chunk) {
          const uint4 e =
              reinterpret_cast<const uint4 *>(embedding[g])[chunk * 4 + t];
          MmaChunk(c, lo[chunk], hi[chunk], e);
        }
      } else {
        // Tiles beyond one per warp (small grids) stream from L2.
        const __nv_bfloat16 *row_lo =
            head + static_cast<i64>(tile * 16 + g) * kRankDim + t * 8;
        for (int chunk = 0; chunk < kChunks; ++chunk) {
          const uint4 e =
              reinterpret_cast<const uint4 *>(embedding[g])[chunk * 4 + t];
          const uint4 a = *reinterpret_cast<const uint4 *>(row_lo + chunk * 32);
          const uint4 b = *reinterpret_cast<const uint4 *>(
              row_lo + 8 * kRankDim + chunk * 32);
          MmaChunk(c, a, b, e);
        }
      }
#pragma unroll
      for (int col = 0; col < 2; ++col) {
        if (!active[col])
          continue;
        const int r = 2 * t + col;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int row = tile * 16 + g + 8 * half;
          const int index = vocab_offset + row;
          const float base_value =
              tile == first_tile ? base[col][half]
                                 : ToFloat(base_logits[r * base_row_stride +
                                                       step * base_step_stride +
                                                       base_column + row]);
          const float noise_value =
              tile == first_tile
                  ? gumbel[col][half]
                  : noise[static_cast<i64>(r * STEPS + step) * vocab_local +
                          row];
          // The bias GEMM's BF16 output, then the eager add's fp32 sum rounded
          // once to the cache dtype.
          const float bias =
              __bfloat162float(__float2bfloat16_rn(c[2 * half + col]));
          CACHE_T logit;
          FromFloat(AddRn(base_value, bias), logit);
          tile_logits[warp][r][g + 8 * half] = logit;
          const float value = ToFloat(logit);
          const float score =
              temp[col] != 0.0F ? AddRn(DivFull(value, temp[col]), noise_value)
                                : value;
          const u64 key = CandidateKey(score, index);
          best[col] = key > best[col] ? key : best[col];
        }
      }
      // The tile's logits leave as 16-byte vectors: one per lane and request
      // row for the local cache and for each peer's staging buffer.
      __syncwarp();
      if (lane < rows * kTileVectors) {
        const int r = lane / kTileVectors;
        const int vector = lane - r * kTileVectors;
        const int req_state = idx_mapping[r * STEPS + step];
        if (req_state >= 0) {
          const uint4 v =
              reinterpret_cast<const uint4 *>(tile_logits[warp][r])[vector];
          const i64 element =
              vocab_offset + tile * 16 + vector * kVectorElements;
          *reinterpret_cast<uint4 *>(logits_cache + req_state * cache_stride0 +
                                     step * cache_stride1 + element) = v;
          const i64 staged = staging_parity +
                             static_cast<i64>(r * STEPS + step) * vocab +
                             element;
          for (int peer = 0; peer < world; ++peer) {
            if (peer == rank)
              continue;
            *reinterpret_cast<uint4 *>(
                reinterpret_cast<CACHE_T *>(PeerBase(peers, peer)) + staged) =
                v;
          }
        }
      }
      __syncwarp();
    }
    if (step + 1 < STEPS)
      load_inputs(step + 1);

    // Each request row's best key over the warp (lanes sharing t), then the
    // CTA.
#pragma unroll
    for (int col = 0; col < 2; ++col) {
#pragma unroll
      for (int offset = 4; offset < 32; offset <<= 1) {
        const u64 other = __shfl_xor_sync(0xffffffffu, best[col], offset);
        best[col] = other > best[col] ? other : best[col];
      }
    }
    if (g == 0) {
      warp_best[warp][2 * t] = best[0];
      warp_best[warp][2 * t + 1] = best[1];
    }
    __syncthreads();
    if (warp == 0) {
      if (lane < rows) {
        u64 key = 0ull;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
          key = warp_best[w][lane] > key ? warp_best[w][lane] : key;
        atomicMax(combine + step * kMaxRows + lane,
                  static_cast<unsigned long long>(key));
      }
      __syncwarp();
      int last = 0;
      if (lane == 0) {
        // The last step's fence also publishes this CTA's peer stores of every
        // step before its arrival is counted.
        if (last_step) {
          __threadfence_system();
        } else {
          __threadfence();
        }
        last = atomicAdd(arrivals + step, 1u) == static_cast<u32>(ctas - 1);
      }
      last = __shfl_sync(0xffffffffu, last, 0);
      // Lane L handles request row L / world and rank L % world.
      const int r = lane / world;
      const int other_rank = lane - r * world;
      const bool exchanging = lane < rows * world;
      // Post every row slot up to max_rows (rows past `rows` carry the zero
      // key) so each slot is rewritten on every launch of its parity. Graphs of
      // different row counts share the launch counter; a slot left from a
      // larger launch could otherwise carry a tag that matches a later one.
      const bool posting = lane < max_rows * world;
      const i64 record =
          ((static_cast<i64>(parity) * kMaxSteps + step) * kMaxRows + r) *
          kMaxRanks;
      if (last) {
        if (lane == 0)
          __threadfence();
        __syncwarp();
        u64 rank_key = 0ull;
        if (lane < rows)
          rank_key = atomicExch(combine + step * kMaxRows + lane, 0ull);
        if (lane == 0)
          atomicExch(arrivals + step, 0u);
        const u64 mine = __shfl_sync(0xffffffffu, rank_key, r);
        if (posting) {
          // Only the last step's record must publish (system scope) the logits
          // this rank pushed to its peers; a system fence on the other steps
          // would wait for those stores to drain.
          if (last_step)
            __threadfence_system();
          StoreRecord(reinterpret_cast<u64 *>(PeerBase(peers, other_rank)) +
                          record + rank,
                      (mine << kTagBits) | tag);
        }
      }
      u64 global = 0ull;
      if (exchanging) {
        global = AwaitRecord(records + record + other_rank, tag, timeout_ns,
                             step, other_rank);
        if (last_step) {
          __threadfence_system();
        } else if (global != 0ull) {
          // Each rank's candidate may be the winner: start its embedding row's
          // HBM read while the other ranks' records are still in flight.
          PrefetchRowL2(embed_table +
                        static_cast<i64>(KeyIndex(global)) * kRankDim);
        }
      }
      for (int offset = 1; offset < world; offset <<= 1) {
        const u64 other = __shfl_xor_sync(0xffffffffu, global, offset);
        global = other > global ? other : global;
      }
      if (exchanging && other_rank == 0)
        winner[r] = global;
    }
    __syncthreads();
    if (blockIdx.x == 0 && static_cast<int>(threadIdx.x) < rows) {
      draft_tokens[threadIdx.x * draft_stride + step] =
          KeyIndex(winner[threadIdx.x]);
    }
    if (step + 1 < STEPS && warp < rows) {
      reinterpret_cast<uint4 *>(embedding[warp])[lane] =
          reinterpret_cast<const uint4 *>(
              embed_table +
              static_cast<i64>(KeyIndex(winner[warp])) * kRankDim)[lane];
    }
    __syncthreads();
  }

  // The peers' shards are in this rank's staging buffer: copy them into the
  // cache.
  {
    const int vectors = vocab_local * static_cast<int>(sizeof(CACHE_T)) / 16;
    const int per_row_step = (world - 1) * vectors;
    const int total = rows * STEPS * per_row_step;
    for (int task = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         task < total; task += static_cast<int>(gridDim.x * blockDim.x)) {
      const int row_step = task / per_row_step;
      const int within = task - row_step * per_row_step;
      int peer = within / vectors;
      const int vector = within - peer * vectors;
      peer += peer >= rank ? 1 : 0;
      const int req_state = idx_mapping[row_step];
      if (req_state < 0)
        continue;
      const int step = row_step % STEPS;
      const i64 element = static_cast<i64>(peer) * vocab_local +
                          static_cast<i64>(vector) * kVectorElements;
      const CACHE_T *staged =
          reinterpret_cast<const CACHE_T *>(PeerBase(peers, rank)) +
          staging_parity + static_cast<i64>(row_step) * vocab + element;
      const uint4 v = __ldcg(reinterpret_cast<const uint4 *>(staged));
      *reinterpret_cast<uint4 *>(logits_cache + req_state * cache_stride0 +
                                 step * cache_stride1 + element) = v;
    }
  }

  if (threadIdx.x == 0) {
    __threadfence();
    const u32 previous = atomicAdd(&control[kEndWord], 1u);
    if (previous == gridDim.x - 1) {
      control[kEndWord] = 0u;
      control[kLaunchWord] = s_count + 1u;
    }
  }
}
