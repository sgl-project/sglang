// QSA prefill block selection without the fp32 scores in memory (AMD CDNA, wave64).
//
// Writes the token selection that the relative TileLang scoring (_tilelang_qsa_mqa_prefill_relative_kernel in
// srt/layers/attention/qsa/mqa.py, unscaled), topk_transform_ragged_v2 with deterministic ties and
// expand_qsa_block_indices write together, for bf16 at 4 heads of 128 dims, without the fp32 [rows, width] scores
// between them. Ties at the k-th score go to the smallest block indices; blocks are written in ascending order.
//
// stream_kernel: a CTA of 512 threads scores 32 query rows over the union of their key ranges, 64 keys a tile,
//   with the TileLang kernel's arithmetic: mfma_f32_16x16x16bf16_1k with the Q fragment first, eight 16-dim steps
//   in order, its LDS layout and swizzle, max(x, 0) per head, then (((0 + h0) + h1) + h2) + h3, so every score has
//   the bits it has there (the tile partition does not enter a score). Q stays in registers; two K buffers in LDS
//   and a register prefetch two tiles ahead. Row b keeps candidates keys[b, 0:count[b]) (order keys) and
//   cols[b, 0:count[b]) (columns relative to starts[b]; widths up to 65,536):
//   - a score is appended when its composite key R = (order key << 16) | (0xFFFF - column) is above the row's
//     threshold theta, so R orders the columns as the selection does and the row's selection is its kTopK
//     largest R;
//   - theta stays below the kTopK-th largest R of the columns seen so far, so the candidates hold that selection at
//     every point and at the end;
//   - before a tile, when a row of the CTA could not take a whole tile (more than kCap - 64 candidates), every row
//     above kLow candidates compacts: an 8-bit histogram of its keys over [lo, hi] (each lane tracks the bounds of
//     what it appended) finds the bin holding the kTopK-th key; when the keys from that bin up number at most kLow
//     the row keeps them and theta becomes (bin edge << 16) - 1, else the bin is refined; a single tied key value
//     that overflows kLow keeps exactly the kTopK largest R and theta becomes the smallest of them minus 1.
// select_kernel: a block of 256 threads a row takes the kTopK largest R of the row's candidates (a superset of the
//   selection in any order gives the same threshold and ties) and writes their blocks in ascending order. Rows of
//   at most kTopK blocks take 0..length-1. Tokens are expanded with expand_qsa_block_indices' arithmetic.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <climits>

namespace sglang {

namespace qsa_prefill_fused_topk_detail {

typedef short bf16x4 __attribute__((__vector_size__(4 * sizeof(short))));
typedef float f32x4 __attribute__((__vector_size__(4 * sizeof(float))));

constexpr int kThreads = 512;
constexpr int kRows = 32;
constexpr int kTile = 64;
constexpr int kTileElems = kTile * 128;
constexpr int kHistBins = 256;
constexpr int kBatch = 8;  // steps of a row's candidates a compaction pass loads before using them
constexpr int kSmem = 2 * kTileElems * 2 + kRows * kHistBins * 4;

struct StreamParams {
  const uint16_t* __restrict__ q;      // [rows * 4, 128] bf16
  const uint16_t* __restrict__ k;      // [keys, 128] bf16
  const int32_t* __restrict__ starts;  // [rows]
  const int32_t* __restrict__ ends;    // [rows]
  uint32_t* __restrict__ keys;         // [rows, kCap]
  uint16_t* __restrict__ cols;         // [rows, kCap]
  int32_t* __restrict__ count;         // [rows]
  uint2* __restrict__ bounds;          // [rows] (lo, hi) of the row's keys
  int nkeys, rows;
};

SGL_DEVICE int swz_w(int tx) {
  return ((((tx & 127) >> 6) + ((tx & 7) >> 2)) & 1) * 32 + ((((tx & 63) >> 5) + ((tx & 3) >> 1)) & 1) * 16 +
         ((((tx & 31) >> 4) + (tx & 1)) & 1) * 8;
}
SGL_DEVICE int swz_r(int tx, int ki) {
  return ((((tx & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32 + ((((tx & 3) >> 1) + (ki & 1)) & 1) * 16 +
         ((((tx & 63) >> 5) + (tx & 1)) & 1) * 8 + ((tx & 31) >> 4) * 4;
}
SGL_DEVICE uint32_t order_key(float x) {
  const uint32_t b = __float_as_uint(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
// Among equal scores the smaller column ranks higher.
SGL_DEVICE uint32_t tie_rank(int c) {
  return 0xFFFFu - static_cast<uint32_t>(c);
}

SGL_DEVICE void load_tile(uint4 (&r)[2], const uint16_t* __restrict__ k, int key_base, int nkeys, int tx) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int key = key_base + i * 32 + (tx >> 4);
    r[i] = key < nkeys ? *reinterpret_cast<const uint4*>(k + static_cast<int64_t>(key) * 128 + (tx & 15) * 8)
                       : make_uint4(0u, 0u, 0u, 0u);
  }
}
SGL_DEVICE void store_tile(uint16_t* kb, const uint4 (&r)[2], int tx) {
#pragma unroll
  for (int i = 0; i < 2; ++i)
    *reinterpret_cast<uint4*>(kb + ((tx & 15) >> 3) * 4096 + i * 2048 + (tx >> 4) * 64 + swz_w(tx)) = r[i];
}

SGL_DEVICE uint32_t group_min(uint32_t v) {
#pragma unroll
  for (int d = 8; d > 0; d >>= 1)
    v = min(v, static_cast<uint32_t>(__shfl_xor(static_cast<int>(v), d, 16)));
  return v;
}
SGL_DEVICE uint32_t group_max(uint32_t v) {
#pragma unroll
  for (int d = 8; d > 0; d >>= 1)
    v = max(v, static_cast<uint32_t>(__shfl_xor(static_cast<int>(v), d, 16)));
  return v;
}

SGL_DEVICE void wave_lds_fence() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
}

// The digit t of a row's 256-bin histogram with suffix(t) >= need > suffix(t + 1); *above_out = suffix(t + 1),
// *bin_out = hist[t], to every lane of the row. Lane l owns bins [240 - 16 l, 256 - 16 l).
SGL_DEVICE void group_find_digit(const int* hist, int need, int* t_out, int* above_out, int* bin_out) {
  const int l = threadIdx.x & 15;
  const int b_lo = kHistBins - 16 * (l + 1);
  int local[16];
  int sum = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    local[i] = hist[b_lo + i];
    sum += local[i];
  }
  int incl = sum;
#pragma unroll
  for (int d = 1; d < 16; d <<= 1) {
    const int u = __shfl_up(incl, d, 16);
    if (l >= d) incl += u;
  }
  int above = incl - sum;
  int t = -1, ab = 0, bc = 0;
#pragma unroll
  for (int i = 15; i >= 0; --i) {
    if (above < need && above + local[i] >= need) {
      t = b_lo + i;
      ab = above;
      bc = local[i];
    }
    above += local[i];
  }
  const uint64_t found = __ballot(t >= 0);
  const int lane = threadIdx.x & 63;
  const int src = __ffsll(static_cast<long long>(found & (0xFFFFull << (lane & 48)))) - 1;
  *t_out = __shfl(t, src, 64);
  *above_out = __shfl(ab, src, 64);
  *bin_out = __shfl(bc, src, 64);
}

template <int kTopK, int kCap>
__global__ __launch_bounds__(kThreads, 4) void stream_kernel(const StreamParams p) {
  static_assert(kCap % (16 * kBatch) == 0 && kCap >= kTopK + 128 + kTile);
  constexpr int kLow = kTopK + 128;
  extern __shared__ __align__(1024) unsigned char smem[];
  uint16_t* lds = reinterpret_cast<uint16_t*>(smem);
  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  const int lane = tx & 63;
  const int l = tx & 15;
  int* hist = reinterpret_cast<int*>(lds + 2 * kTileElems) + (tx >> 4) * kHistBins;

  int start_min = INT_MAX, end_max = INT_MIN;
  for (int qi = 0; qi < kRows; ++qi) {
    const int r = bx * kRows + qi;
    if (r < p.rows) {
      start_min = min(start_min, min(p.starts[r], p.nkeys));
      end_max = max(end_max, min(p.ends[r], p.nkeys));
    }
  }
  const int tiles = max((end_max - start_min + kTile - 1) / kTile, 0);

  const int row = bx * kRows + (tx >> 4);
  const bool row_ok = row < p.rows;
  const int rstart = row_ok ? p.starts[row] : 0;
  const int rlen = row_ok ? p.ends[row] - rstart : 0;
  // One 32-bit row offset for both arrays (rows a launch * kCap < 2^31: the caller's row chunks).
  const int rbase = (row_ok ? row : 0) * kCap;
  uint32_t* __restrict__ keys = p.keys;
  uint16_t* __restrict__ cols = p.cols;
  const uint64_t gmask = 0xFFFFull << (lane & 48);
  const uint64_t below = (1ull << lane) - 1ull;
  int cnt = 0;
  int64_t theta = -1;
  // Bounds of the row's keys: each lane's min and max of what it appended; after a compaction lo is the cut,
  // which every kept and later appended key reaches.
  uint32_t lane_lo = 0xFFFFFFFFu, lane_hi = 0u;

  // Q: 128 (row, head) rows of 128 dims through LDS into registers, one bf16x4 fragment a 16-dim step.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (bx * 128 + i * 32 + (tx >> 4) < p.rows * 4)
      v = *reinterpret_cast<const uint4*>(p.q + static_cast<int64_t>(bx) * 16384 + i * 4096 + tx * 8);
    *reinterpret_cast<uint4*>(lds + ((tx & 15) >> 3) * 8192 + i * 2048 + (tx >> 4) * 64 + swz_w(tx)) = v;
  }
  __syncthreads();
  bf16x4 qf[8];
#pragma unroll
  for (int ki = 0; ki < 8; ++ki)
    qf[ki] =
        *reinterpret_cast<const bf16x4*>(lds + (ki >> 2) * 8192 + (tx >> 6) * 1024 + (tx & 15) * 64 + swz_r(tx, ki));
  __syncthreads();

  // Last tile first: a row's latest keys tend to score highest, so theta rises sooner (fewer appends).
  auto tile_base = [&](int it) { return start_min + (tiles - 1 - it) * kTile; };

  // Exact cut among the row's keys equal to thr: the `need` largest tie ranks; returns the smallest taken.
  auto tie_cut = [&](uint32_t thr, int need) {
    uint32_t base = 0u, range = 0xFFFFu;
    int shift = 8;
#pragma unroll 1
    for (;;) {
#pragma unroll
      for (int i = 0; i < 16; ++i)
        hist[l * 16 + i] = 0;
      wave_lds_fence();
      for (int m0 = 0; m0 * 16 < cnt; m0 += kBatch) {
        uint32_t v[kBatch], c[kBatch];
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
          const int e = l + 16 * (m0 + j);
          v[j] = e < cnt ? keys[rbase + e] : 0u;
          c[j] = e < cnt ? cols[rbase + e] : 0u;
        }
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
          const int e = l + 16 * (m0 + j);
          const uint32_t r = tie_rank(static_cast<int>(c[j]));
          const uint32_t d = r - base;
          if (e < cnt && v[j] == thr && r >= base && d <= range) atomicAdd(&hist[d >> shift], 1);
        }
      }
      wave_lds_fence();
      int t, ab, bc;
      group_find_digit(hist, need, &t, &ab, &bc);
      wave_lds_fence();
      need -= ab;
      base += static_cast<uint32_t>(t) << shift;
      range = min(range - (static_cast<uint32_t>(t) << shift), (1u << shift) - 1u);
      if (shift == 0) break;
      shift = 0;
    }
    return base;
  };

  auto compact = [&]() {
    const uint32_t lo = group_min(lane_lo);
    const uint32_t hi = group_max(lane_hi);
    uint32_t base = lo, range = hi - lo;
    int shift = max(0, 32 - __clz(static_cast<int>(range)) - 8);
    int need = kTopK;  // still to take inside [base, base + range]; kTopK - need lie above it
    uint32_t thr, rank_thr = 0u;
    bool exact = false;
#pragma unroll 1
    for (;;) {
#pragma unroll
      for (int i = 0; i < 16; ++i)
        hist[l * 16 + i] = 0;
      wave_lds_fence();
      for (int m0 = 0; m0 * 16 < cnt; m0 += kBatch) {
        uint32_t v[kBatch];
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
          const int e = l + 16 * (m0 + j);
          v[j] = e < cnt ? keys[rbase + e] : 0u;
        }
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
          const int e = l + 16 * (m0 + j);
          const uint32_t d = v[j] - base;
          if (e < cnt && v[j] >= base && d <= range) atomicAdd(&hist[d >> shift], 1);
        }
      }
      wave_lds_fence();
      int t, ab, bc;
      group_find_digit(hist, need, &t, &ab, &bc);
      wave_lds_fence();
      const uint32_t edge = base + (static_cast<uint32_t>(t) << shift);
      // Keys >= edge: the kTopK - need above the window, ab above bin t inside it, and bin t.
      if (kTopK - need + ab + bc <= kLow) {
        thr = edge;
        break;
      }
      need -= ab;
      if (shift == 0) {  // bin t is the one value edge, and its ties overflow kLow
        thr = edge;
        exact = true;
        rank_thr = tie_cut(thr, need);
        break;
      }
      base = edge;
      range = min(range - (static_cast<uint32_t>(t) << shift), (1u << shift) - 1u);
      shift = max(0, shift - 8);
    }
    // Keep in place: a batch's writes land at or below the entries it read.
    int dst = 0;
    for (int m0 = 0; m0 * 16 < cnt; m0 += kBatch) {
      uint32_t v[kBatch], c[kBatch];
#pragma unroll
      for (int j = 0; j < kBatch; ++j) {
        const int e = l + 16 * (m0 + j);
        v[j] = e < cnt ? keys[rbase + e] : 0u;
        c[j] = e < cnt ? cols[rbase + e] : 0u;
      }
#pragma unroll
      for (int j = 0; j < kBatch; ++j) {
        const int e = l + 16 * (m0 + j);
        bool keep = e < cnt && v[j] >= thr;
        if (exact && keep && v[j] == thr) keep = tie_rank(static_cast<int>(c[j])) >= rank_thr;
        const uint64_t bm = __ballot(keep);
        if (keep) {
          const int at = dst + __popcll(bm & gmask & below);
          keys[rbase + at] = v[j];
          cols[rbase + at] = static_cast<uint16_t>(c[j]);
        }
        dst += __popcll(bm & gmask);
      }
    }
    cnt = dst;
    theta = ((static_cast<int64_t>(thr) << 16) | rank_thr) - 1;
    lane_lo = thr;
  };

  auto score_tile = [&](f32x4(&acc)[4], const uint16_t* kb) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      acc[j] = f32x4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ki = 0; ki < 8; ++ki) {
      bf16x4 a[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        a[i] = *reinterpret_cast<const bf16x4*>(kb + (ki >> 2) * 4096 + i * 1024 + (tx & 15) * 64 + swz_r(tx, ki));
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc[i] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(qf[ki], a[i], acc[i], 0, 0, 0);
    }
  };

  // acc[i][h]: head h of row (tx >> 4) against key key_base + 16 i + (tx & 15). A row's 16 lanes append in lane
  // order through a ballot.
  auto append = [&](const f32x4(&acc)[4], int key_base) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float score = 0.f;
#pragma unroll
      for (int h = 0; h < 4; ++h)
        score = score + max(acc[i][h], 0.f);
      const int column = key_base + i * 16 + (tx & 15) - rstart;
      const uint32_t okey = order_key(score);
      const bool keep = column >= 0 && column < rlen && ((static_cast<int64_t>(okey) << 16) | tie_rank(column)) > theta;
      const uint64_t m = __ballot(keep);
      if (keep) {
        const int at = cnt + __popcll(m & gmask & below);
        keys[rbase + at] = okey;
        cols[rbase + at] = static_cast<uint16_t>(column);
        lane_lo = min(lane_lo, okey);
        lane_hi = max(lane_hi, okey);
      }
      cnt += __popcll(m & gmask);
    }
  };

  // Tile it is in K buffer it & 1; tile it + 1 waits in registers and tile it + 2 loads during tile it.
  uint4 next[2], after[2];
  if (tiles > 0) {
    load_tile(next, p.k, tile_base(0), p.nkeys, tx);
    store_tile(lds, next, tx);
  }
  if (tiles > 1) load_tile(next, p.k, tile_base(1), p.nkeys, tx);
  for (int it = 0; it < tiles; ++it) {
    // The barrier publishes tile it and frees buffer (it + 1) & 1; a row that could not take a whole tile makes
    // it an epoch (the fence orders a row's appends, by its 16 lanes, before they are read back).
    if (__syncthreads_or(cnt > kCap - kTile)) {
      __threadfence_block();
      if (cnt > kLow) compact();
    }
    if (it + 2 < tiles) load_tile(after, p.k, tile_base(it + 2), p.nkeys, tx);
    f32x4 acc[4];
    score_tile(acc, lds + (it & 1) * kTileElems);
    append(acc, tile_base(it));
    if (it + 1 < tiles) store_tile(lds + ((it + 1) & 1) * kTileElems, next, tx);
    next[0] = after[0];
    next[1] = after[1];
  }
  const uint32_t lo = group_min(lane_lo), hi = group_max(lane_hi);
  if (l == 0 && row_ok) {
    p.count[row] = cnt;
    p.bounds[row] = make_uint2(lo, hi);
  }
}

constexpr int kSelThreads = 256;
constexpr int kWave = 64;
constexpr int kWaves = kSelThreads / kWave;
constexpr int kRadixBits = 11;
constexpr int kBins = 1 << kRadixBits;
constexpr int kMapWords = (1 << 16) / 32;  // a bit per column

struct SelectParams {
  const uint32_t* __restrict__ keys;      // [B, kCap]
  const uint16_t* __restrict__ cols;      // [B, kCap]
  const int32_t* __restrict__ count;      // [B]
  const uint2* __restrict__ bounds;       // [B]
  const int32_t* __restrict__ starts;     // [B]
  const int32_t* __restrict__ ends;       // [B]
  const int64_t* __restrict__ positions;  // [B]
  const int32_t* __restrict__ seqlens;    // [B]
  int32_t* __restrict__ out;              // [B, out_stride]
  int64_t out_stride;
};

// Exclusive prefix sum over the block in thread order; *total gets the block sum. Calls are separated by
// barriers (s_wave is reused).
SGL_DEVICE int block_exclusive_scan(int v, int* s_wave, int* total) {
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  int incl = v;
#pragma unroll
  for (int d = 1; d < kWave; d <<= 1) {
    const int u = __shfl_up(incl, d, kWave);
    if (lane >= d) incl += u;
  }
  if (lane == kWave - 1) s_wave[wave] = incl;
  __syncthreads();
  int before = 0, sum = 0;
#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
    const int x = s_wave[w];
    before += w < wave ? x : 0;
    sum += x;
  }
  __syncthreads();
  *total = sum;
  return incl - v + before;
}

template <bool kMax>
SGL_DEVICE uint32_t block_reduce(uint32_t v, uint32_t* s_red) {
#pragma unroll
  for (int d = kWave / 2; d > 0; d >>= 1) {
    const auto u = static_cast<uint32_t>(__shfl_xor(static_cast<int>(v), d, kWave));
    v = kMax ? max(v, u) : min(v, u);
  }
  if (threadIdx.x % kWave == 0) s_red[threadIdx.x / kWave] = v;
  __syncthreads();
  uint32_t r = s_red[0];
#pragma unroll
  for (int w = 1; w < kWaves; ++w)
    r = kMax ? max(r, s_red[w]) : min(r, s_red[w]);
  __syncthreads();
  return r;
}

// The value v among vals[j] (where in[j]) with #(>= v) >= need > #(> v): a radix select over [lo, hi] with
// 11-bit digits of (value - base). need >= 1 and at least need values in [lo, hi].
template <int kPer>
__device__ uint32_t select_kth_largest(
    const uint32_t (&vals)[kPer],
    const bool (&in)[kPer],
    int need,
    uint32_t lo,
    uint32_t hi,
    int* s_hist,
    int* s_wave,
    int* s_digit,
    int* s_above) {
  uint32_t base = lo;
  uint32_t range = hi - lo;
  int shift = max(0, 32 - __clz(static_cast<int>(range)) - kRadixBits);
#pragma unroll 1
  while (range != 0) {
    for (int i = threadIdx.x; i < kBins; i += kSelThreads)
      s_hist[i] = 0;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const uint32_t d = vals[j] - base;
      if (in[j] && vals[j] >= base && d <= range) atomicAdd(&s_hist[d >> shift], 1);
    }
    __syncthreads();
    constexpr int kPerBin = kBins / kSelThreads;
    const int b_lo = kBins - (threadIdx.x + 1) * kPerBin;
    int local[kPerBin];
    int sum = 0;
#pragma unroll
    for (int i = 0; i < kPerBin; ++i) {
      local[i] = s_hist[b_lo + i];
      sum += local[i];
    }
    int total;
    int above = block_exclusive_scan(sum, s_wave, &total);
#pragma unroll
    for (int i = kPerBin - 1; i >= 0; --i) {
      if (above < need && above + local[i] >= need) {
        *s_digit = b_lo + i;
        *s_above = above;
      }
      above += local[i];
    }
    __syncthreads();
    const uint32_t digit = static_cast<uint32_t>(*s_digit) << shift;
    need -= *s_above;
    base += digit;
    range = min(range - digit, (1u << shift) - 1u);
    __syncthreads();
    if (shift == 0) break;
    shift = max(0, shift - kRadixBits);
  }
  return base;
}

template <int kTopK, int kCap, int kRatio, int kTokenTopK>
__global__ __launch_bounds__(kSelThreads) void select_kernel(const SelectParams p) {
  static_assert(kCap % kSelThreads == 0 && kTopK <= kCap && kTokenTopK <= kTopK * kRatio);
  constexpr int kPer = kCap / kSelThreads;
  constexpr int kFinal = kTokenTopK + kRatio - 1;
  __shared__ int s_hist[kBins];
  __shared__ uint32_t s_map[kMapWords];
  __shared__ int s_pref[kMapWords];
  __shared__ int s_sel[kTopK];
  __shared__ int s_wave[kWaves];
  __shared__ uint32_t s_red[kWaves];
  __shared__ int s_digit, s_above;

  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  const int len = p.ends[row] - p.starts[row];

  if (len <= kTopK) {
    for (int i = tx; i < kTopK; i += kSelThreads)
      s_sel[i] = i < len ? i : -1;
  } else {
    const int n = p.count[row];
    const uint32_t* ksrc = p.keys + static_cast<int64_t>(row) * kCap;
    const uint16_t* csrc = p.cols + static_cast<int64_t>(row) * kCap;
    uint32_t key[kPer];
    int col[kPer];
    bool in[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int e = tx + j * kSelThreads;
      in[j] = e < n;
      key[j] = in[j] ? ksrc[e] : 0u;
      col[j] = in[j] ? static_cast<int>(csrc[e]) : 0;
    }
    const uint2 b = p.bounds[row];  // stream_kernel's [lo, hi] of the row's keys
    const uint32_t thr = select_kth_largest(key, in, kTopK, b.x, b.y, s_hist, s_wave, &s_digit, &s_above);

    // Ties at thr: the `need` smallest columns, i.e. the `need` largest tie ranks.
    int n_above = 0, n_tie = 0;
    uint32_t tmin = 0xFFFFFFFFu, tmax = 0u;
    uint32_t rank[kPer];
    bool tie[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      tie[j] = in[j] && key[j] == thr;
      rank[j] = tie_rank(col[j]);
      n_above += in[j] && key[j] > thr;
      n_tie += tie[j];
      if (tie[j]) {
        tmin = min(tmin, rank[j]);
        tmax = max(tmax, rank[j]);
      }
    }
    int total_above, total_tie;
    block_exclusive_scan(n_above, s_wave, &total_above);
    block_exclusive_scan(n_tie, s_wave, &total_tie);
    const int need = kTopK - total_above;
    uint32_t rank_thr = 0u;
    if (total_tie > need) {
      tmin = block_reduce<false>(tmin, s_red);
      tmax = block_reduce<true>(tmax, s_red);
      rank_thr = select_kth_largest(rank, tie, need, tmin, tmax, s_hist, s_wave, &s_digit, &s_above);
    }

    // Output slot of a taken block: the number of taken blocks with a smaller column.
    for (int i = tx; i < kMapWords; i += kSelThreads)
      s_map[i] = 0u;
    __syncthreads();
    bool taken[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      taken[j] = (in[j] && key[j] > thr) || (tie[j] && rank[j] >= rank_thr);
      if (taken[j]) atomicOr(&s_map[col[j] >> 5], 1u << (col[j] & 31));
    }
    __syncthreads();
    constexpr int kWordsPer = kMapWords / kSelThreads;
    int wsum = 0;
#pragma unroll
    for (int i = 0; i < kWordsPer; ++i)
      wsum += __popc(s_map[tx * kWordsPer + i]);
    int total_sel;
    int run = block_exclusive_scan(wsum, s_wave, &total_sel);
#pragma unroll
    for (int i = 0; i < kWordsPer; ++i) {
      s_pref[tx * kWordsPer + i] = run;
      run += __popc(s_map[tx * kWordsPer + i]);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      if (taken[j]) {
        const int w = col[j] >> 5;
        s_sel[s_pref[w] + __popc(s_map[w] & ((1u << (col[j] & 31)) - 1u))] = col[j];
      }
    }
  }
  __syncthreads();

  // Token expansion: expand_qsa_block_indices' arithmetic.
  const int32_t seqlen = p.seqlens[row];
  const int64_t visible = p.positions[row] + 1;
  const int64_t tail_start = (visible / kRatio) * kRatio;
  const int64_t tail_count = visible - tail_start;
  const int valid_blocks = len < kTopK ? len : kTopK;
  const int valid_tokens = min(valid_blocks * kRatio, kTokenTopK);
  int32_t* dst = p.out + static_cast<int64_t>(row) * p.out_stride;
  for (int c = tx; c < kFinal; c += kSelThreads) {
    const int block = (c < kTokenTopK && c / kRatio < kTopK) ? s_sel[c / kRatio] : -1;
    const int64_t expanded = static_cast<int64_t>(block) * kRatio + c % kRatio;
    const bool expanded_valid = c < kTokenTopK && block >= 0 && expanded >= 0 && expanded < seqlen;
    const int64_t tail_offset = c - valid_tokens;
    const int64_t tail = tail_start + tail_offset;
    const bool tail_valid = tail_offset >= 0 && tail_offset < kRatio - 1 && tail_offset < tail_count && tail < seqlen;
    dst[c] = (expanded_valid && c < valid_tokens) ? static_cast<int32_t>(expanded)
                                                  : (tail_valid ? static_cast<int32_t>(tail) : -1);
  }
}

}  // namespace qsa_prefill_fused_topk_detail

template <int kTopK, int kRatio, int kTokenTopK>
struct QSAPrefillFusedTopKKernel {
  // Room for kLow = kTopK + 128 candidates after a compaction plus a tile: 1024 at kTopK 512.
  static constexpr int kCap = 2 * ((kTopK + 255) / 256) * 256;
  static constexpr int kFinal = kTokenTopK + kRatio - 1;

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView starts,
      const tvm::ffi::TensorView ends,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView seqlens,
      const tvm::ffi::TensorView keys,
      const tvm::ffi::TensorView cols,
      const tvm::ffi::TensorView count,
      const tvm::ffi::TensorView bounds,
      const tvm::ffi::TensorView out) {
    using namespace host;
    using namespace qsa_prefill_fused_topk_detail;
    auto B = SymbolicSize{"rows"};
    auto N = SymbolicSize{"keys"};
    auto OS = SymbolicSize{"out_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({B, 4, 128}).with_strides({512, 128, 1}).with_dtype<bf16_t>().with_device(device).verify(q);
    TensorMatcher({N, 128}).with_strides({128, 1}).with_dtype<bf16_t>().with_device(device).verify(k);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(starts);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(ends);
    TensorMatcher({B}).with_dtype<int64_t>().with_device(device).verify(positions);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(seqlens);
    TensorMatcher({B, kCap}).with_strides({kCap, 1}).with_dtype<int32_t>().with_device(device).verify(keys);
    TensorMatcher({B, kCap}).with_strides({kCap, 1}).with_dtype<int16_t>().with_device(device).verify(cols);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(count);
    TensorMatcher({B, 2}).with_strides({2, 1}).with_dtype<int32_t>().with_device(device).verify(bounds);
    TensorMatcher({B, kFinal}).with_strides({OS, 1}).with_dtype<int32_t>().with_device(device).verify(out);
    const auto rows = B.unwrap();
    const auto nkeys = N.unwrap();
    RuntimeCheck(
        rows * kCap <= INT_MAX && nkeys <= INT_MAX, "QSA prefill fused top-k: ", rows, " rows, ", nkeys, " keys");
    if (rows == 0) return;
    const auto stream_params = StreamParams{
        .q = static_cast<const uint16_t*>(q.data_ptr()),
        .k = static_cast<const uint16_t*>(k.data_ptr()),
        .starts = static_cast<const int32_t*>(starts.data_ptr()),
        .ends = static_cast<const int32_t*>(ends.data_ptr()),
        .keys = static_cast<uint32_t*>(keys.data_ptr()),
        .cols = static_cast<uint16_t*>(cols.data_ptr()),
        .count = static_cast<int32_t*>(count.data_ptr()),
        .bounds = static_cast<uint2*>(bounds.data_ptr()),
        .nkeys = static_cast<int>(nkeys),
        .rows = static_cast<int>(rows),
    };
    const auto select_params = SelectParams{
        .keys = static_cast<const uint32_t*>(keys.data_ptr()),
        .cols = static_cast<const uint16_t*>(cols.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .bounds = static_cast<const uint2*>(bounds.data_ptr()),
        .starts = static_cast<const int32_t*>(starts.data_ptr()),
        .ends = static_cast<const int32_t*>(ends.data_ptr()),
        .positions = static_cast<const int64_t*>(positions.data_ptr()),
        .seqlens = static_cast<const int32_t*>(seqlens.data_ptr()),
        .out = static_cast<int32_t*>(out.data_ptr()),
        .out_stride = OS.unwrap(),
    };
    const auto dev = device.unwrap();
    LaunchKernel(static_cast<uint32_t>((rows + kRows - 1) / kRows), kThreads, dev, kSmem)(
        stream_kernel<kTopK, kCap>, stream_params);
    LaunchKernel(static_cast<uint32_t>(rows), kSelThreads, dev)(
        select_kernel<kTopK, kCap, kRatio, kTokenTopK>, select_params);
  }
};

}  // namespace sglang
