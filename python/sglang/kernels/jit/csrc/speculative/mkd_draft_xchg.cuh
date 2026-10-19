// Greedy MTP draft step boundary in one launch: shard argmax of the draft vocabulary-shard logits, a
// multicast push of each rank's candidate (its embedding row, or its fc memo row) plus the
// rank's hidden-half fc partial, and the selection, replacing pack + all-gather + select and
// the fc partial all-gather; with memo rows it also writes the next step's fc output (the
// mkd_fc_sum memo arithmetic). Selection rules are those of vocab_argmax_pack / _select
// (order key, -0.0 == +0.0, NaN key from the caller, lowest vocabulary id on ties).
//
// Transport: a symmetric buffer with a multicast mapping (multi-node NVLink). Every 32-bit
// payload word travels in a 64-bit word whose high half is the call's epoch, stored with
// multimem.st.relaxed.sys.b64 and polled with ld.relaxed.sys.u64, so arrival is read from
// the data itself (single-copy-atomic 8-byte accesses). Three slots rotate by epoch: a slot
// is rewritten only three calls later, after two more exchanges that each need every rank's
// packet. Waits are bounded and trap.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace mkd_xchg {

inline constexpr int kSlots = 3;
inline constexpr uint32_t kSpinLimit = 1u << 27;
// Shards of at least this many 16-byte vectors are scanned in 1/kChunks parts and combined
// through scan_part instead of a full scan per CTA (same keys and ties, so the same token);
// measured faster in A/B.
inline constexpr int kSplitScanMinVecs = 2048;
// Packed partial of a CTA that saw a NaN (no NaN-free packed key reaches it).
inline constexpr uint32_t kNanPart = 0xffffffffu;

SGL_DEVICE uint32_t order_key(float f) {
  uint32_t b = __float_as_uint(f);
  b = b == 0x80000000u ? 0u : b;
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

/// \brief Order keys of two packed bf16 (16 bits each): -0.0 as +0.0, negatives inverted.
/// For non-NaN bf16 the 32-bit key of order_key(float) is (key16 << 16) | (neg ? 0xffff : 0),
/// so comparing key16 orders like the 32-bit keys; NaN halves are flagged in nan_any.
SGL_DEVICE uint32_t packed_key(uint32_t w, uint32_t& nan_any) {
  w &= ~__vcmpeq2(w, 0x80008000u);
  nan_any |= __vcmpgtu2(w & 0x7fff7fffu, 0x7f807f80u);
  const uint32_t neg = __vcmpgtu2(w, 0x7fff7fffu);
  return (w ^ neg) | (~neg & 0x80008000u);
}

SGL_DEVICE bool better(uint32_t v, int64_t i, uint32_t bv, int64_t bi) {
  return v > bv || (v == bv && i < bi);
}

SGL_DEVICE uint32_t clear_negative_zero(uint32_t two_bf16) {
  uint32_t lo = two_bf16 & 0xffffu;
  uint32_t hi = two_bf16 >> 16;
  lo = lo == 0x8000u ? 0u : lo;
  hi = hi == 0x8000u ? 0u : hi;
  return lo | (hi << 16);
}

SGL_DEVICE void mc_store(uint64_t* addr, uint32_t payload, uint32_t tag) {
  const uint64_t w = static_cast<uint64_t>(payload) | (static_cast<uint64_t>(tag) << 32);
  asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" ::"l"(addr), "l"(w) : "memory");
}

SGL_DEVICE uint64_t now_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

}  // namespace mkd_xchg

struct MkdXchgParams {
  const bf16_t* logits;           // [V_local] vocabulary-shard logits, written by the lm_head
  const void* __restrict__ cand;  // [V_local, H] embedding rows (bf16) or memo rows (fp32)
  const int64_t* hot_ids;         // [world * V_local] vocabulary id of each shard row, rank-major
  const float* own_partial;       // [H] this rank's hidden-slice fc partial (hidden ranks only)
  const uint32_t* partial_flag;   // equals the epoch once own_partial is written, or null
  uint64_t* scan_part;            // [kSlots, kChunks] per-CTA shard-argmax partials, or null
  const uint32_t* head_part;      // [head_parts] packed partials of mkd_head (no scan), or null
  int32_t head_parts;
  uint64_t* mc;           // multicast base of the exchange buffer
  const uint64_t* uc;     // this rank's own copy of the exchange buffer
  uint32_t* epoch;        // [1] call counter shared by all ranks, starts at 1
  uint32_t* done;         // [1] CTA arrival counter, zero between calls
  void* out_cand;         // [H] winner's embedding row (bf16) or memo row (fp32)
  float* out_partials;    // [world - embed, H] hidden-slice partials in rank order
  bf16_t* fc_out;         // [H] memo only: the next step's fc output, else null
  int32_t token_only;     // 1: exchange keys only (no next step reads the payload)
  int64_t* topk_index;    // [1]
  float* topk_p;          // [1]
  int64_t* draft_tokens;  // [1, S] or null
  int64_t* positions;     // [1] or null
  int64_t* mrope;         // [3, >= 1] or null
  int32_t vocab;          // V_local, multiple of 8
  uint32_t nan_key;
  int32_t rank;
  int32_t draft_column;
  int32_t mrope_stride;
  uint64_t* trace;      // [trace_slots, kChunks, 9] %globaltimer per phase, or null
  int32_t trace_slots;  // calls kept in the trace ring (slot = epoch % trace_slots)
  const char* pf;       // static weight range prefetched into L2 for a later kernel, or null
  int64_t pf_bytes;
};

/**
 * \brief Step-boundary exchange for one token. Grid = kChunks CTAs; CTA c owns columns
 * [c * H / kChunks, ...) of the candidate and partial rows. Every CTA redoes the shard
 * argmax (a few tens of KB from L2) so no CTA waits for another before sending.
 */
template <int kH, int kWorld, bool kMemo, int kChunks, int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) mkd_xchg_kernel(const MkdXchgParams __grid_constant__ p) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  constexpr int kW = 2 * kH / kWorld;
  constexpr int kEmbed = kH / kW;
  constexpr int kCols = kH / kChunks;
  constexpr int kCandWords = kMemo ? kCols : kCols / 2;
  constexpr int kPartWords = kCols;
  constexpr int kRec = 2 + kCandWords + kPartWords;
  constexpr int kWarps = kThreads / 32;
  static_assert(kH % (kChunks * 2) == 0);

  __shared__ uint32_t s_v[kWarps];
  __shared__ int32_t s_i[kWarps];
  __shared__ uint32_t s_key[kWorld];
  __shared__ uint32_t s_cand[kWorld][kCandWords];
  __shared__ float s_part[kWorld - kEmbed][kPartWords];
  __shared__ int64_t s_gid[kWorld];
  __shared__ int s_win;
  __shared__ uint32_t s_tag;
  __shared__ uint32_t s_best;

  const int c = static_cast<int>(blockIdx.x);
  const int warp = static_cast<int>(threadIdx.x / 32);
  const int lane = static_cast<int>(get_lane_id());

  const uint64_t t_start = mkd_xchg::now_ns();
  if (p.pf != nullptr) {
    // Each CTA hints its share of the range in 64 KB bulk prefetches (static weights only).
    constexpr int64_t kPfChunk = 64 * 1024;
    const int64_t share = (p.pf_bytes + kChunks - 1) / kChunks;
    const int64_t beg = c * share, end = min(p.pf_bytes, beg + share);
    for (int64_t off = beg + static_cast<int64_t>(threadIdx.x) * kPfChunk; off < end;
         off += static_cast<int64_t>(kThreads) * kPfChunk) {
      const uint32_t size = static_cast<uint32_t>(min(kPfChunk, end - off)) & ~15u;
      if (size > 0) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.pf + off), "r"(size) : "memory");
    }
  }
  PDLWaitPrimary<kUsePDL>();
  // The fc launch behind us only prefetches static weights before its own wait.
  PDLTriggerSecondary<kUsePDL>();
  const uint64_t t_wait = mkd_xchg::now_ns();
  // Read the epoch now, publish it after the scan: the load overlaps the logits loads.
  uint32_t epoch = 0;
  if (threadIdx.x == 0) asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(epoch) : "l"(p.epoch) : "memory");

  // Shard argmax, ties to the lowest local row; hot positions order like vocabulary ids.
  // Fast pass on packed 16-bit order keys (exact for NaN-free shards, see packed_key); a shard
  // holding a NaN is rescanned with the 32-bit keys and the caller's NaN key.
  uint32_t best = 0u;  // (key16 << 16) | (0xffff - row)
  uint32_t nan_any = 0u;
  uint64_t t_ld = 0, t_cmp = 0, t_warp = 0;
  constexpr int kScanUnroll = 4;
  const int nvec = p.vocab / 8;
  const bool from_head = p.head_part != nullptr;
  const bool split = !from_head && p.scan_part != nullptr && nvec >= mkd_xchg::kSplitScanMinVecs;
  const int lo = split ? static_cast<int>((static_cast<int64_t>(nvec) * c) / kChunks) : 0;
  const int hi = from_head ? 0 : split ? static_cast<int>((static_cast<int64_t>(nvec) * (c + 1)) / kChunks) : nvec;
  for (int v0 = lo + static_cast<int>(threadIdx.x); v0 < hi; v0 += kThreads * kScanUnroll) {
    uint4 x[kScanUnroll];
#pragma unroll
    for (int u = 0; u < kScanUnroll; ++u) {
      if (v0 + u * kThreads < hi) x[u] = reinterpret_cast<const uint4*>(p.logits)[v0 + u * kThreads];
    }
    if (v0 == lo + static_cast<int>(threadIdx.x)) t_ld = mkd_xchg::now_ns();
#pragma unroll
    for (int u = 0; u < kScanUnroll; ++u) {
      const int v = v0 + u * kThreads;
      if (v < hi) {
        const uint32_t w4[4] = {x[u].x, x[u].y, x[u].z, x[u].w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint32_t key = mkd_xchg::packed_key(w4[q], nan_any);
          const uint32_t row = static_cast<uint32_t>(v * 8 + 2 * q);
          best = max(best, ((key & 0xffffu) << 16) | (0xffffu - row));
          best = max(best, (key & 0xffff0000u) | (0xffffu - row - 1));
        }
      }
    }
  }
  if (from_head && static_cast<int>(threadIdx.x) < p.head_parts) {
    // The lm_head kernel (complete after the PDL wait) left one packed maximum per CTA; every
    // thread takes one, so the block reduce below combines them in one L2 round trip.
    best = p.head_part[threadIdx.x];
  }
  t_cmp = mkd_xchg::now_ns();
  best = __reduce_max_sync(0xffffffffu, best);
  nan_any = __reduce_or_sync(0xffffffffu, nan_any);
  t_warp = mkd_xchg::now_ns();
  if (lane == 0) {
    s_v[warp] = best;
    s_i[warp] = static_cast<int32_t>(nan_any);
  }
  if (threadIdx.x == 0) s_tag = epoch;
  __syncthreads();
  static_assert(kWarps <= 32);
  best = __reduce_max_sync(0xffffffffu, lane < kWarps ? s_v[lane] : 0u);
  nan_any = __reduce_or_sync(0xffffffffu, lane < kWarps ? static_cast<uint32_t>(s_i[lane]) : 0u);
  if (from_head) {
    // kNanPart (all ones) won the max: a head CTA saw a NaN logit, so take the exact rescan.
    nan_any = best == mkd_xchg::kNanPart ? 1u : 0u;
  } else if (split) {
    // Every CTA publishes its sub-shard maximum (tagged with the epoch) and takes the max of all.
    if (warp == 0) {
      const uint32_t tg = __shfl_sync(0xffffffffu, epoch, 0);  // only thread 0 loaded it
      const int part_slot = static_cast<int>(tg % mkd_xchg::kSlots);
      if (lane == 0) {
        const uint64_t w = (static_cast<uint64_t>(tg) << 32) | (nan_any != 0u ? mkd_xchg::kNanPart : best);
        asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p.scan_part + part_slot * kChunks + c), "l"(w)
                     : "memory");
      }
      uint32_t v = 0u;
      if (lane < kChunks) {
        uint64_t w;
        uint32_t n = 0;
        do {
          asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
                       : "=l"(w)
                       : "l"(p.scan_part + part_slot * kChunks + lane)
                       : "memory");
          if (++n > mkd_xchg::kSpinLimit) __trap();
        } while (static_cast<uint32_t>(w >> 32) != tg);
        v = static_cast<uint32_t>(w);
      }
      v = __reduce_max_sync(0xffffffffu, v);
      if (lane == 0) s_best = v;
    }
    __syncthreads();
    best = s_best;
    nan_any = best == mkd_xchg::kNanPart ? 1u : 0u;
  }
  int32_t bidx = static_cast<int32_t>(0xffffu - (best & 0xffffu));
  const uint32_t k16 = best >> 16;
  best = (k16 << 16) | (k16 < 0x8000u ? 0xffffu : 0u);
  if (nan_any != 0u) {
    __syncthreads();
    best = 0u;
    bidx = 0x7fffffff;
    for (int v = static_cast<int>(threadIdx.x); v < nvec; v += kThreads) {
      vec_t xv;
      xv.load(p.logits, v);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float f = cast<fp32_t>(xv[j]);
        const uint32_t key = f == f ? mkd_xchg::order_key(f) : p.nan_key;
        const int32_t idx = v * 8 + j;
        if (key > best || (key == best && idx < bidx)) {
          best = key;
          bidx = idx;
        }
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const uint32_t ov = __shfl_xor_sync(0xffffffffu, best, off);
      const int32_t oi = __shfl_xor_sync(0xffffffffu, bidx, off);
      if (ov > best || (ov == best && oi < bidx)) {
        best = ov;
        bidx = oi;
      }
    }
    if (lane == 0) {
      s_v[warp] = best;
      s_i[warp] = bidx;
    }
    __syncthreads();
    best = s_v[0];
    bidx = s_i[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) {
      if (s_v[w] > best || (s_v[w] == best && s_i[w] < bidx)) {
        best = s_v[w];
        bidx = s_i[w];
      }
    }
  }
  const uint32_t tag = s_tag;
  const int slot = static_cast<int>(tag % mkd_xchg::kSlots);
  const uint64_t t_scan = mkd_xchg::now_ns();
  const uint32_t hot_pos = static_cast<uint32_t>(p.rank * p.vocab + bidx);
  const int64_t brow = bidx;

  // Send this CTA's chunk of the packet to every rank (self included).
  uint64_t* dst = p.mc + ((static_cast<int64_t>(slot) * kWorld + p.rank) * kChunks + c) * kRec;
  if (threadIdx.x == 0) {
    mkd_xchg::mc_store(dst, best, tag);
    mkd_xchg::mc_store(dst + 1, hot_pos, tag);
  }
  const bool payload = p.token_only == 0;
  if (payload) {
    if constexpr (kMemo) {
      const float* row = static_cast<const float*>(p.cand) + brow * kH + c * kCols;
      for (int w = static_cast<int>(threadIdx.x); w < kCandWords; w += kThreads) {
        mkd_xchg::mc_store(dst + 2 + w, __float_as_uint(row[w]), tag);
      }
    } else {
      const uint32_t* row =
          reinterpret_cast<const uint32_t*>(static_cast<const bf16_t*>(p.cand) + brow * kH + c * kCols);
      for (int w = static_cast<int>(threadIdx.x); w < kCandWords; w += kThreads) {
        mkd_xchg::mc_store(dst + 2 + w, mkd_xchg::clear_negative_zero(row[w]), tag);
      }
    }
    if (p.rank >= kEmbed) {
      const float* part = p.own_partial + c * kCols;
      for (int w = static_cast<int>(threadIdx.x); w < kPartWords; w += kThreads) {
        if (p.partial_flag != nullptr) {
          // The partial comes from a side-stream kernel that is not a stream predecessor.
          uint32_t f, n = 0;
          do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(p.partial_flag) : "memory");
            if (++n > mkd_xchg::kSpinLimit) __trap();
          } while (f != tag);
        }
        mkd_xchg::mc_store(dst + 2 + kCandWords + w, __float_as_uint(part[w]), tag);
      }
    }
  }

  const uint64_t t_sent = mkd_xchg::now_ns();
  // Receive in one batch: every rank's key and candidate chunk, and the hidden ranks'
  // partial chunks; the winner's candidate is then picked from shared memory.
  auto rec = [&](int s) { return p.uc + ((static_cast<int64_t>(slot) * kWorld + s) * kChunks + c) * kRec; };
  constexpr int kHead = 2 + kCandWords;
  const int head = payload ? kHead : 2;
  const int heads = kWorld * head;
  const int total = payload ? heads + (kWorld - kEmbed) * kPartWords : heads;
  constexpr int kRecvUnroll = 4;
  for (int w0 = static_cast<int>(threadIdx.x); w0 < total; w0 += kThreads * kRecvUnroll) {
    const uint64_t* addr[kRecvUnroll];
    uint64_t v[kRecvUnroll];
#pragma unroll
    for (int u = 0; u < kRecvUnroll; ++u) {
      const int i = w0 + u * kThreads;
      addr[u] = nullptr;
      if (i < total) {
        if (i < heads) {
          addr[u] = rec(i / head) + i % head;
        } else {
          const int j = i - heads;
          addr[u] = rec(kEmbed + j / kPartWords) + kHead + j % kPartWords;
        }
        asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v[u]) : "l"(addr[u]) : "memory");
      }
    }
    // Re-poll every word still missing in one batch, so the last arrivals cost one round trip.
    uint32_t want = 0, got = 0;
#pragma unroll
    for (int u = 0; u < kRecvUnroll; ++u) {
      if (w0 + u * kThreads < total) {
        want |= 1u << u;
        if (static_cast<uint32_t>(v[u] >> 32) == tag) got |= 1u << u;
      }
    }
    for (uint32_t n = 0; got != want;) {
      if (++n > mkd_xchg::kSpinLimit) __trap();
#pragma unroll
      for (int u = 0; u < kRecvUnroll; ++u) {
        if ((want & ~got) >> u & 1u)
          asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v[u]) : "l"(addr[u]) : "memory");
      }
#pragma unroll
      for (int u = 0; u < kRecvUnroll; ++u) {
        if (((want & ~got) >> u & 1u) && static_cast<uint32_t>(v[u] >> 32) == tag) got |= 1u << u;
      }
    }
#pragma unroll
    for (int u = 0; u < kRecvUnroll; ++u) {
      const int i = w0 + u * kThreads;
      if (i < total) {
        const uint32_t x = static_cast<uint32_t>(v[u]);
        if (i < heads) {
          const int s = i / head, w = i % head;
          if (w == 0) {
            s_key[s] = x;
          } else if (w == 1) {
            s_gid[s] = static_cast<int64_t>(x);
          } else {
            s_cand[s][w - 2] = x;
          }
        } else {
          const int j = i - heads;
          s_part[j / kPartWords][j % kPartWords] = __uint_as_float(x);
        }
      }
    }
  }
  __syncthreads();
  const uint64_t t_recv = mkd_xchg::now_ns();
  if (threadIdx.x == 0) {
    int win = 0;
    uint32_t bv = s_key[0];
    int64_t bi = s_gid[0];
#pragma unroll
    for (int s = 1; s < kWorld; ++s) {
      if (mkd_xchg::better(s_key[s], s_gid[s], bv, bi)) {
        bv = s_key[s];
        bi = s_gid[s];
        win = s;
      }
    }
    s_win = win;
    if (c == 0) {
      const int64_t gid = p.hot_ids[bi];
      p.topk_index[0] = gid;
      p.topk_p[0] = 1.0f;
      if (p.draft_tokens != nullptr) p.draft_tokens[p.draft_column] = gid;
      if (p.positions != nullptr) p.positions[0] = p.positions[0] + 1;
      if (p.mrope != nullptr) {
#pragma unroll
        for (int d = 0; d < 3; ++d)
          p.mrope[d * p.mrope_stride] += 1;
      }
    }
  }
  __syncthreads();
  bool fold = false;
  if constexpr (kMemo) fold = p.fc_out != nullptr;
  if (payload && fold) {
    // The fc output (mkd_fc_sum, memo path): memo row, then the hidden partials in rank order.
    for (int w = static_cast<int>(threadIdx.x); w < kCandWords; w += kThreads) {
      float acc = __uint_as_float(s_cand[s_win][w]);
#pragma unroll
      for (int h = 0; h < kWorld - kEmbed; ++h)
        acc += s_part[h][w];
      p.fc_out[c * kCols + w] = cast<bf16_t>(acc);
    }
  } else if (payload) {
    for (int w = static_cast<int>(threadIdx.x); w < kCandWords; w += kThreads) {
      static_cast<uint32_t*>(p.out_cand)[(kMemo ? c * kCols : c * kCols / 2) + w] = s_cand[s_win][w];
    }
    for (int j = static_cast<int>(threadIdx.x); j < (kWorld - kEmbed) * kPartWords; j += kThreads) {
      p.out_partials[static_cast<int64_t>(j / kPartWords) * kH + c * kCols + j % kPartWords] =
          s_part[j / kPartWords][j % kPartWords];
    }
  }

  __syncthreads();
  if (threadIdx.x == 0 && p.trace != nullptr) {
    uint64_t* t = p.trace + (static_cast<int64_t>(tag % static_cast<uint32_t>(p.trace_slots)) * kChunks + c) * 9;
    t[0] = t_start;
    t[1] = t_wait;
    t[2] = t_ld;
    t[3] = t_cmp;
    t[4] = t_warp;
    t[5] = t_scan;
    t[6] = t_sent;
    t[7] = t_recv;
    t[8] = mkd_xchg::now_ns();
  }
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(p.done, 1u) == kChunks - 1) {
      *p.done = 0u;
      *p.epoch = tag + 1;
    }
  }
}

/** \brief Zero a range of 64-bit words (the exchange buffer before first use). */
__global__ void mkd_xchg_zero_kernel(uint64_t* p, int64_t n) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    p[i] = 0ull;
  }
}

template <int kH, int kWorld, bool kMemo, int kChunks, bool kUsePDL>
struct MkdXchg {
  static constexpr int kThreads = 1024;
  static constexpr int kCols = kH / kChunks;
  static constexpr int kRec = 2 + (kMemo ? kCols : kCols / 2) + kCols;
  static constexpr int64_t kWords = static_cast<int64_t>(mkd_xchg::kSlots) * kWorld * kChunks * kRec;

  static void zero(int64_t uc_ptr) {
    auto* p = reinterpret_cast<uint64_t*>(uc_ptr);
    mkd_xchg_zero_kernel<<<64, 256>>>(p, kWords);
    host::RuntimeDeviceCheck(cudaGetLastError());
  }

  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView cand,
      const tvm::ffi::TensorView hot_ids,
      const tvm::ffi::TensorView own_partial,
      const tvm::ffi::TensorView epoch_done,
      const tvm::ffi::TensorView out_cand,
      const tvm::ffi::TensorView out_partials,
      const tvm::ffi::TensorView fc_out,
      const tvm::ffi::TensorView topk_index,
      const tvm::ffi::TensorView topk_p,
      const tvm::ffi::TensorView draft_tokens,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView mrope_positions,
      int64_t mc_ptr,
      int64_t uc_ptr,
      int64_t rank,
      int64_t draft_column,
      int64_t nan_key,
      int64_t token_only,
      int64_t partial_flag_ptr,
      int64_t scan_part_ptr,
      int64_t head_part_ptr,
      int64_t head_parts,
      int64_t trace_ptr,
      int64_t trace_slots,
      int64_t pf_ptr,
      int64_t pf_bytes) {
    using namespace host;
    constexpr int kW = 2 * kH / kWorld;
    constexpr int kEmbed = kH / kW;
    auto V = SymbolicSize{"vocab_local"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, V}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(logits);
    if constexpr (kMemo) {
      TensorMatcher({V, kH}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(cand);
      TensorMatcher({1, kH}).with_dtype<fp32_t>().with_device(device).verify(out_cand);
    } else {
      TensorMatcher({V, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(cand);
      TensorMatcher({1, kH}).with_dtype<bf16_t>().with_device(device).verify(out_cand);
    }
    TensorMatcher({kWorld * V.unwrap()}).with_dtype<int64_t>().with_device(device).verify(hot_ids);
    TensorMatcher({1, kH}).with_dtype<fp32_t>().with_device(device).verify(own_partial);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(epoch_done);
    TensorMatcher({kWorld - kEmbed, kH}).with_dtype<fp32_t>().with_device(device).verify(out_partials);
    bf16_t* fc = nullptr;
    if (fc_out.numel() > 0) {
      CHECK_HOST(kMemo) << "fc_out needs memo candidates";
      TensorMatcher({1, kH}).with_dtype<bf16_t>().with_device(device).verify(fc_out);
      fc = static_cast<bf16_t*>(fc_out.data_ptr());
    }
    TensorMatcher({1, 1}).with_dtype<int64_t>().with_device(device).verify(topk_index);
    TensorMatcher({1, 1}).with_dtype<fp32_t>().with_device(device).verify(topk_p);
    CHECK_HOST(V.unwrap() % 8 == 0 && V.unwrap() <= 65536) << "local vocab must be a multiple of 8 and <= 65536";
    CHECK_HOST(head_parts <= kThreads) << "at most one head partial per exchange thread";
    CHECK_HOST(trace_slots >= 1) << "trace_slots must be >= 1";
    int64_t* tokens = nullptr;
    if (draft_tokens.numel() > 0) {
      auto S = SymbolicSize{"num_steps"};
      TensorMatcher({1, S}).with_dtype<int64_t>().with_device(device).verify(draft_tokens);
      CHECK_HOST(draft_column >= 0 && draft_column < S.unwrap()) << "draft column out of range";
      tokens = static_cast<int64_t*>(draft_tokens.data_ptr());
    }
    int64_t* pos = nullptr;
    if (positions.numel() > 0) {
      TensorMatcher({1}).with_dtype<int64_t>().with_device(device).verify(positions);
      pos = static_cast<int64_t*>(positions.data_ptr());
    }
    int64_t* mrope = nullptr;
    int32_t mrope_stride = 0;
    if (mrope_positions.numel() > 0) {
      auto R = SymbolicSize{"mrope_row_stride"};
      TensorMatcher({3, 1}).with_strides({R, 1}).with_dtype<int64_t>().with_device(device).verify(mrope_positions);
      mrope = static_cast<int64_t*>(mrope_positions.data_ptr());
      mrope_stride = static_cast<int32_t>(R.unwrap());
    }
    auto* ed = static_cast<uint32_t*>(epoch_done.data_ptr());
    const auto params = MkdXchgParams{
        .logits = static_cast<const bf16_t*>(logits.data_ptr()),
        .cand = cand.data_ptr(),
        .hot_ids = static_cast<const int64_t*>(hot_ids.data_ptr()),
        .own_partial = static_cast<const float*>(own_partial.data_ptr()),
        .partial_flag = reinterpret_cast<const uint32_t*>(partial_flag_ptr),
        .scan_part = reinterpret_cast<uint64_t*>(scan_part_ptr),
        .head_part = reinterpret_cast<const uint32_t*>(head_part_ptr),
        .head_parts = static_cast<int32_t>(head_parts),
        .mc = reinterpret_cast<uint64_t*>(mc_ptr),
        .uc = reinterpret_cast<const uint64_t*>(uc_ptr),
        .epoch = ed,
        .done = ed + 1,
        .out_cand = out_cand.data_ptr(),
        .out_partials = static_cast<float*>(out_partials.data_ptr()),
        .fc_out = fc,
        .token_only = static_cast<int32_t>(token_only != 0),
        .topk_index = static_cast<int64_t*>(topk_index.data_ptr()),
        .topk_p = static_cast<float*>(topk_p.data_ptr()),
        .draft_tokens = tokens,
        .positions = pos,
        .mrope = mrope,
        .vocab = static_cast<int32_t>(V.unwrap()),
        .nan_key = static_cast<uint32_t>(nan_key),
        .rank = static_cast<int32_t>(rank),
        .draft_column = static_cast<int32_t>(draft_column),
        .mrope_stride = mrope_stride,
        .trace = reinterpret_cast<uint64_t*>(trace_ptr),
        .trace_slots = static_cast<int32_t>(trace_slots),
        .pf = reinterpret_cast<const char*>(pf_ptr),
        .pf_bytes = pf_bytes,
    };
    LaunchKernel(kChunks, kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(mkd_xchg_kernel<kH, kWorld, kMemo, kChunks, kThreads, kUsePDL>, params);
  }
};

}  // namespace sglang
