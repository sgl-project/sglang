// SPDX-License-Identifier: Apache-2.0
// Int8 DSpark markov walk for one request (bs = 1): the whole int8 W2 stays on chip (SMEM + registers of every SM) and
// the K draft steps run in one persistent cooperative launch.
//
//   prev = anchor;  for k < K:  logit_k = bf16(base_k + W2 W1[prev]);
//                               tok_k = argmax(logit_k)  or  argmax(logit_k / T + Gumbel_k);  prev = tok_k
//
// Numerics.  W2 is int8 with one fp32 scale per row; u = W1[prev] is two int8 planes, u ~= s_hi q_hi + s_lo q_lo, so
//   W2[r] . u = s_row (s_hi <q_w, q_hi> + s_lo <q_w, q_lo>)
// with both inner products exact in int32 (mma.sync m16n8k32 s8.s8.s32).  The B fragment carries q_hi in the even and
// q_lo in the odd columns, so every lane holds both dots of its rows g and g + 8 of every tile.  The logits base + bias
// are rounded to bf16 once (RNE); argmax and Gumbel-max read exactly those values, and corrected receives exactly those
// bits (the verifier rebuilds q = softmax(corrected / T) from them).
//
// Layout.  One CTA (256 threads) per SM, grid = rows_pad / 1152.  CTA c owns 72 consecutive 16-row tiles: the first
// kTilesSmem = 24 in SMEM, then NG = 6 per warp in registers.  frag[tile][chunk][lane] is the lane's 16-B m16n8k32 A
// fragment of k32-chunk `chunk` (host repack: markov_walk.py _build_frag).  W1q rows (528 B): q_hi[256] |
// q_lo[256] | s_hi f32 | s_lo f32 | pad.
// SMEM: u row (544 B) | base double buffer (2 x 1152 bf16) | Gumbel double buffer (2 x 1152 f32) | bias (1152 f32) |
//       row scales (72 x 16 f32) | 24 W2 tiles (4 KiB each).
// Per step k:
//   1. (132 threads) gather u = W1q[prev] into SMEM; cp.async of base[k + 1] (independent of the walk)
//   2. (all warps) the 9 tiles' mma chains side by side, then each lane's rows' fp32 bias into SMEM
//   3. (144 threads, 8 adjacent rows each) logits, bf16 rounding, corrected (sampling rounds), key, block argmax
//   4. (thread 224) the cross-CTA exchange of the common header on step k's {key, count} pair, after prefetching
//      W1q[local winner] into L2; meanwhile warps 0..6 compute step k + 1's Gumbel noise.
// State: int64 [kStateU64] = round | pad | two sets of 16 {key, count} pairs.  Round r uses set r & 1; CTA 0 clears
// the other set at the start and stores round + 1 after the last step (every CTA has polled every final count by then,
// and the next launch starts after this one ends).  Nothing is reset between launches.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "dspark_markov_walk_common.cuh"
#include <cstdint>

namespace sglang::dspark_markov_walk::i8 {

constexpr int kUSmemBytes = 544;  // SMEM slot of the gathered W1q row (kURowBytes rounded up)
// W2 slice per CTA: TS SMEM tiles + NG register tiles per warp, 72 16-row tiles = 1152 rows
constexpr int NG = 6, TS = 3;
constexpr int kTilesSmem = TS * kWarps, kTilesPerCta = kTilesSmem + NG * kWarps, kRowsCta = kTilesPerCta * 16;
constexpr int kStateU64 = 2 + 4 * kMaxSteps;  // round | pad | two sets of 16-B {key, count} pairs per step

// Owner lanes hold rows l0 + g and + 8 of a tile: c0/c2 = <q_w, q_hi>, c1/c3 = <q_w, q_lo> (B columns 0, 1) -> fp32
// bias into bias_s (local row index)
SGL_DEVICE void
store_bias(const int (&c)[4], float sr_lo, float sr_hi, float u_hi, float u_lo, float* bias_s, int l0, int g) {
  bias_s[l0 + g] = sr_lo * (u_hi * static_cast<float>(c[0]) + u_lo * static_cast<float>(c[1]));
  bias_s[l0 + g + 8] = sr_hi * (u_hi * static_cast<float>(c[2]) + u_lo * static_cast<float>(c[3]));
}

// Gumbel noise of one step for this CTA's rows into dst (local row index); one Philox call -> 4 rows, counter = (global
// row / 4, step, round).  Threads [t, t + nt) share the work.
SGL_DEVICE void
fill_gumbel_v(float* dst, int row_base, int rows_cta, unsigned step, u64 round, uint2 seed, int t, int nt) {
  for (int q = t; q < rows_cta / 4; q += nt) {
    const int r = row_base + q * 4;
    const uint4 x = philox(
        make_uint4(
            static_cast<unsigned>(r >> 2), step, static_cast<unsigned>(round), static_cast<unsigned>(round >> 32)),
        seed);
    const float4 g = make_float4(gumbel(x.x), gumbel(x.y), gumbel(x.z), gumbel(x.w));
    reinterpret_cast<float4*>(dst)[q] = g;
  }
}

__global__ void __launch_bounds__(kThreads, 1) markov_walk_i8_kernel(
    const uint4* __restrict__ frag,
    const float* __restrict__ row_scale,
    const uint32_t* __restrict__ w1q,
    const __nv_bfloat16* __restrict__ base,
    const int64_t* __restrict__ anchor,
    int64_t* __restrict__ tokens,
    __nv_bfloat16* __restrict__ corrected,
    u64* __restrict__ state,
    const float* __restrict__ temps,
    int num_steps,
    int ld,
    int valid_rows,
    u64 seed) {
  constexpr int tiles_smem = kTilesSmem;
  constexpr int tiles_per_cta = kTilesPerCta;
  constexpr int rows_cta = kRowsCta;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  uint32_t* u_s = reinterpret_cast<uint32_t*>(smem_raw);  // kURowBytes
  unsigned char* p = smem_raw + kUSmemBytes;
  __nv_bfloat16* base_s = reinterpret_cast<__nv_bfloat16*>(p);
  p += 4 * rows_cta;  // 2 x bf16
  float* gum_s = reinterpret_cast<float*>(p);
  p += 8 * rows_cta;  // 2 x f32
  float* bias_s = reinterpret_cast<float*>(p);
  p += 4 * rows_cta;  // f32
  float* scale_s = reinterpret_cast<float*>(p);
  p += 64 * tiles_per_cta;                   // all tiles
  uint4* w_s = reinterpret_cast<uint4*>(p);  // 4 KiB/tile
  __shared__ u64 warp_best[kWarps];
  __shared__ unsigned tok_s;
  __shared__ __align__(8) u64 mbar_load;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile0 = blockIdx.x * tiles_per_cta;
  const int row_base = tile0 * 16;
  const uint4* g_frag = frag + static_cast<size_t>(tile0) * kTileVecs;
  // the round's temperature, from device memory (staged by the caller before the launch / graph replay); first
  // load of the kernel, so its latency hides under the W2 load below
  const float temperature = temps[0];

  const u64 round = ld_relaxed(state);
  const int set = static_cast<int>(round & 1);
  // {key, arrivals} pairs, 16-B aligned (state + 2), one pair per step; set r&1, CTA 0 clears the other set
  u64* pairs = state + 2 + set * 2 * kMaxSteps;
  if (blockIdx.x == 0 && threadIdx.x < 2 * kMaxSteps)
    st_relaxed(state + 2 + (set ^ 1) * 2 * kMaxSteps + threadIdx.x, 0);
  const float inv_t = inv_temperature(temperature);
  const bool sampling = inv_t > 0.f;
  // corrected logits only feed the verifier's q = softmax(corrected / T): sampling rounds only
  const bool write_corr = sampling && corrected != nullptr;
  const uint2 seed2 = make_uint2(static_cast<unsigned>(seed), static_cast<unsigned>(seed >> 32));

  // base rows have the caller's stride ld; the last CTA's padding rows (>= valid_rows, whole 8-row vectors) load the
  // last valid vector instead -- never out of bounds -- and the epilogue skips them (r < valid_rows)
  const int vec_last = valid_rows - 8;
  auto prefetch_base = [&](int k) {
    const __nv_bfloat16* src = base + static_cast<size_t>(k) * ld;
    __nv_bfloat16* dst = base_s + (k & 1) * rows_cta;
    for (int i = threadIdx.x; i < rows_cta / 8; i += kThreads)
      cp_async16(dst + i * 8, src + min(row_base + i * 8, vec_last));
    cp_async_commit();
  };

  // ---- once per round: base[0], W2 slice (+ row scales) on-chip, gumbel[0]
  prefetch_base(0);  // base logits: normal L2 priority (just written, and steps 1.. read the other rows)
  const u64 pol = evict_first_policy();
  const unsigned a_mbar = smem_u32(&mbar_load);
  if (threadIdx.x == 0) {
    mbar_init(a_mbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    constexpr unsigned kW2Bytes = tiles_smem * kTileVecs * 16, kScaleBytes = tiles_per_cta * 64, kChunk = 16384;
    mbar_expect_tx(a_mbar, kW2Bytes + kScaleBytes);
    for (unsigned off = 0; off < kW2Bytes; off += kChunk)
      bulk_g2s_hint(
          smem_u32(reinterpret_cast<unsigned char*>(w_s) + off),
          reinterpret_cast<const unsigned char*>(g_frag) + off,
          off + kChunk <= kW2Bytes ? kChunk : kW2Bytes - off,
          a_mbar,
          pol);
    bulk_g2s_hint(smem_u32(scale_s), row_scale + row_base, kScaleBytes, a_mbar, pol);
  }
  if (threadIdx.x == 0) tok_s = clamp_row(anchor[0], valid_rows);
  uint4 a_reg[NG][8];
#pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    const int t = tiles_smem + warp * NG + gi;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      a_reg[gi][j] = ldcg_hint(g_frag + (t * 8 + j) * 32 + lane, pol);
    }
  }
  if (sampling) fill_gumbel_v(gum_s, row_base, rows_cta, 0, round, seed2, threadIdx.x, kThreads);
  cp_async_wait<0>();
  // only thread 0 (which initialized it) waits on the load barrier: another thread could reach a try_wait
  // before the init (nothing orders them); the barrier below then publishes the landed tiles to everyone
  if (threadIdx.x == 0) mbar_wait_cta(a_mbar, 0);
  __syncthreads();

  constexpr int kXchgThread = (kWarps - 1) * 32;
  static_assert(rows_cta / 8 <= kXchgThread, "exchange thread must not own epilogue rows");
  // GEMV over this warp's NG + TS tiles: acc[i] <- int32 dots of tile i's 16 rows with the B columns built from ub (a
  // W1q row's q_hi | q_lo words).  All NG + TS accumulate chains advance together, one k32-chunk at a time; chunk j's
  // SMEM fragments are loaded at the top of iteration j and the register-tile mma run while they are in flight.
  const int g = lane >> 2, tig = lane & 3;
  auto gemv = [&](int (&acc)[NG + TS][4], const uint32_t* ub) {
#pragma unroll
    for (int i = 0; i < NG + TS; ++i)
      acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint4 as[TS];
#pragma unroll
      for (int ts = 0; ts < TS; ++ts)
        as[ts] = w_s[((warp + kWarps * ts) * 8 + j) * 32 + lane];
      // B column n <- q_hi (n even) / q_lo (n odd): every lane (any tig) ends up holding the (hi, lo) dots of its rows
      // g and g + 8 of every tile
      const uint32_t b0 = ub[(g & 1) * 64 + j * 8 + tig];
      const uint32_t b1 = ub[(g & 1) * 64 + j * 8 + 4 + tig];
#pragma unroll
      for (int gi = 0; gi < NG; ++gi)
        mma_s8(acc[gi], a_reg[gi][j], b0, b1);
#pragma unroll
      for (int ts = 0; ts < TS; ++ts)
        mma_s8(acc[NG + ts], as[ts], b0, b1);
    }
  };

  for (int k = 0; k < num_steps; ++k) {
    const unsigned prev = tok_s;
    // %tid.x re-read after the GEMV for the warp-best store: keeps `lane` out of the step's live registers (the step
    // loop uses 253 of 255; this choice is what keeps it free of spills)
    unsigned tid_p = 0;
    if (threadIdx.x < kURowWords) u_s[threadIdx.x] = w1q[static_cast<size_t>(prev) * kURowWords + threadIdx.x];
    if (k + 1 < num_steps) {
      prefetch_base(k + 1);
      cp_async_wait<1>();  // step k's slice (committed a step ago) has landed
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    {
      int acc[NG + TS][4];
      gemv(acc, u_s);
      asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid_p));
      // bias, split over the 4 tig lanes (every lane holds the dots): lane tig converts tiles tig, tig + 4, tig + 8
      {
        constexpr int NT = NG + TS;
        const float u_hi = __uint_as_float(u_s[128]), u_lo = __uint_as_float(u_s[129]);
#pragma unroll
        for (int s4 = 0; s4 < (NT + 3) / 4; ++s4) {
          const int i = 4 * s4 + tig;
          int c[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int v = acc[4 * s4][q];
            if (4 * s4 + 1 < NT) v = tig == 1 ? acc[(4 * s4 + 1) % NT][q] : v;
            if (4 * s4 + 2 < NT) v = tig == 2 ? acc[(4 * s4 + 2) % NT][q] : v;
            if (4 * s4 + 3 < NT) v = tig == 3 ? acc[(4 * s4 + 3) % NT][q] : v;
            c[q] = v;
          }
          if (i < NT) {
            const int t = i < NG ? tiles_smem + warp * NG + i : warp + kWarps * (i - NG);
            store_bias(c, scale_s[t * 16 + g], scale_s[t * 16 + g + 8], u_hi, u_lo, bias_s, t * 16, g);
          }
        }
      }
      __syncthreads();
    }
    const float* bsrc = bias_s;

    // epilogue over all rows of the CTA, 8 adjacent rows per thread (one 16 B base load, 16 B corrected
    // store); rows_cta / 8 work items <= kThreads
    u64 best = 0;
    if (threadIdx.x < rows_cta / 8) {
      const int l = threadIdx.x * 8;
      const int r = row_base + l;
      const uint4 bb = *reinterpret_cast<const uint4*>(base_s + (k & 1) * rows_cta + l);
      const float4 b0 = *reinterpret_cast<const float4*>(bsrc + l);
      const float4 b1 = *reinterpret_cast<const float4*>(bsrc + l + 4);
      const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&bb);
      float lg[8];
      {
        const float2 x0 = __bfloat1622float2(bp[0]), x1 = __bfloat1622float2(bp[1]);
        const float2 x2 = __bfloat1622float2(bp[2]), x3 = __bfloat1622float2(bp[3]);
        lg[0] = x0.x + b0.x;
        lg[1] = x0.y + b0.y;
        lg[2] = x1.x + b0.z;
        lg[3] = x1.y + b0.w;
        lg[4] = x2.x + b1.x;
        lg[5] = x2.y + b1.y;
        lg[6] = x3.x + b1.z;
        lg[7] = x3.y + b1.w;
      }
      // bf16-consistent logits: round once (RNE, 4 F2FP), then argmax / sample the ROUNDED values -- exactly the
      // bits stored to corrected, from which the verifier builds q. The unpack is plain bit ops on the packed
      // word (low half = lg[2i] -> << 16, high half = lg[2i+1] -> & 0xffff0000: 8 ops; __bfloat1622float2 on
      // the bf16x2 struct compiled to 16 PRMT/IMAD).
      uint32_t ow[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162 h = __floats2bfloat162_rn(lg[2 * i], lg[2 * i + 1]);
        ow[i] = *reinterpret_cast<const uint32_t*>(&h);
        lg[2 * i] = __uint_as_float(ow[i] << 16);
        lg[2 * i + 1] = __uint_as_float(ow[i] & 0xffff0000u);
      }
      if (sampling) {
        const float4 g0 = *reinterpret_cast<const float4*>(gum_s + (k & 1) * rows_cta + l);
        const float4 g1 = *reinterpret_cast<const float4*>(gum_s + (k & 1) * rows_cta + l + 4);
        lg[0] = lg[0] * inv_t + g0.x;
        lg[1] = lg[1] * inv_t + g0.y;
        lg[2] = lg[2] * inv_t + g0.z;
        lg[3] = lg[3] * inv_t + g0.w;
        lg[4] = lg[4] * inv_t + g1.x;
        lg[5] = lg[5] * inv_t + g1.y;
        lg[6] = lg[6] * inv_t + g1.z;
        lg[7] = lg[7] * inv_t + g1.w;
      }
      // argmax of the 8 rows as a float max tree + the first row holding it, one key per thread
      const float m =
          fmaxf(fmaxf(fmaxf(lg[0], lg[1]), fmaxf(lg[2], lg[3])), fmaxf(fmaxf(lg[4], lg[5]), fmaxf(lg[6], lg[7])));
      int idx = 7;
#pragma unroll
      for (int i = 6; i >= 0; --i)
        idx = lg[i] == m ? i : idx;  // ties -> smallest row, as torch.argmax
      best = r < valid_rows ? pack_key(m, r + idx) : 0ull;
      // corrected after the key: the argmax heads for the warp max while the store's address is formed
      if (write_corr) {  // block-uniform: sampling rounds with a corrected buffer
        __nv_bfloat16* dst = corrected + static_cast<size_t>(k) * ld + r;
        if (r < valid_rows) *reinterpret_cast<uint4*>(dst) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
      }
    }
    // warp argmax of 64-bit keys with two 32-bit redux.sync: max of the high words, then max of the low
    // words among the lanes that hold it
    {
      const unsigned hi_w = static_cast<unsigned>(best >> 32);
      const unsigned m = __reduce_max_sync(0xffffffffu, hi_w);
      const unsigned lo_w = hi_w == m ? static_cast<unsigned>(best) : 0u;
      const unsigned lo = __reduce_max_sync(0xffffffffu, lo_w);
      best = (static_cast<u64>(m) << 32) | lo;
    }
    if ((tid_p & 31) == 0) warp_best[warp] = best;
    __syncthreads();

    // The exchange runs on thread kXchgThread: it wrote no corrected logits in the epilogue (threads
    // < rows_cta/8 did), so its release does not wait for those global stores to land.
    const u64 n = gridDim.x;
    u64 cta_best = 0;
    if (threadIdx.x == kXchgThread) {
      cta_best = warp_best[0];
#pragma unroll
      for (int w = 1; w < kWarps; ++w)
        cta_best = max(cta_best, warp_best[w]);
      // W1q[local winner] -> L2 now, so the global winner's row is an L2 hit for everyone next step
      const char* row = reinterpret_cast<const char*>(w1q + static_cast<size_t>(key_row(cta_best)) * kURowWords);
#pragma unroll
      for (int line = 0; line < 5; ++line)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(row + line * 128));
      atomicMax(pairs + 2 * k, cta_best);
      red_add_release(pairs + 2 * k + 1, 1);
    }
    if (threadIdx.x == kXchgThread) {
      ulonglong2 v;
      do {
        v = ld_relaxed_v2(pairs + 2 * k);
      } while (v.y < n);
      const unsigned tok = key_row(v.x);
      tok_s = tok;
      if (blockIdx.x == 0) tokens[k] = tok;
    } else if (sampling && warp < kWarps - 1 && k + 1 < num_steps) {
      // off the critical path: next step's noise while the exchange thread waits for the other CTAs
      fill_gumbel_v(
          gum_s + ((k + 1) & 1) * rows_cta,
          row_base,
          rows_cta,
          static_cast<unsigned>(k + 1),
          round,
          seed2,
          threadIdx.x,
          kThreads - 32);
    }
    __syncthreads();
  }
  // every CTA has seen every final count (no CTA still reads this round's state): CTA 0 closes the round
  if (blockIdx.x == 0 && threadIdx.x == 0) st_relaxed(state, round + 1);
}

/**
 * \brief One round of the bs = 1 markov walk: num_steps draft tokens in one cooperative launch.
 *
 * \param frag       int8 [rows_pad * 256], W2 in m16n8k32 A-fragment order (tiles of 16 rows; CTA c owns tiles
 *                   [72 c, 72 c + 72): the first 24 in SMEM, then 6 per warp in registers).
 * \param row_scale  fp32 [rows_pad], W2 row scales; rows_pad = grid x 1152 sets the grid.
 * \param w1q        uint8 [V, 528], W1 rows: q_hi | q_lo | s_hi f32 | s_lo f32 | pad.
 * \param base       bf16 [1, kb, ld], kb >= num_steps, ld >= valid_rows, ld % 8 == 0 (the lm_head's [1, K, V]).
 * \param anchor     int64 [1], clamped into [0, valid_rows).
 * \param tokens     int64 [num_steps], tokens[k] = step k.
 * \param corrected  None, or bf16 shaped like base: the bf16 logits the walk used, written ONLY in sampling rounds,
 *                   rows k < num_steps, vocabulary rows < valid_rows -- zero-initialise it once.
 * \param state      int64 [kStateU64], 16-B aligned, zeroed once and kept across calls (round counter + exchange).
 * \param temps      fp32 [1]: <= 0 or NaN: greedy; T > 0 is clamped to [1e-5, 1e4]; read in-kernel (one graph for
 *                   every temperature).
 * \param valid_rows The vocabulary size: % 8 == 0, <= min(ld, rows_pad, V).
 * \param seed       Philox key; the round counter in state advances every launch.
 */
inline void walk(
    tvm::ffi::TensorView frag,
    tvm::ffi::TensorView row_scale,
    tvm::ffi::TensorView w1q,
    tvm::ffi::TensorView base,
    tvm::ffi::TensorView anchor,
    tvm::ffi::TensorView tokens,
    tvm::ffi::Optional<tvm::ffi::TensorView> corrected,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView temps,
    int64_t num_steps,
    int64_t valid_rows,
    int64_t seed) {
  using namespace host;
  auto rows_pad = SymbolicSize{"rows_pad"};
  auto w1_rows = SymbolicSize{"w1_rows"};
  auto kb = SymbolicSize{"base_steps"};
  auto ld = SymbolicSize{"base_row_stride"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  CHECK_HOST(num_steps >= 1 && num_steps <= kMaxSteps) << "num_steps must be in [1, " << kMaxSteps << "]";
  TensorMatcher({rows_pad}).with_dtype<float>().with_device(device).ensure_alignment(16).verify(row_scale);
  CHECK_HOST(rows_pad.unwrap() >= kRowsCta && rows_pad.unwrap() % kRowsCta == 0)
      << "rows_pad (row_scale.numel()) " << rows_pad.unwrap() << " not a multiple of the CTA rows " << kRowsCta;
  TensorMatcher({rows_pad.unwrap() * 256}).with_dtype<int8_t>().with_device(device).ensure_alignment(16).verify(frag);
  TensorMatcher({w1_rows, kURowBytes}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w1q);
  TensorMatcher({1, kb, ld}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(base);
  if (corrected.has_value()) {
    TensorMatcher({1, kb, ld}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(corrected.value());
  }
  TensorMatcher({1}).with_dtype<int64_t>().with_device(device).ensure_alignment(8).verify(anchor);
  TensorMatcher({1}).with_dtype<float>().with_device(device).ensure_alignment(4).verify(temps);
  TensorMatcher({num_steps}).with_dtype<int64_t>().with_device(device).ensure_alignment(8).verify(tokens);
  TensorMatcher({kStateU64}).with_dtype<int64_t>().with_device(device).ensure_alignment(16).verify(state);
  const int64_t ld_v = ld.unwrap();
  CHECK_HOST(kb.unwrap() >= num_steps) << "base has " << kb.unwrap() << " steps < num_steps " << num_steps;
  CHECK_HOST(
      ld_v % 8 == 0 && ld_v < (int64_t{1} << 31) && valid_rows % 8 == 0 && valid_rows >= 8 && valid_rows <= ld_v &&
      valid_rows <= rows_pad.unwrap() && valid_rows <= w1_rows.unwrap())
      << "need ld % 8 == 0, valid_rows % 8 == 0, 8 <= valid_rows <= min(ld, rows_pad, W1 rows); got ld " << ld_v
      << ", valid_rows " << valid_rows << ", rows_pad " << rows_pad.unwrap() << ", W1 rows " << w1_rows.unwrap();
  static_assert(kRowsCta / 8 <= kThreads, "epilogue covers 8 rows per thread");
  constexpr std::size_t kSmem = kUSmemBytes + 16 * kRowsCta + 64 * kTilesPerCta + kTilesSmem * kTileVecs * 16;
  launch_cooperative(
      markov_walk_i8_kernel,
      device.unwrap(),
      static_cast<uint32_t>(rows_pad.unwrap() / kRowsCta),
      kThreads,
      kSmem,
      kSmem,
      static_cast<const uint4*>(frag.data_ptr()),
      static_cast<const float*>(row_scale.data_ptr()),
      static_cast<const uint32_t*>(w1q.data_ptr()),
      static_cast<const __nv_bfloat16*>(base.data_ptr()),
      static_cast<const int64_t*>(anchor.data_ptr()),
      static_cast<int64_t*>(tokens.data_ptr()),
      corrected.has_value() ? static_cast<__nv_bfloat16*>(corrected.value().data_ptr()) : nullptr,
      static_cast<u64*>(state.data_ptr()),
      static_cast<const float*>(temps.data_ptr()),
      static_cast<int>(num_steps),
      static_cast<int>(ld_v),
      static_cast<int>(valid_rows),
      static_cast<u64>(seed));
}

}  // namespace sglang::dspark_markov_walk::i8
