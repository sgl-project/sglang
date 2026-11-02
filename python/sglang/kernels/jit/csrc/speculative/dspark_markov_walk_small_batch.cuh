// SPDX-License-Identifier: Apache-2.0
// Batched int8 DSpark markov walk (serving: bs 1..4; the launcher takes 1..8): B independent walks share all of W2 on
// chip, in the SMEM and registers of every SM, and the K draft steps run in one persistent cooperative launch.
//
//   for every request b:  prev_b = anchor_b;  for k < K:  logit = bf16(base[b, k] + W2 W1[prev_b]);
//                         tok[b, k] = argmax(logit)  or  argmax(logit / T_b + Gumbel);  prev_b = tok[b, k]
//
// Numerics. W2 is int8 with one fp32 scale per row; u = W1[prev] is two int8 planes, u ~= s_hi q_hi + s_lo q_lo, so
//   W2[r] . u = s_row (s_hi <q_w, q_hi> + s_lo <q_w, q_lo>)
// with both inner products exact in int32 (mma.sync m16n8k32 s8.s8.s32).
//
// Layout. One CTA of 256 threads per SM, grid = rows_pad / 1152. CTA c owns 72 consecutive 16-row tiles: the first
// kTilesSmem in SMEM, then kRegTiles per warp in registers. frag[tile][chunk][lane] is the lane's 16-B A fragment of
// k32 chunk `chunk` (markov_walk.py _build_frag).
//
// One mma pass serves 4 requests: B column n carries request 4p + n / 2 (q_hi for even n, q_lo for odd), so lane tig
// holds request 4p + tig's (hi, lo) int32 dots of its rows, and up to 4 requests cost the tensor work of one. Per step
// and pass p (the next pass's u rows already in flight): GEMV over the CTA's tiles, bias into a double-buffered SMEM
// slab, barrier, then the epilogue of those requests (base + bias, bf16 rounding, corrected for sampling requests,
// Gumbel noise in place, whole warps per request for the argmax); the next pass's base rows are prefetched behind it.
// After the last pass one warp runs the cross-CTA exchange for every request (lane b: request b's {key, count} pair).
//
// The fp32 op sequence before the bf16 rounding is pinned (__fmul_rn / __fmaf_rn / __fadd_rn), so an exact reference
// replays it bit for bit; argmax and Gumbel-max read the rounded values and corrected receives exactly those bits.
// State: int64 [16 + 2 S kStepWords] = round | pad to 128 B | two sets of [S >= num_steps][64 requests] {key, count}
// pairs. Round r uses set r & 1; CTA 0 clears the other set at the start and stores round + 1 after the last step.
// Nothing is reset between launches.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "dspark_markov_walk_common.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace sglang::dspark_markov_walk::small_batch {

inline constexpr int32_t kLaunchBatch = 8;        // two mma passes
inline constexpr int32_t kPassBatch = 4;          // requests per mma pass (column pairs)
inline constexpr int32_t kChunks = kRowsCta / 8;  // epilogue items per request, 8 rows each

struct Smem {
  bf16_t base[2 * kPassBatch * kRowsCta];  // [item parity][request of the pass][row]
  float bias[2 * kPassBatch * kRowsCta];   // [item parity][request of the pass][row]
  float scale[kTilesPerCta * kTileRows];
  uint4 w2[kTilesSmem * kTileVecs];
  uint32_t u[(kLaunchBatch + 1) * kURowWords];  // nb + 1 W1q rows, row nb = zeros; the launch sizes it by nb
};

// The fp32 bias of rows l0 + g and l0 + g + 8, rounding sequence pinned: u_lo c_lo (RN), fma(u_hi, c_hi, .), times
// the row scale (RN). A last-bit difference in fp32 can cross a bf16 rounding boundary, so the exact reference replays
// this order.
SGL_DEVICE void store_bias_rn(
    const int32_t (&c)[4], float scale_g, float scale_g8, float u_hi, float u_lo, float* bias, int32_t l0, int32_t g) {
  bias[l0 + g] =
      __fmul_rn(scale_g, __fmaf_rn(u_hi, static_cast<float>(c[0]), __fmul_rn(u_lo, static_cast<float>(c[1]))));
  bias[l0 + g + 8] =
      __fmul_rn(scale_g8, __fmaf_rn(u_hi, static_cast<float>(c[2]), __fmul_rn(u_lo, static_cast<float>(c[3]))));
}

// Whole warps per request in the epilogue: all 8 for a lone request (sampling's noise is the bulk of the work)
SGL_DEVICE int32_t warps_per_request(int32_t pass_requests) {
  return pass_requests == 1 ? kWarps : pass_requests == 2 ? kWarps / 2 : kWarps / 4;
}

__global__ void __launch_bounds__(kThreads, 1) markov_walk_small_batch_kernel(
    const uint4* __restrict__ frag,
    const float* __restrict__ row_scale,
    const uint32_t* __restrict__ w1q,
    const bf16_t* __restrict__ base,
    const int64_t* __restrict__ anchor,
    int64_t* __restrict__ tokens,
    bf16_t* __restrict__ corrected,
    uint64_t* __restrict__ state,
    const float* __restrict__ temps,
    int32_t nb,
    int32_t num_steps,
    int32_t state_steps,
    int32_t kb,
    int32_t ld,
    int32_t valid_rows,
    uint64_t seed) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  auto& smem = *reinterpret_cast<Smem*>(smem_raw);
  __shared__ uint64_t s_warp_best[kMaxBatch][kWarps];  // per request: its epilogue warps' best keys
  __shared__ uint32_t s_token[kMaxBatch];
  __shared__ float s_inv_t[kMaxBatch];
  __shared__ uint64_t s_mbar_load;
  __shared__ uint64_t s_mbar_u[kMaxBatch / kPassBatch];  // pass p's u rows landed, phase = step parity
  const int32_t lane_id = threadIdx.x & 31, warp_id = threadIdx.x >> 5;
  const int32_t g = lane_id >> 2, tig = lane_id & 3;
  const int32_t first_tile = blockIdx.x * kTilesPerCta;
  const int32_t row_base = first_tile * kTileRows;
  const auto cta_frag = frag + static_cast<size_t>(first_tile) * kTileVecs;

  const auto round = ptx::ld_relaxed_u64(state);
  const int32_t set = static_cast<int32_t>(round & 1);
  const int32_t set_words = state_steps * kStepWords;
  const auto pairs = state + 16 + set * set_words;
  const auto pair = [&](int32_t k, int32_t b) { return pairs + kPairWords * (k * kMaxBatch + b); };
  if (blockIdx.x == 0) {
    // The set round r + 1 uses, which may differ in B, and in K <= state_steps
    for (int32_t i = threadIdx.x; i < state_steps * kMaxBatch * 2; i += kThreads) {
      ptx::st_relaxed_u64(state + 16 + (set ^ 1) * set_words + (i >> 1) * kPairWords + (i & 1), 0);
    }
  }
  const auto philox_key = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
  const int32_t num_passes = (nb + kPassBatch - 1) / kPassBatch;
  const int32_t num_items = num_steps * num_passes;  // (step, pass) in walk order

  // Base rows of item it = (k, pass) into slab it & 1: the pass's requests, this CTA's rows. Chunks at or past
  // valid_rows (the last CTA's tail) copy the last valid chunk instead, in bounds whatever the row stride; the
  // epilogue skips them.
  const int32_t last_chunk_row = valid_rows - 8;
  const auto prefetch_base = [&](int32_t it) {
    const int32_t k = it / num_passes, r0 = (it % num_passes) * kPassBatch, nr = min(kPassBatch, nb - r0);
    const auto dst = smem.base + (it & 1) * kPassBatch * kRowsCta;
    for (int32_t i = threadIdx.x; i < nr * kChunks; i += kThreads) {
      const int32_t rr = i / kChunks, c = i % kChunks;
      ptx::cp_async_cg_16B(
          dst + rr * kRowsCta + c * 8,
          base + (static_cast<size_t>(r0 + rr) * kb + k) * ld + min(row_base + c * 8, last_chunk_row));
    }
    ptx::cp_async_commit();
  };

  // Once per round: the base of item 0, the W2 slice and its row scales on chip
  prefetch_base(0);
  const auto evict_first = ptx::create_policy_evict_first();
  if (threadIdx.x == 0) {
    constexpr uint32_t kW2Vecs = kTilesSmem * kTileVecs;
    constexpr uint32_t kCopyVecs = 1024;  // 16 KiB per bulk copy
    ptx::mbar_init(&s_mbar_load, 1);
    for (int32_t p = 0; p < num_passes; ++p) {
      ptx::mbar_init(&s_mbar_u[p], kThreads);  // every thread arrives
    }
    ptx::fence_mbarrier_init();
    ptx::mbar_arrive_expect_tx(&s_mbar_load, sizeof(smem.w2) + sizeof(smem.scale));
    for (uint32_t i = 0; i < kW2Vecs; i += kCopyVecs) {
      ptx::cp_async_bulk_g2s(
          smem.w2 + i, cta_frag + i, std::min(kCopyVecs, kW2Vecs - i) * 16, &s_mbar_load, evict_first);
    }
    ptx::cp_async_bulk_g2s(smem.scale, row_scale + row_base, sizeof(smem.scale), &s_mbar_load, evict_first);
  }
  for (int32_t b = threadIdx.x; b < nb; b += kThreads) {
    s_token[b] = clamp_row(anchor[b], valid_rows);
    s_inv_t[b] = inv_temperature(temps[b]);
  }
  for (int32_t i = threadIdx.x; i < kURowWords; i += kThreads) {
    smem.u[nb * kURowWords + i] = 0;  // the padding request
  }
  uint4 w2_reg[kRegTiles][8];
#pragma unroll
  for (int32_t i = 0; i < kRegTiles; ++i) {
    const int32_t t = kTilesSmem + warp_id * kRegTiles + i;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      w2_reg[i][j] = ptx::ld_cg_hint_v4(cta_frag + (t * 8 + j) * 32 + lane_id, evict_first);
    }
  }
  // Only thread 0 waits: another thread could reach the wait before the init. The barrier below publishes the tiles.
  if (threadIdx.x == 0) ptx::mbar_wait(&s_mbar_load, 0);
  __syncthreads();

  // The u rows of pass p (requests 4p ..) by cp.async over the threads, fetched one pass ahead; completion is counted
  // on the pass's mbarrier, every thread arriving once per pass and step
  const auto fetch_u = [&](int32_t p) {
    constexpr int32_t kRowVecs = kURowBytes / 16;
    const int32_t r0 = p * kPassBatch, nr = min(kPassBatch, nb - r0);
    // %tid.x re-read here: no threadIdx-derived index is kept live across the step (255 registers, 0 spill)
    const uint32_t tid = ptx::tid_x();
    if (tid < nr * kRowVecs) {
      const int32_t b = r0 + tid / kRowVecs, v = tid % kRowVecs;
      // Indexed in 16-B vectors: word offsets (v * 4) add 6 SASS to the step loop
      ptx::cp_async_cg_16B(
          reinterpret_cast<uint4*>(smem.u) + b * kRowVecs + v,
          reinterpret_cast<const uint4*>(w1q) + static_cast<size_t>(s_token[b]) * kRowVecs + v);
    }
    ptx::cp_async_mbar_arrive_noinc(&s_mbar_u[p]);
  };
  constexpr int32_t kXchgWarp = kWarps - 1;
  for (int32_t k = 0; k < num_steps; ++k) {
    // The u rows of every request, prefetched into L2 by the CTAs that held a local winner
    fetch_u(0);
    for (int32_t p = 0; p < num_passes; ++p) {
      const int32_t it = k * num_passes + p, r0 = p * kPassBatch, nr = min(kPassBatch, nb - r0);
      if (p + 1 < num_passes) fetch_u(p + 1);
      ptx::mbar_wait(&s_mbar_u[p], k & 1);
      {
        const int32_t rq = r0 + (g >> 1);
        int32_t acc[kWarpTiles][4];
        mma_tiles(acc, w2_reg, smem.w2, smem.u + (rq < nb ? rq : nb) * kURowWords, warp_id, lane_id);
        if (tig < nr) {  // lane tig: request r0 + tig, rows g and g + 8 of every tile of this warp
          const auto u = smem.u + (r0 + tig) * kURowWords;
          const float u_hi = __uint_as_float(u[128]), u_lo = __uint_as_float(u[129]);
          const auto dst = smem.bias + ((it & 1) * kPassBatch + tig) * kRowsCta;
#pragma unroll
          for (int32_t i = 0; i < kWarpTiles; ++i) {
            const int32_t t = i < kRegTiles ? kTilesSmem + warp_id * kRegTiles + i : warp_id + kWarps * (i - kRegTiles);
            store_bias_rn(
                acc[i], smem.scale[t * kTileRows + g], smem.scale[t * kTileRows + g + 8], u_hi, u_lo, dst, t * 16, g);
          }
        }
      }
      ptx::cp_async_wait<0>();  // this item's base, the only group in flight
      __syncthreads();          // bias and base of item it visible; every thread is past item it - 1's epilogue
      if (it + 1 < num_items) prefetch_base(it + 1);  // into the slab item it - 1 used
      // Epilogue: lane's 8-row chunks (warp_id % wpr) * 32 + lane_id + 32 wpr j, a plain warp max per request
      const int32_t wpr = warps_per_request(nr);
      const int32_t rr = warp_id / wpr;
      if (rr < nr) {
        const int32_t b = r0 + rr;
        const auto base_s = smem.base + ((it & 1) * kPassBatch + rr) * kRowsCta;
        const auto bias_s = smem.bias + ((it & 1) * kPassBatch + rr) * kRowsCta;
        const float inv_t = s_inv_t[b];
        uint64_t best = 0;
#pragma unroll 1
        for (int32_t c = (warp_id % wpr) * 32 + lane_id; c < kChunks; c += wpr * 32) {
          const int32_t l = c * 8;
          device::AlignedVector<bf16x2_t, 4> base_v;
          device::AlignedVector<float, 4> bias_v[2];
          base_v.load(base_s + l);
          bias_v[0].load(bias_s + l);
          bias_v[1].load(bias_s + l + 4);
          float lg[8];
#pragma unroll
          for (int32_t i = 0; i < 4; ++i) {
            const auto x = __bfloat1622float2(base_v[i]);
            lg[2 * i] = __fadd_rn(x.x, bias_v[i / 2][2 * (i % 2)]);
            lg[2 * i + 1] = __fadd_rn(x.y, bias_v[i / 2][2 * (i % 2) + 1]);
          }
          // The walk's logits are the bf16 roundings (RNE) of these, in corrected's memory order. Greedy takes the
          // max / first argmax on the packed values; the sampler unpacks them (bf16 -> fp32 is a shift).
          device::AlignedVector<uint32_t, 4> rounded;
#pragma unroll
          for (int32_t i = 0; i < 4; ++i) {
            rounded[i] = bf16x2_bits(__floats2bfloat162_rn(lg[2 * i], lg[2 * i + 1]));
          }
          const auto out = (static_cast<size_t>(b) * kb + k) * ld + row_base + l;
          const int32_t r = row_base + l;
          float m;
          int32_t idx;
          if (inv_t > 0.f) {
            // corrected: sampling requests (the verifier's q), valid rows only
            if (corrected && r < valid_rows) rounded.store(corrected + out);
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
              lg[2 * i] = __uint_as_float(rounded[i] << 16);
              lg[2 * i + 1] = __uint_as_float(rounded[i] & 0xFFFF0000u);
            }
            // Gumbel-max, Philox counter (row / 4, step | request << 8, round)
#pragma unroll
            for (int32_t h = 0; h < 2; ++h) {
              const auto x = philox(
                  make_uint4(
                      static_cast<uint32_t>((r + 4 * h) >> 2),
                      static_cast<uint32_t>(k) | (static_cast<uint32_t>(b) << 8),
                      static_cast<uint32_t>(round),
                      static_cast<uint32_t>(round >> 32)),
                  philox_key);
              const float noise[4] = {gumbel_fast(x.x), gumbel_fast(x.y), gumbel_fast(x.z), gumbel_fast(x.w)};
#pragma unroll
              for (int32_t i = 0; i < 4; ++i) {
                lg[4 * h + i] = __fmaf_rn(lg[4 * h + i], inv_t, noise[i]);
              }
            }
            m = fmaxf(fmaxf(fmaxf(lg[0], lg[1]), fmaxf(lg[2], lg[3])), fmaxf(fmaxf(lg[4], lg[5]), fmaxf(lg[6], lg[7])));
            idx = 7;
#pragma unroll
            for (int32_t i = 6; i >= 0; --i) {
              idx = lg[i] == m ? i : idx;  // ties take the smallest row, as torch.argmax
            }
          } else {
            const auto m2 = ptx::bf16x8_max(rounded[0], rounded[1], rounded[2], rounded[3]);
            idx = ptx::bf16x8_first_eq(rounded[0], rounded[1], rounded[2], rounded[3], m2);
            m = __uint_as_float(m2 << 16);
          }
          if (r < valid_rows) best = max(best, pack_key(m, r + idx));
        }
        const auto hi = static_cast<uint32_t>(best >> 32);
        const auto max_hi = device::warp::reduce_max(hi);
        const auto max_lo = device::warp::reduce_max(hi == max_hi ? static_cast<uint32_t>(best) : 0u);
        if (lane_id == 0) s_warp_best[b][warp_id % wpr] = (static_cast<uint64_t>(max_hi) << 32) | max_lo;
      }
    }
    __syncthreads();  // every request's warp bests are in s_warp_best
    if (warp_id == kXchgWarp) {
      // Per request: W1q[local winner] into L2 (the global winner's row is then an L2 hit for every CTA), red.max of
      // the key, then the arrival with release semantics
      for (int32_t b = lane_id; b < nb; b += 32) {
        const int32_t r0 = b / kPassBatch * kPassBatch;
        const int32_t wpr = warps_per_request(min(kPassBatch, nb - r0));
        auto best = s_warp_best[b][0];
        for (int32_t w = 1; w < wpr; ++w) {
          best = max(best, s_warp_best[b][w]);
        }
        const auto row = w1q + static_cast<size_t>(key_row(best)) * kURowWords;
#pragma unroll
        for (int32_t line = 0; line < 5; ++line) {
          ptx::prefetch_l2(row + line * 32);  // 128-B lines
        }
        ptx::red_relaxed_max_u64(pair(k, b), best);
      }
      uint64_t key[kMaxBatch / 32] = {};  // the winner keys of requests lane_id + 32 q; zeroed, so kept in registers
      for (int32_t b = lane_id; b < nb; b += 32) {
        ptx::red_release_add_u64(pair(k, b) + 1, 1);
      }
#pragma unroll
      for (int32_t q = 0; q < kMaxBatch / 32; ++q) {
        const int32_t b = lane_id + 32 * q;
        if (b < nb) {
          uint64_t v[2];
          do {
            ptx::ld_relaxed_v2_u64(pair(k, b), v);
          } while (v[1] < gridDim.x);
          key[q] = v[0];
        }
      }
#pragma unroll
      for (int32_t q = 0; q < kMaxBatch / 32; ++q) {
        const int32_t b = lane_id + 32 * q;
        if (b < nb) {
          const auto token = key_row(key[q]);
          s_token[b] = token;
          if (blockIdx.x == 0) tokens[static_cast<size_t>(b) * num_steps + k] = token;
        }
      }
    }
    __syncthreads();
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) ptx::st_relaxed_u64(state, round + 1);
}

// One round of the batched walk (bs 1..4 in serving; the launcher takes 1..8):
//   frag       int8 [rows_pad * 256], W2 in mma A-fragment order; rows_pad = grid x kRowsCta
//   row_scale  fp32 [rows_pad]; w1q uint8 [V, 528]
//   base       bf16 [B, >= num_steps, ld]; corrected: none, or shaped like base and written for T > 0 requests only
//   anchor     int64 [B]; tokens int64 [B * num_steps], row-major; temps fp32 [B], <= 0 or NaN for greedy
//   state      int64 [16 + 2 S kStepWords] with num_steps <= S <= kMaxSteps, 128-B aligned, zeroed once
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
  auto batch = SymbolicSize{"batch"};
  auto base_steps = SymbolicSize{"base_steps"};
  auto ld = SymbolicSize{"ld"};
  auto device = SymbolicDevice{};
  TensorMatcher({rows_pad})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device)
      .ensure_alignment(16)
      .verify(row_scale);
  TensorMatcher({rows_pad.unwrap() * 256})  //
      .with_dtype<int8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(frag);
  TensorMatcher({-1, kURowBytes})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(w1q);
  TensorMatcher({batch, base_steps, ld})  //
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(base);
  if (corrected.has_value()) {
    TensorMatcher({batch, base_steps, ld})  //
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(corrected.value());
  }
  TensorMatcher({batch})  //
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(anchor);
  TensorMatcher({batch})  //
      .with_dtype<float>()
      .with_device(device)
      .verify(temps);
  TensorMatcher({batch.unwrap() * num_steps})  //
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(tokens);
  TensorMatcher({-1})  //
      .with_dtype<int64_t>()
      .with_device(device)
      .ensure_alignment(128)
      .verify(state);
  const auto nb = batch.unwrap();
  const auto state_steps = (state.size(0) - 16) / (2 * kStepWords);
  CHECK_HOST(
      1 <= num_steps && num_steps <= state_steps && state_steps <= kMaxSteps &&
      state.size(0) == 16 + 2 * kStepWords * state_steps)
      << "markov walk: need int64 state [16 + " << 2 * kStepWords << " S] with 1 <= num_steps " << num_steps
      << " <= S <= " << kMaxSteps;
  CHECK_HOST(1 <= nb && nb <= kLaunchBatch) << "markov walk: batch " << nb << " outside [1, " << kLaunchBatch << "]";
  CHECK_HOST(rows_pad.unwrap() % kRowsCta == 0 && base_steps.unwrap() >= num_steps);
  CHECK_HOST(ld.unwrap() % 8 == 0 && nb * base_steps.unwrap() * ld.unwrap() < (int64_t{1} << 31));
  CHECK_HOST(
      valid_rows % 8 == 0 && 8 <= valid_rows && valid_rows <= std::min({ld.unwrap(), rows_pad.unwrap(), w1q.size(0)}))
      << "markov walk: vocab " << valid_rows << " must be a multiple of 8 within the base logits and the weights";
  constexpr auto kSmemU = offsetof(Smem, u);
  launch_cooperative<markov_walk_small_batch_kernel, sizeof(Smem)>(
      device.unwrap(),
      static_cast<uint32_t>(rows_pad.unwrap() / kRowsCta),
      kThreads,
      kSmemU + static_cast<std::size_t>(nb + 1) * kURowBytes,
      static_cast<const uint4*>(frag.data_ptr()),
      static_cast<const float*>(row_scale.data_ptr()),
      static_cast<const uint32_t*>(w1q.data_ptr()),
      static_cast<const bf16_t*>(base.data_ptr()),
      static_cast<const int64_t*>(anchor.data_ptr()),
      static_cast<int64_t*>(tokens.data_ptr()),
      corrected.has_value() ? static_cast<bf16_t*>(corrected.value().data_ptr()) : nullptr,
      static_cast<uint64_t*>(state.data_ptr()),
      static_cast<const float*>(temps.data_ptr()),
      static_cast<int32_t>(nb),
      static_cast<int32_t>(num_steps),
      static_cast<int32_t>(state_steps),
      static_cast<int32_t>(base_steps.unwrap()),
      static_cast<int32_t>(ld.unwrap()),
      static_cast<int32_t>(valid_rows),
      static_cast<uint64_t>(seed));
}

}  // namespace sglang::dspark_markov_walk::small_batch
