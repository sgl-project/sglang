/*
 * Copyright (c) 2026 by SGLang team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/// \file verify_topp_chain.cuh
/// \brief One target sample per verify row from top-p(softmax(logits / T)), for
///        EAGLE topk = 1 chains, in one kernel and the same on every TP rank.
///
/// For a topk = 1 chain with threshold_single = threshold_acc = 1, the target-only
/// walk (accept draft x with probability p'(x), else draw from p' without x) has
/// the same joint law as: draw y_i ~ p' independently on every verify row, accept
/// while y_i equals the next draft token, and emit the first mismatching y_i (or
/// the last row's y). This kernel produces the y_i; the chain acceptance runs in
/// the existing fused chain kernel, as for greedy rows.
///
/// One 8-CTA cluster per row holds the row in registers. Round 0: every CTA
/// publishes its local max of logit / T and its sum of exp(. - max) to the whole
/// cluster (DSMEM push, one cluster barrier); every CTA then holds the row max M,
/// the normalizer Z and every CTA's prefix. A Philox coin picks the token whose
/// running mass passes u * Z (inverse CDF in a fixed (CTA, thread, item) order);
/// its owner pushes (token, mass) to the cluster (barrier). With top_p < 1 every
/// CTA sums the mass strictly above the sample (barrier): the sample is in the
/// top-p set iff that mass is below top_p * Z (the set flashinfer's AIR top-p
/// renorm keeps: v >= v*, v* the largest value with S(>= v*) >= top_p * Z). If it
/// is not, the sample's mass becomes a lower bound and the next round samples
/// from the masses above it (rejection rounds, as flashinfer's
/// TopPSamplingFromProbKernel); a bounded round count falls back to the argmax,
/// which is always kept.
///
/// Every reduction has a fixed order and the coins come from Philox keyed by the
/// TP-synchronized seed and a device step counter that the last CTA of a call
/// advances, so equal inputs give bitwise-equal outputs on every rank, eagerly and
/// under CUDA-graph replay, and no broadcast is needed.

#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cooperative_groups.h>

#include <cmath>
#include <cstdint>

namespace sglang {

namespace verify_topp {

namespace cg = cooperative_groups;

/// CTAs per row (one portable cluster).
inline constexpr uint32_t kCluster = 8;
/// Bound on sampling rounds per row; the argmax is taken past it.
inline constexpr uint32_t kMaxRounds = 32;
inline constexpr int32_t kNone = INT32_MAX;

/// Persistent per-device state (int64 [4]): coin-stream step, arrival count of the
/// current call (zero between calls), rows that needed more than one round, rows
/// that hit kMaxRounds.
enum StateSlot : int { kStep = 0, kArrive = 1, kMultiRound = 2, kCapped = 3 };

struct Params {
  const float* __restrict__ logits;
  int64_t row_stride;
  const float* __restrict__ temperatures;
  int64_t temperature_stride;
  const float* __restrict__ top_ps;  ///< nullptr: top_p = 1 on every row
  int64_t top_p_stride;
  int32_t* __restrict__ tokens;  ///< [rows]: the target sample
  int32_t* __restrict__ argmax;  ///< [rows]: raw-logit argmax (kArgmax)
  unsigned long long* __restrict__ state;
  uint32_t num_draft;
  uint32_t vocab;
  uint64_t seed;
};

/// What every CTA of a row pushes to every other in round 0.
struct alignas(16) Publish {
  float max_scaled;  ///< max of logit / T over the CTA's tokens (-inf if none)
  float sum_exp;     ///< sum of exp(logit / T - max_scaled)
  float max_logit;   ///< raw-logit argmax value (torch.argmax order)
  int32_t argmax;    ///< its index
};

/// \brief Whether (a, ai) comes before (b, bi) in torch.argmax order: NaN first,
/// then larger values, ties to the lower index.
SGL_DEVICE bool argmax_before(float a, int32_t ai, float b, int32_t bi) {
  const bool a_nan = a != a, b_nan = b != b;
  if (a_nan != b_nan) return a_nan;
  if (a_nan || a == b) return ai < bi;
  return a > b;
}

/// \brief Philox4x32-10 (Random123).
SGL_DEVICE uint4 philox4x32_10(uint4 counter, uint2 key) {
  constexpr uint32_t kM0 = 0xD2511F53u, kM1 = 0xCD9E8D57u;
  constexpr uint32_t kW0 = 0x9E3779B9u, kW1 = 0xBB67AE85u;
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    const uint32_t hi0 = __umulhi(kM0, counter.x), lo0 = kM0 * counter.x;
    const uint32_t hi1 = __umulhi(kM1, counter.z), lo1 = kM1 * counter.z;
    counter = make_uint4(hi1 ^ counter.y ^ key.x, lo1, hi0 ^ counter.w ^ key.y, lo0);
    key.x += kW0;
    key.y += kW1;
  }
  return counter;
}

/// \brief Coin of sampling round `round` of verify row `row` in call `step`, in [0, 1 - 2^-24].
/// Counter = (step, row, round / 4); key = seed.
SGL_DEVICE float philox_coin(uint64_t seed, uint64_t step, uint32_t row, uint32_t round) {
  const uint4 bits = philox4x32_10(
      make_uint4(static_cast<uint32_t>(step), static_cast<uint32_t>(step >> 32), row, round >> 2),
      make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)));
  const uint32_t lane = round & 3u;
  const uint32_t word = lane == 0 ? bits.x : (lane == 1 ? bits.y : (lane == 2 ? bits.z : bits.w));
  return static_cast<float>(word >> 8) * 0x1p-24f;
}

/// Cluster-wide values every CTA reads after an exchange (written by thread 0).
struct RowView {
  float mass[kCluster];    ///< each CTA's mass in the row scale (masked after round 0)
  float prefix[kCluster];  ///< exclusive prefix of `mass` in CTA order
  float total;
  float scale;  ///< this CTA's local-to-row scale exp(cta_max - row_max)
};

/// \brief Thread 0: the round's per-CTA masses into `view` (fixed CTA order).
SGL_DEVICE void fill_view(RowView& view, const float* mass, float scale) {
  float total = 0.0f;
#pragma unroll
  for (uint32_t r = 0; r < kCluster; ++r) {
    view.mass[r] = mass[r];
    view.prefix[r] = total;
    total += mass[r];
  }
  view.total = total;
  view.scale = scale;
}

/**
 * \brief One target sample per verify row (and its raw-logit argmax).
 *
 * Grid (kCluster, rows), cluster (kCluster, 1, 1), kThreads threads. CTA c of a
 * row owns tokens c * kChunk + k * kThreads + t (k < kItems); the sampling order
 * is (c, t, k). Cross-CTA data is pushed into every CTA's shared memory before a
 * cluster barrier, so no CTA reads another's shared memory after its last barrier.
 * Round 0 takes 3 cluster barriers with top_p < 1 (2 without), plus the start barrier, and 4-5 CTA barriers.
 */
template <uint32_t kThreads, uint32_t kItems, bool kArgmax, bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 1) verify_topp_chain_kernel(const Params p) {
  using namespace device;
  constexpr uint32_t kWarps = kThreads / kWarpThreads;
  constexpr uint32_t kChunk = kItems * kThreads;
  static_assert(kThreads % kWarpThreads == 0 && kWarps <= kWarpThreads);

  __shared__ Publish s_pub[kCluster];  // round 0, pushed by every CTA
  __shared__ float s_w[kCluster];      // masked masses of rounds > 0, pushed
  __shared__ float s_g[kCluster];      // mass above the sample, pushed
  __shared__ int32_t s_pick_token;     // pushed by the round's owner
  __shared__ float s_pick_value;
  __shared__ RowView s_view;
  __shared__ int32_t s_amax;
  // One buffer per CTA-wide reduction, so none needs a barrier before its reuse.
  __shared__ float s_wmax[kWarps], s_wval[kWarps], s_wscan[kWarps], s_wsum[kWarps];
  __shared__ int32_t s_widx[kWarps];
  __shared__ int32_t s_first, s_last;

  cg::cluster_group cluster = cg::this_cluster();
  const uint32_t crank = cluster.block_rank();
  const uint32_t row = blockIdx.y;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % kWarpThreads, warp_id = tid / kWarpThreads;
  const uint32_t request = row / p.num_draft;
  const uint32_t base = crank * kChunk;
  const uint64_t step = static_cast<uint64_t>(reinterpret_cast<volatile unsigned long long*>(p.state)[kStep]);
  if (tid == 0) {
    s_first = kNone;
    s_last = -1;
    s_pick_token = -1;
  }
  // A CTA may write another's shared memory only once every CTA of the cluster has
  // started: arrive now, wait before the first push (the load and exp hide the barrier).
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  PDLWaitPrimary<kUsePDL>();

  const float temperature = p.temperatures[static_cast<int64_t>(request) * p.temperature_stride];
  const float top_p = p.top_ps != nullptr ? p.top_ps[static_cast<int64_t>(request) * p.top_p_stride] : 1.0f;
  const float* src = p.logits + static_cast<int64_t>(row) * p.row_stride;
  const bool unit_temperature = temperature == 1.0f;

  // ---- load, scale, local max, raw argmax (the round-cap fallback, always kept) ----
  float e[kItems];
  float local_max = -INFINITY;
  float best_value = -INFINITY;
  int32_t best_index = kNone;
#pragma unroll
  for (uint32_t k = 0; k < kItems; ++k) {
    const uint32_t idx = base + k * kThreads + tid;
    float scaled = -INFINITY;
    if (idx < p.vocab) {
      const float raw = src[idx];
      if (argmax_before(raw, static_cast<int32_t>(idx), best_value, best_index)) {
        best_value = raw;
        best_index = static_cast<int32_t>(idx);
      }
      // IEEE round-to-nearest division, as torch's logits / temperatures (exact at T = 1).
      scaled = unit_temperature ? raw : __fdiv_rn(raw, temperature);
    }
    e[k] = scaled;
    local_max = fmaxf(local_max, scaled);
  }
  // CTA barrier 1: warp max and warp argmax.
  local_max = warp::reduce_max(local_max);
#pragma unroll
  for (uint32_t mask = kWarpThreads / 2; mask >= 1; mask >>= 1) {
    const float other_value = __shfl_xor_sync(warp::kFullMask, best_value, mask);
    const int32_t other_index = __shfl_xor_sync(warp::kFullMask, best_index, mask);
    if (argmax_before(other_value, other_index, best_value, best_index)) {
      best_value = other_value;
      best_index = other_index;
    }
  }
  if (lane == 0) {
    s_wmax[warp_id] = local_max;
    s_wval[warp_id] = best_value;
    s_widx[warp_id] = best_index;
  }
  __syncthreads();
  float cta_max = -INFINITY;
#pragma unroll
  for (uint32_t w = 0; w < kWarps; ++w)
    cta_max = fmaxf(cta_max, s_wmax[w]);

  // CTA barrier 2: exclusive prefix (thread order) of the local masses.
  float thread_mass = 0.0f;
#pragma unroll
  for (uint32_t k = 0; k < kItems; ++k) {
    e[k] = (cta_max == -INFINITY || e[k] == -INFINITY) ? 0.0f : __expf(e[k] - cta_max);
    thread_mass += e[k];
  }
  float excl, cta_sum;
  {
    const float inclusive = warp::inclusive_sum(thread_mass);
    const float in_warp = __shfl_up_sync(warp::kFullMask, inclusive, 1);
    if (lane == kWarpThreads - 1) s_wscan[warp_id] = inclusive;
    __syncthreads();
    float before = 0.0f, sum = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kWarps; ++w) {
      const float v = s_wscan[w];
      if (w < warp_id) before += v;
      sum += v;
    }
    excl = before + (lane == 0 ? 0.0f : in_warp);
    cta_sum = sum;
  }

  // ---- round-0 exchange (cluster barrier 1) ----
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  if (tid == 0) {
    float amax_value = s_wval[0];
    int32_t amax_index = s_widx[0];
#pragma unroll
    for (uint32_t w = 1; w < kWarps; ++w) {
      if (argmax_before(s_wval[w], s_widx[w], amax_value, amax_index)) {
        amax_value = s_wval[w];
        amax_index = s_widx[w];
      }
    }
    const Publish pub{cta_max, cta_sum, amax_value, amax_index};
#pragma unroll
    for (uint32_t r = 0; r < kCluster; ++r)
      *cluster.map_shared_rank(&s_pub[crank], r) = pub;
  }
  cluster.sync();
  // CTA barrier 3: the row view.
  if (tid == 0) {
    float row_max = -INFINITY;
#pragma unroll
    for (uint32_t r = 0; r < kCluster; ++r)
      row_max = fmaxf(row_max, s_pub[r].max_scaled);
    float mass[kCluster];
    float scale = 0.0f;
#pragma unroll
    for (uint32_t r = 0; r < kCluster; ++r) {
      const float m = s_pub[r].max_scaled;
      const float f = m == -INFINITY ? 0.0f : expf(m - row_max);
      if (r == crank) scale = f;
      mass[r] = s_pub[r].sum_exp * f;
    }
    fill_view(s_view, mass, scale);
    float amax_value = s_pub[0].max_logit;
    int32_t amax_index = s_pub[0].argmax;
#pragma unroll
    for (uint32_t r = 1; r < kCluster; ++r) {
      if (argmax_before(s_pub[r].max_logit, s_pub[r].argmax, amax_value, amax_index)) {
        amax_value = s_pub[r].max_logit;
        amax_index = s_pub[r].argmax;
      }
    }
    s_amax = amax_index;
  }
  __syncthreads();
  const float norm = s_view.total;  // Z, the top-p test's reference mass
  const float scale = s_view.scale;

  // ---- sampling rounds ----
  int32_t token = -1;
  float lower = -1.0f;  // round > 0: masses <= lower are out
  bool accepted = false;
  uint32_t round = 0;
  for (; round < kMaxRounds; ++round) {
    if (round > 0) {
      float masked = 0.0f;
#pragma unroll
      for (uint32_t k = 0; k < kItems; ++k)
        masked += e[k] > lower ? e[k] : 0.0f;
      const float inclusive = warp::inclusive_sum(masked);
      const float in_warp = __shfl_up_sync(warp::kFullMask, inclusive, 1);
      if (lane == kWarpThreads - 1) s_wscan[warp_id] = inclusive;
      if (tid == 0) {
        s_first = kNone;
        s_last = -1;
      }
      __syncthreads();
      float before = 0.0f, sum = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < kWarps; ++w) {
        const float v = s_wscan[w];
        if (w < warp_id) before += v;
        sum += v;
      }
      excl = before + (lane == 0 ? 0.0f : in_warp);
      if (tid == 0) {
#pragma unroll
        for (uint32_t r = 0; r < kCluster; ++r)
          *cluster.map_shared_rank(&s_w[crank], r) = sum;
      }
      cluster.sync();
      if (tid == 0) fill_view(s_view, s_w, scale);
      __syncthreads();
    }
    const float target = philox_coin(p.seed, step, row, round) * s_view.total;
    // The owner: the first CTA whose running mass passes the target, else
    // (rounding) the last CTA with mass. Same on every CTA.
    int32_t owner = -1, last_owner = -1;
#pragma unroll
    for (uint32_t r = 0; r < kCluster; ++r) {
      if (s_view.mass[r] > 0.0f) {
        last_owner = static_cast<int32_t>(r);
        if (owner < 0 && s_view.prefix[r] + s_view.mass[r] > target) owner = static_cast<int32_t>(r);
      }
    }
    const bool clamp = owner < 0;
    if (clamp) owner = last_owner;
    if (owner < 0) break;  // no mass at all: the argmax below

    if (static_cast<int32_t>(crank) == owner) {
      // In the CTA's own scale in round 0, in the row scale after.
      const float local_target =
          clamp ? INFINITY : (round == 0 ? (target - s_view.prefix[owner]) / scale : target - s_view.prefix[owner]);
      float running = excl;
      int32_t first = kNone, last = -1;
#pragma unroll
      for (uint32_t k = 0; k < kItems; ++k) {
        const float m = (round == 0 || e[k] > lower) ? e[k] : 0.0f;
        if (m > 0.0f) {
          const int32_t pos = static_cast<int32_t>(tid * kItems + k);
          last = pos;
          running += m;
          if (first == kNone && running > local_target) first = pos;
        }
      }
      if (first != kNone) atomicMin(&s_first, first);
      if (last >= 0) atomicMax(&s_last, last);
      // CTA barrier 4 (owner only).
      __syncthreads();
      const int32_t pos = s_first != kNone ? s_first : s_last;
      if (pos >= 0 && static_cast<uint32_t>(pos) / kItems == tid) {
        const uint32_t k_pick = static_cast<uint32_t>(pos) % kItems;
        float value = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < kItems; ++k)
          if (k == k_pick) value = e[k];
        const int32_t picked_token = static_cast<int32_t>(base + k_pick * kThreads + tid);
        const float picked_value = round == 0 ? value * scale : value;
#pragma unroll
        for (uint32_t r = 0; r < kCluster; ++r) {
          *cluster.map_shared_rank(&s_pick_token, r) = picked_token;
          *cluster.map_shared_rank(&s_pick_value, r) = picked_value;
        }
      }
    }
    // Cluster barrier 2.
    cluster.sync();
    token = s_pick_token;
    const float picked = s_pick_value;
    if (top_p >= 1.0f || token < 0) {
      accepted = token >= 0;
      break;
    }
    // Top-p test: the mass strictly above the sample, in the row scale (CTA barrier 5, cluster barrier 3).
    float above = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < kItems; ++k) {
      const float m = round == 0 ? e[k] * scale : e[k];
      above += m > picked ? m : 0.0f;
    }
    above = warp::reduce_sum(above);
    if (lane == 0) s_wsum[warp_id] = above;
    __syncthreads();
    if (tid == 0) {
      float cta_above = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < kWarps; ++w)
        cta_above += s_wsum[w];
#pragma unroll
      for (uint32_t r = 0; r < kCluster; ++r)
        *cluster.map_shared_rank(&s_g[crank], r) = cta_above;
    }
    cluster.sync();
    float mass_above = 0.0f;
#pragma unroll
    for (uint32_t r = 0; r < kCluster; ++r)
      mass_above += s_g[r];
    if (mass_above < top_p * norm) {
      accepted = true;
      break;
    }
    if (round == 0) {
#pragma unroll
      for (uint32_t k = 0; k < kItems; ++k)
        e[k] *= scale;
    }
    lower = picked;
  }

  if (crank == 0 && tid == 0) {
    const int32_t amax = s_amax != kNone ? s_amax : 0;
    p.tokens[row] = accepted ? token : amax;
    if constexpr (kArgmax) p.argmax[row] = amax;
    if (round > 0) atomicAdd(&p.state[kMultiRound], 1ull);
    if (!accepted) atomicAdd(&p.state[kCapped], 1ull);
  }
  // The last CTA of the call advances the coin stream; every CTA read it before arriving.
  if (tid == 0) {
    __threadfence();
    const unsigned long long ctas = static_cast<unsigned long long>(gridDim.x) * gridDim.y;
    if (atomicAdd(&p.state[kArrive], 1ull) == ctas - 1) {
      p.state[kArrive] = 0;
      atomicAdd(&p.state[kStep], 1ull);
      __threadfence();
    }
  }
}

}  // namespace verify_topp

/**
 * \brief One sample per verify row from top-p(softmax(logits / T)) (kArgmax: and
 *        the raw-logit argmax), the same on every rank.
 *
 * \tparam kThreads threads per CTA (512 or 1024)
 * \tparam kItems tokens per thread; kCluster * kThreads * kItems >= vocab
 * \param logits [rows, vocab] fp32, unit column stride, rows = bs * num_draft
 * \param temperatures [bs, 1] fp32
 * \param top_ps [bs] fp32 (read when has_top_p)
 * \param tokens [rows] int32 (written)
 * \param argmax [rows] int32 (written with kArgmax)
 * \param state [4] int64, persistent, zero-initialized (see StateSlot)
 */
template <uint32_t kThreads, uint32_t kItems, bool kArgmax, bool kUsePDL>
void verify_topp_chain_sample(
    tvm::ffi::TensorView logits,
    tvm::ffi::TensorView temperatures,
    tvm::ffi::TensorView top_ps,
    tvm::ffi::TensorView tokens,
    tvm::ffi::TensorView argmax,
    tvm::ffi::TensorView state,
    int64_t num_draft,
    int64_t has_top_p,
    int64_t seed) {
  using namespace host;
  using namespace verify_topp;

  SymbolicSize rows{"rows"}, vocab{"vocab"}, batch_size{"batch_size"};
  SymbolicSize row_stride{"row_stride"}, temperature_stride{"temperature_stride"};
  SymbolicDevice device;
  device.set_options<kDLCUDA>();
  TensorMatcher({rows, vocab}).with_strides({row_stride, 1}).with_dtype<float>().with_device(device).verify(logits);
  TensorMatcher({batch_size, 1})
      .with_strides({temperature_stride, 1})
      .with_dtype<float>()
      .with_device(device)
      .verify(temperatures);
  TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(tokens);
  TensorMatcher({4}).with_dtype<int64_t>().with_device(device).verify(state);
  const int64_t bs = batch_size.unwrap();
  const int64_t n = rows.unwrap();
  const int64_t v = vocab.unwrap();
  SymbolicSize top_p_stride{"top_p_stride"};
  if (has_top_p) {
    TensorMatcher({bs}).with_strides({top_p_stride}).with_dtype<float>().with_device(device).verify(top_ps);
  }
  if constexpr (kArgmax) {
    TensorMatcher({n}).with_dtype<int32_t>().with_device(device).verify(argmax);
  }
  CHECK_HOST(num_draft > 0 && bs * num_draft == n) << "rows must be bs * num_draft, got " << n;
  CHECK_HOST(v > 0 && v <= int64_t{kCluster} * kThreads * kItems) << "vocab " << v << " exceeds the instantiation";
  CHECK_HOST(n <= 65535) << "too many verify rows: " << n;
  if (n == 0) return;

  Params params{};
  params.logits = static_cast<const float*>(logits.data_ptr());
  params.row_stride = row_stride.unwrap();
  params.temperatures = static_cast<const float*>(temperatures.data_ptr());
  params.temperature_stride = temperature_stride.unwrap();
  params.top_ps = has_top_p ? static_cast<const float*>(top_ps.data_ptr()) : nullptr;
  params.top_p_stride = has_top_p ? top_p_stride.unwrap() : 0;
  params.tokens = static_cast<int32_t*>(tokens.data_ptr());
  params.argmax = kArgmax ? static_cast<int32_t*>(argmax.data_ptr()) : nullptr;
  params.state = static_cast<unsigned long long*>(state.data_ptr());
  params.num_draft = static_cast<uint32_t>(num_draft);
  params.vocab = static_cast<uint32_t>(v);
  params.seed = static_cast<uint64_t>(seed);

  LaunchKernel(dim3(kCluster, static_cast<uint32_t>(n)), kThreads, device.unwrap())
      .config({.use_pdl = kUsePDL, .cluster_dim = dim3(kCluster, 1, 1)})(
          verify_topp_chain_kernel<kThreads, kItems, kArgmax, kUsePDL>, params);
}

}  // namespace sglang
