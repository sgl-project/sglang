// SPDX-License-Identifier: Apache-2.0
// CPU MXFP4 GEMV (marlin_v100 packed) + worker pool.
// Compiled as host C++ (-O3 -march=native), not nvcc.
#pragma once

#include "sm70_dsv41_spill_host_gemv_common.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <thread>
#include <vector>

namespace sglang::sm70_dsv41 {

inline float fp16_to_f32(uint16_t h) {
  return _cvtsh_ss(h);
}

inline uint16_t f32_to_fp16(float x) {
  return static_cast<uint16_t>(
      _cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

/**
 * \brief Marlin kFE2M1f skip_flop=false dequant of one packed word (8 N).
 *
 * Same bit trick as sm70_dsv41_mxfp4_moe_decode::dequant_fe2m1_word.
 */
inline void unpack_fe2m1(uint32_t packed, float out[8]) {
  auto dequant4 = [](int q, float* dst) {
    constexpr int MASK = 0x70007000;
    int Out1 = (q & 0x80008000) | ((q & MASK) >> 3);
    q <<= 4;
    int Out2 = (q & 0x80008000) | ((q & MASK) >> 3);
    const float bias = 16384.f;
    dst[0] = fp16_to_f32(static_cast<uint16_t>(Out2 & 0xffff)) * bias;
    dst[1] = fp16_to_f32(static_cast<uint16_t>((Out2 >> 16) & 0xffff)) * bias;
    dst[2] = fp16_to_f32(static_cast<uint16_t>(Out1 & 0xffff)) * bias;
    dst[3] = fp16_to_f32(static_cast<uint16_t>((Out1 >> 16) & 0xffff)) * bias;
  };
  dequant4(static_cast<int>(packed << 8), out);
  dequant4(static_cast<int>(packed), out + 4);
}

inline float e8m0_to_float(uint8_t byte) {
  int e = static_cast<int>(byte) - 112;
  if (e <= 0) {
    return 0.f;
  }
  if (e > 31) {
    e = 31;
  }
  const uint32_t bits = static_cast<uint32_t>(e + 112) << 23;
  float r;
  std::memcpy(&r, &bits, sizeof(r));
  return r;
}

template <int kSizeN, int kSizeK>
inline void mxfp4_gemv_range(const uint32_t* __restrict__ qw,
                             const uint8_t* __restrict__ sc,
                             const float* __restrict__ x,
                             float* __restrict__ acc,
                             int32_t n0,
                             int32_t n1) {
  static_assert(kSizeN % kPackedMacroN == 0, "N must be a packed-macro multiple");
  static_assert(kSizeK % 32 == 0, "K must be an MXFP4 group multiple");
  constexpr int32_t kTiles = kSizeK / kQuantTileK;
  constexpr int32_t kRow = kSizeN * 2;
  constexpr int32_t kGroupQ = kPackedMacroN / 8 * kQuantTileK;
  constexpr int32_t kNGroups = kSizeN / kPackedMacroN;
  const int32_t g0 = n0 / kPackedMacroN;
  const int32_t g1 = (n1 + kPackedMacroN - 1) / kPackedMacroN;
  alignas(64) float scale_tile[kPackedMacroN];
  for (int32_t g = g0; g < g1 && g < kNGroups; ++g) {
    const int32_t n_origin = g * kPackedMacroN;
    for (int32_t kg = 0; kg < kSizeK / kGroupSize; ++kg) {
      const uint8_t* sc_row = sc + static_cast<int64_t>(kg) * kSizeN + n_origin;
#pragma GCC unroll 8
      for (int32_t i = 0; i < kPackedMacroN; ++i) {
        scale_tile[i] = e8m0_to_float(sc_row[i]);
      }
      for (int32_t kt_off = 0; kt_off < 2; ++kt_off) {
        const int32_t kt = kg * 2 + kt_off;
        const uint32_t* p = qw + static_cast<int64_t>(kt) * kRow +
                            static_cast<int64_t>(g) * kGroupQ;
        for (int32_t local_k = 0; local_k < kQuantTileK; ++local_k) {
          const float xv = x[kt * kQuantTileK + local_k];
          const __m256 vx = _mm256_set1_ps(xv);
          for (int32_t n_vec = 0; n_vec < 8; ++n_vec) {
#pragma GCC unroll 4
            for (int32_t sub = 0; sub < kGroupTiles; ++sub) {
              const int32_t ni = sub * kQuantTileN + n_vec * 8;
              float w[8];
              unpack_fe2m1(*p++, w);
              const __m256 accv = _mm256_fmadd_ps(
                  _mm256_mul_ps(_mm256_loadu_ps(w), _mm256_loadu_ps(scale_tile + ni)),
                  vx,
                  _mm256_loadu_ps(acc + n_origin + ni));
              _mm256_storeu_ps(acc + n_origin + ni, accv);
            }
          }
        }
      }
    }
  }
  (void)kTiles;
}

inline void silu_mul(const float* __restrict__ gate_up, float* __restrict__ act) {
  for (int32_t i = 0; i < kIntermediate; ++i) {
    const float g = gate_up[i];
    const float u = gate_up[kIntermediate + i];
    act[i] = g / (1.f + std::exp(-g)) * u;
  }
}

struct SenseBarrier {
  std::atomic<int> count{0};
  std::atomic<int> sense{0};
  int n = 1;

  void wait() {
    const int s = sense.load(std::memory_order_relaxed);
    if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
      count.store(0, std::memory_order_relaxed);
      sense.store(s + 1, std::memory_order_release);
    } else {
      while (sense.load(std::memory_order_acquire) == s) {
        _mm_pause();
      }
    }
  }
};

struct HostGemvState {
  std::vector<std::thread> threads;
  std::atomic<bool> stop{false};
  std::atomic<uint32_t> job{0};
  SenseBarrier bar;
  int nthreads = 1;
  uint32_t last_seq = 0;
  alignas(64) float x_f[kMaxTok][kHidden];
  alignas(64) float w13_acc[kGateUp];
  alignas(64) float act[kIntermediate];
  alignas(64) float w2_acc[kHidden];
  alignas(64) float y_f[kMaxTok][kHidden];
};

inline HostGemvState& gemv_state() {
  static HostGemvState s;
  return s;
}

template <int kSizeN>
inline void zero_range(float* acc, int32_t n0, int32_t n1) {
  std::memset(acc + n0, 0, static_cast<size_t>(n1 - n0) * sizeof(float));
}

inline void tile_bounds(int32_t n_total, int32_t tid, int32_t nthreads, int32_t* n0, int32_t* n1) {
  const int32_t n_groups = n_total / kPackedMacroN;
  const int32_t g0 = (n_groups * tid) / nthreads;
  const int32_t g1 = (n_groups * (tid + 1)) / nthreads;
  *n0 = g0 * kPackedMacroN;
  *n1 = g1 * kPackedMacroN;
}

inline void run_one_hit(int tid, int nthreads, int32_t host_row, const float* x) {
  auto& st = gemv_state();
  const auto* m = dsv41_spill_mbox_host;
  const bool ok = host_row >= 0 && host_row < m->n_host;
  const uint32_t* w13 = nullptr;
  const uint8_t* s13 = nullptr;
  const uint32_t* w2 = nullptr;
  const uint8_t* s2 = nullptr;
  if (ok) {
    w13 = reinterpret_cast<const uint32_t*>(m->w13_ptr) +
          static_cast<int64_t>(host_row) * kW13KTiles * (kGateUp * 2);
    s13 = reinterpret_cast<const uint8_t*>(m->s13_ptr) +
          static_cast<int64_t>(host_row) * kW13Groups * kGateUp;
    w2 = reinterpret_cast<const uint32_t*>(m->w2_ptr) +
         static_cast<int64_t>(host_row) * kW2KTiles * (kHidden * 2);
    s2 = reinterpret_cast<const uint8_t*>(m->s2_ptr) +
         static_cast<int64_t>(host_row) * kW2Groups * kHidden;
  }
  int32_t n0 = 0;
  int32_t n1 = 0;
  tile_bounds(kGateUp, tid, nthreads, &n0, &n1);
  if (ok) {
    zero_range<kGateUp>(st.w13_acc, n0, n1);
    mxfp4_gemv_range<kGateUp, kHidden>(w13, s13, x, st.w13_acc, n0, n1);
  }
  st.bar.wait();
  if (tid == 0 && ok) {
    silu_mul(st.w13_acc, st.act);
  }
  st.bar.wait();
  tile_bounds(kHidden, tid, nthreads, &n0, &n1);
  if (ok) {
    zero_range<kHidden>(st.w2_acc, n0, n1);
    mxfp4_gemv_range<kHidden, kIntermediate>(w2, s2, st.act, st.w2_acc, n0, n1);
  }
}

inline void worker_loop(int tid) {
  auto& st = gemv_state();
  uint32_t last_job = 0;
  const int nthreads = st.nthreads;
  while (!st.stop.load(std::memory_order_relaxed)) {
    if (tid == 0) {
      while (dsv41_spill_mbox_host->seq == st.last_seq &&
             !st.stop.load(std::memory_order_relaxed)) {
        _mm_pause();
      }
      if (st.stop.load(std::memory_order_relaxed)) {
        st.job.fetch_add(1, std::memory_order_release);
        break;
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      st.last_seq = dsv41_spill_mbox_host->seq;
      if (dsv41_spill_mbox_host->n_hits <= 0) {
        std::memset(dsv41_spill_mbox_host->y, 0, sizeof(dsv41_spill_mbox_host->y));
        std::atomic_thread_fence(std::memory_order_release);
        dsv41_spill_mbox_host->done = st.last_seq;
        continue;
      }
      st.job.fetch_add(1, std::memory_order_release);
    } else {
      while (st.job.load(std::memory_order_acquire) == last_job &&
             !st.stop.load(std::memory_order_relaxed)) {
        _mm_pause();
      }
      if (st.stop.load(std::memory_order_relaxed)) {
        break;
      }
    }
    last_job = st.job.load(std::memory_order_relaxed);
    st.bar.wait();
    if (st.stop.load(std::memory_order_relaxed)) {
      break;
    }
    auto* m = dsv41_spill_mbox_host;
    const int32_t n_tok = m->n_tok < 0 ? 0 : (m->n_tok > kMaxTok ? kMaxTok : m->n_tok);
    const int32_t n_hits = m->n_hits < 0 ? 0 : (m->n_hits > kMaxHits ? kMaxHits : m->n_hits);
    if (tid == 0) {
      for (int32_t t = 0; t < n_tok; ++t) {
        for (int32_t i = 0; i < kHidden; ++i) {
          st.x_f[t][i] = fp16_to_f32(m->x[t][i]);
          st.y_f[t][i] = 0.f;
        }
      }
    }
    st.bar.wait();
    for (int32_t h = 0; h < n_hits; ++h) {
      const int32_t tok = m->tok_of[h];
      if (tok < 0 || tok >= n_tok) {
        st.bar.wait();
        st.bar.wait();
        st.bar.wait();
        continue;
      }
      run_one_hit(tid, nthreads, m->host_rows[h], st.x_f[tok]);
      st.bar.wait();
      if (tid == 0 && m->host_rows[h] >= 0 && m->host_rows[h] < m->n_host) {
        const float w = m->weights[h];
        for (int32_t i = 0; i < kHidden; ++i) {
          st.y_f[tok][i] += st.w2_acc[i] * w;
        }
      }
    }
    st.bar.wait();
    if (tid == 0) {
      for (int32_t t = 0; t < kMaxTok; ++t) {
        if (t < n_tok) {
          for (int32_t i = 0; i < kHidden; ++i) {
            m->y[t][i] = f32_to_fp16(st.y_f[t][i]);
          }
        } else {
          std::memset(m->y[t], 0, sizeof(m->y[t]));
        }
      }
      m->error = 0;
      std::atomic_thread_fence(std::memory_order_release);
      m->done = st.last_seq;
    }
  }
}

void dsv41_mxfp4_moe_expert_serial(const uint16_t* x,
                                   const uint32_t* w13,
                                   const uint8_t* s13,
                                   const uint32_t* w2,
                                   const uint8_t* s2,
                                   float weight,
                                   uint16_t* y) {
  alignas(64) float xf[kHidden];
  alignas(64) float acc13[kGateUp];
  alignas(64) float act[kIntermediate];
  alignas(64) float acc2[kHidden];
  for (int32_t i = 0; i < kHidden; ++i) {
    xf[i] = fp16_to_f32(x[i]);
  }
  std::memset(acc13, 0, sizeof(acc13));
  mxfp4_gemv_range<kGateUp, kHidden>(w13, s13, xf, acc13, 0, kGateUp);
  silu_mul(acc13, act);
  std::memset(acc2, 0, sizeof(acc2));
  mxfp4_gemv_range<kHidden, kIntermediate>(w2, s2, act, acc2, 0, kHidden);
  for (int32_t i = 0; i < kHidden; ++i) {
    y[i] = f32_to_fp16(acc2[i] * weight);
  }
}

void dsv41_host_workers_start(int nthreads) {
  auto& st = gemv_state();
  if (!st.threads.empty()) {
    return;
  }
  int n = nthreads;
  if (n < 1) {
    n = 1;
  }
  if (n > kMaxHostThreads) {
    n = kMaxHostThreads;
  }
  st.nthreads = n;
  st.bar.n = n;
  st.stop.store(false, std::memory_order_relaxed);
  st.job.store(0, std::memory_order_relaxed);
  st.last_seq = 0;
  st.threads.reserve(static_cast<size_t>(n));
  for (int tid = 0; tid < n; ++tid) {
    st.threads.emplace_back(worker_loop, tid);
  }
}

void dsv41_host_workers_stop() {
  auto& st = gemv_state();
  st.stop.store(true, std::memory_order_release);
  st.job.fetch_add(1, std::memory_order_release);
  for (auto& t : st.threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  st.threads.clear();
}

}  // namespace sglang::sm70_dsv41
