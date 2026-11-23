// Ring-buddy replica pick for Muse Spark EP; the fused kernel, the Triton pick and the
// test reference must agree exactly. k[h] = active experts homed on rank h, y[h] = how
// many use the buddy copy on rank h - 1: pick the smallest T = max_r (k[r] - y[r] +
// y[r + 1]), then the smallest y[0], then the minimal chain y[r] = max(k[r] + y[r + 1] - T, 0);
// the y[h] highest-index active experts of home h move to the buddy copy.

#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define MUSE_RING_HD __host__ __device__ __forceinline__
#else
#define MUSE_RING_HD inline
#endif

namespace muse_ring {

MUSE_RING_HD int popc32(uint32_t x) {
#if defined(__CUDA_ARCH__)
  return __popc(x);
#else
  return __builtin_popcount(x);
#endif
}

MUSE_RING_HD int imax(int a, int b) {
  return a > b ? a : b;
}

MUSE_RING_HD int imin(int a, int b) {
  return a < b ? a : b;
}

template <int kRanks_, int kChunksPerHome_>
struct RingPick {
  static constexpr int kRanks = kRanks_;
  static constexpr int kChunksPerHome = kChunksPerHome_;
  static constexpr int kHome = 32 * kChunksPerHome_;
  static constexpr int kChunks = kRanks_ * kChunksPerHome_;
  static constexpr int kLogical = kRanks_ * kHome;
  static constexpr int kKeyBase = kHome + 1;
  static constexpr int kNoKey = 0x7fffffff;

  static_assert(kRanks_ >= 2, "ring-buddy needs >= 2 EP ranks");
  static_assert(kChunksPerHome_ >= 1, "home group must be a multiple of 32 experts");

  MUSE_RING_HD static void counts(const uint32_t (&m)[kChunks], int (&k)[kRanks]) {
#pragma unroll
    for (int h = 0; h < kRanks; ++h) {
      int s = 0;
#pragma unroll
      for (int i = 0; i < kChunksPerHome; ++i)
        s += popc32(m[h * kChunksPerHome + i]);
      k[h] = s;
    }
  }

  MUSE_RING_HD static int candidate_key(const int (&k)[kRanks], int total, int t) {
    if (t < 0 || t > kHome || total > kRanks * t) return kNoKey;
    int b = 0;
#pragma unroll
    for (int r = kRanks - 2; r >= 1; --r)
      b = imax(b + k[r] - t, 0);
    // The feasible y0 form an interval, so testing its least element lo decides t.
    const int lo = imax(k[0] - t + b, 0);
    if (lo > k[0]) return kNoKey;
    int prev = lo;
    bool ok = true;
#pragma unroll
    for (int r = kRanks - 1; r >= 1; --r) {
      ok = ok && prev <= t;
      prev = imax(k[r] + prev - t, 0);
    }
    ok = ok && (k[0] + prev - t <= lo);
    return ok ? t * kKeyBase + lo : kNoKey;
  }

  MUSE_RING_HD static void flows(const int (&k)[kRanks], int t, int y0, int (&y)[kRanks]) {
    y[0] = y0;
    int prev = y0;
#pragma unroll
    for (int r = kRanks - 1; r >= 1; --r) {
      prev = imax(k[r] + prev - t, 0);
      y[r] = prev;
    }
  }

  // e must be active.
  MUSE_RING_HD static int
  physical(const uint32_t (&m)[kChunks], const int (&k)[kRanks], const int (&y)[kRanks], int e) {
    const int c = e >> 5;
    const int bit = e & 31;
    const int h = c / kChunksPerHome;
    const int j = e - h * kHome;
    int excl = 0;
#pragma unroll
    for (int cc = 0; cc < kChunks; ++cc) {
      const uint32_t below = (cc < c) ? 0xffffffffu : ((cc == c) ? ((1u << bit) - 1u) : 0u);
      if (cc / kChunksPerHome == h) excl += popc32(m[cc] & below);
    }
    int kh = 0;
    int yh = 0;
#pragma unroll
    for (int g = 0; g < kRanks; ++g) {
      if (g == h) {
        kh = k[g];
        yh = y[g];
      }
    }
    const bool moved = excl >= kh - yh;
    const int holder = moved ? (h + kRanks - 1) % kRanks : h;
    return holder * (2 * kHome) + (moved ? kHome + j : j);
  }
};

}  // namespace muse_ring
