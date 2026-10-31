#pragma once
#include <sgl_kernel/math.cuh>

#include <bit>
#include <cstdint>

namespace sglang {

namespace device {

// Batcher odd-even merge sort over L keys, padded to a power of two; compare-
// exchanges touching the padding are no-ops and dropped. Built at compile time:
// nvcc does not fully unroll the nested loop form for 16 inputs (stack + loop).
template <uint32_t L>
struct MergeSortNetwork {
 private:
  // Visits every compare-exchange (a, b), a < b, in network order.
  template <typename F>
  static constexpr void for_each_pair(F&& f) {
    const auto n = std::bit_ceil(L);
    for (uint32_t p = 1; p < n; p <<= 1) {
      for (uint32_t k = p; k >= 1; k >>= 1) {
        for (uint32_t j = k % p; j + k < n; j += 2 * k) {
          for (uint32_t i = 0; i < k; ++i) {
            const auto a = i + j, b = i + j + k;
            if (b < L && a / (2 * p) == b / (2 * p)) f(a, b);
          }
        }
      }
    }
  }

  static constexpr uint32_t count_pairs() {
    uint32_t count = 0;
    for_each_pair([&](uint32_t, uint32_t) { ++count; });
    return count;
  }

 public:
  static_assert(L >= 1);
  static constexpr uint32_t kSize = count_pairs();
  // One spare slot keeps the arrays non-empty for L == 1.
  uint32_t x[kSize + 1]{};
  uint32_t y[kSize + 1]{};

  constexpr MergeSortNetwork() {
    uint32_t c = 0;
    for_each_pair([&](uint32_t a, uint32_t b) {
      x[c] = a;
      y[c] = b;
      ++c;
    });
  }
};

// Sorts a[0, N) in descending order.
template <uint32_t N, typename T>
SGL_DEVICE void merge_sort(T* a) {
  constexpr MergeSortNetwork<N> kNet{};
#pragma unroll
  for (uint32_t c = 0; c < kNet.kSize; ++c) {
    const auto hi = math::max(a[kNet.x[c]], a[kNet.y[c]]);
    const auto lo = math::min(a[kNet.x[c]], a[kNet.y[c]]);
    a[kNet.x[c]] = hi;
    a[kNet.y[c]] = lo;
  }
}

template <uint32_t N, typename T>
SGL_DEVICE void merge_sort(T (&a)[N]) {
  return merge_sort<N>(static_cast<T*>(a));
}

}  // namespace device

}  // namespace sglang
