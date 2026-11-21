#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/fusion.cuh>
#include <sgl_kernel/ptx/barrier.cuh>

#include <bit>
#include <cstdint>
#include <type_traits>

namespace sglang::device::distributed::push {

namespace ptx {

using namespace sglang::device::ptx;

/// Stores `value` (any trivially copyable 4-byte type) to the peer CTA's shared memory at `dst` and completes 4 bytes
/// of transaction on the peer's mbarrier `bar`; both are `shared::cluster` addresses from `mapa_shared`.
template <typename T>
SGL_DEVICE void st_async_4B(uint32_t dst, const T& value, uint32_t bar) {
  static_assert(sizeof(T) == 4 && std::is_trivially_copyable_v<T>);
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(dst),
               "r"(std::bit_cast<uint32_t>(value)),
               "r"(bar)
               : "memory");
}

}  // namespace ptx

/**
 * \brief Up to `kNumSums` sums of one float per thread over a row held by a cluster of `kClusterSize` CTAs (ranked by
 *        `%cluster_ctarank`) of `kNumWarps` warps: warp butterfly, then the warps in order through shared memory,
 *        then the CTAs in rank order, each CTA's sum sent to every peer with one 4-byte `st.async` onto the peer's
 *        mbarrier. Every thread of the row gets the same bits. Sum `index` has its own storage and mbarrier and is
 *        used at most once per kernel.
 *
 * The object itself lives in shared memory (`__shared__ ClusterRowSum smem;`). Protocol: `init` at kernel start (every
 * thread); `wait_peers` once before the first `sum`, so that every peer's mbarriers are initialized before they are
 * written. No exit barrier: each CTA waits for every byte sent to it before it uses a sum.
 */
template <uint32_t kClusterSize, uint32_t kNumWarps, uint32_t kNumSums>
struct ClusterRowSum {
 public:
  static_assert(kClusterSize <= kNumWarps * kWarpThreads, "one sending thread per peer CTA");
  static_assert(kNumSums >= 1);
  using Vec = VecStorage<bf16_t>;

  SGL_DEVICE void init() {
    if constexpr (kClusterSize > 1) {
      if (threadIdx.x == 0) {
#pragma unroll
        for (uint32_t i = 0; i < kNumSums; ++i) {
          ptx::mbar_init(&m_full[i], 1);
          ptx::mbar_arrive_expect_tx(&m_full[i], kClusterSize * sizeof(float));
        }
        ptx::fence_barrier_init();
      }
      ptx::bar_cluster_arrive_relaxed();
    }
  }

  SGL_DEVICE void wait_peers() const {
    if constexpr (kClusterSize > 1) ptx::bar_cluster_wait_acquire();
  }

  /// `index` (a constant < kNumSums) picks the sum's storage; each index is summed at most once per kernel.
  SGL_DEVICE float sum(float value, uint32_t index = 0) {
    value = warp::reduce_sum(value);
    // lane 0 only: every lane storing the same value is equivalent but costs 12 registers (ptxas 13.4, DeepSeek V4)
    if (threadIdx.x % kWarpThreads == 0) m_warp_sums[index][threadIdx.x / kWarpThreads] = value;
    __syncthreads();
    if constexpr (kClusterSize == 1) {
      float cta_sum = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < kNumWarps; ++w) {
        cta_sum += m_warp_sums[index][w];
      }
      return cta_sum;
    } else {
      if (threadIdx.x < kClusterSize) {
        float cta_sum = 0.0f;
#pragma unroll
        for (uint32_t w = 0; w < kNumWarps; ++w) {
          cta_sum += m_warp_sums[index][w];
        }
        const auto dst = ptx::mapa_shared(&m_cta_sums[index][ptx::cluster_ctarank()], threadIdx.x);
        const auto bar = ptx::mapa_shared(&m_full[index], threadIdx.x);
        ptx::st_async_4B(dst, cta_sum, bar);
      }
      ptx::mbar_wait_parity(&m_full[index], 0);
      float total = 0.0f;
#pragma unroll
      for (uint32_t c = 0; c < kClusterSize; ++c) {
        total += m_cta_sums[index][c];
      }
      return total;
    }
  }

  /// `bf16(x * weight * rsqrt(mean(x^2) + eps))` with `scale` = 1 / hidden, on sum `index` (see `sum`).
  SGL_DEVICE Vec rms_norm(const Vec& x, const Vec& weight, float eps, float scale, uint32_t index = 0) {
    float acc0 = 0.0f, acc1 = 0.0f;
    bf16_square_sum(x, acc0, acc1);
    const auto r = rms_factor(sum(acc0 + acc1, index), scale, eps);
    return bf16_norm_scale(x, weight, r);
  }

 private:
  alignas(16) float m_warp_sums[kNumSums][kNumWarps];
  alignas(16) float m_cta_sums[kNumSums][kClusterSize];
  uint64_t m_full[kNumSums];  // kClusterSize > 1: the peers' CTA sums have landed
};

}  // namespace sglang::device::distributed::push
