/// \file atomic.cuh
/// \brief Device-side atomic operations.

#pragma once
#include <sgl_kernel/utils.cuh>

#include <type_traits>

namespace sglang {

namespace device::atomic {

/**
 * \brief Atomically computes the maximum of `*addr` and `value`, storing the
 *        result in `*addr`.
 * \param addr Pointer to the value in global/shared memory to be updated.
 * \param value The value to compare against.
 * \return The old value at `*addr` before the update.
 * \note On CUDA, this uses `atomicMax`/`atomicMin` on the reinterpreted
 *       integer representation. On ROCm, a CAS loop is used as a fallback.
 */
SGL_DEVICE float max(float* addr, float value) {
#ifndef USE_ROCM
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                     : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
#else
  int* addr_as_i = (int*)addr;
  int old = *addr_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
#endif
}

namespace ptx {

SGL_DEVICE void red_release_add_u32(uint32_t* ptr, uint32_t n) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(n) : "memory");
}

/**
 * \brief `red_release_add_u32` on the uniform datapath (PTX ISA 8.7, sm_100+);
 * falls back to the plain form where the target does not have it.
 *
 * Typically, one elected lane of a converged warp must issue this, via
 * `device::warp::elect_one_lane` (or read from `%lane_id`).
 * \note If the ptxas cannot prove the instruction issue is from single thread,
 * the generated SASS may have very poor performance.
 */
SGL_DEVICE void red_async_release_add_u32(uint32_t* ptr, uint32_t n) {
#if defined(SGL_CUDA_ARCH) && SGL_CUDA_ARCH >= 1000
  asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(n) : "memory");
#else
  red_release_add_u32(ptr, n);
#endif
}

SGL_DEVICE void red_relaxed_add_u32(uint32_t* ptr, uint32_t n) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(n) : "memory");
}

SGL_DEVICE uint32_t atom_acquire_cas_b32(uint32_t* addr, uint32_t compare, uint32_t swap) {
  uint32_t result;
  asm volatile("atom.acquire.gpu.global.cas.b32 %0, [%1], %2, %3;"
               : "=r"(result)
               : "l"(addr), "r"(compare), "r"(swap)
               : "memory");
  return result;
}

SGL_DEVICE uint32_t load_acquire_u32(uint32_t* addr) {
  uint32_t result;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(result) : "l"(addr) : "memory");
  return result;
}

SGL_DEVICE uint32_t atom_acquire_add_u32(uint32_t* addr, uint32_t n) {
  uint32_t result;
  asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(result) : "l"(addr), "r"(n) : "memory");
  return result;
}

}  // namespace ptx

/**
 * \brief Cross-CTA arrive/wait counter packed into one 32-bit word.
 *
 * Producers call `arrive()`; consumers call `wait_unique()` (exactly one consumer)
 * or `wait()` (several) until every producer has. The word is split: the low
 * `32 - kConsumerBits` bits count producer arrivals, the high bits count
 * consumers that have already been released. The last consumer to be released
 * subtracts the whole thing, so one Event is reusable across launches without a
 * host-side re-zero.
 *
 * \note The handle must be ZERO before first use. Nothing constructs it on the
 *       device, so zero the backing allocation from the host once.
 * \note The storage must be GLOBAL memory: the PTX below names the `.global`
 *       state space, so a shared or local Event is an illegal address.
 * \note `arrive()` is a release and the waits are acquires, so a producer's
 *       writes before `arrive()` are visible to a consumer after the wait.
 * \note Generations must be explicitly ordered: every consumer of generation N
 *       has to be released before any producer arrives for generation N + 1.
 *       The word carries no phase bit, so overlapping two generations on one
 *       Event is undefined behavior.
 */
struct Event {
 public:
  using handle_type = uint32_t;

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  struct DefaultSpin {
    SGL_DEVICE void operator()() const {}
  };

  /// \brief DON'T touch unless you know what you're doing.
  SGL_DEVICE handle_type& unsafe_get_handle() {
    return m_handle;
  }

  /**
   * \brief Increment the producer count by `n`.
   * \param n The number of producers to arrive. Defaults to 1.
   * \note This is a release operation, so any writes before `arrive()` are
   *       visible to a consumer after `wait()`.
   */
  SGL_DEVICE void arrive(uint32_t n = 1) {
    ptx::red_release_add_u32(&m_handle, n);
  }

  /**
   * \brief `arrive()` on the uniform datapath: two instructions instead of four.
   * \param n The number of producers to arrive. Defaults to 1.
   */
  SGL_DEVICE void arrive_async(uint32_t n = 1) {
    ptx::red_async_release_add_u32(&m_handle, n);
  }

  /**
   * \brief Block until `num_producers` producers have arrived.
   * \param num_producers The number of producers to wait for.
   * \param spin A callable invoked each time the wait spins, or the nanoseconds
   *             to sleep per poll.
   *
   * Single-consumer: simpler and faster than `wait()`, but exactly one thread in
   * the whole grid may call it per generation.
   *
   * \note Every poll is an atomic RMW, which serializes on the L2 slice holding
   *       the handle. That is cheap at a low poll rate; pass a `spin` that backs
   *       off where the producers are the bottleneck.
   */
  template <typename Spin = DefaultSpin>
  SGL_DEVICE void wait_unique(uint32_t num_producers, Spin spin = {}) {
    while (ptx::atom_acquire_cas_b32(&m_handle, num_producers, 0) != num_producers)
      s_poll(spin);
  }

  /**
   * \brief Block until `num_producers` producers have arrived, with several
   *        consumers sharing the Event.
   * \tparam kConsumerBits Bits reserved for the consumer half of the word.
   * \param num_producers  Must be `< 1 << (32 - kConsumerBits)`.
   * \param num_consumers  Must be in `[1, 1 << kConsumerBits)`, and the `n` of
   *                       all callers has to sum to exactly this, otherwise the
   *                       Event is never reset.
   * \param n              How many of `num_consumers` this call stands for.
   *                       Defaults to 1, i.e. one calling thread per consumer.
   * \param spin           A callable invoked each time the wait spins, or the
   *                       nanoseconds to sleep per poll.
   */
  template <uint32_t kConsumerBits = 16u, typename Spin = DefaultSpin>
  SGL_DEVICE void wait(uint32_t num_producers, uint32_t num_consumers, uint32_t n = 1, Spin spin = {}) {
    static_assert(kConsumerBits > 0 && kConsumerBits < 32);
    constexpr uint32_t kProducerBits = 32 - kConsumerBits;
    constexpr uint32_t kProducerMask = (1u << kProducerBits) - 1;

    __builtin_assume(num_producers < (1u << kProducerBits));
    __builtin_assume(num_consumers > 0 && num_consumers < (1u << kConsumerBits));

    // Register and observe in the SAME atomic. ticket = consumers ahead of me.
    const auto ticket = ptx::atom_acquire_add_u32(&m_handle, n << kProducerBits);
    if ((ticket & kProducerMask) != num_producers) {
      /// NOTE: when v = 0, a reset has already happened.
      while (const auto v = ptx::load_acquire_u32(&m_handle)) {
        if ((v & kProducerMask) == num_producers) break;
        s_poll(spin);
      }
    }

    // The last consumer to register should reset the counter to 0
    if ((ticket >> kProducerBits) + n == num_consumers) {
      const auto final_value = num_producers | (num_consumers << kProducerBits);
      ptx::red_relaxed_add_u32(&m_handle, -final_value);
    }
  }

 private:
  template <typename Spin>
  SGL_DEVICE static void s_poll(Spin spin) {
    if constexpr (std::is_integral_v<Spin>) {
      static_assert(std::is_unsigned_v<Spin> && sizeof(Spin) <= 4);
      __nanosleep(spin);
    } else {
      spin();
    }
  }

  handle_type m_handle;
};

}  // namespace device::atomic

}  // namespace sglang
