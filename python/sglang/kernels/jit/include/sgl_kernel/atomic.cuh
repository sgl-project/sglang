/// \file atomic.cuh
/// \brief Device-side atomic operations.

#pragma once
#include <sgl_kernel/utils.cuh>

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

#ifndef USE_ROCM

namespace ptx {

SGL_DEVICE void red_release_add_u32(uint32_t* ptr, uint32_t n) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(n) : "memory");
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
 * Producers call `arrive()`; consumers call `wait()` until every producer has.
 * The word is split: the low `32 - kConsumerBits` bits count producer arrivals,
 * the high bits count consumers that have already been released. The last
 * consumer to be released subtracts the whole thing, so one Event is reusable
 * across launches without a host-side re-zero.
 *
 * \note The handle must be ZERO before first use. Nothing constructs it on the
 *       device, so zero the backing allocation from the host once.
 * \note `arrive()` is a release and `wait()` is an acquire, so a producer's
 *       writes before `arrive()` are visible to a consumer after `wait()`.
 */
struct Event {
 public:
  using handle_type = uint32_t;

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

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
   * \brief Block until `num_producers` producers have arrived.
   * \param num_producers The number of producers to wait for.
   *
   * This is a single-consumer version of `wait()`. It is simpler and faster,
   * but it cannot be used with multiple consumers.
   */
  SGL_DEVICE void wait(uint32_t num_producers) {
    while (ptx::atom_acquire_cas_b32(&m_handle, num_producers, 0) != num_producers)
      ;
  }

  /**
   * \brief Block until `num_producers` producers have arrived.
   * \tparam kConsumerBits Bits reserved for the consumer half of the word.
   * \param num_producers  Must be `< 1 << (32 - kConsumerBits).`
   * \param num_consumers  Must be in `[1, 1 << kConsumerBits)`, and every one of them
   *                       has to call this, otherwise the Event will be never reset.
   * \param n              The number of producers to arrive. Defaults to 1.
   */
  template <uint32_t kConsumerBits = 16u>
  SGL_DEVICE void wait(uint32_t num_producers, uint32_t num_consumers, uint32_t n = 1) {
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
      }
    }

    // The last consumer to register should reset the counter to 0
    if ((ticket >> kProducerBits) == num_consumers - 1) {
      const auto final_value = num_producers | (num_consumers << kProducerBits);
      ptx::red_relaxed_add_u32(&m_handle, -final_value);
    }
  }

 private:
  handle_type m_handle;
};

#endif

}  // namespace device::atomic

}  // namespace sglang
