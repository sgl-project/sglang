/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Implementation of a CTA-wide barrier for inter-CTA synchronization.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

namespace detail {

//
// Utilities for abstracting synchronization methods for barriers
//

struct SyncthreadsSync {
  CUTLASS_DEVICE
  static void sync() {
    __syncthreads();
  }
};

struct SyncwarpSync {
  CUTLASS_DEVICE
  static void sync() {
    __syncwarp();
  }
};

template <
  int ThreadCount,
  int BarrierId
>
struct NamedBarrierSync {
  CUTLASS_DEVICE
  static void sync() {
    cutlass::arch::NamedBarrier::sync(ThreadCount, static_cast<arch::ReservedNamedBarriers>(BarrierId));
  }
};

} // namepspace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Group or CTA-wide semaphore for inter-CTA synchronization.
template <class Sync>
struct GenericBarrier {

public:

  /// Flag type
  using T = int;

  /// Initial flag value
  static const T INIT = 0;


protected:

  /// Load flag, as a strong acquire operation (int specialization)
  CUTLASS_DEVICE
  static int ld_acquire(int *ptr)
  {
    int state = 0;

#if (__CUDA_ARCH__ >= 700)
    /// SM70 and newer use memory consistency qualifiers

    // Acquire pattern using acquire modifier
    asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));

#else
    asm volatile ("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif // (__CUDA_ARCH__ >= 700)

    return state;
  }


  /// Reduce into flag, with release pattern (int specialization)
  CUTLASS_DEVICE
  static void red_release(int *ptr, int val)
  {
#if (__CUDA_ARCH__ >= 700)
    /// SM70 and newer use memory consistency qualifiers

    // Release pattern using acq_rel fence + relaxed modifier.  (The fence also releases data
    // that was weakly-written by other threads prior to the last syncthreads)
    asm volatile ("fence.acq_rel.gpu;\n");
    asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(ptr), "r"(val));

#else
    __threadfence();
    atomicAdd(ptr, val);
#endif // (__CUDA_ARCH__ >= 700)
  }


public:

  /// Uses thread[0] to wait for at least the specified count of signals on the given flag counter
  CUTLASS_DEVICE
  static void wait_lt(void *lock_ptr, int thread_idx, int flag_idx, int count)
  {
    T *flag_ptr = reinterpret_cast<T*>(lock_ptr) + flag_idx;

    if (thread_idx == 0)
    {
        // Spin-loop
        #pragma unroll 1
        while(ld_acquire(flag_ptr) < count) {}
    }

    Sync::sync();
  }

  /// Uses thread[0] to wait for at least the specified count of signals on the given flag counter
  CUTLASS_DEVICE
  static void wait_eq(void *lock_ptr, int thread_idx, int flag_idx, T val = 1)
  {
    T *flag_ptr = reinterpret_cast<T*>(lock_ptr) + flag_idx;

    if (thread_idx == 0)
    {
        // Spin-loop
        #pragma unroll 1
        while(ld_acquire(flag_ptr) != val) {}
    }
    Sync::sync();
  }

  /// Uses thread[0] to wait for the specified count of signals on the given flag counter
  CUTLASS_DEVICE
  static void wait_eq_reset(void *lock_ptr, int thread_idx, int flag_idx, T val = 1) {
    T *flag_ptr = reinterpret_cast<T*>(lock_ptr) + flag_idx;

    if (thread_idx == 0)
    {
        // Spin-loop
        #pragma unroll 1
        while(atomicCAS(flag_ptr, val, 0) != val) {}
    }

    Sync::sync();
  }

  /// Increment the arrival count for a flag
  CUTLASS_DEVICE
  static void arrive_inc(void *lock_ptr, int thread_idx, int flag_idx, int val = 1)
  {
    T* flag_ptr = reinterpret_cast<T*>(lock_ptr) + flag_idx;

    Sync::sync();

    if (thread_idx == 0)
    {
      red_release(flag_ptr, val);
    }
  }


  /// Increment the arrival counts for a range of flags
  CUTLASS_DEVICE
  static void arrive_range_inc(void *lock_ptr, int thread_idx, int first_flag_idx, int count = 1, int val = 1)
  {
    int flag_idx = first_flag_idx + thread_idx;
    T* flag_ptr = reinterpret_cast<T*>(lock_ptr) + flag_idx;

    // Barrier to make sure all other threads in group have written their data
    Sync::sync();

    // Select threads increment their flags
    if (thread_idx < count) {
      red_release(flag_ptr, val);
    }
  }
};

using Barrier = GenericBarrier<detail::SyncthreadsSync>;

/////////////////////////////////////////////////////////////////////////////////////////////////

/** Structure for managing multiple NamedBarriers to be used by different warp groups, allowing
 * runtime index values to be used to call into named barriers with compile-time-constant IDs.
 *
 * @param ThreadCount_ Number of threads that will wait on a NamedBarrier with a given ID
 * @param Offset Value added to the ID passed in by the user to determine the NamedBarrier ID to call into
 * @param MaxNumNamedBarriers The maximum number of unique barrier IDs that will be requested on this type
**/
template <
  uint32_t ThreadCount_,
  uint32_t Offset = 0,
  uint32_t MaxNumNamedBarriers = 16
>
struct NamedBarrierManager {

  static_assert(MaxNumNamedBarriers <= arch::NamedBarrier::HardwareMaxNumNamedBarriers);
  static_assert(MaxNumNamedBarriers + Offset <= arch::NamedBarrier::HardwareMaxNumNamedBarriers, "Barrier IDs cannot exceed 15");

  // Number of threads participating in the barrier
  static constexpr uint32_t ThreadCount = ThreadCount_;

  template <uint32_t BarrierId>
  using BarrierSync = cutlass::GenericBarrier<cutlass::detail::NamedBarrierSync<ThreadCount, BarrierId>>;

  // Underlying type used by all barriers for synchronization. Does not depend on
  // template parameter BarrierId, so passing in 0 suffices.
  using T = typename BarrierSync<0>::T;

  using IntegerSequence = cute::make_integer_sequence<uint32_t, MaxNumNamedBarriers>;

  CUTLASS_DEVICE
  static
  void wait_lt(uint32_t idx, void *lock_ptr, int thread_idx, int flag_idx, int count) {
    wait_lt_helper(idx, lock_ptr, thread_idx, flag_idx, count, IntegerSequence{});
  }

  CUTLASS_DEVICE
  static void
  wait_eq(uint32_t idx, void *lock_ptr, int thread_idx, int flag_idx, T val = 1) {
    wait_eq_helper<false>(idx, lock_ptr, thread_idx, flag_idx, val, IntegerSequence{});
  }

  CUTLASS_DEVICE
  static void
  wait_eq_reset(uint32_t idx, void *lock_ptr, int thread_idx, int flag_idx, T val = 1) {
    wait_eq_helper<true>(idx, lock_ptr, thread_idx, flag_idx, val, IntegerSequence{});
  }

  CUTLASS_DEVICE
  static void
  arrive_inc(uint32_t idx, void *lock_ptr, int thread_idx, int flag_idx, int val = 1) {
    arrive_inc_helper(idx, lock_ptr, thread_idx, flag_idx, val, IntegerSequence{});
  }

  CUTLASS_DEVICE
  static void
  arrive_range_inc(uint32_t idx, void *lock_ptr, int thread_idx, int first_flag_idx, int count = 1, int val = 1) {
    arrive_range_inc_helper(idx, lock_ptr, thread_idx, first_flag_idx, count, val, IntegerSequence{});
  }

private:
  CUTLASS_DEVICE
  static void
  check_barrier_in_range([[maybe_unused]] uint32_t idx) {
    assert((idx < MaxNumNamedBarriers) && "Index exceeds barrier count");
  }

  template <uint32_t... Idx>
  CUTLASS_DEVICE
  static void
  wait_lt_helper(uint32_t idx, void *lock_ptr, int thread_idx, int flag_idx, int count, cute::integer_sequence<uint32_t, Idx...>) {
    check_barrier_in_range(idx);
    ((Idx == idx && (BarrierSync<Idx + Offset>::wait_lt(lock_ptr, thread_idx, flag_idx, count), true)) || ...);
  }

  template <bool Reset, uint32_t... Idx>
  CUTLASS_DEVICE
  static void
  wait_eq_helper(uint32_t idx, void *lock_ptr, int thread_idx, int flag_idx, T val, cute::integer_sequence<uint32_t, Idx...>) {
    check_barrier_in_range(idx);
    if constexpr (Reset) {
      ((Idx == idx && (BarrierSync<Idx + Offset>::wait_eq_reset(lock_ptr, thread_idx, flag_idx, val), true)) || ...);
    }
    else {
      ((Idx == idx && (BarrierSync<Idx + Offset>::wait_eq(lock_ptr, thread_idx, flag_idx, val), true)) || ...);
    }
  }

  template <uint32_t... Idx>
  CUTLASS_DEVICE
  static void
  arrive_inc_helper(uint32_t idx, void *lock_ptr, int thread_idx, int flag_idx, int val, cute::integer_sequence<uint32_t, Idx...>) {
    check_barrier_in_range(idx);
    ((Idx == idx && (BarrierSync<Idx + Offset>::arrive_inc(lock_ptr, thread_idx, flag_idx, val), true)) || ...);
  }

  template <uint32_t... Idx>
  CUTLASS_DEVICE
  static void
  arrive_range_inc_helper(uint32_t idx, void *lock_ptr, int thread_idx, int first_flag_idx, int count, int val, cute::integer_sequence<uint32_t, Idx...>) {
    check_barrier_in_range(idx);
    ((Idx == idx && (BarrierSync<Idx + Offset>::arrive_range_inc(lock_ptr, thread_idx, first_flag_idx, count, val), true)) || ...);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/** Structure for synchronizing via contiguous barriers (e.g., __syncwarp, __syncthreads)
 *  via an API that mirrors that of NamedBarrierManager
 *
 * @param Synchronizer Synchronization helper exposing a `sync()` method to perform synchronization
**/
template <
  class Synchronizer,
  uint32_t ThreadCount_
>
struct SyncManager {

  // Number of threads participating in the barrier
  static constexpr uint32_t ThreadCount = ThreadCount_;

  using BarrierSync = cutlass::GenericBarrier<Synchronizer>;

  // Underlying type used by all barriers for synchronization.
  using T = typename BarrierSync::T;

  CUTLASS_DEVICE
  static
  void wait_lt(uint32_t, void *lock_ptr, int thread_idx, int flag_idx, int count) {
    BarrierSync::wait_lt(lock_ptr, thread_idx, flag_idx, count);
  }

  CUTLASS_DEVICE
  static void
  wait_eq(uint32_t, void *lock_ptr, int thread_idx, int flag_idx, T val = 1) {
    BarrierSync::wait_eq(lock_ptr, thread_idx, flag_idx, val);
  }

  CUTLASS_DEVICE
  static void
  wait_eq_reset(uint32_t, void *lock_ptr, int thread_idx, int flag_idx, T val = 1) {
    BarrierSync::wait_eq_reset(lock_ptr, thread_idx, flag_idx, val);
  }

  CUTLASS_DEVICE
  static void
  arrive_inc(uint32_t, void *lock_ptr, int thread_idx, int flag_idx, int val = 1) {
    BarrierSync::arrive_inc(lock_ptr, thread_idx, flag_idx, val);
  }

  CUTLASS_DEVICE
  static void
  arrive_range_inc(uint32_t idx, void *lock_ptr, int thread_idx, int first_flag_idx, int count = 1, int val = 1) {
    BarrierSync::arrive_range_inc(lock_ptr, thread_idx, first_flag_idx, count, val);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
