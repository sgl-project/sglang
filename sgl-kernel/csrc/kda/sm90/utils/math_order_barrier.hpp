// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Copyright (c) 2025 by FlashInfer team.
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

#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>

namespace kda::sm90 {

// cutlass' OrderedSequenceBarrier uses mbarrier
template <
    bool UseReservedNB_,           // treat nb_id as cutlass::ReservedNamedBarriers
    uint32_t... WGIdToNBIdMapping  // say 6,4 is passed, means wg0 use nb6 and wg1 use nb4
    >
struct OrderedNamedBarriers {
  static constexpr bool UseReservedNB = UseReservedNB_;
  static constexpr int NumWG = sizeof...(WGIdToNBIdMapping);
  using NBId_t = std::conditional_t<UseReservedNB, cutlass::arch::ReservedNamedBarriers, uint32_t>;

  CUTE_DEVICE
  OrderedNamedBarriers() : mapping_{NBId_t(WGIdToNBIdMapping)...} {}

  CUTE_DEVICE
  void init(int wg_idx) {  // wg_idx in among all WG participants
    for (int i = wg_idx; i > 0; --i) {
      cutlass::arch::NamedBarrier::arrive(cutlass::NumThreadsPerWarpGroup * NumWG, mapping_[i - 1]);
    }
    // with 3 WGs, init to namedbarrier_id:(arrived_wg,expected_wg)
    // 0:(2,3)
    // 1:(1,3)
    // 2:(0,3)
  }

  CUTE_DEVICE
  ~OrderedNamedBarriers() {
    // FIXME: this will be a problem for persistent scheduler
  }

  CUTE_DEVICE
  void ordered_or_wait(int wg_idx) {  // wg_idx in participants
    // during first call, before
    // 0:(2,3)
    // 1:(1,3)
    // 2:(0,3)
    cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup * NumWG, mapping_[wg_idx]);
    // after
    // 0:(3,3) immediately unblock wg0, and named barrier automatically reset to (0,3)
    // 1:(2,3)
    // 2:(1,3)
  }

  CUTE_DEVICE
  void notify_next_blocked(int wg_idx) {  // wg_idx in participants
    // always call this after ordered_or_wait
    // during first call, before
    // 0:(0,3)
    // 1:(2,3)
    // 2:(1,3)
    CUTE_UNROLL
    for (int i = 1; i < NumWG; ++i) {
      cutlass::arch::NamedBarrier::arrive(cutlass::NumThreadsPerWarpGroup * NumWG, mapping_[(wg_idx + i) % NumWG]);
    }
    // after wg0 called this function
    // 0:(0,3), wg0 has not reached on second ordered_or_wait() or (1,3) wg0 wait on second ordered_or_wait() call
    // 1:(0,3), unblocked wg1's first ordered_or_wait() and reset nb1
    // 2:(2,3), still wait on first ordered_or_wait() call
    //
    // after wg1 called this function
    // 0:(1,3), wg0 has not reached on second ordered_or_wait() or (2,3) wg0 wait on second ordered_or_wait() call
    // 1:(0,3), wg1 has not reached on second ordered_or_wait() or (1,3) wg1 wait on second ordered_or_wait() call
    // 2:(0,3), unblocked wg2's first ordered_or_wait() and reset nb2
    //
    // after wg2 called this function
    // 0:(2,3), wg0 has not reached on second ordered_or_wait() or (0,3) wg0 wait on second ordered_or_wait() call,
    // unblocked 1:(1,3), wg1 has not reached on second ordered_or_wait() or (2,3) wg1 wait on second
    // ordered_or_wait() call, still block 2:(0,3), unblock wg0 ordered_or_wait() and reset
    //
  }

 private:
  cute::array<NBId_t, NumWG> mapping_;
};
}  // namespace kda::sm90
