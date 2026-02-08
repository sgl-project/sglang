/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "cutlass/cutlass.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cute/container/array.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

namespace detail {

// MSVC work-around
template <int Stages>
struct PrefetcherPipelineSharedStorage {
  using TransactionBarrier = cutlass::arch::ClusterTransactionBarrier;
  using Barrier = cutlass::arch::ClusterBarrier;

  TransactionBarrier tma_barrier[Stages];
  Barrier producer_ready_barrier;
};

} // end namespace detail

using namespace cute;

// Prefetcher pipeline is modeled after PipelineTmaAsync, with a cluster transaction
// barrier providing control over the number of concurrent outstanding TMA loads.
// There is also an additional cluster barrier which is only used when `prefetch_ratio` is unset.
// `prefetch_ratio` determines how many K tiles get loaded, and when unset, the prefetcher checks
// whether DMA warps are done waiting on griddepcontrol, and if so, stops issuing more TMA loads.
template <int Stages_>
class PrefetchPipeline {
public :
  static constexpr uint32_t Stages = Stages_;
  using SharedStorage = detail::PrefetcherPipelineSharedStorage<Stages>;

  using TransactionBarrier = typename SharedStorage::TransactionBarrier;
  using Barrier = typename SharedStorage::Barrier;
  using PrefetcherBarrierType = typename TransactionBarrier::ValueType;

  struct Params {
    uint32_t transaction_bytes = 0;
    uint32_t num_prefetchers = 1;
    bool should_prefetch = false;
  };

  // Constructor
  CUTLASS_DEVICE
  PrefetchPipeline(SharedStorage& storage, Params params)
      : params_(params)
      , tma_barrier_ptr_(&storage.tma_barrier[0])
      , producer_ready_barrier_ptr_(&storage.producer_ready_barrier) {

    int lane_predicate = cute::elect_one_sync();
    if (params.should_prefetch && lane_predicate) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        tma_barrier_ptr_[i].init(params.num_prefetchers);
      }
      producer_ready_barrier_ptr_[0].init(1);
    }
  }

  CUTLASS_DEVICE
  void producer_arrive() {
    if (params_.should_prefetch) {
      producer_ready_barrier_ptr_[0].arrive();
    }
  }

  CUTLASS_DEVICE
  bool have_producers_arrived() {
    if (params_.should_prefetch) {
      uint32_t barrier_status_ = producer_ready_barrier_ptr_[0].try_wait(0);
      auto barrier_status = static_cast<BarrierStatus>(barrier_status_);
      if (barrier_status == BarrierStatus::WaitDone) {
        return true; // exit prefetcher loop
      }
      return false;
    }
    return true;
  }

  CUTLASS_DEVICE
  void prefetcher_acquire(uint32_t stage, uint32_t phase, bool should_wait) {
    if (params_.should_prefetch) {
      if (should_wait) {
        tma_barrier_ptr_[stage].wait(phase ^ 1);
      }
      tma_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
    }
  }

  CUTLASS_DEVICE
  void advance_prefetcher_state(uint32_t& stage, uint32_t& phase) {
    if (params_.should_prefetch) {
      stage++;
      if (stage == Stages) {
        stage = 0;
        phase ^= 1;
      }
    }
  }

  CUTLASS_DEVICE
  void prefetcher_tail(uint32_t stage, uint32_t phase) {
    if (params_.should_prefetch) {
      // Wait on any already-issued loads
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < stage; ++i) {
        tma_barrier_ptr_[i].wait(phase);
      }
    }
  }

  CUTLASS_DEVICE
  PrefetcherBarrierType* prefetcher_get_barrier(uint32_t stage) {
    return reinterpret_cast<PrefetcherBarrierType*>(&tma_barrier_ptr_[stage]);
  }

private :
  TransactionBarrier* tma_barrier_ptr_ = nullptr;
  Barrier* producer_ready_barrier_ptr_ = nullptr;
  Params params_;

};

}  // end namespace cutlass
