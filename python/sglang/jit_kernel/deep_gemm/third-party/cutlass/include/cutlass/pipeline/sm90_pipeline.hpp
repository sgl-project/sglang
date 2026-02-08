/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cute/layout.hpp"
#include "cute/layout_composed.hpp"  // cute::composition
#include "cute/swizzle.hpp"             // cute::Swizzle
#include "cute/swizzle_layout.hpp"      // cute::composition
#include "cute/util/type_traits.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/container/array.hpp"
#include "cute/numeric/integral_constant.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/detail/dependent_false.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;

namespace detail {

// Helper function for DEBUG checks
template<class ThreadCategory>
CUTLASS_DEVICE
bool pipeline_is_producer(ThreadCategory role) {
  return (role == ThreadCategory::Producer || role == ThreadCategory::ProducerConsumer);
}

template<class ThreadCategory>
CUTLASS_DEVICE
void pipeline_check_is_producer(ThreadCategory role) {
  #ifndef NDEBUG
  if (!pipeline_is_producer(role)) {
    asm volatile ("brkpt;\n" ::);
  }
  #endif
}

template<class ThreadCategory>
CUTLASS_DEVICE
bool pipeline_is_consumer(ThreadCategory role) {
  return (role == ThreadCategory::Consumer || role == ThreadCategory::ProducerConsumer);
}

template<class ThreadCategory>
CUTLASS_DEVICE
void pipeline_check_is_consumer(ThreadCategory role) {
  #ifndef NDEBUG
  if (!pipeline_is_consumer(role)) {
    asm volatile ("brkpt;\n" ::);
  }
  #endif
}

CUTLASS_DEVICE
cute::tuple<bool, uint32_t> spread_arrivals_to_warp(int thread_idx_in_warp) {
  constexpr uint32_t MaxClusterSize = 16;
  bool is_signaling_thread = (thread_idx_in_warp % (32 / MaxClusterSize)) == 0;
  auto layout = Layout<Shape<_4,_4>,Stride<_4, _1>>{};
  uint32_t thread_row = thread_idx_in_warp / 8;
  uint32_t thread_col = (thread_idx_in_warp % 8) / 2;
  uint32_t dst_blockid = layout(thread_row, thread_col);
  return cute::make_tuple(is_signaling_thread, dst_blockid);
}

CUTLASS_DEVICE
cute::tuple<bool, uint32_t> spread_arrivals_to_warpgroup(int thread_idx_in_warpgroup, int warp_idx) {
  constexpr uint32_t MaxClusterSize = 16;
  bool is_signaling_thread = (thread_idx_in_warpgroup % (NumThreadsPerWarpGroup / MaxClusterSize)) == 0;
  auto layout = cute::composition(Swizzle<2,0,-2>{},
                                  Layout<Shape<_4,_4>,Stride<_4,_1>>{});
  uint32_t thread_row = warp_idx % 4;
  uint32_t thread_col = (thread_idx_in_warpgroup / 8) % 4;
  uint32_t dst_blockid = layout(thread_row, thread_col);
  return cute::make_tuple(is_signaling_thread, dst_blockid);
}
} // namespace detail

enum class BarrierStatus : uint32_t {
  WaitAgain = 0u,
  WaitDone  = 1u,
};

class ArrivalToken {
public:
  CUTLASS_HOST_DEVICE
  ArrivalToken(BarrierStatus barrier_status) : barrier_status_(barrier_status) {}

  CUTLASS_HOST_DEVICE
  ArrivalToken() = delete;

  CUTLASS_HOST_DEVICE
  BarrierStatus get() const {
    return barrier_status_;
  }

  CUTLASS_HOST_DEVICE
  bool operator==(ArrivalToken const& other) const {
    return barrier_status_ == other.get();
  }

private:
  BarrierStatus barrier_status_;

  CUTLASS_HOST_DEVICE
  friend bool operator==(const ArrivalToken& left, const BarrierStatus& right) {
    return left.get() == right;
  }

  CUTLASS_HOST_DEVICE
  friend bool operator==(const BarrierStatus& left, const ArrivalToken& right) {
    return left == right.get();
  }

  CUTLASS_HOST_DEVICE
  friend bool operator!=(const ArrivalToken& left, const BarrierStatus& right) {
    return left.get() != right;
  }

  CUTLASS_HOST_DEVICE
  friend bool operator!=(const BarrierStatus& left, const ArrivalToken& right) {
    return left != right.get();
  }
};

class ProducerToken : public ArrivalToken {
  using ArrivalToken::ArrivalToken;
};

class ConsumerToken : public ArrivalToken {
  using ArrivalToken::ArrivalToken;
};

// Circular Buffer Index + Associated Phase
// Assumes only one operation possible - i.e., ++
template<uint32_t Stages_>
struct PipelineState {

  static constexpr uint32_t Stages = Stages_;

  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t count_ = 0;

  CUTLASS_DEVICE
  PipelineState(): index_{}, phase_{}, count_{} {}

  CUTLASS_DEVICE
  PipelineState(int index, uint32_t phase, uint32_t count)
    : index_(index)
    , phase_(phase)
    , count_(count) {}

  CUTLASS_DEVICE
  int index() const {
    return index_;
  }

  CUTLASS_DEVICE
  uint32_t phase() const {
    return phase_;
  }

  CUTLASS_DEVICE
  uint32_t count() const {
    return count_;
  }

  CUTLASS_DEVICE
  void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;
      }
    }
  }

  CUTLASS_DEVICE
  PipelineState& operator+=(uint32_t num_iterations) {
    return advance(num_iterations);
  }

  CUTLASS_DEVICE
  PipelineState& operator=(PipelineState const& other) {
    index_ = other.index();
    phase_ = other.phase();
    count_ = other.count();
    return *this;
  }

  CUTLASS_DEVICE
  PipelineState& advance(uint32_t num_iterations) {
    if constexpr (Stages > 0) {
      // Number of iterations cross over the stage boundary => flipped phase
      if ((num_iterations < Stages) && (index_ + num_iterations) >= Stages ) {
        phase_ ^= 1;
      }
      // How many times number of iterations cross over the stage boundary and
      // end up on a odd number => flipped phase
      if ((num_iterations >= Stages) && (((index_ + num_iterations) / Stages) % 2) == 1) {
        phase_ ^= 1;
      }
      index_ = (index_ + num_iterations) % Stages;
      count_ += num_iterations;
    }
    return *this;
  }

  CUTLASS_DEVICE
  static PipelineState make_pipeline_state(PipelineState start_state, uint32_t num_iterations) {
    return start_state.advance(num_iterations);
  }
};

template<class Pipeline>
CUTLASS_DEVICE
PipelineState<Pipeline::Stages> make_producer_start_state() {
  // Producer starts with an opposite phase as the buffers are initially empty
  constexpr int InitialProducerStage = 0;
  constexpr uint32_t InitialProducerPhase = 1;
  constexpr uint32_t InitialProducerCount = 0;
  return {InitialProducerStage, InitialProducerPhase, InitialProducerCount};
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMA load (producer) Async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////
// Assumptions : Constructor is visible Cluster-wide (as it needs a Cluster-Sync)
// We have exactly one thread elected in the Producer as the "leader"
// Currently, it is optional to elect a leader for the Consumers
template <int Stages_>
class PipelineTmaAsync {
public:
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = cutlass::PipelineState<Stages>;

  struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    uint32_t transaction_bytes = 0;
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t is_leader = 0;
    uint32_t num_consumers = 0; // Number of consumer threads
    uint32_t num_producers = 1; // Number of producer threads
    int initializing_warp = 0; 
  };

  template <class ClusterShape>
  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params, ClusterShape cluster_shape) {
    int warp_idx = canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == 0);
    is_initializing_warp = (warp_idx == params.initializing_warp); 
    if (is_initializing_warp) {
      // Barrier FULL and EMPTY init
      uint32_t const producer_arv_cnt = params.num_producers;
      uint32_t const num_consumer_warpgroups_per_cluster = cute::ceil_div(params.num_consumers, static_cast<uint32_t>(NumThreadsPerWarpGroup));
      uint32_t multicast_consumer_arrival_count = params.num_consumers; // If cluster_size is 1
      if (cute::size(cluster_shape) > 1) {
        multicast_consumer_arrival_count = (cute::size<0>(cluster_shape) + cute::size<1>(cluster_shape) - 1) *
              num_consumer_warpgroups_per_cluster;
      }
      CUTLASS_ASSERT(multicast_consumer_arrival_count > 0 && "Multicast consumer arrival count must be non-zero");
      CUTLASS_ASSERT(producer_arv_cnt > 0 && "Producer arrival count must be non-zero");
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  template<class ClusterShape, class InitBarriers, class InitMasks>
  CUTLASS_DEVICE
  PipelineTmaAsync(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
      : params_(params)
      , full_barrier_ptr_(&storage.full_barrier_[0])
      , empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    int warp_idx = canonical_warp_idx_sync();
    int thread_idx = threadIdx.x;
    int lane_predicate = cute::elect_one_sync();

    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params_, cluster_shape);
    }

    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      // Logic to optimally schedule Empty Arrives
      // Goal : To divide SYNCS Empty Arrival duty equally amongst the Warp-Group (128 threads)
      dim3 block_id = cute::block_id_in_cluster();
      auto cluster_size = cute::size(cluster_shape);

      if (cluster_size == 1) {
        is_signaling_thread_ = true;
        dst_blockid_ = 0;
      }
      else {
        // STEP 1 : Use Cute Layout function to generate an optimal dst block-id (0-15)
        if (params_.num_consumers % NumThreadsPerWarpGroup == 0) {
          auto [is_signaling_thread, dst_blockid] = detail::spread_arrivals_to_warpgroup(thread_idx % NumThreadsPerWarpGroup, warp_idx);
          is_signaling_thread_ = is_signaling_thread;
          dst_blockid_ = dst_blockid;
        }
        else if (params_.num_consumers == 32) {
          auto [is_signaling_thread, dst_blockid] = detail::spread_arrivals_to_warp(thread_idx % 32);
          is_signaling_thread_ = is_signaling_thread;
          dst_blockid_ = dst_blockid;
        }
        else {
          is_signaling_thread_ = 0;
          #ifndef NDEBUG
            asm volatile ("brkpt;\n" ::);
          #endif
        }

        // STEP 2: Find if this dst block-id needs an arrival for this problem
        is_signaling_thread_ &= dst_blockid_ < cluster_size;
        is_signaling_thread_ &= is_same_row_or_col(dst_blockid_, block_id, cluster_shape);
      }
    }
  }

  // Constructor
  template<class ClusterShape>
  CUTLASS_DEVICE
  PipelineTmaAsync(SharedStorage& storage, Params params, ClusterShape cluster_shape)
      : PipelineTmaAsync(storage, params, cluster_shape, cute::true_type{}, cute::true_type{}) { }
  
  template<class ClusterShape, class InitBarriers>
  CUTLASS_DEVICE
  PipelineTmaAsync(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {})
      : PipelineTmaAsync(storage, params, cluster_shape, InitBarriers{}, cute::true_type{}) { }

  template <class ClusterShape>
  CUTLASS_DEVICE
  bool is_same_row_or_col(int dst_block_id, dim3 block_id, ClusterShape cluster_shape) {
    return (((dst_block_id % cute::size<0>(cluster_shape)) == block_id.x) ||
            (
              ((dst_block_id / cute::size<0>(cluster_shape)) == block_id.y)
            ));
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait.
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the barrier to flip.
  // It opportunistically waits for an implementation-dependent timeout.
  // Whether or not the barrier has flipped yet, the try function will return a token.
  // If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.

  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state) {
    producer_acquire(state.index(), state.phase());
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state, uint32_t bytes) {
    producer_commit(state.index(), bytes);
  }

  template<class UserDefinedArriveOp>
  CUTLASS_DEVICE
  void producer_commit(PipelineState state, UserDefinedArriveOp&& user_defined_arrive_op) {
    cute::forward<UserDefinedArriveOp>(user_defined_arrive_op)(producer_get_barrier(state.index()));;
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    detail::pipeline_check_is_producer(params_.role);
    for (int count = 0; count < Stages; ++count) {
      empty_barrier_ptr_[state.index()].wait(state.phase());
      ++state;
    }
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return producer_get_barrier(state.index());
  }

  CUTLASS_DEVICE
  void producer_expect_transaction(PipelineState state, uint32_t transaction_bytes) {
    producer_expect_transaction(state.index(), transaction_bytes);
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_test_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state) {
    consumer_wait(state.index(), state.phase());
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

private:
  uint32_t dst_blockid_ = 0;
  uint32_t is_signaling_thread_ = 0;
  FullBarrier *full_barrier_ptr_ = nullptr;
  EmptyBarrier *empty_barrier_ptr_ = nullptr;
  Params params_;

  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_producer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase) {
    empty_barrier_ptr_[stage].wait(phase);

    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
    }
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Consumer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }

    // Most likely you have elected more than one leader
    if (params_.is_leader && (threadIdx.x % 32 != 0)) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);
    if (barrier_token != BarrierStatus::WaitDone) {
      empty_barrier_ptr_[stage].wait(phase);
    }

    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
    }
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Consumer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }

    // Most likely you have elected more than one leader
    if (params_.is_leader && (threadIdx.x % 32 != 0)) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }

  CUTLASS_DEVICE
  void producer_expect_transaction(uint32_t stage, uint32_t transaction_bytes) {
    detail::pipeline_check_is_producer(params_.role);
    if (params_.is_leader) {
      full_barrier_ptr_[stage].expect_transaction(transaction_bytes);
    }
  }

  // NOP for TMA based mainloop
  CUTLASS_DEVICE
  void producer_commit(uint32_t stage, uint32_t bytes) {
    // Below code is used only for unit-testing (in the absence of TMA commit)
    #if CUTLASS_UNIT_TEST_PIPELINE
      if (params_.is_leader) {
        // STEP 1 : Commit to self
        full_barrier_ptr_[stage].complete_transaction(bytes);

        // STEP 2 : Commit to other blocks in our cluster
        auto cluster_shape = cute::cluster_shape();
        Layout block_layout_in_cluster = make_layout(cluster_shape);
        dim3 local_block_id = cute::block_id_in_cluster();

        CUTLASS_PRAGMA_UNROLL
        for(int n = 0; n < size<1>(block_layout_in_cluster); ++n) {
          uint32_t dst_block_id = block_layout_in_cluster(local_block_id.x,n,Int<0>{});
          full_barrier_ptr_[stage].complete_transaction(dst_block_id, bytes, n!=local_block_id.y);
        }

        CUTLASS_PRAGMA_UNROLL
        for(int m = 0; m < size<0>(block_layout_in_cluster); ++m) {
          uint32_t dst_block_id = block_layout_in_cluster(m,local_block_id.y,Int<0>{});
          full_barrier_ptr_[stage].complete_transaction(dst_block_id, bytes, m!=local_block_id.x);
        }
      }
    #endif
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].test_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  // Wait for producer to commit transactions (done by TMA)
  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    detail::pipeline_check_is_consumer(params_.role);
    full_barrier_ptr_[stage].wait(phase);
  }

  // Wait for producer to commit transactions (done by TMA)
  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
    detail::pipeline_check_is_consumer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    detail::pipeline_check_is_consumer(params_.role);
    empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signaling_thread_ & (!skip));
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Producer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMA store pipeline class
// producer-only class, no async barriers between threads because consumer is TMA unit
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <
  int Stages_,
  // The number of committed TMA store batches that can be in flight upon return of producer acquire
  int UnacquiredStages_ = Stages_-1
>
class PipelineTmaStore {
public:
  static constexpr uint32_t Stages = Stages_;
  static_assert(Stages_ > 0);
  static_assert(UnacquiredStages_ >= 0);
  static constexpr uint32_t UnacquiredStages = static_cast<uint32_t>(UnacquiredStages_);
  using PipelineState = cutlass::PipelineState<Stages>;

  struct Params {
    bool always_wait = false;
  };

  CUTLASS_DEVICE
  PipelineTmaStore(Params params = {}) : params_(params) {}

  ////////////////////
  // Producer APIs
  ////////////////////
  // Wait for the least recently committed batch of TMA stores to complete
  CUTLASS_DEVICE
  void producer_acquire(PipelineState state) {
    producer_acquire(state.index(), state.count());
  }

  // Commit the most recently issued batch of TMA stores
  CUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    producer_commit(state.index(), state.count());
  }

  // Wait for all TMA stores to complete
  CUTLASS_DEVICE
  void producer_tail([[maybe_unused]] PipelineState state) {
    tma_store_wait<0>();
  }

private:
  Params params_;

  // Wait for the least recently committed batch of TMA stores to complete
  // or until at most UnacquiredStages TMA store batches are in-flight (if specified)
  CUTLASS_DEVICE
  void producer_acquire([[maybe_unused]] uint32_t stage, uint32_t count) {
    if (params_.always_wait || count > UnacquiredStages) {
      tma_store_wait<UnacquiredStages>();
    }
  }

  // Commit the most recently issued batch of TMA stores
  CUTLASS_DEVICE
  void producer_commit([[maybe_unused]] uint32_t stage, [[maybe_unused]] uint32_t count) {
    tma_store_arrive();
  }
};

template <>
class PipelineTmaStore< /* Stages_ = */ 0, /* UnacquiredStages = Stages_ - 1 = */ -1 > {
public:
  static constexpr uint32_t Stages = 0;
  static constexpr uint32_t UnacquiredStages = 0;
  using PipelineState = cutlass::PipelineState<Stages>;

  struct Params {
    bool always_wait = false;
  };

  PipelineTmaStore() = default;
  CUTLASS_DEVICE
    PipelineTmaStore(Params params) : params_(params) {}

  ////////////////////
  // Producer APIs
  ////////////////////

  template<class ThisTemplateParameterExistsOnlyForDependentFalse = int>
  CUTLASS_DEVICE
    void producer_acquire(PipelineState /* state */,
      ThisTemplateParameterExistsOnlyForDependentFalse* /* unused */ = nullptr) {
    static_assert(cutlass::detail::dependent_false<ThisTemplateParameterExistsOnlyForDependentFalse>,
      "It is never valid to call PipelineTmaStore<0>::producer_acquire");
  }

  // Commit the most recently issued batch of TMA stores
  CUTLASS_DEVICE
    void producer_commit(PipelineState state) {
    producer_commit(state.index(), state.count());
  }

  // Wait for all TMA stores to complete
  CUTLASS_DEVICE
    void producer_tail([[maybe_unused]] PipelineState state) {
    tma_store_wait<0>();
  }

private:
  Params params_;

  // Commit the most recently issued batch of TMA stores
  CUTLASS_DEVICE
    void producer_commit([[maybe_unused]] uint32_t stage, [[maybe_unused]] uint32_t count) {
    tma_store_arrive();
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Simple producer-consumer async Pipeline class using producer transaction barriers
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <int Stages_>
class PipelineTransactionAsync {
public:
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = cutlass::PipelineState<Stages>;

  struct SharedStorage {
    cute::array<FullBarrier, Stages> full_barrier_;
    cute::array<EmptyBarrier, Stages> empty_barrier_;
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t transaction_bytes = 0;
    uint32_t producer_arv_count = 1;
    uint32_t consumer_arv_count = 1;
    uint32_t dst_blockid = cute::block_rank_in_cluster();
    int initializing_warp = 0; 
  };

  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params const& params) {
    FullBarrier *full_barrier_ptr = storage.full_barrier_.data();
    EmptyBarrier *empty_barrier_ptr = storage.empty_barrier_.data();
    int warp_idx = canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == 0);
    is_initializing_warp = (warp_idx == params.initializing_warp); 

    if (is_initializing_warp) {
      // Barrier FULL and EMPTY init
      CUTLASS_ASSERT(params.producer_arv_count > 0 && "Producer arrival count must be non-zero");
      CUTLASS_ASSERT(params.consumer_arv_count > 0 && "Consumer arrival count must be non-zero");
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(full_barrier_ptr), decltype(empty_barrier_ptr), Stages>(
          full_barrier_ptr, empty_barrier_ptr, params.producer_arv_count, params.consumer_arv_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  // Constructor
  template<class InitBarriers>
  CUTLASS_DEVICE
  PipelineTransactionAsync(SharedStorage& storage, Params const& params, InitBarriers = cute::true_type{})
    : params_(params)
    , full_barrier_ptr_(storage.full_barrier_.data())
    , empty_barrier_ptr_(storage.empty_barrier_.data()) {

    int warp_idx = canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);

    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params);
    }

  }

  // Constructor
  CUTLASS_DEVICE
  PipelineTransactionAsync(SharedStorage& storage, Params const& params) :
    PipelineTransactionAsync(storage, params, cute::true_type{}) { }

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait.
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the barrier to flip.
  // It opportunistically waits for an implementation-dependent timeout.
  // Whether or not the barrier has flipped yet, the try function will return a token.
  // If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.
  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  // Perform an expect-tx operation on the stage's full barrier. Must be called by 1 thread
  CUTLASS_DEVICE
  void producer_expect_transaction(PipelineState state) {
    producer_expect_transaction(state.index());
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    producer_commit(state.index());
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    for (int count = 0; count < Stages; ++count) {
      producer_acquire(state);
      ++state;
    }
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_test_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

private:
  FullBarrier *full_barrier_ptr_ = nullptr;
  EmptyBarrier *empty_barrier_ptr_ = nullptr;
  Params params_;

  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_producer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }
  }

  // Perform an expect-tx operation on the stage's full barrier. Must be called by 1 thread
  CUTLASS_DEVICE
  void producer_expect_transaction(uint32_t stage) {
    detail::pipeline_check_is_producer(params_.role);
    full_barrier_ptr_[stage].expect_transaction(params_.transaction_bytes);
  }

  CUTLASS_DEVICE
  void producer_commit(uint32_t stage) {
    detail::pipeline_check_is_producer(params_.role);
    full_barrier_ptr_[stage].arrive(params_.dst_blockid);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].test_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
    detail::pipeline_check_is_consumer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    detail::pipeline_check_is_consumer(params_.role);
    empty_barrier_ptr_[stage].arrive(params_.dst_blockid, (not skip));
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Simple producer-consumer async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace PipelineDetail {
  template<int Stages>
  using PipelineAsyncPipelineState = cutlass::PipelineState<Stages>;

  template<int Stages>
  struct PipelineAsyncSharedStorage {
    using FullBarrier = cutlass::arch::ClusterBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;

    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };
};

template <int Stages_>
class PipelineAsync {
public:
  static constexpr uint32_t Stages = Stages_;
  using SharedStorage = PipelineDetail::PipelineAsyncSharedStorage<Stages>;
  using FullBarrier = typename SharedStorage::FullBarrier;
  using EmptyBarrier = typename SharedStorage::EmptyBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  using ConsumerBarrierType = typename EmptyBarrier::ValueType;
  using PipelineState = PipelineDetail::PipelineAsyncPipelineState<Stages>;

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t producer_arv_count = 1;
    uint32_t consumer_arv_count = 1;
    uint32_t dst_blockid = cute::block_rank_in_cluster();
    int initializing_warp = 0; 
  };

  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params) {
    int warp_idx = canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == 0);
    is_initializing_warp = (warp_idx == params.initializing_warp); 
    if (is_initializing_warp) {
      // Barrier FULL and EMPTY init
      CUTLASS_ASSERT(params.producer_arv_count > 0 && "Producer arrival count must be non-zero");
      CUTLASS_ASSERT(params.consumer_arv_count > 0 && "Consumer arrival count must be non-zero");
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, params.producer_arv_count, params.consumer_arv_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  template<class InitBarriers>
  CUTLASS_DEVICE
  PipelineAsync(
    SharedStorage& storage,
    Params const& params,
    InitBarriers = {}) :
      params_(params),
      full_barrier_ptr_(&storage.full_barrier_[0]),
      empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params_);
    }
  }

  CUTLASS_DEVICE
  PipelineAsync(
    SharedStorage& storage,
    Params const& params) :
      PipelineAsync(storage, params, cute::true_type{}) { }

  // Default assumption when only storage is passed is :
  // => single producer, single consumer & they are in the same block (within the Cluster)
  CUTLASS_DEVICE
  PipelineAsync(SharedStorage& storage)
    : PipelineAsync(storage, {}, cute::true_type{}) {}

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait.
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the barrier to flip.
  // It opportunistically waits for an implementation-dependent timeout.
  // Whether or not the barrier has flipped yet, the try function will return a token.
  // If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.
  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    producer_commit(state.index());
  }

  template<class UserDefinedArriveOp>
  CUTLASS_DEVICE
  void producer_commit(PipelineState state, UserDefinedArriveOp&& user_defined_arrive_op) {
    cute::forward<UserDefinedArriveOp>(user_defined_arrive_op)(producer_get_barrier(state.index()));
    producer_commit(state);
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    for (int count = 0; count < Stages; ++count) {
      producer_acquire(state);
      ++state;
    }
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_test_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }

private:
  Params params_;
  FullBarrier *full_barrier_ptr_;
  EmptyBarrier *empty_barrier_ptr_;

  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_producer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void producer_commit(uint32_t stage) {
    detail::pipeline_check_is_producer(params_.role);
    full_barrier_ptr_[stage].arrive();
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].test_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    detail::pipeline_check_is_consumer(params_.role);
    bool done = full_barrier_ptr_[stage].test_wait(phase);
    if (!done) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
    detail::pipeline_check_is_consumer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_release(uint32_t stage) {
    detail::pipeline_check_is_consumer(params_.role);
    empty_barrier_ptr_[stage].arrive(params_.dst_blockid);
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Barrier to ensure an Ordered Sequence between
// SequenceLength number of groups (each with group_size participants) executing SequenceDepth Stages
// i.e., for all i < j - only after id "i" arrives at a particular stage "m"
// will the wait() for id "j" succeed for the same stage
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace PipelineDetail {

template<int SequenceDepth, int SequenceLength>
struct OrderedSequenceBarrierSharedStorage {
  using Barrier = cutlass::arch::ClusterBarrier;
  Barrier barrier_[SequenceDepth][SequenceLength];
};

} // namespace PipelineDetail

template<int SequenceDepth_, int SequenceLength_>
class OrderedSequenceBarrier {
public:
  static constexpr int SequenceDepth = SequenceDepth_;
  static constexpr int SequenceLength = SequenceLength_;
  using SharedStorage =
    PipelineDetail::OrderedSequenceBarrierSharedStorage<SequenceDepth, SequenceLength>;
  using Barrier = typename SharedStorage::Barrier;

  struct Params {
    uint32_t group_id;
    uint32_t group_size;
    int initializing_warp = 0; 
  };

private:
  // In future this Params object can be replaced easily with a CG object
  Params params_;
  Barrier *barrier_ptr_;
  PipelineState<SequenceDepth> stage_;

  static constexpr int Depth = SequenceDepth;
  static constexpr int Length = SequenceLength;

public:
  OrderedSequenceBarrier() = delete;
  OrderedSequenceBarrier(const OrderedSequenceBarrier&) = delete;
  OrderedSequenceBarrier(OrderedSequenceBarrier&&) = delete;
  OrderedSequenceBarrier& operator=(const OrderedSequenceBarrier&) = delete;
  OrderedSequenceBarrier& operator=(OrderedSequenceBarrier&&) = delete;
  ~OrderedSequenceBarrier() = default;

  CUTLASS_DEVICE
  OrderedSequenceBarrier(SharedStorage& storage, Params const& params) :
      params_(params),
      barrier_ptr_(&storage.barrier_[0][0]),
      // Group 0 - starts with an opposite phase
      stage_({0, params.group_id == 0, 0}) {

#if (__CUDA_ARCH__ >= 1000)
    int warp_idx = canonical_warp_idx_sync();

    // Barrier FULL, EMPTY init
    if (warp_idx == params.initializing_warp) {
      int arv_cnt = params.group_size;
      CUTLASS_ASSERT(arv_cnt > 0 && "Arrive count must be non-zero");
      constexpr int Stages = Depth * Length;
      cutlass::arch::detail::initialize_barrier_array_aligned<decltype(barrier_ptr_), Stages>(
          barrier_ptr_, arv_cnt);
    }
#else

    int warp_idx = canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    CUTLASS_ASSERT(params.group_size > 0 && "Group size must be non-zero");

    // Barrier FULL, EMPTY init
    // Init is done only by the one elected thread of the block
    if (warp_idx == 0 && lane_predicate) {
      for (int d = 0; d < Depth; ++d) {
        for (int l = 0; l < Length; ++l) {
          barrier_ptr_[d * Length + l].init(params.group_size);
        }
      }
    }
#endif 
    cutlass::arch::fence_barrier_init();
  }

  // Wait on a stage to be unlocked
  CUTLASS_DEVICE
  void wait() {
    get_barrier_for_current_stage(params_.group_id).wait(stage_.phase());
  }

  // Signal completion of Stage and move to the next stage
  // (group_id) signals to (group_id+1)
  CUTLASS_DEVICE
  void arrive() {
    int signalling_id = (params_.group_id + 1) % Length;
    get_barrier_for_current_stage(signalling_id).arrive();
    ++stage_;
  }

  CUTLASS_DEVICE
  void advance() {
    ++stage_;
  }

private:

  CUTLASS_DEVICE
  Barrier& get_barrier_for_current_stage(int group_id) {
    return barrier_ptr_[stage_.index() * Length + group_id];
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Synchronization call. Blocks until barriers are initialized in shared memory.
CUTLASS_DEVICE
void
pipeline_init_wait(int cluster_size) {
  if (cluster_size > 1) {
    cute::cluster_wait();
  }
  else {
    __syncthreads();
  }
}

// Used to guarantee that the Pipeline init is visible
// to all producers and consumer threadblocks in the cluster
CUTLASS_DEVICE
void
pipeline_init_arrive_relaxed(int cluster_size) {
  if (cluster_size > 1) {
    cute::cluster_arrive_relaxed();
  }
  else {
    __syncthreads();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // end namespace cutlass
