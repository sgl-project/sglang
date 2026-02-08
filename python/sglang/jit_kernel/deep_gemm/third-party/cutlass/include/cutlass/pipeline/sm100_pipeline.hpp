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
//

//

#include "cute/numeric/integral_constant.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "sm90_pipeline.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

using namespace cute;

enum class McastDirection {
  kRow,
  kCol,
  kRowCol
};
namespace detail {

template<McastDirection McastDir, class ClusterShape, class AtomThrShape_MNK>
CUTLASS_DEVICE
uint16_t calculate_multicast_mask(ClusterShape cluster_shape, AtomThrShape_MNK atom_thr_shape, dim3 block_id_in_cluster) {
  auto is_participant = [&](auto x, auto y) {
    if constexpr (McastDir == McastDirection::kRowCol) {
      return (x/size<0>(atom_thr_shape) == block_id_in_cluster.x/size<0>(atom_thr_shape) || // is same MMA cluster col
              y/size<1>(atom_thr_shape) == block_id_in_cluster.y/size<1>(atom_thr_shape));  // is same MMA cluster row
    }
    else if constexpr (McastDir == McastDirection::kRow) {
      return (x/size<0>(atom_thr_shape) == block_id_in_cluster.x/size<0>(atom_thr_shape));  // is same MMA cluster row
    }
    else { // (McastDir == McastDirection::kCol)
      return (y/size<1>(atom_thr_shape) == block_id_in_cluster.y/size<1>(atom_thr_shape));  // is same MMA cluster col
    }
  };
  
  uint16_t block_id_mask = 0;
  auto cluster_layout = make_layout(cluster_shape);
  // When MMA_2x1SM instructions are used, the definition of "same row" changes.
  // With MMA_2x1SM, we need to send the notification for MMA completion to all
  // 2x1 threadblocks of the cluster. Below is a 4x4 example where R are the threadblocks
  // that receives the release for A/B buffers that threadblock (0,0) uses.
  // Row&Col   Row     Col
  // RRRR      RRRR    Cxxx
  // RRRR      RRRR    Cxxx
  // Rxxx      xxxx    Cxxx
  // Rxxx      xxxx    Cxxx
  CUTLASS_PRAGMA_UNROLL
  for (int x = 0; x<size<0>(cluster_shape); x++) {
    CUTLASS_PRAGMA_UNROLL
    for (int y = 0; y<size<1>(cluster_shape); y++) {
      if (is_participant(x,y)) {
        block_id_mask |= (1 << cluster_layout(x,y, Int<0>{}));
      }
    }
  }
  return block_id_mask;
}

template<class ClusterShape, class AtomThrShape_MNK>
CUTLASS_DEVICE
uint16_t calculate_umma_peer_mask(ClusterShape cluster_shape, AtomThrShape_MNK atom_thr_shape, dim3 block_id_in_cluster) {
  uint16_t tmem_sync_mask = 0;
  auto cluster_layout =  make_layout(cluster_shape);
  int block_id_in_cluster_x = (block_id_in_cluster.x / size<0>(AtomThrShape_MNK{})) * size<0>(AtomThrShape_MNK{}) ;
  int block_id_in_cluster_y = (block_id_in_cluster.y / size<1>(AtomThrShape_MNK{})) * size<1>(AtomThrShape_MNK{}) ;
  CUTLASS_PRAGMA_UNROLL
  for (int x = 0; x < size<0>(AtomThrShape_MNK{}); x++) {
    CUTLASS_PRAGMA_UNROLL
    for (int y = 0; y < size<1>(AtomThrShape_MNK{}); y++) {
      tmem_sync_mask |= (1 << cluster_layout(block_id_in_cluster_x + x, block_id_in_cluster_y + y, Int<0>{}));
    }
  }

  return tmem_sync_mask;
}
} // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMA (producer) Async Pipeline class for Blackwell UMMA
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <int Stages_, class AtomThrShape_MNK_ = Shape<_1,_1,_1>>
class PipelineUmmaAsync {
public:
  static constexpr uint32_t Stages = Stages_;
  using AtomThrShape_MNK = AtomThrShape_MNK_;
private:
  using Impl = PipelineAsync<Stages>;
public:
  using FullBarrier  = typename Impl::FullBarrier;
  using EmptyBarrier = typename Impl::EmptyBarrier;
  using ProducerBarrierType = typename Impl::ProducerBarrierType;
  using ConsumerBarrierType = typename Impl::ConsumerBarrierType;
  using PipelineState = typename Impl::PipelineState;
  using SharedStorage = typename Impl::SharedStorage;
  using ThreadCategory = typename Impl::ThreadCategory;
  using Params = typename Impl::Params;

  // Helper function to initialize barriers
  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params) {
    int warp_idx = canonical_warp_idx_sync();
    if (warp_idx == params.initializing_warp) {
      // Barrier FULL and EMPTY init
      CUTLASS_ASSERT(params.producer_arv_count > 0 && "Producer arrival count must be non-zero");
      CUTLASS_ASSERT(params.consumer_arv_count > 0 && "Consumer arrival count must be non-zero");
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, params.producer_arv_count, params.consumer_arv_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  template <class ClusterShape>
  CUTLASS_DEVICE
  void init_masks(ClusterShape cluster_shape, dim3 block_id_in_cluster = cute::block_id_in_cluster()) {
    // Calculate producer mask
    if (params_.role == ThreadCategory::Producer) {
      // The leader threadblock executing the MMA_2x1SM instruction will signal its peer
      // threadblock when it is done with MMA operations. tmem_sync_mask encodes the
      // position of peer SMs in the cluster
      tmem_sync_mask_ = detail::calculate_umma_peer_mask(cluster_shape, AtomThrShape_MNK{}, block_id_in_cluster);
    }
  }

  // Constructor by default initializes barriers and calculates masks. 
  // These operations can be explicity deferred by specifying InitBarriers and InitMasks. 
  // If deferred, user code needs to guarantee init_masks and/or init_barriers is/are called. 
  template<class ClusterShape, class InitBarriers = cute::true_type, class InitMasks = cute::true_type>
  CUTLASS_DEVICE
  PipelineUmmaAsync(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
      : impl_(storage, params, InitBarriers{})
      , params_(params)
      , full_barrier_ptr_(&storage.full_barrier_[0])
      , empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      init_masks(cluster_shape);
    }
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
    return impl_.producer_try_acquire(state, skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    impl_.producer_acquire(state, barrier_token);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    producer_commit(state.index());
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    impl_.producer_tail(state);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return impl_.producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return impl_.consumer_try_wait(state, skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    impl_.consumer_wait(state, barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    detail::pipeline_check_is_consumer(params_.role);
    if constexpr (is_2sm_mma) {
      consumer_release_2x1SM(state.index());
    } else {
      impl_.consumer_release(state);
    }
  }

private:
  Impl impl_;
  Params params_;
  FullBarrier* full_barrier_ptr_ = nullptr;
  EmptyBarrier* empty_barrier_ptr_ = nullptr;
  uint16_t tmem_sync_mask_ = 0;
  static constexpr bool is_2sm_mma = size(AtomThrShape_MNK{}) > 1;

  CUTLASS_DEVICE
  void producer_commit(uint32_t stage) {
    detail::pipeline_check_is_producer(params_.role);
    uint64_t* smem_ptr = reinterpret_cast<uint64_t*>(&full_barrier_ptr_[stage]);
    if constexpr (is_2sm_mma) {
      cutlass::arch::umma_arrive_multicast_2x1SM(smem_ptr, tmem_sync_mask_);
    }
    else {
      cutlass::arch::umma_arrive(smem_ptr);
    }
  }

  CUTLASS_DEVICE
  void consumer_release_2x1SM(uint32_t stage) {
    detail::pipeline_check_is_consumer(params_.role);
    uint64_t* smem_ptr = reinterpret_cast<uint64_t*>(&empty_barrier_ptr_[stage]);
    cutlass::arch::umma_arrive_2x1SM_sm0(smem_ptr);
    static_assert(is_2sm_mma, "ERROR : AtomThrShape_MNK does not correspond to a 2SM MMMA");
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMA (producer) Transform (consumer) Async Pipeline
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <
  int Stages_,
  class AtomThrShape_MNK_ = Shape<_1,_1,_1>
>
class PipelineTmaTransformAsync {
public:
  static constexpr uint32_t Stages = Stages_;
  using AtomThrShape_MNK = AtomThrShape_MNK_;
private:
  using Impl = PipelineTmaAsync<Stages>;
public:
  using FullBarrier  = typename Impl::FullBarrier;
  using EmptyBarrier = typename Impl::EmptyBarrier;
  using ProducerBarrierType = typename Impl::ProducerBarrierType;
  using ConsumerBarrierType = typename Impl::ConsumerBarrierType;
  using PipelineState = typename Impl::PipelineState;
  using SharedStorage = typename Impl::SharedStorage;
  using ThreadCategory = typename Impl::ThreadCategory;
  using Params = typename Impl::Params;

  // Constructor
  template <class ClusterShape, class InitBarriers = cute::true_type, class InitMasks = cute::true_type>
  CUTLASS_DEVICE
  PipelineTmaTransformAsync(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
      : impl_(storage, params, cluster_shape, cute::false_type{}, cute::false_type{})
      , params_(params)
      , full_barrier_ptr_(&storage.full_barrier_[0])
      , empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params_, cluster_shape);
    }

    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      init_masks(cluster_shape);
    }
  }

  template<class ClusterShape, class InitBarriers = cute::true_type, class InitMasks = cute::true_type>
  CUTLASS_DEVICE
  PipelineTmaTransformAsync(SharedStorage& storage, Params params, ClusterShape cluster_shape, McastDirection mcast_direction, InitBarriers = {}, InitMasks = {})
      : impl_(storage, params, cluster_shape, cute::false_type{}, cute::false_type{})
      , params_(params)
      , empty_barrier_ptr_(&storage.empty_barrier_[0])
      , full_barrier_ptr_(&storage.full_barrier_[0]) {
    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params_, cluster_shape, mcast_direction);
    }

    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      init_masks(cluster_shape, mcast_direction);
    }
  }

  // Helper function to initialize barriers
  template <class ClusterShape>
  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params, ClusterShape cluster_shape) {
    int warp_idx = canonical_warp_idx_sync();
    if (warp_idx == params.initializing_warp) {
      // Barrier FULL and EMPTY init
      constexpr int producer_arv_cnt = 1;
      auto atom_thr_shape = AtomThrShape_MNK{};
      static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
      static_assert(IsDynamicCluster or ((cute::size<0>(cluster_shape) % cute::size<0>(atom_thr_shape) == 0) &&
                    (cute::size<1>(cluster_shape) % cute::size<1>(atom_thr_shape) == 0)));
      uint32_t const num_consumer_per_cluster = cute::ceil_div(params.num_consumers, static_cast<uint32_t>(NumThreadsPerWarpGroup));
      uint32_t const multicast_consumer_arrival_count = ((cute::size<0>(cluster_shape) / cute::size<0>(atom_thr_shape)) +
                                     (cute::size<1>(cluster_shape) / cute::size<1>(atom_thr_shape)) - 1) * num_consumer_per_cluster;
      CUTLASS_ASSERT(multicast_consumer_arrival_count > 0 && "Multicast consumer arrival count must be non-zero");
      CUTLASS_ASSERT(producer_arv_cnt > 0 && "Producer arrival count must be non-zero");
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  template <class ClusterShape>
  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params, ClusterShape cluster_shape, McastDirection mcast_direction) {
    auto atom_thr_shape = AtomThrShape_MNK{};

    int warp_idx = canonical_warp_idx_sync();
    if (warp_idx == params.initializing_warp) {
      // Barrier FULL and EMPTY init
      constexpr int producer_arv_cnt = 1;
      uint32_t const num_consumer_per_cluster = params.num_consumers / NumThreadsPerWarpGroup;
      uint32_t const multicast_consumer_arrival_count = (mcast_direction == McastDirection::kRow) ?
        (cute::size<1>(cluster_shape) / cute::size<1>(atom_thr_shape)) * num_consumer_per_cluster : // Mcast with row ctas
        (cute::size<0>(cluster_shape) / cute::size<0>(atom_thr_shape)) * num_consumer_per_cluster;  // Mcast with col ctas

      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);

    }
    cutlass::arch::fence_barrier_init();
  }

  template <class ClusterShape>
  CUTLASS_DEVICE
  void init_masks(ClusterShape cluster_shape, dim3 block_id_in_cluster = cute::block_id_in_cluster(), McastDirection mcast_dir = McastDirection::kRowCol) {
    // Calculate consumer mask
    if (params_.role == ThreadCategory::Consumer) {
      // Logic to optimally schedule Empty Arrives
      // Goal : To divide SYNCS Empty Arrival duty equally amongst the Warp-Group (128 threads)
      int warp_idx = canonical_warp_idx_sync();
      int thread_idx = threadIdx.x;
      auto cluster_size = cute::size(cluster_shape);

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
      if(mcast_dir == McastDirection::kRowCol){
        is_signaling_thread_ &= is_same_row_or_col(dst_blockid_, block_id_in_cluster, cluster_shape);
      }
      if(mcast_dir == McastDirection::kRow){
        is_signaling_thread_ &= is_same_row(dst_blockid_, block_id_in_cluster, cluster_shape);
      }
    }
  }

  template <class ClusterShape>
  CUTLASS_DEVICE
  bool is_same_row(int dst_block_id, dim3 block_id, ClusterShape cluster_shape) {
    return (((dst_block_id % cute::size<0>(cluster_shape)) == block_id.x) 
              // If we are in the same cluster column and using 2CTA MMA, only odd or only even CTAs sync with each other
                 && ((dst_block_id % cute::size<0>(cluster_shape)) % cute::size<0>(AtomThrShape_MNK{}) ==
                      block_id.x % cute::size<0>(AtomThrShape_MNK{}))
            );
  }

  template <class ClusterShape>
  CUTLASS_DEVICE
  bool is_same_row_or_col(int dst_block_id, dim3 block_id, ClusterShape cluster_shape) {
    return (((dst_block_id % cute::size<0>(cluster_shape)) == block_id.x) ||
            (
              ((dst_block_id / cute::size<0>(cluster_shape)) == block_id.y)
              // If we are in the same cluster column and using 2CTA MMA, only odd or only even CTAs sync with each other
                 && ((dst_block_id % cute::size<0>(cluster_shape)) % cute::size<0>(AtomThrShape_MNK{}) ==
                      block_id.x % cute::size<0>(AtomThrShape_MNK{}))
            ));
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    return impl_.producer_try_acquire(state, skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    impl_.producer_acquire(state, barrier_token);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state, uint32_t bytes) {
    impl_.producer_commit(state, bytes);
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    impl_.producer_tail(state);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return impl_.producer_get_barrier(state);
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return impl_.consumer_try_wait(state, skip_wait);
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(PipelineState state, uint32_t skip_wait = false) {
    return impl_.consumer_test_wait(state, skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state) {
    impl_.consumer_wait(state);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token) {
    impl_.consumer_wait(state, barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state, uint32_t skip = false) {
    detail::pipeline_check_is_consumer(params_.role);
    empty_barrier_ptr_[state.index()].arrive(dst_blockid_, is_signaling_thread_ & (!skip));
  }

private:
  Impl impl_;
  uint32_t dst_blockid_ = 0;
  uint32_t is_signaling_thread_ = 0;
  FullBarrier *full_barrier_ptr_ = nullptr;
  EmptyBarrier *empty_barrier_ptr_ = nullptr;
  Params params_;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMA (consumer) Async Pipeline classes for Blackwell UMMA
//
///////////////////////////////////////////////////////////////////////////////////////////////////

// Producer-consumer pipeline implementation
// for UMMA producer. In this case, UMMA barrier arrives are used
// by producer_commit. Use case, accumulator generation as
// the result of MMA instructions.
template <
  int Stages_,
  class ClusterShape = Shape<int,int,_1>,
  class AtomThrShape_MNK_ = Shape<_1,_1,_1>
>
class PipelineTmaUmmaAsync {
public:
  static constexpr uint32_t Stages = Stages_;
  using AtomThrShape_MNK = AtomThrShape_MNK_;
private:
  using Impl = PipelineTmaAsync<Stages>;
public:
  using FullBarrier  = typename Impl::FullBarrier;
  using EmptyBarrier = typename Impl::EmptyBarrier;
  using ProducerBarrierType = typename Impl::ProducerBarrierType;
  using ConsumerBarrierType = typename Impl::ConsumerBarrierType;
  using PipelineState = typename Impl::PipelineState;
  using SharedStorage = typename Impl::SharedStorage;
  using ThreadCategory = typename Impl::ThreadCategory;
  using Params = typename Impl::Params;

  using McastDirection = McastDirection;

  // Helper function to initialize barriers
  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params, ClusterShape cluster_shape) {
    int warp_idx = canonical_warp_idx_sync();
    if (warp_idx == params.initializing_warp) {
      // Barrier FULL and EMPTY init
      constexpr int producer_arv_cnt = 1;
      auto atom_thr_shape = AtomThrShape_MNK{};
      uint32_t const multicast_consumer_arrival_count = (cute::size<0>(cluster_shape) / cute::size<0>(atom_thr_shape)) +
                                     (cute::size<1>(cluster_shape) / cute::size<1>(atom_thr_shape)) - 1;
      CUTLASS_ASSERT(multicast_consumer_arrival_count > 0 && "Multicast consumer arrival count must be non-zero");
      CUTLASS_ASSERT(producer_arv_cnt > 0 && "Producer arrival count must be non-zero");
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params, ClusterShape cluster_shape, McastDirection mcast_direction) {
    auto atom_thr_shape = AtomThrShape_MNK{};

    int warp_idx = canonical_warp_idx_sync();
    if (warp_idx == params.initializing_warp) {
      // Barrier FULL and EMPTY init
      constexpr int producer_arv_cnt = 1;
      uint32_t const multicast_consumer_arrival_count = (mcast_direction == McastDirection::kRow) ?
        cute::size<1>(cluster_shape) / cute::size<1>(atom_thr_shape) : // Mcast with row ctas
        cute::size<0>(cluster_shape) / cute::size<0>(atom_thr_shape);  // Mcast with col ctas

      CUTLASS_ASSERT(multicast_consumer_arrival_count > 0 && "Multicast consumer arrival count must be non-zero");
      CUTLASS_ASSERT(producer_arv_cnt > 0 && "Producer arrival count must be non-zero");
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  CUTLASS_DEVICE
  void init_masks(ClusterShape cluster_shape, dim3 block_id_in_cluster = cute::block_id_in_cluster()) {
    // Calculate consumer mask
    if (params_.role == ThreadCategory::Consumer) {
      auto cluster_layout = make_layout(cluster_shape);
      block_id_mask_ = detail::calculate_multicast_mask<McastDirection::kRowCol>(cluster_shape, AtomThrShape_MNK{}, block_id_in_cluster);
    }
  }

  CUTLASS_DEVICE
  void init_masks(ClusterShape cluster_shape, McastDirection mcast_direction) {
    // Calculate consumer mask
    dim3 block_id_in_cluster = cute::block_id_in_cluster();
    auto cluster_layout = make_layout(cluster_shape);
    if (mcast_direction == McastDirection::kRow) {
      block_id_mask_ = detail::calculate_multicast_mask<McastDirection::kRow>(cluster_shape, AtomThrShape_MNK{}, block_id_in_cluster);
    }
    else {
      block_id_mask_ = detail::calculate_multicast_mask<McastDirection::kCol>(cluster_shape, AtomThrShape_MNK{}, block_id_in_cluster);
    }
  }

  // Constructor by default initializes barriers and calculates masks. 
  // These operations can be explicity deferred by specifying InitBarriers and InitMasks. 
  // If deferred, user code needs to guarantee init_masks and/or init_barriers is/are called. 
  template<typename InitBarriers = cute::true_type, typename InitMasks = cute::true_type>
  CUTLASS_DEVICE
  PipelineTmaUmmaAsync(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
      : impl_(storage, params, cluster_shape, cute::false_type{}, cute::false_type{})
      , params_(params)
      , empty_barrier_ptr_(&storage.empty_barrier_[0])
      , full_barrier_ptr_(&storage.full_barrier_[0]) {
    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params_, cluster_shape);
    }

    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      init_masks(cluster_shape);
    }
  }

  template<typename InitBarriers = cute::true_type, typename InitMasks = cute::true_type>
  CUTLASS_DEVICE
  PipelineTmaUmmaAsync(SharedStorage& storage, Params params, ClusterShape cluster_shape, McastDirection mcast_direction, InitBarriers = {}, InitMasks = {})
      : impl_(storage, params, cluster_shape, cute::false_type{}, cute::false_type{})
      , params_(params)
      , empty_barrier_ptr_(&storage.empty_barrier_[0])
      , full_barrier_ptr_(&storage.full_barrier_[0]) {
    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params_, cluster_shape, mcast_direction);
    }

    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      init_masks(cluster_shape, mcast_direction);
    }
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
    return impl_.producer_try_acquire(state, skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    impl_.producer_acquire(state, barrier_token);
  }

  CUTLASS_DEVICE
  void producer_expect_transaction(PipelineState state, uint32_t transaction_bytes) {
    impl_.producer_expect_transaction(state, transaction_bytes);
  }

  // NOP for TMA based mainloop
  CUTLASS_DEVICE
  void producer_commit(PipelineState state, uint32_t bytes) {
    impl_.producer_commit(state, bytes);
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    impl_.producer_tail(state);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return impl_.producer_get_barrier(state);
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return impl_.consumer_try_wait(state, skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    impl_.consumer_wait(state, barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index(), false);
  }

private:
  Impl impl_;
  Params params_;
  EmptyBarrier *empty_barrier_ptr_;
  FullBarrier *full_barrier_ptr_;
  uint16_t block_id_mask_ = 0;
  static constexpr bool is_2sm_mma = size(AtomThrShape_MNK{}) > 1;

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip) {
    detail::pipeline_check_is_consumer(params_.role);
    uint64_t* smem_ptr = reinterpret_cast<uint64_t*>(&empty_barrier_ptr_[stage]);
    if constexpr (is_2sm_mma) { // Mma cluster shape is 2x1
      if (!skip) {
        cutlass::arch::umma_arrive_multicast_2x1SM(smem_ptr, block_id_mask_);
      }
    }
    else {
      if (!skip) {
        if constexpr (cute::is_static_v<ClusterShape> and size(ClusterShape{}) == 1) {
          cutlass::arch::umma_arrive(smem_ptr);
        }
        else {
          cutlass::arch::umma_arrive_multicast(smem_ptr, block_id_mask_);
        }
      }
    }
  }
};

// Producer-consumer pipeline implementation
// for UMMA consumer. In this case, UMMA barrier arrives are
// used by consumer_release.
template <int Stages_, class AtomThrShape_MNK_ = Shape<_1,_1,_1>>
class PipelineUmmaConsumerAsync {
public:
  static constexpr uint32_t Stages = Stages_;
  using AtomThrShape_MNK = AtomThrShape_MNK_;
private:
  using Impl = PipelineAsync<Stages>;
public:
  using FullBarrier  = typename Impl::FullBarrier;
  using EmptyBarrier = typename Impl::EmptyBarrier;
  using ProducerBarrierType = typename Impl::ProducerBarrierType;
  using ConsumerBarrierType = typename Impl::ConsumerBarrierType;
  using PipelineState = typename Impl::PipelineState;
  using SharedStorage = typename Impl::SharedStorage;
  using ThreadCategory = typename Impl::ThreadCategory;
  using Params = typename Impl::Params;

  template <class ClusterShape>
  CUTLASS_DEVICE
  void init_masks(ClusterShape cluster_shape, dim3 block_id_in_cluster = cute::block_id_in_cluster()) {
    // Calculate consumer mask
    if (params_.role == ThreadCategory::Consumer) {
      // The leader threadblock executing the MMA_2x1SM instruction will signal its peer
      // threadblock when it is done with MMA operations. tmem_sync_mask encodes the
      // position of peer SMs in the cluster
      tmem_sync_mask_ = detail::calculate_umma_peer_mask(cluster_shape, AtomThrShape_MNK{}, block_id_in_cluster);
    }
  }

  // Constructor by default initializes barriers and calculates masks. 
  // These operations can be explicity deferred by specifying InitBarriers and InitMasks. 
  // If deferred, user code needs to guarantee init_masks and/or init_barriers is/are called. 
  template<class ClusterShape, class InitBarriers = cute::true_type, class InitMasks = cute::true_type>
  CUTLASS_DEVICE
  PipelineUmmaConsumerAsync(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
      : impl_(storage, params, InitBarriers{})
      , params_(params)
      , full_barrier_ptr_(&storage.full_barrier_[0])
      , empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      init_masks(cluster_shape);
    }
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    return impl_.producer_try_acquire(state, skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    impl_.producer_acquire(state, barrier_token);
  }

  template<class UserDefinedArriveOp>
  CUTLASS_DEVICE
  void producer_commit(PipelineState state, UserDefinedArriveOp&& user_defined_arrive_op) {
    cute::forward<UserDefinedArriveOp>(user_defined_arrive_op)(producer_get_barrier(state));
    producer_commit(state);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    if constexpr (is_2sm_mma) {
      producer_commit_2x1SM(state.index());
    } else {
      impl_.producer_commit(state);
    }
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    impl_.producer_tail(state);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return impl_.producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return impl_.consumer_try_wait(state, skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      impl_.consumer_wait(state);
    }
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

private:
  Impl impl_;
  Params params_;
  FullBarrier* full_barrier_ptr_ = nullptr;
  EmptyBarrier* empty_barrier_ptr_ = nullptr;
  uint16_t tmem_sync_mask_ = 0;
  static constexpr bool is_2sm_mma = size(AtomThrShape_MNK{}) > 1;

  CUTLASS_DEVICE
  void producer_commit_2x1SM(uint32_t stage) {
    detail::pipeline_check_is_producer(params_.role);
    uint64_t* smem_ptr = reinterpret_cast<uint64_t*>(&full_barrier_ptr_[stage]);
    cutlass::arch::umma_arrive_2x1SM_sm0(smem_ptr);
    static_assert(is_2sm_mma, "ERROR : AtomThrShape_MNK does not correspond to a 2SM MMMA");
  }

  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    detail::pipeline_check_is_consumer(params_.role);
    uint64_t* smem_ptr = reinterpret_cast<uint64_t*>(&empty_barrier_ptr_[stage]);
    if constexpr (is_2sm_mma) {
      cutlass::arch::umma_arrive_multicast_2x1SM(smem_ptr, tmem_sync_mask_);
    }
    else {
      cutlass::arch::umma_arrive(smem_ptr);
    }
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// CLC Async Pipeline class for Blackwell UMMA
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace PipelineDetail {

template<int Stages_>
using PipelineCLCFetchAsyncPipelineState = cutlass::PipelineState<Stages_>;

template<int Stages_>
struct PipelineCLCFetchAsyncSharedStorage {
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;

  FullBarrier full_barrier_[static_cast<size_t>(Stages_)];
  EmptyBarrier empty_barrier_[static_cast<size_t>(Stages_)];
};

} // namespace PipelineDetail

template <int Stages_, class ClusterShape = Shape<int,int,_1>>
class PipelineCLCFetchAsync {

public:
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = PipelineDetail::PipelineCLCFetchAsyncPipelineState<Stages>;
  using SharedStorage = PipelineDetail::PipelineCLCFetchAsyncSharedStorage<Stages>;
  using FullBarrier = typename SharedStorage::FullBarrier;
  using EmptyBarrier = typename SharedStorage::EmptyBarrier;

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
    uint32_t num_consumers = 0;
    uint32_t producer_blockid = 0;
    uint32_t producer_arv_count = 0;
    uint32_t consumer_arv_count = 0;
    int initializing_warp = 0;
  };

  // Constructor
  CUTLASS_DEVICE
  PipelineCLCFetchAsync(SharedStorage& storage, Params const& params) :
  params_(params),
  full_barrier_ptr_(&storage.full_barrier_[0]),
  empty_barrier_ptr_(&storage.empty_barrier_[0]) {
    int warp_idx = canonical_warp_idx_sync();
    if (warp_idx == params.initializing_warp) {
      // Barrier FULL and EMPTY init
      CUTLASS_ASSERT(params.producer_arv_count > 0 && "Producer arrival count must be non-zero");
      CUTLASS_ASSERT(params.consumer_arv_count > 0 && "Consumer arrival count must be non-zero");
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(full_barrier_ptr_), decltype(empty_barrier_ptr_), Stages>(
          full_barrier_ptr_, empty_barrier_ptr_, params_.producer_arv_count, params_.consumer_arv_count);
    }
    cutlass::arch::fence_barrier_init();

    cluster_size_ = []() { auto cs = cute::cluster_shape(); return cs.x * cs.y; }();
  }

  // Constructor
  CUTLASS_DEVICE
  PipelineCLCFetchAsync(SharedStorage& storage, Params const& params, ClusterShape cluster_shape)
  : params_(params)
  , full_barrier_ptr_(&storage.full_barrier_[0])
  , empty_barrier_ptr_(&storage.empty_barrier_[0]) {
    int warp_idx = canonical_warp_idx_sync();
    if (warp_idx == params.initializing_warp) {
      // Barrier FULL and EMPTY init
      CUTLASS_ASSERT(params.producer_arv_count > 0 && "Producer arrival count must be non-zero");
      CUTLASS_ASSERT(params.consumer_arv_count > 0 && "Consumer arrival count must be non-zero");
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(full_barrier_ptr_), decltype(empty_barrier_ptr_), Stages>(
          full_barrier_ptr_, empty_barrier_ptr_, params_.producer_arv_count, params_.consumer_arv_count);
    }
    cutlass::arch::fence_barrier_init();

    cluster_size_ = cute::size<0>(cluster_shape)
                  * cute::size<1>(cluster_shape)
                  * cute::size<2>(cluster_shape);
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
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  // Manual completion of transaction count
  CUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    producer_commit(state.index(), state.phase());
  }

  // Prevents early exit of producer blocks in Cluster.
  // Does NOT reset transaction bytes.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    detail::pipeline_check_is_producer(params_.role);
    for (int count = 0; count < Stages; ++count) {
      bool done = empty_barrier_ptr_[state.index()].test_wait(state.phase());
      if (!done) {
        empty_barrier_ptr_[state.index()].wait(state.phase());
      }
      ++state;
    }
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  // Consumer signalling Producer of completion
  // Notifies the producer block in the Cluster
  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

  CUTLASS_HOST_DEVICE
  uint32_t producer_get_barrier(PipelineState state) {
    return cute::cast_smem_ptr_to_uint(reinterpret_cast<void*>(&full_barrier_ptr_[state.index()]));
  }

private:
  FullBarrier *full_barrier_ptr_ = nullptr;
  EmptyBarrier *empty_barrier_ptr_ = nullptr;
  Params params_;
  int lane_idx_ = canonical_lane_idx();
  int cluster_size_;

  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_producer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_stat = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_stat)};
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);
    // 1. Wait for empty barrier to be ready
    // 2. Set the transaction bytes set to occur on the Full barrier for all blocks
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }

    full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes, lane_idx_, uint32_t(lane_idx_ < cluster_size_));
  }

  CUTLASS_DEVICE
  void producer_commit(uint32_t stage, uint32_t phase) {
    int cluster_size_ = []() { auto cs = cute::cluster_shape(); return cs.x * cs.y; }();
    full_barrier_ptr_[stage].complete_transaction(lane_idx_, params_.transaction_bytes,  uint32_t(lane_idx_ < cluster_size_));
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_stat = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_stat)};
  }

  // Wait for producer to commit transactions
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
    empty_barrier_ptr_[stage].arrive(params_.producer_blockid);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Empty Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////

class PipelineEmpty {
public:
  static constexpr uint32_t Stages = 0;
  using PipelineState = cutlass::PipelineState<0>;
  struct Params {};
  struct SharedStorage {};

  // Constructor
  CUTLASS_DEVICE
  PipelineEmpty(SharedStorage& storage, Params const& params) {}

  // Constructor
  CUTLASS_DEVICE
  PipelineEmpty(SharedStorage&& storage, Params const& params) {}

  // Constructor with throwaway ClusterShape
  template <class ClusterShape = Shape<int,int,_1>>
  CUTLASS_DEVICE
  PipelineEmpty(SharedStorage&& storage, Params const& params, ClusterShape) {}

 CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state) {
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMA (producer - consumer) Async Pipeline classes for Blackwell Sparse UMMA
// This is designed for the pattern that kernel has two different staged tensors. (AB and metadata)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

// Producer-consumer pipeline implementation
// for UMMA producer. In this case, UMMA barrier arrives are used
// by producer_commit. Use case, accumulator generation as
// the result of MMA instructions.
template <
  int Stages_,
  class ClusterShape = Shape<int,int,_1>,
  class AtomThrShape_MNK_ = Shape<_1,_1,_1>
>
class PipelineTmaSparseUmmaAsync {
public:
  static constexpr uint32_t Stages = Stages_;
  using AtomThrShape_MNK = AtomThrShape_MNK_;
private:
  using Impl = PipelineTmaUmmaAsync<Stages, ClusterShape, AtomThrShape_MNK>;
public:
  using FullBarrier  = typename Impl::FullBarrier;
  using EmptyBarrier = typename Impl::EmptyBarrier;
  using ProducerBarrierType = typename Impl::ProducerBarrierType;
  using ConsumerBarrierType = typename Impl::ConsumerBarrierType;
  using PipelineState = typename Impl::PipelineState;
  using SharedStorage = typename Impl::SharedStorage;
  using ThreadCategory = typename Impl::ThreadCategory;
  using Params = typename Impl::Params;

  struct ParamsMetadata {
    uint32_t transaction_bytes = 0;
    uint32_t metadata_transaction_bytes = 0;
  };

  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params, ClusterShape cluster_shape) {
    Impl::init_barriers(storage, params, cluster_shape);
  }

  CUTLASS_DEVICE
  void init_masks(ClusterShape cluster_shape, dim3 block_id_in_cluster = cute::block_id_in_cluster()) {
    impl_.init_masks(cluster_shape, block_id_in_cluster);
  }

  // Constructor by default initializes barriers and calculates masks. 
  // These operations can be deferred by specifying InitBarriers and InitMasks. 
  // If deferred, user code needs to guarantee init_masks and/or init_barriers is/are called. 
  template<typename InitBarriers = cute::true_type, typename InitMasks = cute::true_type>
  CUTLASS_DEVICE
  PipelineTmaSparseUmmaAsync(SharedStorage& storage, Params params, ParamsMetadata params_metadata, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
      : impl_(storage, params, cluster_shape, cute::false_type{}, cute::false_type{})
      , params_(params)
      , params_metadata_(params_metadata)
      , empty_barrier_ptr_(&storage.empty_barrier_[0])
      , full_barrier_ptr_(&storage.full_barrier_[0]) {
    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params_, cluster_shape);
    }

    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      init_masks(cluster_shape);
    }
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
    return impl_.producer_try_acquire(state, skip_wait);
  }

  // Customized for metadata load
  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, bool load_e, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), load_e, barrier_token);
  }

  // Customized for metadata load
  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state, true, barrier_token);
  }

  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    return impl_.producer_tail(state);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return impl_.producer_get_barrier(state);
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return impl_.consumer_try_wait(state, skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    return impl_.consumer_wait(state, barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    return impl_.consumer_release(state);
  }

private:
  Impl impl_;
  Params params_;
  ParamsMetadata params_metadata_;
  EmptyBarrier *empty_barrier_ptr_{nullptr};
  FullBarrier *full_barrier_ptr_{nullptr};

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, bool load_e, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }
    uint32_t bytes_now = load_e ? params_metadata_.transaction_bytes + params_metadata_.metadata_transaction_bytes : params_metadata_.transaction_bytes;

    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_expect_tx(bytes_now);
    }
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
