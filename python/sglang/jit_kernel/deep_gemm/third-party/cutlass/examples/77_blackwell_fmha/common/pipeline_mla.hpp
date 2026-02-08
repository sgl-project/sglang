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
/*!
  \file
  \brief Support the producer to acquire specific bytes of data.
*/

#pragma once

#include "cutlass/pipeline/sm100_pipeline.hpp"

namespace cutlass {

using namespace cute;

template <
  int Stages_,
  class ClusterShape = Shape<int,int,_1>,
  class AtomThrShape_MNK_ = Shape<_1,_1,_1>
>
class PipelineTmaAsyncMla {

public:
  static constexpr uint32_t Stages = Stages_;
  using AtomThrShape_MNK = AtomThrShape_MNK_;

private:
  using Impl = PipelineTmaUmmaAsync<Stages_, ClusterShape, AtomThrShape_MNK_>;

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


public:
  template<typename InitBarriers = cute::true_type, typename InitMasks = cute::true_type>
  CUTLASS_DEVICE
  PipelineTmaAsyncMla(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
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
  PipelineTmaAsyncMla(SharedStorage& storage, Params params, ClusterShape cluster_shape, McastDirection mcast_direction, InitBarriers = {}, InitMasks = {})
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


  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    impl_.producer_acquire(state, barrier_token);
  }

  CUTLASS_DEVICE
  void producer_acquire_bytes(uint32_t stage, uint32_t bytes, uint32_t phase, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);
    if (barrier_token != BarrierStatus::WaitDone) {
      empty_barrier_ptr_[stage].wait(phase);
    }

    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_expect_tx(bytes);
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
  void producer_acquire_bytes(PipelineState state, uint32_t bytes, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire_bytes(state.index(), bytes, state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return impl_.producer_get_barrier(state);
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

}
