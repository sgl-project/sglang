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

namespace cutlass::gemm::collective::detail {

template<
  class ClusterShape_MNK,
  int AccumulatorPipelineStageCount,
  int SchedulerPipelineStageCount,
  int CLCResponseSize,
  bool IsArrayOfPointersGemm,
  int NumTensorMaps=2
>
struct Sm100DenseGemmTmaUmmaCarveout {
  // AccumulatorPipeline = PipelineUmmaAsync
  static constexpr auto AccumulatorPipelineStorage = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage);
  // CLCPipeline = PipelineCLCFetchAsync
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage);
  // LoadOrderBarrier = OrderedSequenceBarrier<1,2>
  static constexpr auto LoadOrderBarrierStorage = sizeof(typename cutlass::OrderedSequenceBarrier<1,2>::SharedStorage);
  // CLC (scheduler) response
  static constexpr auto CLCResponseStorage = SchedulerPipelineStageCount * detail::CLCResponseSize;
  // CLC Throttle pipeline storage
  static constexpr auto CLCThrottlePipelineStorage = sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);
  // Tmem dealloc
  static constexpr auto TmemDeallocStorage = sizeof(cutlass::arch::ClusterBarrier);
  // Tmem ptr storage
  static constexpr auto TmemBasePtrsStorage = SchedulerPipelineStageCount * sizeof(uint32_t);
  // Tensormap Storage
  static constexpr auto TensorMapStorage = 
    IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * NumTensorMaps /* for A and B */ :
    0;
  // Smem usage that's not part of CollectiveEpilogue::SharedStorage & CollectiveMainloop::SharedStorage
  static constexpr auto KernelSmemCarveout = static_cast<int>( AccumulatorPipelineStorage +
                                                               CLCPipelineStorage +
                                                               LoadOrderBarrierStorage +
                                                               TmemDeallocStorage +
                                                               CLCThrottlePipelineStorage +
                                                               CLCResponseStorage +
                                                               TmemBasePtrsStorage +
                                                               TensorMapStorage
                                                              );
};

template<class ClusterShape_MNK, int AccumulatorPipelineStageCount, int SchedulerPipelineStageCount, int CLCResponseSize>
struct Sm100SparseGemmTmaUmmaCarveout {

  // * GemmUniversal::SharedStorage::PipelineStorage
  // LoadOrderBarrier = OrderedSequenceBarrier<1,2>
  static constexpr auto LoadOrderBarrierStorage = sizeof(typename cutlass::OrderedSequenceBarrier<1,2>::SharedStorage);
  // CLCPipelineStorage = PipelineCLCFetchAsync
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage);
  // AccumulatorPipeline = PipelineUmmaAsync
  static constexpr auto AccumulatorPipelineStorage = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage);
  // CLC Throttle pipeline storage
  static constexpr auto CLCThrottlePipelineStorage = sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);
  // Tmem dealloc
  static constexpr auto TmemDeallocStorage = sizeof(cutlass::arch::ClusterBarrier);

  static constexpr auto PipelineStorage = static_cast<int>(cutlass::round_up(
                                                      cutlass::round_up(LoadOrderBarrierStorage, 16) +
                                                      cutlass::round_up(CLCPipelineStorage, 16) +
                                                      cutlass::round_up(AccumulatorPipelineStorage, 16) +
                                                      cutlass::round_up(CLCThrottlePipelineStorage, 16) +
                                                      cutlass::round_up(TmemDeallocStorage, 16),
                                                    16));

  // * GemmUniversal::SharedStorage::Others
  // CLC (scheduler) response
  static constexpr auto CLCQueryResponseStorage = SchedulerPipelineStageCount * CLCResponseSize;
  // Tmem ptr storage
  static constexpr auto TmemBasePtrsStorage = sizeof(uint32_t);

  static constexpr auto OtherStorage = static_cast<int>(cutlass::round_up(
                                                   cutlass::round_up(CLCQueryResponseStorage, 16) +
                                                   cutlass::round_up(TmemBasePtrsStorage, 16),
                                                 16));
  
  // Smem usage that's not part of CollectiveEpilogue::SharedStorage & CollectiveMainloop::SharedStorage
  static constexpr auto KernelSmemCarveout = static_cast<int>( PipelineStorage +
                                                               OtherStorage);
};
} // namespace cutlass::gemm::collective::detail
