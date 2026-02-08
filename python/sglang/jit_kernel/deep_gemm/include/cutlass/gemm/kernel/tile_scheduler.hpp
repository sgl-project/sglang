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

/*! \file
    \brief Utilities for selecting default tile schedulers
*/

#include "cutlass/arch/arch.h"
#include "cutlass/detail/dependent_false.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {

//
// Tags for specifying tile schedulers
//

struct PersistentScheduler { };

struct StreamKScheduler { };

struct GroupScheduler { }; // Only used for Grouped GEMMs

struct DynamicPersistentScheduler { };

struct StaticPersistentScheduler { };

} // namespace cutlass::gemm
////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm100_static_tile_scheduler.hpp" 

#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"            
#include "cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp"   
#include "cutlass/gemm/kernel/sm100_tile_scheduler_group.hpp"      
////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel::detail {

//
// Selectors mapping tile scheduler tag and arch tag to a tile scheduler class
//

template <
  class TileSchedulerTag,
  class ArchTag,
  class TileShape,
  class ClusterShape
  , uint32_t SchedulerPipelineStageCount = 2 
  , class ProblemShapeType = void
>
struct TileSchedulerSelector {
  static_assert(cutlass::detail::dependent_false<ArchTag>,
      "Could not select a tile scheduler for given parameters.");
};

template <
  class ArchTag,
  class TileShape,
  class ClusterShape
  , uint32_t SchedulerPipelineStageCount     
>
struct TileSchedulerSelector<
    PersistentScheduler,
    ArchTag,
    TileShape,
    ClusterShape
    , SchedulerPipelineStageCount              
  > {
  using Scheduler = PersistentTileSchedulerSm90;
};

// Default (void) for Sm90 maps to PersistentTileSchedulerSm90
template <
  class ArchTag,
  class TileShape,
  class ClusterShape
  , uint32_t SchedulerPipelineStageCount     
>
struct TileSchedulerSelector<
    void,
    ArchTag,
    TileShape,
    ClusterShape
    , SchedulerPipelineStageCount              
  > {
  using Scheduler = typename TileSchedulerSelector<
      PersistentScheduler,
      ArchTag,
      TileShape,
      ClusterShape
      , SchedulerPipelineStageCount            
  >::Scheduler;
};

template <
  class TileShape,
  class ClusterShape
  , uint32_t SchedulerPipelineStageCount     
>
struct TileSchedulerSelector<
    StreamKScheduler,
    arch::Sm90,
    TileShape,
    ClusterShape
    , SchedulerPipelineStageCount              
  > {
  using Scheduler = PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>;
};

template <
  class ArchTag,
  class TileShape,
  class ClusterShape, 
  uint32_t SchedulerPipelineStageCount     
>
struct TileSchedulerSelector<
    StaticPersistentScheduler,
    ArchTag,
    TileShape,
    ClusterShape
    , SchedulerPipelineStageCount              
  > {
  using Scheduler = PersistentTileSchedulerSm90;
};

template <
  class TileShape,
  class ClusterShape, 
  uint32_t SchedulerPipelineStageCount, 
  class GroupProblemShape
>
struct TileSchedulerSelector<
    GroupScheduler,
    arch::Sm90,
    TileShape,
    ClusterShape
    , SchedulerPipelineStageCount              
    , GroupProblemShape
  > {
  using Scheduler = PersistentTileSchedulerSm90Group<GroupProblemShape, SchedulerPipelineStageCount>;
};

template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    PersistentScheduler,
    arch::Sm100,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount> {
  using Scheduler = PersistentTileSchedulerSm100<
                        ClusterShape,
                        SchedulerPipelineStageCount>;
};

// Ptr-Array kernel may provide a specialized ArrayProblemShape type
template <class TileShape,
  class ClusterShape,
  uint32_t SchedulerPipelineStageCount,
  class ProblemShape>
struct TileSchedulerSelector<
    PersistentScheduler,
    arch::Sm100,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount,
    ProblemShape> {
  using Scheduler = PersistentTileSchedulerSm100<
                        ClusterShape,
                        SchedulerPipelineStageCount>;
};

// Default (void) for Sm100 maps to PersistentTileSchedulerSm100
template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    void,
    arch::Sm100,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount> {
    using Scheduler = PersistentTileSchedulerSm100<
                ClusterShape,
                SchedulerPipelineStageCount
                >;
};

// Default (void) for Sm100 maps to PersistentTileSchedulerSm100
// Ptr-Array kernel may provide a specialized ArrayProblemShape type
template <class TileShape,
  class ClusterShape,
  uint32_t SchedulerPipelineStageCount,
  class ProblemShape>
struct TileSchedulerSelector<
    void,
    arch::Sm100,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount,
    ProblemShape> {
  using Scheduler = typename TileSchedulerSelector<
      PersistentScheduler,
      arch::Sm100,
      TileShape,
      ClusterShape,
      SchedulerPipelineStageCount>::Scheduler;
};

// SM100 Group tile scheduler
template <
  class TileShape,
  class ClusterShape,
  uint32_t SchedulerPipelineStageCount,
  class GroupProblemShape
>
struct TileSchedulerSelector<
    GroupScheduler,
    arch::Sm100,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount,
    GroupProblemShape
  > {
  using Scheduler = PersistentTileSchedulerSm100Group<GroupProblemShape, SchedulerPipelineStageCount>;
};

// SM100 stream-K scheduler
template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    StreamKScheduler,
    arch::Sm100,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount> {
  using Scheduler = PersistentTileSchedulerSm100StreamK<
                        TileShape,
                        ClusterShape,
                        SchedulerPipelineStageCount>;
};

// SM100 dynamic tile scheduler
template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    DynamicPersistentScheduler,
    arch::Sm100,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount> {
  using Scheduler = PersistentTileSchedulerSm100<
                        ClusterShape,
                        SchedulerPipelineStageCount>;
};

template <
  class TileShape,
  class ClusterShape,
  uint32_t SchedulerPipelineStageCount
>
struct TileSchedulerSelector<
    StaticPersistentScheduler,
    arch::Sm100,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount> {
  using Scheduler = StaticPersistentTileScheduler100;
};

template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    PersistentScheduler,
    arch::Sm103,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount> {
  using Scheduler = PersistentTileSchedulerSm100<
                        ClusterShape,
                        SchedulerPipelineStageCount>;
};

// Ptr-Array kernel may provide a specialized ArrayProblemShape type
template <class TileShape,
  class ClusterShape,
  uint32_t SchedulerPipelineStageCount,
  class ProblemShape>
struct TileSchedulerSelector<
    PersistentScheduler,
    arch::Sm103,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount,
    ProblemShape> {
  using Scheduler = PersistentTileSchedulerSm100<
                        ClusterShape,
                        SchedulerPipelineStageCount>;
};

// SM103 Group tile scheduler
template <
  class TileShape,
  class ClusterShape,
  uint32_t SchedulerPipelineStageCount,
  class GroupProblemShape
>
struct TileSchedulerSelector<
    GroupScheduler,
    arch::Sm103,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount,
    GroupProblemShape
  > {
  using Scheduler = PersistentTileSchedulerSm100Group<GroupProblemShape, SchedulerPipelineStageCount>;
};

template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    StreamKScheduler,
    arch::Sm103,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount> {
  using Scheduler = PersistentTileSchedulerSm100StreamK<
                        TileShape,
                        ClusterShape,
                        SchedulerPipelineStageCount>;
};

// Default (void) for Sm120 maps to PersistentTileSchedulerSm100
template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    void,
    arch::Sm120,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount> {
    using Scheduler = PersistentTileSchedulerSm100<
                ClusterShape,
                SchedulerPipelineStageCount
                >;
};

// PersistentScheduler for Sm120 maps to PersistentTileSchedulerSm100
template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    PersistentScheduler,
    arch::Sm120,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount> {
  using Scheduler = PersistentTileSchedulerSm100<ClusterShape, SchedulerPipelineStageCount>;
};


// StreamKScheduler for Sm120 maps to PersistentTileSchedulerSm100StreamK
template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    StreamKScheduler,
    arch::Sm120,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount> {
  using Scheduler = PersistentTileSchedulerSm100StreamK<
                        TileShape,
                        ClusterShape,
                        SchedulerPipelineStageCount>;
};

// SM120 Group tile scheduler
template <
  class TileShape,
  class ClusterShape, 
  uint32_t SchedulerPipelineStageCount, 
  class GroupProblemShape
>
struct TileSchedulerSelector<
    GroupScheduler,
    arch::Sm120,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount,
    GroupProblemShape
  > {
  using Scheduler = PersistentTileSchedulerSm90Group<GroupProblemShape, SchedulerPipelineStageCount>;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel::detail

////////////////////////////////////////////////////////////////////////////////
