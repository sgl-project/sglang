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

#include <cute/arch/copy.hpp>         // cute::DefaultCopy
#include <cute/util/type_traits.hpp>  // cute::is_base_of_v

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Used to specify epilogue subtile shape or dispatch to automatic computation of subtile shape
struct EpilogueTileAuto {};

// Used to let the builder pick the epilogue schedule automatically.
// Can be overridden with kernel schedule tags in cutlass/gemm/dispatch_policy.hpp
struct EpilogueScheduleAuto {};

template <
  class ArchTag,
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class EpilogueScheduleType,
  class FusionOpOrCallbacks = cutlass::epilogue::fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>,
  class Enable = void
>
struct CollectiveBuilder {
  static_assert(cutlass::detail::dependent_false<ArchTag>,
      "Could not build a collective epilogue for given parameters.");
};

// helper sub-builder for epilogue fusion callbacks (for internal use by CollectiveBuilder only)
namespace detail {

// callbacks builder with operation tag
template<
  class DispatchPolicy,
  class FusionOp,
  class TileShape_MNK,
  class EpilogueTile_MN,
  class ElementAccumulator,
  class AccLoadOp = cute::DefaultCopy,
  class = void
>
struct CallbacksBuilder {
  using Callbacks = fusion::FusionCallbacks<DispatchPolicy, FusionOp, TileShape_MNK, EpilogueTile_MN>;
};

// callbacks builder with callbacks passthrough
template <
  class DispatchPolicy,
  class FusionCallbacks,
  class TileShape_MNK,
  class EpilogueTile_MN,
  class AccLoadOp,
  class ElementAccumulator
>
struct CallbacksBuilder<
  DispatchPolicy,
  FusionCallbacks,
  TileShape_MNK,
  EpilogueTile_MN,
  ElementAccumulator,
  AccLoadOp,
  cute::enable_if_t<not cute::is_base_of_v<fusion::FusionOperation, FusionCallbacks>>
> {
  using Callbacks = FusionCallbacks;
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "builders/sm90_builder.inl"
#include "builders/sm100_builder.inl"  
#include "builders/sm103_builder.inl"
#include "builders/sm120_builder.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////
