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

#include "cute/layout.hpp"     // cute::Shape
#include "cute/numeric/numeric_types.hpp" // cute::sizeof_bits_v
#include "cutlass/arch/mma.h"  // cutlass::arch::OpClassTensorOp, cutlass::OpClassSparseTensorOp
#include "cute/atom/copy_traits_sm100.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/util/type_traits.hpp" // cute::is_same_v

#include "cutlass/detail/dependent_false.hpp" // cutlass::detail::dependent_false
#include "cutlass/detail/layout.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/epilogue/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp"

namespace cutlass::epilogue::collective {
// Alias to sm100 builder
template <
  class OpClass,
  class MmaTileShape_MNK,    // Static MMA tile shape
  class ClusterShape_MNK,    // Static cluster shape or dynamic (int, int, _1)
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
  class FusionOp
>
struct CollectiveBuilder<
    arch::Sm103,
    OpClass,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleType,
    FusionOp 
>
{
  using CollectiveOp = typename CollectiveBuilder<
                          arch::Sm100,
                          OpClass,
                          MmaTileShape_MNK,
                          ClusterShape_MNK,
                          EpilogueTileType,
                          ElementAccumulator,
                          ElementCompute,
                          ElementC,
                          GmemLayoutTagC,
                          AlignmentC,
                          ElementD,
                          GmemLayoutTagD,
                          AlignmentD,
                          EpilogueScheduleType,
                          FusionOp
                        >::CollectiveOp;
};

} // namespace cutlass::epilogue::collective
