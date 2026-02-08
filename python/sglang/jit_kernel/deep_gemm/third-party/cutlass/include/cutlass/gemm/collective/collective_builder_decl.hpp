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

#include <cute/numeric/integral_constant.hpp>
#include <cutlass/detail/dependent_false.hpp>

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Used to specify stage counts or dispatch to automatic computation of stage count
template<int num_stages>
struct StageCount {
  static constexpr int value = num_stages;

  StageCount() = default;
  explicit StageCount(cute::Int<num_stages>) {}
};

template<int carveout_bytes>
struct StageCountAutoCarveout {
  static constexpr int bytes = carveout_bytes;

  StageCountAutoCarveout() = default;
  explicit StageCountAutoCarveout(cute::Int<carveout_bytes>) {}
};

namespace detail {

// Forward Declaration
template<class CollectiveEpilogue>
constexpr int
compute_carveout_from_epi();

} // namespace detail

template<class CollectiveEpilogue>
struct StageCountAutoCarveoutEpi : StageCountAutoCarveout<detail::compute_carveout_from_epi<CollectiveEpilogue>()> {};

using StageCountAuto = StageCountAutoCarveout<0>;

// Used to automatically let the builder pick the kernel schedule.
// Can be overridden with kernel schedule tags in cutlass/gemm/dispatch_policy.hpp
struct KernelScheduleAuto final {};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ArchTag,
  class OpClass,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType,
  class Enable = void
>
struct CollectiveBuilder {
  static_assert(sizeof(ElementA) == 0, "Could not build a collective for given parameters.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

