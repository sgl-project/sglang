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

#pragma once

#include "kda/sm90/collective/mainloop_kda_fwd.hpp"
#include "kda/sm90/kernel/kernel_kda_fwd.hpp"
#include "kda/sm90/kernel/options.hpp"
#include "kda/sm90/kernel/tile_scheduler.hpp"
#include "kda/sm90/utils/type_traits.hpp"

namespace kda::sm90::kernel {

template <
    class Element_,
    class ElementAccumulatorQK_,
    class ElementAccumulatorPV_,
    class TileShape_,  // BlkSeqQO, BlkSeqKV, HeadSize
    class LayoutQ_,
    class LayoutK_,
    class LayoutV_,
    class LayoutO_,
    class DispatchPolicy,
    class Options = DefaultOptions>
struct FlatBuilderKdaFwd;

template <
    class Element,
    class ElementAccumulatorQK,
    class ElementAccumulatorPV,
    class TileShape,  // BlkSeqQO, BlkSeqKV, HeadSize
    class LayoutQ,
    class LayoutK,
    class LayoutV,
    class LayoutO,
    class Options>
struct FlatBuilderKdaFwd<
    Element,
    ElementAccumulatorQK,
    ElementAccumulatorPV,
    TileShape,
    LayoutQ,
    LayoutK,
    LayoutV,
    LayoutO,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    Options> {
  using CollectiveMainloop = kda::sm90::collective::FlatMainloopTmaWarpSpecializedKdaFwd<
      Element,
      ElementAccumulatorQK,
      ElementAccumulatorPV,
      TileShape,
      LayoutQ,
      LayoutK,
      LayoutV,
      LayoutO,
      Options>;

  static constexpr bool kIsPersistent = find_option_t<Tag::kIsPersistent, false_type, Options>::value;
  static_assert(!kIsPersistent, "not implemented");

  using TileScheduler = kda::sm90::kernel::IndividualTileScheduler;
  // using TileScheduler = std::conditional_t<kIsPersistent, kda::sm90::kernel::PersistentTileScheduler,
  // kda::sm90::kernel::IndividualTileScheduler>;

  using Kernel = kda::sm90::kernel::FlatKernelTmaWarpSpecializedKdaFwd<CollectiveMainloop, TileScheduler, Options>;
};

}  // namespace kda::sm90::kernel
