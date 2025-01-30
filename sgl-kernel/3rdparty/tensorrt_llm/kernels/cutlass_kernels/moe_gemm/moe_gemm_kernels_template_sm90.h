/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Ignore CUTLASS warnings about type punning
#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif          // __GNUC__

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"

#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"

#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_launcher_sm90.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_sm90_traits.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

namespace tensorrt_llm
{
using EpilogueFusion = HopperGroupedGemmInput::EpilogueFusion;

template <typename T, typename WeightType, typename OutputType, typename EpilogueTag, EpilogueFusion FUSION,
    typename TileShape, typename ClusterShape>
void dispatchMoeGemmSelectBiasSM90(HopperGroupedGemmInput hopper_input, int num_experts, int multi_processor_count,
    cudaStream_t stream, int* occupancy, size_t* workspace_size)
{
    static_assert(kernels::cutlass_kernels::isValidHopperMOESpecialisation<T, WeightType, EpilogueTag>(),
        "Invalid hopper configuration invoked, fallback to Sm80");

    TLLM_CHECK_WITH_INFO(
        workspace_size || hopper_input.isValid(), "Hopper specialisation is missing additional input information");

    //            auto func = hopper_input.ptr_c ?
    //            kernels::cutlass_kernels::genericMoeGemmKernelLauncherHopper<T, WeightType,
    //                            cutlass::arch::Sm90, EpilogueTag, true>
    //                                           :
    //                                           kernels::cutlass_kernels::genericMoeGemmKernelLauncherHopper<T,
    //                                           WeightType,
    //                                               cutlass::arch::Sm90, EpilogueTag, false>;
    // TODO(dastokes) Re-enable bias when CUTLASS supports it
    auto func = kernels::cutlass_kernels::sm90_generic_moe_gemm_kernelLauncher<T, WeightType, OutputType, EpilogueTag,
        FUSION, TileShape, ClusterShape, false>;
    func(hopper_input, num_experts, multi_processor_count, stream, occupancy, workspace_size);
}

/*
    1x1x1 cluster shape is are supported for any tile shape.

    2x1x1 cluster shape is only supported for when the M tile is at least 128.

    1x2x1 cluster shape is only supported when the N tile is at least 128.

    2x2x1 cluster shape is only supported when both the M and N tiles are at least 128.

    We make the above restrictions are to improve compilation speed in TRT-LLM by pruning kernels
    that may not be very useful in practice.
 */
template <typename CTAShape, typename ClusterShape>
constexpr bool are_tile_shapes_supported()
{
    using namespace cute;
    [[maybe_unused]] constexpr int cta_m = get<0>(CTAShape{});
    [[maybe_unused]] constexpr int cta_n = get<1>(CTAShape{});
    constexpr int cga_m = get<0>(ClusterShape{});
    constexpr int cga_n = get<1>(ClusterShape{});

    if constexpr (cga_m == _1{} && cga_n == _1{})
    {
        return true;
    }
    else if constexpr (cga_m == _2{} && cga_n == _1{} && cta_m >= _128{})
    {
        return true;
    }
    else if constexpr (cga_m == _1{} && cga_n == _2{} && cta_n >= _128{})
    {
        return true;
    }
    else if constexpr (cga_m == _2{} && cga_n == _2{} && cta_m >= _128{} && cta_n >= _128{})
    {
        return true;
    }
    else
    {
        return false;
    }
}

template <typename T, typename WeightType, typename OutputType, typename EpilogueTag, EpilogueFusion FUSION,
    typename TileShape>
void dispatchMoeGemmSelectClusterShapeSM90(HopperGroupedGemmInput hopper_input, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, cudaStream_t stream, int* occupancy,
    size_t* workspace_size)
{
    using namespace cute;
    switch (gemm_config.cluster_shape)
    {
#define SHAPE_CASE(M, N, K)                                                                                            \
    case cutlass_extensions::ClusterShape::ClusterShape_##M##x##N##x##K:                                               \
    {                                                                                                                  \
        using ClusterShape = Shape<_##M, _##N, _##K>;                                                                  \
        if constexpr (are_tile_shapes_supported<TileShape, ClusterShape>())                                            \
        {                                                                                                              \
            dispatchMoeGemmSelectBiasSM90<T, WeightType, OutputType, EpilogueTag, FUSION, TileShape, ClusterShape>(    \
                hopper_input, num_experts, multi_processor_count, stream, occupancy, workspace_size);                  \
            break;                                                                                                     \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_THROW("Unsupported tile and cluster shape combination");                                              \
        }                                                                                                              \
    }

        SHAPE_CASE(1, 1, 1)
        SHAPE_CASE(1, 2, 1)

        SHAPE_CASE(2, 1, 1)
        SHAPE_CASE(2, 2, 1)

#undef SHAPE_CASE
    default: TLLM_THROW("Unsupported config for MoE gemm.");
    }
} // namespace tensorrt_llm

template <typename T, typename WeightType, typename OutputType, typename EpilogueTag, EpilogueFusion FUSION>
void dispatchMoeGemmSelectTileShapeSM90(HopperGroupedGemmInput hopper_input, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, cudaStream_t stream, int* occupancy,
    size_t* workspace_size)
{
    using namespace cute;

    switch (gemm_config.tile_config_sm90)
    {
#define SHAPE_CASE(M, N, K)                                                                                            \
    case cutlass_extensions::CutlassTileConfigSM90::CtaShape##M##x##N##x##K##B:                                        \
    {                                                                                                                  \
        constexpr int KtileBytes = K / sizeof(T);                                                                      \
        using KTileDim = Int<KtileBytes>;                                                                              \
        using TileShape = Shape<_##M, _##N, KTileDim>;                                                                 \
        dispatchMoeGemmSelectClusterShapeSM90<T, WeightType, OutputType, EpilogueTag, FUSION, TileShape>(              \
            hopper_input, num_experts, gemm_config, multi_processor_count, stream, occupancy, workspace_size);         \
        break;                                                                                                         \
    }

        SHAPE_CASE(128, 16, 128)
        SHAPE_CASE(128, 32, 128)
        SHAPE_CASE(128, 64, 128)
        SHAPE_CASE(128, 128, 128)
        SHAPE_CASE(128, 256, 128)
        SHAPE_CASE(256, 128, 128)

#undef SHAPE_CASE
    case cutlass_extensions::CutlassTileConfigSM90::Undefined: TLLM_THROW("GEMM config undefined."); break;
    case cutlass_extensions::CutlassTileConfigSM90::ChooseWithHeuristic:
        TLLM_THROW("GEMM config should have already been set by heuristic.");
        break;
    default: TLLM_THROW("Unsupported config for MoE gemm."); break;
    }
}

template <typename T, typename WeightType, typename OutputType, EpilogueFusion FUSION>
size_t calcMaxWorkspaceSizeSM90(
    int num_experts, cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count)
{
    size_t count;
    // Most of the values are ignored for WS size calculation. We reuse the function to reduce the template bloat
    dispatchMoeGemmSelectTileShapeSM90<T, WeightType, OutputType, cutlass_extensions::EpilogueOpDefault, FUSION>(
        HopperGroupedGemmInput{}, num_experts, gemm_config, multi_processor_count, cudaStream_t{0}, nullptr, &count);
    return count;
}

} // namespace tensorrt_llm
