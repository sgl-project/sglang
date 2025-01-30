/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm_configs.h"
#include <array>
#include <cuda_runtime_api.h>
#include <optional>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/layout/layout.h"

namespace tensorrt_llm
{
template <class T>
constexpr auto transpose_stride(T const& t)
{
    return cute::prepend(cute::prepend(cute::take<2, cute::rank_v<T>>(t), cute::get<0>(t)), cute::get<1>(t));
}

struct HopperGroupedGemmInput
{
    template <class T>
    using TransposeStride = decltype(transpose_stride<T>(T{}));
    template <class Tag>
    using TransposeLayoutTag = std::conditional_t<std::is_same_v<Tag, cutlass::layout::RowMajor>,
        cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;

    static_assert(std::is_same_v<cutlass::layout::RowMajor, TransposeLayoutTag<cutlass::layout::ColumnMajor>>);
    static_assert(std::is_same_v<cutlass::layout::ColumnMajor, TransposeLayoutTag<cutlass::layout::RowMajor>>);

    // Layout for A and B is transposed and then swapped in the implementation
    // This uses B^T * A^T = (A * B)^T to get a better layout for the GEMM
    using LayoutA = TransposeLayoutTag<cutlass::layout::RowMajor>;    // Layout type for A matrix operand
    using LayoutB = TransposeLayoutTag<cutlass::layout::ColumnMajor>; // Layout type for B matrix operand
    using LayoutC = TransposeLayoutTag<cutlass::layout::RowMajor>;    // Layout type for C matrix operand

    using StrideA
        = std::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutA*>>; // Use B because they will be swapped
    using StrideB
        = std::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutB*>>; // Use A because they will be swapped
    using StrideC = std::remove_pointer_t<cutlass::detail::TagToStrideC_t<LayoutC*>>;

    template <class T>
    constexpr static bool IsFP8_v = std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;

    // Currently this should always just be T
    template <class T>
    using OutputTypeAdaptor_t = std::conditional_t<IsFP8_v<T>, nv_bfloat16, T>;

    using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int64_t, int64_t, int64_t>>;

    ProblemShape shape_info{};
    StrideA* stride_a = nullptr;
    StrideB* stride_b = nullptr;

    void const** ptr_a = nullptr;
    void const** ptr_b = nullptr;

    // C is currently the same in both epilogues
    StrideC* stride_c = nullptr;
    void const** ptr_c = nullptr;

    struct DefaultEpilogue
    {
        using LayoutD = TransposeLayoutTag<cutlass::layout::RowMajor>; // Layout type for D matrix operand
        using StrideD = std::remove_pointer_t<cutlass::detail::TagToStrideC_t<LayoutD*>>;

        StrideD* stride_d = nullptr;
        void** ptr_d = nullptr;
    };

    struct FusedFinalizeEpilogue
    {
        using StrideFinalOutput = DefaultEpilogue::StrideD;
        using StrideBias = TransposeStride<cute::Stride<cute::_0, cute::_1, int>>;
        using StrideRouterScales = TransposeStride<cute::Stride<cute::_1, cute::_0>>;

        void* ptr_final_output = nullptr;
        StrideFinalOutput stride_final_output{};

        void const* ptr_bias = nullptr;
        StrideBias stride_bias{};

        float const* ptr_router_scales = nullptr;
        StrideRouterScales stride_router_scales{};

        int64_t const* ptr_expert_first_token_offset = nullptr;
        int const* ptr_source_token_index = nullptr;

        size_t num_rows_in_final_output = 0;
    };

    DefaultEpilogue default_epilogue;
    FusedFinalizeEpilogue fused_finalize_epilogue;

    enum class EpilogueFusion
    {
        NONE,
        ACTIVATION,
        GATED_ACTIVATION,
        FINALIZE
    };
    EpilogueFusion fusion = EpilogueFusion::NONE;

    float const** alpha_scale_ptr_array = nullptr;

    uint8_t* gemm_workspace = nullptr;
    size_t gemm_workspace_size = 0;

    static std::array<size_t, 10> workspaceBuffers(int num_experts);

    static size_t workspaceSize(int num_experts);

    void configureWorkspace(int8_t* start_ptr, int num_experts, void* gemm_workspace, size_t gemm_workspace_size);

    bool isValid() const
    {
        return stride_a != nullptr && ptr_a != nullptr;
    }

    void setFinalizeFusionParams(void* final_output, float const* router_scales,
        int64_t const* expert_first_token_offset, int const* source_token_index, void const* bias, int hidden_size,
        int num_output_tokens);

    std::string toString() const;
};

// Note update moe.py to match
enum class ActivationType
{
    Gelu = 0,
    Relu,
    Silu,
    Swiglu,
    Geglu,
    Identity,
    InvalidType
};

constexpr bool isGatedActivation(ActivationType activation_type)
{
    return activation_type == ActivationType::Swiglu || activation_type == ActivationType::Geglu;
}

template <typename T,                   /*The type used for activations/scales/compute*/
    typename WeightType,                /* The type for the MoE weights */
    typename OutputType,                /* The output type for the GEMM */
    typename ScaleBiasType = OutputType /* The type for the scales/bias */
    >
class MoeGemmRunner
{
public:
    MoeGemmRunner();

#if defined(ENABLE_FP8)
    static constexpr bool use_fp8 = std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;
#else
    static constexpr bool use_fp8 = false;
#endif

    void moeGemmBiasAct(T const* A, WeightType const* B, ScaleBiasType const* weight_scales,
        ScaleBiasType const* biases, bool bias_is_broadcast, void* C, int64_t const* total_tokens_including_expert,
        HopperGroupedGemmInput layout_info, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
        ActivationType activation_type, bool use_fused_moe, float const** alpha_scale_ptr_array, cudaStream_t stream,
        cutlass_extensions::CutlassGemmConfig chosen_conf);

    void moeGemm(T const* A, WeightType const* B, ScaleBiasType const* weight_scales, void* C,
        int64_t const* total_tokens_including_expert, HopperGroupedGemmInput layout_info, int64_t total_rows,
        int64_t gemm_n, int64_t gemm_k, int num_experts, bool use_fused_moe, float const** alpha_scale_ptr_array,
        cudaStream_t stream, cutlass_extensions::CutlassGemmConfig chosen_conf);

    std::vector<cutlass_extensions::CutlassGemmConfig> getConfigs() const;
    static std::vector<cutlass_extensions::CutlassGemmConfig> getConfigs(int sm);
    static std::vector<cutlass_extensions::CutlassGemmConfig> getHopperConfigs(int sm);
    static std::vector<cutlass_extensions::CutlassGemmConfig> getAmpereConfigs(int sm);

    [[nodiscard]] bool isHopperSpecialised(cutlass_extensions::CutlassGemmConfig gemm_config) const;
    [[nodiscard]] bool supportsHopperSpecialisation() const;
    [[nodiscard]] bool isFusedGatedActivation(
        cutlass_extensions::CutlassGemmConfig gemm_config, bool is_gated_activation, int gemm_n, int gemm_k) const;
    [[nodiscard]] bool supportsFusedGatedActivation(bool is_gated_activation, int gemm_n, int gemm_k) const;

    size_t getMaxWorkspaceSize(int num_experts) const;

    [[nodiscard]] int getSM() const;

private:
    template <typename EpilogueTag>
    void dispatchToArch(T const* A, WeightType const* B, ScaleBiasType const* weight_scales,
        ScaleBiasType const* biases, bool bias_is_broadcast, void* C, int64_t const* total_tokens_including_expert,
        HopperGroupedGemmInput layout_info, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
        cutlass_extensions::CutlassGemmConfig gemm_config, bool use_fused_moe, float const** alpha_scale_ptr_array,
        cudaStream_t stream, int* occupancy = nullptr);

    template <typename EpilogueTag>
    void runGemm(T const* A, WeightType const* B, ScaleBiasType const* weight_scales, ScaleBiasType const* biases,
        bool bias_is_broadcast, void* C, int64_t const* total_tokens_including_expert,
        HopperGroupedGemmInput layout_info, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
        bool use_fused_moe, float const** alpha_scale_ptr_array, cudaStream_t stream,
        cutlass_extensions::CutlassGemmConfig chosen_conf);

private:
    int sm_{};
    int multi_processor_count_{};
    mutable int num_experts_ = 0;
    mutable size_t gemm_workspace_size_ = 0;
    size_t calcMaxWorkspaceSize(int num_experts) const;
};

} // namespace tensorrt_llm
