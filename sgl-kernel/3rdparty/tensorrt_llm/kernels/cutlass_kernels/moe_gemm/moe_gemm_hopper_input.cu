/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm
{
std::array<size_t, 10> HopperGroupedGemmInput::workspaceBuffers(int num_experts)
{
    size_t problem_shape_size = sizeof(ProblemShape::UnderlyingProblemShape) * num_experts;
    size_t stride_a_size = sizeof(StrideA) * num_experts;
    size_t stride_b_size = sizeof(StrideB) * num_experts;
    size_t stride_c_size = sizeof(StrideC) * num_experts;
    size_t stride_d_size = sizeof(DefaultEpilogue::StrideD) * num_experts;

    size_t ptr_buf_size = sizeof(void*) * num_experts;
    size_t scale_buf_size = sizeof(float*) * num_experts;

    return std::array{problem_shape_size, stride_a_size, stride_b_size, stride_c_size, stride_d_size, ptr_buf_size,
        ptr_buf_size, ptr_buf_size, ptr_buf_size, scale_buf_size};
}

size_t HopperGroupedGemmInput::workspaceSize(int num_experts)
{
    auto buffers = workspaceBuffers(num_experts);
    return tensorrt_llm::common::calculateTotalWorkspaceSize(buffers.data(), buffers.size());
}

void HopperGroupedGemmInput::configureWorkspace(
    int8_t* start_ptr, int num_experts, void* gemm_workspace, size_t gemm_workspace_size)
{
    auto buffers = workspaceBuffers(num_experts);
    std::array<int8_t*, 10> pointers{};
    TLLM_CHECK_WITH_INFO(pointers.size() == buffers.size(), "Mismatching workspace size and number of buffers");
    for (int i = 0; i < buffers.size(); i++)
    {
        pointers[i] = start_ptr;
        start_ptr = tensorrt_llm::common::nextWorkspacePtr(start_ptr, buffers[i]);
    }

    shape_info.num_groups = num_experts;
    shape_info.problem_shapes = reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(pointers[0]);
    shape_info.host_problem_shapes = nullptr;
    stride_a = reinterpret_cast<StrideA*>(pointers[1]);
    stride_b = reinterpret_cast<StrideB*>(pointers[2]);
    stride_c = reinterpret_cast<StrideC*>(pointers[3]);
    default_epilogue.stride_d = reinterpret_cast<DefaultEpilogue::StrideD*>(pointers[4]);

    ptr_a = reinterpret_cast<void const**>(pointers[5]);
    ptr_b = reinterpret_cast<void const**>(pointers[6]);
    ptr_c = reinterpret_cast<void const**>(pointers[7]);
    default_epilogue.ptr_d = reinterpret_cast<void**>(pointers[8]);

    alpha_scale_ptr_array = reinterpret_cast<float const**>(pointers[9]);

    this->gemm_workspace = reinterpret_cast<uint8_t*>(gemm_workspace);
    this->gemm_workspace_size = gemm_workspace_size;
}

void HopperGroupedGemmInput::setFinalizeFusionParams(void* final_output, float const* router_scales,
    int64_t const* expert_first_token_offset, int const* source_token_index, void const* bias, int hidden_size,
    int num_output_tokens)
{
    fused_finalize_epilogue.ptr_final_output = final_output;
    fused_finalize_epilogue.ptr_router_scales = router_scales;
    fused_finalize_epilogue.ptr_bias = bias;
    fused_finalize_epilogue.ptr_expert_first_token_offset = expert_first_token_offset;
    fused_finalize_epilogue.ptr_source_token_index = source_token_index;

    fused_finalize_epilogue.stride_final_output
        = cutlass::make_cute_packed_stride(FusedFinalizeEpilogue::StrideFinalOutput{},
            transpose_stride(cute::make_shape(num_output_tokens, hidden_size, 1)));
    fused_finalize_epilogue.stride_bias
        = transpose_stride(cute::make_stride(cute::Int<0>{}, cute::Int<1>{}, hidden_size));
    fused_finalize_epilogue.stride_router_scales = {};

    fused_finalize_epilogue.num_rows_in_final_output = num_output_tokens;
}

std::string HopperGroupedGemmInput::toString() const
{
    std::stringstream ss;
    ss << "Hopper Input Information: " << (isValid() ? "valid" : "null") << "\n";
    if (isValid())
    {
        ss << "Ptr A: " << ptr_a << ", Ptr B: " << ptr_b << ", Ptr C: " << ptr_c << "\n";
        ss << "Epilogue Fusion: " << (int) fusion;
        if (fusion == HopperGroupedGemmInput::EpilogueFusion::FINALIZE)
        {
            ss << ",\nFinal Output: " << fused_finalize_epilogue.ptr_final_output;
            ss << " with Stride: " << fused_finalize_epilogue.stride_router_scales;
            ss << ",\nBias: " << fused_finalize_epilogue.ptr_bias;
            ss << " with Stride: " << fused_finalize_epilogue.stride_bias;
            ss << ",\nRouter Scales: " << fused_finalize_epilogue.ptr_router_scales;
            ss << " with Stride: " << fused_finalize_epilogue.stride_router_scales;
            ss << ",\nExpert Offset: " << fused_finalize_epilogue.ptr_expert_first_token_offset;
            ss << ", Source Map: " << fused_finalize_epilogue.ptr_source_token_index;
        }
        else
        {
            ss << ", Ptr D: " << default_epilogue.ptr_d;
        }
        ss << '\n';
        ss << "Alpha scale ptr: " << alpha_scale_ptr_array << "\n";
    }
    return ss.str();
}
} // namespace tensorrt_llm
