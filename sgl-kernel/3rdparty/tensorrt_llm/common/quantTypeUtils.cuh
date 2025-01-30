/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>

namespace tensorrt_llm
{
namespace common
{

template <typename T>
struct QuantTypeStaticVals;

template <>
struct QuantTypeStaticVals<int8_t>
{
    static constexpr float MAX_VAL = 127.f;
    static constexpr float MIN_SCALING_FACTOR = 0.f;
    static constexpr float MIN_SCALING_FACTOR_RCP = FLT_MAX;
};

#ifdef ENABLE_FP8

template <>
struct QuantTypeStaticVals<__nv_fp8_e4m3>
{
    static constexpr float MAX_VAL = 448.f;
    // Ref: https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gen_ai/src/quantize/quantize.cu#L720
    static constexpr float MIN_SCALING_FACTOR = 1.0f / (448.f * 512.f);
    static constexpr float MIN_SCALING_FACTOR_RCP = (448.f * 512.f);
};

#endif // ENABLE_FP8

} // namespace common
} // namespace tensorrt_llm
