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

#include <cutlass/arch/arch.h>

#include <cute/numeric/numeric_types.hpp>

#include "kda/sm90/prefill_kernel_kda_fwd_sm90.cuh"
#include "kda/sm90/utils/common.hpp"

namespace kda::sm90 {

using namespace cute;
using bf16 = cute::bfloat16_t;

// SafeGate=true, InitState=false
template void launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    float const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=true
template void launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    float const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

}  // namespace kda::sm90
