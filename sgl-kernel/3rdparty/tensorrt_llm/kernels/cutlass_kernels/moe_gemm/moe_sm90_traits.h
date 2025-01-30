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

#include "cutlass/arch/mma_sm90.h"
#include "cutlass_extensions/epilogue_helpers.h"

namespace tensorrt_llm::kernels::cutlass_kernels
{

// Hopper arch
template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault>
constexpr bool isValidHopperMOESpecialisation()
{
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
    return cutlass::platform::is_same<T, WeightType>::value
        && cutlass::platform::is_same<EpilogueTag, cutlass_extensions::EpilogueOpDefault>::value;
#else
    return false; // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED is set when Hopper kernels are enabled
#endif
}

// Hopper arch
template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault>
constexpr bool isValidAmpereMOESpecialisation()
{
    return true; // Default to true
}

} // namespace tensorrt_llm::kernels::cutlass_kernels
