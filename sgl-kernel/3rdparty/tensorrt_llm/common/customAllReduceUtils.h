/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstddef>

namespace tensorrt_llm::utils::customAllReduceUtils
{

constexpr size_t NUM_POINTERS_PER_RANK = 7;

// WARNING: MUST BE KEPT IN SYNC with tensorrt_llm/plugin/plugin.py
inline size_t getMaxRequiredWorkspaceSize(int worldSize) noexcept
{
    if (worldSize <= 2)
    {
        return 16 * 1000 * 1000;
    }
    return 8 * 1000 * 1000;
}

} // namespace tensorrt_llm::utils::customAllReduceUtils
