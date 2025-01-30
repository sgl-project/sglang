/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvtx3/nvtx3.hpp>

#include <array>

namespace tensorrt_llm::common::nvtx
{
inline nvtx3::color nextColor()
{
#ifndef NVTX_DISABLE
    constexpr std::array kColors{nvtx3::color{0xff00ff00}, nvtx3::color{0xff0000ff}, nvtx3::color{0xffffff00},
        nvtx3::color{0xffff00ff}, nvtx3::color{0xff00ffff}, nvtx3::color{0xffff0000}, nvtx3::color{0xffffffff}};
    constexpr auto numColors = kColors.size();

    static thread_local std::size_t colorId = 0;
    auto const color = kColors[colorId];
    colorId = colorId + 1 >= numColors ? 0 : colorId + 1;
    return color;
#else
    return nvtx3::color{0};
#endif
}

} // namespace tensorrt_llm::common::nvtx

#define NVTX3_SCOPED_RANGE_WITH_NAME(range, name)                                                                      \
    ::nvtx3::scoped_range range(::tensorrt_llm::common::nvtx::nextColor(), name)
#define NVTX3_SCOPED_RANGE(range) NVTX3_SCOPED_RANGE_WITH_NAME(range##_range, #range)
