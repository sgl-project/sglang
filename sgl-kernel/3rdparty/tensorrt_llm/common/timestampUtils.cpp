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

#include <chrono>
#include <iomanip>
#include <sstream>

#include "tensorrt_llm/common/timestampUtils.h"

namespace tensorrt_llm::common
{

std::string getCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&now_t);

    auto epoch_to_now = now.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch_to_now);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(epoch_to_now - seconds);

    std::ostringstream stream;
    stream << std::put_time(&tm, "%m-%d-%Y %H:%M:%S");
    stream << "." << std::setfill('0') << std::setw(6) << us.count();
    return stream.str();
}

} // namespace tensorrt_llm::common
