/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <functional>
#include <numeric>
#include <optional>
#include <sstream>

namespace tensorrt_llm::common::stl_utils
{

template <typename TInputIt, typename TOutputIt, typename TBinOp>
constexpr TOutputIt basicInclusiveScan(TInputIt first, TInputIt last, TOutputIt dFirst, TBinOp op)
{
    if (first != last)
    {
        auto val = *first;
        while (true)
        {
            *dFirst = val;
            ++dFirst;
            ++first;
            if (first == last)
            {
                break;
            }
            val = op(std::move(val), *first);
        }
    }
    return dFirst;
}

template <typename TInputIt, typename TOutputIt>
constexpr TOutputIt inclusiveScan(TInputIt first, TInputIt last, TOutputIt dFirst)
{
#if defined(__GNUC__) && __GNUC__ <= 8
    return basicInclusiveScan(first, last, dFirst, std::plus<>{});
#else
    return std::inclusive_scan(first, last, dFirst);
#endif
}

template <typename TInputIt, typename TOutputIt, typename T, typename TBinOp>
constexpr TOutputIt basicExclusiveScan(TInputIt first, TInputIt last, TOutputIt dFirst, T init, TBinOp op)
{
    if (first != last)
    {
        while (true)
        {
            T tmp{op(init, *first)};
            *dFirst = init;
            ++dFirst;
            ++first;
            if (first == last)
            {
                break;
            }
            init = std::move(tmp);
        }
    }
    return dFirst;
}

template <typename TInputIt, typename TOutputIt, typename T>
constexpr TOutputIt exclusiveScan(TInputIt first, TInputIt last, TOutputIt dFirst, T init)
{
#if defined(__GNUC__) && __GNUC__ <= 8
    return basicExclusiveScan(first, last, dFirst, std::move(init), std::plus<>{});
#else
    return std::exclusive_scan(first, last, dFirst, std::move(init));
#endif
}

template <typename T, typename = void>
struct HasOperatorOutput : std::false_type
{
};

template <typename T>
struct HasOperatorOutput<T, std::void_t<decltype((std::declval<std::ostream&>() << std::declval<T>()))>>
    : std::true_type
{
};

template <typename T>
std::string toString(T const& t, typename std::enable_if_t<HasOperatorOutput<T>::value, int> = 0)
{
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

template <typename T>
std::string toString(std::optional<T> const& t, typename std::enable_if_t<HasOperatorOutput<T>::value, int> = 0)
{
    std::ostringstream oss;
    if (t)
    {
        oss << t.value();
    }
    else
    {
        oss << "None";
    }
    return oss.str();
}

} // namespace tensorrt_llm::common::stl_utils
