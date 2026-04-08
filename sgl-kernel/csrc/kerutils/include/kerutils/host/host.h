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

/*
 * Copyright (c) 2025 DeepSeek. All Rights Reserved.
 *
 * Licensed under the MIT License.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <cuda_runtime_api.h>

#include <exception>
#include <sstream>
#include <string>

#include "kerutils/common/common.h"

namespace kerutils {

class KUException final : public std::exception {
  std::string message = {};

 public:
  template <typename... Args>
  explicit KUException(const char* name, const char* file, const int line, Args&&... args) {
    std::ostringstream oss;

    oss << name << " error (" << file << ":" << line << "): ";
    (oss << ... << args);
    message = oss.str();
  }

  const char* what() const noexcept override {
    return message.c_str();
  }
};

#define THROW_KU_EXCEPTION(name, ...) throw kerutils::KUException(name, __FILE__, __LINE__, __VA_ARGS__)

#define KU_CUDA_CHECK(call)                                                                         \
  do {                                                                                              \
    cudaError_t status_ = call;                                                                     \
    if (status_ != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
      THROW_KU_EXCEPTION("CUDA", "CUDA error: ", cudaGetErrorString(status_));                      \
    }                                                                                               \
  } while (0)

// This `KU_ASSERT` is triggered no matter if the code is compiled with `-DNDEBUG` or not.
#define KU_ASSERT(cond, ...)                                                         \
  do {                                                                               \
    if (not(cond)) {                                                                 \
      fprintf(stderr, "Assertion `%s` failed (%s:%d): ", #cond, __FILE__, __LINE__); \
      if constexpr (sizeof(#__VA_ARGS__) > 1) {                                      \
        fprintf(stderr, ", " __VA_ARGS__);                                           \
      }                                                                              \
      fprintf(stderr, "\n");                                                         \
      THROW_KU_EXCEPTION("Assertion", "Assertion `", #cond, "` failed.");            \
    }                                                                                \
  } while (0)

#define KU_CHECK_KERNEL_LAUNCH() KU_CUDA_CHECK(cudaGetLastError())

template <typename T>
inline __host__ __device__ constexpr T ceil_div(const T& a, const T& b) {
  return (a + b - 1) / b;
}

template <typename T>
inline __host__ __device__ constexpr T ceil(const T& a, const T& b) {
  return (a + b - 1) / b * b;
}

}  // namespace kerutils
