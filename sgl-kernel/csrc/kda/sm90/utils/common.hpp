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
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

#include "kda/sm90/utils/debug.hpp"

#define FLAT_UNUSED_PARAMETER(x) (void)x

#define CHECK(expr, msg)                                                                           \
  do {                                                                                             \
    if (!(expr)) {                                                                                 \
      std::string buffer(1024, '\0');                                                              \
      sprintf(buffer.data(), "Failed to check %s, %s at %s:%d\n", ##expr, msg __FILE__, __LINE__); \
      throw std::runtime_error(buffer.c_str());                                                    \
    }                                                                                              \
  } while (0)

#define CUDA_CHECK(expr)                                                                                             \
  do {                                                                                                               \
    cudaError_t err = (expr);                                                                                        \
    if (err != cudaSuccess) {                                                                                        \
      std::string buffer(1024, '\0');                                                                                \
      sprintf(buffer.data(), "CUDA Error: %s, Code: %d at %s:%d\n", cudaGetErrorName(err), err, __FILE__, __LINE__); \
      throw std::runtime_error(buffer.c_str());                                                                      \
    }                                                                                                                \
  } while (0)
