/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <sgl_kernel/utils.h>

#include "cutlass/cutlass.h"
#include <algorithm>
#include <cstdint>

namespace sglang {

#define CUTLASS_CHECK(status)                                                              \
  {                                                                                        \
    cutlass::Status error = status;                                                        \
    host::RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

inline uint32_t next_pow_2(uint32_t n) {
  if (n <= 1) {
    return 1;
  }
  return 1u << (32 - __builtin_clz(n - 1));
}

}  // namespace sglang
