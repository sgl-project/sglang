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

#pragma once

#include <cstdio>
#include <cstdlib>

namespace kerutils {}

#define KU_PRINTLN(fmt, ...)         \
  {                                  \
    cute::print(fmt, ##__VA_ARGS__); \
    print("\n");                     \
  }

#define CHECK_CUDA(call)                                                                            \
  do {                                                                                              \
    cudaError_t status_ = call;                                                                     \
    if (status_ != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
      exit(1);                                                                                      \
    }                                                                                               \
  } while (0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

namespace ku = kerutils;
