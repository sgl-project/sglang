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

#include <cute/config.hpp>

#if DEBUG_PIPE
#define PIPE_DEBUG_PRINTF(fmt, ...) \
  if (threadIdx.x == 0) printf("%s:%d " fmt, __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define PIPE_DEBUG_PRINTF(...)
#endif

#ifndef FLAT_DEBUG_PRINT
#define FLAT_DEBUG_PRINT 0
#endif

#if FLAT_DEBUG_PRINT
#define IS_PRINT_BLOCK cute::block(1)
#define DPRINTF(fmt, ...) \
  if (IS_PRINT_BLOCK) printf("%s:%d " fmt, __FILE__, __LINE__, ##__VA_ARGS__)
#define DPRINTF0(fmt, ...) \
  if (IS_PRINT_BLOCK && threadIdx.x == 0) printf("%s:%d " fmt, __FILE__, __LINE__, ##__VA_ARGS__)
#define DPRINTF_W(fmt, ...)            \
  if (IS_PRINT_BLOCK)                  \
  printf(                              \
      "%s:%d [WG%d][W%d][T%-3d] " fmt, \
      __FILE__,                        \
      __LINE__,                        \
      threadIdx.x / 128,               \
      threadIdx.x / 32,                \
      threadIdx.x,                     \
      ##__VA_ARGS__)
#define DPRINTF0_W(fmt, ...)                   \
  if (IS_PRINT_BLOCK && threadIdx.x % 32 == 0) \
  printf(                                      \
      "%s:%d [WG%d][W%d][T%-3d] " fmt,         \
      __FILE__,                                \
      __LINE__,                                \
      threadIdx.x / 128,                       \
      threadIdx.x / 32,                        \
      threadIdx.x,                             \
      ##__VA_ARGS__)
#define DPRINTF_WG(fmt, ...)           \
  if (IS_PRINT_BLOCK)                  \
  printf(                              \
      "%s:%d [WG%d][W%d][T%-3d] " fmt, \
      __FILE__,                        \
      __LINE__,                        \
      threadIdx.x / 128,               \
      threadIdx.x / 32,                \
      threadIdx.x,                     \
      ##__VA_ARGS__)
#define DPRINTF0_WG(fmt, ...)                   \
  if (IS_PRINT_BLOCK && threadIdx.x % 128 == 0) \
  printf(                                       \
      "%s:%d [WG%d][W%d][T%-3d] " fmt,          \
      __FILE__,                                 \
      __LINE__,                                 \
      threadIdx.x / 128,                        \
      threadIdx.x / 32,                         \
      threadIdx.x,                              \
      ##__VA_ARGS__)
#else
#define DPRINTF(...)
#define DPRINTF0(...)
#define DPRINTF_W(...)
#define DPRINTF0_W(...)
#define DPRINTF_WG(...)
#define DPRINTF0_WG(...)
#endif

#if FLAT_DEBUG_PRINT
#define DPRINT_TMA_DESC(tma_dess_addr)                             \
  do {                                                             \
    auto p = reinterpret_cast<const unsigned int*>(tma_dess_addr); \
    DPRINTF(                                                       \
        "\n"                                                       \
        "%08X%08X %08X%08X %08X%08X %08X%08X\n"                    \
        "%08X%08X %08X%08X %08X%08X %08X%08X\n"                    \
        "%08X%08X %08X%08X %08X%08X %08X%08X\n"                    \
        "%08X%08X %08X%08X %08X%08X %08X%08X\n",                   \
        p[0],                                                      \
        p[1],                                                      \
        p[2],                                                      \
        p[3],                                                      \
        p[4],                                                      \
        p[5],                                                      \
        p[6],                                                      \
        p[7],                                                      \
        p[8],                                                      \
        p[9],                                                      \
        p[10],                                                     \
        p[11],                                                     \
        p[12],                                                     \
        p[13],                                                     \
        p[14],                                                     \
        p[15],                                                     \
        p[16],                                                     \
        p[17],                                                     \
        p[18],                                                     \
        p[19],                                                     \
        p[20],                                                     \
        p[21],                                                     \
        p[22],                                                     \
        p[23],                                                     \
        p[24],                                                     \
        p[25],                                                     \
        p[26],                                                     \
        p[27],                                                     \
        p[28],                                                     \
        p[29],                                                     \
        p[30],                                                     \
        p[31]);                                                    \
  } while (0)
#else
#define DPRINT_TMA_DESC(tma_dess_addr)
#endif
