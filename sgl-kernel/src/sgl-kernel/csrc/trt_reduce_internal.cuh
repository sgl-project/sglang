// reference:
// https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.14/cpp/tensorrt_llm/plugins/ncclPlugin/allreducePlugin.cpp
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

#pragma once
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/all.h>

#define FatalError(s)                                                 \
  do {                                                                \
    std::stringstream _where, _message;                               \
    _where << __FILE__ << ':' << __LINE__;                            \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
    std::cerr << _message.str() << "\nAborting...\n";                 \
    assert(false);                                                    \
    exit(1);                                                          \
  } while (0)

#define CHECKCUDA(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

namespace trt_llm {
constexpr size_t WARP_SIZE = 32;
constexpr size_t MAX_ALL_REDUCE_BLOCKS = 24;
constexpr size_t MAX_RANKS_PER_NODE = 8;
constexpr size_t DEFAULT_BLOCK_SIZE = 1024;

enum class AllReduceStrategyType : int8_t {
  RING = 0,
  ONESHOT = 1,
  TWOSHOT = 2,
  AUTO = 3,
};

struct AllReduceParams {
  size_t elts_size;
  size_t elts_total;
  size_t elts_per_rank;
  size_t elts_per_block;
  size_t rank_offset;
  size_t ranks_per_node, rank, local_rank;
  uint32_t barrier_flag;
  uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
  uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
  void* peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
  void* local_input_buffer_ptr;
  void* local_output_buffer_ptr;
};

inline size_t GetMaxRequiredWorkspaceSize(int world_size) {
  if (world_size <= 2) {
    return 16 * 1000 * 1000;
  }
  return 8 * 1000 * 1000;
}

inline AllReduceStrategyType SelectImplementation(size_t message_size, int world_size) {
  const size_t maxWorkspaceSize = GetMaxRequiredWorkspaceSize(world_size);

  if (message_size > maxWorkspaceSize) {
    assert(false && "Custom allreduce do not ring currently");
    return AllReduceStrategyType::RING;
  }

  if (world_size <= 2) {
    return AllReduceStrategyType::ONESHOT;
  }

  if (world_size <= 4) {
    if (message_size < 1 * 1000 * 1000) {
      return AllReduceStrategyType::ONESHOT;
    }
    assert(false && "Custom allreduce do not twoshot currently");
    return AllReduceStrategyType::TWOSHOT;
  }

  if (message_size < 500 * 1000) {
    return AllReduceStrategyType::ONESHOT;
  }
  assert(false && "Custom allreduce do not twoshot currently");
  return AllReduceStrategyType::TWOSHOT;
}

void trtCustomAllReduce(AllReduceParams& params, at::ScalarType data_type, AllReduceStrategyType strat,
                        cudaStream_t stream);

}  // namespace trt_llm
