/* Copyright 2025 SGLang Team. All Rights Reserved.

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

#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/all.h>

#include <cassert>

#include "utils.h"

namespace trt_llm {
constexpr size_t MAX_ALL_REDUCE_BLOCKS = 32;
constexpr size_t MAX_RANKS_PER_NODE = 8;
constexpr size_t DEFAULT_BLOCK_SIZE = 512;

enum class AllReduceStrategyType : int8_t {
  RING = 0,
  ONESHOT = 1,
  TWOSHOT = 2,
  AUTO = 3,
};

struct RankData {
  void* ptrs[MAX_RANKS_PER_NODE];
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
  uint32_t* tmp_result_buffers[MAX_RANKS_PER_NODE];
  RankData* peer_comm_buffer_ptrs;
  void* local_input_buffer_ptr;
  void* local_output_buffer_ptr;
  bool is_capturing;
};

using fptr_t = int64_t;
using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;

class AllReduceMeta {
 public:
  AllReduceMeta(
      int64_t rank_id,
      int64_t world_size,
      torch::Tensor& rank_data,
      const std::vector<fptr_t>& buffers,
      const std::vector<fptr_t>& tmp_result_buffers,
      const std::vector<fptr_t>& barrier_in,
      const std::vector<fptr_t>& barrier_out) {
    this->rank_id = (int)rank_id;
    this->world_size = (int)world_size;
    this->barrier_in = barrier_in;
    this->barrier_out = barrier_out;
    this->tmp_result_buffers = tmp_result_buffers;

    this->rank_data_base = reinterpret_cast<RankData*>(rank_data.data_ptr());
    RankData data;
    for (int i = 0; i < world_size; i++) {
      data.ptrs[i] = (void*)buffers[i];
    }
    auto d_data = this->rank_data_base++;
    CHECK_CUDA_SUCCESS(cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    this->buffers = d_data;
  }

  ~AllReduceMeta() {
    for (auto [_, ptr] : ipc_handles_) {
      CHECK_CUDA_SUCCESS(cudaIpcCloseMemHandle(ptr));
    }
  }

 public:
  int world_size;
  int rank_id;
  std::vector<fptr_t> barrier_in;
  std::vector<fptr_t> barrier_out;
  std::vector<fptr_t> tmp_result_buffers;
  int barrier_flag = 1;
  RankData* buffers;
  RankData* rank_data_base;
  std::vector<void*> graph_unreg_buffers;
  std::map<IPC_KEY, char*> ipc_handles_;
};

inline size_t GetMaxRequiredWorkspaceSize(int world_size) {
  if (world_size <= 2) {
    return 16 * 1024 * 1024;
  }
  return 8 * 1024 * 1024;
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
    if (message_size < 1 * 1024 * 1024) {
      return AllReduceStrategyType::ONESHOT;
    }
    return AllReduceStrategyType::TWOSHOT;
  }

  if (message_size < 512 * 1024) {
    return AllReduceStrategyType::ONESHOT;
  }
  return AllReduceStrategyType::TWOSHOT;
}

inline int divUp(int a, int b) {
  return (a + b - 1) / b;
}

inline int roundUp(int a, int n) {
  return divUp(a, n) * n;
}

void trtCustomAllReduce(
    AllReduceParams& params, at::ScalarType data_type, AllReduceStrategyType strat, cudaStream_t stream);

}  // namespace trt_llm
