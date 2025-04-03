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

#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/all.h>

#include "trt_reduce_internal.cuh"

namespace trt_llm {
constexpr size_t MAX_ALL_TO_ALL_BLOCKS = 64;

struct All2AllParams {
  int64_t elts_size;
  int64_t elts_per_block;            // for sync input
  int64_t elts_per_block_per_round;  // for output
  int64_t round_count;
  int64_t elts_per_round;
  int64_t rank, local_rank;
  int64_t ranks_per_node;
  uint32_t barrier_flag;
  int64_t* buffer_meta_ptrs[MAX_RANKS_PER_NODE];  // current rank output_split_sizes read by other rank
  uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
  uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
  RankData* peer_comm_buffer_ptrs;
  void* local_output_buffer_ptr;
  void* local_input_buffer_ptr;
  int64_t* plan_meta_ptr;
  bool is_capturing;

  int64_t block_size;  // 572 or 512
  int64_t output_elts_total;
  int64_t input_elts_total;
};

struct All2AllPlanParams {
  int64_t elts_size;
  int64_t rank, local_rank;
  int64_t ranks_per_node;
  uint32_t barrier_flag;
  int64_t* buffer_meta_ptrs[MAX_RANKS_PER_NODE];  // current rank output_split_sizes read by other rank
  uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
  uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
  RankData* peer_comm_buffer_ptrs;
  int64_t* output_split_sizes;
  int64_t* input_split_sizes;
  int64_t* output_split_offsets;
  int64_t* input_split_offsets;
  int64_t* plan_meta_ptr;

  int64_t block_size;  // 16*572 or 16*512
  int64_t output_elts_total;
  int64_t input_elts_total;
};
}  // namespace trt_llm
