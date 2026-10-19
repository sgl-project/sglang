/*
 * Copyright (c) 2026 SGLang Team
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

#include <cstdint>

namespace sm90::nvfp4_legacy {

// The GLM-5.2 NVFP4 kernel predates SparseAttnDecodeParams.  Keep the exact
// parameter ABI local to that kernel so newer FlashMLA pins cannot silently
// change its launch layout.
struct DecodingParams {
  using index_t = int64_t;

  int b;
  int s_q;
  int q_seq_per_hk;
  int d, d_v;
  int h_q, h_k;
  int num_blocks;
  int q_head_per_hk;
  bool is_causal;
  float scale_softmax, scale_softmax_log2;
  int topk;

  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ o_ptr;
  void* __restrict__ softmax_lse_ptr;
  int* __restrict__ indices_ptr;

  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t o_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t o_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t o_head_stride;
  index_t indices_batch_stride;
  index_t indices_row_stride;

  int* __restrict__ block_table;
  index_t block_table_batch_stride;
  int page_block_size;
  int* __restrict__ seqlens_k_ptr;

  int* __restrict__ tile_scheduler_metadata_ptr;
  int num_sm_parts;
  int* __restrict__ num_splits_ptr;

  int total_num_splits;
  void* __restrict__ softmax_lseaccum_ptr;
  void* __restrict__ oaccum_ptr;
};

inline constexpr int kTileSchedulerMetadataSize = 8;

}  // namespace sm90::nvfp4_legacy
