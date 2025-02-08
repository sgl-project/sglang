/*
 * Copyright (c) 2025 by SGLang team.
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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// parent_list [bs, topk * (depth - 1) + 1)]
// selected_index [bs, draft_token_num - 1]
// verified_seq_len [bs]
// tree_mask [draft_token*(seq_len[0]+draft_token) | draft_token*(seq_len[1]+draft_token) | ..] =
// [sum(verified_seq_len)*draft_token+bs*draft_token*draft_token] positions [bs * draft_token] retrive_index [b,
// draft_token] retrive_next_token [b, draft_token] retrive_next_sibling [b, draft_token]
__global__ void build_tree_efficient(int64_t* parent_list, int64_t* selected_index, int32_t* verified_seq_len,
                                     bool* tree_mask, int64_t* positions, int64_t* retrive_index,
                                     int64_t* retrive_next_token, int64_t* retrive_next_sibling, int topk, int depth,
                                     int draft_token_num) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= draft_token_num) {
    return;
  }
  int seq_tree_idx = draft_token_num * draft_token_num * bid;
  for (int i = 0; i < bid; i++) {
    seq_tree_idx += verified_seq_len[i] * draft_token_num;
  }
  int seq_len = verified_seq_len[bid];
  int token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len + 1;
  for (int i = 0; i < draft_token_num - 1; i++) {
    tree_mask[token_tree_idx + i] = false;
  }

  int position = 0;
  if (tid == 0) {
    positions[bid * draft_token_num] = seq_len;

    int retrive_index_offset = bid * draft_token_num;
    for (int i = draft_token_num - 1; i > 0; --i) {
      int current_token_idx = retrive_index_offset + i;
      retrive_index[bid * draft_token_num + i] = current_token_idx;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + i - 1] / topk;
      int parent_position = 0;
      if (parent_tb_idx > 0) {
        int parent_token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
        for (; parent_position < draft_token_num; ++parent_position) {
          if (selected_index[bid * (draft_token_num - 1) + parent_position] == parent_token_idx) {
            ++parent_position;
            break;
          }
        }
      }
      if (parent_position == draft_token_num) {
        printf(
            "ERROR: invalid eagle tree!!! Detected a token with no parent token selected. Check the logprob. The token "
            "will be dropped.");
        continue;
      }

      if (retrive_next_token[bid * draft_token_num + parent_position] == -1) {
        retrive_next_token[bid * draft_token_num + parent_position] = i;
      } else {
        int origin_next_token = retrive_next_token[bid * draft_token_num + parent_position];
        retrive_next_token[bid * draft_token_num + parent_position] = i;
        retrive_next_sibling[bid * draft_token_num + i] = origin_next_token;
      }
    }
    retrive_index[bid * draft_token_num] = bid * draft_token_num;
  } else {
    int cur_position = tid - 1;
    while (true) {
      position += 1;
      tree_mask[token_tree_idx + cur_position] = true;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + cur_position] / topk;
      if (parent_tb_idx == 0) {
        break;
      }

      int token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
      for (cur_position = 0; cur_position < draft_token_num; ++cur_position) {
        if (selected_index[bid * (draft_token_num - 1) + cur_position] == token_idx) {
          break;
        }
      }
    }
    positions[bid * draft_token_num + tid] = position + seq_len;
  }
}

void build_tree_kernel_efficient(at::Tensor parent_list, at::Tensor selected_index, at::Tensor verified_seq_len,
                                 at::Tensor tree_mask, at::Tensor positions, at::Tensor retrive_index,
                                 at::Tensor retrive_next_token, at::Tensor retrive_next_sibling, int64_t topk,
                                 int64_t depth, int64_t draft_token_num) {
  // TODO (ying) check shape
  // TODO (ying) check type
  int bs = parent_list.size(0);
  dim3 grid(bs);
  dim3 block(draft_token_num);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  build_tree_efficient<<<grid, block, 0, stream>>>(
      static_cast<int64_t*>(parent_list.data_ptr()), static_cast<int64_t*>(selected_index.data_ptr()),
      static_cast<int32_t*>(verified_seq_len.data_ptr()), static_cast<bool*>(tree_mask.data_ptr()),
      static_cast<int64_t*>(positions.data_ptr()), static_cast<int64_t*>(retrive_index.data_ptr()),
      static_cast<int64_t*>(retrive_next_token.data_ptr()), static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
      int32_t(topk), int32_t(depth), int32_t(draft_token_num));
}

// parent_list [bs, topk * (depth - 1) + 1)]
// selected_index [bs, draft_token_num - 1]
// verified_seq_len [bs]
// tree_mask [draft_token*(seq_len[0]+draft_token) | draft_token*(seq_len[1]+draft_token) | ..] =
// [sum(verified_seq_len)*draft_token+bs*draft_token*draft_token] positions [bs * draft_token] retrive_index [b,
// draft_token, depth + 2]
__global__ void build_tree(int64_t* parent_list, int64_t* selected_index, int32_t* verified_seq_len, bool* tree_mask,
                           int64_t* positions, int64_t* retrive_index, int topk, int depth, int draft_token_num) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= draft_token_num) {
    return;
  }
  int seq_tree_idx = draft_token_num * draft_token_num * bid;
  for (int i = 0; i < bid; i++) {
    seq_tree_idx += verified_seq_len[i] * draft_token_num;
  }
  int seq_len = verified_seq_len[bid];
  int token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len + 1;
  for (int i = 0; i < draft_token_num - 1; i++) {
    tree_mask[token_tree_idx + i] = false;
  }

  int position = 0;
  if (tid == 0) {
    positions[bid * draft_token_num] = seq_len;
    retrive_index[bid * draft_token_num * (depth + 2)] = bid * draft_token_num;
    return;
  }

  int depends_order[10];

  int cur_position = tid - 1;
  while (true) {
    depends_order[position] = cur_position + 1;
    position += 1;
    tree_mask[token_tree_idx + cur_position] = true;
    int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + cur_position] / topk;
    if (parent_tb_idx == 0) {
      break;
    }

    int token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
    for (cur_position = 0; cur_position < draft_token_num; cur_position++) {
      if (selected_index[bid * (draft_token_num - 1) + cur_position] == token_idx) {
        break;
      }
    }
    if (cur_position == draft_token_num) {
      printf(
          "ERROR: invalid eagle tree!!! Detected a token with no parent token selected. Check the logprob. The token "
          "will be dropped.");
      break;
    }
  }
  positions[bid * draft_token_num + tid] = position + seq_len;

  int is_leaf = 0;
  for (int i = 1; i < draft_token_num; i++) {
    if (tree_mask[seq_tree_idx + i * (draft_token_num + seq_len) + seq_len + tid]) {
      is_leaf++;
    }
  }
  if (is_leaf == 1) {
    for (int i = 0; i < position; i++) {
      retrive_index[(bid * (draft_token_num) + tid) * (depth + 2) + position - i] =
          depends_order[i] + bid * draft_token_num;
    }
    retrive_index[(bid * (draft_token_num) + tid) * (depth + 2)] = bid * draft_token_num;
  }
}

void build_tree_kernel(at::Tensor parent_list, at::Tensor selected_index, at::Tensor verified_seq_len,
                       at::Tensor tree_mask, at::Tensor positions, at::Tensor retrive_index, int64_t topk,
                       int64_t depth, int64_t draft_token_num) {
  // TODO (ying) check shape
  // TODO (ying) check type
  int bs = parent_list.size(0);
  dim3 grid(bs);
  dim3 block(draft_token_num);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  build_tree<<<grid, block, 0, stream>>>(
      static_cast<int64_t*>(parent_list.data_ptr()), static_cast<int64_t*>(selected_index.data_ptr()),
      static_cast<int32_t*>(verified_seq_len.data_ptr()), static_cast<bool*>(tree_mask.data_ptr()),
      static_cast<int64_t*>(positions.data_ptr()), static_cast<int64_t*>(retrive_index.data_ptr()), int32_t(topk),
      int32_t(depth), int32_t(draft_token_num));
}
