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
// Adapted from
// https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/speculative/eagle_utils.cu

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

typedef enum { FULL_MASK = 0, QLEN_ONLY = 1, QLEN_ONLY_BITPACKING = 2 } TreeMaskMode;

// parent_list [bs, topk * (depth - 1) + 1)]
// selected_index [bs, draft_token_num - 1]
// verified_seq_len [bs]
// tree_mask [draft_token*(seq_len[0]+draft_token) | draft_token*(seq_len[1]+draft_token) | ..] =
// [sum(verified_seq_len)*draft_token+bs*draft_token*draft_token] positions [bs * draft_token] retrive_index [b,
// draft_token] retrive_next_token [b, draft_token] retrive_next_sibling [b, draft_token]
__global__ void build_tree_efficient(
    int64_t* parent_list,
    int64_t* selected_index,
    int64_t* verified_seq_len,
    bool* tree_mask,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    int tree_mask_mode) {
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
  int token_tree_idx;
  if (tree_mask_mode == FULL_MASK) {
    token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len + 1;
  } else {
    token_tree_idx = draft_token_num * draft_token_num * bid + draft_token_num * tid + 1;
  }
  tree_mask[token_tree_idx - 1] = true;
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
            "WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
            "Please check if the logprob has nan. The token will be ignored to keep proceeding.\n");
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

// parent_list [bs, topk * (depth - 1) + 1)]
// selected_index [bs, draft_token_num - 1]
// verified_seq_len [bs]
// tree_mask: [draft_token*num_bytes_per_item | .. ] = [bs*draft_token*num_bytes_per_item]
// positions [bs * draft_token]
// retrive_index [bs, draft_token]
// retrive_next_token [bs, draft_token]
// retrive_next_sibling [bs, draft_token]
__global__ void build_tree_efficient_partial_packed(
    int64_t* parent_list,
    int64_t* selected_index,
    int64_t* verified_seq_len,
    uint8_t* tree_mask,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    size_t num_bytes_per_item) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= draft_token_num) {
    return;
  }
  int seq_len = verified_seq_len[bid];
  int token_tree_idx = (bid * draft_token_num + tid) * num_bytes_per_item;
  tree_mask[token_tree_idx] = 1;  // little endian

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
            "WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
            "Please check if the logprob has nan. The token will be ignored to keep proceeding.\n");
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
      int byte_idx = (cur_position + 1) / 8;
      int bit_idx = (cur_position + 1) % 8;
      tree_mask[token_tree_idx + byte_idx] |= (1 << bit_idx);
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

template <typename IdType, typename IdType2>
__global__ void VerifyTreeGreedy(
    IdType* predicts,
    IdType* accept_index,
    IdType* accept_token_num,  // mutable
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    IdType2* target_predict,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens) {
  uint32_t bx = blockIdx.x;

  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      IdType2 target_token_id = target_predict[last_accepted_retrive_idx];

      if (draft_token_id == target_token_id) {
        // accept token
        predicts[last_accepted_retrive_idx] = target_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;
  predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx];
}

namespace {

// ---------------------------------------------------------------------------
// tvm-ffi entry points
// ---------------------------------------------------------------------------

// predicts:              [tot_num_draft_tokens] int32, mutable
// accept_index:          [bs, num_spec_step] int32, mutable
// accept_token_num:      [bs] int32, mutable
// candidates:            [bs, num_draft_tokens] int64
// retrive_index:         [bs, num_draft_tokens] int64
// retrive_next_token:    [bs, num_draft_tokens] int64
// retrive_next_sibling:  [bs, num_draft_tokens] int64
// target_predict:        [bs, num_draft_tokens] int64
void verify_tree_greedy(
    tvm::ffi::TensorView predicts,
    tvm::ffi::TensorView accept_index,
    tvm::ffi::TensorView accept_token_num,
    tvm::ffi::TensorView candidates,
    tvm::ffi::TensorView retrive_index,
    tvm::ffi::TensorView retrive_next_token,
    tvm::ffi::TensorView retrive_next_sibling,
    tvm::ffi::TensorView target_predict) {
  using namespace host;

  RuntimeCheck(candidates.device().device_type == kDLCUDA, "candidates must be a CUDA tensor");
  RuntimeCheck(candidates.ndim() == 2, "candidates must be 2D: [bs, num_draft_tokens]");
  RuntimeCheck(candidates.is_contiguous(), "candidates must be contiguous");
  RuntimeCheck(candidates.dtype().code == kDLInt && candidates.dtype().bits == 64, "candidates must be int64");

  uint32_t batch_size = static_cast<uint32_t>(candidates.size(0));
  uint32_t num_draft_tokens = static_cast<uint32_t>(candidates.size(1));

  RuntimeCheck(accept_index.ndim() == 2, "accept_index must be 2D: [bs, num_spec_step]");
  RuntimeCheck(accept_index.dtype().code == kDLInt && accept_index.dtype().bits == 32, "accept_index must be int32");
  RuntimeCheck(accept_index.is_contiguous(), "accept_index must be contiguous");
  RuntimeCheck(static_cast<uint32_t>(accept_index.size(0)) == batch_size, "accept_index batch_size mismatch");
  uint32_t num_spec_step = static_cast<uint32_t>(accept_index.size(1));

  RuntimeCheck(predicts.ndim() == 1, "predicts must be 1D");
  RuntimeCheck(
      static_cast<uint32_t>(predicts.size(0)) == batch_size * num_draft_tokens,
      "predicts size must equal batch_size * num_draft_tokens");
  RuntimeCheck(predicts.dtype().code == kDLInt && predicts.dtype().bits == 32, "predicts must be int32");
  RuntimeCheck(predicts.is_contiguous(), "predicts must be contiguous");

  RuntimeCheck(accept_token_num.ndim() == 1, "accept_token_num must be 1D");
  RuntimeCheck(
      accept_token_num.dtype().code == kDLInt && accept_token_num.dtype().bits == 32, "accept_token_num must be int32");
  RuntimeCheck(accept_token_num.is_contiguous(), "accept_token_num must be contiguous");
  RuntimeCheck(static_cast<uint32_t>(accept_token_num.size(0)) == batch_size, "accept_token_num batch_size mismatch");

  RuntimeCheck(retrive_index.ndim() == 2, "retrive_index must be 2D: [bs, num_draft_tokens]");
  RuntimeCheck(retrive_index.dtype().code == kDLInt && retrive_index.dtype().bits == 64, "retrive_index must be int64");
  RuntimeCheck(retrive_index.is_contiguous(), "retrive_index must be contiguous");
  RuntimeCheck(
      static_cast<uint32_t>(retrive_index.size(0)) == batch_size &&
          static_cast<uint32_t>(retrive_index.size(1)) == num_draft_tokens,
      "retrive_index shape mismatch");

  RuntimeCheck(retrive_next_token.ndim() == 2, "retrive_next_token must be 2D: [bs, num_draft_tokens]");
  RuntimeCheck(
      retrive_next_token.dtype().code == kDLInt && retrive_next_token.dtype().bits == 64,
      "retrive_next_token must be int64");
  RuntimeCheck(retrive_next_token.is_contiguous(), "retrive_next_token must be contiguous");
  RuntimeCheck(
      static_cast<uint32_t>(retrive_next_token.size(0)) == batch_size &&
          static_cast<uint32_t>(retrive_next_token.size(1)) == num_draft_tokens,
      "retrive_next_token shape mismatch");

  RuntimeCheck(retrive_next_sibling.ndim() == 2, "retrive_next_sibling must be 2D: [bs, num_draft_tokens]");
  RuntimeCheck(
      retrive_next_sibling.dtype().code == kDLInt && retrive_next_sibling.dtype().bits == 64,
      "retrive_next_sibling must be int64");
  RuntimeCheck(retrive_next_sibling.is_contiguous(), "retrive_next_sibling must be contiguous");
  RuntimeCheck(
      static_cast<uint32_t>(retrive_next_sibling.size(0)) == batch_size &&
          static_cast<uint32_t>(retrive_next_sibling.size(1)) == num_draft_tokens,
      "retrive_next_sibling shape mismatch");

  RuntimeCheck(target_predict.ndim() == 2, "target_predict must be 2D: [bs, num_draft_tokens]");
  RuntimeCheck(
      target_predict.dtype().code == kDLInt && target_predict.dtype().bits == 64, "target_predict must be int64");
  RuntimeCheck(target_predict.is_contiguous(), "target_predict must be contiguous");
  RuntimeCheck(
      static_cast<uint32_t>(target_predict.size(0)) == batch_size &&
          static_cast<uint32_t>(target_predict.size(1)) == num_draft_tokens,
      "target_predict shape mismatch");

  cudaStream_t stream = LaunchKernel::resolve_device(candidates.device());
  dim3 grid(batch_size);
  dim3 block(1);

  LaunchKernel(grid, block, stream)(
      VerifyTreeGreedy<int32_t, int64_t>,
      static_cast<int32_t*>(predicts.data_ptr()),
      static_cast<int32_t*>(accept_index.data_ptr()),
      static_cast<int32_t*>(accept_token_num.data_ptr()),
      static_cast<int64_t*>(candidates.data_ptr()),
      static_cast<int64_t*>(retrive_index.data_ptr()),
      static_cast<int64_t*>(retrive_next_token.data_ptr()),
      static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
      static_cast<int64_t*>(target_predict.data_ptr()),
      batch_size,
      num_spec_step,
      num_draft_tokens);
}

// parent_list:          [bs, topk * (depth - 1) + 1] int64
// selected_index:       [bs, draft_token_num - 1] int64
// verified_seq_len:     [bs] int64
// tree_mask:            shape depends on tree_mask_mode; uint8 (bool or bitpacked)
// positions:            [bs, draft_token_num] int64, mutable
// retrive_index:        [bs, draft_token_num] int64, mutable
// retrive_next_token:   [bs, draft_token_num] int64, mutable
// retrive_next_sibling: [bs, draft_token_num] int64, mutable
// topk, depth, draft_token_num, tree_mask_mode: scalars
void build_tree_kernel_efficient(
    tvm::ffi::TensorView parent_list,
    tvm::ffi::TensorView selected_index,
    tvm::ffi::TensorView verified_seq_len,
    tvm::ffi::TensorView tree_mask,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView retrive_index,
    tvm::ffi::TensorView retrive_next_token,
    tvm::ffi::TensorView retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode) {
  using namespace host;

  RuntimeCheck(parent_list.device().device_type == kDLCUDA, "parent_list must be a CUDA tensor");
  RuntimeCheck(parent_list.ndim() == 2, "parent_list must be 2D: [bs, topk*(depth-1)+1]");
  RuntimeCheck(parent_list.is_contiguous(), "parent_list must be contiguous");
  RuntimeCheck(parent_list.dtype().code == kDLInt && parent_list.dtype().bits == 64, "parent_list must be int64");

  int bs = static_cast<int>(parent_list.size(0));

  RuntimeCheck(selected_index.ndim() == 2, "selected_index must be 2D: [bs, draft_token_num-1]");
  RuntimeCheck(selected_index.is_contiguous(), "selected_index must be contiguous");
  RuntimeCheck(
      selected_index.dtype().code == kDLInt && selected_index.dtype().bits == 64, "selected_index must be int64");
  RuntimeCheck(static_cast<int>(selected_index.size(0)) == bs, "selected_index batch_size mismatch");
  RuntimeCheck(selected_index.size(1) == draft_token_num - 1, "selected_index dim1 must equal draft_token_num - 1");

  RuntimeCheck(verified_seq_len.ndim() == 1, "verified_seq_len must be 1D: [bs]");
  RuntimeCheck(verified_seq_len.is_contiguous(), "verified_seq_len must be contiguous");
  RuntimeCheck(
      verified_seq_len.dtype().code == kDLInt && verified_seq_len.dtype().bits == 64, "verified_seq_len must be int64");
  RuntimeCheck(static_cast<int>(verified_seq_len.size(0)) == bs, "verified_seq_len batch_size mismatch");

  RuntimeCheck(tree_mask.is_contiguous(), "tree_mask must be contiguous");
  // tree_mask is bool (torch.bool → kDLBool) for FULL_MASK/QLEN_ONLY and
  // uint8 (torch.uint8 → kDLUInt) for QLEN_ONLY_BITPACKING.  Both are 1-byte
  // elements; check only the element size to accept either dtype.
  RuntimeCheck(host::dtype_bytes(tree_mask.dtype()) == 1, "tree_mask element size must be 1 byte (bool or uint8)");

  RuntimeCheck(positions.ndim() == 2, "positions must be 2D: [bs, draft_token_num]");
  RuntimeCheck(positions.is_contiguous(), "positions must be contiguous");
  RuntimeCheck(positions.dtype().code == kDLInt && positions.dtype().bits == 64, "positions must be int64");
  RuntimeCheck(
      static_cast<int>(positions.size(0)) == bs && positions.size(1) == draft_token_num, "positions shape mismatch");

  RuntimeCheck(retrive_index.ndim() == 2, "retrive_index must be 2D: [bs, draft_token_num]");
  RuntimeCheck(retrive_index.is_contiguous(), "retrive_index must be contiguous");
  RuntimeCheck(retrive_index.dtype().code == kDLInt && retrive_index.dtype().bits == 64, "retrive_index must be int64");
  RuntimeCheck(
      static_cast<int>(retrive_index.size(0)) == bs && retrive_index.size(1) == draft_token_num,
      "retrive_index shape mismatch");

  RuntimeCheck(retrive_next_token.ndim() == 2, "retrive_next_token must be 2D: [bs, draft_token_num]");
  RuntimeCheck(retrive_next_token.is_contiguous(), "retrive_next_token must be contiguous");
  RuntimeCheck(
      retrive_next_token.dtype().code == kDLInt && retrive_next_token.dtype().bits == 64,
      "retrive_next_token must be int64");
  RuntimeCheck(
      static_cast<int>(retrive_next_token.size(0)) == bs && retrive_next_token.size(1) == draft_token_num,
      "retrive_next_token shape mismatch");

  RuntimeCheck(retrive_next_sibling.ndim() == 2, "retrive_next_sibling must be 2D: [bs, draft_token_num]");
  RuntimeCheck(retrive_next_sibling.is_contiguous(), "retrive_next_sibling must be contiguous");
  RuntimeCheck(
      retrive_next_sibling.dtype().code == kDLInt && retrive_next_sibling.dtype().bits == 64,
      "retrive_next_sibling must be int64");
  RuntimeCheck(
      static_cast<int>(retrive_next_sibling.size(0)) == bs && retrive_next_sibling.size(1) == draft_token_num,
      "retrive_next_sibling shape mismatch");

  cudaStream_t stream = LaunchKernel::resolve_device(parent_list.device());
  dim3 grid(bs);
  dim3 block(static_cast<unsigned>(draft_token_num));

  if (tree_mask_mode == QLEN_ONLY_BITPACKING) {
    size_t num_bytes_per_item = 1;
    if (draft_token_num > 16) {
      num_bytes_per_item = 4;
    } else if (draft_token_num > 8) {
      num_bytes_per_item = 2;
    }
    LaunchKernel(grid, block, stream)(
        build_tree_efficient_partial_packed,
        static_cast<int64_t*>(parent_list.data_ptr()),
        static_cast<int64_t*>(selected_index.data_ptr()),
        static_cast<int64_t*>(verified_seq_len.data_ptr()),
        static_cast<uint8_t*>(tree_mask.data_ptr()),
        static_cast<int64_t*>(positions.data_ptr()),
        static_cast<int64_t*>(retrive_index.data_ptr()),
        static_cast<int64_t*>(retrive_next_token.data_ptr()),
        static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
        static_cast<int>(topk),
        static_cast<int>(depth),
        static_cast<int>(draft_token_num),
        num_bytes_per_item);
  } else {
    LaunchKernel(grid, block, stream)(
        build_tree_efficient,
        static_cast<int64_t*>(parent_list.data_ptr()),
        static_cast<int64_t*>(selected_index.data_ptr()),
        static_cast<int64_t*>(verified_seq_len.data_ptr()),
        static_cast<bool*>(tree_mask.data_ptr()),
        static_cast<int64_t*>(positions.data_ptr()),
        static_cast<int64_t*>(retrive_index.data_ptr()),
        static_cast<int64_t*>(retrive_next_token.data_ptr()),
        static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
        static_cast<int>(topk),
        static_cast<int>(depth),
        static_cast<int>(draft_token_num),
        static_cast<int>(tree_mask_mode));
  }
}

}  // namespace
