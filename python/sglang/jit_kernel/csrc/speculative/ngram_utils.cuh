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
// https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/speculative/ngram_utils.cu

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

// tree_mask: [bs * draft_token_num * draft_token_num]
// verified_seq_len: [bs]
// positions: [bs * draft_token_num]
// retrive_index: [bs, draft_token_num]
// retrive_next_token: [bs, draft_token_num]
// retrive_next_sibling: [bs, draft_token_num]
__global__ void reconstructIndicesFromTreeMask(
    uint8_t* tree_mask,
    int64_t* verified_seq_len,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int batch_size,
    int draft_token_num) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid >= batch_size || tid >= draft_token_num) {
    return;
  }
  int base_offset = draft_token_num * draft_token_num;
  // token_idx: [bid * draft_token_num, (bid + 1) * draft_token_num)
  int token_idx = bid * draft_token_num;
  // tree_mask_idx: [bid * base_offset, (bid + 1) * base_offset)
  int tree_mask_offset = bid * base_offset;

  int depth = 0;
  int parent_idx = -1;

  for (int i = tid - 1, start_idx = tree_mask_offset + tid * draft_token_num; i >= 0; i--) {
    if (tree_mask[start_idx + i]) {
      depth++;
      if (parent_idx == -1) {
        parent_idx = i;
      }
    }
  }
  retrive_index[token_idx + tid] = token_idx + tid;
  positions[token_idx + tid] = depth + verified_seq_len[bid];

  int next_token_idx = -1;
  for (int i = tid + 1; i < draft_token_num; i++) {
    if (tree_mask[tree_mask_offset + i * draft_token_num + tid]) {
      next_token_idx = i;
      break;
    }
  }
  retrive_next_token[token_idx + tid] = next_token_idx;

  int next_sibling_idx = -1;
  if (parent_idx != -1) {
    for (int i = tid + 1; i < draft_token_num; i++) {
      int start_idx = tree_mask_offset + i * draft_token_num + parent_idx;
      if (tree_mask[start_idx]) {
        bool is_sibling = true;
        int end_idx = tree_mask_offset + i * draft_token_num + i;
        for (int j = start_idx + 1; j < end_idx; ++j) {
          if (tree_mask[j]) {
            is_sibling = false;
            break;
          }
        }
        if (is_sibling) {
          next_sibling_idx = i;
          break;
        }
      }
    }
  }
  retrive_next_sibling[token_idx + tid] = next_sibling_idx;
}

namespace {

// ---------------------------------------------------------------------------
// tvm-ffi entry point
// ---------------------------------------------------------------------------

// tree_mask:            [bs * draft_token_num * draft_token_num] bool
// verified_seq_len:     [bs] int64
// positions:            [bs * draft_token_num] int64, mutable
// retrive_index:        [bs, draft_token_num] int64, mutable
// retrive_next_token:   [bs, draft_token_num] int64, mutable
// retrive_next_sibling: [bs, draft_token_num] int64, mutable
// batch_size, draft_token_num: scalars
void reconstruct_indices_from_tree_mask(
    tvm::ffi::TensorView tree_mask,
    tvm::ffi::TensorView verified_seq_len,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView retrive_index,
    tvm::ffi::TensorView retrive_next_token,
    tvm::ffi::TensorView retrive_next_sibling,
    int64_t batch_size,
    int64_t draft_token_num) {
  using namespace host;

  RuntimeCheck(tree_mask.device().device_type == kDLCUDA, "tree_mask must be a CUDA tensor");
  RuntimeCheck(tree_mask.ndim() == 1, "tree_mask must be 1D: [bs * draft_token_num * draft_token_num]");
  RuntimeCheck(tree_mask.is_contiguous(), "tree_mask must be contiguous");
  RuntimeCheck(host::dtype_bytes(tree_mask.dtype()) == 1, "tree_mask element size must be 1 byte (bool or uint8)");
  RuntimeCheck(
      tree_mask.size(0) == batch_size * draft_token_num * draft_token_num,
      "tree_mask size must equal batch_size * draft_token_num * draft_token_num");

  RuntimeCheck(verified_seq_len.ndim() == 1, "verified_seq_len must be 1D: [bs]");
  RuntimeCheck(verified_seq_len.is_contiguous(), "verified_seq_len must be contiguous");
  RuntimeCheck(
      verified_seq_len.dtype().code == kDLInt && verified_seq_len.dtype().bits == 64, "verified_seq_len must be int64");
  RuntimeCheck(verified_seq_len.size(0) == batch_size, "verified_seq_len size must equal batch_size");

  RuntimeCheck(positions.ndim() == 1, "positions must be 1D: [bs * draft_token_num]");
  RuntimeCheck(positions.is_contiguous(), "positions must be contiguous");
  RuntimeCheck(positions.dtype().code == kDLInt && positions.dtype().bits == 64, "positions must be int64");
  RuntimeCheck(
      positions.size(0) == batch_size * draft_token_num, "positions size must equal batch_size * draft_token_num");

  RuntimeCheck(retrive_index.ndim() == 2, "retrive_index must be 2D: [bs, draft_token_num]");
  RuntimeCheck(retrive_index.is_contiguous(), "retrive_index must be contiguous");
  RuntimeCheck(retrive_index.dtype().code == kDLInt && retrive_index.dtype().bits == 64, "retrive_index must be int64");
  RuntimeCheck(
      retrive_index.size(0) == batch_size && retrive_index.size(1) == draft_token_num,
      "retrive_index shape must be [batch_size, draft_token_num]");

  RuntimeCheck(retrive_next_token.ndim() == 2, "retrive_next_token must be 2D: [bs, draft_token_num]");
  RuntimeCheck(retrive_next_token.is_contiguous(), "retrive_next_token must be contiguous");
  RuntimeCheck(
      retrive_next_token.dtype().code == kDLInt && retrive_next_token.dtype().bits == 64,
      "retrive_next_token must be int64");
  RuntimeCheck(
      retrive_next_token.size(0) == batch_size && retrive_next_token.size(1) == draft_token_num,
      "retrive_next_token shape must be [batch_size, draft_token_num]");

  RuntimeCheck(retrive_next_sibling.ndim() == 2, "retrive_next_sibling must be 2D: [bs, draft_token_num]");
  RuntimeCheck(retrive_next_sibling.is_contiguous(), "retrive_next_sibling must be contiguous");
  RuntimeCheck(
      retrive_next_sibling.dtype().code == kDLInt && retrive_next_sibling.dtype().bits == 64,
      "retrive_next_sibling must be int64");
  RuntimeCheck(
      retrive_next_sibling.size(0) == batch_size && retrive_next_sibling.size(1) == draft_token_num,
      "retrive_next_sibling shape must be [batch_size, draft_token_num]");

  cudaStream_t stream = LaunchKernel::resolve_device(tree_mask.device());
  dim3 grid(static_cast<unsigned>(batch_size));
  dim3 block(static_cast<unsigned>(draft_token_num));

  LaunchKernel(grid, block, stream)(
      reconstructIndicesFromTreeMask,
      static_cast<uint8_t*>(tree_mask.data_ptr()),
      static_cast<int64_t*>(verified_seq_len.data_ptr()),
      static_cast<int64_t*>(positions.data_ptr()),
      static_cast<int64_t*>(retrive_index.data_ptr()),
      static_cast<int64_t*>(retrive_next_token.data_ptr()),
      static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
      static_cast<int>(batch_size),
      static_cast<int>(draft_token_num));
}

}  // namespace
