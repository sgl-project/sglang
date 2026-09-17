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

#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/speculative/eagle.cuh>
#include <sgl_kernel/speculative/ngram.cuh>

namespace sglang {

/// \brief Build the EAGLE tree and its retrieval links in preallocated buffers.
inline void build_tree_kernel_efficient(
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
  SymbolicSize batch_size{"batch_size"}, parent_width{"parent_width"};
  SymbolicDevice device;
  CHECK_HOST(topk > 0 && depth > 0);
  CHECK_HOST(draft_token_num > 0 && draft_token_num <= 1024);
  CHECK_HOST(tree_mask_mode >= 0 && tree_mask_mode <= 2);
  TensorMatcher({batch_size, parent_width})
      .with_strides({parent_list.size(1) == 0 ? -1 : parent_list.size(1), 1})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device)
      .verify(parent_list);
  CHECK_HOST(depth == 1 || parent_width.unwrap() == topk * (depth - 1) + 1);
  TensorMatcher({batch_size, draft_token_num - 1})
      .with_strides({draft_token_num == 1 ? -1 : draft_token_num - 1, 1})
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(selected_index);
  TensorMatcher({batch_size}).with_dtype<int64_t>().with_device(device).verify(verified_seq_len);
  TensorMatcher({batch_size, draft_token_num})
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(retrive_index)
      .verify(retrive_next_token)
      .verify(retrive_next_sibling);
  const int64_t bs = batch_size.unwrap();
  TensorMatcher({bs * draft_token_num}).with_dtype<int64_t>().with_device(device).verify(positions);
  // The mask is raw bytes to these kernels, and callers disagree on how they
  // spell that: bool for an inline allocation or the default preallocated
  // buffer, uint8 for the triton backend's, uint{8,16,32} for a bit-packed one.
  // So constrain its BYTES, never its element type. The bound is a lower one --
  // a buffer preallocated for the captured max batch is larger than what this
  // batch writes, and FULL_MASK spans each request's context length so it has
  // no host-side bound at all.
  size_t num_bytes_per_item = 1;
  int64_t min_mask_bytes = 0;
  if (tree_mask_mode == speculative::QLEN_ONLY_BITPACKING) {
    CHECK_HOST(draft_token_num <= 32);
    // Width comes from draft_token_num alone, exactly as the AOT launcher does.
    num_bytes_per_item = draft_token_num > 16 ? 4 : (draft_token_num > 8 ? 2 : 1);
    min_mask_bytes = bs * draft_token_num * static_cast<int64_t>(num_bytes_per_item);
  } else if (tree_mask_mode == speculative::QLEN_ONLY) {
    min_mask_bytes = bs * draft_token_num * draft_token_num;
  }
  // -1 leaves the extent unconstrained; the byte bound below is the real guard.
  TensorMatcher({-1}).with_device(device).verify(tree_mask);
  const int64_t mask_elem_bytes = tree_mask.dtype().bits / 8;
  CHECK_HOST(tree_mask.dtype().bits % 8 == 0 && mask_elem_bytes > 0);
  CHECK_HOST(tree_mask.numel() * mask_elem_bytes >= min_mask_bytes);
  if (bs == 0) return;

  auto launch = LaunchKernel(static_cast<uint32_t>(bs), static_cast<uint32_t>(draft_token_num), device.unwrap());
  if (tree_mask_mode == speculative::QLEN_ONLY_BITPACKING) {
    launch(
        speculative::build_tree_efficient_partial_packed,
        static_cast<int64_t*>(parent_list.data_ptr()),
        static_cast<int64_t*>(selected_index.data_ptr()),
        static_cast<int64_t*>(verified_seq_len.data_ptr()),
        static_cast<uint8_t*>(tree_mask.data_ptr()),
        static_cast<int64_t*>(positions.data_ptr()),
        static_cast<int64_t*>(retrive_index.data_ptr()),
        static_cast<int64_t*>(retrive_next_token.data_ptr()),
        static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
        static_cast<int32_t>(topk),
        static_cast<int32_t>(depth),
        static_cast<int32_t>(draft_token_num),
        num_bytes_per_item);
  } else {
    launch(
        speculative::build_tree_efficient,
        static_cast<int64_t*>(parent_list.data_ptr()),
        static_cast<int64_t*>(selected_index.data_ptr()),
        static_cast<int64_t*>(verified_seq_len.data_ptr()),
        static_cast<bool*>(tree_mask.data_ptr()),
        static_cast<int64_t*>(positions.data_ptr()),
        static_cast<int64_t*>(retrive_index.data_ptr()),
        static_cast<int64_t*>(retrive_next_token.data_ptr()),
        static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
        static_cast<int32_t>(topk),
        static_cast<int32_t>(depth),
        static_cast<int32_t>(draft_token_num),
        static_cast<int32_t>(tree_mask_mode));
  }
}

/// \brief Verify a draft tree against greedy target predictions.
inline void verify_tree_greedy(
    tvm::ffi::TensorView predicts,
    tvm::ffi::TensorView accept_index,
    tvm::ffi::TensorView accept_token_num,
    tvm::ffi::TensorView candidates,
    tvm::ffi::TensorView retrive_index,
    tvm::ffi::TensorView retrive_next_token,
    tvm::ffi::TensorView retrive_next_sibling,
    tvm::ffi::TensorView target_predict) {
  using namespace host;
  SymbolicSize batch_size{"batch_size"}, draft_tokens{"draft_tokens"}, spec_tokens{"spec_tokens"};
  SymbolicDevice device;
  TensorMatcher({batch_size, draft_tokens})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device)
      .verify(candidates)
      .verify(retrive_index)
      .verify(retrive_next_token)
      .verify(retrive_next_sibling)
      .verify(target_predict);
  TensorMatcher({batch_size, spec_tokens}).with_dtype<int32_t>().with_device(device).verify(accept_index);
  TensorMatcher({batch_size}).with_dtype<int32_t>().with_device(device).verify(accept_token_num);
  TensorMatcher({batch_size.unwrap() * draft_tokens.unwrap()})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(predicts);
  CHECK_HOST(draft_tokens.unwrap() > 0 && spec_tokens.unwrap() > 0);
  if (batch_size.unwrap() == 0) return;
  LaunchKernel(static_cast<uint32_t>(batch_size.unwrap()), 1, device.unwrap())(
      speculative::VerifyTreeGreedy<int32_t, int64_t>,
      static_cast<int32_t*>(predicts.data_ptr()),
      static_cast<int32_t*>(accept_index.data_ptr()),
      static_cast<int32_t*>(accept_token_num.data_ptr()),
      static_cast<int64_t*>(candidates.data_ptr()),
      static_cast<int64_t*>(retrive_index.data_ptr()),
      static_cast<int64_t*>(retrive_next_token.data_ptr()),
      static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
      static_cast<int64_t*>(target_predict.data_ptr()),
      static_cast<uint32_t>(batch_size.unwrap()),
      static_cast<uint32_t>(spec_tokens.unwrap()),
      static_cast<uint32_t>(draft_tokens.unwrap()));
}

/// \brief Reconstruct retrieval links and positions from an ngram tree mask.
inline void reconstruct_indices_from_tree_mask(
    tvm::ffi::TensorView tree_mask,
    tvm::ffi::TensorView verified_seq_len,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView retrive_index,
    tvm::ffi::TensorView retrive_next_token,
    tvm::ffi::TensorView retrive_next_sibling,
    int64_t batch_size,
    int64_t draft_token_num) {
  using namespace host;
  SymbolicDevice device;
  CHECK_HOST(batch_size >= 0 && draft_token_num > 0 && draft_token_num <= 1024);
  // Bytes, not element type -- same reasoning as build_tree_kernel_efficient:
  // the kernel casts straight to bool* and callers are free to spell a 1-byte
  // mask as bool or uint8.
  TensorMatcher({-1}).with_device<kDLCUDA>(device).verify(tree_mask);
  CHECK_HOST(tree_mask.dtype().bits == 8);
  CHECK_HOST(tree_mask.numel() >= batch_size * draft_token_num * draft_token_num);
  TensorMatcher({batch_size}).with_dtype<int64_t>().with_device(device).verify(verified_seq_len);
  TensorMatcher({batch_size * draft_token_num}).with_dtype<int64_t>().with_device(device).verify(positions);
  TensorMatcher({batch_size, draft_token_num})
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(retrive_index)
      .verify(retrive_next_token)
      .verify(retrive_next_sibling);
  if (batch_size == 0) return;
  LaunchKernel(static_cast<uint32_t>(batch_size), static_cast<uint32_t>(draft_token_num), device.unwrap())(
      speculative::reconstructIndicesFromTreeMask,
      static_cast<bool*>(tree_mask.data_ptr()),
      static_cast<int64_t*>(verified_seq_len.data_ptr()),
      static_cast<int64_t*>(positions.data_ptr()),
      static_cast<int64_t*>(retrive_index.data_ptr()),
      static_cast<int64_t*>(retrive_next_token.data_ptr()),
      static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
      static_cast<int32_t>(batch_size),
      static_cast<int32_t>(draft_token_num));
}

}  // namespace sglang
