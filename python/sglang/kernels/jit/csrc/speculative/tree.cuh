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
  SymbolicDType mask_dtype;
  int64_t mask_size = bs * draft_token_num;
  if (tree_mask_mode == speculative::QLEN_ONLY_BITPACKING) {
    CHECK_HOST(draft_token_num <= 32);
    const uint8_t bits = draft_token_num > 16 ? 32 : (draft_token_num > 8 ? 16 : 8);
    mask_dtype.set_value(DLDataType{kDLUInt, bits, 1});
  } else {
    mask_dtype.set_value(DLDataType{kDLBool, 8, 1});
    mask_size = tree_mask_mode == speculative::QLEN_ONLY ? mask_size * draft_token_num : -1;
  }
  TensorMatcher({mask_size}).with_dtype(mask_dtype).with_device(device).verify(tree_mask);
  if (bs == 0) return;

  auto launch = LaunchKernel(static_cast<uint32_t>(bs), static_cast<uint32_t>(draft_token_num), device.unwrap());
  if (tree_mask_mode == speculative::QLEN_ONLY_BITPACKING) {
    const size_t num_bytes_per_item = tree_mask.dtype().bits / 8;
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
  SymbolicDType mask_dtype;
  mask_dtype.set_value(DLDataType{kDLBool, 8, 1});
  CHECK_HOST(batch_size >= 0 && draft_token_num > 0 && draft_token_num <= 1024);
  TensorMatcher({batch_size * draft_token_num * draft_token_num})
      .with_dtype(mask_dtype)
      .with_device<kDLCUDA>(device)
      .verify(tree_mask);
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
