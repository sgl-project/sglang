#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef USE_ROCM
#include "pytorch_extension_utils.h"
#else
#include "pytorch_extension_utils_rocm.h"
#endif

template <typename IdType>
__global__ void LookaheadVerifyTreeGreedy(
    IdType* accept_index,       // mutable
    IdType* accept_token_ids,   // mutable
    IdType* accept_token_num,   // mutable
    IdType* total_accept_num,   // mutable
    IdType* last_verified_ids,  // mutable
    IdType* candidates,
    IdType* retrive_index,
    IdType* retrive_next_token,
    IdType* retrive_next_sibling,
    IdType* target_predict,
    uint32_t batch_size,
    uint32_t num_draft_tokens,
    uint32_t eos_token_id) {
  uint32_t bx = blockIdx.x;

  IdType offset = bx * num_draft_tokens;
  IdType last_accepted_retrive_idx = retrive_index[offset];
  accept_index[offset] = last_accepted_retrive_idx;
  accept_token_ids[offset] = target_predict[last_accepted_retrive_idx];
  uint32_t num_accepted_tokens = 1;
  IdType cur_index = 0;
  bool is_eos = false;

  for (uint32_t j = 1; j < num_draft_tokens && !is_eos; ++j) {
    cur_index = retrive_next_token[offset + cur_index];
    while (cur_index != -1) {
      IdType draft_index = retrive_index[offset + cur_index];
      IdType draft_token_id = candidates[offset + cur_index];
      IdType target_token_id = target_predict[last_accepted_retrive_idx];
      if (draft_token_id == target_token_id) {
        if (draft_token_id == eos_token_id) {
          is_eos = true;
          break;
        }

        accept_token_ids[offset + num_accepted_tokens] = target_predict[draft_index];
        accept_index[offset + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        ++num_accepted_tokens;
        break;
      } else {
        cur_index = retrive_next_sibling[offset + cur_index];
      }
    }
    if (cur_index == -1) break;
  }

  for (int i = num_accepted_tokens; i < num_draft_tokens; ++i) {
    accept_token_ids[offset + i] = -1;
  }

  accept_token_num[bx] = num_accepted_tokens;
  last_verified_ids[bx] = target_predict[last_accepted_retrive_idx];
  atomicAdd(&total_accept_num[0], num_accepted_tokens);
}

template <typename IdType>
__global__ void AcceptFlattenIndex(
    IdType* accept_index,
    IdType* flatten_index,  // mutable
    IdType* accept_token_num,
    IdType* total_accept_num,
    uint32_t num_draft_tokens) {
  uint32_t bx = blockIdx.x;

  int start_flatten_accept_idx = 0;
  int start_flatten_evict_idx = total_accept_num[0];
  for (int i = 0; i < bx; i++) {
    start_flatten_accept_idx += accept_token_num[i];
    start_flatten_evict_idx += num_draft_tokens - accept_token_num[i];
  }

  int start_accept_idx = bx * num_draft_tokens;
  int last_accept_idx = start_accept_idx - 1;
  for (int i = 0; i < accept_token_num[bx]; i++) {
    flatten_index[start_flatten_accept_idx + i] = accept_index[start_accept_idx + i];
    for (int j = last_accept_idx + 1; j < accept_index[start_accept_idx + i]; j++) {
      flatten_index[start_flatten_evict_idx++] = j;
    }
    last_accept_idx = accept_index[start_accept_idx + i];
  }
  int end_flatten_evict_idx = (bx + 1) * num_draft_tokens;
  for (int i = last_accept_idx + 1; i < end_flatten_evict_idx; i++) {
    flatten_index[start_flatten_evict_idx++] = i;
  }
}

// accept_token_num: [bs]
// accept_token_ids: [bs, num_draft_tokens]
// last_verified_ids: [bs]
// flatten_index: [bs*num_draft_tokens]
// total_accept_num: [1]
// candidates: [bs, num_draft_tokens]
// retrive_index: [bs, num_draft_tokens]
// retrive_next_token: [bs, num_draft_tokens]
// retrive_next_sibling: [bs, num_draft_tokens]
// target_predict: [bs, num_draft_tokens]
void lookahead_verify_tree_greedy(
    at::Tensor accept_token_num,   // mutable
    at::Tensor accept_token_ids,   // mutable
    at::Tensor last_verified_ids,  // mutable
    at::Tensor flatten_index,      // mutable
    at::Tensor total_accept_num,   // mutable
    at::Tensor candidates,
    at::Tensor retrive_index,  // 感觉这个不需要
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor target_predict,
    int64_t eos_token_id) {
  CHECK_INPUT(candidates);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_INPUT(target_predict);
  auto device = target_predict.device();
  CHECK_EQ(candidates.device(), device);
  CHECK_EQ(retrive_index.device(), device);
  CHECK_EQ(retrive_next_token.device(), device);
  CHECK_EQ(retrive_next_sibling.device(), device);
  CHECK_EQ(target_predict.device(), device);
  CHECK_DIM(1, accept_token_num);
  CHECK_DIM(1, total_accept_num);
  CHECK_DIM(2, accept_token_ids);
  CHECK_DIM(1, candidates);
  CHECK_DIM(2, retrive_index);
  CHECK_DIM(2, retrive_next_token);
  CHECK_DIM(2, retrive_next_sibling);
  CHECK_DIM(1, target_predict);
  unsigned int batch_size = accept_token_ids.size(0);
  unsigned int num_draft_tokens = accept_token_ids.size(1);
  CHECK_EQ(batch_size, accept_token_num.size(0));
  CHECK_EQ(1, total_accept_num.size(0));
  CHECK_EQ(batch_size, retrive_index.size(0));
  CHECK_EQ(batch_size, retrive_next_token.size(0));
  CHECK_EQ(batch_size, retrive_next_sibling.size(0));
  CHECK_EQ(num_draft_tokens, retrive_index.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_token.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_sibling.size(1));
  CHECK_EQ(batch_size, accept_token_num.size(0));

  if (accept_token_num.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'accept_token_num' to be of type int (torch.int32).");
  }
  if (candidates.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'candidates' to be of type int (torch.int32).");
  }
  if (retrive_index.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'retrive_index' to be of type int (torch.int32).");
  }
  if (retrive_next_token.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'retrive_next_token' to be of type int (torch.int32).");
  }
  if (retrive_next_sibling.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'retrive_next_sibling' to be of type int (torch.int32).");
  }
  if (target_predict.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'target_predict' to be of type int (torch.int32).");
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(batch_size);
  dim3 block(1);

  at::TensorOptions options = at::TensorOptions().dtype(at::kInt).device(device);
  at::Tensor accept_index = at::full({batch_size, num_draft_tokens}, -1, options);
  total_accept_num[0] = 0;

  LookaheadVerifyTreeGreedy<int><<<grid, block, 0, stream>>>(
      static_cast<int*>(accept_index.data_ptr()),
      static_cast<int*>(accept_token_ids.data_ptr()),
      static_cast<int*>(accept_token_num.data_ptr()),
      static_cast<int*>(total_accept_num.data_ptr()),
      static_cast<int*>(last_verified_ids.data_ptr()),
      static_cast<int*>(candidates.data_ptr()),
      static_cast<int*>(retrive_index.data_ptr()),
      static_cast<int*>(retrive_next_token.data_ptr()),
      static_cast<int*>(retrive_next_sibling.data_ptr()),
      static_cast<int*>(target_predict.data_ptr()),
      batch_size,
      num_draft_tokens,
      eos_token_id);

  AcceptFlattenIndex<int><<<grid, block, 0, stream>>>(
      static_cast<int*>(accept_index.data_ptr()),
      static_cast<int*>(flatten_index.data_ptr()),
      static_cast<int*>(accept_token_num.data_ptr()),
      static_cast<int*>(total_accept_num.data_ptr()),
      num_draft_tokens);
}

// tree_mask: [bs*draft_token_num * draft_token_num]
// verified_seq_len: [bs]
// positions: [bs * draft_token_num]
// retrive_index: [bs, draft_token_num]
// retrive_next_token: [bs, draft_token_num]
// retrive_next_sibling: [bs, draft_token_num]
__global__ void reconstructIndicesFromTreeMask(
    bool* tree_mask,
    int* verified_seq_len,
    int* positions,
    int* retrive_index,
    int* retrive_next_token,
    int* retrive_next_sibling,
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

  // 按列查找第 tid 列 tid+1 行开始的第一个不为 false 的元素作为 next_token_idx
  int next_token_idx = -1;
  for (int i = tid + 1; i < draft_token_num; i++) {
    if (tree_mask[tree_mask_offset + i * draft_token_num + tid]) {
      next_token_idx = i;
      break;
    }
  }
  retrive_next_token[token_idx + tid] = next_token_idx;

  // 根据 parent_idx 查找 next_sibling_idx
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

void reconstruct_indices_from_tree_mask(
    at::Tensor tree_mask,
    at::Tensor verified_seq_len,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t batch_size,
    int64_t draft_token_num) {
  dim3 grid(batch_size);
  dim3 block(draft_token_num);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  reconstructIndicesFromTreeMask<<<grid, block, 0, stream>>>(
      static_cast<bool*>(tree_mask.data_ptr()),
      static_cast<int*>(verified_seq_len.data_ptr()),
      static_cast<int*>(positions.data_ptr()),
      static_cast<int*>(retrive_index.data_ptr()),
      static_cast<int*>(retrive_next_token.data_ptr()),
      static_cast<int*>(retrive_next_sibling.data_ptr()),
      int(batch_size),
      int(draft_token_num));
}
