#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef USE_ROCM
#include "pytorch_extension_utils.h"
#else
#include "pytorch_extension_utils_rocm.h"
#endif

// tree_mask: [bs * draft_token_num * draft_token_num]
// verified_seq_len: [bs]
// positions: [bs * draft_token_num]
// retrive_index: [bs, draft_token_num]
// retrive_next_token: [bs, draft_token_num]
// retrive_next_sibling: [bs, draft_token_num]
__global__ void reconstructIndicesFromTreeMask(
    bool* tree_mask,
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
      static_cast<int64_t*>(verified_seq_len.data_ptr()),
      static_cast<int64_t*>(positions.data_ptr()),
      static_cast<int64_t*>(retrive_index.data_ptr()),
      static_cast<int64_t*>(retrive_next_token.data_ptr()),
      static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
      int(batch_size),
      int(draft_token_num));
}
