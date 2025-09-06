#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef USE_ROCM
#include "pytorch_extension_utils.h"
#else
#include "pytorch_extension_utils_rocm.h"
#endif

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
