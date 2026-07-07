// InfLLM-V2 variable-length max pooling, AOT build.
//
// Migrated from `3rdparty/infllmv2_cuda_impl/csrc/max_pooling_1d.cuh`. The
// device kernels are kept faithful to the original; only the host-side
// launchers are rewritten from the raw `cudaStream_t` + `data_ptr` pybind
// interface to the sgl-kernel `at::Tensor` + torch.ops convention.
//
// Notes vs. the original implementation:
//   * `TypeTraits<T>::inf()` is replaced by `static_cast<T>(INFINITY)`, and the
//     pooling max is accumulated in fp32 so we don't rely on half/bf16
//     comparison operators.
//   * Outputs are pre-allocated on the Python side and passed in. The kernel
//     writes every element for packed varlen inputs, so no pre-zeroing is needed.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "utils.h"

namespace {

// input:  [num_heads, total_q, max_seqlen_k]
// output: [num_heads, total_q, out_len]
template <typename T>
__global__ void max_pooling_1d_varlen_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_k,
    const int* __restrict__ cache_lens,
    int batch_size,
    int num_heads,
    int max_seqlen_k,
    int out_len,
    int kernel_size,
    int stride,
    int padding,
    int block_size,
    int local_blocks,
    int init_blocks) {
  const int bidh = blockIdx.y;         // head index
  const int bidq_global = blockIdx.x;  // global query index across all batches

  int lo = 0;
  int hi = batch_size;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (bidq_global >= cu_seqlens_q[mid + 1]) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int batch_idx = lo;

  const int q_start = cu_seqlens_q[batch_idx];
  const int bidq_local = bidq_global - q_start;
  const int seqlen_q = cu_seqlens_q[batch_idx + 1] - q_start;
  const int seqlen_k = cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx];
  if (bidq_local >= seqlen_q) return;

  const size_t total_q_all = static_cast<size_t>(cu_seqlens_q[batch_size]);
  const size_t in_offset =
      static_cast<size_t>(bidh) * total_q_all * max_seqlen_k + static_cast<size_t>(bidq_global) * max_seqlen_k;
  const T* in = input + in_offset;
  const size_t out_offset =
      static_cast<size_t>(bidh) * total_q_all * out_len + static_cast<size_t>(bidq_global) * out_len;
  T* out = output + out_offset;

  const int cache_len = cache_lens[batch_idx];
  const int off_bq = (bidq_local + cache_len) / block_size;
  const T pos_inf = static_cast<T>(static_cast<float>(INFINITY));

  for (int k = threadIdx.x; k < out_len; k += blockDim.x) {
    const int off_bk = k;
    const bool should_mask_inf = (off_bk < init_blocks) || ((off_bq >= off_bk) && (off_bq <= off_bk + local_blocks));

    if (should_mask_inf) {
      out[k] = pos_inf;
    } else {
      int start = k * stride - padding;
      int end = start + kernel_size;
      start = max(start, 0);
      end = min(end, seqlen_k);

      float max_val = -INFINITY;
      for (int i = start; i < end; i++) {
        const float v = static_cast<float>(in[i]);
        if (v > max_val) max_val = v;
      }
      out[k] = static_cast<T>(max_val);
    }
  }
}

}  // namespace

void infllm_v2_max_pooling_1d_varlen(
    at::Tensor input,
    at::Tensor output,
    at::Tensor cu_seqlens_q,
    at::Tensor cu_seqlens_k,
    at::Tensor cache_lens,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t block_size,
    int64_t local_blocks,
    int64_t init_blocks,
    int64_t total_q) {
  TORCH_CHECK(input.dim() == 3, "input must be 3D [num_heads, total_q, max_k]");
  TORCH_CHECK(output.dim() == 3, "output must be 3D [num_heads, total_q, out_len]");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt, "cu_seqlens_q must be int32");
  TORCH_CHECK(cu_seqlens_k.scalar_type() == at::kInt, "cu_seqlens_k must be int32");
  TORCH_CHECK(cache_lens.scalar_type() == at::kInt, "cache_lens must be int32");

  const int batch_size = static_cast<int>(cu_seqlens_q.size(0)) - 1;
  const int num_heads = static_cast<int>(input.size(0));
  const int out_len = static_cast<int>(output.size(2));
  const int grid_q = static_cast<int>(total_q > 0 ? total_q : input.size(1));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(grid_q, num_heads);
  const dim3 block(256);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    max_pooling_1d_varlen_kernel<c_type><<<grid, block, 0, stream>>>(
        static_cast<const c_type*>(input.data_ptr()),
        static_cast<c_type*>(output.data_ptr()),
        cu_seqlens_q.data_ptr<int>(),
        cu_seqlens_k.data_ptr<int>(),
        cache_lens.data_ptr<int>(),
        batch_size,
        num_heads,
        static_cast<int>(max_seqlen_k),
        out_len,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(block_size),
        static_cast<int>(local_blocks),
        static_cast<int>(init_blocks));
    return true;
  });
}
