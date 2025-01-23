#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>

#include "flashinfer/vec_dtypes.cuh"
#include "pytorch_extension_utils.h"
#include "utils.h"

template <typename scalar_t>
__global__ void sampling_scaling_penalties_kernel(const scalar_t* logits, const scalar_t* scaling_penalties,
                                                  scalar_t* output, const int32_t numel) {
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t stride = blockDim.x * gridDim.x;

  constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
  using vec_t = flashinfer::vec_t<scalar_t, vec_size>;

  const int32_t num_vec_elems = numel / vec_size;

#pragma unroll 1
  for (int32_t i = tid; i < num_vec_elems; i += stride) {
    vec_t logits_vec, penalties_vec, out_vec;
    logits_vec.cast_load(logits + i * vec_size);
    penalties_vec.cast_load(scaling_penalties + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      out_vec[j] = logits_vec[j] > scalar_t(0.0f) ? logits_vec[j] / penalties_vec[j] : logits_vec[j] * penalties_vec[j];
    }

    out_vec.cast_store(output + i * vec_size);
  }

  // process the remaining elements
  const int32_t start_idx = num_vec_elems * vec_size;
  for (int32_t i = start_idx + tid; i < numel; i += stride) {
    scalar_t logit = logits[i];
    scalar_t penalty = scaling_penalties[i];
    output[i] = logit > scalar_t(0.0f) ? logit / penalty : logit * penalty;
  }
}

torch::Tensor sampling_scaling_penalties(const torch::Tensor& logits, const torch::Tensor& scaling_penalties) {
  auto output = torch::empty_like(logits);
  const auto numel = logits.numel();
  const int threads = 512;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(logits.scalar_type(), scalar_t, [&] {
    uint32_t vec_size = 16 / sizeof(scalar_t);
    const int blocks = (numel + threads * vec_size - 1) / (threads * vec_size);
    sampling_scaling_penalties_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        static_cast<scalar_t*>(logits.data_ptr()), static_cast<scalar_t*>(scaling_penalties.data_ptr()),
        static_cast<scalar_t*>(output.data_ptr()), numel);
    return true;
  });

  return output;
}
