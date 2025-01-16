#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>

#include "utils.hpp"
#include "vectorization.cuh"

template <typename scalar_t>
__global__ void sampling_scaling_penalties_kernel(const scalar_t* logits, const scalar_t* scaling_penalties,
                                                  scalar_t* output, const int32_t numel) {
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t stride = blockDim.x * gridDim.x;

  auto const* vectorized_logits = reinterpret_cast<vec4_t<scalar_t> const*>(logits);
  auto const* vectorized_penalties = reinterpret_cast<vec4_t<scalar_t> const*>(scaling_penalties);
  auto* vectorized_output = reinterpret_cast<vec4_t<scalar_t>*>(output);

  const int32_t num_vec_elems = numel >> 2;

#pragma unroll 4
  for (int32_t i = tid; i < num_vec_elems; i += stride) {
    vec4_t<scalar_t> logits_vec = vectorized_logits[i];
    vec4_t<scalar_t> penalties_vec = vectorized_penalties[i];
    vec4_t<scalar_t> out_vec;

    out_vec.x = logits_vec.x > 0 ? logits_vec.x / penalties_vec.x : logits_vec.x * penalties_vec.x;
    out_vec.y = logits_vec.y > 0 ? logits_vec.y / penalties_vec.y : logits_vec.y * penalties_vec.y;
    out_vec.z = logits_vec.z > 0 ? logits_vec.z / penalties_vec.z : logits_vec.z * penalties_vec.z;
    out_vec.w = logits_vec.w > 0 ? logits_vec.w / penalties_vec.w : logits_vec.w * penalties_vec.w;

    vectorized_output[i] = out_vec;
  }

  const int32_t start_idx = num_vec_elems * 4;
  for (int32_t i = start_idx + tid; i < numel; i += stride) {
    scalar_t logit = logits[i];
    scalar_t penalty = scaling_penalties[i];
    output[i] = logit > 0 ? logit / penalty : logit * penalty;
  }
}

torch::Tensor sampling_scaling_penalties(const torch::Tensor& logits, const torch::Tensor& scaling_penalties) {
  auto output = torch::empty_like(logits);
  const auto numel = logits.numel();
  const int threads = 512;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(), "sampling_scaling_penalties_kernel", ([&] {
        const int blocks = (numel + threads * 4 - 1) / (threads * 4);
        sampling_scaling_penalties_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            logits.data_ptr<scalar_t>(), scaling_penalties.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), numel);
      }));

  return output;
}
