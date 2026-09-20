// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "pr_hot.cuh"

namespace {
__global__ void gather_contiguous(const uint4* src, const float* scale,
    uint4* dst, float* dst_scale, const int* permutation, int count) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = v / 8, part = v % 8;
  if (row < count) {
    const int source = permutation[row];
    dst[v] = src[(int64_t)source*8+part];
    if (part == 0) dst_scale[row] = scale[source];
  }
}

void prepare(torch::Tensor hot, torch::Tensor epoch_storage, torch::Tensor permutation,
    torch::Tensor swap_a, torch::Tensor swap_b, torch::Tensor counts, int common_end,
    torch::Tensor src, torch::Tensor scales, torch::Tensor dst, torch::Tensor dst_scales) {
  TORCH_CHECK(hot.numel() == 12288 && hot.scalar_type() == torch::kLong);
  TORCH_CHECK(common_end >= 12288 && common_end <= src.size(0));
  auto stream = c10::cuda::getCurrentCUDAStream();
  // A fixed epoch plus device reset works for arbitrary CUDA graph replays.
  C10_CUDA_CHECK(cudaMemsetAsync(epoch_storage.data_ptr(), 0, epoch_storage.nbytes(), stream));
  const int64_t* hot_ptr = hot.data_ptr<int64_t>();
  int* ep = epoch_storage.data_ptr<int>();
  int* perm = permutation.data_ptr<int>();
  int* a = swap_a.data_ptr<int>();
  int* b = swap_b.data_ptr<int>();
  int* cnt = counts.data_ptr<int>();
  int size = 12288, window_start = 0, epoch = 1;
  void* args[] = {&hot_ptr, &ep, &perm, &a, &b, &cnt, &size, &window_start, &common_end, &epoch};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      (const void*)pair_swap_gather::cooperative_plan_kernel<int64_t>, dim3(48), dim3(256), args, 0, stream));
  gather_contiguous<<<(src.size(0)*8+255)/256,256,0,stream>>>(
      reinterpret_cast<const uint4*>(src.data_ptr()), scales.data_ptr<float>(),
      reinterpret_cast<uint4*>(dst.data_ptr()), dst_scales.data_ptr<float>(),
      perm, src.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void bucket_edges(const int* buckets, float* edge, int Q) {
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  if (row < Q) edge[row] = float(buckets[row]+1);
}

void seed(torch::Tensor scores, torch::Tensor origin, torch::Tensor inv,
    torch::Tensor buckets, torch::Tensor edge, torch::Tensor values,
    torch::Tensor indices, torch::Tensor counts, torch::Tensor histogram) {
  TORCH_CHECK(scores.size(1) == 12288 && values.size(1) >= 12288);
  auto stream = c10::cuda::getCurrentCUDAStream();
  seed_prep_kernel<true,12288,256><<<scores.size(0),256,4*256*sizeof(int),stream>>>(
      scores.data_ptr<float>(), scores.stride(0), 12288, 256, 2048, 0.0f,
      origin.data_ptr<float>(), inv.data_ptr<float>(), buckets.data_ptr<int>(),
      values.data_ptr<float>(), indices.data_ptr<int>(), counts.data_ptr<int>(),
      values.size(1), 0, histogram.data_ptr<int>());
  bucket_edges<<<(scores.size(0)+255)/256,256,0,stream>>>(
      buckets.data_ptr<int>(), edge.data_ptr<float>(), scores.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void map_vote_kernel(int* out, const int* permutation, int* votes, int Q, int K) {
  const int row = blockIdx.x;
  for (int j = threadIdx.x; j < K; j += blockDim.x) {
    const int64_t off = (int64_t)row*K+j;
    const int physical = out[off];
    const int original = physical >= 0 ? permutation[physical] : -1;
    out[off] = original;
    if (original >= 0 && row >= max(0,Q-1536)) atomicAdd(votes+original,1);
  }
}

void map_vote(torch::Tensor out, torch::Tensor permutation, torch::Tensor votes) {
  map_vote_kernel<<<out.size(0),256,0,c10::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<int>(), permutation.data_ptr<int>(), votes.data_ptr<int>(),
      out.size(0), out.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &prepare);
  m.def("seed", &seed);
  m.def("map_vote", &map_vote);
  m.def("carry", &carry_votes_topk_reset_litetopk_);
}
