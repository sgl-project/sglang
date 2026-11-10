// SPDX-License-Identifier: Apache-2.0
// Parallel reduction of QSA split-attention outputs on Volta.
#include <cuda_fp16.h>
#include <sgl_kernel/tensor.h>
#include <tvm/ffi/container/tensor.h>

#include <cub/block/block_reduce.cuh>
#include <sgl_kernel/utils.cuh>

namespace sglang::sm70_qsa_combine {
__global__ void kernel(const half* __restrict__ partial,
                       const float* __restrict__ lse,
                       const int* __restrict__ lengths, half* __restrict__ out,
                       int splits, int selected_tokens, int tokens_per_split) {
  constexpr int NT = 128, D_TILE = 16;
  const int batch = blockIdx.z, head = blockIdx.y;
  const int d = blockIdx.x * D_TILE + threadIdx.x % D_TILE;
  const int group = threadIdx.x / D_TILE;
  const int context = min(lengths[batch], selected_tokens);
  const int active =
      min(splits, max(1, (context + tokens_per_split - 1) / tokens_per_split));
  using Reduce = cub::BlockReduce<float, NT>;
  __shared__ typename Reduce::TempStorage temp;
  __shared__ float maximum, denominator, weights[160], sums[8][D_TILE];
  float local_max = -1073741824.f;
  for (int s = threadIdx.x; s < active; s += NT)
    local_max = fmaxf(local_max, lse[(batch * splits + s) * 6 + head]);
  float mx = Reduce(temp).Reduce(local_max, cub::Max());
  if (threadIdx.x == 0) maximum = mx;
  __syncthreads();
  float local_sum = 0.f;
  for (int s = threadIdx.x; s < active; s += NT) {
    float w = exp2f(lse[(batch * splits + s) * 6 + head] - maximum);
    weights[s] = w;
    local_sum += w;
  }
  float total = Reduce(temp).Sum(local_sum);
  if (threadIdx.x == 0) denominator = total;
  __syncthreads();
  float accum = 0.f;
  for (int s = group; s < active; s += 8) {
    float w = weights[s] / denominator;
    accum = fmaf(
        w, __half2float(partial[((batch * splits + s) * 6 + head) * 256 + d]),
        accum);
  }
  sums[group][threadIdx.x % D_TILE] = accum;
  __syncthreads();
  if (threadIdx.x < D_TILE) {
    float result = 0.f;
#pragma unroll
    for (int g = 0; g < 8; ++g) result += sums[g][threadIdx.x];
    out[(batch * 6 + head) * 256 + d] = __float2half_rn(result);
  }
}

inline void combine(tvm::ffi::TensorView partial, tvm::ffi::TensorView lse,
                    tvm::ffi::TensorView lengths, tvm::ffi::TensorView out,
                    int64_t selected_tokens, int64_t tokens_per_split) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto batch = SymbolicSize{}, splits = SymbolicSize{};
  TensorMatcher({batch, splits, 6, 256})
      .with_dtype<half>()
      .with_device(device)
      .verify(partial);
  TensorMatcher({batch, splits, 6})
      .with_dtype<float>()
      .with_device(device)
      .verify(lse);
  TensorMatcher({batch}).with_dtype<int>().with_device(device).verify(lengths);
  TensorMatcher({batch, 6, 256})
      .with_dtype<half>()
      .with_device(device)
      .verify(out);
  RuntimeCheck(batch.unwrap() >= 1 && batch.unwrap() <= 4 &&
                   splits.unwrap() >= 1 && splits.unwrap() <= 160,
               "Unsupported QSA combine shape");
  RuntimeCheck(selected_tokens > 0 && tokens_per_split > 0,
               "Invalid QSA split sizes");
  LaunchKernel(dim3(16, 6, batch.unwrap()), 128, device.unwrap())(
      kernel, static_cast<const half*>(partial.data_ptr()),
      static_cast<const float*>(lse.data_ptr()),
      static_cast<const int*>(lengths.data_ptr()),
      static_cast<half*>(out.data_ptr()), int(splits.unwrap()),
      int(selected_tokens), int(tokens_per_split));
}
}  // namespace sglang::sm70_qsa_combine
