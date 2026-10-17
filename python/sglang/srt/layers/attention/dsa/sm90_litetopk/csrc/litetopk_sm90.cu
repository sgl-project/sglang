// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <dlfcn.h>
#include "sm90_litetopk.cuh"

#ifndef LT_BQ
#define LT_BQ 4
#endif
#ifndef LT_BK
#define LT_BK 128
#endif
#ifndef LT_STAGES
#define LT_STAGES 3
#endif
#ifndef LT_MATH
#define LT_MATH 256
#endif
#ifndef LT_REGS
#define LT_REGS 224
#endif

namespace {
CUtensorMap make_map(void* ptr, CUtensorMapDataType dtype, int bytes,
                     int inner, int outer, int tile_inner, int tile_outer,
                     int stride, bool swizzled) {
  using Encode = decltype(&cuTensorMapEncodeTiled);
  static void* driver = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  static Encode encode = reinterpret_cast<Encode>(dlsym(driver, "cuTensorMapEncodeTiled"));
  TORCH_CHECK(encode != nullptr, "cuTensorMapEncodeTiled unavailable");
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(inner), static_cast<cuuint64_t>(outer)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(stride * bytes)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(tile_inner), static_cast<cuuint32_t>(tile_outer)};
  const cuuint32_t elem[2] = {1, 1};
  auto status = encode(&map, dtype, 2, ptr, dims, strides, box, elem,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzled ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(status == CUDA_SUCCESS, "TMA descriptor failed ", int(status));
  return map;
}

template<bool Dense, bool Bucket = false>
void launch_impl(torch::Tensor q, torch::Tensor k, torch::Tensor scales, torch::Tensor weights,
            torch::Tensor starts, torch::Tensor ends, torch::Tensor threshold,
            torch::Tensor values, torch::Tensor indices, torch::Tensor counts, int splits,
            const int32_t* active, int range_start, int range_end,
            const float* origin = nullptr, const float* inv_delta = nullptr, const int* seed_hist = nullptr) {
  const int Q = q.size(0), S = k.size(0), cap = values.size(1);
  TORCH_CHECK(q.is_cuda() && q.is_contiguous() && k.is_contiguous());
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn && k.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(q.size(1) == 32 && q.size(2) == 128 && k.size(1) == 128);
  TORCH_CHECK(weights.is_contiguous() && scales.is_contiguous() && S % 4 == 0);
  TORCH_CHECK(values.size(0) == Q && values.scalar_type() == torch::kFloat32);
  const auto tq = make_map(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, 128, Q*32, 128, LT_BQ*32, 128, true);
  const auto tk = make_map(k.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, 128, S, 128, LT_BK, 128, true);
  const auto ts = make_map(scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, S, 1, LT_BK, 1, 0, false);
  const auto tw = make_map(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, 32, Q, 32, LT_BQ, 32, false);
  constexpr int smem = LT_BQ * 32 * 128 + LT_STAGES * LT_BK * 128
      + LT_BQ * 32 * 4 + LT_STAGES * LT_BK * 4 + (1 + 2 * LT_STAGES) * 8
      + ((LT_WARP_BUFFER == 2 && !Dense) ? (LT_MATH / 32 * LT_BQ * 64 * 8) : 0)
      + ((Bucket && LT_ONLINE) ? (LT_BQ * 256 + LT_BQ * 2 + 1) * 4 : 0)
      + ((Bucket && LT_ONLINE == 3) ? LT_MATH / 32 * LT_BQ * 34 * 4 : 0);
  auto kernel = &litetopk_sm90::score<LT_BQ, LT_BK, LT_STAGES, LT_MATH, LT_REGS, Dense, Bucket>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  int qblocks = (Q + LT_BQ - 1) / LT_BQ;
  if (splits <= 0) splits = std::max(1, (132 * 4 + qblocks - 1) / qblocks);
  kernel<<<dim3(qblocks, splits), LT_MATH + 128, smem, c10::cuda::getCurrentCUDAStream()>>>(
      Q, S, starts.data_ptr<int32_t>(), ends.data_ptr<int32_t>(),
      threshold.data_ptr<float>(), splits, values.data_ptr<float>(),
      indices.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), cap,
      active, range_start, range_end, origin, inv_delta, seed_hist, tq, tk, ts, tw);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<bool Dense>
void launch(torch::Tensor q, torch::Tensor k, torch::Tensor scales, torch::Tensor weights,
            torch::Tensor starts, torch::Tensor ends, torch::Tensor threshold,
            torch::Tensor values, torch::Tensor indices, torch::Tensor counts, int splits) {
  launch_impl<Dense>(q,k,scales,weights,starts,ends,threshold,values,indices,counts,
                     splits,nullptr,0,k.size(0));
}

void launch_masked(torch::Tensor q, torch::Tensor k, torch::Tensor scales, torch::Tensor weights,
            torch::Tensor starts, torch::Tensor ends, torch::Tensor threshold,
            torch::Tensor values, torch::Tensor indices, torch::Tensor counts, int splits,
            torch::Tensor active, int range_start, int range_end) {
  TORCH_CHECK(range_start % 256 == 0 && range_end <= k.size(0) && range_end >= range_start);
  launch_impl<false>(q,k,scales,weights,starts,ends,threshold,values,indices,counts,
                    splits,active.data_ptr<int32_t>(),range_start,range_end);
}

void launch_hot(torch::Tensor q, torch::Tensor k, torch::Tensor scales, torch::Tensor weights,
    torch::Tensor starts, torch::Tensor ends, torch::Tensor threshold,
    torch::Tensor values, torch::Tensor indices, torch::Tensor counts, int splits,
    torch::Tensor origin, torch::Tensor inv_delta, torch::Tensor seed_hist) {
  launch_impl<false, true>(q,k,scales,weights,starts,ends,threshold,values,indices,counts,
      splits,nullptr,12288,k.size(0),origin.data_ptr<float>(),inv_delta.data_ptr<float>(),seed_hist.data_ptr<int>());
}

__global__ void prepare_seed_kernel(const float* logits, const int32_t* selected,
    float* thresholds, float* values, int32_t* indices, int32_t* counts,
    int rows, int stride, int K, int cap) {
  const int row = blockIdx.x, tid = threadIdx.x;
  float t = CUDART_INF_F;
  for (int j = tid; j < K; j += blockDim.x) {
    int idx = selected[static_cast<int64_t>(row) * K + j];
    float value = logits[static_cast<int64_t>(row) * stride + idx];
    values[static_cast<int64_t>(row) * cap + j] = value;
    indices[static_cast<int64_t>(row) * cap + j] = idx;
    t = fminf(t, value);
  }
  #pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    t = fminf(t, __shfl_down_sync(0xffffffff, t, delta));
  __shared__ float mins[8];
  if ((tid & 31) == 0) mins[tid / 32] = t;
  __syncthreads();
  if (tid < 32) {
    t = tid < 8 ? mins[tid] : CUDART_INF_F;
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
      t = fminf(t, __shfl_down_sync(0xffffffff, t, delta));
    if (tid == 0) {
      thresholds[row] = t;
      counts[row] = K;
    }
  }
}

void prepare_seed(torch::Tensor scores, torch::Tensor seed_indices,
                  torch::Tensor threshold, torch::Tensor values,
                  torch::Tensor indices, torch::Tensor counts) {
  prepare_seed_kernel<<<scores.size(0), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      scores.data_ptr<float>(), seed_indices.data_ptr<int32_t>(), threshold.data_ptr<float>(),
      values.data_ptr<float>(), indices.data_ptr<int32_t>(), counts.data_ptr<int32_t>(),
      scores.size(0), scores.stride(0), seed_indices.size(1), values.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void map_kernel(const int32_t* selected, const int32_t* indices,
                          const int32_t* counts, int32_t* out, int cap, int K) {
  const int row = blockIdx.x;
  for (int j = threadIdx.x; j < K; j += blockDim.x) {
    const int64_t offset = static_cast<int64_t>(row) * K + j;
    const int idx = selected[offset];
    out[offset] = counts[row] <= cap && idx >= 0 && idx < cap
        ? indices[static_cast<int64_t>(row) * cap + idx] : -1;
  }
}

void map_indices(torch::Tensor selected, torch::Tensor indices, torch::Tensor counts, torch::Tensor out) {
  map_kernel<<<out.size(0), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      selected.data_ptr<int32_t>(), indices.data_ptr<int32_t>(), counts.data_ptr<int32_t>(),
      out.data_ptr<int32_t>(), indices.size(1), out.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("score_sparse", &launch<false>);
  m.def("score_hot", &launch_hot);
  m.def("score_masked", &launch_masked);
  m.def("score_dense", &launch<true>);
  m.def("prepare_seed", &prepare_seed);
  m.def("map_indices", &map_indices);
}
