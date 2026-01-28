/* Adapted from https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/norm/rms_norm.cu */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "utils.h"

// Array type for vectorized operations
template <typename T, int N>
struct Array {
  T data[N];

  __device__ __forceinline__ T& operator[](int i) {
    return data[i];
  }
  __device__ __forceinline__ const T& operator[](int i) const {
    return data[i];
  }
};

// Efficient vectorized load
template <typename T, int N>
inline __device__ void Load(Array<T, N>& dst, const T* src) {
  if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
    (uint4&)dst = *(const uint4*)src;
  } else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
    (uint2&)dst = *(const uint2*)src;
  } else if constexpr (sizeof(Array<T, N>) == sizeof(uint)) {
    (uint&)dst = *(const uint*)src;
  } else if constexpr (sizeof(Array<T, N>) % sizeof(uint4) == 0) {  //  uncoalesced
    constexpr int M = sizeof(Array<T, N>) / sizeof(uint4);
#pragma unroll
    for (int i = 0; i < M; ++i) {
      *((uint4*)&dst + i) = *((uint4*)src + i);
    }
  } else {
    static_assert(!std::is_same_v<T, T>);
  }
}

// Efficient vectorized load with __ldg
template <typename T, int N>
inline __device__ void Ldg(Array<T, N>& dst, const T* src) {
  static_assert(sizeof(Array<T, N>) <= sizeof(uint4));

  if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
    (uint4&)dst = __ldg((const uint4*)src);
  } else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
    (uint2&)dst = __ldg((const uint2*)src);
  } else if constexpr (sizeof(Array<T, N>) == sizeof(uint)) {
    (uint&)dst = __ldg((const uint*)src);
  } else {
    static_assert(!std::is_same_v<T, T>);
  }
}

// Efficient vectorized store
template <typename T, int N>
inline __device__ void Store(T* __restrict__ dst, const Array<T, N>& src) {
  if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
    *(uint4*)dst = (const uint4&)src;
  } else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
    *(uint2*)dst = (const uint2&)src;
  } else if constexpr (sizeof(Array<T, N>) == sizeof(uint)) {
    *(uint*)dst = (const uint&)src;
  } else if constexpr (sizeof(Array<T, N>) % sizeof(uint4) == 0) {  //  uncoalesced
    constexpr int M = sizeof(Array<T, N>) / sizeof(uint4);
#pragma unroll
    for (int i = 0; i < M; ++i) {
      *((uint4*)dst + i) = *((uint4*)&src + i);
    }
  } else {
    static_assert(!std::is_same_v<T, T>);
  }
}

template <class T, int vec_size>
__global__ void RMSNorm(T* data, const T* weight, int dim, int total_heads, float eps, float inv_dim) {
  // vec_size = 16/2 = 8 for float16, 4 for float32
  constexpr int thr_per_qk = 128 / vec_size;  // process 16 elements per thread if float16

  const int bi = (threadIdx.x + blockIdx.x * blockDim.x) / thr_per_qk;
  const int di = threadIdx.x % thr_per_qk * vec_size;

  if (bi >= total_heads) {
    return;
  }

  data += bi * dim;

  // Load data vector efficiently
  Array<T, vec_size> vec{};
  if (di < dim) {
    Load(vec, &data[di]);
  }

  // Convert to float and compute sum of squares
  float acc[vec_size];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    acc[i] = static_cast<float>(vec[i]);
    sum += acc[i] * acc[i];
  }

  // Reduce sum within warp
  for (int mask = thr_per_qk / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync((uint32_t)-1, sum, mask);
  }

  // Compute RMS
  float rms = rsqrtf(sum * inv_dim + eps);

  // Load weight and apply normalization
  if (di < dim) {
    Array<T, vec_size> w{};
    Ldg(w, &weight[di]);

#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      vec[i] = static_cast<T>(acc[i] * rms * static_cast<float>(w[i]));
    }

    // Store back efficiently
    Store(&data[di], vec);
  }
}

void turbomind_rms_norm(torch::Tensor& data, const torch::Tensor& weight, double eps) {
  TORCH_CHECK(data.dim() == 2, "data must be a 2D tensor");
  int head_dim = data.stride(0);
  int total_heads = data.size(0);  // token_num * head_num
  TORCH_CHECK(head_dim <= 128, "head_dim must be <= 128");

  constexpr int vec_size = sizeof(uint4) / sizeof(float);  // 16/4 = 4 for float32
  constexpr int thr_per_qk = 128 / vec_size;

  TORCH_CHECK(head_dim % vec_size == 0, "head_dim must be divisible by vec_size");

  const int threads = total_heads * thr_per_qk;
  const int block_dim = 512;
  const int grid_dim = (threads + block_dim - 1) / block_dim;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(data.scalar_type(), c_type, [&] {
    // Determine vec_size based on data type
    constexpr int actual_vec_size = sizeof(uint4) / sizeof(c_type);
    if (actual_vec_size == 4) {  // float32
      RMSNorm<c_type, 4><<<grid_dim, block_dim, 0, stream>>>(
          static_cast<c_type*>(data.data_ptr()),
          static_cast<const c_type*>(weight.data_ptr()),
          head_dim,
          total_heads,
          eps,
          1.f / head_dim);
    } else if (actual_vec_size == 8) {  // float16, bfloat16
      RMSNorm<c_type, 8><<<grid_dim, block_dim, 0, stream>>>(
          static_cast<c_type*>(data.data_ptr()),
          static_cast<const c_type*>(weight.data_ptr()),
          head_dim,
          total_heads,
          eps,
          1.f / head_dim);
    }
    return true;
  });
}
