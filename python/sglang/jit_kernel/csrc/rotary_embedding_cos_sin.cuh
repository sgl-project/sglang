#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

// Convert scalar types to float for computation.
template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}
template <>
__device__ __forceinline__ float to_float<half>(half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) {
  return __bfloat162float(v);
}

// Convert float back to scalar types (round-to-nearest for fp16/bf16).
template <typename T>
__device__ __forceinline__ T from_float(float v) {
  return static_cast<T>(v);
}
template <>
__device__ __forceinline__ half from_float<half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

// Vectorize loads/stores in 16 bytes (float4) regardless of scalar type.
template <typename scalar_t, bool interleaved>
struct RotaryVecData;

template <typename scalar_t>
struct RotaryVecData<scalar_t, true> {
  float4 vec;
  float2 cos_vec;
  float2 sin_vec;
};

template <typename scalar_t>
struct RotaryVecData<scalar_t, false> {
  float4 vec_x;
  float4 vec_y;
  float4 cos_x;
  float4 sin_x;
  float4 cos_y;
  float4 sin_y;
};

// Vector load helper:
//   - load a 16B tile from q/k (either aligned float4 or scalar gather)
//   - always vector-load cos/sin (aligned by launch-time checks)
//
// rot_offset is the "pair index":
//   - interleaved:     pair i -> q[2*i], q[2*i+1]
//   - non-interleaved: pair i -> q[i], q[i + embed_dim]
// embed_dim is the number of pairs per head.
template <typename scalar_t, bool interleaved, bool aligned_qk>
__device__ __forceinline__ RotaryVecData<scalar_t, interleaved> load_rotary_vec(
    const scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  RotaryVecData<scalar_t, interleaved> data;

  constexpr int kVecBytes = 16;
  constexpr int kElePerVec = kVecBytes / sizeof(scalar_t);

  if constexpr (interleaved) {
    const int base = rot_offset * 2;

    if constexpr (aligned_qk) {
      data.vec = *reinterpret_cast<const float4*>(arr + base);
    } else {
      union VecU {
        float4 v;
        scalar_t e[kElePerVec];
      } tmp;
#pragma unroll
      for (int i = 0; i < kElePerVec; ++i)
        tmp.e[i] = arr[base + i];
      data.vec = tmp.v;
    }

    data.cos_vec = *reinterpret_cast<const float2*>(cos_ptr + rot_offset);
    data.sin_vec = *reinterpret_cast<const float2*>(sin_ptr + rot_offset);
  } else {
    const int bx = rot_offset;
    const int by = rot_offset + embed_dim;

    if constexpr (aligned_qk) {
      data.vec_x = *reinterpret_cast<const float4*>(arr + bx);
      data.vec_y = *reinterpret_cast<const float4*>(arr + by);
    } else {
      union VecU {
        float4 v;
        scalar_t e[kElePerVec];
      } tx, ty;
#pragma unroll
      for (int i = 0; i < kElePerVec; ++i) {
        tx.e[i] = arr[bx + i];
        ty.e[i] = arr[by + i];
      }
      data.vec_x = tx.v;
      data.vec_y = ty.v;
    }

    data.cos_x = *reinterpret_cast<const float4*>(cos_ptr + bx);
    data.sin_x = *reinterpret_cast<const float4*>(sin_ptr + bx);
    data.cos_y = *reinterpret_cast<const float4*>(cos_ptr + by);
    data.sin_y = *reinterpret_cast<const float4*>(sin_ptr + by);
  }

  return data;
}

// Vector compute + store:
// Apply RoPE to the loaded 16B tile and write back to q/k.
//
// Interleaved math (per pair):
//   x' = x*cos - y*sin
//   y' = y*cos + x*sin
//
// Non-interleaved math (general form, allows distinct cos/sin for x and y halves):
//   x' = x*cos_x - y*sin_x
//   y' = y*cos_y + x*sin_y
template <typename scalar_t, bool interleaved, bool aligned_qk>
__device__ __forceinline__ void compute_store_rotary_vec(
    scalar_t* __restrict__ arr, const RotaryVecData<scalar_t, interleaved>& data, int rot_offset, int embed_dim) {
  constexpr int kVecBytes = 16;
  constexpr int kElePerVec = kVecBytes / sizeof(scalar_t);

  if constexpr (interleaved) {
    union VecU {
      float4 v;
      scalar_t e[kElePerVec];
    } v;
    v.v = data.vec;

    union CSU {
      float2 v;
      scalar_t e[kElePerVec / 2];
    } c, s;
    c.v = data.cos_vec;
    s.v = data.sin_vec;

#pragma unroll
    for (int i = 0; i < kElePerVec; i += 2) {
      const int idx = i / 2;
      const float cos_val = to_float(c.e[idx]);
      const float sin_val = to_float(s.e[idx]);
      const float x = to_float(v.e[i]);
      const float y = to_float(v.e[i + 1]);
      v.e[i] = from_float<scalar_t>(x * cos_val - y * sin_val);
      v.e[i + 1] = from_float<scalar_t>(y * cos_val + x * sin_val);
    }

    const int base = rot_offset * 2;
    if constexpr (aligned_qk) {
      *reinterpret_cast<float4*>(arr + base) = v.v;
    } else {
#pragma unroll
      for (int i = 0; i < kElePerVec; ++i)
        arr[base + i] = v.e[i];
    }
  } else {
    union VecU {
      float4 v;
      scalar_t e[kElePerVec];
    } vx, vy;
    vx.v = data.vec_x;
    vy.v = data.vec_y;

    union CSU {
      float4 v;
      scalar_t e[kElePerVec];
    } cx, sx, cy, sy;
    cx.v = data.cos_x;
    sx.v = data.sin_x;
    cy.v = data.cos_y;
    sy.v = data.sin_y;

#pragma unroll
    for (int i = 0; i < kElePerVec; ++i) {
      const float cos_x = to_float(cx.e[i]);
      const float sin_x = to_float(sx.e[i]);
      const float cos_y = to_float(cy.e[i]);
      const float sin_y = to_float(sy.e[i]);

      const float x = to_float(vx.e[i]);
      const float y = to_float(vy.e[i]);

      vx.e[i] = from_float<scalar_t>(x * cos_x - y * sin_x);
      vy.e[i] = from_float<scalar_t>(y * cos_y + x * sin_y);
    }

    const int bx = rot_offset;
    const int by = rot_offset + embed_dim;
    if constexpr (aligned_qk) {
      *reinterpret_cast<float4*>(arr + bx) = vx.v;
      *reinterpret_cast<float4*>(arr + by) = vy.v;
    } else {
#pragma unroll
      for (int i = 0; i < kElePerVec; ++i) {
        arr[bx + i] = vx.e[i];
        arr[by + i] = vy.e[i];
      }
    }
  }
}

// Scalar fallback: Apply RoPE for exactly one (x,y) pair per iteration.
template <typename scalar_t, bool interleaved>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr,            // [head_size]
    const scalar_t* __restrict__ cos_ptr,  // [rot_dim]
    const scalar_t* __restrict__ sin_ptr,  // [rot_dim]
    int rot_offset,
    int embed_dim) {
  if constexpr (interleaved) {
    const int x_index = 2 * rot_offset;
    const int y_index = x_index + 1;

    const float cos_val = to_float(cos_ptr[rot_offset]);
    const float sin_val = to_float(sin_ptr[rot_offset]);
    const float x = to_float(arr[x_index]);
    const float y = to_float(arr[y_index]);

    arr[x_index] = from_float<scalar_t>(x * cos_val - y * sin_val);
    arr[y_index] = from_float<scalar_t>(y * cos_val + x * sin_val);
  } else {
    const int x_index = rot_offset;
    const int y_index = rot_offset + embed_dim;

    const float cos_val_x = to_float(cos_ptr[rot_offset]);
    const float sin_val_x = to_float(sin_ptr[rot_offset]);
    const float cos_val_y = to_float(cos_ptr[rot_offset + embed_dim]);
    const float sin_val_y = to_float(sin_ptr[rot_offset + embed_dim]);

    const float x = to_float(arr[x_index]);
    const float y = to_float(arr[y_index]);

    arr[x_index] = from_float<scalar_t>(x * cos_val_x - y * sin_val_x);
    arr[y_index] = from_float<scalar_t>(y * cos_val_y + x * sin_val_y);
  }
}

// 2D-grid kernel:
//   blockIdx.x -> token index
//   blockIdx.y -> "sub-block" index within the token (tile along pairs dimension)
//
// For very small T (few tokens) but large per-token work, using multiple blocks
// per token can improve occupancy/throughput compared to one-block-per-token.
template <typename scalar_t, bool interleaved, bool vectorized, bool aligned_qk, int ROT_EMBED_DIM>
__global__ void rotary_embedding_kernel_2d(
    const scalar_t* __restrict__ cos_data,  // [num_tokens, rot_dim_arg]
    const scalar_t* __restrict__ sin_data,  // [num_tokens, rot_dim_arg]
    scalar_t* __restrict__ query_total,     // [num_tokens, num_heads, head_size] contiguous
    scalar_t* __restrict__ key_total,       // [num_tokens, num_kv_heads, head_size] contiguous or nullptr
    const int rot_dim_arg,
    const int embed_dim_for_rotation_arg,
    const int64_t query_token_stride,
    const int64_t key_token_stride,
    const int64_t head_stride_query,
    const int64_t head_stride_key,
    const int num_heads,
    const int num_kv_heads,
    const int head_size_arg,
    const int blocks_per_token) {
  const int token_idx = blockIdx.x;
  if (token_idx >= gridDim.x) return;

  const scalar_t* current_token_cos_ptr = cos_data + token_idx * rot_dim_arg;
  const scalar_t* current_token_sin_ptr = sin_data + token_idx * rot_dim_arg;

  scalar_t* query_for_token = query_total + token_idx * (int)query_token_stride;
  scalar_t* key_for_token = (key_total != nullptr) ? (key_total + token_idx * (int)key_token_stride) : nullptr;

  const int local_block_idx = blockIdx.y;
  const int embed_dim_for_rotation = (ROT_EMBED_DIM > 0) ? ROT_EMBED_DIM : embed_dim_for_rotation_arg;

  if constexpr (vectorized) {
    constexpr int kVecBytes = 16;
    constexpr int kElePerVec = kVecBytes / sizeof(scalar_t);
    constexpr int pairs_per_step = interleaved ? (kElePerVec / 2) : kElePerVec;

    const int pair_stride = blockDim.x * blocks_per_token * pairs_per_step;
    int i = (local_block_idx * blockDim.x + threadIdx.x) * pairs_per_step;
    const int nq_pairs = num_heads * embed_dim_for_rotation;

    RotaryVecData<scalar_t, interleaved> curr_data;
    if (i < nq_pairs) {
      const int head_idx = i / embed_dim_for_rotation;
      const int rot_offset = i % embed_dim_for_rotation;
      scalar_t* ptr = query_for_token + head_idx * (int)head_stride_query;
      curr_data = load_rotary_vec<scalar_t, interleaved, aligned_qk>(
          ptr, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
    }

    int next_i = i + pair_stride;
    for (; i < nq_pairs; i += pair_stride, next_i += pair_stride) {
      RotaryVecData<scalar_t, interleaved> next_data;
      const bool active_next = (next_i < nq_pairs);

      if (active_next) {
        const int head_idx_next = next_i / embed_dim_for_rotation;
        const int rot_offset_next = next_i % embed_dim_for_rotation;
        scalar_t* ptr_next = query_for_token + head_idx_next * (int)head_stride_query;
        next_data = load_rotary_vec<scalar_t, interleaved, aligned_qk>(
            ptr_next, current_token_cos_ptr, current_token_sin_ptr, rot_offset_next, embed_dim_for_rotation);
      }

      const int head_idx = i / embed_dim_for_rotation;
      const int rot_offset = i % embed_dim_for_rotation;
      scalar_t* ptr = query_for_token + head_idx * (int)head_stride_query;
      compute_store_rotary_vec<scalar_t, interleaved, aligned_qk>(ptr, curr_data, rot_offset, embed_dim_for_rotation);

      if (active_next) curr_data = next_data;
    }

    if (key_for_token != nullptr) {
      const int nk_pairs = num_kv_heads * embed_dim_for_rotation;
      int k_i = (local_block_idx * blockDim.x + threadIdx.x) * pairs_per_step;

      RotaryVecData<scalar_t, interleaved> curr_data_k;
      if (k_i < nk_pairs) {
        const int head_idx = k_i / embed_dim_for_rotation;
        const int rot_offset = k_i % embed_dim_for_rotation;
        scalar_t* ptr = key_for_token + head_idx * (int)head_stride_key;
        curr_data_k = load_rotary_vec<scalar_t, interleaved, aligned_qk>(
            ptr, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
      }

      int next_k_i = k_i + pair_stride;
      for (; k_i < nk_pairs; k_i += pair_stride, next_k_i += pair_stride) {
        RotaryVecData<scalar_t, interleaved> next_data_k;
        const bool active_next = (next_k_i < nk_pairs);
        if (active_next) {
          const int head_idx_next = next_k_i / embed_dim_for_rotation;
          const int rot_offset_next = next_k_i % embed_dim_for_rotation;
          scalar_t* ptr_next = key_for_token + head_idx_next * (int)head_stride_key;
          next_data_k = load_rotary_vec<scalar_t, interleaved, aligned_qk>(
              ptr_next, current_token_cos_ptr, current_token_sin_ptr, rot_offset_next, embed_dim_for_rotation);
        }

        const int head_idx = k_i / embed_dim_for_rotation;
        const int rot_offset = k_i % embed_dim_for_rotation;
        scalar_t* ptr = key_for_token + head_idx * (int)head_stride_key;
        compute_store_rotary_vec<scalar_t, interleaved, aligned_qk>(
            ptr, curr_data_k, rot_offset, embed_dim_for_rotation);

        if (active_next) curr_data_k = next_data_k;
      }
    }
  } else {
    // Fallback to scalar implementation
    const int pair_stride = blockDim.x * blocks_per_token;
    const int thread_pair_offset = local_block_idx * blockDim.x + threadIdx.x;

    const int nq_pairs = num_heads * embed_dim_for_rotation;
    for (int i = thread_pair_offset; i < nq_pairs; i += pair_stride) {
      const int head_idx = i / embed_dim_for_rotation;
      const int rot_offset = i % embed_dim_for_rotation;
      scalar_t* query_for_token_head = query_for_token + head_idx * (int)head_stride_query;
      apply_token_rotary_embedding<scalar_t, interleaved>(
          query_for_token_head, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
    }

    if (key_for_token != nullptr) {
      const int nk_pairs = num_kv_heads * embed_dim_for_rotation;
      for (int i = thread_pair_offset; i < nk_pairs; i += pair_stride) {
        const int head_idx = i / embed_dim_for_rotation;
        const int rot_offset = i % embed_dim_for_rotation;
        scalar_t* key_for_token_head = key_for_token + head_idx * (int)head_stride_key;
        apply_token_rotary_embedding<scalar_t, interleaved>(
            key_for_token_head, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
      }
    }
  }
}

// 1D-grid kernel:
//   blockIdx.x -> token index
//   exactly one block per token
//
// The default path for most sizes.
template <typename scalar_t, bool interleaved, bool vectorized, bool aligned_qk, int ROT_EMBED_DIM>
__launch_bounds__(512) __global__ void rotary_embedding_kernel_1d(
    const scalar_t* __restrict__ cos_data,  // [num_tokens, rot_dim_arg]
    const scalar_t* __restrict__ sin_data,  // [num_tokens, rot_dim_arg]
    scalar_t* __restrict__ query_total,
    scalar_t* __restrict__ key_total,
    const int rot_dim_arg,
    const int embed_dim_for_rotation_arg,
    const int64_t query_token_stride,
    const int64_t key_token_stride,
    const int64_t head_stride_query,
    const int64_t head_stride_key,
    const int num_heads,
    const int num_kv_heads,
    const int head_size_arg) {
  const int token_idx = blockIdx.x;
  if (token_idx >= gridDim.x) return;

  const scalar_t* current_token_cos_ptr = cos_data + token_idx * rot_dim_arg;
  const scalar_t* current_token_sin_ptr = sin_data + token_idx * rot_dim_arg;

  scalar_t* query_for_token = query_total + token_idx * (int)query_token_stride;
  scalar_t* key_for_token = (key_total != nullptr) ? (key_total + token_idx * (int)key_token_stride) : nullptr;

  const int embed_dim_for_rotation = (ROT_EMBED_DIM > 0) ? ROT_EMBED_DIM : embed_dim_for_rotation_arg;

  if constexpr (vectorized) {
    constexpr int kVecBytes = 16;
    constexpr int kElePerVec = kVecBytes / sizeof(scalar_t);
    constexpr int pairs_per_step = interleaved ? (kElePerVec / 2) : kElePerVec;

    const int nq_pairs = num_heads * embed_dim_for_rotation;
    const int stride = blockDim.x * pairs_per_step;
    int i = threadIdx.x * pairs_per_step;

    RotaryVecData<scalar_t, interleaved> curr_data;
    if (i < nq_pairs) {
      const int head_idx = i / embed_dim_for_rotation;
      const int rot_offset = i % embed_dim_for_rotation;
      scalar_t* ptr = query_for_token + head_idx * (int)head_stride_query;
      curr_data = load_rotary_vec<scalar_t, interleaved, aligned_qk>(
          ptr, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
    }

    int next_i = i + stride;
    for (; i < nq_pairs; i += stride, next_i += stride) {
      RotaryVecData<scalar_t, interleaved> next_data;
      if (next_i < nq_pairs) {
        const int head_idx = next_i / embed_dim_for_rotation;
        const int rot_offset = next_i % embed_dim_for_rotation;
        scalar_t* ptr = query_for_token + head_idx * (int)head_stride_query;
        next_data = load_rotary_vec<scalar_t, interleaved, aligned_qk>(
            ptr, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
      }

      if (i < nq_pairs) {
        const int head_idx = i / embed_dim_for_rotation;
        const int rot_offset = i % embed_dim_for_rotation;
        scalar_t* ptr = query_for_token + head_idx * (int)head_stride_query;
        compute_store_rotary_vec<scalar_t, interleaved, aligned_qk>(ptr, curr_data, rot_offset, embed_dim_for_rotation);
      }
      curr_data = next_data;
    }

    if (key_for_token != nullptr) {
      const int nk_pairs = num_kv_heads * embed_dim_for_rotation;
      int k_i = threadIdx.x * pairs_per_step;
      RotaryVecData<scalar_t, interleaved> curr_data_k;

      if (k_i < nk_pairs) {
        const int head_idx = k_i / embed_dim_for_rotation;
        const int rot_offset = k_i % embed_dim_for_rotation;
        scalar_t* ptr = key_for_token + head_idx * (int)head_stride_key;
        curr_data_k = load_rotary_vec<scalar_t, interleaved, aligned_qk>(
            ptr, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
      }

      int next_k_i = k_i + stride;
      for (; k_i < nk_pairs; k_i += stride, next_k_i += stride) {
        RotaryVecData<scalar_t, interleaved> next_data_k;
        if (next_k_i < nk_pairs) {
          const int head_idx = next_k_i / embed_dim_for_rotation;
          const int rot_offset = next_k_i % embed_dim_for_rotation;
          scalar_t* ptr = key_for_token + head_idx * (int)head_stride_key;
          next_data_k = load_rotary_vec<scalar_t, interleaved, aligned_qk>(
              ptr, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
        }

        if (k_i < nk_pairs) {
          const int head_idx = k_i / embed_dim_for_rotation;
          const int rot_offset = k_i % embed_dim_for_rotation;
          scalar_t* ptr = key_for_token + head_idx * (int)head_stride_key;
          compute_store_rotary_vec<scalar_t, interleaved, aligned_qk>(
              ptr, curr_data_k, rot_offset, embed_dim_for_rotation);
        }
        curr_data_k = next_data_k;
      }
    }
  } else {
    const int nq_pairs = num_heads * embed_dim_for_rotation;
    for (int i = threadIdx.x; i < nq_pairs; i += blockDim.x) {
      const int head_idx = i / embed_dim_for_rotation;
      const int rot_offset = i % embed_dim_for_rotation;
      scalar_t* query_for_token_head = query_for_token + head_idx * (int)head_stride_query;
      apply_token_rotary_embedding<scalar_t, interleaved>(
          query_for_token_head, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
    }

    if (key_for_token != nullptr) {
      const int nk_pairs = num_kv_heads * embed_dim_for_rotation;
      for (int i = threadIdx.x; i < nk_pairs; i += blockDim.x) {
        const int head_idx = i / embed_dim_for_rotation;
        const int rot_offset = i % embed_dim_for_rotation;
        scalar_t* key_for_token_head = key_for_token + head_idx * (int)head_stride_key;
        apply_token_rotary_embedding<scalar_t, interleaved>(
            key_for_token_head, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
      }
    }
  }
}

// Kernel variant dispatcher:
// Select one of:
//   - grid shape: 2D (multi-block per token) vs 1D (one-block per token)
//   - layout: interleaved vs non-interleaved
//   - compute: vectorized 16B tiles vs scalar
//   - memory: q/k aligned 16B load/store vs scalar gather/scatter
template <int ROT, typename scalar_t>
__forceinline__ void dispatch_rotary_launch(
    bool use_grid_2d,
    dim3 grid2d,
    dim3 grid1d,
    dim3 block,
    const decltype(host::LaunchKernel::resolve_device(std::declval<DLDevice>())) stream,
    bool interleaved,
    bool use_vec,
    bool qk_aligned16,
    const scalar_t* cos_ptr,
    const scalar_t* sin_ptr,
    scalar_t* q_ptr,
    scalar_t* k_ptr,
    int rot_dim_from_cache,
    int embed_dim_for_rotation,
    int64_t query_token_stride,
    int64_t key_token_stride,
    int64_t head_stride_query,
    int64_t head_stride_key,
    int num_heads,
    int num_kv_heads,
    int head_size,
    int blocks_per_token) {
  // 2D grid path
  if (use_grid_2d) {
    if (interleaved) {
      if (use_vec) {
        if (qk_aligned16) {
          host::LaunchKernel(grid2d, block, stream)(
              rotary_embedding_kernel_2d<scalar_t, true, true, true, ROT>,
              cos_ptr,
              sin_ptr,
              q_ptr,
              k_ptr,
              rot_dim_from_cache,
              embed_dim_for_rotation,
              query_token_stride,
              key_token_stride,
              head_stride_query,
              head_stride_key,
              num_heads,
              num_kv_heads,
              head_size,
              blocks_per_token);
        } else {
          host::LaunchKernel(grid2d, block, stream)(
              rotary_embedding_kernel_2d<scalar_t, true, true, false, ROT>,
              cos_ptr,
              sin_ptr,
              q_ptr,
              k_ptr,
              rot_dim_from_cache,
              embed_dim_for_rotation,
              query_token_stride,
              key_token_stride,
              head_stride_query,
              head_stride_key,
              num_heads,
              num_kv_heads,
              head_size,
              blocks_per_token);
        }
      } else {
        host::LaunchKernel(grid2d, block, stream)(
            rotary_embedding_kernel_2d<scalar_t, true, false, true, ROT>,
            cos_ptr,
            sin_ptr,
            q_ptr,
            k_ptr,
            rot_dim_from_cache,
            embed_dim_for_rotation,
            query_token_stride,
            key_token_stride,
            head_stride_query,
            head_stride_key,
            num_heads,
            num_kv_heads,
            head_size,
            blocks_per_token);
      }
    } else {
      if (use_vec) {
        if (qk_aligned16) {
          host::LaunchKernel(grid2d, block, stream)(
              rotary_embedding_kernel_2d<scalar_t, false, true, true, ROT>,
              cos_ptr,
              sin_ptr,
              q_ptr,
              k_ptr,
              rot_dim_from_cache,
              embed_dim_for_rotation,
              query_token_stride,
              key_token_stride,
              head_stride_query,
              head_stride_key,
              num_heads,
              num_kv_heads,
              head_size,
              blocks_per_token);
        } else {
          host::LaunchKernel(grid2d, block, stream)(
              rotary_embedding_kernel_2d<scalar_t, false, true, false, ROT>,
              cos_ptr,
              sin_ptr,
              q_ptr,
              k_ptr,
              rot_dim_from_cache,
              embed_dim_for_rotation,
              query_token_stride,
              key_token_stride,
              head_stride_query,
              head_stride_key,
              num_heads,
              num_kv_heads,
              head_size,
              blocks_per_token);
        }
      } else {
        host::LaunchKernel(grid2d, block, stream)(
            rotary_embedding_kernel_2d<scalar_t, false, false, true, ROT>,
            cos_ptr,
            sin_ptr,
            q_ptr,
            k_ptr,
            rot_dim_from_cache,
            embed_dim_for_rotation,
            query_token_stride,
            key_token_stride,
            head_stride_query,
            head_stride_key,
            num_heads,
            num_kv_heads,
            head_size,
            blocks_per_token);
      }
    }
    return;
  }

  // 1D grid path
  if (interleaved) {
    if (use_vec) {
      if (qk_aligned16) {
        host::LaunchKernel(grid1d, block, stream)(
            rotary_embedding_kernel_1d<scalar_t, true, true, true, ROT>,
            cos_ptr,
            sin_ptr,
            q_ptr,
            k_ptr,
            rot_dim_from_cache,
            embed_dim_for_rotation,
            query_token_stride,
            key_token_stride,
            head_stride_query,
            head_stride_key,
            num_heads,
            num_kv_heads,
            head_size);
      } else {
        host::LaunchKernel(grid1d, block, stream)(
            rotary_embedding_kernel_1d<scalar_t, true, true, false, ROT>,
            cos_ptr,
            sin_ptr,
            q_ptr,
            k_ptr,
            rot_dim_from_cache,
            embed_dim_for_rotation,
            query_token_stride,
            key_token_stride,
            head_stride_query,
            head_stride_key,
            num_heads,
            num_kv_heads,
            head_size);
      }
    } else {
      host::LaunchKernel(grid1d, block, stream)(
          rotary_embedding_kernel_1d<scalar_t, true, false, true, ROT>,
          cos_ptr,
          sin_ptr,
          q_ptr,
          k_ptr,
          rot_dim_from_cache,
          embed_dim_for_rotation,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          num_heads,
          num_kv_heads,
          head_size);
    }
  } else {
    if (use_vec) {
      if (qk_aligned16) {
        host::LaunchKernel(grid1d, block, stream)(
            rotary_embedding_kernel_1d<scalar_t, false, true, true, ROT>,
            cos_ptr,
            sin_ptr,
            q_ptr,
            k_ptr,
            rot_dim_from_cache,
            embed_dim_for_rotation,
            query_token_stride,
            key_token_stride,
            head_stride_query,
            head_stride_key,
            num_heads,
            num_kv_heads,
            head_size);
      } else {
        host::LaunchKernel(grid1d, block, stream)(
            rotary_embedding_kernel_1d<scalar_t, false, true, false, ROT>,
            cos_ptr,
            sin_ptr,
            q_ptr,
            k_ptr,
            rot_dim_from_cache,
            embed_dim_for_rotation,
            query_token_stride,
            key_token_stride,
            head_stride_query,
            head_stride_key,
            num_heads,
            num_kv_heads,
            head_size);
      }
    } else {
      host::LaunchKernel(grid1d, block, stream)(
          rotary_embedding_kernel_1d<scalar_t, false, false, true, ROT>,
          cos_ptr,
          sin_ptr,
          q_ptr,
          k_ptr,
          rot_dim_from_cache,
          embed_dim_for_rotation,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          num_heads,
          num_kv_heads,
          head_size);
    }
  }
}

template <typename scalar_t>
inline void launch_rotary(
    const tvm::ffi::TensorView cos,
    const tvm::ffi::TensorView sin,
    const tvm::ffi::TensorView q,
    const tvm::ffi::TensorView* k,
    int64_t head_size,
    bool interleaved) {
  using namespace host;

  auto T = SymbolicSize{"T"};
  auto Hq = SymbolicSize{"Hq"};
  auto Hk = SymbolicSize{"Hk"};
  auto D = SymbolicSize{"D"};
  auto R = SymbolicSize{"R"};
  auto dtype = SymbolicDType{};
  auto device = SymbolicDevice{};

  TensorMatcher({T, R}).with_dtype<float, half, nv_bfloat16>(dtype).with_device<kDLCUDA>(device).verify(cos).verify(
      sin);

  TensorMatcher({T, Hq, D}).with_dtype<float, half, nv_bfloat16>(dtype).with_device<kDLCUDA>(device).verify(q);

  RuntimeCheck(D.unwrap() == head_size, "head_size mismatch: got ", D.unwrap(), " expected ", head_size);
  RuntimeCheck(cos.size(1) == sin.size(1), "cos/sin dim mismatch");

  const int64_t t = T.unwrap();
  const int64_t hq = Hq.unwrap();
  const int64_t d = D.unwrap();
  const int64_t r = R.unwrap();

  RuntimeCheck(t > 0 && hq > 0 && d > 0 && r > 0, "invalid shape");
  if (!interleaved) {
    RuntimeCheck(r % 2 == 0, "non-interleaved requires even R, got ", r);
  }
  const int embed_dim_for_rotation = interleaved ? (int)r : (int)(r / 2);
  RuntimeCheck(embed_dim_for_rotation > 0, "embed_dim_for_rotation must be > 0");
  RuntimeCheck(2LL * embed_dim_for_rotation <= head_size, "rotate dim exceeds head_size");

  // Strides: JIT wrapper guarantees contiguous [T, H, D] tensors.
  const int64_t query_token_stride = hq * d;
  const int64_t head_stride_query = d;

  int64_t key_token_stride = 0;
  int64_t head_stride_key = d;
  int hk = 0;
  if (k != nullptr) {
    TensorMatcher({T, Hk, D}).with_dtype<float, half, nv_bfloat16>(dtype).with_device<kDLCUDA>(device).verify(*k);
    hk = (int)Hk.unwrap();
    RuntimeCheck(hk > 0, "invalid key shape");
    key_token_stride = (int64_t)hk * d;
  }

  const int max_pairs_to_rotate_per_token =
      (k == nullptr) ? ((int)hq * embed_dim_for_rotation)
                     : std::max((int)hq * embed_dim_for_rotation, (int)hk * embed_dim_for_rotation);

  constexpr int kVecBytes = 16;
  const int elem_bytes = (int)sizeof(scalar_t);
  const int kElePerVec = kVecBytes / elem_bytes;
  const int pairs_per_step = interleaved ? (kElePerVec / 2) : kElePerVec;

  bool can_vec_compute = true;
  if ((embed_dim_for_rotation % pairs_per_step) != 0) can_vec_compute = false;
  if ((reinterpret_cast<uintptr_t>(cos.data_ptr()) % kVecBytes) != 0) can_vec_compute = false;
  if ((reinterpret_cast<uintptr_t>(sin.data_ptr()) % kVecBytes) != 0) can_vec_compute = false;
  if (((r * elem_bytes) % kVecBytes) != 0) can_vec_compute = false;

  bool qk_aligned16 = true;
  if ((reinterpret_cast<uintptr_t>(q.data_ptr()) % kVecBytes) != 0) qk_aligned16 = false;
  if (((query_token_stride * elem_bytes) % kVecBytes) != 0) qk_aligned16 = false;
  if (((head_stride_query * elem_bytes) % kVecBytes) != 0) qk_aligned16 = false;
  if (k != nullptr) {
    if ((reinterpret_cast<uintptr_t>(k->data_ptr()) % kVecBytes) != 0) qk_aligned16 = false;
    if (((key_token_stride * elem_bytes) % kVecBytes) != 0) qk_aligned16 = false;
    if (((head_stride_key * elem_bytes) % kVecBytes) != 0) qk_aligned16 = false;
  }

  const bool use_vec = can_vec_compute;
  const int launch_pairs_per_thread = use_vec ? pairs_per_step : 1;
  const int total_threads_needed =
      (max_pairs_to_rotate_per_token + launch_pairs_per_thread - 1) / launch_pairs_per_thread;

  auto round_up32 = [](int x) { return ((x + 31) / 32) * 32; };

  // Case 1: 2D Grid (only when num_tokens<=4 and needs multiple blocks per token)
  const int threads_per_block_2d = std::min<int>(512, std::max(128, round_up32(std::min(total_threads_needed, 512))));
  const int blocks_per_token_2d = (total_threads_needed + threads_per_block_2d - 1) / threads_per_block_2d;
  const bool use_grid_2d = ((int)t <= 4) && (blocks_per_token_2d > 1);

  // Case 2: 1D Grid
  const int threads_per_block_1d = std::min<int>(512, std::max(128, round_up32(total_threads_needed)));

  const int threads_per_block = use_grid_2d ? threads_per_block_2d : threads_per_block_1d;
  const int blocks_per_token = use_grid_2d ? blocks_per_token_2d : 1;

  const auto stream = LaunchKernel::resolve_device(device.unwrap());
  const scalar_t* cos_ptr = static_cast<const scalar_t*>(cos.data_ptr());
  const scalar_t* sin_ptr = static_cast<const scalar_t*>(sin.data_ptr());
  scalar_t* q_ptr = static_cast<scalar_t*>(q.data_ptr());
  scalar_t* k_ptr = (k != nullptr) ? static_cast<scalar_t*>(k->data_ptr()) : nullptr;

  const dim3 grid2d((int)t, std::max(1, blocks_per_token));
  const dim3 grid1d((int)t);
  const dim3 block(threads_per_block);

  // Compile-time specializations for common embed dims; fallback to runtime (ROT=0).
  switch (embed_dim_for_rotation) {
    case 32:
      dispatch_rotary_launch<32, scalar_t>(
          use_grid_2d,
          grid2d,
          grid1d,
          block,
          stream,
          interleaved,
          use_vec,
          qk_aligned16,
          cos_ptr,
          sin_ptr,
          q_ptr,
          k_ptr,
          (int)r,
          embed_dim_for_rotation,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          (int)hq,
          (int)hk,
          (int)head_size,
          blocks_per_token);
      break;
    case 40:
      dispatch_rotary_launch<40, scalar_t>(
          use_grid_2d,
          grid2d,
          grid1d,
          block,
          stream,
          interleaved,
          use_vec,
          qk_aligned16,
          cos_ptr,
          sin_ptr,
          q_ptr,
          k_ptr,
          (int)r,
          embed_dim_for_rotation,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          (int)hq,
          (int)hk,
          (int)head_size,
          blocks_per_token);
      break;
    case 64:
      dispatch_rotary_launch<64, scalar_t>(
          use_grid_2d,
          grid2d,
          grid1d,
          block,
          stream,
          interleaved,
          use_vec,
          qk_aligned16,
          cos_ptr,
          sin_ptr,
          q_ptr,
          k_ptr,
          (int)r,
          embed_dim_for_rotation,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          (int)hq,
          (int)hk,
          (int)head_size,
          blocks_per_token);
      break;
    case 80:
      dispatch_rotary_launch<80, scalar_t>(
          use_grid_2d,
          grid2d,
          grid1d,
          block,
          stream,
          interleaved,
          use_vec,
          qk_aligned16,
          cos_ptr,
          sin_ptr,
          q_ptr,
          k_ptr,
          (int)r,
          embed_dim_for_rotation,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          (int)hq,
          (int)hk,
          (int)head_size,
          blocks_per_token);
      break;
    case 128:
      dispatch_rotary_launch<128, scalar_t>(
          use_grid_2d,
          grid2d,
          grid1d,
          block,
          stream,
          interleaved,
          use_vec,
          qk_aligned16,
          cos_ptr,
          sin_ptr,
          q_ptr,
          k_ptr,
          (int)r,
          embed_dim_for_rotation,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          (int)hq,
          (int)hk,
          (int)head_size,
          blocks_per_token);
      break;
    case 160:
      dispatch_rotary_launch<160, scalar_t>(
          use_grid_2d,
          grid2d,
          grid1d,
          block,
          stream,
          interleaved,
          use_vec,
          qk_aligned16,
          cos_ptr,
          sin_ptr,
          q_ptr,
          k_ptr,
          (int)r,
          embed_dim_for_rotation,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          (int)hq,
          (int)hk,
          (int)head_size,
          blocks_per_token);
      break;
    default:
      dispatch_rotary_launch<0, scalar_t>(
          use_grid_2d,
          grid2d,
          grid1d,
          block,
          stream,
          interleaved,
          use_vec,
          qk_aligned16,
          cos_ptr,
          sin_ptr,
          q_ptr,
          k_ptr,
          (int)r,
          embed_dim_for_rotation,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          (int)hq,
          (int)hk,
          (int)head_size,
          blocks_per_token);
      break;
  }

  RuntimeDeviceCheck();
}

}  // namespace

struct RotaryEmbeddingCosSinKernel {
  static void run_q(
      const tvm::ffi::TensorView cos,
      const tvm::ffi::TensorView sin,
      const tvm::ffi::TensorView query,
      int64_t head_size,
      bool interleaved) {
    const auto dt = query.dtype();
    if (host::is_type<half>(dt)) {
      launch_rotary<half>(cos, sin, query, nullptr, head_size, interleaved);
    } else if (host::is_type<nv_bfloat16>(dt)) {
      launch_rotary<nv_bfloat16>(cos, sin, query, nullptr, head_size, interleaved);
    } else if (host::is_type<float>(dt)) {
      launch_rotary<float>(cos, sin, query, nullptr, head_size, interleaved);
    } else {
      host::Panic("Unsupported dtype for rotary_embedding_cos_sin");
    }
  }

  static void run_qk(
      const tvm::ffi::TensorView cos,
      const tvm::ffi::TensorView sin,
      const tvm::ffi::TensorView query,
      const tvm::ffi::TensorView key,
      int64_t head_size,
      bool interleaved) {
    const auto dt = query.dtype();
    if (host::is_type<half>(dt)) {
      launch_rotary<half>(cos, sin, query, &key, head_size, interleaved);
    } else if (host::is_type<nv_bfloat16>(dt)) {
      launch_rotary<nv_bfloat16>(cos, sin, query, &key, head_size, interleaved);
    } else if (host::is_type<float>(dt)) {
      launch_rotary<float>(cos, sin, query, &key, head_size, interleaved);
    } else {
      host::Panic("Unsupported dtype for rotary_embedding_cos_sin");
    }
  }
};
