// Adapted from
// https://github.com/vllm-project/vllm/blob/014ece97c7aa49084a1119dca792af081a18dbc1/csrc/pos_encoding_kernels.cu

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = SGLANG_LDG(cos_ptr + x_index);
    sin = SGLANG_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = SGLANG_LDG(cos_ptr + x_index / 2);
    sin = SGLANG_LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    const scalar_t* cache_ptr,
    const int head_size,
    const int num_heads,
    const int num_kv_heads,
    const int rot_dim,
    const int token_idx,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head = token_idx * key_stride + head_idx * head_stride;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, IS_NEOX>(key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
    }
  }
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,       // [batch_size, seq_len] or
                                                 // [num_tokens]
    scalar_t* __restrict__ query,                // [batch_size, seq_len, num_heads,
                                                 // head_size] or [num_tokens, num_heads,
                                                 // head_size]
    scalar_t* __restrict__ key,                  // nullptr or
                                                 // [batch_size, seq_len, num_kv_heads,
                                                 // head_size] or [num_tokens, num_kv_heads,
                                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<scalar_t, IS_NEOX>(
      query,
      key,
      cache_ptr,
      head_size,
      num_heads,
      num_kv_heads,
      rot_dim,
      token_idx,
      query_stride,
      key_stride,
      head_stride);
}

// Helper struct to launch kernel
template <typename scalar_t, bool IS_NEOX>
void launch_kernel(
    const int64_t* positions_data_ptr,
    void* query_ptr,
    void* key_ptr,
    const void* cos_sin_cache_ptr,
    int rot_dim,
    int64_t query_stride,
    int64_t key_stride,
    int64_t head_stride,
    int num_heads,
    int num_kv_heads,
    int head_size,
    dim3 grid,
    dim3 block,
    const cudaStream_t stream) {
  rotary_embedding_kernel<scalar_t, IS_NEOX><<<grid, block, 0, stream>>>(
      positions_data_ptr,
      static_cast<scalar_t*>(query_ptr),
      static_cast<scalar_t*>(key_ptr),
      static_cast<const scalar_t*>(cos_sin_cache_ptr),
      rot_dim,
      query_stride,
      key_stride,
      head_stride,
      num_heads,
      num_kv_heads,
      head_size);
};

// Helper macro to reduce repetition
#define DISPATCH_DTYPE(DTYPE_CODE, DTYPE_BITS, IS_NEOX, ...)                                                      \
  if (DTYPE_CODE == kDLFloat && DTYPE_BITS == 32) {                                                               \
    launch_kernel<float, IS_NEOX>(__VA_ARGS__);                                                                   \
  } else if (DTYPE_CODE == kDLFloat && DTYPE_BITS == 16) {                                                        \
    launch_kernel<half, IS_NEOX>(__VA_ARGS__);                                                                    \
  } else if (DTYPE_CODE == kDLBfloat && DTYPE_BITS == 16) {                                                       \
    launch_kernel<nv_bfloat16, IS_NEOX>(__VA_ARGS__);                                                             \
  } else {                                                                                                        \
    RuntimeCheck(                                                                                                 \
        false, "Unsupported data type for rotary embedding. Only float32, float16, and bfloat16 are supported."); \
  }

// Helper function to dispatch based on data type
template <bool IS_NEOX>
void dispatch_by_dtype(
    const int64_t* positions_data_ptr,
    DLDataType query_dtype,
    void* query_ptr,
    void* key_ptr,
    void* cos_sin_cache_ptr,
    int rot_dim,
    int64_t query_stride,
    int64_t key_stride,
    int64_t head_stride,
    int num_heads,
    int num_kv_heads,
    int head_size,
    dim3 grid,
    dim3 block,
    const cudaStream_t stream) {
  using namespace host;
  DISPATCH_DTYPE(
      query_dtype.code,
      query_dtype.bits,
      IS_NEOX,
      positions_data_ptr,
      query_ptr,
      key_ptr,
      cos_sin_cache_ptr,
      rot_dim,
      query_stride,
      key_stride,
      head_stride,
      num_heads,
      num_kv_heads,
      head_size,
      grid,
      block,
      stream);
}

struct RotaryEmbeddingKernel {
  static void
  run(tvm::ffi::TensorView positions,  // [batch_size, seq_len] or [num_tokens]
      tvm::ffi::TensorView query,      // [batch_size, seq_len, num_heads * head_size] or
                                       // [num_tokens, num_heads * head_size] or
                                       // [batch_size, seq_len, num_heads, head_size] or
                                       // [num_tokens, num_heads, head_size]
      tvm::ffi::Optional<tvm::ffi::TensorView> key,
      // null or
      // [batch_size, seq_len, num_kv_heads * head_size] or
      // [num_tokens, num_kv_heads * head_size] or
      // [batch_size, seq_len, num_heads, head_size] or
      // [num_tokens, num_heads, head_size]
      int64_t head_size,
      tvm::ffi::TensorView cos_sin_cache,  // [max_position, rot_dim]
      bool is_neox) {
    using namespace host;

    // num_tokens = batch_size * seq_len
    int64_t num_tokens = positions.numel();
    int32_t positions_ndim = positions.ndim();

    // Make sure num_tokens dim is consistent across positions, query, and key
    RuntimeCheck(
        positions_ndim == 1 || positions_ndim == 2, "positions must have shape [num_tokens] or [batch_size, seq_len]");
    if (positions_ndim == 1) {
      RuntimeCheck(
          query.size(0) == positions.size(0) && (!key.has_value() || key.value().size(0) == positions.size(0)),
          "query, key and positions must have the same number of tokens");
    }
    if (positions_ndim == 2) {
      RuntimeCheck(
          query.size(0) == positions.size(0) && (!key.has_value() || key.value().size(0) == positions.size(0)) &&
              query.size(1) == positions.size(1) && (!key.has_value() || key.value().size(1) == positions.size(1)),
          "query, key and positions must have the same batch_size and seq_len");
    }

    // Make sure head_size is valid for query and key
    // hidden_size = num_heads * head_size
    int query_hidden_size = query.numel() / num_tokens;
    int key_hidden_size = key.has_value() ? key.value().numel() / num_tokens : 0;
    RuntimeCheck(query_hidden_size % head_size == 0);
    RuntimeCheck(key_hidden_size % head_size == 0);

    // Make sure query and key have consistent number of heads
    int num_heads = query_hidden_size / head_size;
    int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
    RuntimeCheck(num_heads % num_kv_heads == 0);

    int rot_dim = cos_sin_cache.size(1);
    int seq_dim_idx = positions_ndim - 1;
    int64_t query_stride = query.stride(seq_dim_idx);
    int64_t key_stride = key.has_value() ? key.value().stride(seq_dim_idx) : 0;
    // Determine head stride: for [*, heads, head_size] use stride of last dim;
    // for flat [*, heads*head_size], heads blocks are contiguous of size
    // head_size
    int query_ndim = query.dim();
    int64_t head_stride = (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

    dim3 grid(num_tokens);
    dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));

    auto device = query.device();
    const cudaStream_t stream = LaunchKernel::resolve_device(device);

    auto positions_data_ptr = static_cast<const int64_t*>(positions.data_ptr());

    if (is_neox) {
      dispatch_by_dtype<true>(
          positions_data_ptr,
          query.dtype(),
          query.data_ptr(),
          key.has_value() ? key.value().data_ptr() : nullptr,
          cos_sin_cache.data_ptr(),
          rot_dim,
          query_stride,
          key_stride,
          head_stride,
          num_heads,
          num_kv_heads,
          head_size,
          grid,
          block,
          stream);
    } else {
      dispatch_by_dtype<false>(
          positions_data_ptr,
          query.dtype(),
          query.data_ptr(),
          key.has_value() ? key.value().data_ptr() : nullptr,
          cos_sin_cache.data_ptr(),
          rot_dim,
          query_stride,
          key_stride,
          head_stride,
          num_heads,
          num_kv_heads,
          head_size,
          grid,
          block,
          stream);
    }
  }
};

}  // namespace
