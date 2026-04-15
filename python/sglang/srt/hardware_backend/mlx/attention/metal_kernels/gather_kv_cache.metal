#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

// Convert from cache type to output type, with optional FP8 dequantization.
template <typename CACHE_T, typename OUT_T>
inline OUT_T from_cache(CACHE_T v) = delete;

// Identity conversions (cache_t == out_t)
template <> inline float from_cache<float, float>(float v) { return v; }
template <> inline bfloat16_t from_cache<bfloat16_t, bfloat16_t>(bfloat16_t v) {
  return v;
}
template <> inline half from_cache<half, half>(half v) { return v; }

// FP8 E4M3 -> output type conversions
template <> inline float from_cache<uchar, float>(uchar v) {
  return fp8_e4m3_to_float(v);
}
template <> inline half from_cache<uchar, half>(uchar v) {
  return (half)fp8_e4m3_to_float(v);
}
template <> inline bfloat16_t from_cache<uchar, bfloat16_t>(uchar v) {
  return (bfloat16_t)fp8_e4m3_to_float(v);
}

constant bool use_fp8_scales [[function_constant(10)]];

/// Gather K and V from paged KV cache into contiguous output tensors.
///
/// One threadgroup per output token. Threads cooperatively copy
/// kv_heads * head_size elements for both K and V.
///
/// Uses binary search on cu_seq_lens to find batch_id.
///
/// K/V cache layout: [num_blocks, block_size, kv_heads, head_size]
/// K/V output:       [num_tokens, kv_heads, head_size]
template <typename CACHE_T, typename OUT_T>
[[kernel]] void gather_kv_cache(
    const device CACHE_T *__restrict__ key_cache
    [[buffer(0)]], // [num_blocks, block_size, kv_heads, head_size]
    const device CACHE_T *__restrict__ value_cache
    [[buffer(1)]], // [num_blocks, block_size, kv_heads, head_size]
    device OUT_T *__restrict__ k_out
    [[buffer(2)]], // [num_tokens, kv_heads, head_size]
    device OUT_T *__restrict__ v_out
    [[buffer(3)]], // [num_tokens, kv_heads, head_size]
    const device float *__restrict__ k_scale
    [[buffer(4), function_constant(use_fp8_scales)]],
    const device float *__restrict__ v_scale
    [[buffer(5), function_constant(use_fp8_scales)]],
    const device int *__restrict__ block_table
    [[buffer(6)]], // [batch, max_blocks]
    const device int *__restrict__ cu_seq_lens [[buffer(7)]], // [batch + 1]
    device const int &num_tokens [[buffer(8)]],
    device const int &num_seqs [[buffer(9)]],
    device const int &block_size [[buffer(10)]],
    device const int &block_table_stride [[buffer(11)]],
    device const int &num_kv_heads [[buffer(12)]],
    device const int &head_size [[buffer(13)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  const int token_id = gid;
  if (token_id >= num_tokens) {
    return;
  }

  // Binary search cu_seq_lens to find batch_id
  int lo = 0, hi = num_seqs;
  while (lo < hi) {
    int mid = (lo + hi + 1) / 2;
    if (cu_seq_lens[mid] <= token_id) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int batch_id = lo;

  const int batch_offset = token_id - cu_seq_lens[batch_id];
  const int block_table_id = batch_offset / block_size;
  const int slot = batch_offset % block_size;
  const int block_id =
      block_table[batch_id * block_table_stride + block_table_id];

  const int n = num_kv_heads * head_size;
  const long out_base = (long)token_id * num_kv_heads * head_size;

  // Cache layout: [num_blocks, block_size, num_kv_heads, head_size]
  // Both K and V use the same layout — index is identical.
  const long cache_block_stride = (long)block_size * num_kv_heads * head_size;
  const long cache_token_stride = (long)num_kv_heads * head_size;

  for (int i = tid; i < n; i += threads_per_threadgroup) {
    const int head_idx = i / head_size;
    const int d = i % head_size;

    const long src_idx = (long)block_id * cache_block_stride +
                         slot * cache_token_stride +
                         head_idx * head_size + d;

    if (use_fp8_scales) {
      k_out[out_base + i] = OUT_T(
          (float)from_cache<CACHE_T, OUT_T>(key_cache[src_idx]) * (*k_scale));
      v_out[out_base + i] =
          OUT_T((float)from_cache<CACHE_T, OUT_T>(value_cache[src_idx]) *
                (*v_scale));
    } else {
      k_out[out_base + i] = from_cache<CACHE_T, OUT_T>(key_cache[src_idx]);
      v_out[out_base + i] = from_cache<CACHE_T, OUT_T>(value_cache[src_idx]);
    }
  }
}

#define instantiate_gather_kv_cache(cache_type, out_type)                      \
  template [[host_name("gather_kv_cache_cache_" #cache_type                    \
                       "_out_" #out_type)]] [[kernel]] void                    \
  gather_kv_cache<cache_type, out_type>(                                       \
      const device cache_type *__restrict__ key_cache [[buffer(0)]],           \
      const device cache_type *__restrict__ value_cache [[buffer(1)]],         \
      device out_type *__restrict__ k_out [[buffer(2)]],                       \
      device out_type *__restrict__ v_out [[buffer(3)]],                       \
      const device float *__restrict__ k_scale                                 \
      [[buffer(4), function_constant(use_fp8_scales)]],                        \
      const device float *__restrict__ v_scale                                 \
      [[buffer(5), function_constant(use_fp8_scales)]],                        \
      const device int *__restrict__ block_table [[buffer(6)]],                \
      const device int *__restrict__ cu_seq_lens [[buffer(7)]],                \
      device const int &num_tokens [[buffer(8)]],                              \
      device const int &num_seqs [[buffer(9)]],                                \
      device const int &block_size [[buffer(10)]],                             \
      device const int &block_table_stride [[buffer(11)]],                     \
      device const int &num_kv_heads [[buffer(12)]],                           \
      device const int &head_size [[buffer(13)]],                              \
      uint gid [[threadgroup_position_in_grid]],                               \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint threads_per_threadgroup [[threads_per_threadgroup]]);

// Same-type (no dequant)
instantiate_gather_kv_cache(float, float);
instantiate_gather_kv_cache(bfloat16_t, bfloat16_t);
instantiate_gather_kv_cache(half, half);

// FP8 E4M3 -> compute type (dequant)
instantiate_gather_kv_cache(uchar, float);
instantiate_gather_kv_cache(uchar, bfloat16_t);
instantiate_gather_kv_cache(uchar, half);