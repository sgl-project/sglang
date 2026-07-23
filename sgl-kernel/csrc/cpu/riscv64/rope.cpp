#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#if defined(CPU_CAPABILITY_RVV)

#include <cmath>
#include <cstdint>

#include "common.h"
#include "riscv64/vector_helpers.h"

namespace {

// 3D kernel: non-neox, out-of-place (query_out, key_out)
// query shape: [num_tokens, num_heads, head_size]
// key shape:   [num_tokens, 1, head_size]  (num_kv_heads == 1)
// Rotation: adjacent pairs (h, h+1) with cos[h/2], sin[h/2]
// Cache layout: [cos(rotary_dim/2) | sin(rotary_dim/2)] per position
template <typename scalar_t>
void rotary_embedding_3D_kernel_impl(
    scalar_t* __restrict__ query_out,
    scalar_t* __restrict__ key_out,
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_size,
    int64_t rotary_dim,
    int64_t query_stride_s,
    int64_t query_out_stride_s,
    int64_t key_out_stride_s,
    int64_t key_stride_s,
    int64_t query_stride_h,
    int64_t query_out_stride_h) {
  int64_t embed_dim = rotary_dim / 2;
  const ptrdiff_t stride = 2 * static_cast<ptrdiff_t>(sizeof(scalar_t));

  at::parallel_for(0, num_tokens * num_heads, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t seq = idx / num_heads;
      int64_t head_id = idx % num_heads;
      // Scratch only used by generic scalar fallback in load/store helpers; dead for BF16/FP16.
      alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
      int64_t p = positions[seq];
      scalar_t* cos_ptr = cos_sin_cache + p * rotary_dim;
      scalar_t* sin_ptr = cos_ptr + embed_dim;

      // Vectorized adjacent-pair rotation for query (out-of-place).
      scalar_t* q_in = query + seq * query_stride_s + head_id * query_stride_h;
      scalar_t* q_out = query_out + seq * query_out_stride_s + head_id * query_out_stride_h;
      size_t vl = 0;
      for (int64_t j = 0; j < embed_dim; j += static_cast<int64_t>(vl)) {
        vl = __riscv_vsetvl_e32m4(embed_dim - j);
        vfloat32m4_t v_cos = load_as_float_m4(cos_ptr + j, vl, scratch);
        vfloat32m4_t v_sin = load_as_float_m4(sin_ptr + j, vl, scratch);
        vfloat32m4_t v_x = load_strided_as_float_m4(q_in + 2 * j, stride, vl, scratch);
        vfloat32m4_t v_y = load_strided_as_float_m4(q_in + 2 * j + 1, stride, vl, scratch);
        vfloat32m4_t v_out_x = __riscv_vfmul_vv_f32m4(v_x, v_cos, vl);
        v_out_x = __riscv_vfnmsac_vv_f32m4(v_out_x, v_y, v_sin, vl);
        vfloat32m4_t v_out_y = __riscv_vfmul_vv_f32m4(v_y, v_cos, vl);
        v_out_y = __riscv_vfmacc_vv_f32m4(v_out_y, v_x, v_sin, vl);
        store_strided_from_float_m4(q_out + 2 * j, stride, v_out_x, vl, scratch);
        store_strided_from_float_m4(q_out + 2 * j + 1, stride, v_out_y, vl, scratch);
      }
      // Copy non-rotated tail of query.
      for (int64_t h = rotary_dim; h < head_size; ++h)
        q_out[h] = q_in[h];

      // Key has only 1 head in 3D layout (num_kv_heads == 1, enforced by CHECK_EQ).
      // Only the first query head writes the key output to avoid redundant stores.
      if (head_id == 0) {
        scalar_t* k_in = key + seq * key_stride_s;
        scalar_t* k_out = key_out + seq * key_out_stride_s;
        size_t vl_k = 0;
        for (int64_t j = 0; j < embed_dim; j += static_cast<int64_t>(vl_k)) {
          vl_k = __riscv_vsetvl_e32m4(embed_dim - j);
          vfloat32m4_t v_cos = load_as_float_m4(cos_ptr + j, vl_k, scratch);
          vfloat32m4_t v_sin = load_as_float_m4(sin_ptr + j, vl_k, scratch);
          vfloat32m4_t v_x = load_strided_as_float_m4(k_in + 2 * j, stride, vl_k, scratch);
          vfloat32m4_t v_y = load_strided_as_float_m4(k_in + 2 * j + 1, stride, vl_k, scratch);
          vfloat32m4_t v_out_x = __riscv_vfmul_vv_f32m4(v_x, v_cos, vl_k);
          v_out_x = __riscv_vfnmsac_vv_f32m4(v_out_x, v_y, v_sin, vl_k);
          vfloat32m4_t v_out_y = __riscv_vfmul_vv_f32m4(v_y, v_cos, vl_k);
          v_out_y = __riscv_vfmacc_vv_f32m4(v_out_y, v_x, v_sin, vl_k);
          store_strided_from_float_m4(k_out + 2 * j, stride, v_out_x, vl_k, scratch);
          store_strided_from_float_m4(k_out + 2 * j + 1, stride, v_out_y, vl_k, scratch);
        }
        // Copy non-rotated tail of key.
        for (int64_t h = rotary_dim; h < head_size; ++h)
          k_out[h] = k_in[h];
      }
    }
  });
}

// Neox 4D kernel: in-place
// Paired indices: (j, embed_dim + j) — front half / back half
// Cache layout: [cos(rotary_dim/2) | sin(rotary_dim/2)] per position
template <typename scalar_t>
void rotary_embedding_neox_4D_kernel_impl(
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t rotary_dim,
    int64_t query_stride_b,
    int64_t query_stride_s,
    int64_t query_stride_h,
    int64_t key_stride_b,
    int64_t key_stride_s,
    int64_t key_stride_h,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t batch_size,
    int64_t seq_len) {
  int64_t embed_dim = rotary_dim / 2;

  // scratch is per-call-site (inside lambda) to avoid data race under OMP parallel
  auto compute_head = [&](scalar_t* cache_ptr, scalar_t* qk, int64_t token_head) {
    alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
    size_t vl = 0;
    for (int64_t j = 0; j < embed_dim; j += vl) {
      vl = __riscv_vsetvl_e32m4(embed_dim - j);
      int64_t x_index = j;
      int64_t y_index = embed_dim + j;

      // Load cos, sin from cache
      vfloat32m4_t v_cos = load_as_float_m4(cache_ptr + x_index, vl, scratch);
      vfloat32m4_t v_sin = load_as_float_m4(cache_ptr + y_index, vl, scratch);

      // Load x (front half) and y (back half) of head
      vfloat32m4_t v_x = load_as_float_m4(qk + token_head + x_index, vl, scratch);
      vfloat32m4_t v_y = load_as_float_m4(qk + token_head + y_index, vl, scratch);

      // out_x = x * cos - y * sin
      vfloat32m4_t v_out_x = __riscv_vfmul_vv_f32m4(v_x, v_cos, vl);
      v_out_x = __riscv_vfnmsac_vv_f32m4(v_out_x, v_y, v_sin, vl);

      // out_y = y * cos + x * sin
      vfloat32m4_t v_out_y = __riscv_vfmul_vv_f32m4(v_y, v_cos, vl);
      v_out_y = __riscv_vfmacc_vv_f32m4(v_out_y, v_x, v_sin, vl);

      store_from_float_m4(qk + token_head + x_index, v_out_x, vl, scratch);
      store_from_float_m4(qk + token_head + y_index, v_out_y, vl, scratch);
    }
  };

  at::parallel_for(0, batch_size * seq_len, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t bs = idx / seq_len;
      int64_t seq = idx % seq_len;
      int64_t pos = positions[bs * seq_len + seq];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;

      for (int64_t i = 0; i < num_heads; ++i) {
        int64_t token_head = bs * query_stride_b + seq * query_stride_s + i * query_stride_h;
        compute_head(cache_ptr, query, token_head);
      }

      for (int64_t i = 0; i < num_kv_heads; ++i) {
        int64_t token_head = bs * key_stride_b + seq * key_stride_s + i * key_stride_h;
        compute_head(cache_ptr, key, token_head);
      }
    }
  });
}
// Non-neox 4D kernel: in-place
// Paired indices: (2j, 2j+1) — adjacent pairs
// Cache layout: [cos(rotary_dim/2) | sin(rotary_dim/2)] per position
template <typename scalar_t>
void rotary_embedding_4D_kernel_impl(
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t rotary_dim,
    int64_t query_stride_b,
    int64_t query_stride_s,
    int64_t query_stride_h,
    int64_t key_stride_b,
    int64_t key_stride_s,
    int64_t key_stride_h,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t batch_size,
    int64_t seq_len) {
  int64_t embed_dim = rotary_dim / 2;

  // Vectorized adjacent-pair rotation using stride-2 load/store.
  // cache_ptr points to [cos(rotary_dim/2) | sin(rotary_dim/2)] for the token's position.
  // head_ptr points to the start of one head: layout [x0, y0, x1, y1, ...].
  // stride = 2 * sizeof(scalar_t) so strided loads pick up every other element.
  auto compute_head = [&](scalar_t* cache_ptr, scalar_t* head_ptr) {
    alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
    scalar_t* cos_ptr = cache_ptr;
    scalar_t* sin_ptr = cache_ptr + embed_dim;
    const ptrdiff_t stride = 2 * static_cast<ptrdiff_t>(sizeof(scalar_t));
    size_t vl = 0;
    for (int64_t j = 0; j < embed_dim; j += static_cast<int64_t>(vl)) {
      vl = __riscv_vsetvl_e32m4(embed_dim - j);
      vfloat32m4_t v_cos = load_as_float_m4(cos_ptr + j, vl, scratch);
      vfloat32m4_t v_sin = load_as_float_m4(sin_ptr + j, vl, scratch);
      // Stride-2 load: x = head[0,2,4,...], y = head[1,3,5,...]
      vfloat32m4_t v_x = load_strided_as_float_m4(head_ptr + 2 * j, stride, vl, scratch);
      vfloat32m4_t v_y = load_strided_as_float_m4(head_ptr + 2 * j + 1, stride, vl, scratch);

      // out_x = x * cos - y * sin
      vfloat32m4_t v_out_x = __riscv_vfmul_vv_f32m4(v_x, v_cos, vl);
      v_out_x = __riscv_vfnmsac_vv_f32m4(v_out_x, v_y, v_sin, vl);
      // out_y = y * cos + x * sin
      vfloat32m4_t v_out_y = __riscv_vfmul_vv_f32m4(v_y, v_cos, vl);
      v_out_y = __riscv_vfmacc_vv_f32m4(v_out_y, v_x, v_sin, vl);

      store_strided_from_float_m4(head_ptr + 2 * j, stride, v_out_x, vl, scratch);
      store_strided_from_float_m4(head_ptr + 2 * j + 1, stride, v_out_y, vl, scratch);
    }
  };

  at::parallel_for(0, batch_size * seq_len, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t bs = idx / seq_len;
      int64_t seq = idx % seq_len;
      int64_t pos = positions[bs * seq_len + seq];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;
      for (int64_t i = 0; i < num_heads; ++i) {
        int64_t token_head = bs * query_stride_b + seq * query_stride_s + i * query_stride_h;
        compute_head(cache_ptr, query + token_head);
      }
      for (int64_t i = 0; i < num_kv_heads; ++i) {
        int64_t token_head = bs * key_stride_b + seq * key_stride_s + i * key_stride_h;
        compute_head(cache_ptr, key + token_head);
      }
    }
  });
}

}  // namespace

// Top-level dispatch: rotary_embedding_cpu
std::tuple<at::Tensor, at::Tensor> rotary_embedding_cpu(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    bool is_neox) {
  RECORD_FUNCTION("sgl-kernel::rotary_embedding_cpu", std::vector<c10::IValue>({query, key}));
  CHECK_DIM(1, positions);
  const auto input_dim = query.dim();
  const auto input_dtype = query.scalar_type();
  TORCH_CHECK(
      input_dim == 2 || input_dim == 3 || input_dim == 4,
      " Query/Key must be 2D [num_tokens, num_heads*head_size] or 3D [num_tokens, num_heads, head_size] or 4D "
      "[batch_size, seq_len, num_heads, head_size] tensor");
  CHECK_DIM(2, cos_sin_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);

  int64_t rotary_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even, got ", rotary_dim);
  if (input_dim == 3) {
    TORCH_CHECK(
        query.size(-1) >= rotary_dim,
        "3D rotary: head_size (",
        query.size(-1),
        ") must be >= rotary_dim (",
        rotary_dim,
        ")");
    CHECK_EQ(key.size(1), 1);
  }

  int64_t num_tokens = positions.numel();
  if (input_dim <= 3) {
    CHECK_EQ(key.size(0), num_tokens);
    CHECK_EQ(query.size(0), num_tokens);
  }

  TORCH_CHECK(positions.scalar_type() == at::kLong, "expect positions to be int64, got ", positions.scalar_type());
  TORCH_CHECK(input_dtype == key.scalar_type(), "query and key must have the same data type");
  TORCH_CHECK(input_dtype == cos_sin_cache.scalar_type(), "query and cos_sin_cache must have the same data type");

  int64_t num_heads = input_dim == 2 ? query.size(-1) / head_size : query.size(-2);
  int64_t num_kv_heads = input_dim == 2 ? key.size(-1) / head_size : key.size(-2);
  int64_t key_stride_s = key.stride(0);
  int64_t query_stride_s = query.stride(0);

  int64_t query_stride_h = input_dim == 2 ? head_size : query.stride(-2);
  int64_t key_stride_h = input_dim == 2 ? head_size : key.stride(-2);
  // Output tensors: allocated here only for the 3D out-of-place path.
  // 2D/4D in-place paths reassign query_out/key_out to the input tensors inside the dispatch.
  at::Tensor query_out, key_out;
  int64_t query_out_stride_s = 0, key_out_stride_s = 0, query_out_stride_h = 0;
  if (input_dim == 3) {
    query_out = at::empty_like(query);
    key_out = at::empty_like(key);
    query_out_stride_s = query_out.stride(0);
    key_out_stride_s = key_out.stride(0);
    query_out_stride_h = query_out.stride(1);
  }
  int64_t batch_size = 1;
  int64_t seq_len = num_tokens;
  int64_t query_stride_b = 0;
  int64_t key_stride_b = 0;
  if (input_dim == 4) {
    batch_size = query.size(0);
    seq_len = query.size(1);
    query_stride_b = query.stride(0);
    key_stride_b = key.stride(0);
    query_stride_s = query.stride(1);
    key_stride_s = key.stride(1);
    CHECK_EQ(batch_size, key.size(0));
    CHECK_EQ(seq_len, key.size(1));
    CHECK_EQ(key.size(0) * key.size(1), num_tokens);
    CHECK_EQ(query.size(0) * query.size(1), num_tokens);
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input_dtype, "rotary_embedding_cpu", [&] {
    if (input_dim == 2 || input_dim == 4) {
      if (is_neox) {
        rotary_embedding_neox_4D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            query_stride_b,
            query_stride_s,
            query_stride_h,
            key_stride_b,
            key_stride_s,
            key_stride_h,
            num_heads,
            num_kv_heads,
            head_size,
            batch_size,
            seq_len);
      } else {
        rotary_embedding_4D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            query_stride_b,
            query_stride_s,
            query_stride_h,
            key_stride_b,
            key_stride_s,
            key_stride_h,
            num_heads,
            num_kv_heads,
            head_size,
            batch_size,
            seq_len);
      }
      query_out = query;
      key_out = key;

    } else {
      TORCH_CHECK(
          is_neox == false, " Query/Key with 3D [num_tokens, num_heads, head_size] does not support neox rope yet");
      rotary_embedding_3D_kernel_impl<scalar_t>(
          query_out.data_ptr<scalar_t>(),
          key_out.data_ptr<scalar_t>(),
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          num_tokens,
          num_heads,
          head_size,
          rotary_dim,
          query_stride_s,
          query_out_stride_s,
          key_out_stride_s,
          key_stride_s,
          query_stride_h,
          query_out_stride_h);
    }
  });
  return std::make_tuple(query_out, key_out);
}

#endif  // CPU_CAPABILITY_RVV
