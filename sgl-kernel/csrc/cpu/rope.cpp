#include "common.h"
#include "vec.h"

namespace {

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
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t rotary_dim,
    int64_t query_stride_s,
    int64_t query_out_stride_s,
    int64_t key_out_stride_s,
    int64_t key_stride_s,
    int64_t query_stride_h,
    int64_t query_out_stride_h) {
  int64_t HR = rotary_dim;
  int64_t HK = rotary_dim;
  int64_t COFF = HR / 2;
  at::parallel_for(0, num_tokens * num_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t seq{0}, head_id{0};
    data_index_init(begin, seq, num_tokens, head_id, num_heads);
    for (int64_t i = begin; i < end; ++i) {
      int64_t in_offset_q = seq * query_stride_s + head_id * query_stride_h;
      int64_t out_offset_q = seq * query_out_stride_s + head_id * query_out_stride_h;
      int64_t out_offset_k = seq * key_out_stride_s;
      int64_t p = 0;
      scalar_t* sin_start = nullptr;
      scalar_t* cos_start = nullptr;
      // step 0) get the rotary position embedding for the current position
      p = positions[seq];
      sin_start = cos_sin_cache + p * HR + COFF;
      cos_start = cos_sin_cache + p * HR;
      // step 1) apply_rotary_pos_emb for the rotary_dim elements in every
      // head of query/key
      for (int64_t h = 0; h < rotary_dim; h += 2) {
        scalar_t cos = cos_start[h >> 1];
        scalar_t sin = sin_start[h >> 1];
        scalar_t in1 = query[in_offset_q + h];
        scalar_t in2 = query[in_offset_q + h + 1];
        scalar_t out1 = in1 * cos - in2 * sin;
        scalar_t out2 = in2 * cos + in1 * sin;
        query_out[out_offset_q + h] = out1;
        query_out[out_offset_q + h + 1] = out2;
      }
      for (int64_t h = 0; h < HK; h += 2) {
        scalar_t cos = cos_start[h >> 1];
        scalar_t sin = sin_start[h >> 1];
        int64_t k_pe_offset = seq * key_stride_s;
        scalar_t in1_k = key[k_pe_offset + h];
        scalar_t in2_k = key[k_pe_offset + h + 1];
        scalar_t out1_k = in1_k * cos - in2_k * sin;
        scalar_t out2_k = in2_k * cos + in1_k * sin;
        key_out[out_offset_k + h] = out1_k;
        key_out[out_offset_k + h + 1] = out2_k;
      }
      // move to the next index
      data_index_step(seq, num_tokens, head_id, num_heads);
    }
  });
}

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
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t bVecSize = bVec::size();

  int64_t embed_dim = rotary_dim / 2;
  bool flag = (embed_dim % bVecSize == 0);
  int64_t loop_upper = flag ? embed_dim : embed_dim - bVecSize;

  auto compute_loop = [&](int64_t token_head, scalar_t* cache_ptr, scalar_t* qk) {
    int64_t j = 0;
    for (; j < loop_upper; j += bVecSize) {
      int64_t rot_offset = j;
      int64_t x_index = rot_offset;
      int64_t y_index = embed_dim + rot_offset;

      int64_t out_x = token_head + x_index;
      int64_t out_y = token_head + y_index;

      bVec _cos = bVec::loadu(cache_ptr + x_index);
      bVec _sin = bVec::loadu(cache_ptr + y_index);

      bVec _q_x = bVec::loadu(qk + out_x);
      bVec _q_y = bVec::loadu(qk + out_y);
      fVec _cos_0, _cos_1;
      std::tie(_cos_0, _cos_1) = at::vec::convert_to_float(_cos);
      fVec _sin_0, _sin_1;
      std::tie(_sin_0, _sin_1) = at::vec::convert_to_float(_sin);
      fVec _q_x_0, _q_x_1;
      std::tie(_q_x_0, _q_x_1) = at::vec::convert_to_float(_q_x);
      fVec _q_y_0, _q_y_1;
      std::tie(_q_y_0, _q_y_1) = at::vec::convert_to_float(_q_y);

      auto out1_0 = _q_x_0 * _cos_0 - _q_y_0 * _sin_0;
      auto out1_1 = _q_x_1 * _cos_1 - _q_y_1 * _sin_1;
      auto out1 = convert_from_float_ext<scalar_t>(out1_0, out1_1);
      out1.store(qk + out_x);

      auto out2_0 = _q_y_0 * _cos_0 + _q_x_0 * _sin_0;
      auto out2_1 = _q_y_1 * _cos_1 + _q_x_1 * _sin_1;
      auto out2 = convert_from_float_ext<scalar_t>(out2_0, out2_1);
      out2.store(qk + out_y);
    }
    if (!flag) {
      for (; j < embed_dim; ++j) {
        int64_t x_index = j;
        int64_t y_index = embed_dim + j;

        int64_t out_x = token_head + x_index;
        int64_t out_y = token_head + y_index;

        float _cos = cache_ptr[x_index];
        float _sin = cache_ptr[y_index];

        float _q_x = qk[out_x];
        float _q_y = qk[out_y];

        qk[out_x] = _q_x * _cos - _q_y * _sin;
        qk[out_y] = _q_y * _cos + _q_x * _sin;
      }
    }
  };

#pragma omp parallel for collapse(2)
  for (int64_t bs = 0; bs < batch_size; ++bs) {
    for (int64_t seq = 0; seq < seq_len; ++seq) {
      int64_t pos = positions[bs * seq_len + seq];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;

      for (int64_t i = 0; i < num_heads; ++i) {
        int64_t head_idx = i;
        int64_t token_head = bs * query_stride_b + seq * query_stride_s + head_idx * query_stride_h;
        compute_loop(token_head, cache_ptr, query);
      }

      for (int64_t i = 0; i < num_kv_heads; ++i) {
        int64_t head_idx = i;
        int64_t token_head = bs * key_stride_b + seq * key_stride_s + head_idx * key_stride_h;
        compute_loop(token_head, cache_ptr, key);
      }
    }
  }
}

template <typename scalar_t>
void apply_rotary_pos_emb_kernel_impl(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    float* __restrict__ cos,
    float* __restrict__ sin,
    int64_t query_stride_s,
    int64_t key_stride_s,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_tokens) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t bVecSize = bVec::size();
  constexpr int64_t fVecSize = fVec::size();

  int64_t embed_dim = head_size / 2;
  bool flag = (embed_dim % bVecSize == 0);
  int64_t loop_upper = flag ? embed_dim : embed_dim - bVecSize;

  auto compute_loop = [&](int64_t token_head, float* cos_ptr, float* sin_ptr, scalar_t* qk) {
    int64_t j = 0;
    for (; j < loop_upper; j += bVecSize) {
      int64_t rot_offset = j;
      int64_t x_index = rot_offset;
      int64_t y_index = embed_dim + rot_offset;

      int64_t out_x = token_head + x_index;
      int64_t out_y = token_head + y_index;

      fVec _cos_x_0 = fVec::loadu(cos_ptr + x_index);
      fVec _sin_x_0 = fVec::loadu(sin_ptr + x_index);
      fVec _cos_x_1 = fVec::loadu(cos_ptr + x_index + fVecSize);
      fVec _sin_x_1 = fVec::loadu(sin_ptr + x_index + fVecSize);

      fVec _cos_y_0 = fVec::loadu(cos_ptr + y_index);
      fVec _sin_y_0 = fVec::loadu(sin_ptr + y_index);
      fVec _cos_y_1 = fVec::loadu(cos_ptr + y_index + fVecSize);
      fVec _sin_y_1 = fVec::loadu(sin_ptr + y_index + fVecSize);

      bVec _q_x = bVec::loadu(qk + out_x);
      bVec _q_y = bVec::loadu(qk + out_y);
      fVec _q_x_0, _q_x_1;
      std::tie(_q_x_0, _q_x_1) = at::vec::convert_to_float(_q_x);
      fVec _q_y_0, _q_y_1;
      std::tie(_q_y_0, _q_y_1) = at::vec::convert_to_float(_q_y);

      auto out1_0 = _q_x_0 * _cos_x_0 - _q_y_0 * _sin_x_0;
      auto out1_1 = _q_x_1 * _cos_x_1 - _q_y_1 * _sin_x_1;
      auto out1 = convert_from_float_ext<scalar_t>(out1_0, out1_1);
      out1.store(qk + out_x);

      auto out2_0 = _q_y_0 * _cos_y_0 + _q_x_0 * _sin_y_0;
      auto out2_1 = _q_y_1 * _cos_y_1 + _q_x_1 * _sin_y_1;
      auto out2 = convert_from_float_ext<scalar_t>(out2_0, out2_1);
      out2.store(qk + out_y);
    }
    if (!flag) {
      for (; j < embed_dim; ++j) {
        int64_t x_index = j;
        int64_t y_index = embed_dim + j;

        int64_t out_x = token_head + x_index;
        int64_t out_y = token_head + y_index;

        float _cos_x = cos_ptr[x_index];
        float _sin_x = sin_ptr[x_index];
        float _cos_y = cos_ptr[y_index];
        float _sin_y = sin_ptr[y_index];

        float _q_x = qk[out_x];
        float _q_y = qk[out_y];

        qk[out_x] = _q_x * _cos_x - _q_y * _sin_x;
        qk[out_y] = _q_y * _cos_y + _q_x * _sin_y;
      }
    }
  };

  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    int64_t token_idx = {0};
    data_index_init(begin, token_idx, num_tokens);
    for (int i = begin; i < end; ++i) {
      float* cos_ptr = cos + token_idx * head_size;
      float* sin_ptr = sin + token_idx * head_size;

      for (int64_t i = 0; i < num_heads; ++i) {
        int64_t head_idx = i;
        int64_t token_head = token_idx * query_stride_s + head_idx * head_size;
        compute_loop(token_head, cos_ptr, sin_ptr, query);
      }

      for (int64_t i = 0; i < num_kv_heads; ++i) {
        int64_t head_idx = i;
        int64_t token_head = token_idx * key_stride_s + head_idx * head_size;
        compute_loop(token_head, cos_ptr, sin_ptr, key);
      }
      data_index_step(token_idx, num_tokens);
    }
  });
}

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

  at::parallel_for(0, batch_size * seq_len * num_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t bs = {0}, seq = {0}, i = {0};
    data_index_init(begin, bs, batch_size, seq, seq_len, i, num_heads);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t pos = positions[bs * seq_len + seq];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;
      scalar_t* cos_cache_ptr = cache_ptr;
      scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      int64_t head_idx = i;
      int64_t token_head = bs * query_stride_b + seq * query_stride_s + head_idx * query_stride_h;
      scalar_t* head_query = token_head + query;
      for (int64_t j = 0; j < embed_dim; j += 1) {
        int64_t rot_offset = j;
        int64_t x_index = 2 * rot_offset;
        int64_t y_index = 2 * rot_offset + 1;

        float cos = cos_cache_ptr[rot_offset];
        float sin = sin_cache_ptr[rot_offset];

        float x = head_query[x_index];
        float y = head_query[y_index];

        head_query[x_index] = x * cos - y * sin;
        head_query[y_index] = y * cos + x * sin;
      }
      data_index_step(bs, batch_size, seq, seq_len, i, num_heads);
    }
  });

  at::parallel_for(0, batch_size * seq_len * num_kv_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t bs = {0}, seq = {0}, i = {0};
    data_index_init(begin, bs, batch_size, seq, seq_len, i, num_kv_heads);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t pos = positions[bs * seq_len + seq];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;
      scalar_t* cos_cache_ptr = cache_ptr;
      scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      int64_t head_idx = i;
      int64_t token_head = bs * key_stride_b + seq * key_stride_s + head_idx * head_size;
      scalar_t* head_key = key + token_head;
      for (int64_t j = 0; j < embed_dim; j += 1) {
        int64_t rot_offset = j;
        int64_t x_index = 2 * rot_offset;
        int64_t y_index = 2 * rot_offset + 1;

        float cos = cos_cache_ptr[rot_offset];
        float sin = sin_cache_ptr[rot_offset];

        float x = head_key[x_index];
        float y = head_key[y_index];

        head_key[x_index] = x * cos - y * sin;
        head_key[y_index] = y * cos + x * sin;
      }
      data_index_step(bs, batch_size, seq, seq_len, i, num_kv_heads);
    }
  });
}

}  // namespace

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
  if (input_dim == 3) {
    // TODO: add support for head_dim != rotary_dim case when input_dim=3
    CHECK_EQ(query.size(-1), rotary_dim);
    // TODO: add support for kv_head != 1
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
  at::Tensor query_out = at::empty_like(query);
  at::Tensor key_out = at::empty_like(key);
  int64_t query_out_stride_s = query_out.stride(0);
  int64_t key_out_stride_s = key_out.stride(0);
  // output stride of num head dim is meaningful only when input dim = 3
  int64_t query_out_stride_h = input_dim == 3 ? query_out.stride(1) : -1;
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
      // TODO: add neox style support for rope impl with 3D inputs
      rotary_embedding_3D_kernel_impl<scalar_t>(
          query_out.data_ptr<scalar_t>(),
          key_out.data_ptr<scalar_t>(),
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          num_tokens,
          num_heads,
          num_kv_heads,
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

// query: [num_tokens, num_heads, head_size]
// key: [num_tokens, num_heads, head_size]
// cos: [num_tokens, head_size]
// sin: [num_tokens, head_size]
std::tuple<at::Tensor, at::Tensor>
apply_rotary_pos_emb_cpu(at::Tensor& query, at::Tensor& key, at::Tensor& cos, at::Tensor& sin) {
  RECORD_FUNCTION("sgl-kernel::apply_rotary_pos_emb_cpu", std::vector<c10::IValue>({query, key}));
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
  CHECK_INPUT(cos);
  CHECK_INPUT(sin);
  CHECK_DIM(3, query);
  CHECK_DIM(3, key);
  CHECK_DIM(2, cos);
  CHECK_DIM(2, sin);
  const auto input_dtype = query.scalar_type();
  int64_t num_tokens = query.size(0);
  CHECK_EQ(num_tokens, key.size(0));
  CHECK_EQ(num_tokens, cos.size(0));
  CHECK_EQ(num_tokens, sin.size(0));
  int64_t num_heads = query.size(1);
  CHECK_EQ(num_heads, key.size(1));
  int64_t head_size = query.size(2);
  CHECK_EQ(head_size, key.size(2));
  CHECK_EQ(head_size, cos.size(1));
  CHECK_EQ(head_size, sin.size(1));
  int64_t q_stride_s = query.stride(0);
  int64_t k_stride_s = key.stride(0);
  TORCH_CHECK(input_dtype == key.scalar_type(), "query and key must have the same data type");
  TORCH_CHECK(cos.scalar_type() == at::kFloat, "expect cos to be float32, got ", cos.scalar_type());
  TORCH_CHECK(sin.scalar_type() == at::kFloat, "expect sin to be float32, got ", sin.scalar_type());
  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "apply_rotary_pos_emb_cpu", [&] {
    apply_rotary_pos_emb_kernel_impl<scalar_t>(
        query.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        q_stride_s,
        k_stride_s,
        num_heads,
        num_heads,
        head_size,
        num_tokens);
  });
  return std::make_tuple(query, key);
}
