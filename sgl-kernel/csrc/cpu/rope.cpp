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
void rotary_embedding_neox_2D_kernel_impl(
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t rotary_dim,
    int64_t query_stride_s,
    int64_t key_stride_s,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_tokens) {
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

#pragma omp parallel for
  for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    int64_t pos = positions[token_idx];
    scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;

    for (int64_t i = 0; i < num_heads; ++i) {
      int64_t head_idx = i;
      int64_t token_head = token_idx * query_stride_s + head_idx * head_size;
      compute_loop(token_head, cache_ptr, query);
    }

    for (int64_t i = 0; i < num_kv_heads; ++i) {
      int64_t head_idx = i;
      int64_t token_head = token_idx * key_stride_s + head_idx * head_size;
      compute_loop(token_head, cache_ptr, key);
    }
  }
}

template <typename scalar_t>
void rotary_embedding_2D_kernel_impl(
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t rotary_dim,
    int64_t query_stride_s,
    int64_t key_stride_s,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_tokens) {
  int64_t embed_dim = rotary_dim / 2;

  at::parallel_for(0, num_tokens * num_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t token_idx = {0}, i = {0};
    data_index_init(begin, token_idx, num_tokens, i, num_heads);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t pos = positions[token_idx];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;
      scalar_t* cos_cache_ptr = cache_ptr;
      scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      int64_t head_idx = i;
      int64_t token_head = token_idx * query_stride_s + head_idx * head_size;
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
      data_index_step(token_idx, num_tokens, i, num_heads);
    }
  });

  at::parallel_for(0, num_tokens * num_kv_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t token_idx{0}, i = {0};
    data_index_init(begin, token_idx, num_tokens, i, num_kv_heads);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t pos = positions[token_idx];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;
      scalar_t* cos_cache_ptr = cache_ptr;
      scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      int64_t head_idx = i;
      int64_t token_head = token_idx * key_stride_s + head_idx * head_size;
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
      data_index_step(token_idx, num_tokens, i, num_kv_heads);
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
      input_dim == 2 || input_dim == 3,
      " Query/Key must be 2D [num_tokens, num_heads*head_size] or 3D [num_tokens, num_heads, head_size] tensor");
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
  CHECK_EQ(key.size(0), num_tokens);
  CHECK_EQ(query.size(0), num_tokens);

  TORCH_CHECK(positions.scalar_type() == at::kLong, "expect positions to be int64, got ", positions.scalar_type());
  TORCH_CHECK(input_dtype == key.scalar_type(), "query and key must have the same data type");
  TORCH_CHECK(input_dtype == cos_sin_cache.scalar_type(), "query and cos_sin_cache must have the same data type");

  int64_t num_heads = input_dim == 2 ? query.size(-1) / head_size : query.size(1);
  int64_t num_kv_heads = input_dim == 2 ? key.size(-1) / head_size : key.size(1);
  int64_t key_stride_s = key.stride(0);
  int64_t query_stride_s = query.stride(0);

  // input stride of num head dim is meaningful only when input dim = 3
  int64_t query_stride_h = input_dim == 3 ? query.stride(1) : -1;
  at::Tensor query_out = at::empty_like(query);
  at::Tensor key_out = at::empty_like(key);
  int64_t query_out_stride_s = query_out.stride(0);
  int64_t key_out_stride_s = key_out.stride(0);
  // output stride of num head dim is meaningful only when input dim = 3
  int64_t query_out_stride_h = input_dim == 3 ? query_out.stride(1) : -1;

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input_dtype, "rotary_embedding_cpu", [&] {
    if (input_dim == 2) {
      if (is_neox) {
        rotary_embedding_neox_2D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            query_stride_s,
            key_stride_s,
            num_heads,
            num_kv_heads,
            head_size,
            num_tokens);
      } else {
        rotary_embedding_2D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            query_stride_s,
            key_stride_s,
            num_heads,
            num_kv_heads,
            head_size,
            num_tokens);
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
