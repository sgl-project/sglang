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
    scalar_t* query,
    scalar_t* query_out,
    scalar_t* key,
    scalar_t* key_out,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t rotary_dim,
    int64_t query_stride_b,
    int64_t query_stride_s,
    int64_t query_stride_h,
    int64_t query_out_stride_b,
    int64_t query_out_stride_s,
    int64_t query_out_stride_h,
    int64_t key_stride_b,
    int64_t key_stride_s,
    int64_t key_stride_h,
    int64_t key_out_stride_b,
    int64_t key_out_stride_s,
    int64_t key_out_stride_h,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t batch_size,
    int64_t seq_len,
    bool inplace = false) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t bVecSize = bVec::size();

  int64_t embed_dim = rotary_dim / 2;
  bool flag = (embed_dim % bVecSize == 0);
  int64_t loop_upper = flag ? embed_dim : embed_dim - bVecSize;

  auto compute_loop = [&](int64_t in_offset, int64_t out_offset, scalar_t* cache_ptr, scalar_t* qk, scalar_t* qk_out) {
    int64_t j = 0;
    for (; j < loop_upper; j += bVecSize) {
      int64_t rot_offset = j;
      int64_t x_index = rot_offset;
      int64_t y_index = embed_dim + rot_offset;

      bVec _cos = bVec::loadu(cache_ptr + x_index);
      bVec _sin = bVec::loadu(cache_ptr + y_index);

      bVec _q_x = bVec::loadu(qk + in_offset + x_index);
      bVec _q_y = bVec::loadu(qk + in_offset + y_index);
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
      out1.store(qk_out + out_offset + x_index);

      auto out2_0 = _q_y_0 * _cos_0 + _q_x_0 * _sin_0;
      auto out2_1 = _q_y_1 * _cos_1 + _q_x_1 * _sin_1;
      auto out2 = convert_from_float_ext<scalar_t>(out2_0, out2_1);
      out2.store(qk_out + out_offset + y_index);
    }
    if (!flag) {
      for (; j < embed_dim; ++j) {
        int64_t x_index = j;
        int64_t y_index = embed_dim + j;

        float _cos = cache_ptr[x_index];
        float _sin = cache_ptr[y_index];

        float _q_x = qk[in_offset + x_index];
        float _q_y = qk[in_offset + y_index];

        qk_out[out_offset + x_index] = _q_x * _cos - _q_y * _sin;
        qk_out[out_offset + y_index] = _q_y * _cos + _q_x * _sin;
      }
    }
    // Copy non-rotary elements from input to output
    if (!inplace && rotary_dim < head_size) {
      std::memcpy(
          qk_out + out_offset + rotary_dim, qk + in_offset + rotary_dim, (head_size - rotary_dim) * sizeof(scalar_t));
    }
  };

#pragma omp parallel for collapse(2)
  for (int64_t bs = 0; bs < batch_size; ++bs) {
    for (int64_t seq = 0; seq < seq_len; ++seq) {
      int64_t pos = positions[bs * seq_len + seq];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;

      for (int64_t i = 0; i < num_heads; ++i) {
        int64_t head_idx = i;
        int64_t in_offset = bs * query_stride_b + seq * query_stride_s + head_idx * query_stride_h;
        int64_t out_offset = bs * query_out_stride_b + seq * query_out_stride_s + head_idx * query_out_stride_h;
        compute_loop(in_offset, out_offset, cache_ptr, query, query_out);
      }

      for (int64_t i = 0; i < num_kv_heads; ++i) {
        int64_t head_idx = i;
        int64_t in_offset = bs * key_stride_b + seq * key_stride_s + head_idx * key_stride_h;
        int64_t out_offset = bs * key_out_stride_b + seq * key_out_stride_s + head_idx * key_out_stride_h;
        compute_loop(in_offset, out_offset, cache_ptr, key, key_out);
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
void apply_rotary_pos_emb_kernel_impl(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos,
    scalar_t* __restrict__ sin,
    int64_t query_stride_s,
    int64_t key_stride_s,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_tokens) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t bVecSize = bVec::size();

  int64_t embed_dim = head_size / 2;
  bool flag = (embed_dim % bVecSize == 0);
  int64_t loop_upper = flag ? embed_dim : embed_dim - bVecSize;

  auto compute_loop = [&](int64_t token_head, scalar_t* cos_ptr, scalar_t* sin_ptr, scalar_t* qk) {
    int64_t j = 0;
    for (; j < loop_upper; j += bVecSize) {
      int64_t rot_offset = j;
      int64_t x_index = rot_offset;
      int64_t y_index = embed_dim + rot_offset;

      int64_t out_x = token_head + x_index;
      int64_t out_y = token_head + y_index;

      bVec _cos_x = bVec::loadu(cos_ptr + x_index);
      bVec _sin_x = bVec::loadu(sin_ptr + x_index);
      bVec _cos_y = bVec::loadu(cos_ptr + y_index);
      bVec _sin_y = bVec::loadu(sin_ptr + y_index);
      fVec _cos_x_0, _cos_x_1;
      std::tie(_cos_x_0, _cos_x_1) = at::vec::convert_to_float(_cos_x);
      fVec _sin_x_0, _sin_x_1;
      std::tie(_sin_x_0, _sin_x_1) = at::vec::convert_to_float(_sin_x);
      fVec _cos_y_0, _cos_y_1;
      std::tie(_cos_y_0, _cos_y_1) = at::vec::convert_to_float(_cos_y);
      fVec _sin_y_0, _sin_y_1;
      std::tie(_sin_y_0, _sin_y_1) = at::vec::convert_to_float(_sin_y);

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
      scalar_t* cos_ptr = cos + token_idx * head_size;
      scalar_t* sin_ptr = sin + token_idx * head_size;

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
inline scalar_t* get_cache_ptr(
    int64_t j,
    scalar_t* cache_t_ptr,
    scalar_t* cache_h_ptr,
    scalar_t* cache_w_ptr,
    int64_t mrope_section_t,
    int64_t mrope_section_h,
    int64_t mrope_section_w,
    bool mrope_interleaved) {
  if (mrope_interleaved) {
    if (j % 3 == 1 && j <= mrope_section_h * 3) return cache_h_ptr;
    if (j % 3 == 2 && j <= mrope_section_w * 3) return cache_w_ptr;
    return cache_t_ptr;
  }
  if (j < mrope_section_t) return cache_t_ptr;
  if (j < mrope_section_t + mrope_section_h) return cache_h_ptr;
  return cache_w_ptr;
}

template <typename scalar_t>
void multimodal_rotary_embedding_neox_2D_kernel_impl(
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
    int64_t num_tokens,
    int64_t mrope_section_t,
    int64_t mrope_section_h,
    int64_t mrope_section_w,
    int64_t positions_stride0,
    bool mrope_interleaved) {
  int64_t embed_dim = rotary_dim / 2;
  auto compute_loop =
      [&](int64_t token_head, scalar_t* cache_t_ptr, scalar_t* cache_h_ptr, scalar_t* cache_w_ptr, scalar_t* qk) {
        for (int64_t j = 0; j < embed_dim; ++j) {
          int64_t x_index = j;
          int64_t y_index = embed_dim + j;

          int64_t out_x = token_head + x_index;
          int64_t out_y = token_head + y_index;

          scalar_t* cache_ptr = get_cache_ptr(
              j,
              cache_t_ptr,
              cache_h_ptr,
              cache_w_ptr,
              mrope_section_t,
              mrope_section_h,
              mrope_section_w,
              mrope_interleaved);
          float _cos = cache_ptr[x_index];
          float _sin = cache_ptr[y_index];

          float _q_x = qk[out_x];
          float _q_y = qk[out_y];

          qk[out_x] = _q_x * _cos - _q_y * _sin;
          qk[out_y] = _q_y * _cos + _q_x * _sin;
        }
      };
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    int64_t token_idx = {0};
    data_index_init(begin, token_idx, num_tokens);
    for (int i = begin; i < end; ++i) {
      int64_t pos_t = positions[token_idx];
      int64_t pos_h = positions[positions_stride0 + token_idx];
      int64_t pos_w = positions[positions_stride0 * 2 + token_idx];
      scalar_t* cache_t_ptr = cos_sin_cache + pos_t * rotary_dim;
      scalar_t* cache_h_ptr = cos_sin_cache + pos_h * rotary_dim;
      scalar_t* cache_w_ptr = cos_sin_cache + pos_w * rotary_dim;

      for (int64_t i = 0; i < num_heads; ++i) {
        int64_t head_idx = i;
        int64_t token_head = token_idx * query_stride_s + head_idx * head_size;
        compute_loop(token_head, cache_t_ptr, cache_h_ptr, cache_w_ptr, query);
      }

      for (int64_t i = 0; i < num_kv_heads; ++i) {
        int64_t head_idx = i;
        int64_t token_head = token_idx * key_stride_s + head_idx * head_size;
        compute_loop(token_head, cache_t_ptr, cache_h_ptr, cache_w_ptr, key);
      }
      data_index_step(token_idx, num_tokens);
    }
  });
}

template <typename scalar_t>
void rotary_embedding_4D_kernel_impl(
    int64_t* __restrict__ positions,
    scalar_t* query,
    scalar_t* query_out,
    scalar_t* key,
    scalar_t* key_out,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t rotary_dim,
    int64_t query_stride_b,
    int64_t query_stride_s,
    int64_t query_stride_h,
    int64_t query_out_stride_b,
    int64_t query_out_stride_s,
    int64_t query_out_stride_h,
    int64_t key_stride_b,
    int64_t key_stride_s,
    int64_t key_stride_h,
    int64_t key_out_stride_b,
    int64_t key_out_stride_s,
    int64_t key_out_stride_h,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t batch_size,
    int64_t seq_len,
    bool inplace = false) {
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
      int64_t in_offset = bs * query_stride_b + seq * query_stride_s + head_idx * query_stride_h;
      int64_t out_offset = bs * query_out_stride_b + seq * query_out_stride_s + head_idx * query_out_stride_h;
      scalar_t* head_query = in_offset + query;
      scalar_t* head_query_out = out_offset + query_out;
      for (int64_t j = 0; j < embed_dim; j += 1) {
        int64_t rot_offset = j;
        int64_t x_index = 2 * rot_offset;
        int64_t y_index = 2 * rot_offset + 1;

        float cos = cos_cache_ptr[rot_offset];
        float sin = sin_cache_ptr[rot_offset];

        float x = head_query[x_index];
        float y = head_query[y_index];

        head_query_out[x_index] = x * cos - y * sin;
        head_query_out[y_index] = y * cos + x * sin;
      }
      // Copy non-rotary elements from input to output
      if (!inplace && rotary_dim < head_size) {
        std::memcpy(head_query_out + rotary_dim, head_query + rotary_dim, (head_size - rotary_dim) * sizeof(scalar_t));
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
      int64_t in_offset = bs * key_stride_b + seq * key_stride_s + head_idx * key_stride_h;
      int64_t out_offset = bs * key_out_stride_b + seq * key_out_stride_s + head_idx * key_out_stride_h;
      scalar_t* head_key = key + in_offset;
      scalar_t* head_key_out = key_out + out_offset;
      for (int64_t j = 0; j < embed_dim; j += 1) {
        int64_t rot_offset = j;
        int64_t x_index = 2 * rot_offset;
        int64_t y_index = 2 * rot_offset + 1;

        float cos = cos_cache_ptr[rot_offset];
        float sin = sin_cache_ptr[rot_offset];

        float x = head_key[x_index];
        float y = head_key[y_index];

        head_key_out[x_index] = x * cos - y * sin;
        head_key_out[y_index] = y * cos + x * sin;
      }
      // Copy non-rotary elements from input to output
      if (!inplace && rotary_dim < head_size) {
        std::memcpy(head_key_out + rotary_dim, head_key + rotary_dim, (head_size - rotary_dim) * sizeof(scalar_t));
      }
      data_index_step(bs, batch_size, seq, seq_len, i, num_kv_heads);
    }
  });
}

template <typename scalar_t>
void multimodal_rotary_embedding_2D_kernel_impl(
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
    int64_t num_tokens,
    int64_t mrope_section_t,
    int64_t mrope_section_h,
    int64_t mrope_section_w,
    int64_t positions_stride0,
    bool mrope_interleaved) {
  int64_t embed_dim = rotary_dim / 2;
  auto compute_loop = [&](scalar_t* cache_t_ptr, scalar_t* cache_h_ptr, scalar_t* cache_w_ptr, scalar_t* head_query) {
    for (int64_t j = 0; j < embed_dim; j += 1) {
      int64_t rot_offset = j;
      int64_t x_index = 2 * rot_offset;
      int64_t y_index = 2 * rot_offset + 1;

      scalar_t* cache_ptr = get_cache_ptr(
          j,
          cache_t_ptr,
          cache_h_ptr,
          cache_w_ptr,
          mrope_section_t,
          mrope_section_h,
          mrope_section_w,
          mrope_interleaved);
      float cos = cache_ptr[rot_offset];
      float sin = cache_ptr[rot_offset + embed_dim];

      float x = head_query[x_index];
      float y = head_query[y_index];

      head_query[x_index] = x * cos - y * sin;
      head_query[y_index] = y * cos + x * sin;
    }
  };
  at::parallel_for(0, num_tokens * num_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t token_idx = {0}, i = {0};
    data_index_init(begin, token_idx, num_tokens, i, num_heads);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t pos_t = positions[token_idx];
      int64_t pos_h = positions[positions_stride0 + token_idx];
      int64_t pos_w = positions[positions_stride0 * 2 + token_idx];
      scalar_t* cache_t_ptr = cos_sin_cache + pos_t * rotary_dim;
      scalar_t* cache_h_ptr = cos_sin_cache + pos_h * rotary_dim;
      scalar_t* cache_w_ptr = cos_sin_cache + pos_w * rotary_dim;
      int64_t head_idx = i;
      int64_t token_head = token_idx * query_stride_s + head_idx * head_size;
      scalar_t* head_query = token_head + query;
      compute_loop(cache_t_ptr, cache_h_ptr, cache_w_ptr, head_query);
      data_index_step(token_idx, num_tokens, i, num_heads);
    }
  });

  at::parallel_for(0, num_tokens * num_kv_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t token_idx{0}, i = {0};
    data_index_init(begin, token_idx, num_tokens, i, num_kv_heads);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t pos_t = positions[token_idx];
      int64_t pos_h = positions[positions_stride0 + token_idx];
      int64_t pos_w = positions[positions_stride0 * 2 + token_idx];
      scalar_t* cache_t_ptr = cos_sin_cache + pos_t * rotary_dim;
      scalar_t* cache_h_ptr = cos_sin_cache + pos_h * rotary_dim;
      scalar_t* cache_w_ptr = cos_sin_cache + pos_w * rotary_dim;
      int64_t head_idx = i;
      int64_t token_head = token_idx * key_stride_s + head_idx * head_size;
      scalar_t* head_key = key + token_head;
      compute_loop(cache_t_ptr, cache_h_ptr, cache_w_ptr, head_key);
      data_index_step(token_idx, num_tokens, i, num_kv_heads);
    }
  });
}

// AVX512 helpers: load/store fVecSize reduced-precision elements as float vectors.
// Used by the half-width vectorized path when embed_dim < bVecSize but >= fVecSize.
#if defined(CPU_CAPABILITY_AVX512)
template <typename T>
inline at::vec::Vectorized<float> load_as_fvec(const T* ptr);

template <>
inline at::vec::Vectorized<float> load_as_fvec<at::BFloat16>(const at::BFloat16* ptr) {
  return at::vec::Vectorized<float>(CVT_BF16_TO_FP32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))));
}

template <>
inline at::vec::Vectorized<float> load_as_fvec<at::Half>(const at::Half* ptr) {
  return at::vec::Vectorized<float>(CVT_FP16_TO_FP32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))));
}

template <>
inline at::vec::Vectorized<float> load_as_fvec<float>(const float* ptr) {
  return at::vec::Vectorized<float>::loadu(ptr);
}

template <typename T>
inline void store_fvec(T* ptr, const at::vec::Vectorized<float>& v);

template <>
inline void store_fvec<at::BFloat16>(at::BFloat16* ptr, const at::vec::Vectorized<float>& v) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), (__m256i)_mm512_cvtneps_pbh(__m512(v)));
}

template <>
inline void store_fvec<at::Half>(at::Half* ptr, const at::vec::Vectorized<float>& v) {
  _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(ptr), _mm512_cvtps_ph(__m512(v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
#endif

// Apply multidimensional RoPE to query and key tensors simultaneously.
// q/k: [num_tokens, num_heads, head_dim], cos/sin: [num_tokens, head_dim]
// Splits head_dim into ndim=2 chunks and applies standard rotary to each independently.
// Fusing query and key into a single pass improves cos/sin cache reuse.
// cos/sin layout per chunk (chunk_size elements): [cos_half0, cos_half1] matching
// rotate_half pattern: out_x = x_first * cos_first - x_second * sin_first
//                      out_y = x_second * cos_second + x_first * sin_second
template <typename scalar_t, typename param_t>
void apply_multidimensional_rope_kernel_impl(
    scalar_t* __restrict__ q,
    scalar_t* __restrict__ k,
    param_t* __restrict__ cos,
    param_t* __restrict__ sin,
    int64_t q_stride_s,
    int64_t k_stride_s,
    int64_t num_heads,
    int64_t head_dim,
    int64_t num_tokens) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t bVecSize = bVec::size();
  constexpr int64_t fVecSize = fVec::size();

  constexpr int64_t ndim = 2;
  int64_t chunk_size = head_dim / ndim;
  int64_t embed_dim = chunk_size / 2;

  // Vectorized compute loop for a single chunk within a single head.
  // x: pointer to tensor data (query or key)
  // token_head: offset into x for (token, head, chunk)
  // cos_ptr/sin_ptr: pointer to the cos/sin data for the chunk
  auto compute_loop = [&](scalar_t* x, int64_t token_head, param_t* cos_ptr, param_t* sin_ptr) {
    int64_t j = 0;
    // Full-width vectorized loop (bVecSize elements per iteration)
    for (; j + bVecSize <= embed_dim; j += bVecSize) {
      int64_t x_index = j;
      int64_t y_index = embed_dim + j;

      int64_t out_x = token_head + x_index;
      int64_t out_y = token_head + y_index;

      // Load cos/sin vectors
      fVec _cos_x_0, _cos_x_1, _sin_x_0, _sin_x_1;
      fVec _cos_y_0, _cos_y_1, _sin_y_0, _sin_y_1;
      if constexpr (std::is_same_v<param_t, float>) {
        _cos_x_0 = fVec::loadu(cos_ptr + x_index);
        _sin_x_0 = fVec::loadu(sin_ptr + x_index);
        _cos_x_1 = fVec::loadu(cos_ptr + x_index + fVecSize);
        _sin_x_1 = fVec::loadu(sin_ptr + x_index + fVecSize);

        _cos_y_0 = fVec::loadu(cos_ptr + y_index);
        _sin_y_0 = fVec::loadu(sin_ptr + y_index);
        _cos_y_1 = fVec::loadu(cos_ptr + y_index + fVecSize);
        _sin_y_1 = fVec::loadu(sin_ptr + y_index + fVecSize);
      } else {
        using pVec = at::vec::Vectorized<param_t>;
        pVec _cos_x = pVec::loadu(cos_ptr + x_index);
        pVec _sin_x = pVec::loadu(sin_ptr + x_index);
        pVec _cos_y = pVec::loadu(cos_ptr + y_index);
        pVec _sin_y = pVec::loadu(sin_ptr + y_index);
        std::tie(_cos_x_0, _cos_x_1) = at::vec::convert_to_float(_cos_x);
        std::tie(_sin_x_0, _sin_x_1) = at::vec::convert_to_float(_sin_x);
        std::tie(_cos_y_0, _cos_y_1) = at::vec::convert_to_float(_cos_y);
        std::tie(_sin_y_0, _sin_y_1) = at::vec::convert_to_float(_sin_y);
      }

      bVec _q_x = bVec::loadu(x + out_x);
      bVec _q_y = bVec::loadu(x + out_y);
      fVec _q_x_0, _q_x_1;
      std::tie(_q_x_0, _q_x_1) = at::vec::convert_to_float(_q_x);
      fVec _q_y_0, _q_y_1;
      std::tie(_q_y_0, _q_y_1) = at::vec::convert_to_float(_q_y);

      auto out1_0 = _q_x_0 * _cos_x_0 - _q_y_0 * _sin_x_0;
      auto out1_1 = _q_x_1 * _cos_x_1 - _q_y_1 * _sin_x_1;
      auto out1 = convert_from_float_ext<scalar_t>(out1_0, out1_1);
      out1.store(x + out_x);

      auto out2_0 = _q_y_0 * _cos_y_0 + _q_x_0 * _sin_y_0;
      auto out2_1 = _q_y_1 * _cos_y_1 + _q_x_1 * _sin_y_1;
      auto out2 = convert_from_float_ext<scalar_t>(out2_0, out2_1);
      out2.store(x + out_y);
    }
#if defined(CPU_CAPABILITY_AVX512)
    // Half-width vectorized loop (fVecSize elements per iteration).
    // Handles the case where embed_dim < bVecSize but >= fVecSize,
    // e.g. head_dim=64 -> embed_dim=16 == fVecSize on AVX512.
    for (; j + fVecSize <= embed_dim; j += fVecSize) {
      int64_t x_index = j;
      int64_t y_index = embed_dim + j;

      int64_t out_x = token_head + x_index;
      int64_t out_y = token_head + y_index;

      fVec _cos_x = load_as_fvec(cos_ptr + x_index);
      fVec _sin_x = load_as_fvec(sin_ptr + x_index);
      fVec _cos_y = load_as_fvec(cos_ptr + y_index);
      fVec _sin_y = load_as_fvec(sin_ptr + y_index);

      fVec _q_x = load_as_fvec(x + out_x);
      fVec _q_y = load_as_fvec(x + out_y);

      auto res_x = _q_x * _cos_x - _q_y * _sin_x;
      auto res_y = _q_y * _cos_y + _q_x * _sin_y;

      store_fvec(x + out_x, res_x);
      store_fvec(x + out_y, res_y);
    }
#endif
    // Scalar fallback for remaining elements
    for (; j < embed_dim; ++j) {
      int64_t x_index = j;
      int64_t y_index = embed_dim + j;

      int64_t out_x = token_head + x_index;
      int64_t out_y = token_head + y_index;

      float _cos_x = cos_ptr[x_index];
      float _sin_x = sin_ptr[x_index];
      float _cos_y = cos_ptr[y_index];
      float _sin_y = sin_ptr[y_index];

      float _q_x = x[out_x];
      float _q_y = x[out_y];

      x[out_x] = _q_x * _cos_x - _q_y * _sin_x;
      x[out_y] = _q_y * _cos_y + _q_x * _sin_y;
    }
  };

  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    int64_t token_idx = {0};
    data_index_init(begin, token_idx, num_tokens);
    for (int64_t i = begin; i < end; ++i) {
      for (int64_t d = 0; d < ndim; ++d) {
        int64_t chunk_offset = d * chunk_size;
        param_t* cos_ptr = cos + token_idx * head_dim + chunk_offset;
        param_t* sin_ptr = sin + token_idx * head_dim + chunk_offset;

        for (int64_t h = 0; h < num_heads; ++h) {
          int64_t q_token_head = token_idx * q_stride_s + h * head_dim + chunk_offset;
          compute_loop(q, q_token_head, cos_ptr, sin_ptr);
          int64_t k_token_head = token_idx * k_stride_s + h * head_dim + chunk_offset;
          compute_loop(k, k_token_head, cos_ptr, sin_ptr);
        }
      }
      data_index_step(token_idx, num_tokens);
    }
  });
}

}  // namespace

// query: [num_tokens, num_heads, head_dim]
// key:   [num_tokens, num_heads, head_dim]
// cos:   [num_tokens, head_dim]
// sin:   [num_tokens, head_dim]
// Applies 2-D multidimensional RoPE: splits head_dim into 2 chunks and applies
// standard rotary embedding to each independently (in-place on query and key).
std::tuple<at::Tensor, at::Tensor>
apply_multidimensional_rope_cpu(at::Tensor& query, at::Tensor& key, at::Tensor& cos, at::Tensor& sin) {
  RECORD_FUNCTION("sgl-kernel::apply_multidimensional_rope_cpu", std::vector<c10::IValue>({query, key}));
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
  int64_t head_dim = query.size(2);
  CHECK_EQ(head_dim, key.size(2));
  CHECK_EQ(head_dim, cos.size(1));
  CHECK_EQ(head_dim, sin.size(1));
  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be divisible by 2 (ndim=2)");
  TORCH_CHECK(head_dim % 4 == 0, "head_dim must be divisible by 4 so each RoPE chunk is even");
  TORCH_CHECK(
      query.stride(1) == head_dim,
      "query must be contiguous across heads: expected stride(1) == head_dim, got stride(1)=",
      query.stride(1),
      " and head_dim=",
      head_dim);
  TORCH_CHECK(
      key.stride(1) == head_dim,
      "key must be contiguous across heads: expected stride(1) == head_dim, got stride(1)=",
      key.stride(1),
      " and head_dim=",
      head_dim);
  int64_t q_stride_s = query.stride(0);
  int64_t k_stride_s = key.stride(0);
  TORCH_CHECK(input_dtype == key.scalar_type(), "query and key must have the same data type");
  TORCH_CHECK(cos.scalar_type() == sin.scalar_type(), "cos and sin must have the same data type");
  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(input_dtype, cos.scalar_type(), "apply_multidimensional_rope_cpu", [&] {
    apply_multidimensional_rope_kernel_impl<scalar_t, param_t>(
        query.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        cos.data_ptr<param_t>(),
        sin.data_ptr<param_t>(),
        q_stride_s,
        k_stride_s,
        num_heads,
        head_dim,
        num_tokens);
  });
  return std::make_tuple(query, key);
}

std::tuple<at::Tensor, at::Tensor> rotary_embedding_cpu(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    bool is_neox) {
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
  int64_t query_out_stride_h = input_dim == 2 ? head_size : query_out.stride(-2);
  int64_t key_out_stride_h = input_dim == 2 ? head_size : key_out.stride(-2);
  int64_t query_out_stride_b = 0;
  int64_t key_out_stride_b = 0;
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
    query_out_stride_b = query_out.stride(0);
    key_out_stride_b = key_out.stride(0);
    query_out_stride_s = query_out.stride(1);
    key_out_stride_s = key_out.stride(1);
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
            query_out.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            key_out.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            query_stride_b,
            query_stride_s,
            query_stride_h,
            query_out_stride_b,
            query_out_stride_s,
            query_out_stride_h,
            key_stride_b,
            key_stride_s,
            key_stride_h,
            key_out_stride_b,
            key_out_stride_s,
            key_out_stride_h,
            num_heads,
            num_kv_heads,
            head_size,
            batch_size,
            seq_len);
      } else {
        rotary_embedding_4D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            query_out.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            key_out.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            query_stride_b,
            query_stride_s,
            query_stride_h,
            query_out_stride_b,
            query_out_stride_s,
            query_out_stride_h,
            key_stride_b,
            key_stride_s,
            key_stride_h,
            key_out_stride_b,
            key_out_stride_s,
            key_out_stride_h,
            num_heads,
            num_kv_heads,
            head_size,
            batch_size,
            seq_len);
      }

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
  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "apply_rotary_pos_emb_cpu", [&] {
    if (cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat) {
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
    } else if (cos.scalar_type() == input_dtype && sin.scalar_type() == input_dtype) {
      apply_rotary_pos_emb_kernel_impl<scalar_t>(
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos.data_ptr<scalar_t>(),
          sin.data_ptr<scalar_t>(),
          q_stride_s,
          k_stride_s,
          num_heads,
          num_heads,
          head_size,
          num_tokens);
    } else {
      TORCH_CHECK(
          false, "cos and sin must have the same data type, and must be either float or the same type as query/key");
    }
  });
  return std::make_tuple(query, key);
}

// positions: [num_tokens] (text only) or [3, num_tokens] (T/H/W positions with multimodal inputs)
// query: [num_tokens, num_heads * head_size]
// key: [num_tokens, num_kv_heads * head_size]
// cos_sin_cache: [max_position_embeddings, rotary_dim]
// mrope_section: [t, h, w]
void multimodal_rotary_embedding_cpu(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    const std::optional<std::vector<int64_t>>& mrope_section,
    bool mrope_interleaved,
    bool is_neox) {
  TORCH_CHECK(positions.dim() == 1 || positions.dim() == 2, "positions must be a 1D or 2D tensor");
  CHECK_DIM(2, query);
  CHECK_DIM(2, key);
  CHECK_DIM(2, cos_sin_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
  int64_t rotary_dim = cos_sin_cache.size(1);
  int64_t num_tokens = positions.size(-1);
  CHECK_EQ(key.size(0), num_tokens);
  CHECK_EQ(query.size(0), num_tokens);
  const auto input_dtype = query.scalar_type();
  TORCH_CHECK(positions.scalar_type() == at::kLong, "expect positions to be int64, got ", positions.scalar_type());
  TORCH_CHECK(input_dtype == key.scalar_type(), "query and key must have the same data type");
  TORCH_CHECK(input_dtype == cos_sin_cache.scalar_type(), "query and cos_sin_cache must have the same data type");

  int64_t num_heads = query.size(-1) / head_size;
  int64_t num_kv_heads = key.size(-1) / head_size;
  int64_t key_stride_s = key.stride(0);
  int64_t query_stride_s = query.stride(0);

  if (positions.dim() == 2) {
    TORCH_CHECK(mrope_section.has_value(), "mrope_section must be provided when positions is 2D");
    auto mrope_section_val = mrope_section.value();
    CHECK_EQ(mrope_section_val.size(), 3);
    CHECK_EQ(positions.size(0), 3);
    int64_t mrope_section_t = mrope_section_val[0];
    int64_t mrope_section_h = mrope_section_val[1];
    int64_t mrope_section_w = mrope_section_val[2];
    int64_t positions_stride0 = positions.stride(0);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input_dtype, "rotary_embedding_cpu", [&] {
      if (is_neox) {
        multimodal_rotary_embedding_neox_2D_kernel_impl<scalar_t>(
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
            num_tokens,
            mrope_section_t,
            mrope_section_h,
            mrope_section_w,
            positions_stride0,
            mrope_interleaved);
      } else {
        multimodal_rotary_embedding_2D_kernel_impl<scalar_t>(
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
            num_tokens,
            mrope_section_t,
            mrope_section_h,
            mrope_section_w,
            positions_stride0,
            mrope_interleaved);
      }
    });
  } else {  // positions.dim() == 1
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input_dtype, "rotary_embedding_cpu", [&] {
      if (is_neox) {
        rotary_embedding_neox_4D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            // query input strides (batch, seq, head)
            0,
            query_stride_s,
            head_size,
            // query output strides (same as input: in-place)
            0,
            query_stride_s,
            head_size,
            // key input strides (batch, seq, head)
            0,
            key_stride_s,
            head_size,
            // key output strides (same as input: in-place)
            0,
            key_stride_s,
            head_size,
            num_heads,
            num_kv_heads,
            head_size,
            1,
            num_tokens,
            true);
      } else {
        rotary_embedding_4D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            // query input strides (batch, seq, head)
            0,
            query_stride_s,
            head_size,
            // query output strides (same as input: in-place)
            0,
            query_stride_s,
            head_size,
            // key input strides (batch, seq, head)
            0,
            key_stride_s,
            head_size,
            // key output strides (same as input: in-place)
            0,
            key_stride_s,
            head_size,
            num_heads,
            num_kv_heads,
            head_size,
            1,
            num_tokens,
            true);
      }
    });
  }
}
