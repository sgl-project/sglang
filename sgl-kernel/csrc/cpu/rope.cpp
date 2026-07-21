#include "common.h"
#include "vec.h"

namespace {

struct RopeParams {
  // Treat all tensors as [B, S, H, D]
  //   2D [S, H * D] -> [1, S, H, D]
  //   3D [S, H, D]  -> [1, S, H, D]
  //   4D               [B, S, H, D]
  int64_t rotary_dim{0};
  int64_t head_size{0};
  int64_t batches{1}, seqlen{1}, num_heads{1}, num_heads_kv{1};
  int64_t q_strideB{0}, q_strideS{0}, q_strideH{0};
  int64_t k_strideB{0}, k_strideS{0}, k_strideH{0};

  RopeParams(const at::Tensor& query, const at::Tensor& key, int64_t head_size_, int64_t rotary_dim_)
      : rotary_dim(rotary_dim_), head_size(head_size_) {
    int64_t ndim = query.dim();
    switch (ndim) {
      case 2:
        seqlen = query.size(0);
        num_heads = query.size(1) / head_size;
        num_heads_kv = key.size(1) / head_size;
        q_strideS = query.stride(0);
        k_strideS = key.stride(0);
        q_strideH = head_size;
        k_strideH = head_size;
        break;
      case 3:
        seqlen = query.size(0);
        num_heads = query.size(1);
        num_heads_kv = key.size(1);
        q_strideS = query.stride(0);
        k_strideS = key.stride(0);
        q_strideH = query.stride(1);
        k_strideH = key.stride(1);
        break;
      case 4:
        batches = query.size(0);
        seqlen = query.size(1);
        num_heads = query.size(2);
        num_heads_kv = key.size(2);
        q_strideB = query.stride(0);
        k_strideB = key.stride(0);
        q_strideS = query.stride(1);
        k_strideS = key.stride(1);
        q_strideH = query.stride(2);
        k_strideH = key.stride(2);
        break;
      default:
        TORCH_CHECK(false, "Expected a 2D/3D/4D tensor, got ", ndim, "D.");
    }
  }

  inline int64_t rows() const {
    return batches * seqlen;
  }
  inline int64_t q_offset(int64_t b, int64_t s, int64_t h) const {
    return b * q_strideB + s * q_strideS + h * q_strideH;
  }
  inline int64_t k_offset(int64_t b, int64_t s, int64_t h) const {
    return b * k_strideB + s * k_strideS + h * k_strideH;
  }
  inline int64_t q_out_offset(int64_t b, int64_t s, int64_t h) const {
    return ((b * seqlen + s) * num_heads + h) * head_size;
  }
  inline int64_t k_out_offset(int64_t b, int64_t s, int64_t h) const {
    return ((b * seqlen + s) * num_heads_kv + h) * head_size;
  }
};

enum class RotaryMode {
  Interleaved,  // GPT-J
  Neox,
};

template <typename scalar_t, RotaryMode rotary_mode>
struct RotaryEmbedInternal;

template <typename scalar_t>
struct RotaryEmbedInternal<scalar_t, RotaryMode::Interleaved> {
  static inline void
  apply(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const scalar_t* __restrict__ cache, int size) {
    constexpr int kVecSize = at::vec::Vectorized<scalar_t>::size();
    const int half_size = size / 2;

    int d = 0;
    for (; d <= size - kVecSize; d += kVecSize) {
      auto [xy0, xy1] = load_float_vec2(input + d);
      auto [x, y] = at::vec::deinterleave2(xy0, xy1);
      auto cos = load_float_vec(cache + d / 2);
      auto sin = load_float_vec(cache + half_size + d / 2);
      auto out0 = x * cos - y * sin;
      auto out1 = y * cos + x * sin;
      std::tie(xy0, xy1) = at::vec::interleave2(out0, out1);
      convert_from_float_ext<scalar_t>(xy0, xy1).store(out + d);
    }
    for (; d < size; d += 2) {
      float x = input[d], y = input[d + 1];
      float cos = cache[d >> 1], sin = cache[half_size + (d >> 1)];
      out[d] = static_cast<scalar_t>(x * cos - y * sin);
      out[d + 1] = static_cast<scalar_t>(y * cos + x * sin);
    }
  }
};

template <typename scalar_t>
struct RotaryEmbedInternal<scalar_t, RotaryMode::Neox> {
  static inline void
  apply(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const scalar_t* __restrict__ cache, int size) {
    constexpr int kVecSize = at::vec::Vectorized<scalar_t>::size();

    const int half_size = size / 2;
    int d = 0;
    for (; d <= half_size - kVecSize; d += kVecSize) {
      auto [x0, x1] = load_float_vec2(input + d);
      auto [y0, y1] = load_float_vec2(input + half_size + d);
      auto [cos0, cos1] = load_float_vec2(cache + d);
      auto [sin0, sin1] = load_float_vec2(cache + half_size + d);
      auto out0 = x0 * cos0 - y0 * sin0;
      auto out1 = x1 * cos1 - y1 * sin1;
      auto out2 = y0 * cos0 + x0 * sin0;
      auto out3 = y1 * cos1 + x1 * sin1;
      convert_from_float_ext<scalar_t>(out0, out1).store(out + d);
      convert_from_float_ext<scalar_t>(out2, out3).store(out + half_size + d);
    }
    for (; d < half_size; ++d) {
      float x = input[d], y = input[d + half_size];
      float cos = cache[d], sin = cache[d + half_size];
      out[d] = static_cast<scalar_t>(x * cos - y * sin);
      out[d + half_size] = static_cast<scalar_t>(y * cos + x * sin);
    }
  }
};

template <typename scalar_t, RotaryMode mode, bool inplace, typename CachePos>
void rotary_embedding_kernel_impl(
    scalar_t* __restrict__ query_out,
    scalar_t* __restrict__ key_out,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const RopeParams& p,
    const CachePos& cache_pos) {
  at::parallel_for(0, p.rows(), 0, [&](int64_t begin, int64_t end) {
    int64_t bs = 0, seq = 0;
    data_index_init(begin, bs, p.batches, seq, p.seqlen);
    for (int64_t i = begin; i < end; ++i) {
      const scalar_t* cache = cache_pos(bs * p.seqlen + seq);
      for (int64_t h = 0; h < p.num_heads; ++h) {
        scalar_t* q_in = query + p.q_offset(bs, seq, h);
        scalar_t* q_out;
        if constexpr (inplace) {
          q_out = q_in;
        } else {
          q_out = query_out + p.q_out_offset(bs, seq, h);
        }
        RotaryEmbedInternal<scalar_t, mode>::apply(q_out, q_in, cache, p.rotary_dim);
      }
      for (int64_t h = 0; h < p.num_heads_kv; ++h) {
        scalar_t* k_in = key + p.k_offset(bs, seq, h);
        scalar_t* k_out;
        if constexpr (inplace) {
          k_out = k_in;
        } else {
          k_out = key_out + p.k_out_offset(bs, seq, h);
        }
        RotaryEmbedInternal<scalar_t, mode>::apply(k_out, k_in, cache, p.rotary_dim);
      }
      data_index_step(bs, p.batches, seq, p.seqlen);
    }
  });
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

}  // namespace

// 2D : [num_tokens, num_heads*head_size] inplace
// 3D : [num_tokens, num_heads, head_size] outplace
// 4D : [batch_size, seq_len, num_heads, head_size] inplace
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

  CHECK_DIM(2, cos_sin_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
  TORCH_CHECK(positions.scalar_type() == at::kLong, "expect positions to be int64, got ", positions.scalar_type());
  TORCH_CHECK(input_dtype == key.scalar_type(), "query and key must have the same data type");
  TORCH_CHECK(input_dtype == cos_sin_cache.scalar_type(), "query and cos_sin_cache must have the same data type");

  int64_t rotary_dim = cos_sin_cache.size(1);
  const RopeParams p{query, key, head_size, rotary_dim};
  TORCH_CHECK(positions.numel() == p.rows(), "positions.numel() must equal batch * seqlen");

  if (input_dim == 2) {
    TORCH_CHECK(query.size(-1) % head_size == 0, "query last dim must be divisible by head_size");
    TORCH_CHECK(key.size(-1) % head_size == 0, "key last dim must be divisible by head_size");
  }
  if (input_dim == 3) {
    // out-of-place path: align with legacy behavior, no partial rotary
    CHECK_EQ(query.size(-1), rotary_dim);
    CHECK_EQ(key.size(-1), rotary_dim);
    CHECK_EQ(head_size, rotary_dim);
  }
  if (input_dim == 4) {
    CHECK_EQ(query.size(0), key.size(0));
    CHECK_EQ(query.size(1), key.size(1));
  }

  at::Tensor query_out = (input_dim != 3) ? query : at::empty_like(query);
  at::Tensor key_out = (input_dim != 3) ? key : at::empty_like(key);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input_dtype, "rotary_embedding_cpu", [&] {
    AT_DISPATCH_BOOL(input_dim != 3, inplace, [&] {
      const scalar_t* cache_base = cos_sin_cache.data_ptr<scalar_t>();
      const int64_t* pos_ptr = positions.data_ptr<int64_t>();
      auto cache_pos = [cache_base, pos_ptr, rotary_dim](int64_t token) -> const scalar_t* {
        return cache_base + pos_ptr[token] * rotary_dim;
      };

      scalar_t* q_ptr = query.data_ptr<scalar_t>();
      scalar_t* k_ptr = key.data_ptr<scalar_t>();
      scalar_t* q_out_ptr = query_out.data_ptr<scalar_t>();
      scalar_t* k_out_ptr = key_out.data_ptr<scalar_t>();

      if (is_neox) {
        rotary_embedding_kernel_impl<scalar_t, RotaryMode::Neox, inplace>(
            q_out_ptr, k_out_ptr, q_ptr, k_ptr, p, cache_pos);
      } else {
        rotary_embedding_kernel_impl<scalar_t, RotaryMode::Interleaved, inplace>(
            q_out_ptr, k_out_ptr, q_ptr, k_ptr, p, cache_pos);
      }
    });
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
std::tuple<at::Tensor, at::Tensor> multimodal_rotary_embedding_cpu(
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

  const RopeParams p{query, key, head_size, rotary_dim};
  TORCH_CHECK(p.rotary_dim <= p.head_size, "rotary_dim must be <= head_size");
  TORCH_CHECK(p.rotary_dim % 2 == 0, "rotary_dim must be even");
  TORCH_CHECK(positions.size(-1) == p.rows(), "positions numel must equal batch*seqlen");

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
      const scalar_t* cache_base = cos_sin_cache.data_ptr<scalar_t>();
      const int64_t* pos_ptr = positions.data_ptr<int64_t>();
      auto cache_pos = [cache_base, pos_ptr, rotary_dim](int64_t token) -> const scalar_t* {
        return cache_base + pos_ptr[token] * rotary_dim;
      };

      scalar_t* q_ptr = query.data_ptr<scalar_t>();
      scalar_t* k_ptr = key.data_ptr<scalar_t>();

      if (is_neox) {
        rotary_embedding_kernel_impl<scalar_t, RotaryMode::Neox, true>(q_ptr, k_ptr, q_ptr, k_ptr, p, cache_pos);
      } else {
        rotary_embedding_kernel_impl<scalar_t, RotaryMode::Interleaved, true>(q_ptr, k_ptr, q_ptr, k_ptr, p, cache_pos);
      }
    });
  }
  return std::make_tuple(query, key);
}
