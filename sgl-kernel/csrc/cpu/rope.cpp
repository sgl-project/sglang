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
  Interleaved,  // GPT-J / packed [cos|sin]
  Neox,         // packed [cos|sin]
  NeoxFull,     // split cos/sin each of length head_size (HF rotate_half)
};

// Already-indexed cos/sin rows for apply_rotary_pos_emb style.
template <typename param_t>
struct SplitCosSinRow {
  const param_t* cos;
  const param_t* sin;
};

// Already-indexed T/H/W cache rows for 2D mRoPE (no gathered buffer).
template <typename scalar_t>
struct MropeCosSinRow {
  const scalar_t* cache_t;
  const scalar_t* cache_h;
  const scalar_t* cache_w;
  int64_t section_t;
  int64_t section_h;
  int64_t section_w;
  bool interleaved;

  inline const scalar_t* ptr_at(int64_t j) const {
    if (interleaved) {
      if (j % 3 == 1 && j <= section_h * 3) return cache_h;
      if (j % 3 == 2 && j <= section_w * 3) return cache_w;
      return cache_t;
    }
    if (j < section_t) return cache_t;
    if (j < section_t + section_h) return cache_h;
    return cache_w;
  }
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

  // mRoPE: cos/sin may come from different T/H/W rows per pair index.
  static inline void
  apply(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, MropeCosSinRow<scalar_t> cache, int size) {
    const int half_size = size / 2;
    for (int j = 0; j < half_size; ++j) {
      const scalar_t* src = cache.ptr_at(j);
      float cos = src[j], sin = src[j + half_size];
      float x = input[2 * j], y = input[2 * j + 1];
      out[2 * j] = static_cast<scalar_t>(x * cos - y * sin);
      out[2 * j + 1] = static_cast<scalar_t>(y * cos + x * sin);
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

  // mRoPE: cos/sin may come from different T/H/W rows per rotary index.
  static inline void
  apply(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, MropeCosSinRow<scalar_t> cache, int size) {
    const int half_size = size / 2;
    for (int j = 0; j < half_size; ++j) {
      const scalar_t* src = cache.ptr_at(j);
      float cos = src[j], sin = src[j + half_size];
      float x = input[j], y = input[j + half_size];
      out[j] = static_cast<scalar_t>(x * cos - y * sin);
      out[j + half_size] = static_cast<scalar_t>(y * cos + x * sin);
    }
  }
};

template <typename scalar_t>
struct RotaryEmbedInternal<scalar_t, RotaryMode::NeoxFull> {
  template <typename CosT>
  static inline void
  apply(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, SplitCosSinRow<CosT> cache, int size) {
    constexpr int kVecSize = at::vec::Vectorized<scalar_t>::size();
    const int half_size = size / 2;
    int d = 0;
    for (; d <= half_size - kVecSize; d += kVecSize) {
      auto [x0, x1] = load_float_vec2(input + d);
      auto [y0, y1] = load_float_vec2(input + half_size + d);
      auto [cos_x0, cos_x1] = load_float_vec2(cache.cos + d);
      auto [sin_x0, sin_x1] = load_float_vec2(cache.sin + d);
      auto [cos_y0, cos_y1] = load_float_vec2(cache.cos + half_size + d);
      auto [sin_y0, sin_y1] = load_float_vec2(cache.sin + half_size + d);
      auto out0 = x0 * cos_x0 - y0 * sin_x0;
      auto out1 = x1 * cos_x1 - y1 * sin_x1;
      auto out2 = y0 * cos_y0 + x0 * sin_y0;
      auto out3 = y1 * cos_y1 + x1 * sin_y1;
      convert_from_float_ext<scalar_t>(out0, out1).store(out + d);
      convert_from_float_ext<scalar_t>(out2, out3).store(out + half_size + d);
    }
    for (; d < half_size; ++d) {
      float x = input[d], y = input[d + half_size];
      float cos_x = static_cast<float>(cache.cos[d]);
      float sin_x = static_cast<float>(cache.sin[d]);
      float cos_y = static_cast<float>(cache.cos[d + half_size]);
      float sin_y = static_cast<float>(cache.sin[d + half_size]);
      out[d] = static_cast<scalar_t>(x * cos_x - y * sin_x);
      out[d + half_size] = static_cast<scalar_t>(y * cos_y + x * sin_y);
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
      auto cache = cache_pos(bs * p.seqlen + seq);
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

// Apply multidimensional RoPE to a single tensor x.
// x: [num_tokens, num_heads, head_dim], cos/sin: [num_tokens, head_dim]
// Splits head_dim into ndim=2 chunks and applies standard rotary to each independently.
// cos/sin layout per chunk (chunk_size elements): [cos_half0, cos_half1] matching
// rotate_half pattern: out_x = x_first * cos_first - x_second * sin_first
//                      out_y = x_second * cos_second + x_first * sin_second
template <typename scalar_t, typename param_t>
void apply_multidimensional_rope_kernel_impl(
    scalar_t* __restrict__ x,
    param_t* __restrict__ cos,
    param_t* __restrict__ sin,
    int64_t x_stride_s,
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

  bool flag = (embed_dim % bVecSize == 0);
  int64_t loop_upper = flag ? embed_dim : embed_dim - bVecSize;

  // Vectorized compute loop for a single chunk within a single head.
  // token_head: offset into x for (token, head)
  // cos_ptr/sin_ptr: pointer to the cos/sin data for the chunk
  auto compute_loop = [&](int64_t token_head, param_t* cos_ptr, param_t* sin_ptr) {
    int64_t j = 0;
    for (; j < loop_upper; j += bVecSize) {
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

        float _q_x = x[out_x];
        float _q_y = x[out_y];

        x[out_x] = _q_x * _cos_x - _q_y * _sin_x;
        x[out_y] = _q_y * _cos_y + _q_x * _sin_y;
      }
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
          int64_t token_head = token_idx * x_stride_s + h * head_dim + chunk_offset;
          compute_loop(token_head, cos_ptr, sin_ptr);
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
        cos.data_ptr<param_t>(),
        sin.data_ptr<param_t>(),
        q_stride_s,
        num_heads,
        head_dim,
        num_tokens);
    apply_multidimensional_rope_kernel_impl<scalar_t, param_t>(
        key.data_ptr<scalar_t>(),
        cos.data_ptr<param_t>(),
        sin.data_ptr<param_t>(),
        k_stride_s,
        num_heads,
        head_dim,
        num_tokens);
  });
  return std::make_tuple(query, key);
}

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
  TORCH_CHECK(input_dim >= 2 && input_dim <= 4, "Query/Key must be 2D/3D/4D, got ", input_dim, "D.");

  CHECK_DIM(2, cos_sin_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
  TORCH_CHECK(positions.scalar_type() == at::kLong, "expect positions to be int64, got ", positions.scalar_type());
  TORCH_CHECK(input_dtype == key.scalar_type(), "query and key must have the same data type");
  TORCH_CHECK(input_dtype == cos_sin_cache.scalar_type(), "query and cos_sin_cache must have the same data type");

  int64_t rotary_dim = cos_sin_cache.size(1);
  const RopeParams p{query, key, head_size, rotary_dim};
  TORCH_CHECK(p.rotary_dim <= p.head_size, "rotary_dim must be <= head_size");
  TORCH_CHECK(p.rotary_dim % 2 == 0, "rotary_dim must be even");
  TORCH_CHECK(positions.numel() == p.rows(), "positions.numel() must equal batch * seqlen");

  if (input_dim <= 3) {
    CHECK_EQ(key.size(0), query.size(0));
  }
  if (input_dim == 2) {
    CHECK_EQ(query.size(1), p.num_heads * p.head_size);
    CHECK_EQ(key.size(1), p.num_heads_kv * p.head_size);
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

  at::Tensor query_out = (input_dim != 3) ? query : at::empty(query.sizes(), query.options());
  at::Tensor key_out = (input_dim != 3) ? key : at::empty(key.sizes(), key.options());
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
  CHECK_DIM(3, query);
  const auto input_dtype = query.scalar_type();
  int64_t num_tokens = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t head_size = query.size(2);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_INPUT_SHAPE_DTYPE<true>(key, {num_tokens, num_heads, head_size}, input_dtype);
  CHECK_INPUT_SHAPE_DTYPE<false>(cos, {num_tokens, head_size}, cos.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<false>(sin, {num_tokens, head_size}, sin.scalar_type());
  CHECK_EQ(cos.scalar_type(), sin.scalar_type());
  TORCH_CHECK(head_size % 2 == 0, "head_size must be even");

  const RopeParams p{query, key, head_size, head_size};
  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(input_dtype, cos.scalar_type(), [&] {
    scalar_t* q_ptr = query.data_ptr<scalar_t>();
    scalar_t* k_ptr = key.data_ptr<scalar_t>();
    const param_t* cos_ptr = cos.data_ptr<param_t>();
    const param_t* sin_ptr = sin.data_ptr<param_t>();
    auto cache_pos = [cos_ptr, sin_ptr, head_size](int64_t token) -> SplitCosSinRow<param_t> {
      return {cos_ptr + token * head_size, sin_ptr + token * head_size};
    };
    rotary_embedding_kernel_impl<scalar_t, RotaryMode::NeoxFull, true>(q_ptr, k_ptr, q_ptr, k_ptr, p, cache_pos);
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
  CHECK_EQ(positions.scalar_type(), at::kLong);
  CHECK_DIM(2, query);

  const auto input_dtype = query.scalar_type();
  int64_t rotary_dim = cos_sin_cache.size(1);
  int64_t num_tokens = positions.size(-1);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
  CHECK_EQ(query.size(0), num_tokens);
  CHECK_EQ(key.size(0), num_tokens);
  CHECK_EQ(query.size(-1) % head_size, 0);
  CHECK_EQ(key.size(-1) % head_size, 0);
  CHECK_EQ(input_dtype, key.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<false>(cos_sin_cache, {cos_sin_cache.size(0), rotary_dim}, input_dtype);

  const RopeParams p{query, key, head_size, rotary_dim};
  TORCH_CHECK(p.rotary_dim <= p.head_size, "rotary_dim must be <= head_size");
  TORCH_CHECK(p.rotary_dim % 2 == 0, "rotary_dim must be even");
  TORCH_CHECK(positions.size(-1) == p.rows(), "positions.size(-1) must equal batch * seqlen");

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input_dtype, "multimodal_rotary_embedding_cpu", [&] {
    const scalar_t* cache_base = cos_sin_cache.data_ptr<scalar_t>();
    const int64_t* pos_ptr = positions.data_ptr<int64_t>();
    scalar_t* q_ptr = query.data_ptr<scalar_t>();
    scalar_t* k_ptr = key.data_ptr<scalar_t>();

    if (positions.dim() == 2) {
      TORCH_CHECK(mrope_section.has_value(), "mrope_section must be provided when positions is 2D");
      auto mrope_section_val = mrope_section.value();
      CHECK_EQ(mrope_section_val.size(), 3);
      CHECK_EQ(positions.size(0), 3);
      const int64_t section_t = mrope_section_val[0];
      const int64_t section_h = mrope_section_val[1];
      const int64_t section_w = mrope_section_val[2];
      const int64_t p_stride0 = positions.stride(0);
      auto cache_pos = [=](int64_t token) -> MropeCosSinRow<scalar_t> {
        return {
            cache_base + pos_ptr[0 * p_stride0 + token] * rotary_dim,
            cache_base + pos_ptr[1 * p_stride0 + token] * rotary_dim,
            cache_base + pos_ptr[2 * p_stride0 + token] * rotary_dim,
            section_t,
            section_h,
            section_w,
            mrope_interleaved};
      };
      if (is_neox) {
        rotary_embedding_kernel_impl<scalar_t, RotaryMode::Neox, true>(q_ptr, k_ptr, q_ptr, k_ptr, p, cache_pos);
      } else {
        rotary_embedding_kernel_impl<scalar_t, RotaryMode::Interleaved, true>(q_ptr, k_ptr, q_ptr, k_ptr, p, cache_pos);
      }
    } else {  // positions.dim() == 1
      auto cache_pos = [cache_base, pos_ptr, rotary_dim](int64_t token) -> const scalar_t* {
        return cache_base + pos_ptr[token] * rotary_dim;
      };

      if (is_neox) {
        rotary_embedding_kernel_impl<scalar_t, RotaryMode::Neox, true>(q_ptr, k_ptr, q_ptr, k_ptr, p, cache_pos);
      } else {
        rotary_embedding_kernel_impl<scalar_t, RotaryMode::Interleaved, true>(q_ptr, k_ptr, q_ptr, k_ptr, p, cache_pos);
      }
    }
  });
  return std::make_tuple(query, key);
}
