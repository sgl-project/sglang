#include <type_traits>

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

// In-place rotary embedding on interleaved real/imag pairs.
// Supports [T, rope_dim] and [T, nh, rope_dim] with explicit outer strides.
namespace {

#ifdef CPU_CAPABILITY_AVX512
static inline __m512 rotary_emb_sign_mask(bool inverse) {
  // Forward negates even lanes of the sin broadcast; inverse negates odd lanes.
  return _mm512_castsi512_ps(
      inverse ? _mm512_set_epi32(
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0)
              : _mm512_set_epi32(
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000,
                    0,
                    (int)0x80000000));
}
#endif

template <typename scalar_t>
static inline void apply_rotary_emb_row(
    scalar_t* __restrict__ xp,
    const float* __restrict__ fp,
    int64_t rope_dim,
    bool inverse
#ifdef CPU_CAPABILITY_AVX512
    ,
    const __m512 avx_sign
#endif
) {
#ifdef CPU_CAPABILITY_AVX512
  {
    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int64_t kVecSize = bVec::size();
    constexpr int64_t kFVecSize = fVec::size();

    int64_t k = 0;
    // double is dispatched but handled by the scalar path below.
    if constexpr (!std::is_same_v<scalar_t, double>) {
      for (; k <= rope_dim - kVecSize; k += kVecSize) {
        if constexpr (std::is_same_v<scalar_t, float>) {
          const __m512 xv0 = _mm512_loadu_ps(xp + k);
          const __m512 fv0 = _mm512_loadu_ps(fp + k);
          // 0xA0 broadcasts even lanes within each complex pair, 0xF5 broadcasts
          // odd lanes, and 0xB1 swaps real/imag lanes within each pair.
          const __m512 out0 = _mm512_fmadd_ps(
              xv0,
              _mm512_permute_ps(fv0, 0xA0),
              _mm512_mul_ps(_mm512_permute_ps(xv0, 0xB1), _mm512_xor_ps(_mm512_permute_ps(fv0, 0xF5), avx_sign)));
          _mm512_storeu_ps(xp + k, out0);
        } else {
          fVec x0_v, x1_v;
          std::tie(x0_v, x1_v) = at::vec::convert_to_float(bVec::loadu(xp + k));
          const __m512 xv0 = x0_v;
          const __m512 xv1 = x1_v;

          const __m512 fv0 = _mm512_loadu_ps(fp + k);
          const __m512 fv1 = _mm512_loadu_ps(fp + k + kFVecSize);

          const __m512 out0 = _mm512_fmadd_ps(
              xv0,
              _mm512_permute_ps(fv0, 0xA0),
              _mm512_mul_ps(_mm512_permute_ps(xv0, 0xB1), _mm512_xor_ps(_mm512_permute_ps(fv0, 0xF5), avx_sign)));
          const __m512 out1 = _mm512_fmadd_ps(
              xv1,
              _mm512_permute_ps(fv1, 0xA0),
              _mm512_mul_ps(_mm512_permute_ps(xv1, 0xB1), _mm512_xor_ps(_mm512_permute_ps(fv1, 0xF5), avx_sign)));

          at::vec::convert_from_float<scalar_t>(fVec(out0), fVec(out1)).store(xp + k);
        }
      }
    }  // if constexpr (!double)

    // Scalar tail.
    for (; k < rope_dim; k += 2) {
      const float xr = static_cast<float>(xp[k]);
      const float xi = static_cast<float>(xp[k + 1]);
      const float cr = fp[k], ci = fp[k + 1];
      if (inverse) {
        xp[k] = static_cast<scalar_t>(xr * cr + xi * ci);
        xp[k + 1] = static_cast<scalar_t>(xi * cr - xr * ci);
      } else {
        xp[k] = static_cast<scalar_t>(xr * cr - xi * ci);
        xp[k + 1] = static_cast<scalar_t>(xr * ci + xi * cr);
      }
    }
  }
#else
  // Scalar fallback.
  for (int64_t k = 0; k < rope_dim; k += 2) {
    const float xr = static_cast<float>(xp[k]);
    const float xi = static_cast<float>(xp[k + 1]);
    const float cr = fp[k], ci = fp[k + 1];
    if (inverse) {
      xp[k] = static_cast<scalar_t>(xr * cr + xi * ci);
      xp[k + 1] = static_cast<scalar_t>(xi * cr - xr * ci);
    } else {
      xp[k] = static_cast<scalar_t>(xr * cr - xi * ci);
      xp[k + 1] = static_cast<scalar_t>(xr * ci + xi * cr);
    }
  }
#endif
}

template <typename scalar_t, typename index_t>
static void apply_rotary_emb_impl(
    scalar_t* __restrict__ q,
    scalar_t* __restrict__ k,
    const float* __restrict__ freqs,
    const index_t* __restrict__ positions,
    int64_t T,
    int64_t q_nh,
    int64_t k_nh,
    int64_t rope_dim,
    int64_t q_stride_t,
    int64_t q_stride_h,
    int64_t k_stride_t,
    int64_t k_stride_h,
    int64_t freqs_stride_t,
    int64_t positions_stride,
    bool inverse) {
  // Interleaved complex rotation:
  //   x = [xr0, xi0, xr1, xi1, ...]
  //   f = [c0,  s0,  c1,  s1,  ...]
  // AVX-512 uses permutes to broadcast cos/sin within each pair and a sign mask
  // to switch between forward and inverse rotation without branching.

#ifdef CPU_CAPABILITY_AVX512
  const __m512 avx_sign = rotary_emb_sign_mask(inverse);
#endif

  constexpr int64_t kRopeBlock = 256;
  const int64_t max_nh = std::max<int64_t>(q_nh, k == nullptr ? 0 : k_nh);
  const int64_t outer_rows = T * max_nh;
  const int64_t grain = std::max<int64_t>(1, GRAIN_SIZE / rope_dim);

  at::parallel_for(0, outer_rows, grain, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t t = i / max_nh;
      const int64_t h = i % max_nh;
      const int64_t freq_t = positions == nullptr ? t : static_cast<int64_t>(positions[t * positions_stride]);
      const float* fp = freqs + freq_t * freqs_stride_t;
      const bool do_q = h < q_nh;
      const bool do_k = k != nullptr && h < k_nh;

      if (!do_q && !do_k) {
        continue;
      }

      scalar_t* qx = do_q ? q + t * q_stride_t + h * q_stride_h : nullptr;
      scalar_t* kx = do_k ? k + t * k_stride_t + h * k_stride_h : nullptr;

      for (int64_t d0 = 0; d0 < rope_dim; d0 += kRopeBlock) {
        const int64_t dlen = std::min<int64_t>(kRopeBlock, rope_dim - d0);

        if (do_q) {
          apply_rotary_emb_row<scalar_t>(
              qx + d0,
              fp + d0,
              dlen,
              inverse
#ifdef CPU_CAPABILITY_AVX512
              ,
              avx_sign
#endif
          );
        }

        if (do_k) {
          apply_rotary_emb_row<scalar_t>(
              kx + d0,
              fp + d0,
              dlen,
              inverse
#ifdef CPU_CAPABILITY_AVX512
              ,
              avx_sign
#endif
          );
        }
      }
    }
  });
}

}  // namespace

at::Tensor apply_rotary_emb_interleaved_cpu(
    at::Tensor& x,
    at::Tensor& freqs,
    bool inverse,
    const std::optional<at::Tensor>& positions,
    const std::optional<at::Tensor>& k_opt) {
  TORCH_CHECK(
      x.dim() == 2 || x.dim() == 3,
      "apply_rotary_emb_interleaved_cpu: x must be 2D [T,rope_dim] or 3D [T,nh,rope_dim]");
  TORCH_CHECK(
      x.device().is_cpu() && freqs.device().is_cpu(), "apply_rotary_emb_interleaved_cpu: all tensors must be on CPU");

  const int64_t T = x.size(0);
  const int64_t rope_dim = x.size(-1);
  const int64_t nh = x.dim() == 3 ? x.size(1) : 1;
  const bool has_k = k_opt.has_value();

  at::Tensor freqs_real;
  if (freqs.is_complex()) {
    TORCH_CHECK(freqs.dim() == 2, "apply_rotary_emb_interleaved_cpu: freqs_cis must be 2D [N, rope_dim/2]");
    TORCH_CHECK(
        freqs.scalar_type() == at::kComplexFloat, "apply_rotary_emb_interleaved_cpu: complex freqs must be complex64");
    freqs_real = at::view_as_real(freqs).flatten(-2);
  } else {
    TORCH_CHECK(
        freqs.dim() == 2 && freqs.scalar_type() == at::kFloat,
        "apply_rotary_emb_interleaved_cpu: freqs must be float32 [N, rope_dim] or complex64 [N, rope_dim/2]");
    freqs_real = freqs;
  }

  TORCH_CHECK(rope_dim % 2 == 0, "apply_rotary_emb_interleaved_cpu: rope_dim must be even");

  TORCH_CHECK(
      freqs_real.size(1) == rope_dim, "apply_rotary_emb_interleaved_cpu: frequency rope dim must match x last dim");

  const bool has_positions = positions.has_value();
  if (has_positions) {
    const at::Tensor& pos = positions.value();
    TORCH_CHECK(pos.device().is_cpu(), "apply_rotary_emb_interleaved_cpu: positions must be on CPU");
    TORCH_CHECK(pos.dim() == 1, "apply_rotary_emb_interleaved_cpu: positions must be 1D [T]");
    TORCH_CHECK(pos.size(0) == T, "apply_rotary_emb_interleaved_cpu: positions must have shape [T]");
    TORCH_CHECK(
        pos.scalar_type() == at::kLong || pos.scalar_type() == at::kInt,
        "apply_rotary_emb_interleaved_cpu: positions must be int64 or int32");
  } else {
    TORCH_CHECK(freqs_real.size(0) == T, "apply_rotary_emb_interleaved_cpu: freqs must have shape [T, rope_dim]");
  }

  if (freqs_real.stride(-1) != 1) {
    freqs_real = freqs_real.contiguous();
  }

  // The rope dimension must be contiguous. Outer strides are handled explicitly,
  // so non-contiguous [T] or [head] slices are still supported.
  TORCH_CHECK(
      x.stride(-1) == 1,
      "apply_rotary_emb_interleaved_cpu: x inner (rope) dim must be contiguous (stride[-1]==1); "
      "got stride[-1]=",
      x.stride(-1));

  int64_t k_nh = 0;
  int64_t k_stride_t = 0;
  int64_t k_stride_h = 0;
  if (has_k) {
    const at::Tensor& k = k_opt.value();
    TORCH_CHECK(k.device().is_cpu(), "apply_rotary_emb_interleaved_cpu: k must be on CPU");
    TORCH_CHECK(k.dim() == 2 || k.dim() == 3, "apply_rotary_emb_interleaved_cpu: k must be 2D or 3D");
    TORCH_CHECK(k.scalar_type() == x.scalar_type(), "apply_rotary_emb_interleaved_cpu: x and k must have same dtype");
    TORCH_CHECK(k.size(0) == T, "apply_rotary_emb_interleaved_cpu: k must have same T as x");
    TORCH_CHECK(k.size(-1) == rope_dim, "apply_rotary_emb_interleaved_cpu: k rope dim must match x");
    TORCH_CHECK(k.stride(-1) == 1, "apply_rotary_emb_interleaved_cpu: k inner (rope) dim must be contiguous");
    k_nh = k.dim() == 3 ? k.size(1) : 1;
    k_stride_t = k.stride(0);
    k_stride_h = k.dim() == 3 ? k.stride(1) : 0;
  }

  const int64_t x_stride_t = x.stride(0);
  const int64_t x_stride_h = x.dim() == 3 ? x.stride(1) : 0;
  const int64_t freqs_stride_t = freqs_real.stride(0);
  const int64_t positions_stride = has_positions ? positions.value().stride(0) : 0;
  const bool pos_i64 = has_positions && positions.value().scalar_type() == at::kLong;

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, x.scalar_type(), "apply_rotary_emb_interleaved_cpu", [&] {
    scalar_t* k_ptr = nullptr;
    if (has_k) {
      k_ptr = k_opt.value().data_ptr<scalar_t>();
    }

    auto run_with_positions = [&](const auto* positions_ptr) {
      using index_t = typename std::remove_cv<typename std::remove_pointer<decltype(positions_ptr)>::type>::type;
      apply_rotary_emb_impl<scalar_t, index_t>(
          x.data_ptr<scalar_t>(),
          k_ptr,
          freqs_real.data_ptr<float>(),
          positions_ptr,
          T,
          nh,
          k_nh,
          rope_dim,
          x_stride_t,
          x_stride_h,
          k_stride_t,
          k_stride_h,
          freqs_stride_t,
          positions_stride,
          inverse);
    };

    if (!has_positions || pos_i64) {
      run_with_positions(has_positions ? positions.value().data_ptr<int64_t>() : nullptr);
    } else {
      run_with_positions(positions.value().data_ptr<int32_t>());
    }
  });
  return x;
}
