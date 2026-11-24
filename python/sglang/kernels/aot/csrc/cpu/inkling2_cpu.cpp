#include <ATen/Parallel.h>

#include <cmath>
#include <vector>

#include "common.h"

namespace {

template <typename scalar_t>
inline float read_scalar_as_float(const scalar_t* p) {
  return static_cast<float>(*p);
}

template <typename scalar_t>
inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

inline float read_any_as_float(const void* ptr, at::ScalarType st, int64_t idx) {
  switch (st) {
    case at::kFloat:
      return static_cast<const float*>(ptr)[idx];
    case at::kBFloat16:
      return static_cast<float>(static_cast<const at::BFloat16*>(ptr)[idx]);
    case at::kHalf:
      return static_cast<float>(static_cast<const at::Half*>(ptr)[idx]);
    default:
      TORCH_CHECK(false, "Unsupported scalar type for Inkling2 CPU op: ", st);
  }
}

inline void write_any_from_float(void* ptr, at::ScalarType st, int64_t idx, float val) {
  switch (st) {
    case at::kFloat:
      static_cast<float*>(ptr)[idx] = val;
      return;
    case at::kBFloat16:
      static_cast<at::BFloat16*>(ptr)[idx] = static_cast<at::BFloat16>(val);
      return;
    case at::kHalf:
      static_cast<at::Half*>(ptr)[idx] = static_cast<at::Half>(val);
      return;
    default:
      TORCH_CHECK(false, "Unsupported scalar type for Inkling2 CPU op: ", st);
  }
}

template <typename scalar_t>
void inkling2_short_convolution_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    at::Tensor& cache,
    const int64_t* safe_idx_ptr,
    const uint8_t* cache_mask_ptr,
    const int64_t* cu_ptr,
    const int64_t* si_ptr,
    const std::optional<at::Tensor>& dynamic_weights,
    bool decode,
    bool use_residual,
    const uint8_t* track_mask_ptr,
    const int64_t* track_indices_ptr,
    const std::optional<at::Tensor>& window,
    at::Tensor& out) {
  const int64_t T = x.size(0);
  const int64_t D = x.size(1);
  const bool dynamic = dynamic_weights.has_value();
  const int64_t width = dynamic ? weight.size(0) : weight.size(1);
  const int64_t n_heads = dynamic ? dynamic_weights->size(1) : 1;
  const int64_t head_dim = dynamic ? (D / n_heads) : D;
  const float scale =
      static_cast<float>(std::pow(static_cast<double>(width * (dynamic ? 2 : 1) + (use_residual ? 1 : 0)), -0.5));

  at::Tensor weight_f;
  at::Tensor dyn_f;
  at::Tensor weight_static;
  const float* weight_f_ptr = nullptr;
  const float* dyn_f_ptr = nullptr;
  const scalar_t* weight_s_ptr = nullptr;
  int64_t w_s0 = 0, w_s1 = 0;
  if (dynamic) {
    weight_f = weight.to(at::kFloat).contiguous();
    dyn_f = dynamic_weights->to(at::kFloat).contiguous();
    weight_f_ptr = weight_f.data_ptr<float>();
    dyn_f_ptr = dyn_f.data_ptr<float>();
  } else {
    weight_static = weight.to(x.scalar_type()).contiguous();
    weight_s_ptr = weight_static.data_ptr<scalar_t>();
    w_s0 = weight_static.stride(0);
    w_s1 = weight_static.stride(1);
  }

  const scalar_t* x_ptr = x.data_ptr<scalar_t>();
  const int64_t x_s0 = x.stride(0);
  scalar_t* out_ptr = out.data_ptr<scalar_t>();
  const int64_t out_s0 = out.stride(0);

  scalar_t* cache_ptr = cache.data_ptr<scalar_t>();
  const int64_t c_s0 = cache.stride(0);
  const int64_t c_s1 = cache.stride(1);
  const int64_t c_s2 = cache.stride(2);

  const bool has_window = window.has_value();
  void* win_ptr = has_window ? window->data_ptr() : nullptr;
  const at::ScalarType win_st = has_window ? window->scalar_type() : x.scalar_type();
  const int64_t win_s0 = has_window ? window->stride(0) : 0;
  const int64_t win_s1 = has_window ? window->stride(1) : 0;
  const int64_t win_s2 = has_window ? window->stride(2) : 0;
  const int64_t win_s3 = has_window ? window->stride(3) : 0;
  const bool has_track = (track_mask_ptr != nullptr && track_indices_ptr != nullptr);

  constexpr int64_t COL_CHUNK = 64;
  const int64_t num_col_chunks = (D + COL_CHUNK - 1) / COL_CHUNK;

  at::parallel_for(0, T * num_col_chunks, 1, [&](int64_t begin, int64_t end) {
    float acc[COL_CHUNK];
    std::vector<float> hist_buf;
    if (decode && width > 1) {
      hist_buf.resize(static_cast<size_t>((width - 1) * COL_CHUNK));
    }

    for (int64_t task = begin; task < end; ++task) {
      const int64_t row = task / num_col_chunks;
      const int64_t c0 = (task % num_col_chunks) * COL_CHUNK;
      const int64_t chunk_len = std::min(COL_CHUNK, D - c0);

      const int64_t seq = decode ? row : si_ptr[row];
      const int64_t bos = decode ? row : cu_ptr[seq];
      const int64_t slot = safe_idx_ptr[seq];
      const bool in_pool = (cache_mask_ptr[seq] != 0);
      const bool has_cache = in_pool && (slot > 0);
      const bool do_track = decode && has_track && has_cache && (track_mask_ptr[seq] != 0);
      const int64_t track_slot = do_track ? track_indices_ptr[seq] : 0;

      for (int64_t j = 0; j < chunk_len; ++j) {
        acc[j] = 0.0f;
      }

      for (int64_t tap = 0; tap < width; ++tap) {
        const int64_t lag = dynamic ? tap : (width - 1 - tap);
        const int64_t src = row - lag;
        const bool from_x = (src >= bos);
        const int64_t cache_pos = src - bos + (width - 1);
        const bool save_win = has_window && (lag < width - 1) && in_pool;
        const int64_t win_base = save_win ? (seq * win_s0 + (row - bos) * win_s1 + (width - 2 - lag) * win_s2) : 0;

        for (int64_t j = 0; j < chunk_len; ++j) {
          const int64_t col = c0 + j;
          float val = 0.0f;
          float raw_prefix = 0.0f;
          if (from_x) {
            val = static_cast<float>(x_ptr[src * x_s0 + col]);
          } else {
            raw_prefix = in_pool ? static_cast<float>(cache_ptr[slot * c_s0 + cache_pos * c_s1 + col * c_s2]) : 0.0f;
            val = has_cache ? raw_prefix : 0.0f;
            if (decode && width > 1) {
              hist_buf[cache_pos * COL_CHUNK + j] = raw_prefix;
            }
          }

          if (save_win) {
            const float w_val_save = from_x ? val : raw_prefix;
            write_any_from_float(win_ptr, win_st, win_base + col * win_s3, w_val_save);
          }

          float w_val = 0.0f;
          if (dynamic) {
            const int64_t head = col / head_dim;
            w_val = weight_f_ptr[lag * D + col] + dyn_f_ptr[(row * n_heads + head) * width + lag];
          } else {
            w_val = static_cast<float>(weight_s_ptr[col * w_s0 + tap * w_s1]);
          }
          acc[j] += val * w_val;
        }
      }

      for (int64_t j = 0; j < chunk_len; ++j) {
        const int64_t col = c0 + j;
        const float cur = static_cast<float>(x_ptr[row * x_s0 + col]);
        const float total = use_residual ? (acc[j] + cur) : acc[j];
        out_ptr[row * out_s0 + col] = static_cast<scalar_t>(total * scale);
      }

      if (decode && has_cache && width > 1) {
        for (int64_t tap = 0; tap < width - 1; ++tap) {
          for (int64_t j = 0; j < chunk_len; ++j) {
            const int64_t col = c0 + j;
            const float val =
                (tap == width - 2) ? static_cast<float>(x_ptr[row * x_s0 + col]) : hist_buf[(tap + 1) * COL_CHUNK + j];
            const scalar_t c_val = static_cast<scalar_t>(val);
            cache_ptr[slot * c_s0 + tap * c_s1 + col * c_s2] = c_val;
            if (do_track) {
              cache_ptr[track_slot * c_s0 + tap * c_s1 + col * c_s2] = c_val;
            }
          }
        }
      }
    }
  });
}

template <typename down_t>
void inkling2_routed_sconv_epilogue_impl(
    const at::Tensor& x_src,
    const at::Tensor& weight,
    at::Tensor& cache,
    const int64_t* safe_idx_ptr,
    const uint8_t* cache_mask_ptr,
    const int64_t* cu_ptr,
    const int64_t* si_ptr,
    const at::Tensor& attention,
    const std::optional<at::Tensor>& partial,
    const uint8_t* track_mask_ptr,
    const int64_t* track_indices_ptr,
    const std::optional<at::Tensor>& window,
    at::Tensor& out) {
  const int64_t rows = attention.size(0);
  const int64_t D = attention.size(1);
  const int64_t width = weight.size(1);
  const bool decode = !window.has_value();
  const float scale = static_cast<float>(std::pow(static_cast<double>(width + 1), -0.5));

  at::Tensor weight_cast = weight.to(x_src.scalar_type()).contiguous();
  const down_t* weight_ptr = weight_cast.data_ptr<down_t>();
  const int64_t w_s0 = weight_cast.stride(0);
  const int64_t w_s1 = weight_cast.stride(1);

  const down_t* x_ptr = x_src.data_ptr<down_t>();
  const int64_t x_s0 = x_src.stride(0);

  down_t* cache_ptr = cache.data_ptr<down_t>();
  const int64_t c_s0 = cache.stride(0);
  const int64_t c_s1 = cache.stride(1);
  const int64_t c_s2 = cache.stride(2);

  at::Tensor attn_c = attention.contiguous();
  const void* attn_ptr = attn_c.data_ptr();
  const at::ScalarType attn_st = attn_c.scalar_type();

  const bool has_partial = partial.has_value();
  at::Tensor partial_c;
  const void* partial_ptr = nullptr;
  at::ScalarType partial_st = attn_st;
  if (has_partial) {
    partial_c = partial->contiguous();
    partial_ptr = partial_c.data_ptr();
    partial_st = partial_c.scalar_type();
  }

  void* out_ptr = out.data_ptr();
  const at::ScalarType out_st = out.scalar_type();

  const bool has_window = window.has_value();
  void* win_ptr = has_window ? window->data_ptr() : nullptr;
  const at::ScalarType win_st = has_window ? window->scalar_type() : out_st;
  const int64_t win_s0 = has_window ? window->stride(0) : 0;
  const int64_t win_s1 = has_window ? window->stride(1) : 0;
  const int64_t win_s2 = has_window ? window->stride(2) : 0;
  const int64_t win_s3 = has_window ? window->stride(3) : 0;
  const bool has_track = (track_mask_ptr != nullptr && track_indices_ptr != nullptr);

  constexpr int64_t COL_CHUNK = 64;
  const int64_t num_col_chunks = (D + COL_CHUNK - 1) / COL_CHUNK;

  at::parallel_for(0, rows * num_col_chunks, 1, [&](int64_t begin, int64_t end) {
    float acc[COL_CHUNK];
    std::vector<float> hist_buf;
    if (decode && width > 1) {
      hist_buf.resize(static_cast<size_t>((width - 1) * COL_CHUNK));
    }

    for (int64_t task = begin; task < end; ++task) {
      const int64_t row = task / num_col_chunks;
      const int64_t c0 = (task % num_col_chunks) * COL_CHUNK;
      const int64_t chunk_len = std::min(COL_CHUNK, D - c0);

      const int64_t seq = decode ? row : si_ptr[row];
      const int64_t bos = decode ? row : cu_ptr[seq];
      const int64_t slot = safe_idx_ptr[seq];
      const bool in_pool = (cache_mask_ptr[seq] != 0);
      const bool has_cache = in_pool && (slot > 0);
      const bool do_track = decode && has_track && has_cache && (track_mask_ptr[seq] != 0);
      const int64_t track_slot = do_track ? track_indices_ptr[seq] : 0;

      for (int64_t j = 0; j < chunk_len; ++j) {
        acc[j] = 0.0f;
      }

      for (int64_t tap = 0; tap < width; ++tap) {
        const int64_t lag = width - 1 - tap;
        const int64_t src = row - lag;
        const bool from_x = (src >= bos);
        const int64_t cache_pos = src - bos + (width - 1);
        const bool save_win = has_window && (lag < width - 1) && in_pool;
        const int64_t win_base = save_win ? (seq * win_s0 + (row - bos) * win_s1 + (width - 2 - lag) * win_s2) : 0;

        for (int64_t j = 0; j < chunk_len; ++j) {
          const int64_t col = c0 + j;
          float val = 0.0f;
          float raw_prefix = 0.0f;
          if (from_x) {
            val = static_cast<float>(x_ptr[src * x_s0 + col]);
          } else {
            raw_prefix = in_pool ? static_cast<float>(cache_ptr[slot * c_s0 + cache_pos * c_s1 + col * c_s2]) : 0.0f;
            val = has_cache ? raw_prefix : 0.0f;
            if (decode && width > 1) {
              hist_buf[cache_pos * COL_CHUNK + j] = raw_prefix;
            }
          }

          if (save_win) {
            const float w_val_save = from_x ? val : raw_prefix;
            write_any_from_float(win_ptr, win_st, win_base + col * win_s3, w_val_save);
          }

          const float w_val = static_cast<float>(weight_ptr[col * w_s0 + tap * w_s1]);
          acc[j] += val * w_val;
        }
      }

      for (int64_t j = 0; j < chunk_len; ++j) {
        const int64_t col = c0 + j;
        const float cur = static_cast<float>(x_ptr[row * x_s0 + col]);
        const float y_f = (acc[j] + cur) * scale;
        const float y = static_cast<float>(static_cast<down_t>(y_f));
        const int64_t off = row * D + col;
        const float attn_v = read_any_as_float(attn_ptr, attn_st, off);
        const float total =
            has_partial ? ((read_any_as_float(partial_ptr, partial_st, off) + attn_v) + y) : (attn_v + y);
        write_any_from_float(out_ptr, out_st, off, total);
      }

      if (decode && has_cache && width > 1) {
        for (int64_t tap = 0; tap < width - 1; ++tap) {
          for (int64_t j = 0; j < chunk_len; ++j) {
            const int64_t col = c0 + j;
            const float val =
                (tap == width - 2) ? static_cast<float>(x_ptr[row * x_s0 + col]) : hist_buf[(tap + 1) * COL_CHUNK + j];
            const down_t c_val = static_cast<down_t>(val);
            cache_ptr[slot * c_s0 + tap * c_s1 + col * c_s2] = c_val;
            if (do_track) {
              cache_ptr[track_slot * c_s0 + tap * c_s1 + col * c_s2] = c_val;
            }
          }
        }
      }
    }
  });
}

}  // namespace

at::Tensor inkling2_short_convolution_cpu(
    const at::Tensor& x,
    const at::Tensor& weight,
    at::Tensor& cache,
    const at::Tensor& safe_idx,
    const at::Tensor& cache_mask,
    const at::Tensor& cu,
    const at::Tensor& si,
    const std::optional<at::Tensor>& dynamic_weights,
    bool decode,
    bool use_residual,
    const std::optional<at::Tensor>& track_mask,
    const std::optional<at::Tensor>& track_indices,
    const std::optional<at::Tensor>& window) {
  TORCH_CHECK(x.dim() == 2 && x.stride(-1) == 1, "x must be 2D [T, D] with last dim contiguous");
  if (x.size(0) == 0) {
    return at::empty_like(x, at::MemoryFormat::Contiguous);
  }
  const int64_t D = x.size(1);
  const bool dynamic = dynamic_weights.has_value();
  if (dynamic) {
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == D, "dynamic weight must be [width, D]");
    TORCH_CHECK(
        dynamic_weights->dim() == 3 && dynamic_weights->size(0) == x.size(0) &&
            dynamic_weights->size(2) == weight.size(0),
        "dynamic_weights must be [T, n_heads, width]");
    TORCH_CHECK(D % dynamic_weights->size(1) == 0, "D must be divisible by n_heads");
  } else {
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == D, "static weight must be [D, width]");
  }
  const int64_t width = dynamic ? weight.size(0) : weight.size(1);
  TORCH_CHECK(
      cache.dim() == 3 && cache.size(1) >= width - 1 && cache.size(2) == D, "cache must be [num_slots, >=width-1, D]");
  TORCH_CHECK(cache.scalar_type() == x.scalar_type(), "cache dtype must match x dtype");

  at::Tensor out = at::empty({x.size(0), D}, x.options().memory_format(at::MemoryFormat::Contiguous));
  auto safe_idx_l = safe_idx.to(at::kLong).contiguous();
  auto cache_mask_u = cache_mask.to(at::kByte).contiguous();
  auto cu_l = cu.to(at::kLong).contiguous();
  auto si_l = si.to(at::kLong).contiguous();

  at::Tensor track_mask_u;
  at::Tensor track_indices_l;
  const uint8_t* track_mask_ptr = nullptr;
  const int64_t* track_indices_ptr = nullptr;
  if (track_mask.has_value() && track_indices.has_value()) {
    track_mask_u = track_mask->to(at::kByte).contiguous();
    track_indices_l = track_indices->to(at::kLong).contiguous();
    track_mask_ptr = track_mask_u.data_ptr<uint8_t>();
    track_indices_ptr = track_indices_l.data_ptr<int64_t>();
  }

  switch (x.scalar_type()) {
    case at::kFloat:
      inkling2_short_convolution_impl<float>(
          x,
          weight,
          cache,
          safe_idx_l.data_ptr<int64_t>(),
          cache_mask_u.data_ptr<uint8_t>(),
          cu_l.data_ptr<int64_t>(),
          si_l.data_ptr<int64_t>(),
          dynamic_weights,
          decode,
          use_residual,
          track_mask_ptr,
          track_indices_ptr,
          window,
          out);
      break;
    case at::kBFloat16:
      inkling2_short_convolution_impl<at::BFloat16>(
          x,
          weight,
          cache,
          safe_idx_l.data_ptr<int64_t>(),
          cache_mask_u.data_ptr<uint8_t>(),
          cu_l.data_ptr<int64_t>(),
          si_l.data_ptr<int64_t>(),
          dynamic_weights,
          decode,
          use_residual,
          track_mask_ptr,
          track_indices_ptr,
          window,
          out);
      break;
    case at::kHalf:
      inkling2_short_convolution_impl<at::Half>(
          x,
          weight,
          cache,
          safe_idx_l.data_ptr<int64_t>(),
          cache_mask_u.data_ptr<uint8_t>(),
          cu_l.data_ptr<int64_t>(),
          si_l.data_ptr<int64_t>(),
          dynamic_weights,
          decode,
          use_residual,
          track_mask_ptr,
          track_indices_ptr,
          window,
          out);
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for inkling2_short_convolution_cpu: ", x.scalar_type());
  }
  return out;
}

at::Tensor inkling2_routed_sconv_epilogue_cpu(
    const at::Tensor& down,
    const std::optional<at::Tensor>& src2dst,
    int64_t top_k,
    const at::Tensor& weight,
    at::Tensor& cache,
    const at::Tensor& safe_idx,
    const at::Tensor& cache_mask,
    const at::Tensor& cu,
    const at::Tensor& si,
    const at::Tensor& attention,
    const std::optional<at::Tensor>& partial,
    const std::optional<at::Tensor>& track_mask,
    const std::optional<at::Tensor>& track_indices,
    const std::optional<at::Tensor>& window,
    bool round_bf16) {
  TORCH_CHECK(attention.dim() == 2, "attention must be 2D [T, D]");
  const at::Tensor& out_proto = partial.has_value() ? *partial : attention;
  at::Tensor out = at::empty_like(out_proto, at::MemoryFormat::Contiguous);
  const int64_t rows = attention.size(0);
  const int64_t D = attention.size(1);
  if (rows == 0) {
    return out;
  }
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == D, "weight must be [D, W]");
  const int64_t width = weight.size(1);
  TORCH_CHECK(
      cache.dim() == 3 && cache.size(1) >= width - 1 && cache.size(2) == D, "cache must be [num_slots, >=width-1, D]");

  at::Tensor x_comb;
  if (top_k > 0) {
    TORCH_CHECK(src2dst.has_value(), "src2dst required when top_k > 0");
    auto s2d_l = src2dst->to(at::kLong).contiguous();
    const int64_t* s2d_ptr = s2d_l.data_ptr<int64_t>();
    at::Tensor down_c = down.contiguous();
    const void* down_ptr = down_c.data_ptr();
    const at::ScalarType down_st = down_c.scalar_type();
    x_comb = at::empty({rows, D}, down.options().memory_format(at::MemoryFormat::Contiguous));
    void* xc_ptr = x_comb.data_ptr();

    constexpr int64_t COL_CHUNK = 128;
    const int64_t num_chunks = (D + COL_CHUNK - 1) / COL_CHUNK;
    at::parallel_for(0, rows * num_chunks, 1, [&](int64_t begin, int64_t end) {
      for (int64_t task = begin; task < end; ++task) {
        const int64_t r = task / num_chunks;
        const int64_t c0 = (task % num_chunks) * COL_CHUNK;
        const int64_t c1 = std::min(D, c0 + COL_CHUNK);
        const int64_t r_base = r * top_k;
        for (int64_t d = c0; d < c1; ++d) {
          float sum = 0.0f;
          for (int64_t k = 0; k < top_k; ++k) {
            const int64_t idx = s2d_ptr[r_base + k];
            sum += read_any_as_float(down_ptr, down_st, idx * D + d);
          }
          if (down_st == at::kBFloat16) {
            sum = static_cast<float>(static_cast<at::BFloat16>(sum));
          } else if (down_st == at::kHalf) {
            sum = static_cast<float>(static_cast<at::Half>(sum));
          }
          if (round_bf16) {
            sum = static_cast<float>(static_cast<at::BFloat16>(sum));
          }
          write_any_from_float(xc_ptr, down_st, r * D + d, sum);
        }
      }
    });
  } else {
    TORCH_CHECK(down.dim() == 2 && down.size(0) == rows && down.size(1) == D, "down must be [rows, D] when top_k <= 0");
    x_comb = down.contiguous();
  }

  TORCH_CHECK(cache.scalar_type() == x_comb.scalar_type(), "cache dtype must match down dtype");

  auto safe_idx_l = safe_idx.to(at::kLong).contiguous();
  auto cache_mask_u = cache_mask.to(at::kByte).contiguous();
  auto cu_l = cu.to(at::kLong).contiguous();
  auto si_l = si.to(at::kLong).contiguous();

  at::Tensor track_mask_u;
  at::Tensor track_indices_l;
  const uint8_t* track_mask_ptr = nullptr;
  const int64_t* track_indices_ptr = nullptr;
  if (track_mask.has_value() && track_indices.has_value()) {
    track_mask_u = track_mask->to(at::kByte).contiguous();
    track_indices_l = track_indices->to(at::kLong).contiguous();
    track_mask_ptr = track_mask_u.data_ptr<uint8_t>();
    track_indices_ptr = track_indices_l.data_ptr<int64_t>();
  }

  switch (x_comb.scalar_type()) {
    case at::kFloat:
      inkling2_routed_sconv_epilogue_impl<float>(
          x_comb,
          weight,
          cache,
          safe_idx_l.data_ptr<int64_t>(),
          cache_mask_u.data_ptr<uint8_t>(),
          cu_l.data_ptr<int64_t>(),
          si_l.data_ptr<int64_t>(),
          attention,
          partial,
          track_mask_ptr,
          track_indices_ptr,
          window,
          out);
      break;
    case at::kBFloat16:
      inkling2_routed_sconv_epilogue_impl<at::BFloat16>(
          x_comb,
          weight,
          cache,
          safe_idx_l.data_ptr<int64_t>(),
          cache_mask_u.data_ptr<uint8_t>(),
          cu_l.data_ptr<int64_t>(),
          si_l.data_ptr<int64_t>(),
          attention,
          partial,
          track_mask_ptr,
          track_indices_ptr,
          window,
          out);
      break;
    case at::kHalf:
      inkling2_routed_sconv_epilogue_impl<at::Half>(
          x_comb,
          weight,
          cache,
          safe_idx_l.data_ptr<int64_t>(),
          cache_mask_u.data_ptr<uint8_t>(),
          cu_l.data_ptr<int64_t>(),
          si_l.data_ptr<int64_t>(),
          attention,
          partial,
          track_mask_ptr,
          track_indices_ptr,
          window,
          out);
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for inkling2_routed_sconv_epilogue_cpu: ", x_comb.scalar_type());
  }
  return out;
}

namespace {
template <typename scalar_t>
void update_sconv_cache_strided_impl(
    const at::Tensor& x_cast,
    at::Tensor& sconv_cache,
    const int64_t* ci_ptr,
    const uint8_t* his_ptr,
    const int64_t* qsl_ptr,
    int64_t B,
    int64_t W1,
    int64_t D) {
  const scalar_t* x_ptr = x_cast.data_ptr<scalar_t>();
  scalar_t* c_ptr = sconv_cache.data_ptr<scalar_t>();
  const int64_t x_s0 = x_cast.stride(0);
  const int64_t x_s1 = x_cast.stride(1);
  const int64_t c_s0 = sconv_cache.stride(0);
  const int64_t c_s1 = sconv_cache.stride(1);
  const int64_t c_s2 = sconv_cache.stride(2);
  const scalar_t zero = static_cast<scalar_t>(0.0f);

  at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      const int64_t ci = ci_ptr[b];
      const int64_t start = qsl_ptr[b];
      const int64_t stop = qsl_ptr[b + 1];
      const int64_t qlen = stop - start;
      if (ci == -1 || qlen <= 0) {
        continue;
      }
      const bool has_init = (his_ptr[b] != 0);
      scalar_t* slot_base = c_ptr + ci * c_s0;
      for (int64_t d = 0; d < D; ++d) {
        scalar_t* d_base = slot_base + d * c_s2;
        for (int64_t w = 0; w < W1; ++w) {
          if (qlen >= W1 - w) {
            const int64_t src_row = stop - W1 + w;
            d_base[w * c_s1] = x_ptr[src_row * x_s0 + d * x_s1];
          } else if (has_init) {
            d_base[w * c_s1] = d_base[(w + qlen) * c_s1];
          } else {
            d_base[w * c_s1] = zero;
          }
        }
      }
    }
  });
}
}  // namespace

void inkling2_update_sconv_cache_cpu(
    const at::Tensor& x,
    at::Tensor& sconv_cache,
    const at::Tensor& cache_indices,
    const at::Tensor& has_initial_state,
    const at::Tensor& query_start_loc) {
  TORCH_CHECK(x.dim() == 2, "x must be 2D [T, D]");
  TORCH_CHECK(sconv_cache.dim() == 3, "sconv_cache must be 3D [pool, W1, D]");
  const int64_t B = cache_indices.size(0);
  if (B == 0) {
    return;
  }
  const int64_t W1 = sconv_cache.size(1);
  const int64_t D = x.size(1);
  TORCH_CHECK(sconv_cache.size(2) == D, "sconv_cache hidden dim mismatch");

  auto ci_l = cache_indices.to(at::kLong).contiguous();
  auto his_u = has_initial_state.to(at::kByte).contiguous();
  auto qsl_l = query_start_loc.to(at::kLong).contiguous();
  const int64_t* ci_ptr = ci_l.data_ptr<int64_t>();
  const uint8_t* his_ptr = his_u.data_ptr<uint8_t>();
  const int64_t* qsl_ptr = qsl_l.data_ptr<int64_t>();

  at::Tensor x_cast = (x.scalar_type() == sconv_cache.scalar_type()) ? x : x.to(sconv_cache.scalar_type());
  if (x_cast.stride(1) == 1 && sconv_cache.stride(2) == 1) {
    const int64_t x_s0 = x_cast.stride(0);
    const int64_t c_s0 = sconv_cache.stride(0);
    const int64_t c_s1 = sconv_cache.stride(1);
    const char* x_bytes = reinterpret_cast<const char*>(x_cast.data_ptr());
    char* c_bytes = reinterpret_cast<char*>(sconv_cache.data_ptr());
    const size_t elem_size = sconv_cache.element_size();
    const size_t row_bytes = static_cast<size_t>(D) * elem_size;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const int64_t ci = ci_ptr[b];
        const int64_t start = qsl_ptr[b];
        const int64_t stop = qsl_ptr[b + 1];
        const int64_t qlen = stop - start;
        if (ci == -1 || qlen <= 0) {
          continue;
        }
        const bool has_init = (his_ptr[b] != 0);
        char* slot_base = c_bytes + static_cast<size_t>(ci * c_s0) * elem_size;
        for (int64_t w = 0; w < W1; ++w) {
          char* dst = slot_base + static_cast<size_t>(w * c_s1) * elem_size;
          if (qlen >= W1 - w) {
            const int64_t src_row = stop - W1 + w;
            const char* src = x_bytes + static_cast<size_t>(src_row * x_s0) * elem_size;
            std::memcpy(dst, src, row_bytes);
          } else if (has_init) {
            const char* src = slot_base + static_cast<size_t>((w + qlen) * c_s1) * elem_size;
            std::memmove(dst, src, row_bytes);
          } else {
            std::memset(dst, 0, row_bytes);
          }
        }
      }
    });
    return;
  }

  switch (sconv_cache.scalar_type()) {
    case at::kBFloat16:
      update_sconv_cache_strided_impl<at::BFloat16>(x_cast, sconv_cache, ci_ptr, his_ptr, qsl_ptr, B, W1, D);
      break;
    case at::kHalf:
      update_sconv_cache_strided_impl<at::Half>(x_cast, sconv_cache, ci_ptr, his_ptr, qsl_ptr, B, W1, D);
      break;
    case at::kFloat:
      update_sconv_cache_strided_impl<float>(x_cast, sconv_cache, ci_ptr, his_ptr, qsl_ptr, B, W1, D);
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for inkling2_update_sconv_cache_cpu: ", sconv_cache.scalar_type());
  }
}
