#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "common.h"
#include "gemm.h"
#include "vec.h"

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

  const int64_t total_tasks1 = T * num_col_chunks;
  auto run_range1 = [&](int64_t begin, int64_t end) {
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
  };
  if (total_tasks1 <= 64) {
    run_range1(0, total_tasks1);
  } else {
    at::parallel_for(0, total_tasks1, 4, run_range1);
  }
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

  const int64_t total_tasks2 = rows * num_col_chunks;
  auto run_range2 = [&](int64_t begin, int64_t end) {
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
  };
  if (total_tasks2 <= 64) {
    run_range2(0, total_tasks2);
  } else {
    at::parallel_for(0, total_tasks2, 4, run_range2);
  }
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
    const int64_t total_tasks3 = rows * num_chunks;
    auto run_range3 = [&](int64_t begin, int64_t end) {
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
    };
    if (total_tasks3 <= 64) {
      run_range3(0, total_tasks3);
    } else {
      at::parallel_for(0, total_tasks3, 4, run_range3);
    }
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

  auto run_upd_strided = [&](int64_t begin, int64_t end) {
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
  };
  if (B <= 4) {
    run_upd_strided(0, B);
  } else {
    at::parallel_for(0, B, 1, run_upd_strided);
  }
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

    auto run_upd_cont = [&](int64_t begin, int64_t end) {
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
    };
    if (B <= 4) {
      run_upd_cont(0, B);
    } else {
      at::parallel_for(0, B, 1, run_upd_cont);
    }
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

namespace {

template <typename x_t, typename out_t>
void inkling2_rms_norm_kernel_impl(
    const x_t* __restrict__ x,
    const void* __restrict__ w,
    at::ScalarType w_dtype,
    const void* __restrict__ pw,
    at::ScalarType pw_dtype,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t width,
    int64_t x_stride,
    float eps) {
  using fVec = at::vec::Vectorized<float>;
  using xVec = at::vec::Vectorized<x_t>;
  using oVec = at::vec::Vectorized<out_t>;
  constexpr int kFVec = fVec::size();
  constexpr int kOVec = oVec::size();
  static_assert(kOVec == xVec::size() || sizeof(x_t) != sizeof(out_t));

  auto load_f32 = [](const void* ptr, at::ScalarType st, int64_t d) -> fVec {
    if (st == at::kFloat) {
      return fVec::loadu(static_cast<const float*>(ptr) + d);
    } else if (st == at::kBFloat16) {
      auto [v0, _] = at::vec::convert_to_float<at::BFloat16>(
          at::vec::Vectorized<at::BFloat16>::loadu(static_cast<const at::BFloat16*>(ptr) + d));
      return v0;
    } else {
      auto [v0, _] = at::vec::convert_to_float<at::Half>(
          at::vec::Vectorized<at::Half>::loadu(static_cast<const at::Half*>(ptr) + d));
      return v0;
    }
  };

  auto process_rows = [&](int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; ++r) {
      const x_t* __restrict__ x_row = x + r * x_stride;
      out_t* __restrict__ out_row = out + r * width;

      fVec sum_v0(0.f);
      fVec sum_v1(0.f);
      int64_t d = 0;
      if constexpr (std::is_same_v<x_t, float>) {
        for (; d <= width - 2 * kFVec; d += 2 * kFVec) {
          fVec v0 = fVec::loadu(x_row + d);
          fVec v1 = fVec::loadu(x_row + d + kFVec);
          sum_v0 += v0 * v0;
          sum_v1 += v1 * v1;
        }
        sum_v0 += sum_v1;
      } else {
        constexpr int kXVec = xVec::size();
        for (; d <= width - kXVec; d += kXVec) {
          auto [v0, v1] = at::vec::convert_to_float<x_t>(xVec::loadu(x_row + d));
          sum_v0 += v0 * v0;
          sum_v1 += v1 * v1;
        }
        sum_v0 += sum_v1;
      }
      float ss = vec_reduce_sum(sum_v0);
      for (; d < width; ++d) {
        float v = static_cast<float>(x_row[d]);
        ss += v * v;
      }
      const float rstd = 1.0f / std::sqrt(ss / static_cast<float>(width) + eps);
      const fVec rstd_v(rstd);

      d = 0;
      if constexpr (std::is_same_v<x_t, out_t> && !std::is_same_v<x_t, float>) {
        constexpr int kVec = oVec::size();
        for (; d <= width - kVec; d += kVec) {
          auto [v0, v1] = at::vec::convert_to_float<x_t>(xVec::loadu(x_row + d));
          if (pw != nullptr) {
            v0 *= load_f32(pw, pw_dtype, d);
            v1 *= load_f32(pw, pw_dtype, d + kFVec);
          }
          v0 *= rstd_v;
          v1 *= rstd_v;
          if (w != nullptr) {
            v0 *= load_f32(w, w_dtype, d);
            v1 *= load_f32(w, w_dtype, d + kFVec);
          }
          convert_from_float_ext<out_t>(v0, v1).store(out_row + d);
        }
      } else if constexpr (std::is_same_v<x_t, float> && std::is_same_v<out_t, float>) {
        for (; d <= width - kFVec; d += kFVec) {
          fVec v0 = fVec::loadu(x_row + d);
          if (pw != nullptr) {
            v0 *= load_f32(pw, pw_dtype, d);
          }
          v0 *= rstd_v;
          if (w != nullptr) {
            v0 *= load_f32(w, w_dtype, d);
          }
          v0.store(out_row + d);
        }
      }
      for (; d < width; ++d) {
        float val = static_cast<float>(x_row[d]);
        if (pw != nullptr) {
          val *= read_any_as_float(pw, pw_dtype, d);
        }
        val *= rstd;
        if (w != nullptr) {
          val *= read_any_as_float(w, w_dtype, d);
        }
        out_row[d] = static_cast<out_t>(val);
      }
    }
  };

  if (rows * width <= 32768) {
    process_rows(0, rows);
  } else {
    at::parallel_for(0, rows, 1, process_rows);
  }
}

}  // namespace

void inkling2_rms_norm_cpu(
    const at::Tensor& x,
    const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& pre_weight_opt,
    at::Tensor& out,
    double eps) {
  TORCH_CHECK(
      x.stride(-1) == 1 && out.is_contiguous(),
      "inkling2_rms_norm_cpu requires last-dim contiguous x and contiguous out");
  const int64_t width = x.size(-1);
  const int64_t rows = x.numel() / width;
  if (rows == 0) {
    return;
  }
  auto x_2d = x.reshape({rows, width});
  const int64_t x_stride = (rows > 1) ? x_2d.stride(0) : width;

  const void* w_ptr = weight_opt.has_value() ? weight_opt.value().data_ptr() : nullptr;
  at::ScalarType w_dtype = weight_opt.has_value() ? weight_opt.value().scalar_type() : at::kFloat;
  const void* pw_ptr = pre_weight_opt.has_value() ? pre_weight_opt.value().data_ptr() : nullptr;
  at::ScalarType pw_dtype = pre_weight_opt.has_value() ? pre_weight_opt.value().scalar_type() : at::kFloat;

  const auto x_st = x.scalar_type();
  const auto o_st = out.scalar_type();
  if (x_st == at::kBFloat16 && o_st == at::kBFloat16) {
    inkling2_rms_norm_kernel_impl<at::BFloat16, at::BFloat16>(
        x_2d.data_ptr<at::BFloat16>(),
        w_ptr,
        w_dtype,
        pw_ptr,
        pw_dtype,
        out.data_ptr<at::BFloat16>(),
        rows,
        width,
        x_stride,
        static_cast<float>(eps));
  } else if (x_st == at::kFloat && o_st == at::kFloat) {
    inkling2_rms_norm_kernel_impl<float, float>(
        x_2d.data_ptr<float>(),
        w_ptr,
        w_dtype,
        pw_ptr,
        pw_dtype,
        out.data_ptr<float>(),
        rows,
        width,
        x_stride,
        static_cast<float>(eps));
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, x_st, "inkling2_rms_norm_cpu_x", [&] {
      const scalar_t* x_p = x_2d.data_ptr<scalar_t>();
      void* o_p = out.data_ptr();
      auto run = [&](int64_t begin, int64_t end) {
        for (int64_t r = begin; r < end; ++r) {
          const scalar_t* xr = x_p + r * x_stride;
          double ss = 0.0;
          for (int64_t d = 0; d < width; ++d) {
            double v = static_cast<double>(xr[d]);
            ss += v * v;
          }
          float rstd = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(width) + eps));
          for (int64_t d = 0; d < width; ++d) {
            float v = static_cast<float>(xr[d]);
            if (pw_ptr != nullptr) {
              v *= read_any_as_float(pw_ptr, pw_dtype, d);
            }
            v *= rstd;
            if (w_ptr != nullptr) {
              v *= read_any_as_float(w_ptr, w_dtype, d);
            }
            write_any_from_float(o_p, o_st, r * width + d, v);
          }
        }
      };
      if (rows * width <= 32768) {
        run(0, rows);
      } else {
        at::parallel_for(0, rows, 1, run);
      }
    });
  }
}

void inkling2_qk_l2_clip_cpu(
    const at::Tensor& x,
    const std::optional<at::Tensor>& scale_opt,
    const std::optional<at::Tensor>& output_scale_opt,
    at::Tensor& out,
    double threshold,
    double eps,
    int64_t rows_per_output_scale) {
  TORCH_CHECK(out.is_contiguous(), "inkling2_qk_l2_clip_cpu requires contiguous out");
  const int64_t N = x.size(-1);
  const int64_t M = x.numel() / N;
  if (M == 0) {
    return;
  }
  auto x_2d = x.reshape({M, N});
  const bool x_last_cont = (x.stride(-1) == 1);
  auto x_c = x_last_cont ? x_2d : x_2d.contiguous();
  const int64_t x_stride = (M > 1) ? x_c.stride(0) : N;

  const void* sc_ptr = nullptr;
  at::ScalarType sc_dtype = at::kFloat;
  int64_t sc_heads = 1;
  int64_t sc_stride0 = 0;
  if (scale_opt.has_value()) {
    const auto& sc = scale_opt.value();
    TORCH_CHECK(sc.stride(-1) == 1, "scale must be last-dim contiguous");
    sc_ptr = sc.data_ptr();
    sc_dtype = sc.scalar_type();
    if (sc.ndimension() == 2) {
      sc_heads = sc.size(0);
      sc_stride0 = sc.stride(0);
    }
  }

  const void* os_ptr = nullptr;
  at::ScalarType os_dtype = at::kFloat;
  if (output_scale_opt.has_value()) {
    os_ptr = output_scale_opt.value().data_ptr();
    os_dtype = output_scale_opt.value().scalar_type();
  }

  const auto x_st = x_c.scalar_type();
  const auto o_st = out.scalar_type();
  const void* x_ptr = x_c.data_ptr();
  void* o_ptr = out.data_ptr();
  const float thresh_f = static_cast<float>(threshold);
  const float eps_f = static_cast<float>(eps);

  auto run_rows = [&](int64_t begin, int64_t end) {
    std::vector<float> tmp(static_cast<size_t>(N));
    for (int64_t m = begin; m < end; ++m) {
      const int64_t x_off = m * x_stride;
      const int64_t sc_off = (sc_ptr != nullptr && sc_heads > 1) ? ((m % sc_heads) * sc_stride0) : 0;
      float sum_sq = 0.0f;
      if (x_st == at::kBFloat16 && sc_ptr == nullptr) {
        const at::BFloat16* xr = static_cast<const at::BFloat16*>(x_ptr) + x_off;
        using bVec = at::vec::Vectorized<at::BFloat16>;
        using fVec = at::vec::Vectorized<float>;
        fVec s0(0.f), s1(0.f);
        int64_t d = 0;
        for (; d <= N - bVec::size(); d += bVec::size()) {
          auto [v0, v1] = at::vec::convert_to_float<at::BFloat16>(bVec::loadu(xr + d));
          v0.store(tmp.data() + d);
          v1.store(tmp.data() + d + fVec::size());
          s0 += v0 * v0;
          s1 += v1 * v1;
        }
        sum_sq = vec_reduce_sum(s0 + s1);
        for (; d < N; ++d) {
          float v = static_cast<float>(xr[d]);
          tmp[d] = v;
          sum_sq += v * v;
        }
      } else {
        for (int64_t d = 0; d < N; ++d) {
          float v = read_any_as_float(x_ptr, x_st, x_off + d);
          if (sc_ptr != nullptr) {
            v *= read_any_as_float(sc_ptr, sc_dtype, sc_off + d);
          }
          tmp[d] = v;
          sum_sq += v * v;
        }
      }
      const float norm = (sum_sq > 0.0f) ? std::sqrt(sum_sq) : 0.0f;
      const float coeff = std::min(thresh_f / (norm + eps_f), 1.0f);
      const float out_sc = (os_ptr != nullptr) ? read_any_as_float(os_ptr, os_dtype, m / rows_per_output_scale) : 1.0f;

      if (o_st == at::kBFloat16 && os_ptr == nullptr) {
        at::BFloat16* orow = static_cast<at::BFloat16*>(o_ptr) + m * N;
        using bVec = at::vec::Vectorized<at::BFloat16>;
        using fVec = at::vec::Vectorized<float>;
        const fVec cv(coeff);
        int64_t d = 0;
        for (; d <= N - bVec::size(); d += bVec::size()) {
          fVec v0 = fVec::loadu(tmp.data() + d) * cv;
          fVec v1 = fVec::loadu(tmp.data() + d + fVec::size()) * cv;
          convert_from_float_ext<at::BFloat16>(v0, v1).store(orow + d);
        }
        for (; d < N; ++d) {
          orow[d] = static_cast<at::BFloat16>(tmp[d] * coeff);
        }
      } else {
        for (int64_t d = 0; d < N; ++d) {
          float y = tmp[d] * coeff;
          if (os_ptr != nullptr) {
            // Round to output_dtype first, then multiply by output_scale matching _qk_l2_clip_cpu
            if (o_st == at::kBFloat16) {
              y = static_cast<float>(static_cast<at::BFloat16>(y)) * out_sc;
            } else if (o_st == at::kHalf) {
              y = static_cast<float>(static_cast<at::Half>(y)) * out_sc;
            } else {
              y *= out_sc;
            }
          }
          write_any_from_float(o_ptr, o_st, m * N + d, y);
        }
      }
    }
  };

  if (M * N <= 32768) {
    run_rows(0, M);
  } else {
    at::parallel_for(0, M, 1, run_rows);
  }
}

void inkling2_gate_sink_cpu(
    const at::Tensor& output,
    const at::Tensor& gate,
    const at::Tensor& lse,
    const at::Tensor& sinks,
    at::Tensor& out,
    int64_t heads,
    int64_t group,
    int64_t padded_group) {
  const int64_t tokens = out.size(0);
  if (tokens == 0) {
    return;
  }
  const int64_t head_dim = out.size(1) / heads;
  const int64_t num_groups = heads / group;
  const int64_t out_pad_heads = num_groups * padded_group;

  auto out_3d = output.reshape({tokens, out_pad_heads, head_dim});
  const int64_t o_s0 = out_3d.stride(0);
  const int64_t o_s1 = out_3d.stride(1);
  const int64_t o_s2 = out_3d.stride(2);

  auto gate_2d = gate.narrow(0, 0, tokens);
  const int64_t g_s0 = gate_2d.stride(0);
  const int64_t g_s1 = gate_2d.stride(1);

  auto lse_2d = lse.narrow(0, 0, tokens).reshape({tokens, out_pad_heads});
  const int64_t l_s0 = lse_2d.stride(0);
  const int64_t l_s1 = lse_2d.stride(1);

  auto sinks_2d = sinks.narrow(0, 0, tokens);
  const int64_t s_s0 = sinks_2d.stride(0);
  const int64_t s_s1 = sinks_2d.stride(1);

  const void* op = out_3d.data_ptr();
  const void* gp = gate_2d.data_ptr();
  const void* lp = lse_2d.data_ptr();
  const void* sp = sinks_2d.data_ptr();
  void* dst = out.data_ptr();

  const auto o_st = out_3d.scalar_type();
  const auto g_st = gate_2d.scalar_type();
  const auto l_st = lse_2d.scalar_type();
  const auto s_st = sinks_2d.scalar_type();
  const auto dst_st = out.scalar_type();

  const bool fast_bf16 =
      (o_s2 == 1 && g_s1 == 1 && o_st == at::kFloat && g_st == at::kBFloat16 && dst_st == at::kBFloat16);

  auto run_th = [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t t = idx / heads;
      const int64_t h = idx % heads;
      const int64_t grp_idx = h / group;
      const int64_t in_grp = h % group;
      const int64_t src_head = grp_idx * padded_group + in_grp;

      const float lse_v = read_any_as_float(lp, l_st, t * l_s0 + src_head * l_s1);
      const float sink_v = read_any_as_float(sp, s_st, t * s_s0 + h * s_s1);
      const float sink_sig = 1.0f / (1.0f + std::exp(-(lse_v - sink_v)));

      if (fast_bf16) {
        const float* o_row = static_cast<const float*>(op) + t * o_s0 + src_head * o_s1;
        const at::BFloat16* g_row = static_cast<const at::BFloat16*>(gp) + t * g_s0 + h * head_dim;
        at::BFloat16* d_row = static_cast<at::BFloat16*>(dst) + t * (heads * head_dim) + h * head_dim;
        using bVec = at::vec::Vectorized<at::BFloat16>;
        using fVec = at::vec::Vectorized<float>;
        const fVec one(1.0f);
        const fVec ssig(sink_sig);
        int64_t d = 0;
        for (; d <= head_dim - bVec::size(); d += bVec::size()) {
          fVec ov0 = fVec::loadu(o_row + d);
          fVec ov1 = fVec::loadu(o_row + d + fVec::size());
          auto [gv0, gv1] = at::vec::convert_to_float<at::BFloat16>(bVec::loadu(g_row + d));
          fVec sig0 = one / (one + (fVec(0.0f) - gv0).exp());
          fVec sig1 = one / (one + (fVec(0.0f) - gv1).exp());
          convert_from_float_ext<at::BFloat16>(ov0 * sig0 * ssig, ov1 * sig1 * ssig).store(d_row + d);
        }
        for (; d < head_dim; ++d) {
          float gv = static_cast<float>(g_row[d]);
          float gsig = 1.0f / (1.0f + std::exp(-gv));
          d_row[d] = static_cast<at::BFloat16>(o_row[d] * gsig * sink_sig);
        }
      } else {
        for (int64_t d = 0; d < head_dim; ++d) {
          float ov = read_any_as_float(op, o_st, t * o_s0 + src_head * o_s1 + d * o_s2);
          float gv = read_any_as_float(gp, g_st, t * g_s0 + (h * head_dim + d) * g_s1);
          float gsig = 1.0f / (1.0f + std::exp(-gv));
          write_any_from_float(dst, dst_st, t * (heads * head_dim) + h * head_dim + d, ov * gsig * sink_sig);
        }
      }
    }
  };

  const int64_t total_items = tokens * heads;
  if (total_items * head_dim <= 32768) {
    run_th(0, total_items);
  } else {
    at::parallel_for(0, total_items, 4, run_th);
  }
}

std::vector<at::Tensor> inkling2_attn_res_cpu(
    at::TensorList blocks,
    const std::optional<at::Tensor>& partial_addend_opt,
    const at::Tensor& queries,
    bool normalize_values) {
  const int64_t n_sources = static_cast<int64_t>(blocks.size());
  TORCH_CHECK(n_sources > 0, "attn_res requires at least one block");
  const at::Tensor& base = blocks[0];
  const int64_t tokens = base.size(0);
  const int64_t width = base.size(1);
  const int64_t n_queries = queries.size(0);
  const auto b_st = base.scalar_type();
  const auto p_st = partial_addend_opt.has_value() ? partial_addend_opt.value().scalar_type() : b_st;

  std::vector<at::Tensor> outs;
  outs.reserve(static_cast<size_t>(n_queries));
  std::vector<float*> out_ptrs(static_cast<size_t>(n_queries));
  for (int64_t q = 0; q < n_queries; ++q) {
    outs.push_back(at::empty({tokens, width}, base.options().dtype(at::kFloat)));
    out_ptrs[static_cast<size_t>(q)] = outs.back().data_ptr<float>();
  }
  if (tokens == 0) {
    return outs;
  }

  std::vector<const void*> blk_ptrs(static_cast<size_t>(n_sources));
  for (int64_t s = 0; s < n_sources; ++s) {
    blk_ptrs[static_cast<size_t>(s)] = blocks[s].data_ptr();
  }
  const void* part_ptr = partial_addend_opt.has_value() ? partial_addend_opt.value().data_ptr() : nullptr;
  const float* q_ptr = queries.data_ptr<float>();

  auto run_tokens = [&](int64_t begin, int64_t end) {
    std::vector<float> vals(static_cast<size_t>(n_sources * width));
    std::vector<float> kbuf(static_cast<size_t>(n_sources * width));
    std::vector<float> logits(static_cast<size_t>(n_queries * n_sources));
    std::vector<float> weights(static_cast<size_t>(n_queries * n_sources));

    for (int64_t t = begin; t < end; ++t) {
      for (int64_t s = 0; s < n_sources; ++s) {
        float* vrow = vals.data() + s * width;
        float* krow = kbuf.data() + s * width;
        const void* src_ptr = blk_ptrs[static_cast<size_t>(s)];
        const bool add_part = (s == n_sources - 1 && part_ptr != nullptr);
        double ss = 0.0;
        if (b_st == at::kBFloat16 && !add_part) {
          const at::BFloat16* br = static_cast<const at::BFloat16*>(src_ptr) + t * width;
          using bVec = at::vec::Vectorized<at::BFloat16>;
          using fVec = at::vec::Vectorized<float>;
          fVec sv(0.f);
          int64_t d = 0;
          for (; d <= width - bVec::size(); d += bVec::size()) {
            auto [v0, v1] = at::vec::convert_to_float<at::BFloat16>(bVec::loadu(br + d));
            v0.store(vrow + d);
            v1.store(vrow + d + fVec::size());
            sv += v0 * v0 + v1 * v1;
          }
          ss = static_cast<double>(vec_reduce_sum(sv));
          for (; d < width; ++d) {
            float v = static_cast<float>(br[d]);
            vrow[d] = v;
            ss += static_cast<double>(v) * v;
          }
        } else {
          for (int64_t d = 0; d < width; ++d) {
            float v = read_any_as_float(src_ptr, b_st, t * width + d);
            if (add_part) {
              v += read_any_as_float(part_ptr, p_st, t * width + d);
            }
            vrow[d] = v;
            ss += static_cast<double>(v) * v;
          }
        }
        const float rstd = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(width) + 1e-6));
        using fVec = at::vec::Vectorized<float>;
        const fVec rstd_v(rstd);
        int64_t d = 0;
        for (; d <= width - fVec::size(); d += fVec::size()) {
          (fVec::loadu(vrow + d) * rstd_v).store(krow + d);
        }
        for (; d < width; ++d) {
          krow[d] = vrow[d] * rstd;
        }

        for (int64_t q = 0; q < n_queries; ++q) {
          const float* qr = q_ptr + q * width;
          fVec dsum(0.f);
          int64_t i = 0;
          for (; i <= width - fVec::size(); i += fVec::size()) {
            dsum += fVec::loadu(krow + i) * fVec::loadu(qr + i);
          }
          float dot_v = vec_reduce_sum(dsum);
          for (; i < width; ++i) {
            dot_v += krow[i] * qr[i];
          }
          logits[static_cast<size_t>(q * n_sources + s)] = dot_v;
        }
      }

      const float* val_src = normalize_values ? kbuf.data() : vals.data();
      for (int64_t q = 0; q < n_queries; ++q) {
        const float* lrow = logits.data() + q * n_sources;
        float* wrow = weights.data() + q * n_sources;
        float mx = lrow[0];
        for (int64_t s = 1; s < n_sources; ++s) {
          mx = std::max(mx, lrow[s]);
        }
        float sm = 0.0f;
        for (int64_t s = 0; s < n_sources; ++s) {
          float e = std::exp(lrow[s] - mx);
          wrow[s] = e;
          sm += e;
        }
        const float inv_sm = 1.0f / sm;
        for (int64_t s = 0; s < n_sources; ++s) {
          wrow[s] *= inv_sm;
        }

        float* dst = out_ptrs[static_cast<size_t>(q)] + t * width;
        using fVec = at::vec::Vectorized<float>;
        int64_t d = 0;
        for (; d <= width - fVec::size(); d += fVec::size()) {
          fVec acc(0.f);
          for (int64_t s = 0; s < n_sources; ++s) {
            acc += fVec::loadu(val_src + s * width + d) * fVec(wrow[s]);
          }
          acc.store(dst + d);
        }
        for (; d < width; ++d) {
          float acc = 0.0f;
          for (int64_t s = 0; s < n_sources; ++s) {
            acc += val_src[s * width + d] * wrow[s];
          }
          dst[d] = acc;
        }
      }
    }
  };

  if (tokens <= 4) {
    run_tokens(0, tokens);
  } else {
    at::parallel_for(0, tokens, 1, run_tokens);
  }
  return outs;
}

void inkling2_route_topk_cpu(
    const at::Tensor& partials,
    const std::optional<at::Tensor>& bias_opt,
    at::Tensor& logits,
    at::Tensor& ids,
    const std::optional<at::Tensor>& weights_opt,
    int64_t topk,
    double route_scale,
    const std::optional<at::Tensor>& global_scale_opt) {
  const int64_t groups = partials.size(0);
  const int64_t m = partials.size(1);
  const int64_t n = partials.size(2);
  if (m == 0) {
    return;
  }
  auto part_c = partials.scalar_type() == at::kFloat ? partials.contiguous() : partials.to(at::kFloat).contiguous();
  const float* part_ptr = part_c.data_ptr<float>();
  const void* bias_ptr = bias_opt.has_value() ? bias_opt.value().data_ptr() : nullptr;
  const auto bias_st = bias_opt.has_value() ? bias_opt.value().scalar_type() : at::kFloat;
  float* logits_ptr = logits.data_ptr<float>();
  int32_t* ids_ptr = ids.data_ptr<int32_t>();
  float* w_ptr = weights_opt.has_value() ? weights_opt.value().data_ptr<float>() : nullptr;
  float gs = 1.0f;
  if (global_scale_opt.has_value()) {
    gs = read_any_as_float(global_scale_opt.value().data_ptr(), global_scale_opt.value().scalar_type(), 0);
  }
  const float scale_mul = static_cast<float>(route_scale) * gs;

  auto run_rows = [&](int64_t begin, int64_t end) {
    std::vector<float> sel(static_cast<size_t>(n));
    std::vector<int32_t> order(static_cast<size_t>(n));
    std::vector<float> topk_lp(static_cast<size_t>(topk));

    for (int64_t r = begin; r < end; ++r) {
      float* lrow = logits_ptr + r * n;
      for (int64_t j = 0; j < n; ++j) {
        float sum = part_ptr[r * n + j];
        for (int64_t g = 1; g < groups; ++g) {
          sum += part_ptr[(g * m + r) * n + j];
        }
        lrow[j] = sum;
        float s = 1.0f / (1.0f + std::exp(-sum));
        if (bias_ptr != nullptr) {
          s += read_any_as_float(bias_ptr, bias_st, j);
        }
        sel[static_cast<size_t>(j)] = s;
        order[static_cast<size_t>(j)] = static_cast<int32_t>(j);
      }
      std::partial_sort(order.begin(), order.begin() + topk, order.end(), [&](int32_t a, int32_t b) {
        const float sa = sel[static_cast<size_t>(a)];
        const float sb = sel[static_cast<size_t>(b)];
        if (sa != sb) {
          return sa > sb;
        }
        return a < b;
      });
      int32_t* irow = ids_ptr + r * topk;
      for (int64_t k = 0; k < topk; ++k) {
        irow[k] = order[static_cast<size_t>(k)];
      }
      if (w_ptr != nullptr) {
        float mx = -std::numeric_limits<float>::infinity();
        for (int64_t k = 0; k < topk; ++k) {
          float x = lrow[irow[k]];
          float lp = (x >= 0.0f) ? (-std::log1p(std::exp(-x))) : (x - std::log1p(std::exp(x)));
          topk_lp[static_cast<size_t>(k)] = lp;
          mx = std::max(mx, lp);
        }
        float sm = 0.0f;
        for (int64_t k = 0; k < topk; ++k) {
          float e = std::exp(topk_lp[static_cast<size_t>(k)] - mx);
          topk_lp[static_cast<size_t>(k)] = e;
          sm += e;
        }
        const float inv = scale_mul / sm;
        float* wrow = w_ptr + r * topk;
        for (int64_t k = 0; k < topk; ++k) {
          wrow[k] = topk_lp[static_cast<size_t>(k)] * inv;
        }
      }
    }
  };

  if (m <= 8) {
    run_rows(0, m);
  } else {
    at::parallel_for(0, m, 1, run_rows);
  }
}

at::Tensor inkling2_bf16_linear_fp32_cpu(
    const at::Tensor& x, const at::Tensor& packed_weight, const std::optional<at::Tensor>& bias_opt) {
  TORCH_CHECK(
      x.scalar_type() == at::kBFloat16 && packed_weight.scalar_type() == at::kBFloat16,
      "inkling2_bf16_linear_fp32_cpu requires bfloat16 tensors");
  TORCH_CHECK(x.is_contiguous() && packed_weight.is_contiguous(), "Inputs must be contiguous");
  const int64_t K = x.size(-1);
  const int64_t M = x.numel() / K;
  const int64_t N = packed_weight.size(0);
  TORCH_CHECK(packed_weight.size(1) == K && K % 2 == 0 && N % 32 == 0, "Shape alignment mismatch");

  auto out = at::empty({M, N}, x.options().dtype(at::kFloat));
  if (M == 0) {
    return out;
  }

  const at::BFloat16* A = x.data_ptr<at::BFloat16>();
  const at::BFloat16* B = packed_weight.data_ptr<at::BFloat16>();
  float* C = out.data_ptr<float>();
  const void* bias_ptr = bias_opt.has_value() ? bias_opt.value().data_ptr() : nullptr;
  const auto bias_st = bias_opt.has_value() ? bias_opt.value().scalar_type() : at::kFloat;

  constexpr int64_t BLOCK_N = 32;
  const int64_t NB = N / BLOCK_N;

#if defined(CPU_CAPABILITY_AVX512)
  if (M <= 4) {
    auto run_nb = [&](int64_t begin, int64_t end) {
      const float* a_f32 = reinterpret_cast<const float*>(A);
      const int64_t K2 = K >> 1;
      for (int64_t nb = begin; nb < end; ++nb) {
        const at::BFloat16* Bn = B + nb * BLOCK_N * K;
        __m512 vc[4][2];
        for (int64_t r = 0; r < M; ++r) {
          vc[r][0] = _mm512_setzero_ps();
          vc[r][1] = _mm512_setzero_ps();
        }
        for (int64_t k = 0; k < K2; ++k) {
          const at::BFloat16* brow = Bn + k * (BLOCK_N * 2);
          __m512bh vb0 = (__m512bh)(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(brow)));
          __m512bh vb1 = (__m512bh)(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(brow + 32)));
          for (int64_t r = 0; r < M; ++r) {
            __m512bh va = (__m512bh)(_mm512_set1_ps(a_f32[r * K2 + k]));
            vc[r][0] = _mm512_dpbf16_ps(vc[r][0], va, vb0);
            vc[r][1] = _mm512_dpbf16_ps(vc[r][1], va, vb1);
          }
        }
        for (int64_t r = 0; r < M; ++r) {
          _mm512_storeu_ps(C + r * N + nb * BLOCK_N, vc[r][0]);
          _mm512_storeu_ps(C + r * N + nb * BLOCK_N + 16, vc[r][1]);
          if (bias_ptr != nullptr) {
            for (int64_t j = 0; j < BLOCK_N; ++j) {
              C[r * N + nb * BLOCK_N + j] += read_any_as_float(bias_ptr, bias_st, nb * BLOCK_N + j);
            }
          }
        }
      }
    };
    if (M * N * K <= 2097152) {
      run_nb(0, NB);
    } else {
      at::parallel_for(0, NB, 1, run_nb);
    }
    return out;
  }
#endif

#if !defined(SGLANG_CPU_ARM64_SKIP_X86_ONLY_OPS)
  constexpr int64_t BLOCK_M = 32;
  const int64_t MB = (M + BLOCK_M - 1) / BLOCK_M;
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    loop_2d<at::BFloat16>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t) {
      int64_t m_size = std::min(M - mb * BLOCK_M, BLOCK_M);
      const at::BFloat16* Am = A + mb * BLOCK_M * K;
      const at::BFloat16* Bn = B + nb * BLOCK_N * K;
      float* Cm = C + mb * BLOCK_M * N + nb * BLOCK_N;
      at::native::cpublas::brgemm(m_size, BLOCK_N, K, K, BLOCK_N, N, false, Am, Bn, Cm);
      if (bias_ptr != nullptr) {
        for (int64_t r = 0; r < m_size; ++r) {
          for (int64_t j = 0; j < BLOCK_N; ++j) {
            Cm[r * N + j] += read_any_as_float(bias_ptr, bias_st, nb * BLOCK_N + j);
          }
        }
      }
    });
    at::native::cpublas::brgemm_release();
  });
#else
  for (int64_t r = 0; r < M; ++r) {
    for (int64_t nb = 0; nb < NB; ++nb) {
      const at::BFloat16* Bn = B + nb * BLOCK_N * K;
      for (int64_t j = 0; j < BLOCK_N; ++j) {
        float acc = (bias_ptr != nullptr) ? read_any_as_float(bias_ptr, bias_st, nb * BLOCK_N + j) : 0.0f;
        for (int64_t k = 0; k < K / 2; ++k) {
          acc += static_cast<float>(A[r * K + k * 2 + 0]) * static_cast<float>(Bn[k * BLOCK_N * 2 + j * 2 + 0]);
          acc += static_cast<float>(A[r * K + k * 2 + 1]) * static_cast<float>(Bn[k * BLOCK_N * 2 + j * 2 + 1]);
        }
        C[r * N + nb * BLOCK_N + j] = acc;
      }
    }
  }
#endif
  return out;
}
