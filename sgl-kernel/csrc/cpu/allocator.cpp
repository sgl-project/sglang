#include "common.h"

namespace {

inline int64_t ceil_div(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// Ensure tensor is of the given dtype and contiguous, avoiding copy when already matching.
inline at::Tensor as_dtype(at::Tensor& t, at::ScalarType dtype) {
  if (t.scalar_type() == dtype && t.is_contiguous()) return t;
  return t.to(dtype).contiguous();
}

// Check if tensor is of given dtype and contiguous
inline bool is_dtype_contig(const at::Tensor& t, at::ScalarType dtype) {
  return t.scalar_type() == dtype && t.is_contiguous();
}

// ---------------------------------------------------------------------------
// alloc_extend_kernel_impl: templated on input type, always writes int64 output
// ---------------------------------------------------------------------------
template <typename in_t>
void alloc_extend_kernel_impl(
    const in_t* pre_lens_ptr,
    const in_t* seq_lens_ptr,
    const in_t* last_loc_ptr,
    const in_t* free_pages_ptr,
    int64_t* out_ptr,
    int64_t bs,
    int64_t page_size) {
  std::vector<int64_t> extend_lens(bs);
  std::vector<int64_t> output_start(bs);
  std::vector<int64_t> num_new_pages_vec(bs);
  std::vector<int64_t> new_page_start(bs);

  int64_t extend_cumsum = 0;
  int64_t page_cumsum = 0;
  for (int64_t i = 0; i < bs; ++i) {
    int64_t pre_len = static_cast<int64_t>(pre_lens_ptr[i]);
    int64_t seq_len = static_cast<int64_t>(seq_lens_ptr[i]);
    int64_t ext_len = seq_len - pre_len;
    extend_lens[i] = ext_len;

    output_start[i] = extend_cumsum;
    extend_cumsum += ext_len;

    int64_t pages_after = ceil_div(seq_len, page_size);
    int64_t pages_before = ceil_div(pre_len, page_size);
    int64_t new_pages = pages_after - pages_before;
    num_new_pages_vec[i] = new_pages;

    new_page_start[i] = page_cumsum;
    page_cumsum += new_pages;
  }

  at::parallel_for(0, bs, 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t pre_len = static_cast<int64_t>(pre_lens_ptr[i]);
      const int64_t seq_len = static_cast<int64_t>(seq_lens_ptr[i]);
      const int64_t ext_len = extend_lens[i];
      if (ext_len == 0) continue;

      const int64_t out_start = output_start[i];
      const int64_t pg_start = new_page_start[i];
      const int64_t num_new_pages_self = num_new_pages_vec[i];
      const int64_t last_loc_i = static_cast<int64_t>(last_loc_ptr[i]);
      int64_t* out = out_ptr + out_start;

      // Part 1: fill the old partial page
      const int64_t page_boundary = ceil_div(pre_len, page_size) * page_size;
      const int64_t num_part1 = std::min(seq_len, page_boundary) - pre_len;
      for (int64_t j = 0; j < num_part1; ++j) {
        out[j] = last_loc_i + 1 + j;
      }

      if (pre_len + num_part1 == seq_len) continue;

      // Part 2: fill new full pages
      const int64_t full_page_start = ceil_div(pre_len, page_size) * page_size;
      const int64_t full_page_end = (seq_len / page_size) * page_size;
      const int64_t num_part2 = full_page_end - full_page_start;
      if (num_part2 > 0) {
        int64_t* out2 = out + num_part1;
        for (int64_t j = 0; j < num_part2; ++j) {
          int64_t page_idx = j / page_size;
          int64_t in_page_off = j % page_size;
          out2[j] = static_cast<int64_t>(free_pages_ptr[pg_start + page_idx]) * page_size + in_page_off;
        }
      }

      if (pre_len + num_part1 + num_part2 == seq_len) continue;

      // Part 3: fill the new partial page at the end
      const int64_t num_part3 = seq_len - (seq_len / page_size) * page_size;
      if (num_part3 > 0) {
        int64_t start_page = static_cast<int64_t>(free_pages_ptr[pg_start + num_new_pages_self - 1]);
        int64_t* out3 = out + num_part1 + num_part2;
        for (int64_t j = 0; j < num_part3; ++j) {
          out3[j] = start_page * page_size + j;
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// alloc_decode_kernel_impl: templated on input type, always writes int64 output
// ---------------------------------------------------------------------------
template <typename in_t>
void alloc_decode_kernel_impl(
    const in_t* seq_lens_ptr,
    const in_t* last_loc_ptr,
    const in_t* free_pages_ptr,
    int64_t* out_ptr,
    int64_t bs,
    int64_t page_size) {
  std::vector<int64_t> num_new_pages_vec(bs);
  std::vector<int64_t> new_page_start(bs);

  int64_t page_cumsum = 0;
  for (int64_t i = 0; i < bs; ++i) {
    int64_t seq_len = static_cast<int64_t>(seq_lens_ptr[i]);
    int64_t pre_len = seq_len - 1;
    int64_t pages_after = ceil_div(seq_len, page_size);
    int64_t pages_before = ceil_div(pre_len, page_size);
    int64_t new_pages = pages_after - pages_before;
    num_new_pages_vec[i] = new_pages;
    new_page_start[i] = page_cumsum;
    page_cumsum += new_pages;
  }

  at::parallel_for(0, bs, 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      if (num_new_pages_vec[i] == 0) {
        out_ptr[i] = static_cast<int64_t>(last_loc_ptr[i]) + 1;
      } else {
        int64_t page = static_cast<int64_t>(free_pages_ptr[new_page_start[i]]);
        out_ptr[i] = page * page_size;
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Prepare output buffer: always int64, reuse tensor if already int64+contiguous
// ---------------------------------------------------------------------------
inline std::pair<int64_t*, at::Tensor> prepare_out_i64(at::Tensor& out_indices) {
  if (out_indices.scalar_type() == at::kLong && out_indices.is_contiguous()) {
    return {out_indices.data_ptr<int64_t>(), at::Tensor()};
  }
  auto buf = at::empty({out_indices.numel()}, out_indices.options().dtype(at::kLong));
  return {buf.data_ptr<int64_t>(), buf};
}

inline void copy_back_if_needed(at::Tensor& out_indices, const at::Tensor& buf) {
  if (buf.defined()) {
    out_indices.copy_(buf);
  }
}

} // anonymous namespace

// ============================================================================
// alloc_extend_kernel_cpu
// ============================================================================
void alloc_extend_kernel_cpu(
    at::Tensor& pre_lens,
    at::Tensor& seq_lens,
    at::Tensor& last_loc,
    at::Tensor& free_pages,
    at::Tensor& out_indices,
    int64_t page_size) {
  const int64_t bs = pre_lens.size(0);
  auto [out_ptr, out_buf] = prepare_out_i64(out_indices);

  if (pre_lens.scalar_type() == at::kInt) {
    auto pl = as_dtype(pre_lens, at::kInt);
    auto sl = as_dtype(seq_lens, at::kInt);
    auto ll = as_dtype(last_loc, at::kInt);
    auto fp = as_dtype(free_pages, at::kInt);
    alloc_extend_kernel_impl<int32_t>(
        pl.data_ptr<int32_t>(), sl.data_ptr<int32_t>(),
        ll.data_ptr<int32_t>(), fp.data_ptr<int32_t>(),
        out_ptr, bs, page_size);
  } else {
    auto pl = as_dtype(pre_lens, at::kLong);
    auto sl = as_dtype(seq_lens, at::kLong);
    auto ll = as_dtype(last_loc, at::kLong);
    auto fp = as_dtype(free_pages, at::kLong);
    alloc_extend_kernel_impl<int64_t>(
        pl.data_ptr<int64_t>(), sl.data_ptr<int64_t>(),
        ll.data_ptr<int64_t>(), fp.data_ptr<int64_t>(),
        out_ptr, bs, page_size);
  }

  copy_back_if_needed(out_indices, out_buf);
}

// ============================================================================
// alloc_decode_kernel_cpu
// ============================================================================
void alloc_decode_kernel_cpu(
    at::Tensor& seq_lens,
    at::Tensor& last_loc,
    at::Tensor& free_pages,
    at::Tensor& out_indices,
    int64_t page_size) {
  const int64_t bs = seq_lens.size(0);
  auto [out_ptr, out_buf] = prepare_out_i64(out_indices);

  if (seq_lens.scalar_type() == at::kInt) {
    auto sl = as_dtype(seq_lens, at::kInt);
    auto ll = as_dtype(last_loc, at::kInt);
    auto fp = as_dtype(free_pages, at::kInt);
    alloc_decode_kernel_impl<int32_t>(
        sl.data_ptr<int32_t>(), ll.data_ptr<int32_t>(),
        fp.data_ptr<int32_t>(), out_ptr, bs, page_size);
  } else {
    auto sl = as_dtype(seq_lens, at::kLong);
    auto ll = as_dtype(last_loc, at::kLong);
    auto fp = as_dtype(free_pages, at::kLong);
    alloc_decode_kernel_impl<int64_t>(
        sl.data_ptr<int64_t>(), ll.data_ptr<int64_t>(),
        fp.data_ptr<int64_t>(), out_ptr, bs, page_size);
  }

  copy_back_if_needed(out_indices, out_buf);
}
