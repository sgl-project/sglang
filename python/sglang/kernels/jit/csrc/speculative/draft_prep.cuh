#include <sgl_kernel/tensor.h>  // For SymbolicDevice, is_type
#include <sgl_kernel/utils.h>   // For div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

/// Row r, column c < cols: out_cache_loc[r * cols + c] = req_to_token[req_pool_indices[r]][seq_lens[r] + c];
/// column 0 also writes positions[r] = seq_lens[r] and, with mrope, mrope[a][r] for a < 3.
template <typename Len>
__global__ void draft_cache_locs_kernel(
    const int64_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ req_to_token,
    const Len* __restrict__ seq_lens,
    int64_t* __restrict__ out_cache_loc,
    Len* __restrict__ positions,
    int64_t* __restrict__ mrope,
    int64_t bs,
    int64_t cols,
    int64_t pool_len) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= bs * cols) return;
  const int64_t r = i / cols;
  const int64_t c = i - r * cols;
  const Len start = seq_lens[r];
  out_cache_loc[i] =
      static_cast<int64_t>(req_to_token[req_pool_indices[r] * pool_len + static_cast<int64_t>(start) + c]);
  if (c == 0) {
    positions[r] = start;
    if (mrope != nullptr) {
      for (int64_t a = 0; a < 3; ++a) {
        mrope[a * bs + r] = static_cast<int64_t>(start);
      }
    }
  }
}

/// Token i < bs * width belongs to request i / width at offset i % width of its window, which starts at
/// max(seq_lens, 0); entries i < bs also carry the per-request prefix, extend and post lengths and the start.
template <typename Len>
__global__ void draft_extend_layout_kernel(
    const Len* __restrict__ seq_lens,
    int64_t seq_stride,
    int32_t* __restrict__ prefix_lens,
    int32_t* __restrict__ extend_lens,
    Len* __restrict__ post_lens,
    int32_t* __restrict__ start_locs,
    int64_t* __restrict__ positions,
    int64_t* __restrict__ mrope,
    int64_t bs,
    int64_t width) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < bs) {
    const int64_t base = static_cast<int64_t>(seq_lens[i * seq_stride]);
    prefix_lens[i] = static_cast<int32_t>(base > 0 ? base : 0);
    extend_lens[i] = static_cast<int32_t>(width);
    post_lens[i] = static_cast<Len>(base + width);
    start_locs[i] = static_cast<int32_t>(i * width);
  }
  if (i < bs * width) {
    const int64_t req = i / width;
    const int64_t base = static_cast<int64_t>(seq_lens[req * seq_stride]);
    const int32_t prefix = static_cast<int32_t>(base > 0 ? base : 0);
    const int64_t position = static_cast<int64_t>(prefix + static_cast<int32_t>(i - req * width));
    positions[i] = position;
    if (mrope != nullptr) {
      for (int64_t a = 0; a < 3; ++a) {
        mrope[a * bs * width + i] = position;
      }
    }
  }
}

constexpr uint32_t kDraftPrepBlock = 128;

struct DraftPrep {
  /// \brief Draft-decode write slots and start positions of a topk-1 draft in one launch.
  /// \param req_pool_indices [bs] int64.
  /// \param req_to_token [rows, pool_len] int32, contiguous.
  /// \param seq_lens [bs] int32 or int64.
  /// \param out_cache_loc [bs * cols] int64 output.
  /// \param positions [bs] output with the dtype of seq_lens.
  /// \param mrope Optional [3, bs] int64 output, contiguous.
  /// \param cols Slots written per request.
  /// \return 0 after launching (or when bs is 0); 1 without launching when a dtype, shape or stride
  ///         is outside the supported layout, so the caller can take the generic path.
  static int64_t cache_locs(
      tvm::ffi::TensorView req_pool_indices,
      tvm::ffi::TensorView req_to_token,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView out_cache_loc,
      tvm::ffi::TensorView positions,
      tvm::ffi::Optional<tvm::ffi::TensorView> mrope,
      int64_t cols) {
    using namespace host;

    if (seq_lens.ndim() != 1 || cols < 1) return 1;
    const int64_t bs = seq_lens.size(0);
    const bool len64 = is_type<int64_t>(seq_lens.dtype());
    if (!len64 && !is_type<int32_t>(seq_lens.dtype())) return 1;
    if (!unit_vector(seq_lens, bs) || !unit_vector(positions, bs) || !(positions.dtype() == seq_lens.dtype())) return 1;
    if (!unit_vector(req_pool_indices, bs) || !is_type<int64_t>(req_pool_indices.dtype())) return 1;
    if (!unit_vector(out_cache_loc, bs * cols) || !is_type<int64_t>(out_cache_loc.dtype())) return 1;
    if (req_to_token.ndim() != 2 || !is_type<int32_t>(req_to_token.dtype()) || req_to_token.stride(1) != 1 ||
        req_to_token.stride(0) != req_to_token.size(1))
      return 1;
    if (mrope.has_value() && !rope_rows(mrope.value(), bs)) return 1;

    SymbolicDevice device_;
    device_.set_options<kDLCUDA, kDLROCM>();
    device_.verify(seq_lens.device());
    device_.verify(req_pool_indices.device());
    device_.verify(req_to_token.device());
    device_.verify(out_cache_loc.device());
    device_.verify(positions.device());
    if (mrope.has_value()) device_.verify(mrope.value().device());
    if (bs == 0) return 0;

    auto* rope = mrope.has_value() ? static_cast<int64_t*>(mrope.value().data_ptr()) : nullptr;
    const auto grid = static_cast<uint32_t>(div_ceil(bs * cols, int64_t{kDraftPrepBlock}));
    const auto* rows = static_cast<const int64_t*>(req_pool_indices.data_ptr());
    const auto* table = static_cast<const int32_t*>(req_to_token.data_ptr());
    auto* locs = static_cast<int64_t*>(out_cache_loc.data_ptr());
    const int64_t pool_len = req_to_token.size(1);
    if (len64) {
      LaunchKernel(grid, kDraftPrepBlock, device_.unwrap())(
          draft_cache_locs_kernel<int64_t>,
          rows,
          table,
          static_cast<const int64_t*>(seq_lens.data_ptr()),
          locs,
          static_cast<int64_t*>(positions.data_ptr()),
          rope,
          bs,
          cols,
          pool_len);
    } else {
      LaunchKernel(grid, kDraftPrepBlock, device_.unwrap())(
          draft_cache_locs_kernel<int32_t>,
          rows,
          table,
          static_cast<const int32_t*>(seq_lens.data_ptr()),
          locs,
          static_cast<int32_t*>(positions.data_ptr()),
          rope,
          bs,
          cols,
          pool_len);
    }
    return 0;
  }

  /// \brief Draft-extend layout (prefix, extend and post lengths, window starts, positions, optional mrope)
  ///        for a window of `width` tokens per request without front rows, in one launch.
  /// \param seq_lens [bs] int32 or int64, any stride.
  /// \param prefix_lens [bs] int32 output.
  /// \param extend_lens [bs] int32 output.
  /// \param post_lens [bs] output with the dtype of seq_lens.
  /// \param start_locs [bs] int32 output.
  /// \param positions [bs * width] int64 output.
  /// \param mrope Optional [3, bs * width] int64 output, contiguous.
  /// \param width Window tokens per request.
  /// \return 0 after launching (or when bs is 0); 1 without launching when a dtype, shape or stride
  ///         is outside the supported layout, so the caller can take the generic path.
  static int64_t extend_layout(
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView prefix_lens,
      tvm::ffi::TensorView extend_lens,
      tvm::ffi::TensorView post_lens,
      tvm::ffi::TensorView start_locs,
      tvm::ffi::TensorView positions,
      tvm::ffi::Optional<tvm::ffi::TensorView> mrope,
      int64_t width) {
    using namespace host;

    if (seq_lens.ndim() != 1 || width < 1) return 1;
    const int64_t bs = seq_lens.size(0);
    const bool len64 = is_type<int64_t>(seq_lens.dtype());
    if (!len64 && !is_type<int32_t>(seq_lens.dtype())) return 1;
    if (!unit_vector(post_lens, bs) || !(post_lens.dtype() == seq_lens.dtype())) return 1;
    for (const auto& t : {prefix_lens, extend_lens, start_locs}) {
      if (!unit_vector(t, bs) || !is_type<int32_t>(t.dtype())) return 1;
    }
    if (!unit_vector(positions, bs * width) || !is_type<int64_t>(positions.dtype())) return 1;
    if (mrope.has_value() && !rope_rows(mrope.value(), bs * width)) return 1;

    SymbolicDevice device_;
    device_.set_options<kDLCUDA, kDLROCM>();
    device_.verify(seq_lens.device());
    for (const auto& t : {prefix_lens, extend_lens, post_lens, start_locs, positions}) {
      device_.verify(t.device());
    }
    if (mrope.has_value()) device_.verify(mrope.value().device());
    if (bs == 0) return 0;

    auto* rope = mrope.has_value() ? static_cast<int64_t*>(mrope.value().data_ptr()) : nullptr;
    const auto grid = static_cast<uint32_t>(div_ceil(bs * width, int64_t{kDraftPrepBlock}));
    auto* prefix = static_cast<int32_t*>(prefix_lens.data_ptr());
    auto* extend = static_cast<int32_t*>(extend_lens.data_ptr());
    auto* starts = static_cast<int32_t*>(start_locs.data_ptr());
    auto* pos = static_cast<int64_t*>(positions.data_ptr());
    const int64_t stride = seq_lens.stride(0);
    if (len64) {
      LaunchKernel(grid, kDraftPrepBlock, device_.unwrap())(
          draft_extend_layout_kernel<int64_t>,
          static_cast<const int64_t*>(seq_lens.data_ptr()),
          stride,
          prefix,
          extend,
          static_cast<int64_t*>(post_lens.data_ptr()),
          starts,
          pos,
          rope,
          bs,
          width);
    } else {
      LaunchKernel(grid, kDraftPrepBlock, device_.unwrap())(
          draft_extend_layout_kernel<int32_t>,
          static_cast<const int32_t*>(seq_lens.data_ptr()),
          stride,
          prefix,
          extend,
          static_cast<int32_t*>(post_lens.data_ptr()),
          starts,
          pos,
          rope,
          bs,
          width);
    }
    return 0;
  }

 private:
  /// A 1-D tensor of n elements with unit stride (any stride when n <= 1).
  static bool unit_vector(const tvm::ffi::TensorView& t, int64_t n) {
    return t.ndim() == 1 && t.size(0) == n && (n <= 1 || t.stride(0) == 1);
  }

  /// A contiguous [3, n] int64 tensor.
  static bool rope_rows(const tvm::ffi::TensorView& t, int64_t n) {
    return t.ndim() == 2 && t.size(0) == 3 && t.size(1) == n && host::is_type<int64_t>(t.dtype()) &&
           (n == 0 || ((t.stride(1) == 1 || n == 1) && t.stride(0) == n));
  }
};

}  // namespace sglang
