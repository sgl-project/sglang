#include <sgl_kernel/tensor.h>  // For SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace sglang {

/// Upper bound on the fields one launch copies; the parameter block is passed by value.
constexpr int32_t kSmallFieldCopyMaxFields = 16;

/// Element kinds: raw copies of 1/2/4/8-byte elements, and the two integer width casts.
enum SmallFieldCopyKind : int32_t {
  kCopy8 = 0,
  kCopy16 = 1,
  kCopy32 = 2,
  kCopy64 = 3,
  kInt32ToInt64 = 4,
  kInt64ToInt32 = 5,
};

/// Number of int64 entries describing one field in the packed descriptor array.
constexpr int64_t kSmallFieldCopyEntries = 7;

struct SmallFieldCopyParams {
  const void* src[kSmallFieldCopyMaxFields];
  void* dst[kSmallFieldCopyMaxFields];
  int64_t rows[kSmallFieldCopyMaxFields];
  int64_t cols[kSmallFieldCopyMaxFields];
  int64_t src_row_stride[kSmallFieldCopyMaxFields];
  int64_t dst_row_stride[kSmallFieldCopyMaxFields];
  int32_t kind[kSmallFieldCopyMaxFields];
};

/// \brief Copies up to kSmallFieldCopyMaxFields small row-major fields in one launch.
///
/// blockIdx.y selects the field; elements are visited in row-major order with unit
/// column stride on both sides.
__global__ void small_field_copy_kernel(const __grid_constant__ SmallFieldCopyParams params) {
  const int32_t field = static_cast<int32_t>(blockIdx.y);
  const int64_t cols = params.cols[field];
  const int64_t n = params.rows[field] * cols;
  const int64_t src_stride = params.src_row_stride[field];
  const int64_t dst_stride = params.dst_row_stride[field];
  const int32_t kind = params.kind[field];
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = i / cols;
    const int64_t col = i - row * cols;
    const int64_t s = row * src_stride + col;
    const int64_t d = row * dst_stride + col;
    switch (kind) {
      case kCopy8:
        static_cast<uint8_t*>(params.dst[field])[d] = static_cast<const uint8_t*>(params.src[field])[s];
        break;
      case kCopy16:
        static_cast<uint16_t*>(params.dst[field])[d] = static_cast<const uint16_t*>(params.src[field])[s];
        break;
      case kCopy32:
        static_cast<uint32_t*>(params.dst[field])[d] = static_cast<const uint32_t*>(params.src[field])[s];
        break;
      case kCopy64:
        static_cast<uint64_t*>(params.dst[field])[d] = static_cast<const uint64_t*>(params.src[field])[s];
        break;
      case kInt32ToInt64:
        static_cast<int64_t*>(params.dst[field])[d] =
            static_cast<int64_t>(static_cast<const int32_t*>(params.src[field])[s]);
        break;
      case kInt64ToInt32:
        static_cast<int32_t*>(params.dst[field])[d] =
            static_cast<int32_t>(static_cast<const int64_t*>(params.src[field])[s]);
        break;
      default:
        break;
    }
  }
}

/// Internal kind of the vectorized launch: a raw field moved in 16-byte units.
constexpr int32_t kCopy128 = 6;

struct SmallFieldCopyVecParams {
  const void* src[kSmallFieldCopyMaxFields];
  void* dst[kSmallFieldCopyMaxFields];
  int64_t rows[kSmallFieldCopyMaxFields];
  int64_t cols[kSmallFieldCopyMaxFields];
  int64_t src_row_stride[kSmallFieldCopyMaxFields];
  int64_t dst_row_stride[kSmallFieldCopyMaxFields];
  int32_t kind[kSmallFieldCopyMaxFields];
  int32_t block_start[kSmallFieldCopyMaxFields + 1];
  int32_t num_fields;
};

/// \brief Same copies as small_field_copy_kernel, with each field owning its own block range
///        (sized to the field) and aligned raw fields moved in 16-byte units.
__global__ void small_field_copy_vec_kernel(const __grid_constant__ SmallFieldCopyVecParams params) {
  const int32_t block = static_cast<int32_t>(blockIdx.x);
  int32_t field = 0;
  while (field + 1 < params.num_fields && block >= params.block_start[field + 1]) {
    ++field;
  }
  const int64_t rows = params.rows[field];
  const int64_t cols = params.cols[field];
  const int64_t n = rows * cols;
  const int64_t src_stride = params.src_row_stride[field];
  const int64_t dst_stride = params.dst_row_stride[field];
  const int32_t kind = params.kind[field];
  const int64_t stride = static_cast<int64_t>(params.block_start[field + 1] - params.block_start[field]) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(block - params.block_start[field]) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const int64_t row = rows == 1 ? 0 : i / cols;
    const int64_t col = i - row * cols;
    const int64_t s = row * src_stride + col;
    const int64_t d = row * dst_stride + col;
    switch (kind) {
      case kCopy8:
        static_cast<uint8_t*>(params.dst[field])[d] = static_cast<const uint8_t*>(params.src[field])[s];
        break;
      case kCopy16:
        static_cast<uint16_t*>(params.dst[field])[d] = static_cast<const uint16_t*>(params.src[field])[s];
        break;
      case kCopy32:
        static_cast<uint32_t*>(params.dst[field])[d] = static_cast<const uint32_t*>(params.src[field])[s];
        break;
      case kCopy64:
        static_cast<uint64_t*>(params.dst[field])[d] = static_cast<const uint64_t*>(params.src[field])[s];
        break;
      case kInt32ToInt64:
        static_cast<int64_t*>(params.dst[field])[d] =
            static_cast<int64_t>(static_cast<const int32_t*>(params.src[field])[s]);
        break;
      case kInt64ToInt32:
        static_cast<int32_t*>(params.dst[field])[d] =
            static_cast<int32_t>(static_cast<const int64_t*>(params.src[field])[s]);
        break;
      case kCopy128:
        static_cast<uint4*>(params.dst[field])[d] = static_cast<const uint4*>(params.src[field])[s];
        break;
      default:
        break;
    }
  }
}

constexpr int64_t kSmallFieldCopyBlock = 256;
constexpr int64_t kSmallFieldCopyMaxBlocks = 32;
// Per-field block cap of the vectorized launch; arbitrary, sized for ~0.5 MB fields in one pass.
constexpr int64_t kSmallFieldCopyVecMaxBlocks = 128;

struct SmallFieldCopy {
  /// \brief Copies every field described in `desc` with one launch on the current stream.
  /// \param anchor Any CUDA tensor on the target device; selects the device and its stream.
  /// \param desc Per field: src_ptr, dst_ptr, rows, cols, src_row_stride, dst_row_stride, kind
  ///             (strides in elements; unit column stride on both sides).
  /// \return 0 after launching (or when there is nothing to copy); 1 without launching when two
  ///         different fields overlap, so the caller can fall back to ordered copies.
  static int64_t run(tvm::ffi::TensorView anchor, tvm::ffi::Array<int64_t> desc) {
    using namespace host;

    const int64_t num_entries = static_cast<int64_t>(desc.size());
    CHECK_HOST(num_entries <= kSmallFieldCopyMaxFields * kSmallFieldCopyEntries)
        << "too many fields " << num_entries / kSmallFieldCopyEntries;
    int64_t entries[kSmallFieldCopyMaxFields * kSmallFieldCopyEntries];
    for (int64_t i = 0; i < num_entries; ++i) {
      entries[i] = desc[i];
    }
    return launch(anchor, entries, num_entries, false);
  }

  /// \brief Same as `run`, with the descriptor in a host int64 tensor of 7 entries per field.
  static int64_t run_host_desc(tvm::ffi::TensorView anchor, tvm::ffi::TensorView desc) {
    return launch(anchor, host_desc(desc), desc.size(0), false);
  }

  /// \brief `run` through small_field_copy_vec_kernel: same bytes, same refusals.
  static int64_t run_vec(tvm::ffi::TensorView anchor, tvm::ffi::Array<int64_t> desc) {
    using namespace host;

    const int64_t num_entries = static_cast<int64_t>(desc.size());
    CHECK_HOST(num_entries <= kSmallFieldCopyMaxFields * kSmallFieldCopyEntries)
        << "too many fields " << num_entries / kSmallFieldCopyEntries;
    int64_t entries[kSmallFieldCopyMaxFields * kSmallFieldCopyEntries];
    for (int64_t i = 0; i < num_entries; ++i) {
      entries[i] = desc[i];
    }
    return launch(anchor, entries, num_entries, true);
  }

  /// \brief `run_host_desc` through small_field_copy_vec_kernel.
  static int64_t run_host_desc_vec(tvm::ffi::TensorView anchor, tvm::ffi::TensorView desc) {
    return launch(anchor, host_desc(desc), desc.size(0), true);
  }

 private:
  static const int64_t* host_desc(tvm::ffi::TensorView desc) {
    using namespace host;

    CHECK_HOST(
        desc.device().device_type == kDLCPU && is_type<int64_t>(desc.dtype()) && desc.ndim() == 1 &&
        (desc.size(0) <= 1 || desc.stride(0) == 1))
        << "descriptor must be a contiguous host int64 vector";
    return static_cast<const int64_t*>(desc.data_ptr());
  }

  /// Moves a validated field to the vectorized layout: a contiguous field becomes one row, and
  /// a raw field whose rows, strides and base addresses are 16-byte aligned moves in uint4 units.
  static void vectorize(SmallFieldCopyVecParams& p, int32_t i, int64_t elem_size) {
    if (p.rows[i] > 1 && p.src_row_stride[i] == p.cols[i] && p.dst_row_stride[i] == p.cols[i]) {
      p.cols[i] *= p.rows[i];
      p.rows[i] = 1;
      p.src_row_stride[i] = p.dst_row_stride[i] = p.cols[i];
    }
    if (p.kind[i] > kCopy64) return;
    const int64_t row_bytes = p.cols[i] * elem_size;
    const int64_t src_bytes = p.src_row_stride[i] * elem_size;
    const int64_t dst_bytes = p.dst_row_stride[i] * elem_size;
    const bool aligned = row_bytes % 16 == 0 && reinterpret_cast<uintptr_t>(p.src[i]) % 16 == 0 &&
                         reinterpret_cast<uintptr_t>(p.dst[i]) % 16 == 0 &&
                         (p.rows[i] == 1 || (src_bytes % 16 == 0 && dst_bytes % 16 == 0));
    if (!aligned) return;
    p.kind[i] = kCopy128;
    p.cols[i] = row_bytes / 16;
    p.src_row_stride[i] = p.rows[i] == 1 ? p.cols[i] : src_bytes / 16;
    p.dst_row_stride[i] = p.rows[i] == 1 ? p.cols[i] : dst_bytes / 16;
  }

  static int64_t launch(tvm::ffi::TensorView anchor, const int64_t* desc, int64_t num_entries, bool vec) {
    using namespace host;

    SymbolicDevice device_;
    device_.set_options<kDLGPU, kDLROCM>();
    device_.verify(anchor.device());

    CHECK_HOST(num_entries % kSmallFieldCopyEntries == 0) << "bad descriptor length " << num_entries;
    const int64_t num_described = num_entries / kSmallFieldCopyEntries;
    CHECK_HOST(num_described <= kSmallFieldCopyMaxFields) << "too many fields " << num_described;

    SmallFieldCopyParams params{};
    uintptr_t src_begin[kSmallFieldCopyMaxFields], src_end[kSmallFieldCopyMaxFields];
    uintptr_t dst_begin[kSmallFieldCopyMaxFields], dst_end[kSmallFieldCopyMaxFields];
    int32_t num_fields = 0;
    int64_t largest = 0;
    for (int64_t f = 0; f < num_described; ++f) {
      const int64_t base = f * kSmallFieldCopyEntries;
      const auto src = static_cast<uintptr_t>(desc[base + 0]);
      const auto dst = static_cast<uintptr_t>(desc[base + 1]);
      const int64_t rows = desc[base + 2];
      const int64_t cols = desc[base + 3];
      const int64_t src_stride = desc[base + 4];
      const int64_t dst_stride = desc[base + 5];
      const auto kind = static_cast<int32_t>(desc[base + 6]);
      CHECK_HOST(kind >= kCopy8 && kind <= kInt64ToInt32) << "bad kind " << kind;
      CHECK_HOST(rows >= 0 && cols >= 0) << "bad shape " << rows << "x" << cols;
      if (rows == 0 || cols == 0) continue;
      const bool raw = kind <= kCopy64;
      if (raw && src == dst && src_stride == dst_stride) continue;  // copy onto itself
      const int64_t src_size = kind == kInt64ToInt32 ? 8 : kind == kInt32ToInt64 ? 4 : (int64_t{1} << kind);
      const int64_t dst_size = kind == kInt32ToInt64 ? 8 : kind == kInt64ToInt32 ? 4 : (int64_t{1} << kind);
      const int32_t i = num_fields++;
      params.src[i] = reinterpret_cast<const void*>(src);
      params.dst[i] = reinterpret_cast<void*>(dst);
      params.rows[i] = rows;
      params.cols[i] = cols;
      params.src_row_stride[i] = src_stride;
      params.dst_row_stride[i] = dst_stride;
      params.kind[i] = kind;
      src_begin[i] = src;
      src_end[i] = src + static_cast<uintptr_t>(((rows - 1) * src_stride + cols) * src_size);
      dst_begin[i] = dst;
      dst_end[i] = dst + static_cast<uintptr_t>(((rows - 1) * dst_stride + cols) * dst_size);
      largest = rows * cols > largest ? rows * cols : largest;
    }
    // Fields run concurrently: a write that lands on another field's read or write would make
    // the result depend on block order.
    for (int32_t i = 0; i < num_fields; ++i) {
      for (int32_t j = 0; j < num_fields; ++j) {
        if (i != j && dst_begin[i] < src_end[j] && src_begin[j] < dst_end[i]) return 1;
        if (i < j && dst_begin[i] < dst_end[j] && dst_begin[j] < dst_end[i]) return 1;
      }
      if (dst_begin[i] < src_end[i] && src_begin[i] < dst_end[i]) return 1;
    }
    if (num_fields == 0) return 0;

    if (vec) {
      SmallFieldCopyVecParams vp{};
      vp.num_fields = num_fields;
      for (int32_t i = 0; i < num_fields; ++i) {
        vp.src[i] = params.src[i];
        vp.dst[i] = params.dst[i];
        vp.rows[i] = params.rows[i];
        vp.cols[i] = params.cols[i];
        vp.src_row_stride[i] = params.src_row_stride[i];
        vp.dst_row_stride[i] = params.dst_row_stride[i];
        vp.kind[i] = params.kind[i];
        vectorize(vp, i, params.kind[i] <= kCopy64 ? (int64_t{1} << params.kind[i]) : 0);
        const int64_t units = vp.rows[i] * vp.cols[i];
        const int64_t blocks = div_ceil(units, kSmallFieldCopyBlock);
        vp.block_start[i + 1] =
            vp.block_start[i] +
            static_cast<int32_t>(blocks < kSmallFieldCopyVecMaxBlocks ? blocks : kSmallFieldCopyVecMaxBlocks);
      }
      LaunchKernel(
          dim3(static_cast<uint32_t>(vp.block_start[num_fields])),
          static_cast<uint32_t>(kSmallFieldCopyBlock),
          device_.unwrap())(small_field_copy_vec_kernel, vp);
      return 0;
    }

    // Every field gets the same block count; the grid-stride loop covers larger fields, so a
    // small cap keeps the idle blocks of the small fields cheap. Tunable, not measured.
    const int64_t blocks = div_ceil(largest, kSmallFieldCopyBlock);
    const dim3 grid(
        static_cast<uint32_t>(blocks < kSmallFieldCopyMaxBlocks ? blocks : kSmallFieldCopyMaxBlocks),
        static_cast<uint32_t>(num_fields));
    LaunchKernel(grid, static_cast<uint32_t>(kSmallFieldCopyBlock), device_.unwrap())(small_field_copy_kernel, params);
    return 0;
  }
};

}  // namespace sglang
