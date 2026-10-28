// MiniMax-H3 VAE output: temporal window blend + write, and ImageNet
// de-normalization + clamp.
//
// Every arithmetic step is the eager klvae chain spelled with explicit
// round-to-nearest intrinsics -- blend(a, b, k, n) = a * (1 - w) + b * w with
// w = k * rcp(n), six separately rounded fp32 ops (two for the weight, four
// for the blend); (x - mean) / std as two -- so both ops are bit-exact against
// it, on the strided and the vectorized path alike.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace sglang {

namespace minimax_h3_vae_output {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kVecWidth = 4;  // fp32 lanes per 128-bit load
constexpr int64_t kVecAlignment = kVecWidth * sizeof(float);
constexpr int64_t kIndexMax = std::numeric_limits<uint32_t>::max();

/// \brief Sizes and element strides of a view with `kDims` dimensions.
template <int kDims>
struct Layout {
  int64_t strides[kDims];
  uint32_t sizes[kDims];
};

/// \brief De-normalization constants of the three RGB channels.
struct ChannelStats {
  float mean[3];
  float std[3];
};

/// \brief klvae.blend's linear ramp, a * (1 - w) + b * w, each op rounded once.
///
/// The eager weight is `positions / extent` on a CUDA tensor with a Python int
/// divisor, which aten's div_true_kernel_cuda evaluates as k * (1 / n) with the
/// reciprocal rounded first; a correctly rounded k / n differs by one ulp for
/// extents such as 7, 12 or 15.
SGL_DEVICE float blend_rn(float a, float b, uint32_t k, uint32_t n) {
  const float weight = __fmul_rn(static_cast<float>(k), __frcp_rn(static_cast<float>(n)));
  return __fadd_rn(__fmul_rn(a, __fsub_rn(1.f, weight)), __fmul_rn(b, weight));
}

/// \brief torchvision Normalize followed by clamp(0, 1); NaN passes through as in aten's clamp.
SGL_DEVICE float denorm_clamp_rn(float value, float mean, float std) {
  const float scaled = __fdiv_rn(__fsub_rn(value, mean), std);
  return scaled < 0.f ? 0.f : (scaled > 1.f ? 1.f : scaled);
}

SGL_DEVICE int64_t element_offset(const Layout<5>& layout, uint32_t b, uint32_t c, uint32_t t, uint32_t y, uint32_t x) {
  return b * layout.strides[0] + c * layout.strides[1] + t * layout.strides[2] + y * layout.strides[3] +
         x * layout.strides[4];
}

template <int kDims>
inline Layout<kDims> make_layout(tvm::ffi::TensorView view) {
  Layout<kDims> layout{};
  for (int d = 0; d < kDims; ++d) {
    layout.strides[d] = view.stride(d);
    layout.sizes[d] = static_cast<uint32_t>(view.size(d));
  }
  return layout;
}

/// \brief Byte span a view can touch; an empty view maps to an empty range.
struct ByteRange {
  uintptr_t begin;
  uintptr_t end;
};

inline ByteRange byte_range(tvm::ffi::TensorView view) {
  int64_t last = 0;
  for (int d = 0; d < view.dim(); ++d) {
    if (view.size(d) == 0) return ByteRange{0, 0};
    last += (view.size(d) - 1) * view.stride(d);
  }
  const auto begin = reinterpret_cast<uintptr_t>(view.data_ptr());
  const auto bytes = static_cast<uintptr_t>(host::dtype_bytes(view.dtype()));
  return ByteRange{begin, begin + static_cast<uintptr_t>(last) * bytes + bytes};
}

/// \brief Byte span of frames `[start, start + count)` of a [B, C, T, H, W] view.
inline ByteRange frame_byte_range(tvm::ffi::TensorView view, int64_t start, int64_t count) {
  if (count == 0) return ByteRange{0, 0};
  int64_t last = (count - 1) * view.stride(2);
  for (int d = 0; d < 5; ++d) {
    if (d == 2) continue;
    if (view.size(d) == 0) return ByteRange{0, 0};
    last += (view.size(d) - 1) * view.stride(d);
  }
  const auto bytes = static_cast<uintptr_t>(host::dtype_bytes(view.dtype()));
  const auto begin =
      reinterpret_cast<uintptr_t>(view.data_ptr()) + static_cast<uintptr_t>(start * view.stride(2)) * bytes;
  return ByteRange{begin, begin + static_cast<uintptr_t>(last) * bytes + bytes};
}

inline bool overlaps(ByteRange a, ByteRange b) {
  return a.begin < b.end && b.begin < a.end;
}

/**
 * \brief Write `count` frames of `part` to `out[start : start + count]`, blending
 *        the first `extent` frames with the trailing `extent` frames of `overlap`
 *        on klvae's linear ramp. Grid is 1-D over the written elements.
 *
 * \tparam kVectorized Innermost stride 1, 16-byte aligned tensors and `W % 4 == 0`:
 *                     128-bit lanes.
 * \param part [B, C, Tp, H, W] fp32 window to write.
 * \param overlap [B, C, To, H, W] fp32 previous window; read only when `extent > 0`.
 * \param out [B, C, Tout, H, W] fp32 destination; frames outside the window stay untouched.
 * \param part_layout Sizes and strides of `part`.
 * \param overlap_layout Sizes and strides of `overlap`.
 * \param out_layout Sizes and strides of `out`.
 * \param start First destination frame.
 * \param count Frames written.
 * \param extent Frames blended, at most `min(Tp, To)`; 0 disables the blend.
 * \param total Vectors written: B * C * count * H * W / width.
 */
template <bool kVectorized>
__global__ void temporal_blend_write_kernel(
    const float* __restrict__ part,
    const float* __restrict__ overlap,
    float* __restrict__ out,
    const Layout<5> __grid_constant__ part_layout,
    const Layout<5> __grid_constant__ overlap_layout,
    const Layout<5> __grid_constant__ out_layout,
    uint32_t start,
    uint32_t count,
    uint32_t extent,
    uint32_t total) {
  constexpr uint32_t kWidth = kVectorized ? kVecWidth : 1;
  using Vec = device::AlignedVector<float, kWidth>;

  uint32_t index = blockIdx.x * kBlockSize + threadIdx.x;
  if (index >= total) return;
  const uint32_t cols = part_layout.sizes[4] / kWidth;
  const uint32_t x = (index % cols) * kWidth;
  index /= cols;
  const uint32_t y = index % part_layout.sizes[3];
  index /= part_layout.sizes[3];
  const uint32_t t = index % count;
  index /= count;
  const uint32_t c = index % part_layout.sizes[1];
  const uint32_t b = index / part_layout.sizes[1];

  Vec value;
  value.load(part + element_offset(part_layout, b, c, t, y, x));
  if (t < extent) {
    Vec previous;
    previous.load(overlap + element_offset(overlap_layout, b, c, overlap_layout.sizes[2] - extent + t, y, x));
#pragma unroll
    for (uint32_t e = 0; e < kWidth; ++e) {
      value[e] = blend_rn(previous[e], value[e], t, extent);
    }
  }
  value.store(out + element_offset(out_layout, b, c, start + t, y, x));
}

/**
 * \brief out = clamp((value - mean[c]) / std[c], 0, 1) over [B, 3, T, H, W];
 *        NaN propagates. Grid is 1-D over the contiguous output.
 *
 * \tparam kVectorized Innermost stride 1, 16-byte aligned tensors and `W % 4 == 0`:
 *                     128-bit lanes.
 * \param value [B, 3, T, H, W] fp32; any strides on the scalar path.
 * \param out Contiguous [B, 3, T, H, W] fp32.
 * \param in_layout Sizes and strides of `value`.
 * \param stats Per-channel mean and std.
 * \param total Vectors written: numel / width.
 */
template <bool kVectorized>
__global__ void denorm_clamp_kernel(
    const float* value,  // may be `out` itself (contiguous, in place)
    float* out,
    const Layout<5> __grid_constant__ in_layout,
    const ChannelStats __grid_constant__ stats,
    uint32_t total) {
  constexpr uint32_t kWidth = kVectorized ? kVecWidth : 1;
  using Vec = device::AlignedVector<float, kWidth>;

  const uint32_t index = blockIdx.x * kBlockSize + threadIdx.x;
  if (index >= total) return;
  uint32_t rest = index;
  const uint32_t cols = in_layout.sizes[4] / kWidth;
  const uint32_t x = (rest % cols) * kWidth;
  rest /= cols;
  const uint32_t y = rest % in_layout.sizes[3];
  rest /= in_layout.sizes[3];
  const uint32_t t = rest % in_layout.sizes[2];
  rest /= in_layout.sizes[2];
  const uint32_t c = rest % 3;
  const uint32_t b = rest / 3;

  Vec pixels;
  pixels.load(value + element_offset(in_layout, b, c, t, y, x));
#pragma unroll
  for (uint32_t e = 0; e < kWidth; ++e) {
    pixels[e] = denorm_clamp_rn(pixels[e], stats.mean[c], stats.std[c]);
  }
  pixels.store(out + static_cast<int64_t>(index) * kWidth);
}

/**
 * \brief Validate the window, overlap and destination, then launch
 *        `temporal_blend_write_kernel`.
 *
 * \tparam kVectorized Chosen by the Python wrapper; enforced here.
 */
template <bool kVectorized>
struct TemporalBlendWriteKernel {
  /**
   * \param part [B, C, Tp, H, W] fp32 on CUDA.
   * \param overlap [B, C, To, H, W] fp32 on CUDA; ignored when `blend_extent == 0`
   *                (the wrapper passes `part` in that case).
   * \param out [B, C, Tout, H, W] fp32 on CUDA; must not alias `part` or `overlap`.
   * \param start First destination frame.
   * \param count Frames written, `count <= Tp` and `start + count <= Tout`.
   * \param blend_extent Frames blended, `blend_extent <= min(Tp, To)`.
   */
  static void
  run(tvm::ffi::TensorView part,
      tvm::ffi::TensorView overlap,
      tvm::ffi::TensorView out,
      int64_t start,
      int64_t count,
      int64_t blend_extent) {
    using namespace host;

    auto B = SymbolicSize{"batch"};
    auto C = SymbolicSize{"channels"};
    auto Tp = SymbolicSize{"part_frames"};
    auto To = SymbolicSize{"overlap_frames"};
    auto Tout = SymbolicSize{"out_frames"};
    auto H = SymbolicSize{"height"};
    auto W = SymbolicSize{"width"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    if constexpr (kVectorized) {
      TensorMatcher({B, C, Tp, H, W})
          .with_strides({-1, -1, -1, -1, 1})
          .with_dtype<fp32_t>()
          .with_device(device)
          .ensure_alignment(kVecAlignment)
          .verify(part);
      TensorMatcher({B, C, To, H, W})
          .with_strides({-1, -1, -1, -1, 1})
          .with_dtype<fp32_t>()
          .with_device(device)
          .ensure_alignment(kVecAlignment)
          .verify(overlap);
      TensorMatcher({B, C, Tout, H, W})
          .with_strides({-1, -1, -1, -1, 1})
          .with_dtype<fp32_t>()
          .with_device(device)
          .ensure_alignment(kVecAlignment)
          .verify(out);
    } else {
      TensorMatcher({B, C, Tp, H, W})
          .with_strides({-1, -1, -1, -1, -1})
          .with_dtype<fp32_t>()
          .with_device(device)
          .verify(part);
      TensorMatcher({B, C, To, H, W})
          .with_strides({-1, -1, -1, -1, -1})
          .with_dtype<fp32_t>()
          .with_device(device)
          .verify(overlap);
      TensorMatcher({B, C, Tout, H, W})
          .with_strides({-1, -1, -1, -1, -1})
          .with_dtype<fp32_t>()
          .with_device(device)
          .verify(out);
    }

    const int64_t part_frames = Tp.unwrap();
    const int64_t overlap_frames = To.unwrap();
    const int64_t out_frames = Tout.unwrap();
    const int64_t width = W.unwrap();
    CHECK_HOST(start >= 0 && count >= 0 && blend_extent >= 0)
        << "temporal_blend_write: start, count and blend_extent must be non-negative";
    CHECK_HOST(count <= part_frames) << "temporal_blend_write: count=" << count << " exceeds the " << part_frames
                                     << " frames of part";
    CHECK_HOST(start + count <= out_frames) << "temporal_blend_write: frames [" << start << ", " << start + count
                                            << ") exceed the " << out_frames << " frames of out";
    CHECK_HOST(blend_extent <= overlap_frames && blend_extent <= part_frames)
        << "temporal_blend_write: blend_extent=" << blend_extent << " exceeds min(" << part_frames << ", "
        << overlap_frames << ")";
    CHECK_HOST(
        B.unwrap() <= kIndexMax && C.unwrap() <= kIndexMax && out_frames <= kIndexMax && H.unwrap() <= kIndexMax &&
        width <= kIndexMax)
        << "temporal_blend_write: dimensions exceed uint32 indexing";
    if constexpr (kVectorized) {
      CHECK_HOST(width % kVecWidth == 0) << "temporal_blend_write: vectorized path needs W % 4 == 0, got W=" << width;
    }
    // Only the frames the kernel touches count: the read window of part, the
    // trailing blend_extent frames of overlap, the written window of out.
    const ByteRange written = frame_byte_range(out, start, count);
    CHECK_HOST(!overlaps(written, frame_byte_range(part, 0, count))) << "temporal_blend_write: out must not alias part";
    CHECK_HOST(!overlaps(written, frame_byte_range(overlap, overlap_frames - blend_extent, blend_extent)))
        << "temporal_blend_write: out must not alias overlap";

    constexpr uint32_t kWidth = kVectorized ? kVecWidth : 1;
    const int64_t total = B.unwrap() * C.unwrap() * count * H.unwrap() * width / kWidth;
    if (total == 0) return;
    CHECK_HOST(total <= kIndexMax) << "temporal_blend_write: work items exceed uint32 indexing";
    LaunchKernel(static_cast<uint32_t>(div_ceil(total, static_cast<int64_t>(kBlockSize))), kBlockSize, device.unwrap())(
        temporal_blend_write_kernel<kVectorized>,
        static_cast<const float*>(part.data_ptr()),
        static_cast<const float*>(overlap.data_ptr()),
        static_cast<float*>(out.data_ptr()),
        make_layout<5>(part),
        make_layout<5>(overlap),
        make_layout<5>(out),
        static_cast<uint32_t>(start),
        static_cast<uint32_t>(count),
        static_cast<uint32_t>(blend_extent),
        static_cast<uint32_t>(total));
  }
};

/**
 * \brief Validate the RGB tensor and its output, then launch `denorm_clamp_kernel`.
 *
 * \tparam kVectorized Chosen by the Python wrapper; enforced here.
 */
template <bool kVectorized>
struct DenormClampKernel {
  /**
   * \param value [B, 3, T, H, W] fp32 on CUDA; any strides on the scalar path.
   * \param out Contiguous [B, 3, T, H, W] fp32 on CUDA; either disjoint from
   *            `value` or `value` itself (then `value` must be contiguous, so
   *            every thread reads and writes the same elements).
   * \param mean0 Channel 0 mean (subtracted first).
   * \param mean1 Channel 1 mean.
   * \param mean2 Channel 2 mean.
   * \param std0 Channel 0 std (divided second).
   * \param std1 Channel 1 std.
   * \param std2 Channel 2 std.
   */
  static void
  run(tvm::ffi::TensorView value,
      tvm::ffi::TensorView out,
      double mean0,
      double mean1,
      double mean2,
      double std0,
      double std1,
      double std2) {
    using namespace host;

    auto B = SymbolicSize{"batch"};
    auto T = SymbolicSize{"time"};
    auto H = SymbolicSize{"height"};
    auto W = SymbolicSize{"width"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    if constexpr (kVectorized) {
      TensorMatcher({B, 3, T, H, W})
          .with_strides({-1, -1, -1, -1, 1})
          .with_dtype<fp32_t>()
          .with_device(device)
          .ensure_alignment(kVecAlignment)
          .verify(value);
      TensorMatcher({B, 3, T, H, W})
          .with_dtype<fp32_t>()
          .with_device(device)
          .ensure_alignment(kVecAlignment)
          .verify(out);
    } else {
      TensorMatcher({B, 3, T, H, W})
          .with_strides({-1, -1, -1, -1, -1})
          .with_dtype<fp32_t>()
          .with_device(device)
          .verify(value);
      TensorMatcher({B, 3, T, H, W}).with_dtype<fp32_t>().with_device(device).verify(out);
    }

    const int64_t width = W.unwrap();
    CHECK_HOST(B.unwrap() <= kIndexMax && T.unwrap() <= kIndexMax && H.unwrap() <= kIndexMax && width <= kIndexMax)
        << "denorm_clamp: dimensions exceed uint32 indexing";
    if constexpr (kVectorized) {
      CHECK_HOST(width % kVecWidth == 0) << "denorm_clamp: vectorized path needs W % 4 == 0, got W=" << width;
    }
    if (out.data_ptr() == value.data_ptr()) {
      const auto layout = make_layout<5>(value);
      int64_t expected = 1;
      bool contiguous = true;
      for (int d = 4; d >= 0; --d) {
        contiguous = contiguous && (layout.sizes[d] == 1 || layout.strides[d] == expected);
        expected *= layout.sizes[d];
      }
      CHECK_HOST(contiguous) << "denorm_clamp: in-place value must be contiguous";
    } else {
      CHECK_HOST(!overlaps(byte_range(out), byte_range(value))) << "denorm_clamp: out must not alias value";
    }

    constexpr uint32_t kWidth = kVectorized ? kVecWidth : 1;
    const int64_t total = B.unwrap() * 3 * T.unwrap() * H.unwrap() * width / kWidth;
    if (total == 0) return;
    CHECK_HOST(total <= kIndexMax) << "denorm_clamp: work items exceed uint32 indexing";
    const ChannelStats stats{
        {static_cast<float>(mean0), static_cast<float>(mean1), static_cast<float>(mean2)},
        {static_cast<float>(std0), static_cast<float>(std1), static_cast<float>(std2)}};
    LaunchKernel(static_cast<uint32_t>(div_ceil(total, static_cast<int64_t>(kBlockSize))), kBlockSize, device.unwrap())(
        denorm_clamp_kernel<kVectorized>,
        static_cast<const float*>(value.data_ptr()),
        static_cast<float*>(out.data_ptr()),
        make_layout<5>(value),
        stats,
        static_cast<uint32_t>(total));
  }
};

}  // namespace minimax_h3_vae_output

}  // namespace sglang
