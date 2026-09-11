#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>
#include <sgl_kernel/deepseek_v4/kv_layout.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <cuda_fp8.h>
#include <optional>

namespace sglang {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

struct FusedStoreCacheParam {
  const void* __restrict__ input;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  uint32_t num_tokens;
};

/// Parameters of the V4.1 (fp8 / fp4) FlashMLA store; `freqs_cis` is the per-token
/// (real, imag) pairs of the 64 RoPE dims, or nullptr when the input is already rotated.
struct FusedStoreCacheV41Param {
  const void* __restrict__ input;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  const float* __restrict__ freqs_cis;
  uint32_t num_tokens;
};

template <typename Float, typename IndicesT, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_store_flashmla_cache(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  using Paged = deepseek_v4::PagedKV<deepseek_v4::KVLayout::V4, kPageBits>;

  // each warp handles 64 elements, 8 warps, each block handles 1 row
  const auto& [input, cache, indices, num_tokens] = param;
  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t wid = tid / 32;

  PDLWaitPrimary<kUsePDL>();

  // prefetch the index
  const auto index = static_cast<const IndicesT*>(indices)[bid];
  // always load the value from input (don't store if invalid)
  using Float2 = packed_t<Float>;
  const auto elems = static_cast<const Float2*>(input)[tid + bid * 256];
  if (wid != 7) {
    const auto [x, y] = cast<fp32x2_t>(elems);
    const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
    const auto scale_raw = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
    const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    const auto result = pack_fp8(x * inv_scale, y * inv_scale);
    const auto row = Paged::row(static_cast<uint8_t*>(cache), index);
    reinterpret_cast<fp8x2_e4m3_t*>(row.data)[tid] = result;
    row.scale[wid] = scale_ue8m0;
  } else {
    const auto result = cast<bf16x2_t>(elems);
    const auto row = Paged::row(static_cast<uint8_t*>(cache), index);
    reinterpret_cast<bf16x2_t*>(row.data + 448)[tid - 7 * 32] = result;
  }

  PDLTriggerSecondary<kUsePDL>();
}

/// V4.1 store: one 256-thread block per token, thread `tx` owns elements (2tx, 2tx + 1) of the
/// 512-wide row. With kRope the last warp first rotates its (real, imag) pairs -- the RoPE
/// tail -- and rounds them back to the input dtype, as `rope_tail` does, so that the caller can
/// hand in the un-rotated, un-quantized latent and the e2m1 / e4m3 rounding happens exactly once.
/// Elements per thread of the V4.1 store, `512 / vec` threads per token: 4 for the fp8 rows
/// (8-byte loads, half the threads), 2 for the fp4 rows, whose per-element IEEE divisions
/// are better spread over more threads (measured on B200, bs 1..512).
constexpr uint32_t v41_store_vec_size(deepseek_v4::KVLayout layout) {
  return layout == deepseek_v4::KVLayout::V41 ? 4 : 2;
}

template <
    typename Float,
    typename IndicesT,
    uint32_t kPageBits,
    bool kUsePDL,
    deepseek_v4::KVLayout kLayout,
    bool kRope>
__global__ void fused_store_flashmla_cache_v41(const __grid_constant__ FusedStoreCacheV41Param param) {
  using namespace device;
  using Paged = deepseek_v4::PagedKV<kLayout, kPageBits>;
  static_assert(kLayout != deepseek_v4::KVLayout::V4, "the V4 layout has its own kernel above");

  constexpr uint32_t kVecSize = v41_store_vec_size(kLayout);
  constexpr uint32_t kNopeLanes = (512 - 64) / kVecSize;  // threads from here on hold the RoPE tail
  using Packed = packed_t<Float>;
  using Vec = AlignedVector<Packed, kVecSize / 2>;

  const auto& [input, cache, indices, freqs_cis, num_tokens] = param;
  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;

  PDLWaitPrimary<kUsePDL>();

  const auto index = static_cast<const IndicesT*>(indices)[bid];
  Vec elems;
  elems.load(static_cast<const Float*>(input) + bid * 512, tid);
  float v[kVecSize];
#pragma unroll
  for (uint32_t i = 0; i < kVecSize / 2; ++i) {
    const auto [x, y] = cast<fp32x2_t>(elems[i]);
    v[2 * i] = x;
    v[2 * i + 1] = y;
  }
  if constexpr (kRope) {
    if (tid >= kNopeLanes) {
      // (real, imag) pairs of the tail, rotated and rounded back to the input dtype as
      // `rope_tail` does, so that the caller can also pass pre-rotated rows.
      AlignedVector<float, kVecSize> freq;
      freq.load(freqs_cis + bid * 64, tid - kNopeLanes);
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        const auto x = v[2 * i];
        const auto y = v[2 * i + 1];
        const auto rotated = cast<fp32x2_t>(
            cast<Packed>(fp32x2_t{x * freq[2 * i] - y * freq[2 * i + 1], x * freq[2 * i + 1] + y * freq[2 * i]}));
        v[2 * i] = rotated.x;
        v[2 * i + 1] = rotated.y;
      }
    }
  }

  const auto row = Paged::row(static_cast<uint8_t*>(cache), index);
  deepseek_v4::v41::store_row<kLayout>(row.data, row.scale, tid, v);

  PDLTriggerSecondary<kUsePDL>();
}

template <typename Float, typename IndicesT, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_store_indexer_cache(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  /// NOTE: 132 = 128 + 4
  constexpr int64_t kPageBytes = 132 << kPageBits;

  // each warp handles 128 elements, 1 warp, each block handles multiple rows
  const auto& [input, cache, indices, num_tokens] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;

  if (global_wid >= num_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  // prefetch the index
  const auto index = static_cast<const IndicesT*>(indices)[global_wid];
  // always load the value from input (don't store if invalid)
  using Float2 = packed_t<Float>;
  using InStorage = AlignedVector<Float2, 2>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;
  const auto elems = static_cast<const InStorage*>(input)[global_tid];
  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);
  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);
  const auto local_max = fmaxf(fmaxf(fabs(x0), fabs(x1)), fmaxf(fabs(y0), fabs(y1)));
  const auto abs_max = warp::reduce_max(local_max);
  // use normal fp32 scale
  const auto scale = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
  const auto inv_scale = 1.0f / scale;
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  const auto page_ptr = pointer::offset(cache, page * kPageBytes);
  const auto value_ptr = pointer::offset(page_ptr, offset * 128);
  const auto scale_ptr = pointer::offset(page_ptr, 128 << kPageBits, offset * 4);
  OutStorage result;
  result[0] = pack_fp8(x0 * inv_scale, x1 * inv_scale);
  result[1] = pack_fp8(y0 * inv_scale, y1 * inv_scale);
  static_cast<OutStorage*>(value_ptr)[lane_id] = result;
  static_cast<float*>(scale_ptr)[0] = scale;

  PDLTriggerSecondary<kUsePDL>();
}

template <
    typename Float,
    typename IndicesT,
    uint32_t kPageSize,
    bool kUsePDL,
    deepseek_v4::KVLayout kLayout = deepseek_v4::KVLayout::V4>
struct FusedStoreCacheFlashMLAKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr bool kIsV4 = kLayout == deepseek_v4::KVLayout::V4;
  static constexpr int64_t kPageBytes = deepseek_v4::kv_page_bytes<kLayout>(kPageSize);
  static_assert(!kIsV4 || kPageBytes == host::div_ceil(584 * kPageSize, 576) * 576);

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);

  template <bool kRope>
  static constexpr auto v41_kernel = fused_store_flashmla_cache_v41<Float, IndicesT, kLogSize, kUsePDL, kLayout, kRope>;

  /// Store rows that are already normed and rotated.
  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
    launch(input, cache, indices, std::nullopt);
  }

  /// V4.1 layouts only: rotate the RoPE tail in-kernel with the per-token `freqs_cis`
  /// (`[num_tokens, 64]` fp32, real / imag interleaved) before quantizing.
  static void run_rope(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView indices,
      tvm::ffi::TensorView freqs_cis) {
    static_assert(!kIsV4, "the V4 layout keeps its RoPE dims in bf16 and has no in-kernel RoPE");
    launch(input, cache, indices, freqs_cis);
  }

 private:
  static void launch(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView indices,
      std::optional<tvm::ffi::TensorView> freqs_cis) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 512})  // input
        .with_dtype<Float>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({N})  // indices
        .with_dtype<IndicesT>()
        .with_device(device_)
        .verify(indices);
    if (freqs_cis.has_value()) {
      // Real / imag interleaved, so the trailing dim is 64, not 32.
      TensorMatcher({N, 64}).with_dtype<float>().with_device(device_).verify(*freqs_cis);
    }
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    const auto kBlockSize = 256;
    const auto num_blocks = num_tokens;
    if constexpr (kIsV4) {
      RuntimeCheck(!freqs_cis.has_value(), "the V4 layout has no in-kernel RoPE");
      const auto params = FusedStoreCacheParam{
          .input = input.data_ptr(),
          .cache = cache.data_ptr(),
          .indices = indices.data_ptr(),
          .num_tokens = num_tokens,
      };
      constexpr auto kernel = fused_store_flashmla_cache<Float, IndicesT, kLogSize, kUsePDL>;
      LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
    } else {
      const auto params = FusedStoreCacheV41Param{
          .input = input.data_ptr(),
          .cache = cache.data_ptr(),
          .indices = indices.data_ptr(),
          .freqs_cis = freqs_cis.has_value() ? static_cast<const float*>(freqs_cis->data_ptr()) : nullptr,
          .num_tokens = num_tokens,
      };
      const auto kernel = freqs_cis.has_value() ? v41_kernel<true> : v41_kernel<false>;
      LaunchKernel(num_blocks, 512 / v41_store_vec_size(kLayout), device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
    }
  }
};

template <typename Float, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct FusedStoreCacheIndexerKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = 132 * kPageSize;
  static constexpr auto kernel = fused_store_indexer_cache<Float, IndicesT, kLogSize, kUsePDL>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 128})  // input
        .with_dtype<Float>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({N})  // indices
        .with_dtype<IndicesT>()
        .with_device(device_)
        .verify(indices);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedStoreCacheParam{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
    };
    const auto kBlockSize = 128;
    const auto num_blocks = div_ceil(num_tokens * 32, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
