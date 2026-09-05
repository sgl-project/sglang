#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

struct FusedStoreCacheParam {
  const void* __restrict__ input;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  uint32_t num_tokens;
};

struct FusedQuantizePackedParam {
  const void* __restrict__ input;
  void* __restrict__ packed;
  uint32_t num_tokens;
};

struct StorePackedCacheParam {
  const void* __restrict__ packed;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  uint32_t num_tokens;
};

struct StoreRankMajorPackedCacheParam {
  const void* __restrict__ packed;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  uint32_t num_tokens;
  uint32_t cp_mask;
  uint32_t cp_bits;
  uint32_t rows_per_rank;
};

[[maybe_unused]] SGL_DEVICE float fp8_e4m3_clip(float val) {
  namespace math = device::math;
  return math::max(math::min(val, kFP8E4M3Max), -kFP8E4M3Max);
}

[[maybe_unused]] SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {
  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};
}

// Quantize one bf16 index-K row into a transport-friendly 132-byte record:
// 128 fp8 payload bytes followed by one fp32 scale.  The arithmetic is kept
// byte-for-byte identical to fused_store_indexer_cache below; only the output
// address changes.  CP can therefore communicate the compact representation
// without changing the cache contents consumed by the indexer.
template <typename KeyT>
__global__ void fused_quantize_indexer_packed(const __grid_constant__ FusedQuantizePackedParam param) {
  using namespace device;

  const auto& [input, packed, num_tokens] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;
  if (global_wid >= num_tokens) return;

  using KeyT2 = packed_t<KeyT>;
  using InStorage = AlignedVector<KeyT2, 2>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;
  const auto elems = static_cast<const InStorage*>(input)[global_tid];
  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);
  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);
  const auto local_max = fmaxf(fmaxf(fabs(x0), fabs(x1)), fmaxf(fabs(y0), fabs(y1)));
  const auto abs_max = warp::reduce_max(local_max);
  const auto scale = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
  const auto inv_scale = 1.0f / scale;

  auto* packed_row = pointer::offset(packed, global_wid * 132);
  OutStorage result;
  result[0] = pack_fp8(x0 * inv_scale, x1 * inv_scale);
  result[1] = pack_fp8(y0 * inv_scale, y1 * inv_scale);
  static_cast<OutStorage*>(packed_row)[lane_id] = result;
  if (lane_id == 0) {
    *static_cast<float*>(pointer::offset(packed_row, 128)) = scale;
  }
}

// Store already-quantized 132-byte rows into the ordinary page-64 index-K
// cache.  A warp copies one row; no dequantization or requantization occurs.
template <typename IndicesT, uint32_t kPageBits>
__global__ void store_packed_indexer_cache(const __grid_constant__ StorePackedCacheParam param) {
  using namespace device;
  constexpr int64_t kPageBytes = 132 << kPageBits;

  const auto& [packed, cache, indices, num_tokens] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;
  if (global_wid >= num_tokens) return;

  const auto index = static_cast<const IndicesT*>(indices)[global_wid];
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  auto* page_ptr = pointer::offset(cache, page * kPageBytes);
  auto* value_ptr = pointer::offset(page_ptr, offset * 128);
  auto* scale_ptr = pointer::offset(page_ptr, 128 << kPageBits, offset * 4);

  const auto* packed_words = static_cast<const uint32_t*>(packed) + static_cast<int64_t>(global_wid) * 33;
  static_cast<uint32_t*>(value_ptr)[lane_id] = packed_words[lane_id];
  if (lane_id == 0) {
    *static_cast<uint32_t*>(scale_ptr) = packed_words[32];
  }
}

// Store a CP all-gather's rank-major output directly into logical-token cache
// order.  For equal interleave shards, logical row i lives at rank-major row
// (i % cp_size) * rows_per_rank + i / cp_size.  Folding that source lookup
// into the paged destination scatter removes the global index_select and its
// equally large temporary.  The production adapter selects this helper only
// under the default-off packed-transport experiment after proving equal
// power-of-two Interleave shards; every other layout retains the old reorder.
template <typename IndicesT, uint32_t kPageBits>
__global__ void store_rank_major_packed_indexer_cache(const __grid_constant__ StoreRankMajorPackedCacheParam param) {
  using namespace device;
  constexpr int64_t kPageBytes = 132 << kPageBits;

  const auto& [packed, cache, indices, num_tokens, cp_mask, cp_bits, rows_per_rank] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto logical_row = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;
  if (logical_row >= num_tokens) return;

  const auto source_row = (logical_row & cp_mask) * rows_per_rank + (logical_row >> cp_bits);
  const auto index = static_cast<const IndicesT*>(indices)[logical_row];
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  auto* page_ptr = pointer::offset(cache, page * kPageBytes);
  auto* value_ptr = pointer::offset(page_ptr, offset * 128);
  auto* scale_ptr = pointer::offset(page_ptr, 128 << kPageBits, offset * 4);

  const auto* packed_words = static_cast<const uint32_t*>(packed) + static_cast<int64_t>(source_row) * 33;
  static_cast<uint32_t*>(value_ptr)[lane_id] = packed_words[lane_id];
  if (lane_id == 0) {
    *static_cast<uint32_t*>(scale_ptr) = packed_words[32];
  }
}

template <typename KeyT, typename IndicesT, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_store_indexer_cache(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  /// NOTE: 132 = 128 + 4
  constexpr int64_t kPageBytes = 132 << kPageBits;

  // each warp handles 128 elements, each block handles multiple rows
  const auto& [input, cache, indices, num_tokens] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;

  if (global_wid >= num_tokens) return;

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  // prefetch the index
  const auto index = static_cast<const IndicesT*>(indices)[global_wid];
  // always load the value from input (don't store if invalid)
  using KeyT2 = packed_t<KeyT>;
  using InStorage = AlignedVector<KeyT2, 2>;
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

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

template <typename KeyT, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct FusedStoreCacheIndexerKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  /// NOTE: 132 = 128 + 4 (128 represent K and 4 represent scale)
  static constexpr int64_t kPageBytes = 132 * kPageSize;
  static constexpr auto kernel = fused_store_indexer_cache<KeyT, IndicesT, kLogSize, kUsePDL>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 128})  // input
        .with_dtype<KeyT>()
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

template <typename KeyT, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct FusedQuantizePackedIndexerKernel {
  static constexpr auto kernel = fused_quantize_indexer_packed<KeyT>;

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView packed) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 128}).with_dtype<KeyT>().with_device(device_).verify(input);
    TensorMatcher({N, 132}).with_dtype<uint8_t>().with_device(device_).verify(packed);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedQuantizePackedParam{
        .input = input.data_ptr(),
        .packed = packed.data_ptr(),
        .num_tokens = num_tokens,
    };
    constexpr int kBlockSize = 128;
    const auto num_blocks = div_ceil(num_tokens * 32, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())(kernel, params);
  }
};

template <typename KeyT, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct StorePackedCacheIndexerKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = 132 * kPageSize;
  static constexpr auto kernel = store_packed_indexer_cache<IndicesT, kLogSize>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");

  static void run(tvm::ffi::TensorView packed, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 132}).with_dtype<uint8_t>().with_device(device_).verify(packed);
    TensorMatcher({-1, -1}).with_strides({kPageBytes, 1}).with_dtype<uint8_t>().with_device(device_).verify(cache);
    TensorMatcher({N}).with_dtype<IndicesT>().with_device(device_).verify(indices);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = StorePackedCacheParam{
        .packed = packed.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
    };
    constexpr int kBlockSize = 128;
    const auto num_blocks = div_ceil(num_tokens * 32, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())(kernel, params);
  }
};

template <typename KeyT, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct StoreRankMajorPackedCacheIndexerKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = 132 * kPageSize;
  static constexpr auto kernel = store_rank_major_packed_indexer_cache<IndicesT, kLogSize>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");

  static void
  run(tvm::ffi::TensorView packed, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices, int64_t cp_size) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 132}).with_dtype<uint8_t>().with_device(device_).verify(packed);
    TensorMatcher({-1, -1}).with_strides({kPageBytes, 1}).with_dtype<uint8_t>().with_device(device_).verify(cache);
    TensorMatcher({N}).with_dtype<IndicesT>().with_device(device_).verify(indices);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    RuntimeCheck(cp_size > 1, "cp_size must be greater than one");
    RuntimeCheck(std::has_single_bit(static_cast<uint64_t>(cp_size)), "cp_size must be a power of two");
    RuntimeCheck(num_tokens % cp_size == 0, "rank-major packed store requires equal CP shards");
    const auto params = StoreRankMajorPackedCacheParam{
        .packed = packed.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
        .cp_mask = static_cast<uint32_t>(cp_size - 1),
        .cp_bits = static_cast<uint32_t>(std::countr_zero(static_cast<uint64_t>(cp_size))),
        .rows_per_rank = static_cast<uint32_t>(num_tokens / cp_size),
    };
    constexpr int kBlockSize = 128;
    const auto num_blocks = div_ceil(num_tokens * 32, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())(kernel, params);
  }
};

}  // namespace sglang
