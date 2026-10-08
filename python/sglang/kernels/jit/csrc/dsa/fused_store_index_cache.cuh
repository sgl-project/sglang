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
  void* __restrict__ shard_cache = nullptr;
  const void* __restrict__ logical_indices = nullptr;
  uint32_t shard_size = 1;
  uint32_t shard_rank = 0;
};

[[maybe_unused]]
SGL_DEVICE float fp8_e4m3_clip(float val) {
  namespace math = device::math;
  return math::max(math::min(val, kFP8E4M3Max), -kFP8E4M3Max);
}

[[maybe_unused]]
SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {
  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};
}

template <typename KeyT, typename IndicesT, uint32_t kPageBits, bool kUsePDL, bool kSharded = false>
__global__ void fused_store_indexer_cache(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  /// NOTE: 132 = 128 + 4
  constexpr int64_t kPageBytes = 132 << kPageBits;

  // each warp handles 128 elements, each block handles multiple rows
  const auto& [input, cache, indices, num_tokens, shard_cache, logical_indices, shard_size, shard_rank] = param;
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
  if constexpr (kSharded) {
    if (lane_id == 0) *static_cast<float*>(scale_ptr) = scale;
  } else {
    static_cast<float*>(scale_ptr)[0] = scale;
  }

  if constexpr (kSharded) {
    // Scratch consumes every token; persistent storage only consumes this
    // rank's pages. Reuse the exact same quantized bytes and FP32 scale.
    const auto logical = static_cast<const IndicesT*>(logical_indices)[global_wid];
    const auto logical_page = logical >> kPageBits;
    if (logical_page % shard_size == shard_rank) {
      const auto local_page = logical_page / shard_size;
      const auto local_offset = logical & ((1 << kPageBits) - 1);
      const auto local_ptr = pointer::offset(shard_cache, local_page * kPageBytes);
      static_cast<OutStorage*>(pointer::offset(local_ptr, local_offset * 128))[lane_id] = result;
      if (lane_id == 0) {
        *static_cast<float*>(pointer::offset(local_ptr, 128 << kPageBits, local_offset * 4)) = scale;
      }
    }
  }

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

  static void run_sharded(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView scratch,
      tvm::ffi::TensorView scratch_indices,
      tvm::ffi::TensorView shard_cache,
      tvm::ffi::TensorView logical_indices,
      int64_t shard_size,
      int64_t shard_rank) {
    using namespace host;
    RuntimeCheck(shard_size > 1 && shard_size <= UINT32_MAX, "invalid shard size");
    RuntimeCheck(shard_rank >= 0 && shard_rank < shard_size, "invalid shard rank");
    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 128}).with_dtype<KeyT>().with_device(device_).verify(input);
    TensorMatcher({-1, kPageBytes})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(scratch);
    TensorMatcher({-1, kPageBytes})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(shard_cache);
    TensorMatcher({N}).with_dtype<IndicesT>().with_device(device_).verify(scratch_indices);
    TensorMatcher({N}).with_dtype<IndicesT>().with_device(device_).verify(logical_indices);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    const auto params = FusedStoreCacheParam{
        .input = input.data_ptr(),
        .cache = scratch.data_ptr(),
        .indices = scratch_indices.data_ptr(),
        .num_tokens = num_tokens,
        .shard_cache = shard_cache.data_ptr(),
        .logical_indices = logical_indices.data_ptr(),
        .shard_size = static_cast<uint32_t>(shard_size),
        .shard_rank = static_cast<uint32_t>(shard_rank),
    };
    constexpr auto sharded_kernel = fused_store_indexer_cache<KeyT, IndicesT, kLogSize, kUsePDL, true>;
    LaunchKernel(div_ceil(num_tokens, 4u), 128, device_.unwrap()).enable_pdl(kUsePDL)(sharded_kernel, params);
  }

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

}  // namespace sglang
