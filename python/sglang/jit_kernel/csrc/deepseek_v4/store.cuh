#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <cuda_fp8.h>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

template <bool kOwnerSharded>
struct FusedStoreCacheParam;

template <>
struct FusedStoreCacheParam<false> {
  const void* __restrict__ input;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  uint32_t num_tokens;
};

template <>
struct FusedStoreCacheParam<true> {
  const void* __restrict__ input;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  uint32_t num_tokens;
  uint32_t owner_rank;
  uint32_t owner_size;
};

template <typename Float, typename IndicesT, uint32_t kPageBits, bool kUsePDL, bool kOwnerSharded>
__global__ void fused_store_flashmla_cache(const __grid_constant__ FusedStoreCacheParam<kOwnerSharded> param) {
  using namespace device;

  /// NOTE: 584 = 576 + 8
  constexpr int64_t kPageBytes = host::div_ceil(584 << kPageBits, 576) * 576;

  // each warp handles 64 elements, 8 warps, each block handles 1 row
  const auto input = param.input;
  const auto cache = param.cache;
  const auto indices = param.indices;
  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t wid = tid / 32;

  PDLWaitPrimary<kUsePDL>();

  // prefetch the index
  const auto index = static_cast<const IndicesT*>(indices)[bid];
  int32_t physical_page = index >> kPageBits;
  if constexpr (kOwnerSharded) {
    if (index < 0 || static_cast<uint32_t>(physical_page) % param.owner_size != param.owner_rank) {
      PDLTriggerSecondary<kUsePDL>();
      return;
    }
    physical_page /= static_cast<int32_t>(param.owner_size);
  }
  const int32_t offset = index & ((1 << kPageBits) - 1);
  // always load the value from input (don't store if invalid)
  using Float2 = packed_t<Float>;
  const auto elems = static_cast<const Float2*>(input)[tid + bid * 256];
  const auto page_ptr = pointer::offset(cache, physical_page * kPageBytes);
  if (wid != 7) {
    const auto [x, y] = cast<fp32x2_t>(elems);
    const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
    const auto scale_raw = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
    const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    const auto result = pack_fp8(x * inv_scale, y * inv_scale);
    const auto value_ptr = pointer::offset(page_ptr, offset * 576);
    const auto scale_ptr = pointer::offset(page_ptr, 576 << kPageBits, offset * 8);
    static_cast<fp8x2_e4m3_t*>(value_ptr)[tid] = result;
    static_cast<uint8_t*>(scale_ptr)[wid] = scale_ue8m0;
  } else {
    const auto result = cast<bf16x2_t>(elems);
    const auto value_ptr = pointer::offset(page_ptr, offset * 576, 448);
    static_cast<bf16x2_t*>(value_ptr)[tid - 7 * 32] = result;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename Float, typename IndicesT, uint32_t kPageBits, bool kUsePDL, bool kOwnerSharded>
__global__ void fused_store_indexer_cache(const __grid_constant__ FusedStoreCacheParam<kOwnerSharded> param) {
  using namespace device;

  /// NOTE: 132 = 128 + 4
  constexpr int64_t kPageBytes = 132 << kPageBits;

  // each warp handles 128 elements, 1 warp, each block handles multiple rows
  const auto input = param.input;
  const auto cache = param.cache;
  const auto indices = param.indices;
  const auto num_tokens = param.num_tokens;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;

  if (global_wid >= num_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  // prefetch the index
  const auto index = static_cast<const IndicesT*>(indices)[global_wid];
  int32_t physical_page = index >> kPageBits;
  if constexpr (kOwnerSharded) {
    if (index < 0 || static_cast<uint32_t>(physical_page) % param.owner_size != param.owner_rank) {
      PDLTriggerSecondary<kUsePDL>();
      return;
    }
    physical_page /= static_cast<int32_t>(param.owner_size);
  }
  // always load the value from input (don't store if invalid)
  using Float2 = packed_t<Float>;
  using InStorage = AlignedVector<Float2, 2>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;
  const auto elems = static_cast<const InStorage*>(input)[global_tid];
  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);
  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);
  const auto local_max = fmaxf(fmaxf(fabs(x0), fabs(x1)), fmaxf(fabs(y0), fabs(y1)));
  const auto abs_max = warp::reduce_max(local_max);
  const auto scale = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
  const auto inv_scale = 1.0f / scale;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  const auto page_ptr = pointer::offset(cache, physical_page * kPageBytes);
  const auto value_ptr = pointer::offset(page_ptr, offset * 128);
  const auto scale_ptr = pointer::offset(page_ptr, 128 << kPageBits, offset * 4);
  OutStorage result;
  result[0] = pack_fp8(x0 * inv_scale, x1 * inv_scale);
  result[1] = pack_fp8(y0 * inv_scale, y1 * inv_scale);
  static_cast<OutStorage*>(value_ptr)[lane_id] = result;
  static_cast<float*>(scale_ptr)[0] = scale;

  PDLTriggerSecondary<kUsePDL>();
}

template <typename Float, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct FusedStoreCacheFlashMLAKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = host::div_ceil(584 * kPageSize, 576) * 576;
  static constexpr auto kernel = fused_store_flashmla_cache<Float, IndicesT, kLogSize, kUsePDL, false>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
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
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedStoreCacheParam<false>{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
    };
    const auto kBlockSize = 256;
    const auto num_blocks = num_tokens;
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

template <typename Float, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct FusedStoreCacheFlashMLASharedKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = host::div_ceil(584 * kPageSize, 576) * 576;
  static constexpr auto kernel = fused_store_flashmla_cache<Float, IndicesT, kLogSize, kUsePDL, true>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");

  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView indices,
      int64_t owner_rank,
      int64_t owner_size) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 512}).with_dtype<Float>().with_device(device_).verify(input);
    TensorMatcher({-1, -1}).with_strides({kPageBytes, 1}).with_dtype<uint8_t>().with_device(device_).verify(cache);
    TensorMatcher({N}).with_dtype<IndicesT>().with_device(device_).verify(indices);
    const auto params = FusedStoreCacheParam<true>{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = static_cast<uint32_t>(N.unwrap()),
        .owner_rank = static_cast<uint32_t>(owner_rank),
        .owner_size = static_cast<uint32_t>(owner_size),
    };
    LaunchKernel(N.unwrap(), 256, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

template <typename Float, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct FusedStoreCacheIndexerKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = 132 * kPageSize;
  static constexpr auto kernel = fused_store_indexer_cache<Float, IndicesT, kLogSize, kUsePDL, false>;

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
    const auto params = FusedStoreCacheParam<false>{
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

template <typename Float, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct FusedStoreCacheIndexerSharedKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = 132 * kPageSize;
  static constexpr auto kernel = fused_store_indexer_cache<Float, IndicesT, kLogSize, kUsePDL, true>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");

  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView indices,
      int64_t owner_rank,
      int64_t owner_size) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 128}).with_dtype<Float>().with_device(device_).verify(input);
    TensorMatcher({-1, -1}).with_strides({kPageBytes, 1}).with_dtype<uint8_t>().with_device(device_).verify(cache);
    TensorMatcher({N}).with_dtype<IndicesT>().with_device(device_).verify(indices);
    const auto params = FusedStoreCacheParam<true>{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = static_cast<uint32_t>(N.unwrap()),
        .owner_rank = static_cast<uint32_t>(owner_rank),
        .owner_size = static_cast<uint32_t>(owner_size),
    };
    const auto num_blocks = div_ceil(N.unwrap() * 32, 128);
    LaunchKernel(num_blocks, 128, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
