#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

struct HicacheKernelParams {
  void* __restrict__ k_cache_dst;
  void* __restrict__ v_cache_dst;
  const void* __restrict__ indices_dst;
  void* __restrict__ k_cache_src;
  void* __restrict__ v_cache_src;
  const void* __restrict__ indices_src;
  std::size_t length;
  std::size_t kv_cache_src_stride;
  std::size_t kv_cache_dst_stride;
  std::size_t num_layers = 0;  // only used in all_layer transfer
};

template <
    std::integral T,
    std::size_t kElementSize,
    std::size_t kUnroll,
    std::size_t kBlockQuota,
    std::size_t kNumThreads,
    std::size_t kMaxOccupancy>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void hicache_transfer_per_layer(
    const __grid_constant__ HicacheKernelParams params) {
  // each warp acts as a worker
  using namespace device;
  static_assert(kNumThreads % kWarpThreads == 0);
  static_assert(kWarpThreads % kUnroll == 0);

  constexpr auto kWarpThreads = device::kWarpThreads / kUnroll;
  constexpr auto kWarpsPerBlock = kNumThreads / kWarpThreads;
  constexpr auto kWorkers = kWarpsPerBlock * kBlockQuota;

  const auto& [
    k_cache_dst, v_cache_dst, indices_dst, // dst
    k_cache_src, v_cache_src, indices_src, // src
    length, kv_cache_src_stride, kv_cache_dst_stride, _ // metadata
  ] = params;
  const auto warp_id = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpThreads;

  // force to transfer 128 bytes per iteration
  // since the PCIe transaction size is 128 bytes aligned
  constexpr auto kGranularity = 128 / kWarpThreads;

  for (auto i = warp_id; i < length; i += kWorkers) {
    const auto pos_src = static_cast<const T*>(indices_src)[i];
    const auto pos_dst = static_cast<const T*>(indices_dst)[i];
    const auto src_k = pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_k = pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
    const auto src_v = pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_v = pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
    const auto vec_k = warp::load_vec<kElementSize, kGranularity, kWarpThreads>(src_k);
    const auto vec_v = warp::load_vec<kElementSize, kGranularity, kWarpThreads>(src_v);
    warp::store_vec<kElementSize, kGranularity, kWarpThreads>(dst_k, vec_k);
    warp::store_vec<kElementSize, kGranularity, kWarpThreads>(dst_v, vec_v);
  }
}

template <
    std::integral T,
    std::size_t kElementSize,
    std::size_t kUnroll,
    std::size_t kBlockQuota,
    std::size_t kNumThreads,
    std::size_t kMaxOccupancy>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void hicache_transfer_all_layer(
    const __grid_constant__ HicacheKernelParams params) {
  // each warp acts as a worker
  using namespace device;
  using src_ptr_t = std::add_pointer_t<const void* const>;
  using dst_ptr_t = std::add_pointer_t<void* const>;

  static_assert(kNumThreads % kWarpThreads == 0);
  constexpr auto kWarpThreads = device::kWarpThreads / kUnroll;
  constexpr auto kWarpsPerBlock = static_cast<uint32_t>(kNumThreads) / kWarpThreads;
  constexpr auto kWorkers = kWarpsPerBlock * kBlockQuota;

  const auto& [
    k_ptr_dst, v_ptr_dst, indices_dst, // dst
    k_ptr_src, v_ptr_src, indices_src, // src
    length, kv_cache_src_stride, kv_cache_dst_stride, num_layers // metadata
  ] = params;
  const auto warp_id = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpThreads;

  // force to transfer 128 bytes per iteration
  // since the PCIe transaction size is 128 bytes aligned
  constexpr auto kGranularity = 128 / kWarpThreads;

  for (auto i = warp_id; i < length; i += kWorkers) {
    const auto pos_src = static_cast<const T*>(indices_src)[i];
    const auto pos_dst = static_cast<const T*>(indices_dst)[i];
    for (std::size_t layer = 0; layer < num_layers; ++layer) {
      const auto k_cache_src = static_cast<src_ptr_t>(k_ptr_src)[layer];
      const auto v_cache_src = static_cast<src_ptr_t>(v_ptr_src)[layer];
      const auto k_cache_dst = static_cast<dst_ptr_t>(k_ptr_dst)[layer];
      const auto v_cache_dst = static_cast<dst_ptr_t>(v_ptr_dst)[layer];
      const auto src_k = pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
      const auto dst_k = pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
      const auto src_v = pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
      const auto dst_v = pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
      const auto vec_k = warp::load_vec<kElementSize, kGranularity, kWarpThreads>(src_k);
      const auto vec_v = warp::load_vec<kElementSize, kGranularity, kWarpThreads>(src_v);
      warp::store_vec<kElementSize, kGranularity, kWarpThreads>(dst_k, vec_k);
      warp::store_vec<kElementSize, kGranularity, kWarpThreads>(dst_v, vec_v);
    }
  }
}

template <
    std::size_t kElementSize,
    std::size_t kUnroll,
    std::size_t kBlockQuota,
    std::size_t kNumThreads,
    std::size_t kMaxOccupancy>
struct HiCacheKernel {
  template <typename T>
  static constexpr auto _kernel_one =
      hicache_transfer_per_layer<T, kElementSize, kUnroll, kBlockQuota, kNumThreads, kMaxOccupancy>;
  template <typename T>
  static constexpr auto _kernel_all =
      hicache_transfer_all_layer<T, kElementSize, kUnroll, kBlockQuota, kNumThreads, kMaxOccupancy>;

  static void run_one(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_cache_src,
      const tvm::ffi::TensorView v_cache_src,
      const tvm::ffi::TensorView indices_src) {
    using namespace host;

    auto D = SymbolicSize{"D"};  // cache dimension
    auto N = SymbolicSize{"N"};  // src kv stride
    auto M = SymbolicSize{"M"};  // dst kv stride
    auto L = SymbolicSize{"L"};  // indices length
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto indices_device = SymbolicDevice{};

    TensorMatcher({-1, D})  //
        .with_strides({N, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_src)
        .verify(v_cache_src);
    TensorMatcher({-1, D})  //
        .with_strides({M, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_dst)
        .verify(v_cache_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    // verify dimension match
    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    const auto element_bytes = D.unwrap() * dtype_size;
    RuntimeCheck(kElementSize == element_bytes, "HicacheKernel: cache dimension mismatch.");

    const auto k_cache_dst_ptr = k_cache_dst.data_ptr();
    const auto v_cache_dst_ptr = v_cache_dst.data_ptr();
    const auto k_cache_src_ptr = k_cache_src.data_ptr();
    const auto v_cache_src_ptr = v_cache_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<std::size_t>(L.unwrap());
    const auto kv_cache_src_stride = static_cast<std::size_t>(N.unwrap()) * dtype_size;
    const auto kv_cache_dst_stride = static_cast<std::size_t>(M.unwrap()) * dtype_size;
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    const auto device = indices_device.unwrap();

    constexpr auto kWorkersPerBlock = kNumThreads / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = k_cache_dst_ptr,
        .v_cache_dst = v_cache_dst_ptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = k_cache_src_ptr,
        .v_cache_src = v_cache_src_ptr,
        .indices_src = indices_src_ptr,
        .length = length,
        .kv_cache_src_stride = kv_cache_src_stride,
        .kv_cache_dst_stride = kv_cache_dst_stride,
    };
    const auto kernel = use_int32 ? _kernel_one<int32_t> : _kernel_one<int64_t>;
    LaunchKernel(num_blocks, kNumThreads, device)(kernel, params);
  }

  static void run_all(
      const tvm::ffi::TensorView k_ptr_dst,
      const tvm::ffi::TensorView v_ptr_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_ptr_src,
      const tvm::ffi::TensorView v_ptr_src,
      const tvm::ffi::TensorView indices_src,
      const std::size_t kv_src_stride,
      const std::size_t kv_dst_stride) {
    using namespace host;

    auto N = SymbolicSize{"N"};  // num layers
    auto L = SymbolicSize{"L"};  // indices length
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({N})  //
        .with_dtype<uint64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(k_ptr_src)
        .verify(v_ptr_src)
        .verify(k_ptr_dst)
        .verify(v_ptr_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(indices_src)
        .verify(indices_dst);

    // verify dimension match
    const auto k_cache_dst_ptr = k_ptr_dst.data_ptr();
    const auto v_cache_dst_ptr = v_ptr_dst.data_ptr();
    const auto k_cache_src_ptr = k_ptr_src.data_ptr();
    const auto v_cache_src_ptr = v_ptr_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<std::size_t>(L.unwrap());
    const auto use_int32 = dtype_.unwrap().bits == 32;
    const auto device = device_.unwrap();

    constexpr auto kWorkersPerBlock = kNumThreads / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = k_cache_dst_ptr,
        .v_cache_dst = v_cache_dst_ptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = k_cache_src_ptr,
        .v_cache_src = v_cache_src_ptr,
        .indices_src = indices_src_ptr,
        .length = length,
        .kv_cache_src_stride = kv_src_stride,
        .kv_cache_dst_stride = kv_dst_stride,
        .num_layers = static_cast<std::size_t>(N.unwrap()),
    };
    const auto kernel = use_int32 ? _kernel_all<int32_t> : _kernel_all<int64_t>;
    LaunchKernel(num_blocks, kNumThreads, device)(kernel, params);
  }
};

}  // namespace
