#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cassert>
#include <cstdint>

namespace sglang {

struct StoreKVCacheParams {
  const void* __restrict__ k;
  const void* __restrict__ v;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ indices;
  int64_t stride_k_bytes;
  int64_t stride_v_bytes;
  // Independent slot strides: head_dim != v_head_dim gives K and V different row widths.
  int64_t stride_k_cache_bytes;
  int64_t stride_v_cache_bytes;
  int64_t stride_indices;
  uint32_t batch_size;
  int64_t size_limit;
  int64_t reserved_skip_index;
};

/**
 * \brief Kernel to store key-value pairs into the KV cache.
 * Each element is split into multiple parts to allow parallel memory copy.
 * \tparam kKBytes The size of each key element in bytes.
 * \tparam kVBytes The size of each value element in bytes.
 * \tparam kNumThreads Threads cooperating on one KV item; a multiple of the
 *         warp size. The block shape is chosen at launch, independently.
 * \tparam kUsePDL Whether to use PDL feature.
 * \tparam TLoc The data type of the indices (`int32_t` or `int64_t`).
 */
template <int64_t kKBytes, int64_t kVBytes, uint32_t kNumThreads, bool kUsePDL, typename TLoc>
__global__ void store_kvcache_kernel(const __grid_constant__ StoreKVCacheParams params) {
  using namespace device;
  static_assert(kNumThreads % kWarpThreads == 0, "TODO: support sub-warp copy for small items");
  constexpr uint32_t kNumSplit = kNumThreads / kWarpThreads;
  // Integer division below would silently drop the remainder of every row.
  static_assert(kKBytes % kNumSplit == 0 && kVBytes % kNumSplit == 0, "the split must divide both rows exactly");
  constexpr uint32_t kKSplitBytes = static_cast<uint32_t>(kKBytes) / kNumSplit;
  constexpr uint32_t kVSplitBytes = static_cast<uint32_t>(kVBytes) / kNumSplit;

  const auto warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  const auto item_id = warp_id / kNumSplit;
  const auto split_id = warp_id % kNumSplit;

  const auto& [
    k_input, v_input, k_cache, v_cache, indices, // ptr
    stride_k, stride_v, stride_k_cache, stride_v_cache, stride_indices, batch_size, // size
    size_limit, reserved_skip_index // bounds and reserved sink
  ] = params;
  if (item_id >= batch_size) return;

  PDLWaitPrimary<kUsePDL>();
  const auto index = static_cast<const TLoc*>(indices)[item_id * stride_indices];
  const auto k_src = pointer::offset(k_input, item_id * stride_k, split_id * kKSplitBytes);
  const auto v_src = pointer::offset(v_input, item_id * stride_v, split_id * kVSplitBytes);

  using enum warp::LoadStorePattern::type;
  const auto k = warp::load_bytes<kKSplitBytes, WARP_UNIFORM_16B>(k_src);
  const auto v = warp::load_bytes<kVSplitBytes, WARP_UNIFORM_16B>(v_src);

  PDLTriggerSecondary<kUsePDL>();
  assert(index >= 0 && index < size_limit);
  if (index != reserved_skip_index) {
    const auto k_dst = pointer::offset(k_cache, index * stride_k_cache, split_id * kKSplitBytes);
    const auto v_dst = pointer::offset(v_cache, index * stride_v_cache, split_id * kVSplitBytes);
    warp::store_bytes<kKSplitBytes, WARP_UNIFORM_16B>(k_dst, k);
    warp::store_bytes<kVSplitBytes, WARP_UNIFORM_16B>(v_dst, v);
  }
}

template <int64_t kKBytes, int64_t kVBytes, uint32_t kNumThreads, bool kUsePDL>
struct StoreKVCacheKernel {
  template <typename T>
  static constexpr auto store_kernel = store_kvcache_kernel<kKBytes, kVBytes, kNumThreads, kUsePDL, T>;

  static void
  run(const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView indices,
      const int64_t size_limit,
      const int64_t reserved_skip_index) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto DK = SymbolicSize{"k_element_size"};
    auto DV = SymbolicSize{"v_element_size"};
    auto dtype = SymbolicDType{};
    auto device_ = SymbolicDevice{};
    auto idx_dtype = SymbolicDType{};
    device_.set_options<kDLGPU>();

    using device::warp::LoadStorePattern;
    using enum LoadStorePattern::type;
    // Feed get_vec_bytes the SPLIT width, i.e. the exact value the kernel hands
    // to load_bytes -- the full row can resolve to a narrower vector and would
    // then under-constrain the strides.
    constexpr uint32_t kNumSplit = kNumThreads / device::kWarpThreads;
    constexpr int64_t kAlignK = LoadStorePattern::get_vec_bytes<kKBytes / kNumSplit, WARP_UNIFORM_16B>();
    constexpr int64_t kAlignV = LoadStorePattern::get_vec_bytes<kVBytes / kNumSplit, WARP_UNIFORM_16B>();

    TensorMatcher({B, DK})  //
        .with_strides({-1, 1})
        .with_dtype(dtype)
        .with_device(device_)
        .ensure_alignment(kAlignK)
        .verify(k);
    TensorMatcher({B, DV})  //
        .with_strides({-1, 1})
        .with_dtype(dtype)
        .with_device(device_)
        .ensure_alignment(kAlignV)
        .verify(v);
    TensorMatcher({-1, DK})  //
        .with_strides({-1, 1})
        .with_dtype(dtype)
        .with_device(device_)
        .ensure_alignment(kAlignK)
        .verify(k_cache);
    TensorMatcher({-1, DV})  //
        .with_strides({-1, 1})
        .with_dtype(dtype)
        .with_device(device_)
        .ensure_alignment(kAlignV)
        .verify(v_cache);
    TensorMatcher({B})  //
        .with_strides({-1})
        .with_dtype<int32_t, int64_t>(idx_dtype)
        .with_device(device_)
        .verify(indices);

    const auto dtype_size = static_cast<int64_t>(dtype_bytes(dtype.unwrap()));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto device = device_.unwrap();
    CHECK_HOST(kKBytes == dtype_size * DK.unwrap());
    CHECK_HOST(kVBytes == dtype_size * DV.unwrap());

    if (batch_size == 0) return;

    const auto params = StoreKVCacheParams{
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .stride_k_bytes = k.stride(0) * dtype_size,
        .stride_v_bytes = v.stride(0) * dtype_size,
        .stride_k_cache_bytes = k_cache.stride(0) * dtype_size,
        .stride_v_cache_bytes = v_cache.stride(0) * dtype_size,
        .stride_indices = indices.stride(0),
        .batch_size = batch_size,
        .size_limit = size_limit,
        .reserved_skip_index = reserved_skip_index,
    };

    const auto kernel = idx_dtype.is_type<int32_t>() ? store_kernel<int32_t> : store_kernel<int64_t>;
    const auto total_warps = batch_size * kNumSplit;
    const auto num_warps = [&] {
      const auto sm_count = runtime::get_sm_count(device.device_id);
#pragma unroll
      for (uint32_t n : {1, 2, 4}) {
        if (total_warps <= sm_count * n) return n;
      }
      return 8u;
    }();
    const auto num_blocks = div_ceil(total_warps, num_warps);
    LaunchKernel(num_blocks, {device::kWarpThreads, num_warps}, device)  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
