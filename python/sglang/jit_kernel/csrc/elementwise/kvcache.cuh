#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct StoreKVCacheParams {
  const void* __restrict__ k;
  const void* __restrict__ v;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ indices;
  int64_t stride_k_bytes;
  int64_t stride_v_bytes;
  int64_t stride_cache_bytes;
  int64_t stride_indices;
  uint32_t batch_size;
};

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;

template <int64_t kElementBytes>
__device__ void copy_impl(
    const void* __restrict__ k_src,
    const void* __restrict__ v_src,
    void* __restrict__ k_dst,
    void* __restrict__ v_dst) {
  using namespace device;
  constexpr int64_t kAlignment = (kElementBytes % (16 * kWarpThreads) == 0) ? 16
                                 : kElementBytes % (8 * kWarpThreads) == 0  ? 8
                                 : kElementBytes % (4 * kWarpThreads) == 0  ? 4
                                 : kElementBytes % 4 == 0                   ? 4
                                                                            : 0;

  static_assert(kAlignment > 0, "Element size must be multiple of 4 bytes");

  using vec_t = aligned_vector<uint32_t, kAlignment / 4>;
  constexpr auto kLoopBytes = sizeof(vec_t) * kWarpThreads;
  constexpr auto kLoopCount = kElementBytes / kLoopBytes;

#pragma unroll kLoopCount
  for (int64_t i = 0; i < kLoopCount; ++i) {
    const auto k = warp::load<vec_t>(pointer::offset(k_src, i * kLoopBytes));
    const auto v = warp::load<vec_t>(pointer::offset(v_src, i * kLoopBytes));
    warp::store(pointer::offset(k_dst, i * kLoopBytes), k);
    warp::store(pointer::offset(v_dst, i * kLoopBytes), v);
  }

  // handle the epilogue if any
  if constexpr (kLoopCount * kLoopBytes < kElementBytes) {
    constexpr auto kOffset = kLoopCount * kLoopBytes;
    if ((threadIdx.x % kWarpThreads) * sizeof(vec_t) < kElementBytes - kOffset) {
      const auto k = warp::load<vec_t>(pointer::offset(k_src, kOffset));
      const auto v = warp::load<vec_t>(pointer::offset(v_src, kOffset));
      warp::store(pointer::offset(k_dst, kOffset), k);
      warp::store(pointer::offset(v_dst, kOffset), v);
    }
  }
}

// Each warp handles one item
template <int64_t kElementBytes, int kSplit, bool kUsePDL, typename T>
__global__ void store_kvcache(const __grid_constant__ StoreKVCacheParams params) {
  using namespace device;
  constexpr auto kSplitSize = kElementBytes / kSplit;
  const uint32_t warp_id = blockIdx.x * kNumWarps + threadIdx.x / kWarpThreads;
  const uint32_t item_id = warp_id / kSplit;
  const uint32_t split_id = warp_id % kSplit;
  const auto& [
    k_input, v_input, k_cache, v_cache, indices, // ptr
    stride_k, stride_v, stride_cache, stride_indices, batch_size // size
  ] = params;
  if (item_id >= batch_size) return;

  const auto index_ptr = static_cast<const T*>(indices) + item_id * stride_indices;
  PDLWaitPrimary<kUsePDL>();

  const auto index = *index_ptr;
  const auto k_src = pointer::offset(k_input, item_id * stride_k, split_id * kSplitSize);
  const auto v_src = pointer::offset(v_input, item_id * stride_v, split_id * kSplitSize);
  const auto k_dst = pointer::offset(k_cache, index * stride_cache, split_id * kSplitSize);
  const auto v_dst = pointer::offset(v_cache, index * stride_cache, split_id * kSplitSize);

  copy_impl<kSplitSize>(k_src, v_src, k_dst, v_dst);
  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kElementBytes, bool kUsePDL>
struct StoreKVCacheKernel {
  static_assert(kElementBytes > 0 && kElementBytes % 4 == 0);

  template <int kSplit, typename T>
  static constexpr auto store_kernel = store_kvcache<kElementBytes, kSplit, kUsePDL, T>;

  template <typename T>
  static auto get_kernel(const int num_split) {
    using namespace host;
    // only apply split optimization when element size is aligned
    if constexpr (kElementBytes % (4 * 128) == 0) {
      if (num_split == 4) return store_kernel<4, T>;
    }
    if constexpr (kElementBytes % (2 * 128) == 0) {
      if (num_split == 2) return store_kernel<2, T>;
    }
    if (num_split == 1) return store_kernel<1, T>;
    Panic("Unsupported num_split {} for element size {}", num_split, kElementBytes);
  }

  static void
  run(const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView indices,
      const int num_split) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto D = SymbolicSize{"element_size"};
    auto KS = SymbolicSize{"k_stride"};
    auto VS = SymbolicSize{"v_stride"};
    auto S = SymbolicSize{"cache_stride"};
    auto I = SymbolicSize{"indices_stride"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    auto indice_dtype = SymbolicDType{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, D})  //
        .with_strides({KS, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(k);
    TensorMatcher({B, D})  //
        .with_strides({VS, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(v);
    TensorMatcher({-1, D})  //
        .with_strides({S, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({B})  //
        .with_strides({I})
        .with_dtype<int32_t, int64_t>(indice_dtype)
        .with_device(device)
        .verify(indices);

    const int64_t dtype_size = dtype_bytes(dtype.unwrap());
    const uint32_t num_elements = static_cast<uint32_t>(B.unwrap());
    RuntimeCheck(kElementBytes == dtype_size * D.unwrap());

    const auto params = StoreKVCacheParams{
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .stride_k_bytes = KS.unwrap() * dtype_size,
        .stride_v_bytes = VS.unwrap() * dtype_size,
        .stride_cache_bytes = S.unwrap() * dtype_size,
        .stride_indices = I.unwrap(),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
    };
    // select kernel and update num_split if needed
    const auto use_int32 = indice_dtype.is_type<int32_t>();
    const auto kernel = use_int32 ? get_kernel<int32_t>(num_split) : get_kernel<int64_t>(num_split);
    const auto num_blocks = div_ceil(num_elements * num_split, kNumWarps);
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
