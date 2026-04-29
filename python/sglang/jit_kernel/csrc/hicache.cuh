#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstdint>
#include <dlfcn.h>
#include <limits>
#include <vector>

namespace device {

namespace details {

template <typename T, uint32_t N>
struct LocalStorage {
  T data[N];
};

template <int kUnit>
inline constexpr auto get_mem_package() {
  if constexpr (kUnit == 16) {
    return uint4{};
  } else if constexpr (kUnit == 8) {
    return uint2{};
  } else if constexpr (kUnit == 4) {
    return uint1{};
  } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4, "Unsupported memory package size");
  }
}

template <int kUnit>
using PackageType = decltype(get_mem_package<kUnit>());

SGL_DEVICE uint1 load_nc(const uint1* __restrict__ src) {
  uint32_t tmp;
  asm volatile("ld.global.L1::no_allocate.b32 %0,[%1];" : "=r"(tmp) : "l"(src));
  return uint1{tmp};
}

SGL_DEVICE uint2 load_nc(const uint2* __restrict__ src) {
  uint32_t tmp0, tmp1;
  asm volatile("ld.global.L1::no_allocate.v2.b32 {%0,%1},[%2];" : "=r"(tmp0), "=r"(tmp1) : "l"(src));
  return uint2{tmp0, tmp1};
}

SGL_DEVICE uint4 load_nc(const uint4* __restrict__ src) {
  uint32_t tmp0, tmp1, tmp2, tmp3;
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];"
               : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3)
               : "l"(src));
  return uint4{tmp0, tmp1, tmp2, tmp3};
}

SGL_DEVICE void store_nc(uint1* __restrict__ dst, const uint1& value) {
  uint32_t tmp = value.x;
  asm volatile("st.global.L1::no_allocate.b32 [%0],%1;" ::"l"(dst), "r"(tmp));
}

SGL_DEVICE void store_nc(uint2* __restrict__ dst, const uint2& value) {
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  asm volatile("st.global.L1::no_allocate.v2.b32 [%0],{%1,%2};" ::"l"(dst), "r"(tmp0), "r"(tmp1));
}

SGL_DEVICE void store_nc(uint4* __restrict__ dst, const uint4& value) {
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  uint32_t tmp2 = value.z;
  uint32_t tmp3 = value.w;
  asm volatile(
      "st.global.L1::no_allocate.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(dst), "r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3));
}

}  // namespace details

template <int64_t kBytes, uint32_t kNumThreads>
SGL_DEVICE auto load_vec(const void* __restrict__ src) {
  static_assert(kBytes % 128 == 0, "kBytes must be multiple of 128 bytes");
  static_assert(128 % kNumThreads == 0, "kNumThreads must divide 128 bytes");
  constexpr uint32_t kLoopCount = kBytes / 128;
  using Package = details::PackageType<128 / kNumThreads>;
  using Storage = details::LocalStorage<Package, kLoopCount>;

  const auto src_packed = static_cast<const Package*>(src);
  const auto lane_id = threadIdx.x % kNumThreads;
  Storage vec;

#pragma unroll kLoopCount
  for (uint32_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kNumThreads + lane_id;
    vec.data[i] = details::load_nc(&src_packed[j]);
  }

  return vec;
}

template <int64_t kBytes, uint32_t kNumThreads, typename Storage>
SGL_DEVICE void store_vec(void* __restrict__ dst, const Storage& vec) {
  using Package = std::decay_t<decltype(vec.data[0])>;
  constexpr uint32_t kBytesPerLoop = sizeof(Package) * kNumThreads;
  constexpr uint32_t kLoopCount = kBytes / kBytesPerLoop;
  static_assert(kBytes % kBytesPerLoop == 0, "Invalid Storage configuration");

  const auto dst_packed = static_cast<Package*>(dst);
  const auto lane_id = threadIdx.x % kNumThreads;

#pragma unroll kLoopCount
  for (uint32_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kNumThreads + lane_id;
    details::store_nc(&dst_packed[j], vec.data[i]);
  }
}

}  // namespace device

namespace {

#define SGL_HICACHE_KERNEL __global__ __launch_bounds__(kBlockSize, 1)

struct HicacheKernelParams {
  void* __restrict__ k_cache_dst;
  void* __restrict__ v_cache_dst;
  const void* __restrict__ indices_dst;
  void* __restrict__ k_cache_src;
  void* __restrict__ v_cache_src;
  const void* __restrict__ indices_src;
  int64_t kv_cache_src_stride;
  int64_t kv_cache_dst_stride;
  uint32_t length;
  uint32_t num_layers = 0;  // only used in all_layer transfer
};

struct HicacheRelayoutParams {
  void* __restrict__ k_cache_dst;
  void* __restrict__ v_cache_dst;
  const void* __restrict__ indices_src;
  const void* __restrict__ k_ptr_src;
  const void* __restrict__ v_ptr_src;
  uint32_t num_pages;
  uint32_t num_layers;
  uint32_t page_size;
};

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
#if CUDA_VERSION >= 13000
using CudaMemcpyBatchPtr = const void*;
using CudaMemcpyBatchAsyncFn = cudaError_t (*)(
    CudaMemcpyBatchPtr*,
    CudaMemcpyBatchPtr*,
    const size_t*,
    size_t,
    cudaMemcpyAttributes*,
    size_t*,
    size_t,
    cudaStream_t);
#else
using CudaMemcpyBatchPtr = void*;
using CudaMemcpyBatchAsyncFn = cudaError_t (*)(
    CudaMemcpyBatchPtr*,
    CudaMemcpyBatchPtr*,
    size_t*,
    size_t,
    cudaMemcpyAttributes*,
    size_t*,
    size_t,
    size_t*,
    cudaStream_t);
#endif

inline auto get_cuda_memcpy_batch_async() -> CudaMemcpyBatchAsyncFn {
  static CudaMemcpyBatchAsyncFn cuda_memcpy_batch_async = []() {
    void* symbol = dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync");
    return reinterpret_cast<CudaMemcpyBatchAsyncFn>(symbol);
  }();
  return cuda_memcpy_batch_async;
}

inline auto call_cuda_memcpy_batch_async(
    CudaMemcpyBatchAsyncFn copy_fn,
    CudaMemcpyBatchPtr* dsts,
    CudaMemcpyBatchPtr* srcs,
    size_t* sizes,
    size_t count,
    cudaMemcpyAttributes* attrs,
    size_t* attrs_idxs,
    size_t num_attrs,
    cudaStream_t stream) -> cudaError_t {
#if CUDA_VERSION >= 13000
  return copy_fn(dsts, srcs, sizes, count, attrs, attrs_idxs, num_attrs, stream);
#else
  size_t fail_idx = std::numeric_limits<size_t>::max();
  return copy_fn(dsts, srcs, sizes, count, attrs, attrs_idxs, num_attrs, &fail_idx, stream);
#endif
}
#endif

template <
    typename T,
    int64_t kElementSize,
    uint32_t kUnroll,
    uint32_t kBlockQuota,
    uint32_t kBlockSize,
    bool kIsMLA = false>
SGL_HICACHE_KERNEL void hicache_transfer_per_layer(const __grid_constant__ HicacheKernelParams params) {
  using namespace device;
  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kWarpThreads % kUnroll == 0);

  constexpr uint32_t kNumThreads = kWarpThreads / kUnroll;
  constexpr uint32_t kWorkersPerBlock = kBlockSize / kNumThreads;
  constexpr uint32_t kNumWorkers = kWorkersPerBlock * kBlockQuota;

  const auto& [
    k_cache_dst, v_cache_dst, indices_dst, // dst
    k_cache_src, v_cache_src, indices_src, // src
    kv_cache_src_stride, kv_cache_dst_stride, length, _ // metadata
  ] = params;

  const uint32_t work_id = blockIdx.x * kWorkersPerBlock + threadIdx.x / kNumThreads;
  for (uint32_t i = work_id; i < length; i += kNumWorkers) {
    const auto pos_src = static_cast<const T*>(indices_src)[i];
    const auto pos_dst = static_cast<const T*>(indices_dst)[i];
    const auto src_k = pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_k = pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
    const auto vec_k = load_vec<kElementSize, kNumThreads>(src_k);
    store_vec<kElementSize, kNumThreads>(dst_k, vec_k);
    if constexpr (!kIsMLA) {
      const auto src_v = pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
      const auto dst_v = pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
      const auto vec_v = load_vec<kElementSize, kNumThreads>(src_v);
      store_vec<kElementSize, kNumThreads>(dst_v, vec_v);
    }
  }
}

template <
    typename T,
    int64_t kElementSize,
    uint32_t kUnroll,
    uint32_t kBlockQuota,
    uint32_t kBlockSize,
    bool kIsMLA = false>
SGL_HICACHE_KERNEL void hicache_transfer_all_layer(const __grid_constant__ HicacheKernelParams params) {
  using namespace device;
  using src_ptr_t = const void*;
  using dst_ptr_t = void*;

  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kWarpThreads % kUnroll == 0);

  constexpr uint32_t kNumThreads = kWarpThreads / kUnroll;
  constexpr uint32_t kWorkersPerBlock = kBlockSize / kNumThreads;
  constexpr uint32_t kNumWorkers = kWorkersPerBlock * kBlockQuota;

  const auto& [
    k_ptr_dst, v_ptr_dst, indices_dst, // dst
    k_ptr_src, v_ptr_src, indices_src, // src
    kv_cache_src_stride, kv_cache_dst_stride, length, num_layers // metadata
  ] = params;

  const uint32_t work_id = blockIdx.x * kWorkersPerBlock + threadIdx.x / kNumThreads;
  for (uint32_t i = work_id; i < length; i += kNumWorkers) {
    const auto pos_src = static_cast<const T*>(indices_src)[i];
    const auto pos_dst = static_cast<const T*>(indices_dst)[i];
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
      const auto k_cache_src = static_cast<const src_ptr_t*>(k_ptr_src)[layer];
      const auto k_cache_dst = static_cast<const dst_ptr_t*>(k_ptr_dst)[layer];
      const auto src_k = pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
      const auto dst_k = pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
      const auto vec_k = load_vec<kElementSize, kNumThreads>(src_k);
      store_vec<kElementSize, kNumThreads>(dst_k, vec_k);
      if constexpr (!kIsMLA) {
        const auto v_cache_src = static_cast<const src_ptr_t*>(v_ptr_src)[layer];
        const auto v_cache_dst = static_cast<const dst_ptr_t*>(v_ptr_dst)[layer];
        const auto src_v = pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
        const auto dst_v = pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
        const auto vec_v = load_vec<kElementSize, kNumThreads>(src_v);
        store_vec<kElementSize, kNumThreads>(dst_v, vec_v);
      }
    }
  }
}

template <typename IndexType, int64_t kElementSize, bool kIsMLA>
__global__ void hicache_relayout_kernel(const __grid_constant__ HicacheRelayoutParams params) {
  using namespace device;
  using pack_t = uint4;
  static_assert(kElementSize % 16 == 0, "hicache_relayout_kernel requires 16-byte aligned element size");
  constexpr uint32_t kVecBytes = 16;
  constexpr uint32_t kVecPerItem = kElementSize / kVecBytes;

  const auto& [k_cache_dst, v_cache_dst, indices_src, k_ptr_src, v_ptr_src, num_pages, num_layers, page_size] = params;
  const auto k_ptr_src_arr = static_cast<const void* const*>(k_ptr_src);
  const auto v_ptr_src_arr = static_cast<const void* const*>(v_ptr_src);
  const auto tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const auto total_vecs = static_cast<uint64_t>(num_pages) * page_size * num_layers * kVecPerItem;

  for (uint64_t linear_vec_id = tid; linear_vec_id < total_vecs; linear_vec_id += stride) {
    const auto page_id =
        static_cast<uint32_t>(linear_vec_id / (static_cast<uint64_t>(page_size) * num_layers * kVecPerItem));
    const auto page_vec_id =
        static_cast<uint32_t>(linear_vec_id % (static_cast<uint64_t>(page_size) * num_layers * kVecPerItem));
    const auto token_in_page = page_vec_id / (num_layers * kVecPerItem);
    const auto token_vec_id = page_vec_id % (num_layers * kVecPerItem);
    const auto layer_id = token_vec_id / kVecPerItem;
    const auto vec_id = token_vec_id % kVecPerItem;
    const auto src_page = static_cast<uint32_t>(static_cast<const IndexType*>(indices_src)[page_id]);
    const auto src_token = src_page + token_in_page;
    const auto src_k = pointer::offset(
        static_cast<const void*>(k_ptr_src_arr[layer_id]),
        static_cast<int64_t>(src_token) * kElementSize + static_cast<int64_t>(vec_id) * kVecBytes);
    const auto dst_k =
        pointer::offset(static_cast<void*>(k_cache_dst), static_cast<int64_t>(linear_vec_id) * kVecBytes);
    const auto vec_k = details::load_nc(reinterpret_cast<const pack_t*>(src_k));
    details::store_nc(reinterpret_cast<pack_t*>(dst_k), vec_k);

    if constexpr (!kIsMLA) {
      const auto src_v = pointer::offset(
          static_cast<const void*>(v_ptr_src_arr[layer_id]),
          static_cast<int64_t>(src_token) * kElementSize + static_cast<int64_t>(vec_id) * kVecBytes);
      const auto dst_v =
          pointer::offset(static_cast<void*>(v_cache_dst), static_cast<int64_t>(linear_vec_id) * kVecBytes);
      const auto vec_v = details::load_nc(reinterpret_cast<const pack_t*>(src_v));
      details::store_nc(reinterpret_cast<pack_t*>(dst_v), vec_v);
    }
  }
}

inline void copy_page_first_pages_fallback(
    const std::vector<tvm::ffi::TensorView>& src_ptrs,
    std::vector<tvm::ffi::TensorView> dst_ptrs,
    const int64_t* dst_indices_ptr,
    int64_t num_pages,
    int64_t page_size,
    cudaStream_t stream) {
  using namespace host;

  RuntimeCheck(src_ptrs.size() == dst_ptrs.size(), "Source and destination tensors must have the same count");
  for (const auto tensor_id : irange(src_ptrs.size())) {
    RuntimeCheck(
        src_ptrs[tensor_id].dtype() == dst_ptrs[tensor_id].dtype(),
        "Source and destination tensors must have the same dtype");
    const int64_t elem_size = host::dtype_bytes(src_ptrs[tensor_id].dtype());
    const int64_t src_stride0 = src_ptrs[tensor_id].stride(0);
    const int64_t dst_stride0 = dst_ptrs[tensor_id].stride(0);
    const size_t src_page_bytes = static_cast<size_t>(page_size * src_stride0 * elem_size);
    const size_t dst_page_bytes = static_cast<size_t>(page_size * dst_stride0 * elem_size);
    RuntimeCheck(src_page_bytes == dst_page_bytes, "Source and destination page spans must match");
    for (const auto page_offset : irange(num_pages)) {
      const char* src_ptr = static_cast<const char*>(src_ptrs[tensor_id].data_ptr()) +
                            static_cast<size_t>(page_offset * page_size * src_stride0 * elem_size);
      char* dst_ptr = static_cast<char*>(dst_ptrs[tensor_id].data_ptr()) +
                      static_cast<size_t>(dst_indices_ptr[page_offset * page_size] * dst_stride0 * elem_size);
      RuntimeDeviceCheck(cudaMemcpyAsync(dst_ptr, src_ptr, src_page_bytes, cudaMemcpyDeviceToHost, stream));
    }
  }
}

inline bool try_copy_page_first_pages_batch(
    const std::vector<tvm::ffi::TensorView>& src_ptrs,
    std::vector<tvm::ffi::TensorView> dst_ptrs,
    const int64_t* dst_indices_ptr,
    int64_t num_pages,
    int64_t page_size,
    int device_id,
    cudaStream_t stream) {
#if defined(USE_ROCM) || !defined(CUDA_VERSION) || (CUDA_VERSION < 12080)
  return false;
#else
  host::RuntimeCheck(src_ptrs.size() == dst_ptrs.size(), "Source and destination tensors must have the same count");
  constexpr size_t kLargeCopyThresholdBytes = 128 * 1024;
  thread_local std::vector<CudaMemcpyBatchPtr> batch_srcs;
  thread_local std::vector<CudaMemcpyBatchPtr> batch_dsts;
  thread_local std::vector<size_t> batch_sizes;

  int driver_version = 0;
  cudaError_t driver_version_err = cudaDriverGetVersion(&driver_version);
  if (driver_version_err != cudaSuccess || driver_version < 12080) {
    return false;
  }

  auto copy_fn = get_cuda_memcpy_batch_async();
  if (copy_fn == nullptr) {
    return false;
  }

  const size_t num_copies = static_cast<size_t>(src_ptrs.size()) * static_cast<size_t>(num_pages);
  batch_srcs.clear();
  batch_dsts.clear();
  batch_sizes.clear();
  batch_srcs.reserve(num_copies);
  batch_dsts.reserve(num_copies);
  batch_sizes.reserve(num_copies);

  size_t first_page_bytes = 0;
  for (const auto tensor_id : host::irange(src_ptrs.size())) {
    host::RuntimeCheck(
        src_ptrs[tensor_id].dtype() == dst_ptrs[tensor_id].dtype(),
        "Source and destination tensors must have the same dtype");
    const int64_t elem_size = host::dtype_bytes(src_ptrs[tensor_id].dtype());
    const int64_t src_stride0 = src_ptrs[tensor_id].stride(0);
    const int64_t dst_stride0 = dst_ptrs[tensor_id].stride(0);
    const size_t src_page_bytes = static_cast<size_t>(page_size * src_stride0 * elem_size);
    const size_t dst_page_bytes = static_cast<size_t>(page_size * dst_stride0 * elem_size);
    host::RuntimeCheck(src_page_bytes == dst_page_bytes, "Source and destination page spans must match");
    if (tensor_id == 0) {
      first_page_bytes = src_page_bytes;
    }
    for (const auto page_offset : host::irange(num_pages)) {
      char* src_ptr = static_cast<char*>(src_ptrs[tensor_id].data_ptr()) +
                      static_cast<size_t>(page_offset * page_size * src_stride0 * elem_size);
      char* dst_ptr = static_cast<char*>(dst_ptrs[tensor_id].data_ptr()) +
                      static_cast<size_t>(dst_indices_ptr[page_offset * page_size] * dst_stride0 * elem_size);
      batch_srcs.push_back(src_ptr);
      batch_dsts.push_back(dst_ptr);
      batch_sizes.push_back(src_page_bytes);
    }
  }
  if (first_page_bytes >= kLargeCopyThresholdBytes) {
    return false;
  }

  std::vector<size_t> attrs_idxs(1, 0);
  cudaMemcpyAttributes attrs{};
  attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attrs.srcLocHint.type = cudaMemLocationTypeDevice;
  attrs.srcLocHint.id = device_id;
  attrs.dstLocHint.type = cudaMemLocationTypeHost;
  attrs.dstLocHint.id = 0;
  attrs.flags = 0;

  cudaError_t err = call_cuda_memcpy_batch_async(
      copy_fn,
      batch_dsts.data(),
      batch_srcs.data(),
      batch_sizes.data(),
      num_copies,
      &attrs,
      attrs_idxs.data(),
      1,
      stream);
  if (err == cudaErrorNotSupported || err == cudaErrorCallRequiresNewerDriver || err == cudaErrorInvalidValue) {
    (void)cudaGetLastError();
    return false;
  }
  host::RuntimeCheck(err == cudaSuccess, "cudaMemcpyBatchAsync failed. error=", cudaGetErrorString(err));
  return true;
#endif
}

template <int64_t kElementSize, uint32_t kUnroll, uint32_t kBlockQuota, uint32_t kBlockSize>
struct HiCacheKernel {
  template <typename T>
  static constexpr auto kernel_one = hicache_transfer_per_layer<T, kElementSize, kUnroll, kBlockQuota, kBlockSize>;
  template <typename T>
  static constexpr auto kernel_all = hicache_transfer_all_layer<T, kElementSize, kUnroll, kBlockQuota, kBlockSize>;
  template <typename T>
  static constexpr auto kernel_one_mla =
      hicache_transfer_per_layer<T, kElementSize, kUnroll, kBlockQuota, kBlockSize, true>;
  template <typename T>
  static constexpr auto kernel_all_mla =
      hicache_transfer_all_layer<T, kElementSize, kUnroll, kBlockQuota, kBlockSize, true>;

  static void run_one(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_cache_src,
      const tvm::ffi::TensorView v_cache_src,
      const tvm::ffi::TensorView indices_src) {
    using namespace host;

    auto D = SymbolicSize{"head dimension"};
    auto N = SymbolicSize{"src kv stride"};
    auto M = SymbolicSize{"dst kv stride"};
    auto L = SymbolicSize{"indices length"};
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
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto kv_cache_src_stride = static_cast<int64_t>(N.unwrap() * dtype_size);
    const auto kv_cache_dst_stride = static_cast<int64_t>(M.unwrap() * dtype_size);
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    const auto device = indices_device.unwrap();

    constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = k_cache_dst_ptr,
        .v_cache_dst = v_cache_dst_ptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = k_cache_src_ptr,
        .v_cache_src = v_cache_src_ptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = kv_cache_src_stride,
        .kv_cache_dst_stride = kv_cache_dst_stride,
        .length = length,
    };
    const auto kernel = use_int32 ? kernel_one<int32_t> : kernel_one<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }

  static void run_all(
      const tvm::ffi::TensorView k_ptr_dst,
      const tvm::ffi::TensorView v_ptr_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_ptr_src,
      const tvm::ffi::TensorView v_ptr_src,
      const tvm::ffi::TensorView indices_src,
      const int64_t kv_src_stride_bytes,
      const int64_t kv_dst_stride_bytes) {
    using namespace host;

    auto N = SymbolicSize{"num_layers"};
    auto L = SymbolicSize{"indices length"};
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
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto use_int32 = dtype_.unwrap().bits == 32;
    const auto device = device_.unwrap();

    constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = k_cache_dst_ptr,
        .v_cache_dst = v_cache_dst_ptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = k_cache_src_ptr,
        .v_cache_src = v_cache_src_ptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = kv_src_stride_bytes,
        .kv_cache_dst_stride = kv_dst_stride_bytes,
        .length = length,
        .num_layers = static_cast<uint32_t>(N.unwrap()),
    };
    const auto kernel = use_int32 ? kernel_all<int32_t> : kernel_all<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }

  static void run_one_mla(
      const tvm::ffi::TensorView cache_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView cache_src,
      const tvm::ffi::TensorView indices_src) {
    using namespace host;

    auto D = SymbolicSize{"head dimension"};
    auto N = SymbolicSize{"src stride"};
    auto M = SymbolicSize{"dst stride"};
    auto L = SymbolicSize{"indices length"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto indices_device = SymbolicDevice{};

    TensorMatcher({-1, D})  //
        .with_strides({N, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(cache_src);
    TensorMatcher({-1, D})  //
        .with_strides({M, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(cache_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    const auto element_bytes = D.unwrap() * dtype_size;
    RuntimeCheck(kElementSize == element_bytes, "HicacheKernel MLA: cache dimension mismatch.");

    const auto cache_dst_ptr = cache_dst.data_ptr();
    const auto cache_src_ptr = cache_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto cache_src_stride = static_cast<int64_t>(N.unwrap() * dtype_size);
    const auto cache_dst_stride = static_cast<int64_t>(M.unwrap() * dtype_size);
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    const auto device = indices_device.unwrap();

    constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = cache_dst_ptr,
        .v_cache_dst = nullptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = cache_src_ptr,
        .v_cache_src = nullptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = cache_src_stride,
        .kv_cache_dst_stride = cache_dst_stride,
        .length = length,
    };
    const auto kernel = use_int32 ? kernel_one_mla<int32_t> : kernel_one_mla<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }

  static void run_all_mla(
      const tvm::ffi::TensorView ptr_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView ptr_src,
      const tvm::ffi::TensorView indices_src,
      const int64_t src_stride_bytes,
      const int64_t dst_stride_bytes) {
    using namespace host;

    auto N = SymbolicSize{"num_layers"};
    auto L = SymbolicSize{"indices length"};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({N})  //
        .with_dtype<uint64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(ptr_src)
        .verify(ptr_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(indices_src)
        .verify(indices_dst);

    const auto cache_dst_ptr = ptr_dst.data_ptr();
    const auto cache_src_ptr = ptr_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto use_int32 = dtype_.unwrap().bits == 32;
    const auto device = device_.unwrap();

    constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = cache_dst_ptr,
        .v_cache_dst = nullptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = cache_src_ptr,
        .v_cache_src = nullptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = src_stride_bytes,
        .kv_cache_dst_stride = dst_stride_bytes,
        .length = length,
        .num_layers = static_cast<uint32_t>(N.unwrap()),
    };
    const auto kernel = use_int32 ? kernel_all_mla<int32_t> : kernel_all_mla<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }

 private:
  template <bool kIsMLA>
  static void launch_relayout_kernel(
      const HicacheRelayoutParams& params,
      int64_t num_pages,
      int64_t num_layers,
      int64_t page_size,
      bool use_int32,
      DLDevice device) {
    using namespace host;

    constexpr uint32_t kRelayoutBlockSize = 256;
    constexpr uint32_t kVecPerItem = kElementSize / 16;
    const auto total_vecs = static_cast<uint64_t>(num_pages) * page_size * num_layers * kVecPerItem;
    const auto kernel = use_int32 ? hicache_relayout_kernel<int32_t, kElementSize, kIsMLA>
                                  : hicache_relayout_kernel<int64_t, kElementSize, kIsMLA>;
    const auto max_occ = runtime::get_blocks_per_sm(kernel, kRelayoutBlockSize);
    const auto num_sm = runtime::get_sm_count(device.device_id);
    const auto grid =
        std::min<uint64_t>(num_sm * max_occ, div_ceil(total_vecs, static_cast<uint64_t>(kRelayoutBlockSize)));
    LaunchKernel(static_cast<uint32_t>(grid), kRelayoutBlockSize, device)(kernel, params);
  }

  template <bool kIsMLA>
  static void run_staged_impl(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView dst_indices_cpu,
      const tvm::ffi::TensorView staging_k,
      const tvm::ffi::TensorView staging_v,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView k_ptr_src,
      const tvm::ffi::TensorView v_ptr_src,
      const int64_t page_size) {
    using namespace host;

    auto T = SymbolicSize{"num_tokens"};
    auto N = SymbolicSize{"num_layers"};
    auto D = SymbolicSize{"element_dim"};
    auto P = SymbolicSize{"num_pages"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto dst_indices_dtype = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({T, N, D})  //
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA>(device_)
        .verify(staging_k);
    if constexpr (!kIsMLA) {
      TensorMatcher({T, N, D})  //
          .with_dtype(cache_dtype)
          .with_device<kDLCUDA>(device_)
          .verify(staging_v);
    }
    TensorMatcher({-1, N, D})  //
        .with_dtype(cache_dtype)
        .with_device<kDLCPU, kDLCUDAHost>()
        .verify(k_cache_dst);
    if constexpr (!kIsMLA) {
      TensorMatcher({-1, N, D})  //
          .with_dtype(cache_dtype)
          .with_device<kDLCPU, kDLCUDAHost>()
          .verify(v_cache_dst);
    }
    TensorMatcher({N})  //
        .with_dtype<uint64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(k_ptr_src);
    if constexpr (!kIsMLA) {
      TensorMatcher({N})  //
          .with_dtype<uint64_t>()
          .with_device<kDLCUDA>(device_)
          .verify(v_ptr_src);
    }
    TensorMatcher({P})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(device_)
        .verify(page_indices_src);
    TensorMatcher({T})  //
        .with_dtype<int64_t>(dst_indices_dtype)
        .with_device<kDLCPU, kDLCUDAHost>()
        .verify(dst_indices_cpu);

    RuntimeCheck(page_size > 0, "HiCache staged relayout: page_size must be positive");
    RuntimeCheck(T.unwrap() == P.unwrap() * page_size, "HiCache staged relayout: staging token count mismatch");
    RuntimeCheck(
        kElementSize == D.unwrap() * dtype_bytes(cache_dtype.unwrap()),
        "HiCache staged relayout: element size mismatch");
    RuntimeCheck(kElementSize % 16 == 0, "HiCache staged relayout: element size must be 16-byte aligned");

    const auto params = HicacheRelayoutParams{
        .k_cache_dst = staging_k.data_ptr(),
        .v_cache_dst = kIsMLA ? nullptr : staging_v.data_ptr(),
        .indices_src = page_indices_src.data_ptr(),
        .k_ptr_src = k_ptr_src.data_ptr(),
        .v_ptr_src = kIsMLA ? nullptr : v_ptr_src.data_ptr(),
        .num_pages = static_cast<uint32_t>(P.unwrap()),
        .num_layers = static_cast<uint32_t>(N.unwrap()),
        .page_size = static_cast<uint32_t>(page_size),
    };
    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    launch_relayout_kernel<kIsMLA>(params, P.unwrap(), N.unwrap(), page_size, use_int32, device);

    auto stream = LaunchKernel::resolve_device(device);
    const int64_t* dst_indices_ptr = static_cast<const int64_t*>(dst_indices_cpu.data_ptr());
    if constexpr (kIsMLA) {
      if (!try_copy_page_first_pages_batch(
              {staging_k}, {k_cache_dst}, dst_indices_ptr, P.unwrap(), page_size, device.device_id, stream)) {
        copy_page_first_pages_fallback({staging_k}, {k_cache_dst}, dst_indices_ptr, P.unwrap(), page_size, stream);
      }
    } else {
      if (!try_copy_page_first_pages_batch(
              {staging_k, staging_v},
              {k_cache_dst, v_cache_dst},
              dst_indices_ptr,
              P.unwrap(),
              page_size,
              device.device_id,
              stream)) {
        copy_page_first_pages_fallback(
            {staging_k, staging_v}, {k_cache_dst, v_cache_dst}, dst_indices_ptr, P.unwrap(), page_size, stream);
      }
    }
  }

 public:
  static void run_all_lf_pf_staged(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView dst_indices_cpu,
      const tvm::ffi::TensorView staging_k,
      const tvm::ffi::TensorView staging_v,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView k_ptr_src,
      const tvm::ffi::TensorView v_ptr_src,
      const int64_t page_size) {
    run_staged_impl<false>(
        k_cache_dst,
        v_cache_dst,
        dst_indices_cpu,
        staging_k,
        staging_v,
        page_indices_src,
        k_ptr_src,
        v_ptr_src,
        page_size);
  }

  static void run_all_mla_lf_pf_staged(
      const tvm::ffi::TensorView cache_dst,
      const tvm::ffi::TensorView dst_indices_cpu,
      const tvm::ffi::TensorView staging,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView ptr_src,
      const int64_t page_size) {
    run_staged_impl<true>(
        cache_dst, cache_dst, dst_indices_cpu, staging, staging, page_indices_src, ptr_src, ptr_src, page_size);
  }
};

#undef SGL_HICACHE_KERNEL

}  // namespace
