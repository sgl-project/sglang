#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace sglang {

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

// NVIDIA exposes an explicit "do not allocate in L1" cache hint via PTX. ROCm
// has no equivalent PTX, but non-temporal (streaming) loads/stores express the
// same intent for one-shot HiCache write-back traffic that should not pollute
// the cache. Guard the PTX behind USE_ROCM so the JIT module also compiles with
// hipcc; see python/sglang/kernels/jit/utils/compile.py for the ROCm build flags.
#ifdef USE_ROCM
// Native Clang vector types so a single __builtin_nontemporal_{load,store} maps
// to one vectorized global_{load,store}_dwordx{2,4}. Issuing N independent
// 32-bit nontemporal ops instead leaves merging to the LoadStoreVectorizer,
// which is not guaranteed and may drop the nontemporal hint, throttling HiCache
// bandwidth. uint2/uint4 already carry 8B/16B alignment matching the vector
// types, so the pointer reinterpret_casts stay correctly aligned.
typedef uint32_t native_uint2 __attribute__((ext_vector_type(2)));
typedef uint32_t native_uint4 __attribute__((ext_vector_type(4)));
#endif

SGL_DEVICE uint1 load_nc(const uint1* __restrict__ src) {
#ifndef USE_ROCM
  uint32_t tmp;
  asm volatile("ld.global.L1::no_allocate.b32 %0,[%1];" : "=r"(tmp) : "l"(src));
  return uint1{tmp};
#else
  return uint1{__builtin_nontemporal_load(&src->x)};
#endif
}

SGL_DEVICE uint2 load_nc(const uint2* __restrict__ src) {
#ifndef USE_ROCM
  uint32_t tmp0, tmp1;
  asm volatile("ld.global.L1::no_allocate.v2.b32 {%0,%1},[%2];" : "=r"(tmp0), "=r"(tmp1) : "l"(src));
  return uint2{tmp0, tmp1};
#else
  native_uint2 tmp = __builtin_nontemporal_load(reinterpret_cast<const native_uint2*>(src));
  return __builtin_bit_cast(uint2, tmp);
#endif
}

SGL_DEVICE uint4 load_nc(const uint4* __restrict__ src) {
#ifndef USE_ROCM
  uint32_t tmp0, tmp1, tmp2, tmp3;
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];"
               : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3)
               : "l"(src));
  return uint4{tmp0, tmp1, tmp2, tmp3};
#else
  native_uint4 tmp = __builtin_nontemporal_load(reinterpret_cast<const native_uint4*>(src));
  return __builtin_bit_cast(uint4, tmp);
#endif
}

SGL_DEVICE void store_nc(uint1* __restrict__ dst, const uint1& value) {
#ifndef USE_ROCM
  uint32_t tmp = value.x;
  asm volatile("st.global.L1::no_allocate.b32 [%0],%1;" ::"l"(dst), "r"(tmp));
#else
  __builtin_nontemporal_store(value.x, &dst->x);
#endif
}

SGL_DEVICE void store_nc(uint2* __restrict__ dst, const uint2& value) {
#ifndef USE_ROCM
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  asm volatile("st.global.L1::no_allocate.v2.b32 [%0],{%1,%2};" ::"l"(dst), "r"(tmp0), "r"(tmp1));
#else
  __builtin_nontemporal_store(__builtin_bit_cast(native_uint2, value), reinterpret_cast<native_uint2*>(dst));
#endif
}

SGL_DEVICE void store_nc(uint4* __restrict__ dst, const uint4& value) {
#ifndef USE_ROCM
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  uint32_t tmp2 = value.z;
  uint32_t tmp3 = value.w;
  asm volatile(
      "st.global.L1::no_allocate.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(dst), "r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3));
#else
  __builtin_nontemporal_store(__builtin_bit_cast(native_uint4, value), reinterpret_cast<native_uint4*>(dst));
#endif
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

/// \brief Operands of \ref hicache_transfer_per_layer_page_unified.
struct HicachePageUnifiedKernelParams {
  void* __restrict__ k_cache_dst;
  void* __restrict__ v_cache_dst;
  const void* __restrict__ src;
  const void* __restrict__ src_indices;
  const void* __restrict__ dst_indices;
  uint64_t total_vecs;
  int64_t num_items;
  int64_t num_groups;
  int64_t num_layers;
  int64_t page_size;
  int64_t layer_id;
};

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

/**
 * \brief Read one layer out of page-unified host pages into a per-layer device buffer.
 *
 * The source page block is (head_group, layer, 2, page_size, head_in_group, dim)
 * with K=0 and V=1 -- the exact byte order the page-unified write-back leaves
 * behind, so a page round-trips unchanged. MLA drops the head-group, K/V and
 * head axes, leaving (layer, page_size, dim).
 *
 * The destination is the device pool's own per-layer rows, (token, head, dim),
 * where head group \c g owns the contiguous run [g * kElementSize, (g+1) * kElementSize).
 * That run is exactly one source cell, which is why the permutation costs
 * nothing: it only reorders addresses the copy already computes per group.
 *
 * \c kElementSize is one head group's token row here, not the whole row the
 * other kernels in this file copy: one source cell is what a thread block
 * walks, and the device row is \c num_groups of them side by side.
 *
 * Each thread moves 16 bytes. The linear index runs dim-fastest, then item, so
 * lanes of a warp read consecutive host bytes for tokens of one (head group,
 * layer, K/V) cell -- the reads cross PCIe, so their coalescing is what sets
 * the achieved bandwidth.
 */
template <typename T, int64_t kElementSize, uint32_t kBlockSize, bool kIsMLA = false>
SGL_HICACHE_KERNEL void
hicache_transfer_per_layer_page_unified(const __grid_constant__ HicachePageUnifiedKernelParams params) {
  using namespace device;
  static_assert(kElementSize > 0 && kElementSize % 16 == 0);
  constexpr int64_t kGroupVecs = kElementSize / 16;
  constexpr int64_t kComponents = kIsMLA ? 1 : 2;
  const int64_t num_groups = kIsMLA ? 1 : params.num_groups;

  const auto tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t i = tid; i < params.total_vecs; i += stride) {
    const auto vec = static_cast<int64_t>(i % kGroupVecs);
    auto remaining = static_cast<int64_t>(i / kGroupVecs);
    const auto item = remaining % params.num_items;
    remaining /= params.num_items;
    const auto kv = remaining % kComponents;
    const auto group = remaining / kComponents;

    const auto src_token = static_cast<int64_t>(static_cast<const T*>(params.src_indices)[item]);
    const auto dst_token = static_cast<int64_t>(static_cast<const T*>(params.dst_indices)[item]);
    const auto page = src_token / params.page_size;
    const auto token_in_page = src_token % params.page_size;

    const auto src_offset =
        (((((page * num_groups + group) * params.num_layers + params.layer_id) * kComponents + kv) * params.page_size +
          token_in_page) *
         kElementSize) +
        vec * 16;
    const auto dst_offset = (dst_token * num_groups + group) * kElementSize + vec * 16;
    const auto dst_base = kv == 0 ? params.k_cache_dst : params.v_cache_dst;

    const auto src = pointer::offset(static_cast<const void*>(params.src), src_offset);
    const auto dst = pointer::offset(static_cast<void*>(dst_base), dst_offset);
    const auto value = details::load_nc(reinterpret_cast<const uint4*>(src));
    details::store_nc(reinterpret_cast<uint4*>(dst), value);
  }
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
  template <typename T, bool kIsMLA>
  static constexpr auto kernel_one_page_unified =
      hicache_transfer_per_layer_page_unified<T, kElementSize, kBlockSize, kIsMLA>;

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
        .with_device<kDLGPU, kDLGPUHost, kDLCPU>()
        .verify(k_cache_src)
        .verify(v_cache_src);
    TensorMatcher({-1, D})  //
        .with_strides({M, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLGPU, kDLGPUHost, kDLCPU>()
        .verify(k_cache_dst)
        .verify(v_cache_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLGPU>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    // verify dimension match
    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    const auto element_bytes = D.unwrap() * dtype_size;
    RuntimeCheck(kElementSize == element_bytes, "HicacheKernel: cache dimension mismatch.");

    const auto device = indices_device.unwrap();
    const auto k_cache_dst_ptr = runtime::get_device_accessible_ptr(k_cache_dst);
    const auto v_cache_dst_ptr = runtime::get_device_accessible_ptr(v_cache_dst);
    const auto k_cache_src_ptr = runtime::get_device_accessible_ptr(k_cache_src);
    const auto v_cache_src_ptr = runtime::get_device_accessible_ptr(v_cache_src);
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto kv_cache_src_stride = static_cast<int64_t>(N.unwrap() * dtype_size);
    const auto kv_cache_dst_stride = static_cast<int64_t>(M.unwrap() * dtype_size);
    const auto use_int32 = indices_dtype.unwrap().bits == 32;

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
        .with_device<kDLGPU>(device_)
        .verify(k_ptr_src)
        .verify(v_ptr_src)
        .verify(k_ptr_dst)
        .verify(v_ptr_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(dtype_)
        .with_device<kDLGPU>(device_)
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
        .with_device<kDLGPU, kDLGPUHost, kDLCPU>()
        .verify(cache_src);
    TensorMatcher({-1, D})  //
        .with_strides({M, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLGPU, kDLGPUHost, kDLCPU>()
        .verify(cache_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLGPU>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    const auto element_bytes = D.unwrap() * dtype_size;
    RuntimeCheck(kElementSize == element_bytes, "HicacheKernel MLA: cache dimension mismatch.");

    const auto device = indices_device.unwrap();
    const auto cache_dst_ptr = runtime::get_device_accessible_ptr(cache_dst);
    const auto cache_src_ptr = runtime::get_device_accessible_ptr(cache_src);
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto cache_src_stride = static_cast<int64_t>(N.unwrap() * dtype_size);
    const auto cache_dst_stride = static_cast<int64_t>(M.unwrap() * dtype_size);
    const auto use_int32 = indices_dtype.unwrap().bits == 32;

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
        .with_device<kDLGPU>(device_)
        .verify(ptr_src)
        .verify(ptr_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(dtype_)
        .with_device<kDLGPU>(device_)
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

  /**
   * \brief Load-back half of the page-unified HiCache layout: host pages -> device pool.
   *
   * The mirror of the page-unified write-back. Where write-back stages a whole
   * page on the device and hands the bulk transfer to the copy engine, load-back
   * is driven per layer by the SMs reading the pinned host mapping directly: the
   * transfer engine walks layers and unblocks the forward pass one layer at a
   * time, so staging a whole page per layer would move \c num_layers times the
   * bytes the layer actually needs.
   *
   * \tparam kIsMLA      Whether the pool has a single latent component and no head axis
   * \param k_cache_dst One layer of the device K pool, flattened to (token, head * dim)
   * \param v_cache_dst One layer of the device V pool; ignored when \c kIsMLA
   * \param src         Host pool flattened to (page, page elements), page-unified order
   * \param src_indices Host-pool TOKEN indices, on the destination device
   * \param dst_indices Device-pool token indices, same length and order
   * \param layer_id    Layer to read, indexing the host pool's layer axis
   * \param num_layers  Layer extent of the host pool's layer axis
   * \param num_groups  Head groups the page block is cut into; 1 when \c kIsMLA
   * \param page_size   Tokens per page
   */
  template <bool kIsMLA>
  static void run_one_page_unified(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView src,
      const tvm::ffi::TensorView src_indices,
      const tvm::ffi::TensorView dst_indices,
      const int64_t layer_id,
      const int64_t num_layers,
      const int64_t num_groups,
      const int64_t page_size) {
    using namespace host;

    auto B = SymbolicSize{"page elements"};
    auto C = SymbolicSize{"layer elements per token"};
    auto N = SymbolicSize{"indices length"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    // The device pool binds the launch device: the source is host memory read
    // over its pinned mapping, so it cannot name a stream.
    TensorMatcher({-1, C}).with_dtype(cache_dtype).with_device<kDLGPU>(device_).verify(k_cache_dst);
    if constexpr (!kIsMLA) {
      TensorMatcher({-1, C}).with_dtype(cache_dtype).with_device<kDLGPU>(device_).verify(v_cache_dst);
    }
    TensorMatcher({-1, B}).with_dtype(cache_dtype).with_device<kDLCPU, kDLGPUHost, kDLGPU>().verify(src);
    TensorMatcher({N})
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLGPU>(device_)
        .verify(src_indices)
        .verify(dst_indices);

    constexpr int64_t kComponents = kIsMLA ? 1 : 2;
    RuntimeCheck(num_layers > 0 && num_groups > 0 && page_size > 0, "Page-unified load-back: invalid dimensions");
    RuntimeCheck(layer_id >= 0 && layer_id < num_layers, "Page-unified load-back: layer id out of range");
    RuntimeCheck(!kIsMLA || num_groups == 1, "Page-unified MLA load-back: expected one latent group");
    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    RuntimeCheck(
        B.unwrap() * dtype_size == num_groups * num_layers * kComponents * page_size * kElementSize,
        "Page-unified load-back: page byte size mismatch");
    RuntimeCheck(
        C.unwrap() * dtype_size == num_groups * kElementSize, "Page-unified load-back: device row byte size mismatch");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(k_cache_dst.data_ptr()) % 16 == 0 &&
            (kIsMLA || reinterpret_cast<uintptr_t>(v_cache_dst.data_ptr()) % 16 == 0),
        "Page-unified load-back: buffers must be 16-byte aligned");
    if (N.unwrap() == 0) {
      return;
    }

    const auto params = HicachePageUnifiedKernelParams{
        .k_cache_dst = k_cache_dst.data_ptr(),
        .v_cache_dst = kIsMLA ? nullptr : v_cache_dst.data_ptr(),
        .src = src.data_ptr(),
        .src_indices = src_indices.data_ptr(),
        .dst_indices = dst_indices.data_ptr(),
        .total_vecs = static_cast<uint64_t>(N.unwrap()) * num_groups * kComponents * (kElementSize / 16),
        .num_items = N.unwrap(),
        .num_groups = num_groups,
        .num_layers = num_layers,
        .page_size = page_size,
        .layer_id = layer_id,
    };
    // A quota-capped persistent grid, as in the other entry points: the load
    // overlaps the forward pass, so more blocks buy bandwidth by taking SMs
    // from it.
    const auto num_blocks = std::min(div_ceil(params.total_vecs, uint64_t{kBlockSize}), uint64_t{kBlockQuota});
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    const auto kernel = use_int32 ? kernel_one_page_unified<int32_t, kIsMLA> : kernel_one_page_unified<int64_t, kIsMLA>;
    LaunchKernel(static_cast<uint32_t>(num_blocks), kBlockSize, device_.unwrap())(kernel, params);
  }
};

#undef SGL_HICACHE_KERNEL

}  // namespace sglang
