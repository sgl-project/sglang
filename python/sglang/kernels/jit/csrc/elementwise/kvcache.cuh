#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
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

struct StoreKVCacheQuantParams {
  const void* __restrict__ k;
  const void* __restrict__ v;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ indices;
  // Per-tensor scales, either as a device scalar (preferred: no host sync when
  // the caller holds them as 0-dim GPU tensors) or as host-precomputed
  // reciprocals. A null pointer selects the host value.
  const float* __restrict__ k_scale;
  const float* __restrict__ v_scale;
  float k_inv_scale;
  float v_inv_scale;
  int64_t stride_k;  // source/cache row strides, in elements
  int64_t stride_v;
  int64_t stride_cache;
  int64_t stride_indices;
  uint32_t batch_size;
  int64_t size_limit;
  int64_t reserved_skip_index;  // reserved sink; -1 disables skipping
};

/**
 * \brief Kernel to quantize key-value pairs to FP8 and store them into the KV
 * cache in a single pass. Fuses the unfused eager sequence
 * ``k.div_(scale); k.to(fp8)`` + byte store (5 kernel launches with scales)
 * into one launch. One warp handles one item.
 * \tparam kRowElems The number of elements per key/value row.
 * \tparam TSrc The source data type (`bf16_t`, `fp16_t` or `fp32_t`).
 * \tparam TDst The quantized cache data type (`fp8_e4m3_t`).
 * \tparam kUsePDL Whether to use PDL feature.
 * \tparam T The data type of the indices (`int32_t` or `int64_t`).
 */
template <int64_t kRowElems, typename TSrc, typename TDst, bool kUsePDL, typename T>
__global__ void store_kvcache_quant(const __grid_constant__ StoreKVCacheQuantParams params) {
  using namespace device;
  constexpr uint32_t kVecElems = 16 / sizeof(TSrc);
  using src_vec_t = AlignedVector<TSrc, kVecElems>;
  using dst_vec_t = AlignedVector<TDst, kVecElems>;
  constexpr int64_t kVecsPerRow = kRowElems / kVecElems;
  constexpr int64_t kLoopCount = kVecsPerRow / kWarpThreads;
  constexpr int64_t kTailVecs = kVecsPerRow % kWarpThreads;

  const uint32_t item_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (item_id >= params.batch_size) return;

  const auto index_ptr = static_cast<const T*>(params.indices) + item_id * params.stride_indices;
  PDLWaitPrimary<kUsePDL>();

  const auto index = *index_ptr;
  // A stale/OOB slot id would cause an illegal memory access in the store below;
  // fail fast at the culprit instead. always-on (kvcache JIT compiles without NDEBUG).
  assert(index >= 0 && index < params.size_limit);
  if (index == params.reserved_skip_index) {
    PDLTriggerSecondary<kUsePDL>();
    return;
  }

  const float k_inv = params.k_scale != nullptr ? 1.0f / *params.k_scale : params.k_inv_scale;
  const float v_inv = params.v_scale != nullptr ? 1.0f / *params.v_scale : params.v_inv_scale;

  const auto k_src = static_cast<const TSrc*>(params.k) + item_id * params.stride_k;
  const auto v_src = static_cast<const TSrc*>(params.v) + item_id * params.stride_v;
  const auto k_dst = static_cast<TDst*>(params.k_cache) + index * params.stride_cache;
  const auto v_dst = static_cast<TDst*>(params.v_cache) + index * params.stride_cache;

  const auto gmem_src = tile::Memory<src_vec_t>::warp();
  const auto gmem_dst = tile::Memory<dst_vec_t>::warp();

  // Clip to the finite FP8 range before conversion (same convention as
  // per_tensor_quant_fp8): saturate instead of overflowing to NaN.
  const auto quant_vec = [&](const src_vec_t& in, const float inv_scale) {
    dst_vec_t out;
#pragma unroll
    for (uint32_t j = 0; j < kVecElems; ++j) {
      const float value = static_cast<float>(in[j]) * inv_scale;
      out[j] = static_cast<TDst>(math::max(math::min(value, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX));
    }
    return out;
  };

#pragma unroll
  for (int64_t i = 0; i < kLoopCount; ++i) {
    gmem_dst.store(k_dst, quant_vec(gmem_src.load(k_src, i), k_inv), i);
    gmem_dst.store(v_dst, quant_vec(gmem_src.load(v_src, i), v_inv), i);
  }

  // handle the epilogue if any
  if constexpr (kTailVecs > 0) {
    if (gmem_src.in_bound(kVecsPerRow, kLoopCount)) {
      gmem_dst.store(k_dst, quant_vec(gmem_src.load(k_src, kLoopCount), k_inv), kLoopCount);
      gmem_dst.store(v_dst, quant_vec(gmem_src.load(v_src, kLoopCount), v_inv), kLoopCount);
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kRowElems, typename TSrc, typename TDst, bool kUsePDL>
struct StoreKVCacheQuantKernel {
  static constexpr uint32_t kVecElems = 16 / sizeof(TSrc);
  static_assert(kRowElems > 0 && kRowElems % kVecElems == 0, "Row must be a multiple of the vector width");

  template <typename T>
  static constexpr auto kernel = store_kvcache_quant<kRowElems, TSrc, TDst, kUsePDL, T>;

  static void
  run(const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::Optional<tvm::ffi::TensorView> k_scale,
      const tvm::ffi::Optional<tvm::ffi::TensorView> v_scale,
      const double k_inv_scale,
      const double v_inv_scale,
      const int64_t size_limit,
      const int64_t reserved_skip_index) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto D = SymbolicSize{"element_size"};
    auto KS = SymbolicSize{"k_stride"};
    auto VS = SymbolicSize{"v_stride"};
    auto S = SymbolicSize{"cache_stride"};
    auto I = SymbolicSize{"indices_stride"};
    auto device = SymbolicDevice{};
    auto indice_dtype = SymbolicDType{};
    // CUDA only: the fp8 conversion relies on the __nv_fp8 conversion
    // operators; on ROCm fp8_e4m3_t is a plain byte type (see utils.cuh).
    device.set_options<kDLCUDA>();

    TensorMatcher({B, D})  //
        .with_strides({KS, 1})
        .with_dtype<TSrc>()
        .with_device(device)
        .verify(k);
    TensorMatcher({B, D})  //
        .with_strides({VS, 1})
        .with_dtype<TSrc>()
        .with_device(device)
        .verify(v);
    TensorMatcher({-1, D})  //
        .with_strides({S, 1})
        .with_dtype<TDst>()
        .with_device(device)
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({B})  //
        .with_strides({I})
        .with_dtype<int32_t, int64_t>(indice_dtype)
        .with_device(device)
        .verify(indices);
    for (const auto& scale : {k_scale, v_scale}) {
      if (scale.has_value()) {
        TensorMatcher({1})  //
            .with_dtype<float>()
            .with_device(device)
            .verify(scale.value());
      }
    }

    RuntimeCheck(kRowElems == D.unwrap());

    const auto params = StoreKVCacheQuantParams{
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .k_scale = k_scale.has_value() ? static_cast<const float*>(k_scale.value().data_ptr()) : nullptr,
        .v_scale = v_scale.has_value() ? static_cast<const float*>(v_scale.value().data_ptr()) : nullptr,
        .k_inv_scale = static_cast<float>(k_inv_scale),
        .v_inv_scale = static_cast<float>(v_inv_scale),
        .stride_k = KS.unwrap(),
        .stride_v = VS.unwrap(),
        .stride_cache = S.unwrap(),
        .stride_indices = I.unwrap(),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .size_limit = size_limit,
        .reserved_skip_index = reserved_skip_index,
    };
    const auto use_int32 = indice_dtype.is_type<int32_t>();
    const auto kernel_ptr = use_int32 ? kernel<int32_t> : kernel<int64_t>;
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto num_warps = [&] {
      const auto sm_count = runtime::get_sm_count(device.unwrap().device_id);
#pragma unroll
      for (uint32_t n : {1, 2, 4}) {
        if (batch_size <= sm_count * n) return n;
      }
      return 8u;
    }();
    const auto num_blocks = div_ceil(batch_size, num_warps);
    LaunchKernel(num_blocks, {device::kWarpThreads, num_warps}, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel_ptr, params);
  }
};

}  // namespace sglang
