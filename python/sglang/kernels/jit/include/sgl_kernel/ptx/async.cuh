// Async copies (TMA, cp.async, st.async), the async-proxy fence, and host-side TMA tensor maps (`host::tma`).
// CUDA only: a ROCm kernel must not include it.

#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/ptx/barrier.cuh>

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <initializer_list>

namespace sglang {

namespace device::ptx {

// ---- TMA (PTX ISA 9.7.9.25) ------------------------------------------------

/// Prefetches the tensor-map descriptor, so the first TMA does not pay its fetch.
SGL_DEVICE void prefetch_tensormap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

/// One 2D tile of `tmap` into `dst`, completing on `bar`. `x` is the stride-1 (innermost) coordinate.
template <SmemAddress Addr, MBarAddress Bar>
SGL_DEVICE void cp_async_bulk_tensor_2d(Addr dst, const CUtensorMap* tmap, int32_t x, int32_t y, Bar bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"  //
      ::"r"(to_shared(dst)),
      "l"(tmap),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar))
      : "memory");
}

/// One 3D tile at coordinates (x, y, z), innermost first, of `tmap` into `dst`, completing on `bar`.
template <SmemAddress Addr, MBarAddress Bar>
SGL_DEVICE void cp_async_bulk_tensor_3d(Addr dst, const CUtensorMap* tmap, int32_t x, int32_t y, int32_t z, Bar bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"  //
      ::"r"(to_shared(dst)),
      "l"(tmap),
      "r"(x),
      "r"(y),
      "r"(z),
      "r"(to_shared(bar))
      : "memory");
}

/// The same 2D tile with an L2 cache-policy hint (`policy` from `createpolicy_evict_first` or similar).
template <SmemAddress Addr, MBarAddress Bar>
SGL_DEVICE void
cp_async_bulk_tensor_2d_hint(Addr dst, const CUtensorMap* tmap, int32_t x, int32_t y, Bar bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3}], [%4], %5;"  //
      ::"r"(to_shared(dst)),
      "l"(tmap),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar)),
      "l"(policy)
      : "memory");
}

/// 1D bulk copy global -> shared, completing on `bar` (16-byte aligned, `bytes % 16 == 0`).
template <SmemAddress Addr, MBarAddress Bar>
SGL_DEVICE void cp_async_bulk_g2s(Addr dst, const void* src, uint32_t bytes, Bar bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"  //
               ::"r"(to_shared(dst)),
               "l"(src),
               "r"(bytes),
               "r"(to_shared(bar))
               : "memory");
}

/// Prefetch `bytes` (16-byte granular) of global memory into L2.
SGL_DEVICE void cp_async_bulk_prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

// ---- cp.async (PTX ISA 9.7.9.24) -------------------------------------------

/// 16 B global -> shared, cached in L2 only (`.cg`): the usual choice for data read once per CTA.
template <SmemAddress Addr>
SGL_DEVICE void cp_async_16B(Addr dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(to_shared(dst)), "l"(src) : "memory");
}

// `.ca` variants: cached in L1 too (the only form for 4 B and 8 B copies).
template <SmemAddress Addr>
SGL_DEVICE void cp_async_ca_4B(Addr dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(to_shared(dst)), "l"(src) : "memory");
}
template <SmemAddress Addr>
SGL_DEVICE void cp_async_ca_8B(Addr dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(to_shared(dst)), "l"(src) : "memory");
}
template <SmemAddress Addr>
SGL_DEVICE void cp_async_ca_16B(Addr dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(to_shared(dst)), "l"(src) : "memory");
}

/// `cp_async_16B` with an L2 cache-policy hint (`policy` from `createpolicy_evict_first` or similar).
template <SmemAddress Addr>
SGL_DEVICE void cp_async_16B_hint(Addr dst, const void* src, uint64_t policy) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"  //
               ::"r"(to_shared(dst)),
               "l"(src),
               "l"(policy)
               : "memory");
}

/// Waits for all of this thread's prior cp.async copies (committed or not).
SGL_DEVICE void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}

/// Arrives on `bar` once this thread's prior cp.async complete; without `.noinc` it does not count toward the
/// expected arrivals.
template <MBarAddress Bar>
SGL_DEVICE void cp_async_mbarrier_arrive(Bar bar) {
  asm volatile("cp.async.mbarrier.arrive.b64 [%0];" ::"r"(to_shared(bar)) : "memory");
}

/// Arrives on `bar` once this thread's prior cp.async complete, counting as one of the expected arrivals (`.noinc`).
template <MBarAddress Bar>
SGL_DEVICE void cp_async_mbarrier_arrive_noinc(Bar bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(to_shared(bar)) : "memory");
}

// ---- L2 cache policies (PTX ISA 9.7.9.17) ------------------------------------

/// A policy for `.L2::cache_hint` operands: lines this access installs are evicted first (data read once).
SGL_DEVICE uint64_t createpolicy_evict_first() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

// ---- proxy fence -----------------------------------------------------------

/// Orders this thread's generic-proxy smem accesses before its later async-proxy (TMA / UMMA / tcgen05.cp) ones.
SGL_DEVICE void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ---- st.async (PTX ISA 9.7.9.12) -------------------------------------------
// Store into a (possibly remote) CTA's smem, completing on that CTA's mbarrier. `dst` and `bar` must be in the same
// CTA: shared::cluster addresses from `mapa_shared`, or local smem pointers. `V` is any type of the stored size (copied
// with memcpy, so it need not be trivially copyable, e.g. bf16x2 vectors).

template <SmemAddress Addr, typename V, MBarAddress Bar>
SGL_DEVICE void st_async_4B(Addr dst, const V& value, Bar bar) {
  static_assert(sizeof(V) == 4);
  uint32_t packed;
  __builtin_memcpy(&packed, &value, sizeof(packed));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"  //
               ::"r"(to_shared(dst)),
               "r"(packed),
               "r"(to_shared(bar))
               : "memory");
}

template <SmemAddress Addr, typename V, MBarAddress Bar>
SGL_DEVICE void st_async_8B(Addr dst, const V& value, Bar bar) {
  static_assert(sizeof(V) == 8);
  uint64_t packed;
  __builtin_memcpy(&packed, &value, sizeof(packed));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];"  //
               ::"r"(to_shared(dst)),
               "l"(packed),
               "r"(to_shared(bar))
               : "memory");
}

template <SmemAddress Addr, typename V, MBarAddress Bar>
SGL_DEVICE void st_async_16B(Addr dst, const V& value, Bar bar) {
  static_assert(sizeof(V) == 16);
  ulonglong2 packed;
  __builtin_memcpy(&packed, &value, sizeof(packed));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b64 [%0], {%1, %2}, [%3];"  //
               ::"r"(to_shared(dst)),
               "l"(packed.x),
               "l"(packed.y),
               "r"(to_shared(bar))
               : "memory");
}

}  // namespace device::ptx

}  // namespace sglang

namespace sglang::host::tma {

/// The driver's `cuTensorMapEncodeTiled`, resolved once through the runtime.
inline PFN_cuTensorMapEncodeTiled_v12000 encode_tiled_fn() {
  static const auto fn = [] {
    void* sym = nullptr;
    cudaDriverEntryPointQueryResult status;
    CHECK_CUDA(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &sym, 12000, cudaEnableDefault, &status));
    CHECK_HOST(status == cudaDriverEntryPointSuccess && sym != nullptr) << "cuTensorMapEncodeTiled unavailable";
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(sym);
  }();
  return fn;
}

struct TensorMapParams {
  CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  // NOTE: it's more common to use promote 256B, so we set that as default rather than NONE.
  CUtensorMapL2promotion l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
  CUtensorMapFloatOOBfill oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
  // Traversal strides per dimension, innermost first; empty (the usual case) means 1 in every dimension.
  std::initializer_list<cuuint32_t> elem_strides = {};
};

/// A tiled tensor map of `base`. `dims` and `box` list the dimensions innermost first; `strides` holds the byte
/// strides of dimensions 1 .. rank - 1. The rest comes from `params`, e.g. `{.swizzle = CU_TENSOR_MAP_SWIZZLE_128B}`.
inline CUtensorMap make_map(
    const void* base,
    CUtensorMapDataType dtype,
    std::initializer_list<cuuint64_t> dims,
    std::initializer_list<cuuint64_t> strides,
    std::initializer_list<cuuint32_t> box,
    const TensorMapParams& params = {}) {
  const auto rank = static_cast<uint32_t>(dims.size());
  const auto num_elem_strides = params.elem_strides.size();
  CHECK_HOST(rank >= 1 && rank <= 5) << rank << " not in [1, 5]";
  CHECK_HOST(strides.size() == rank - 1 && box.size() == rank)
      << "tensor map rank " << rank << " with " << strides.size() << " strides and " << box.size() << " box dims";
  CHECK_HOST(num_elem_strides == 0 || num_elem_strides == rank)
      << "tensor map rank " << rank << " with " << num_elem_strides << " element strides";
  static constexpr cuuint32_t kUnitStrides[5] = {1, 1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult res = encode_tiled_fn()(
      &map,
      dtype,
      rank,
      const_cast<void*>(base),
      dims.begin(),
      strides.begin(),
      box.begin(),
      num_elem_strides == 0 ? kUnitStrides : params.elem_strides.begin(),
      params.interleave,
      params.swizzle,
      params.l2_promotion,
      params.oob_fill);
  CHECK_HOST(res == CUDA_SUCCESS) << "cuTensorMapEncodeTiled failed: " << static_cast<int>(res);
  return map;
}

}  // namespace sglang::host::tma
