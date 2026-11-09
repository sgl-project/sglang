// Asynchronous copies: TMA (tensor and 1D bulk), cp.async, and the proxy fence that orders generic-proxy smem
// writes before async-proxy reads. Copies that complete on an mbarrier take it from ptx/barrier.cuh. The host side
// builds the TMA tensor maps (`host::tma`). CUDA only: a ROCm kernel must not include it.

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

// ---- TMA (PTX ISA §9.7.9.25) -----------------------------------------------

/// Warms the cache line holding the tensor-map descriptor, so the first TMA does not pay the descriptor fetch on top
/// of the DRAM latency.
SGL_DEVICE void prefetch_tensormap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

/// One 2D tile at coordinates (x = inner, y = outer) of `tmap` into `dst_smem`, completing on `bar`. The map's
/// globalDim is (inner, outer), dim 0 the stride-1 axis; swapped coordinates load a transposed tile with plausible
/// magnitudes and scrambled pairing.
SGL_DEVICE void
cp_async_bulk_tensor_2d(uint32_t dst_smem, const CUtensorMap* tmap, int32_t x, int32_t y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
          dst_smem),
      "l"(tmap),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar))
      : "memory");
}

/// 1D bulk copy global -> shared, completing on `bar` (16-byte aligned, `bytes % 16 == 0`).
SGL_DEVICE void cp_async_bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(to_shared(dst)),
      "l"(src),
      "r"(bytes),
      "r"(to_shared(bar))
      : "memory");
}

/// Prefetch `bytes` (16-byte granular) of global memory into L2.
SGL_DEVICE void cp_async_bulk_prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

// ---- cp.async (PTX ISA §9.7.9.24) ------------------------------------------

/// 16 B global -> shared, cached in L2 only (`.cg`): the usual choice for data read once per CTA.
SGL_DEVICE void cp_async_16B(uint32_t dst_smem, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst_smem), "l"(src) : "memory");
}

// `.ca` variants: cached in L1 too (the only form for 4 B and 8 B copies).
SGL_DEVICE void cp_async_ca_4B(uint32_t dst_smem, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst_smem), "l"(src) : "memory");
}
SGL_DEVICE void cp_async_ca_8B(uint32_t dst_smem, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst_smem), "l"(src) : "memory");
}
SGL_DEVICE void cp_async_ca_16B(uint32_t dst_smem, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst_smem), "l"(src) : "memory");
}

/// Arrives on `bar` once this thread's prior cp.async completed; the pending count is raised first, so the barrier's
/// expected arrivals do not change.
SGL_DEVICE void cp_async_mbarrier_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.b64 [%0];" ::"r"(to_shared(bar)) : "memory");
}

// ---- proxy fence -----------------------------------------------------------

/// Orders this thread's generic-proxy smem accesses before its later async-proxy (TMA / UMMA / tcgen05.cp) ones.
SGL_DEVICE void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
