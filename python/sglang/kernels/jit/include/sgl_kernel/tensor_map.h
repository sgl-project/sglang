/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <sgl_kernel/cuda_check.h>

#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ================= common/tensor_map.h =================
// Helpers around cuTensorMapEncodeTiled — wraps the driver-API call so the rest
// of the kernels module can describe a 2D tile in one line.
//
// References:
//   CUDA Driver API: cuTensorMapEncodeTiled
//   PTX ISA 9.2 §5.5 Tensors and §5.5.2 Tensor Access Modes
//   recipes/tma_alignment_rules/README.md — full derivation of every rule here.
//
// ENCODED RULES (per repo rule #10 — encode what can be encoded):
//
//   tmap::box_inner_bytes<Swizzle>()
//       Returns the canonical box inner byte width for this swizzle mode.
//       Rule: box_inner_bytes = swizzle_width for swizzled paths; 16 for NONE
//       (minimum legal inner box).  Use this as BLOCK_K_BYTES in your kernel
//       instead of computing manually.
//
//   tmap::block_k<Swizzle, Dtype>()
//       Returns box_inner_bytes / element_size_bytes.  For _ALIGN8B (FP4 dense),
//       elements are 0.5 bytes so result is box_inner_bytes * 2.  For _ALIGN16B
//       (FP4 padded) the boxDim is ALWAYS 128 U4 elements regardless of swizzle.
//       static_assert fires at compile time on illegal Swizzle × Dtype combos.
//
//   tmap::validate_shape<Dtype>(global_k, stride_bytes)
//       Runtime check — mirrors what the encoder would reject.  Call before
//       tmap::encode_tiled_2d to get a descriptive error instead of the
//       driver's opaque CUDA_ERROR_INVALID_VALUE.  Returns bool; the overload
//       tmap::check_shape<Dtype>(global_k, stride_bytes) aborts on failure.


namespace tmap {

// ---------------------------------------------------------------------------
// Compile-time swizzle ↔ box-width accessors
// ---------------------------------------------------------------------------

// Returns the canonical "box inner bytes" for a given swizzle mode.
// For NONE: the minimum legal inner box is 16 bytes.
// For 32B / 64B / 128B: the canonical value equals the swizzle width.
// Rule: always set box_inner_bytes == swizzle_width for MMA-feed paths
// (the MMA smem descriptor assumes equality; smaller-than-swizzle inner boxes
// are encoder-legal but silently corrupt the MMA load).
template <CUtensorMapSwizzle Swz>
constexpr int box_inner_bytes() {
    // Use ternary so the full expression evaluates without triggering static_assert
    // on a branch that is never taken.  All four documented swizzle modes supported.
    return (Swz == CU_TENSOR_MAP_SWIZZLE_NONE)  ? 16 :
           (Swz == CU_TENSOR_MAP_SWIZZLE_32B)   ? 32 :
           (Swz == CU_TENSOR_MAP_SWIZZLE_64B)   ? 64 :
           (Swz == CU_TENSOR_MAP_SWIZZLE_128B)  ? 128 : 0;
}

// Returns the number of elements per inner box (= BLOCK_K for a K-major
// MMA feed operand).
//
// Dtype-specific notes:
//   - UINT8 / UINT16 / BFLOAT16 / FLOAT16 / UINT32:
//       block_k = box_inner_bytes / sizeof(element).
//   - 16U4_ALIGN8B (FP4 dense, kind::mxf4):
//       elements are 4-bit (0.5 bytes); block_k = box_inner_bytes * 2.
//       Legal swizzles: NONE / 32B / 64B / 128B — all accepted by encoder.
//   - 16U4_ALIGN16B (FP4 with-padding, kind::mxf8f6f4):
//       boxDim[0] is FIXED at 128 U4 elements by the encoder (no other value
//       is accepted).  The only production swizzle is 128B (NONE also legal).
//       32B is rejected; 64B is encoder-accepted but undocumented — avoid.
//       static_assert rejects 32B at compile time.
//
// Usage:
//   constexpr int BLOCK_K = tmap::block_k<CU_TENSOR_MAP_SWIZZLE_128B,
//                                         CU_TENSOR_MAP_DATA_TYPE_BFLOAT16>();
// Helper: compile-time element size for regular TMA dtypes (in bytes).
// Returns 0 for FP4 sub-byte types (handled separately in block_k).
template <CUtensorMapDataType Dtype>
constexpr int tma_elem_bytes() {
    return
        (Dtype == CU_TENSOR_MAP_DATA_TYPE_UINT8)    ? 1 :
        (Dtype == CU_TENSOR_MAP_DATA_TYPE_UINT16   ||
         Dtype == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 ||
         Dtype == CU_TENSOR_MAP_DATA_TYPE_FLOAT16)  ? 2 :
        (Dtype == CU_TENSOR_MAP_DATA_TYPE_UINT32   ||
         Dtype == CU_TENSOR_MAP_DATA_TYPE_INT32    ||
         Dtype == CU_TENSOR_MAP_DATA_TYPE_FLOAT32)  ? 4 :
        (Dtype == CU_TENSOR_MAP_DATA_TYPE_UINT64   ||
         Dtype == CU_TENSOR_MAP_DATA_TYPE_INT64    ||
         Dtype == CU_TENSOR_MAP_DATA_TYPE_FLOAT64)  ? 8 : 0;
}

template <CUtensorMapSwizzle Swz, CUtensorMapDataType Dtype>
constexpr int block_k() {
    // _ALIGN16B: encoder mandates boxDim[0] == 128 regardless of swizzle.
    // 32B is encoder-rejected; 64B is encoder-accepted but undocumented — both
    // are caught below.  The static_assert fires at instantiation time for the
    // caller that passes 32B, giving a readable error instead of an encoder abort.
    if constexpr (Dtype == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B) {
        static_assert(Swz != CU_TENSOR_MAP_SWIZZLE_32B,
            "16U4_ALIGN16B (FP4 padded) with 32B swizzle is rejected by the encoder. "
            "Use SWIZZLE_128B (production) or SWIZZLE_NONE (load-only path).");
        return 128; // fixed by the encoder — there is no other legal value
    } else if constexpr (Dtype == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B) {
        // FP4 dense: 0.5 bytes/element → block_k = box_inner_bytes * 2.
        return box_inner_bytes<Swz>() * 2;
    } else {
        // Regular dtypes: derive from byte width.
        constexpr int eb = tma_elem_bytes<Dtype>();
        static_assert(eb > 0,
            "tmap::block_k: unknown TMA dtype — cannot compute element count");
        return box_inner_bytes<Swz>() / eb;
    }
}

// ---------------------------------------------------------------------------
// Runtime shape validation (call before encode_tiled_2d / encode_tiled_3d)
// ---------------------------------------------------------------------------

// Returns true if (global_k, stride_bytes) satisfies the encoder's constraints
// for the given dtype.  The check mirrors what cuTensorMapEncodeTiled would
// return CUDA_ERROR_INVALID_VALUE for (so you get a readable error first).
//
// This is a host-side runtime check; it cannot be constexpr because K / stride
// are typically runtime values.
template <CUtensorMapDataType Dtype>
inline bool validate_shape(uint64_t global_k, uint64_t stride_bytes) {
    if constexpr (Dtype == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B) {
        // globalDim[0] must be multiple of 128 U4 elements.
        if (global_k % 128 != 0) return false;
        // stride must be multiple of 32 bytes.
        if (stride_bytes % 32 != 0) return false;
    } else if constexpr (Dtype == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B) {
        // globalDim[0] must be even (nibbles → whole bytes).
        if (global_k % 2 != 0) return false;
        // stride must be multiple of 16 bytes.
        if (stride_bytes % 16 != 0) return false;
    } else {
        // Regular dtypes: stride multiple of 16 bytes only.
        if (stride_bytes % 16 != 0) return false;
    }
    return true;
}

// Like validate_shape but aborts with a diagnostic message on failure.
template <CUtensorMapDataType Dtype>
inline void check_shape(uint64_t global_k, uint64_t stride_bytes,
                        const char* caller = "tmap::check_shape") {
    if (!validate_shape<Dtype>(global_k, stride_bytes)) {
        std::fprintf(stderr,
            "%s: illegal TMA shape — dtype=%d global_k=%llu stride_bytes=%llu\n"
            "  See recipes/tma_alignment_rules/README.md for padding rules.\n",
            caller, (int)Dtype,
            (unsigned long long)global_k, (unsigned long long)stride_bytes);
        std::abort();
    }
}

// ---------------------------------------------------------------------------
// encode_tiled_2d / encode_tiled_3d — thin wrappers around the driver call
// ---------------------------------------------------------------------------

inline CUtensorMap encode_tiled_2d(void* global_ptr,
                                   CUtensorMapDataType dtype,
                                   uint64_t global_rows, uint64_t global_cols,
                                   uint64_t row_stride_bytes,
                                   uint32_t box_rows, uint32_t box_cols,
                                   CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE,
                                   CUtensorMapL2promotion promo = CU_TENSOR_MAP_L2_PROMOTION_NONE) {
    // Coordinate convention: dim 0 = innermost (cols), dim 1 = outer (rows).
    // The driver treats globalDim[0] as the stride-1 axis.
    cuuint64_t global_dim[2]    = { global_cols, global_rows };
    // globalStrides has length (rank - 1) — the stride for dim 1+, in BYTES.
    cuuint64_t global_strides[1] = { row_stride_bytes };
    cuuint32_t box_dim[2]        = { box_cols, box_rows };
    cuuint32_t element_strides[2]= { 1, 1 };

    CUtensorMap m{};
    CU_CHECK(cuTensorMapEncodeTiled(
        &m, dtype, /*rank=*/2, global_ptr,
        global_dim, global_strides, box_dim, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return m;
}

// 3D variant. Coordinate convention: dim 0 = innermost (cols), dim 1 = middle
// (rows), dim 2 = outermost (depth, e.g. expert index). `row_stride_bytes` is
// the stride of dim 1; `depth_stride_bytes` is the stride of dim 2 (typically
// `rows * row_stride_bytes`).
inline CUtensorMap encode_tiled_3d(void* global_ptr,
                                   CUtensorMapDataType dtype,
                                   uint64_t global_depth, uint64_t global_rows, uint64_t global_cols,
                                   uint64_t row_stride_bytes, uint64_t depth_stride_bytes,
                                   uint32_t box_rows, uint32_t box_cols,
                                   CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
    cuuint64_t global_dim[3]    = { global_cols, global_rows, global_depth };
    // globalStrides has length (rank - 1); dim 1+ strides in BYTES.
    cuuint64_t global_strides[2] = { row_stride_bytes, depth_stride_bytes };
    cuuint32_t box_dim[3]        = { box_cols, box_rows, 1u };
    cuuint32_t element_strides[3]= { 1, 1, 1 };

    CUtensorMap m{};
    CU_CHECK(cuTensorMapEncodeTiled(
        &m, dtype, /*rank=*/3, global_ptr,
        global_dim, global_strides, box_dim, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return m;
}


}  // namespace tmap
