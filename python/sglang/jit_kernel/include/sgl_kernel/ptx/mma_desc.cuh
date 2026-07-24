/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ================= common/mma_desc.cuh =================
// Builders for tcgen05.mma matrix descriptors. PTX ISA 9.2 §9.7.16.4.
// The public PTX spec has documentation errors in the descriptor bit layout;
// the bit-field layout below is verified via our gate tests.
//
// All builders are constexpr/__forceinline__ — zero runtime cost.


namespace ptx {

// ---- Major mode (instruction descriptor bits 15/16) -------------------------
//
// 0 = K-major (innermost dim is K) — the "TN" convention.
// 1 = MN-major (innermost is M for A, N for B) — the "NN" convention.
// FP4/FP6 (E2M1/E2M3/E3M2) only support K-major. FP8/F16/BF16/TF32 support both.
enum class Major : uint8_t {
    K  = 0,
    MN = 1,
};

// ---- Smem matrix descriptor bit layout ----------------------
//
// Bit layout (the public PTX spec has errors at bits 46-60;
// verified layout from gate tests and first-principles derivation):
//   0-13:  start_address >> 4
//   16-29: leading_byte_offset >> 4   (LBO)
//   32-45: stride_byte_offset >> 4    (SBO)
//   46-47: version (=1, Blackwell — must be set; small-shape MMAs silently
//          tolerate version=0, but larger shapes like 128B-swizzle
//          BLOCK_K>16 produce garbage without it.)
//   49-51: base_offset (0 if matrix is at the swizzle's natural boundary,
//          else (pattern_start_addr >> 7) & 0x7)
//   52:    lbo_mode (=0, relative byte offset — the legacy default)
//   61-63: layout_type code (0=None, 1=128B_BASE32B, 2=128B, 4=64B, 6=32B)
//
// Byte-offset fields encode as `(byte_value >> 4)` (= u128 units), 14 bits,
// so byte values up to 256K-1 in multiples of 16.
//
// `swizzle_bytes` is the byte period: 0 (none), 32, 64, or 128. Internally
// mapped to the 3-bit code at bits 61-63. Prefer the high-level helpers
// below (mma_smem_desc_k_major, ...) for typical setups; reach for this
// raw builder when you need something unusual.

__host__ __device__ static __forceinline__ constexpr uint64_t mma_smem_desc(
        uint32_t matrix_addr, uint32_t lbo, uint32_t sbo,
        uint32_t base_offset, int swizzle_bytes) {
    auto enc = [](uint32_t x) -> uint64_t {
        return (uint64_t)((x & 0x3FFFFu) >> 4);
    };
    // 3-bit layout_type code at bits 61-63.
    uint8_t code = (swizzle_bytes == 128) ? 2u
                 : (swizzle_bytes == 64)  ? 4u
                 : (swizzle_bytes == 32)  ? 6u
                 :                          0u;
    uint64_t d = 0;
    d |= enc(matrix_addr);                              // bits  0-13
    d |= enc(lbo) << 16;                                // bits 16-29
    d |= enc(sbo) << 32;                                // bits 32-45
    d |= uint64_t(1u) << 46;                            // bits 46-47 = version = 1
    d |= uint64_t(base_offset & 0x7u) << 49;            // bits 49-51
    d |= uint64_t(code & 0x7u) << 61;                   // bits 61-63
    return d;
}


// ---- High-level: K-major operand (inner = K) -------------------------------
//
// Used for A (M, K) and B (N, K) in TN GEMMs. Computes LBO/SBO and enforces
// the swizzle-vs-K_BYTES invariant from the layout parameters; static_assert
// fires at compile time on misuse.
//
//   K_BYTES        = BLOCK_K * sizeof(T)
//   Required       SWIZZLE_BYTES == K_BYTES  (DeepGEMM mma/sm90.cuh:251)
//   Derived        LBO = 0,  SBO = 8 * K_BYTES
//
// The `T` template parameter is a size proxy — `uint16_t` for any 16-bit
// dtype (BF16/FP16), `uint8_t` for FP8. The actual dtype semantics live in
// the instruction descriptor (mma_inst_desc_*), not the smem descriptor.
//
// Examples:
//   BF16/FP16, BLOCK_K=16, 32B  → mma_smem_desc_k_major<uint16_t, 16, 32>(addr)
//   BF16/FP16, BLOCK_K=64, 128B → mma_smem_desc_k_major<uint16_t, 64, 128>(addr)
//   FP8,        BLOCK_K=32, 32B → mma_smem_desc_k_major<uint8_t,  32, 32>(addr)
template <typename T, int BLOCK_K, int SWIZZLE_BYTES>
__host__ __device__ static __forceinline__ constexpr uint64_t
mma_smem_desc_k_major(uint32_t addr, uint32_t base_offset = 0) {
    constexpr int K_BYTES = BLOCK_K * int(sizeof(T));
    static_assert(SWIZZLE_BYTES == K_BYTES,
                  "K-major requires swizzle bytes == BLOCK_K * sizeof(T) "
                  "(DeepGEMM mma/sm90.cuh:251).");
    return mma_smem_desc(addr, /*lbo=*/0u, /*sbo=*/8u * uint32_t(K_BYTES),
                         base_offset, SWIZZLE_BYTES);
}


// ---- Instruction descriptor (PTX ISA Table 44) -----------

enum class FP8Type : uint8_t { E4M3 = 0, E5M2 = 1, E2M3 = 3, E3M2 = 4, E2M1 = 5 };
enum class F16Type : uint8_t { F16  = 0, BF16 = 1 };       // for kind::f16
enum class DType   : uint8_t { F16  = 0, F32  = 1, S32  = 2 };

__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_f8f6f4(
        uint32_t M, uint32_t N,
        FP8Type a_type, FP8Type b_type,
        DType   d_type   = DType::F32,
        Major   a_major  = Major::K,
        Major   b_major  = Major::K,
        bool    negate_a = false,
        bool    negate_b = false) {
    uint32_t d = 0;
    // bits 0-1 sparse_id2, 2 sparse_flag, 3 saturate (all 0 here)
    // bits 4-5 c_format (D matrix dtype)
    d |= (static_cast<uint32_t>(d_type) & 0x3u) << 4;
    // bit 6 unused
    // bits 7-9 a_format
    d |= (static_cast<uint32_t>(a_type) & 0x7u) << 7;
    // bits 10-12 b_format
    d |= (static_cast<uint32_t>(b_type) & 0x7u) << 10;
    // bit 13 a_negate, bit 14 b_negate
    if (negate_a) d |= 1u << 13;
    if (negate_b) d |= 1u << 14;
    // bit 15 a_major (0 = K, 1 = MN), bit 16 b_major
    d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
    d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
    // bits 17-22 N >> 3 (so encode N=8 as 1, N=16 as 2, ..., N=256 as 32)
    d |= ((N >> 3) & 0x3Fu) << 17;
    // bit 23 unused
    // bits 24-28 M >> 4 (so M=64 → 4, M=128 → 8, M=256 → 16)
    d |= ((M >> 4) & 0x1Fu) << 24;
    // bit 29 unused; bits 30-31 max_shift (0 for non-.ws)
    return d;
}

// ---- Instruction descriptor for kind::f16 (F16/BF16 inputs) ----------------
//
// Same Table 44 layout as kind::f8f6f4 but the atype/btype field encodes
// F16=0 / BF16=1 (vs E4M3=0 / E5M2=1 / E2M3=3 / E3M2=4 / E2M1=5 for f8f6f4).
// The MMA-Kind on the instruction itself disambiguates which decoding to use.
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_f16(
        uint32_t M, uint32_t N,
        F16Type a_type   = F16Type::BF16,
        F16Type b_type   = F16Type::BF16,
        DType   d_type   = DType::F32,
        Major   a_major  = Major::K,
        Major   b_major  = Major::K,
        bool    negate_a = false,
        bool    negate_b = false) {
    uint32_t d = 0;
    d |= (static_cast<uint32_t>(d_type)  & 0x3u) <<  4;
    d |= (static_cast<uint32_t>(a_type)  & 0x7u) <<  7;
    d |= (static_cast<uint32_t>(b_type)  & 0x7u) << 10;
    if (negate_a) d |= 1u << 13;
    if (negate_b) d |= 1u << 14;
    d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
    d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
    d |= ((N >> 3) & 0x3Fu) << 17;
    d |= ((M >> 4) & 0x1Fu) << 24;
    return d;
}

// ---- Instruction descriptor for kind::tf32 (TF32 inputs) -------------------
//
// Same Table 44 bit layout as kind::f16, but the atype/btype field encodes
// TF32 = 2 (vs F16=0 / BF16=1 for kind::f16). The operand is stored as fp32
// in smem (4 B/elem); the MMA truncates each fp32 word to the 19-bit tf32
// mantissa internally — no host-side cvt. D dtype is F32. Provenance:
// `kernels/qr/studies/inhouse_gemm/tf32_mma.cuh` (wave-26 cta_group::1
// wrapper proved TF32=2 in bits 7/10). The negate flags are accepted for
// API parity with the f16/f8 builders (TF32 negate is HW-supported).
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_tf32(
        uint32_t M, uint32_t N,
        DType   d_type   = DType::F32,
        Major   a_major  = Major::K,
        Major   b_major  = Major::K,
        bool    negate_a = false,
        bool    negate_b = false) {
    constexpr uint32_t TF32 = 2u;
    uint32_t d = 0;
    d |= (static_cast<uint32_t>(d_type)  & 0x3u) <<  4;
    d |= TF32 <<  7;                                      // atype = TF32 = 2
    d |= TF32 << 10;                                      // btype = TF32 = 2
    if (negate_a) d |= 1u << 13;
    if (negate_b) d |= 1u << 14;
    d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
    d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
    d |= ((N >> 3) & 0x3Fu) << 17;
    d |= ((M >> 4) & 0x1Fu) << 24;
    return d;
}

// ---- Block-scaled instruction descriptor (Table 45) -------------------------
//
// Note carefully: kind::mxf8f6f4 uses a DIFFERENT instruction-descriptor format
// than kind::f8f6f4 (Table 44). The shifted M field moves: f8f6f4 puts
// `M >> 4` at bits [24:28]; mxf8f6f4 puts `M >> 7` at bits [27:28] with bits
// [24:26] reserved (= 0). The scale fields (a_sf_id at [29:30], scale_format
// at [23], b_sf_id at [4:5]) replace the f8f6f4 fields at the same positions.
//
// Reference: PTX ISA 9.2 Table 45 (verified via gate tests).
//
// `tmem_sfa_addr` / `tmem_sfb_addr` are the 32-bit TMEM register addresses
// where SF tiles for A and B live. The top 2 bits of each become the
// `a_sf_id` / `b_sf_id` fields in the descriptor — matching the values
// passed in the runtime `[sf_a]`, `[sf_b]` MMA operands.

enum class ScaleFormat : uint8_t { E4M3 = 0, E8M0 = 1 };


// Instruction descriptor for `tcgen05.mma.kind::mxf4nvf4.block_scale.block16`.
// Unified builder covering ALL THREE shape paths kind::mxf4nvf4 supports:
//
//   * K=64 dense (block16, scale_vec::4X)         — k_dim=0, sparse=0
//     (Table 46 line 850; Table 58 line 2702 — Mx4 / 4xN SF layout)
//   * K=96 dense (block16, scale_vec::6X; sm_103a) — k_dim=1, sparse=0
//     (refs/sections/9_7_16_*:224-229 — K=96 sm_103a-exclusive; Table 58
//      line 2703 — Mx6 / 6xN SF layout)
//   * K=128 sparse 4:8 on A (block16, scale_vec::4X; sm_103a) — sparse=1
//     (refs/sections/9_7_16_*:3971-3975 — sm_103a support;
//      §9.7.16.2.1 lines 224-229 — Table 41 256xNxK at K=128 sparse;
//      Table 46 lines 849-850 — bit 31 k_dim AND bit 2 sparse_flag;
//      Table 58 lines 2702-2705 — Mx4 / 4xN at K=64/K=128, Mx6/6xN at K=96)
//
// Sparse semantics (Table 46 line 849-850): for mxf4nvf4 with sparsity flag
// set, K=128 sparse and K=64 dense both encode bit-31 k_dim=0. The sparse
// flag at bit 2 disambiguates: sparse + k_dim=0 → K=128 sparse; dense +
// k_dim=0 → K=64 dense; dense + k_dim=1 → K=96 dense. K=96 sparse is NOT
// supported (Mx6 SF layout incompatible with sparse pipe; this helper
// rejects that via the static-friendly default ordering — K=96 sparse is
// constructable but undefined at runtime, caller's responsibility).
//
// Scale-format encoding (Table 59 lines 2776-2780): kind::mxf4nvf4 supports
// BOTH UE4M3 (bit 23 = 0; the NVIDIA-custom 8-bit unsigned E4M3 format
// unique to mxf4nvf4) AND UE8M0 (bit 23 = 1; same as kind::mxf4). This
// is a runtime choice; default UE4M3 matches the production NVFP4-out
// kernel (`kernels/fused_gemm_2cta_sf` mxf4_nvfp4out_path) at K=64 dense.
//
// The asm string for `tcgen05.mma.cta_group::*.kind::mxf4nvf4.block_scale.
// block16` is BYTE-IDENTICAL between K=64 and K=96 (proven by the §21.3
// micro-cycle bench reusing the kind::mxf4 wrapper at K=96 with only bit 31
// flipped — `recipes/_other/cublas_nvfp4_ref/README.md` §21.2). The sparse
// variant uses a DIFFERENT asm string (`tcgen05.mma.sp...`, 4 operands;
// see `tcgen05_mma_mxf4nvf4_block16_sp[_2sm]` in common/ptx/tcgen05.cuh).
// K-dim selection lives in idesc bit 31; sparse selection lives in BOTH
// idesc bit 2 AND the sp variant of the asm string.
//
// Promotion provenance: K=64 dense extracted from
// `kernels/fused_gemm_2cta_sf/kernel.cu:359-373` (which had no `k_dim` arg);
// K=96 dense added for `recipes/_other/microbench_tcgen05_mma` R1 SFA_ID-walk
// verification + LEVER 9 K=96 NVFP4 kernel
// (`kernels/fused_gemm_2cta_sf/LEVER9_K96_DESIGN.md` Phase A); K=128
// sparse added for T11 Cat-D probe (`recipes/_other/cublas_nvfp4_ref` §21.10).
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_mxf4nvf4_block16(
        uint32_t M, uint32_t N,
        uint32_t tmem_sfa_addr = 0,
        uint32_t tmem_sfb_addr = 0,
        Major a_major = Major::K,
        Major b_major = Major::K,
        bool  negate_a = false,
        bool  negate_b = false,
        bool  k_dim    = false,                  // bit 31: 1 = K=96 dense (sm_103a-only)
        ScaleFormat sf = ScaleFormat::E4M3,      // bit 23: E4M3 = 0 = UE4M3 (NVFP4 native default); E8M0 = 1 = UE8M0
        bool  sparse   = false) {                // bit 2: 1 = K=128 sparse 4:8 on A (sm_103a-only)
    constexpr uint32_t MXF4_E2M1 = 1u;
    uint32_t d = 0;
    // bit 2 sparse flag (Table 46 line 849-850: Dense=0, Sparse=1). K=128
    // sparse encodes (sparse=1, k_dim=0); see header comment.
    if (sparse) d |= 1u << 2;
    // bits 4-5 b_sf_id = top 2 bits of tmem_sfb_addr (block16 + K=64/K=128
    // mandate 0 — Figures 233, 242; block16 + K=96 allows {0, 2} sub-byte
    // offset within the 4-byte TMEM SF word).
    d |= ((tmem_sfb_addr & 0xC0000000u) >> 30) << 4;
    // bits 7-9 a_format (E2M1 = 1 for kind::mxf4nvf4; Table 46 / MXF4Format).
    d |= MXF4_E2M1 << 7;
    // bits 10-11 b_format (E2M1 = 1).
    d |= MXF4_E2M1 << 10;
    if (negate_a) d |= 1u << 13;
    if (negate_b) d |= 1u << 14;
    // bits 15/16 transpose A/B. kind::mxf4nvf4 only supports K-major
    // (Table 53 line 2196: "Is Transpose A/B supported = No"). Caller
    // must keep both at K (= 0).
    d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
    d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
    // bits 17-22 N >> 3.
    d |= ((N >> 3) & 0x3Fu) << 17;
    // bit 23 scale_format (E4M3 = 0 = UE4M3 NVFP4 native; E8M0 = 1 = UE8M0
    // shared with kind::mxf4). Table 59 lines 2776-2780: both valid for
    // kind::mxf4nvf4. The default (E4M3) matches the production NVFP4-out
    // kernel at K=64 dense.
    d |= (static_cast<uint32_t>(sf) & 0x1u) << 23;
    // bits 24-26 reserved (= 0).
    // bits 27-28 m_dim = M >> 7 (M=128 → 1, M=256 → 2). Encode via (M >> 4)
    // << 24 to land at bits 27-28 with bits 24-26 = 0
    // (for M ∈ {128, 256} only).
    d |= ((M >> 4) & 0x1Fu) << 24;
    // bits 29-30 a_sf_id = top 2 bits of tmem_sfa_addr (block16 + K=64/K=128
    // mandate 0; block16 + K=96 allows {0, 2}).
    d |= ((tmem_sfa_addr & 0xC0000000u) >> 30) << 29;
    // bit 31 k_dim:
    //   dense   k_dim=0 → K=64
    //   dense   k_dim=1 → K=96  (sm_103a-only)
    //   sparse  k_dim=0 → K=128 sparse 4:8 on A  (sm_103a-only)
    // K=96 sparse is NOT supported.
    if (k_dim) d |= 1u << 31;
    return d;
}


}  // namespace ptx
