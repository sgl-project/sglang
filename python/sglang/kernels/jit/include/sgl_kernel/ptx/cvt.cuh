/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ================= common/ptx/cvt.cuh =================
// Templated cvt wrappers — pack 2 FP32 inputs into the dst dtype's packed
// representation in a single PTX `cvt` instruction.
//
// Naming: `cvt_pack_f32x2_to<Dst>` — explicit on BOTH sides:
//   - input  shape: `f32x2` (the 2 FP32 inputs)
//   - output shape: `<Dst>` is a tag whose `packed2_t` is the packed pair
//     (e.g. `e2m1` → `uint8_t` holding 2 nibbles), mirroring PTX's
//     `cvt.<dst>x2.f32` op naming. A call reads
//     `cvt_pack_f32x2_to<ptx::e2m1>(a, b)` = "cvt+pack f32x2 to e2m1(x2)".
// Renamed from the pre-2026-05-04 `cvt_pack_f32<Dst>` (rule #7 in
// README "Orchestrator rules from user feedback": elided "two inputs" /
// "pair output").
//
// Tag types parameterize the destination format. Each tag's `packed2_t`
// declares the right container size for its packed pair:
//
//   tag         PTX                                       packed2_t  half_bits
//   ptx::bf16   cvt.rn.bf16x2.f32                         uint32_t   16
//   ptx::e4m3   cvt.rn.satfinite.e4m3x2.f32               uint16_t    8
//   ptx::e5m2   cvt.rn.satfinite.e5m2x2.f32               uint16_t    8
//   ptx::e2m1   cvt.rn.satfinite.e2m1x2.f32               uint8_t     4
//   ptx::ue8m0  cvt.rz.satfinite.ue8m0x2.f32              uint16_t    8
//
// PER-FORMAT GOTCHAS (verified in ptx/d_cvt_pack/test):
//
//   bf16  : NaN propagates with quiet-NaN bit set. Round-to-nearest-even —
//           ties go to even, NOT to away (`65504.5` → `65504`, not `65505`).
//   e4m3  : .satfinite is mandatory. Max finite = 448; |x| > 448 saturates
//           (sign-preserved). NaN → 0x7F (no negative-NaN encoding).
//           Min normal = 2^-6; smaller magnitudes are subnormal.
//   e5m2  : Like e4m3 but ±57344 max, 2 mantissa bits. Common for gradients.
//   e2m1  : Only 7 distinct positive values (0, 0.5, 1, 1.5, 2, 3, 4, 6).
//           NaN → 0x7 = MAX_NORM positive (no NaN encoding in e2m1). Always
//           paired with a UE8M0 per-32-element scale factor (mxfp4).
//   ue8m0 : NO SIGN BIT — hardware uses |x|. So `-1.0f` → 0x7F (= 2^0 = 1.0),
//           NOT 0x00. Code = floor(log2(|x|)) + 127, clamped. NaN → 0xFF.
//           **Use `.rz` rounding, NOT `.rn`**: ptxas rejects `.rn.satfinite.
//           ue8m0x2.f32` as "Illegal rounding modifier" on sm_103a despite
//           the spec listing it.
//
// PACKED-PAIR BYTE ORDERING (the wrapper signature `(float a, float b)` does
// not reveal this — it bites you when storing the packed result to smem and
// reading it back as an array of the smaller dtype):
//
//   `cvt.{f16x2,bf16x2,e4m3x2,e5m2x2,e2m1x2,ue8m0x2}.f32 d, a, b` packs
//   cvt(a) into d's UPPER half-bits and cvt(b) into d's LOWER half-bits.
//   Stored to little-endian memory and read back as a contiguous array of
//   the half-dtype, the LOWER bits land at the smaller-byte offset (col i)
//   and UPPER at the larger (col i+1).
//
//   So if you have FP32 cells (c0, c1) destined for adjacent slots [i, i+1]
//   in natural order, pass them as `cvt_pack_f32x2_to<bf16>(c1, c0)` —
//   c1 → upper = slot i+1, c0 → lower = slot i. Off-by-one swaps every
//   adjacent pair (fused_gemm/v1 hit this; rule 6b in its README is the
//   post-mortem).
//
// IMPLEMENTATION GOTCHAS:
//   - .b8 destinations require an inline `.reg .b8` (PTX has no 8-bit machine
//     register) plus a `cvt.u32.u8` to surface the byte to a C++ register.
//   - Inline-asm constraint table:
//       .b16 / .h16  → "h"     (uint16_t)
//       .b32 / .u32  → "r"     (uint32_t)
//       .b64 / .u64  → "l"     (uint64_t)
//       .f32         → "f"     (float)
//     There's no .b8 constraint — wrap with the .reg trick above.
//
// Spec: PTX ISA 9.2 §9.7.9.21. Per-format deep dive: ptx/d_cvt_pack/README.md.


namespace ptx {

// Tag types + their packed-pair container size.
struct bf16  { using packed2_t = uint32_t; };   // bf16x2  = .b32
struct f16   { using packed2_t = uint32_t; };   // f16x2   = .b32 (not exercised)
struct e4m3  { using packed2_t = uint16_t; };   // e4m3x2  = .b16
struct e5m2  { using packed2_t = uint16_t; };   // e5m2x2  = .b16
struct e2m1  { using packed2_t = uint8_t;  };   // e2m1x2  = .b8
struct ue8m0 { using packed2_t = uint16_t; };   // ue8m0x2 = .b16

// Forward declaration — instantiated via specializations.
template <typename Dst>
static __device__ __forceinline__ typename Dst::packed2_t cvt_pack_f32x2_to(float a, float b);

template <>
__device__ __forceinline__ uint32_t cvt_pack_f32x2_to<bf16>(float a, float b) {
    uint32_t d;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(a), "f"(b));
    return d;
}

// f16 specialization — packed pair via `cvt.rn.f16x2.f32`. Same packed-pair
// byte-ordering convention as the other dtypes (`a` → upper half, `b` →
// lower half). Used by the multirank cooperative L1/L2 epi when staging
// the TMEM-D drain through smem_cd as FP16 (vs the BF16-direct path used
// in single-rank kernels).
template <>
__device__ __forceinline__ uint32_t cvt_pack_f32x2_to<f16>(float a, float b) {
    uint32_t d;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(a), "f"(b));
    return d;
}

template <>
__device__ __forceinline__ uint16_t cvt_pack_f32x2_to<e4m3>(float a, float b) {
    uint16_t d;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(d) : "f"(a), "f"(b));
    return d;
}

template <>
__device__ __forceinline__ uint16_t cvt_pack_f32x2_to<e5m2>(float a, float b) {
    uint16_t d;
    asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
                 : "=h"(d) : "f"(a), "f"(b));
    return d;
}

template <>
__device__ __forceinline__ uint8_t cvt_pack_f32x2_to<e2m1>(float a, float b) {
    // .b8 destination: declare a .b8 reg in inline-asm and surface via cvt.u32.u8.
    uint32_t d;
    asm volatile(
        "{ .reg .b8 v;"
        "  cvt.rn.satfinite.e2m1x2.f32 v, %1, %2;"
        "  cvt.u32.u8 %0, v;"
        "}"
        : "=r"(d) : "f"(a), "f"(b));
    return uint8_t(d & 0xffu);
}

template <>
__device__ __forceinline__ uint16_t cvt_pack_f32x2_to<ue8m0>(float a, float b) {
    // Note: .rn is rejected by ptxas for ue8m0x2.f32 (despite the spec); use .rz.
    uint16_t d;
    asm volatile("cvt.rz.satfinite.ue8m0x2.f32 %0, %1, %2;"
                 : "=h"(d) : "f"(a), "f"(b));
    return d;
}

}  // namespace ptx
