#pragma once

#include <cute/numeric/math.hpp>
#include <cuda/std/cstdint>
#include <cuda_fp8.h>
#include <cutlass/numeric_types.h>
#include <deep_gemm/common/compile.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::math {

/// Pointer operations
template <typename dtype_t = void>
CUTLASS_HOST_DEVICE dtype_t* advance_ptr(void* ptr, const uint64_t num_bytes) {
    return reinterpret_cast<dtype_t*>(static_cast<uint8_t*>(ptr) + num_bytes);
}

/// Math functions
template <typename T>
CUTLASS_HOST_DEVICE T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T, bool kDoCeilAlignment = true>
CUTLASS_HOST_DEVICE T align(T a, T b) {
    return (kDoCeilAlignment ? ceil_div(a, b) : (a / b)) * b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
CUTLASS_DEVICE void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

#ifdef DG_IN_CUDA_COMPILATION
CUTLASS_DEVICE float2 fma2(const float2& a, const float2& b, const float2& c) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    return __ffma2_rn(a, b, c);
#else
    return make_float2(
        __fmaf_rn(a.x, b.x, c.x),
        __fmaf_rn(a.y, b.y, c.y)
    );
#endif
}

CUTLASS_HOST_DEVICE float fast_rcp(const float& x) {
    float ret;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}

/// Casting
template <typename old_t>
CUTLASS_DEVICE nv_bfloat162 cast_into_bf16x2(old_t& x, old_t& y) {
    return __float22bfloat162_rn({*reinterpret_cast<float*>(&x), *reinterpret_cast<float*>(&y)});
}

template <typename old_t>
CUTLASS_DEVICE int cast_into_bf16_and_pack(old_t& x, old_t& y) {
    auto bf16x2 = cast_into_bf16x2(x, y);
    return *reinterpret_cast<int*>(&bf16x2);
}

/// Packed BF16 quantization
// Tree-reduce the amax of packed BF16 pairs (HMNMX2)
template <uint32_t kNumPacked>
CUTLASS_DEVICE nv_bfloat16 get_packed_bf16_amax(const nv_bfloat162 (&values)[kNumPacked]) {
    DG_STATIC_ASSERT(kNumPacked >= 2 and (kNumPacked & (kNumPacked - 1)) == 0,
                     "The tree requires a power-of-two size");
    nv_bfloat162 tree[kNumPacked / 2];
    #pragma unroll
    for (uint32_t i = 0; i < kNumPacked / 2; ++ i)
        tree[i] = __hmax2(__habs2(values[i]), __habs2(values[i + kNumPacked / 2]));
    #pragma unroll
    for (uint32_t stride = kNumPacked / 4; stride > 0; stride /= 2) {
        #pragma unroll
        for (uint32_t i = 0; i < stride; ++ i)
            tree[i] = __hmax2(tree[i], tree[i + stride]);
    }
    return __hmax(tree[0].x, tree[0].y);
}

// Scale two BF16 pairs by power-of-two BF16 sf_invs (exact) and pack into four FP8 bytes
CUTLASS_DEVICE uint32_t scale_bf16x2_into_fp8x4(const nv_bfloat162& lower, const nv_bfloat162& upper,
                                                const nv_bfloat162& sf_inv_lower, const nv_bfloat162& sf_inv_upper) {
    return __nv_fp8x4_e4m3(__hmul2(lower, sf_inv_lower), __hmul2(upper, sf_inv_upper)).__x;
}

// Select a power-of-two UE8M0 SF mapping `amax` into the finite range of `quant_dtype_t`:
// the carry of the integer addition performs the exponent ceiling (carry iff the amax
// mantissa exceeds the max finite value's mantissa)
template <typename quant_dtype_t = cutlass::float_e4m3_t, typename dtype_t>
CUTLASS_DEVICE uint32_t get_ue8m0_sf_exp(const dtype_t& amax) {
    constexpr bool kIsFP32 = cute::is_same_v<dtype_t, float>;
    DG_STATIC_ASSERT((kIsFP32 or cute::is_same_v<dtype_t, nv_bfloat16>), "The input type must be FP32 or BF16");
    DG_STATIC_ASSERT((cute::is_same_v<quant_dtype_t, cutlass::float_e4m3_t> or cute::is_same_v<quant_dtype_t, cutlass::float_e2m1_t>),
                     "The quantized type must be E4M3 or E2M1");
    constexpr bool kIsFP8 = cute::is_same_v<quant_dtype_t, cutlass::float_e4m3_t>;
    constexpr uint32_t kMantissaBits = kIsFP32 ? 23 : 7;
    constexpr uint32_t kMantissaMask = (1u << kMantissaBits) - 1;
    constexpr uint32_t kQuantMaxMantissa = (kIsFP8 ? 0x60u : 0x40u) << (kMantissaBits - 7);   // mantissa(1.75) or mantissa(1.5)
    constexpr uint32_t kQuantMaxExponent = kIsFP8 ? 8 : 2;                                    // 448 = 1.75 * 2 ^ 8, 6 = 1.5 * 2 ^ 2
    // The exponent floors match the per-token cast kernel's amax clamps:
    // `max(amax, 1e-4)` for E4M3 (2 ^ -22), `max(amax, 6 * 2 ^ -126)` for E2M1 (2 ^ -126)
    constexpr uint32_t kMinSFExponent = kIsFP8 ? 105 : 1;
    uint32_t amax_bits;
    if constexpr (kIsFP32)
        amax_bits = __float_as_uint(amax);
    else
        amax_bits = __bfloat16_as_ushort(amax);
    const auto rounded_exp = (amax_bits + kMantissaMask - kQuantMaxMantissa) >> kMantissaBits;
    return cute::max(rounded_exp, kMinSFExponent + kQuantMaxExponent) - kQuantMaxExponent;
}

// The `dtype_t` sf_inv of a UE8M0 SF exponent (reciprocal biased exponents sum to 254):
// a power of two, exact in both output formats
template <typename dtype_t>
CUTLASS_DEVICE auto get_ue8m0_sf_inv(const uint32_t& sf_exp) {
    if constexpr (cute::is_same_v<dtype_t, float>) {
        return __uint_as_float((254u - sf_exp) << 23);
    } else {
        DG_STATIC_ASSERT((cute::is_same_v<dtype_t, nv_bfloat16>), "The output type must be FP32 or BF16");
        return __ushort_as_bfloat16((254u - sf_exp) << 7);
    }
}

CUTLASS_DEVICE float fast_pow2(const int& x) {
    uint32_t bits_x = (x + 127) << 23;
    return *reinterpret_cast<float*>(&bits_x);
}

CUTLASS_DEVICE int fast_log2_ceil(float x) {
    const auto bits = *reinterpret_cast<uint32_t*>(&x);
    const auto exp = bits >> 23;
    const auto man = bits & ((1 << 23) - 1);
    return exp - 127 + (man != 0);
}

template <bool kUseUE8M0 = true>
CUTLASS_DEVICE void get_e4m3_sf_and_sf_inv(const float2& amax, float2& sf, float2& sf_inv) {
    DG_STATIC_ASSERT(kUseUE8M0, "Must use UE8M0");
    const float2 finfo_factor = {1.0 / 448.0, 1.0 / 448.0};
    const auto scaled = __fmul2_rn(amax, finfo_factor);
    const auto exp_x = fast_log2_ceil(scaled.x);
    const auto exp_y = fast_log2_ceil(scaled.y);
    sf.x = fast_pow2(exp_x), sf_inv.x = fast_pow2(-exp_x);
    sf.y = fast_pow2(exp_y), sf_inv.y = fast_pow2(-exp_y);
}

// E2M1 (FP4) variant: divisor is finfo_max=6 instead of 448. Same UE8M0
// SF protocol; only the per-element clipping range and dtype differ.
// 1/6 = 0x3E2AAAAB exactly in FP32 RN.
template <bool kUseUE8M0 = true>
CUTLASS_DEVICE void get_e2m1_sf_and_sf_inv(const float2& amax, float2& sf, float2& sf_inv) {
    DG_STATIC_ASSERT(kUseUE8M0, "Must use UE8M0");
    const float2 finfo_factor = {1.0f / 6.0f, 1.0f / 6.0f};
    const auto scaled = __fmul2_rn(amax, finfo_factor);
    const auto exp_x = fast_log2_ceil(scaled.x);
    const auto exp_y = fast_log2_ceil(scaled.y);
    sf.x = fast_pow2(exp_x), sf_inv.x = fast_pow2(-exp_x);
    sf.y = fast_pow2(exp_y), sf_inv.y = fast_pow2(-exp_y);
}

CUTLASS_DEVICE void get_nvfp4_sf_and_sf_inv(const float2& amax, float2& sf, float2& sf_inv, uint2& sf_bits) {
    constexpr float kInvMax = 1.0f / 6.0f;
    const __nv_fp8_e4m3 qx(amax.x * kInvMax), qy(amax.y * kInvMax);
    sf_bits = {qx.__x, qy.__x};
    sf = {static_cast<float>(qx), static_cast<float>(qy)};
    sf_inv.x = sf.x > 0.0f ? __frcp_rn(sf.x) : 0.0f;
    sf_inv.y = sf.y > 0.0f ? __frcp_rn(sf.y) : 0.0f;
}

// Built with `__frcp_rn`, so a lookup is bit-identical to computing the reciprocal.
constexpr uint32_t kNumNVFP4SFInvLutEntries = 256;

CUTLASS_DEVICE void init_nvfp4_sf_inv_lut(float* lut, const uint32_t& lane_idx) {
    #pragma unroll
    for (uint32_t b = lane_idx; b < kNumNVFP4SFInvLutEntries; b += 32) {
        __nv_fp8_e4m3 q;
        q.__x = static_cast<__nv_fp8_storage_t>(b);
        const float sf = static_cast<float>(q);
        lut[b] = sf > 0.0f ? __frcp_rn(sf) : 0.0f;
    }
}

CUTLASS_DEVICE void get_nvfp4_sf_inv_from_lut(const float2& amax, const float* lut,
                                              float2& sf_inv, uint2& sf_bits) {
    constexpr float kInvMax = 1.0f / 6.0f;
    const __nv_fp8_e4m3 qx(amax.x * kInvMax), qy(amax.y * kInvMax);
    sf_bits = {qx.__x, qy.__x};
    sf_inv = {lut[qx.__x], lut[qy.__x]};
}

CUTLASS_DEVICE uint32_t cast_into_e2m1x2_pairs(const float2& lower, const float2& upper) {
    uint32_t packed;
    asm volatile(
        "{\n\t"
        ".reg .b8 byte0;\n\t"
        ".reg .b8 byte1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %3, %1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %2;\n\t"
        "mov.b32 %0, {byte0, byte1, byte0, byte1};\n\t"
        "}\n"
        : "=r"(packed)
        : "f"(lower.x), "f"(lower.y), "f"(upper.x), "f"(upper.y));
    return packed;
}

// Pack two FP32 values into one FP4 (E2M1) byte: lower nibble = a, upper = b.
// Matches PTX `cvt.rn.satfinite.e2m1x2.f32 d, b, a` (b → upper, a → lower).
CUTLASS_DEVICE uint32_t cvt_pack_f32_to_e2m1x2(const float& a, const float& b) {
    uint32_t out;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "cvt.u32.u8 %0, byte0;\n"
        "}"
        : "=r"(out) : "f"(a), "f"(b));
    return out;
}

// Pack four FP32 values into one uint16 (FP4 nibbles, 4 elements / 2 bytes).
// Layout: bits[0:4]=a, [4:8]=b, [8:12]=c, [12:16]=d. Compatible with
// `cvt.rn.satfinite.e2m1x2.f32` whose output is "low nibble = first arg".
CUTLASS_DEVICE uint32_t cvt_pack_f32x4_to_e2m1x4(
        const float& a, const float& b, const float& c, const float& d) {
    uint32_t out;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
        ".reg .b16 hword;\n"
        "mov.b16 hword, {byte0, byte1};\n"
        "cvt.u32.u16 %0, hword;\n"
        "}"
        : "=r"(out) : "f"(a), "f"(b), "f"(c), "f"(d));
    return out;
}

/// Reduction
template <typename T>
CUTLASS_DEVICE T warp_inclusive_sum(T value, const uint32_t lane_idx) {
    #pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const T synced = __shfl_up_sync(0xffffffff, value, offset);
        if (lane_idx >= offset)
            value += synced;
    }
    return value;
}

// Block-wide prefix sum. Synchronize the block before reusing `warp_sums`.
template <uint32_t kNumThreads, typename T>
CUTLASS_DEVICE T cta_exclusive_sum(const T value, T* warp_sums, T& total) {
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    DG_STATIC_ASSERT(kNumThreads >= 32 and kNumThreads % 32 == 0 and kNumWarps <= 32, "Invalid CTA scan shape");

    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const T lane_sum = warp_inclusive_sum(value, lane_idx);
    if (lane_idx == 31)
        warp_sums[warp_idx] = lane_sum;
    __syncthreads();

    const T warp_total = lane_idx < kNumWarps ? warp_sums[lane_idx] : 0;
    const T warp_sum = warp_inclusive_sum(warp_total, lane_idx);
    total = ptx::exchange(warp_sum, kNumWarps - 1);
    return lane_sum - value + ptx::exchange(warp_sum - warp_total, warp_idx);
}

template <uint32_t kNumThreads, typename T>
CUTLASS_DEVICE T cta_exclusive_sum(const T value, T* warp_sums) {
    T total;
    return cta_exclusive_sum<kNumThreads>(value, warp_sums, total);
}

// Operation functors
// NOTES: `fmaxf`/`fminf` lower to single `FMNMX` instructions; the generic float ternaries
//        cannot (mismatched NaN semantics) and lower to `FSETP + FSEL` pairs instead,
//        doubling both the instruction count and the dependency chain latency
template <typename T> struct ReduceSum { CUTLASS_DEVICE T operator()(T a, T b) const { return a + b; } };
template <typename T> struct ReduceMax { CUTLASS_DEVICE T operator()(T a, T b) const { return a > b ? a : b; } };
template <typename T> struct ReduceMin { CUTLASS_DEVICE T operator()(T a, T b) const { return a < b ? a : b; } };
template <> struct ReduceMax<float> { CUTLASS_DEVICE float operator()(float a, float b) const { return fmaxf(a, b); } };
template <> struct ReduceMin<float> { CUTLASS_DEVICE float operator()(float a, float b) const { return fminf(a, b); } };
template <> struct ReduceMax<nv_bfloat16> { CUTLASS_DEVICE nv_bfloat16 operator()(nv_bfloat16 a, nv_bfloat16 b) const { return __hmax(a, b); } };
template <> struct ReduceMax<nv_bfloat162> { CUTLASS_DEVICE nv_bfloat162 operator()(nv_bfloat162 a, nv_bfloat162 b) const { return __hmax2(a, b); } };
template <typename T> struct ReduceAnd { CUTLASS_DEVICE T operator()(T a, T b) const { return a & b; } };
template <typename T> struct ReduceOr  { CUTLASS_DEVICE T operator()(T a, T b) const { return a | b; } };

// Unified reduction function
template <uint32_t kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
CUTLASS_DEVICE T warp_reduce(T value, Op op) {
    DG_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or
                     kNumLanesPerGroup ==  4 or kNumLanesPerGroup == 2  or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
    constexpr uint32_t mask = 0xffffffff;
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <=  1) value = op(value, __shfl_xor_sync(mask, value,  1));
        if constexpr (kNumLanesPerGroup <=  2) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup <=  4) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup <=  8) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup <= 16) value = op(value, __shfl_xor_sync(mask, value, 16));
    } else {
        if constexpr (kNumLanesPerGroup >= 32) value = op(value, __shfl_xor_sync(mask, value, 16));
        if constexpr (kNumLanesPerGroup >= 16) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup >=  8) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup >=  4) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup >=  2) value = op(value, __shfl_xor_sync(mask, value,  1));
    }
    return value;
}

// Convenience aliases
template <uint32_t kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
CUTLASS_DEVICE T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}

template <uint32_t kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
CUTLASS_DEVICE T warp_reduce_max(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMax<T>{});
}

#endif

} // namespace deep_gemm
