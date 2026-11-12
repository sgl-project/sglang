#pragma once

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

#include <deep_gemm/common/exception.cuh>

#if defined(DG_IN_CUDA_COMPILATION)
namespace deep_gemm::ptx {

CUTLASS_DEVICE uint32_t get_sm_idx() {
    uint32_t sm_idx;
    asm ("mov.u32 %0, %%smid;" : "=r"(sm_idx));
    return sm_idx;
}

CUTLASS_DEVICE uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

CUTLASS_DEVICE uint64_t get_grid_idx() {
    uint64_t grid_idx;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(grid_idx));
    return grid_idx;
}

CUTLASS_DEVICE void sync_aligned(const uint32_t& num_threads, const uint32_t& barrier_idx) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_idx), "r"(num_threads));
}

CUTLASS_DEVICE void sync_unaligned(const uint32_t& num_threads, const uint32_t& barrier_idx) {
    asm volatile("barrier.sync %0, %1;" : : "r"(barrier_idx), "r"(num_threads));
}

template <typename dtype_t>
CUTLASS_DEVICE dtype_t exchange(dtype_t ptr, const uint32_t& src_lane_idx) {
    DG_STATIC_ASSERT(sizeof(dtype_t) % sizeof(uint32_t) == 0, "");
    const auto send_int_values = reinterpret_cast<uint32_t*>(&ptr);
    dtype_t recv_dtype;
    auto recv_int_values = reinterpret_cast<uint32_t*>(&recv_dtype);
    #pragma unroll
    for (uint32_t i = 0; i < sizeof(dtype_t) / sizeof(uint32_t); ++ i)
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], static_cast<int>(src_lane_idx));
    return recv_dtype;
}

CUTLASS_DEVICE float reduce_max_sync(const float& value) {
    float ret;
    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(ret) : "f"(value));
    return ret;
}

CUTLASS_DEVICE nv_bfloat162 cvt_relu_bf16x2_f32(const float2& v) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    uint32_t packed;
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(packed) : "f"(v.y), "f"(v.x));
    return *reinterpret_cast<const nv_bfloat162*>(&packed);
#else
    return __floats2bfloat162_rn(fmaxf(v.x, 0.0f), fmaxf(v.y, 0.0f));
#endif
}

CUTLASS_DEVICE void accumulate(float2& a, nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    // Use `add.rn.f32.bf16` instruction to perform fused (cast + add) operation on SM100
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.x) : "h"(*reinterpret_cast<uint16_t*>(&b.x)));
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.y) : "h"(*reinterpret_cast<uint16_t*>(&b.y)));
#else
    const auto [x, y] = __bfloat1622float2(b);
    a.x += x, a.y += y;
#endif
}

CUTLASS_DEVICE void accumulate_square(float2& a, nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    // Accumulate BF16 squares directly into FP32 without separate conversion instructions.
    asm("fma.rn.f32.bf16 %0, %1, %1, %0;\n" : "+f"(a.x) : "h"(*reinterpret_cast<uint16_t*>(&b.x)));
    asm("fma.rn.f32.bf16 %0, %1, %1, %0;\n" : "+f"(a.y) : "h"(*reinterpret_cast<uint16_t*>(&b.y)));
#else
    const float2 b_f32 = __bfloat1622float2(b);
    a.x += b_f32.x * b_f32.x;
    a.y += b_f32.y * b_f32.y;
#endif
}

} // namespace deep_gemm::ptx
#endif
