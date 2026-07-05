#include <metal_stdlib>
using namespace metal;

constant uint  H           [[function_constant(0)]];
constant float  EPS         [[function_constant(1)]];

template <typename T>
inline void rms_norm_impl(
    const device T* x,
    const device T* w,
    device T* out,
    uint3 gid,
    uint3 tid,
    uint3 tptg,
    threadgroup float* buf
) {
    uint row = gid.y;
    uint lane = tid.x;
    uint tg = tptg.x;

    uint simd_lane_id = lane % 32;      // my position within my 32-lane group
    uint simd_group_id = lane / 32;      // which group I'm in

    const device T* xrow = x + row * H;
    device T* orow = out + row * H;

    // Pass 1: sum of squares in fp32. Vectorized when H is a multiple of 4
    // (the common case: 1024/4096/...). H is a function constant, so this
    // branch is resolved at pipeline-compile time, not at runtime.
    float local = 0.0;
    if (H % 4 == 0) {
        const device vec<T, 4>* xv = (const device vec<T, 4>*)xrow;
        for (uint i = lane; i < H / 4; i += tg) {
            float4 v = float4(xv[i]);
            local += dot(v, v);
        }
    } else {
        for (uint i = lane; i < H; i += tg) {
            float v = float(xrow[i]);
            local += v * v;
        }
    }

    // Two-tier reduction: simd_sum within each SIMD group, then across groups.
    local = simd_sum(local);
    if (simd_lane_id == 0) {
        buf[simd_group_id] = local;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0;
    for (uint g = 0; g < (tg + 31) / 32; g++) {
        total += buf[g];
    }
    float inv = metal::rsqrt(total / H + EPS);

    // Pass 2: normalize, scale by weight, cast back to T on store.
    if (H % 4 == 0) {
        const device vec<T, 4>* xv = (const device vec<T, 4>*)xrow;
        const device vec<T, 4>* wv = (const device vec<T, 4>*)w;
        device vec<T, 4>* ov = (device vec<T, 4>*)orow;
        for (uint i = lane; i < H / 4; i += tg) {
            float4 xf = float4(xv[i]);
            float4 wf = float4(wv[i]);
            ov[i] = vec<T, 4>(xf * inv * wf);
        }
    } else {
        for (uint i = lane; i < H; i += tg) {
            orow[i] = T(float(xrow[i]) * inv * float(w[i]));
        }
    }
}

#define INSTANTIATE(NAME, T)                                                      \
    [[host_name("rms_norm_" #NAME)]] [[kernel]] void rms_norm_impl##NAME(         \
        const device T*       x          [[buffer(0)]],                           \
        const device T*       w          [[buffer(1)]],                           \
        device T*       out        [[buffer(2)]],                                 \
        uint3 gid [[thread_position_in_grid]],                                    \
        uint3 tid [[thread_position_in_threadgroup]],                             \
        uint3 tptg [[threads_per_threadgroup]]) {                                 \
        threadgroup float buf[8];                                                 \
        rms_norm_impl<T>(x, w, out, gid, tid, tptg, buf);                              \
    }

INSTANTIATE(f16,  half)
INSTANTIATE(bf16, bfloat)
INSTANTIATE(f32,  float)