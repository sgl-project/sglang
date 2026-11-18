#include <metal_stdlib>
using namespace metal;

// act[p, i] = silu(w1[e, i] . x[t]) * (w1[e, I + i] . x[t]) for pair p = (t, j).
// Threadgroups run row-chunk major over the pairs sorted by expert, so pairs
// sharing an expert read the same weight rows back to back (from cache).
kernel void moe_gate_up(
    device const bfloat* x [[buffer(0)]],    // [T, H]
    device const bfloat* w1 [[buffer(1)]],   // [E, 2I, H]
    device const int* ids [[buffer(2)]],     // [T * K]
    device const int* order [[buffer(3)]],   // [T * K] pairs sorted by expert
    device float* act [[buffer(4)]],         // [T * K, I]
    constant long& H [[buffer(5)]],
    constant long& I [[buffer(6)]],
    constant long& K [[buffer(7)]],
    constant long& N [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const long p = order[tg % N], i = (tg / N) * ROWS + sg;
  if (i >= I) return;
  const long e = ids[p];
  device const bfloat4* xv = (device const bfloat4*)(x + (p / K) * H);
  device const bfloat4* g = (device const bfloat4*)(w1 + (e * 2 * I + i) * H);
  device const bfloat4* u = (device const bfloat4*)(w1 + (e * 2 * I + I + i) * H);
  float sg_acc = 0.0f, su_acc = 0.0f;
  for (long c = lane; c < H / 4; c += 32) {
    const float4 xf = float4(xv[c]);
    sg_acc += dot(float4(g[c]), xf);
    su_acc += dot(float4(u[c]), xf);
  }
  const float gate = simd_sum(sg_acc), up = simd_sum(su_acc);
  if (lane == 0) act[p * I + i] = gate / (1.0f + exp(-gate)) * up;
}

// out[t, h] = sum_j weight[t, j] * (w2[e_tj, h] . act[t * K + j]); row-chunk
// major over tokens for the same cache reuse.
kernel void moe_down(
    device const float* act [[buffer(0)]],   // [T * K, I]
    device const bfloat* w2 [[buffer(1)]],   // [E, H, I]
    device const int* ids [[buffer(2)]],     // [T * K]
    device const float* weights [[buffer(3)]],  // [T * K]
    device bfloat* out [[buffer(4)]],        // [T, H]
    constant long& H [[buffer(5)]],
    constant long& I [[buffer(6)]],
    constant long& K [[buffer(7)]],
    constant long& T [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const long t = tg % T, h = (tg / T) * ROWS + sg;
  if (h >= H) return;
  float total = 0.0f;
  for (long j = 0; j < K; ++j) {
    const long p = t * K + j;
    device const float4* av = (device const float4*)(act + p * I);
    device const bfloat4* w = (device const bfloat4*)(w2 + (long(ids[p]) * H + h) * I);
    float s = 0.0f;
    for (long c = lane; c < I / 4; c += 32) s += dot(float4(w[c]), av[c]);
    total += weights[p] * simd_sum(s);
  }
  if (lane == 0) out[t * H + h] = bfloat(total);
}
