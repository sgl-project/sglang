#include <metal_stdlib>
using namespace metal;

constant uint H [[function_constant(0)]];
constant uint KH [[function_constant(1)]];
constant uint SPLITS [[function_constant(2)]];
constant float SCALE [[function_constant(3)]];
constant bool HAS_TAIL [[function_constant(4)]];
constant uint ROWS [[function_constant(5)]];
constant uint WIDTH [[function_constant(6)]];
constant uint SLOTS [[function_constant(7)]];

template <typename T, typename O, uint D, uint W, uint G, bool PARTIAL>
inline void radix_impl(
    const device T* q, const device T* k, const device T* v,
    const device T* kp, const device T* vp, const device int* table,
    const device long* requests, const device long* lengths,
    const device T* tail_k, const device T* tail_v, device O* out,
    uint3 group, uint warp, uint lane,
    threadgroup float* maxima, threadgroup float* sums, threadgroup float* partial) {
  constexpr uint OD = PARTIAL ? D + 2 : D;
  const uint b = group.z, h = group.y * G, kh = h / (H / KH);
  const long length = lengths[b], request = requests[b];
  const ulong row = ulong(b) * H + h;
  if (request < 0 || request >= ROWS || length < 1 + uint(HAS_TAIL) || length > WIDTH) {
    if (warp == 0)
      for (uint g = 0; g < G; ++g)
        for (uint d = lane; d < OD; d += 32)
          out[((row + g) * SPLITS + group.x) * OD + d] = O(NAN);
    return;
  }
  float query[G][D / 32], acc[G][D / 32], maximum[G], sum[G];
  for (uint g = 0; g < G; ++g) {
    maximum[g] = -INFINITY;
    sum[g] = 0;
    for (uint i = 0; i < D / 32; ++i) {
      query[g][i] = float(q[(row + g) * D + lane + i * 32]) * SCALE;
      acc[g][i] = 0;
    }
  }
  const long span = (length + SPLITS - 1) / SPLITS;
  const long end = min(length, (group.x + 1) * span);
  for (long token = group.x * span + warp; token < end; token += W) {
    const bool current = token == length - 1;
    const bool previous = HAS_TAIL && token == length - 2;
    const long slot = (current || previous) ? long(b) : long(table[request * WIDTH + token]);
    if (slot < 0 || (!current && !previous && slot >= SLOTS)) {
      for (uint g = 0; g < G; ++g) { maximum[g] = NAN; sum[g] = NAN; }
      break;
    }
    const ulong base = (ulong(slot) * KH + kh) * D;
    float dots[G] = {};
    for (uint i = 0; i < D / 32; ++i) {
      const ulong offset = base + lane + i * 32;
      const float key = float(previous ? tail_k[offset] : (current ? k[offset] : kp[offset]));
      for (uint g = 0; g < G; ++g) dots[g] += query[g][i] * key;
    }
    float rescale[G], weights[G];
    for (uint g = 0; g < G; ++g) {
      const float dot = simd_sum(dots[g]);
      const float next_max = max(maximum[g], dot);
      rescale[g] = metal::fast::exp(maximum[g] - next_max);
      weights[g] = metal::fast::exp(dot - next_max);
      sum[g] = sum[g] * rescale[g] + weights[g];
      maximum[g] = next_max;
    }
    for (uint i = 0; i < D / 32; ++i) {
      const ulong offset = base + lane + i * 32;
      const float value = float(previous ? tail_v[offset] : (current ? v[offset] : vp[offset]));
      for (uint g = 0; g < G; ++g)
        acc[g][i] = acc[g][i] * rescale[g] + weights[g] * value;
    }
  }
  for (uint g = 0; g < G; ++g) {
    if (lane == 0) { maxima[g * W + warp] = maximum[g]; sums[g * W + warp] = sum[g]; }
    for (uint i = 0; i < D / 32; ++i)
      partial[(g * W + warp) * D + lane + i * 32] = acc[g][i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (warp == 0) {
    for (uint g = 0; g < G; ++g) {
      float row_max = -INFINITY, denominator = 0, factors[W];
      for (uint w = 0; w < W; ++w) row_max = max(row_max, maxima[g * W + w]);
      for (uint w = 0; w < W; ++w) {
        factors[w] = sums[g * W + w] == 0 ? 0 : metal::fast::exp(maxima[g * W + w] - row_max);
        denominator += sums[g * W + w] * factors[w];
      }
      const ulong ob = ((row + g) * SPLITS + group.x) * OD;
      for (uint i = 0; i < D / 32; ++i) {
        float numerator = 0;
        for (uint w = 0; w < W; ++w)
          numerator += partial[(g * W + w) * D + lane + i * 32] * factors[w];
        out[ob + lane + i * 32] = O(PARTIAL ? numerator : numerator / denominator);
      }
      if (PARTIAL && lane == 0) { out[ob + D] = O(denominator); out[ob + D + 1] = O(row_max); }
    }
  }
}

template <typename T, uint D>
inline void reduce_impl(const device float* partials, device T* out, uint3 group, uint lane) {
  const ulong row = ulong(group.z) * H + group.y;
  const ulong base = row * SPLITS * (D + 2);
  float maximum = -INFINITY;
  for (uint s = 0; s < SPLITS; ++s) maximum = max(maximum, partials[base + s * (D + 2) + D + 1]);
  float denominator = 0;
  float numerator[D / 32] = {};
  for (uint s = 0; s < SPLITS; ++s) {
    const ulong offset = base + s * (D + 2);
    const float sum = partials[offset + D];
    const float factor = sum == 0 ? 0 : metal::fast::exp(partials[offset + D + 1] - maximum);
    denominator += sum * factor;
    for (uint i = 0; i < D / 32; ++i) numerator[i] += partials[offset + lane + i * 32] * factor;
  }
  for (uint i = 0; i < D / 32; ++i) out[row * D + lane + i * 32] = T(numerator[i] / denominator);
}

#define RADIX(NAME, T, O, D, W, G, P) \
[[host_name("radix_" #NAME "_d" #D "_w" #W "_g" #G "_p" #P)]] \
kernel void radix_##NAME##_d##D##_w##W##_g##G##_p##P( \
    const device T* q [[buffer(0)]], const device T* k [[buffer(1)]], const device T* v [[buffer(2)]], \
    const device T* kp [[buffer(3)]], const device T* vp [[buffer(4)]], const device int* table [[buffer(5)]], \
    const device long* requests [[buffer(6)]], const device long* lengths [[buffer(7)]], \
    const device T* tail_k [[buffer(8)]], const device T* tail_v [[buffer(9)]], device O* out [[buffer(10)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  threadgroup float maxima[G * W], sums[G * W], partial[G * W * D]; \
  radix_impl<T, O, D, W, G, P>(q, k, v, kp, vp, table, requests, lengths, tail_k, tail_v, out, \
      group, warp, lane, maxima, sums, partial); \
}
#define PLAN(NAME, T, D, W, G) RADIX(NAME, T, T, D, W, G, 0) RADIX(NAME, T, float, D, W, G, 1)
#define WARP(NAME, T, D, W) PLAN(NAME, T, D, W, 1) PLAN(NAME, T, D, W, 2)
#define DIM(NAME, T, D) WARP(NAME, T, D, 4) \
[[host_name("radix_reduce_" #NAME "_d" #D)]] kernel void radix_reduce_##NAME##_d##D( \
    const device float* partials [[buffer(0)]], device T* out [[buffer(1)]], \
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
  reduce_impl<T, D>(partials, out, group, lane); \
}
#define TYPE(NAME, T) DIM(NAME, T, 64) DIM(NAME, T, 128) DIM(NAME, T, 256)
TYPE(f16, half)
TYPE(bf16, bfloat)
TYPE(f32, float)
