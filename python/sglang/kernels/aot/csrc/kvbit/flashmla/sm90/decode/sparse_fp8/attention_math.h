#pragma once

#include <cmath>

namespace kvbit::dsv4 {

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
    float
    lse_with_sink_log2(float sum, float max_log2, float sink_log2) {
  // FlashMLA uses +inf as its empty, sink-free state sentinel.
  if (sink_log2 == INFINITY) return INFINITY;
  if (sum == 0.0f) return sink_log2 == -INFINITY ? INFINITY : sink_log2;
  const float kv_lse_log2 = max_log2 + log2f(sum);
  const float high = fmaxf(kv_lse_log2, sink_log2);
  return high + log2f(exp2f(kv_lse_log2 - high) + exp2f(sink_log2 - high));
}

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
    float
    no_split_lse(float sum, float max_log2, float sink_log2) {
  return lse_with_sink_log2(sum, max_log2, sink_log2) * 0.6931471805599453f;
}

}  // namespace kvbit::dsv4
