#pragma once
#include "../elementwise/deterministic_topk.cuh"

namespace sglang::q38_mk_short512_detail {
constexpr uint32_t kThreadsPerBlock = 512;
constexpr size_t kSmemBytes = 8 * 1024 * sizeof(uint32_t);

template <int kTopK>
SGL_DEVICE void naive_topk(int32_t* output, int32_t length) {
  for (int i = threadIdx.x; i < kTopK; i += kThreadsPerBlock) output[i] = i < length ? i : -1;
}

template <int kTopK>
SGL_DEVICE void radix_select_topk(const float* input, int* output, int start, int length) {
  deterministic_topk_detail::radix_select<kTopK, kThreadsPerBlock, true>(input, output, start, length);
}
}  // namespace sglang::q38_mk_short512_detail
