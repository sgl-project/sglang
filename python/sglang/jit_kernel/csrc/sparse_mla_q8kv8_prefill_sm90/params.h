#pragma once

#include "cutlass/bfloat16.h"
#include <cstdint>
#include <cuda_runtime.h>

struct SparseMlaQ8Kv8PrefillParams {
  int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
  float sm_scale_div_log2;

  const uint8_t* __restrict__ q;
  const uint8_t* __restrict__ kv;
  int* __restrict__ indices;
  float* __restrict__ attn_sink;
  int* __restrict__ topk_length;

  const float* __restrict__ q_scale_ptr;
  const float* __restrict__ kv_scale_ptr;

  int stride_q_s_q;
  int stride_q_h_q;
  int64_t stride_kv_s_kv;
  int stride_kv_h_kv;
  int stride_indices_s_q;
  int stride_indices_h_kv;

  cutlass::bfloat16_t* __restrict__ out;
  float* __restrict__ max_logits;
  float* __restrict__ lse;

  int num_sm;
  cudaStream_t stream;
};
