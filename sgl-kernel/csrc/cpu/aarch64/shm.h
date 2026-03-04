#pragma once
#include <arm_neon.h>

#define VECTOR_LENGTH_IN_BYTES 16

__attribute__((target("+bf16"))) inline float32x4x2_t cvt_bf16_to_fp32(const bfloat16x8_t src) {
  float32x4x2_t y;
  y.val[0] = vcvtq_low_f32_bf16(src);
  y.val[1] = vcvtq_high_f32_bf16(src);
  return y;
}

__attribute__((target("+bf16"))) inline bfloat16x8_t cvt_fp32_to_bf16(const float32x4x2_t src) {
  return vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(src.val[0]), src.val[1]);
}

__attribute__((target("+bf16"))) inline void
reduce_bf16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers, int world_size) {
  const int element_size = 2;
  const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
  int main_elements = num_elements - (num_elements % vector_length);
  int remain_elements = num_elements % vector_length;

  // process aligned part
#pragma omp parallel for
  for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
       i += VECTOR_LENGTH_IN_BYTES) {
    float32x4x2_t inout_val = cvt_bf16_to_fp32(vld1q_bf16((const bfloat16_t*)(buffers[0] + i)));
    for (int j = 1; j < world_size; j++) {
      const float32x4x2_t in_val = cvt_bf16_to_fp32(vld1q_bf16((const bfloat16_t*)(buffers[j] + i)));
      inout_val.val[0] = vaddq_f32(inout_val.val[0], in_val.val[0]);
      inout_val.val[1] = vaddq_f32(inout_val.val[1], in_val.val[1]);
    }
    vst1q_bf16((bfloat16_t*)(to_buffer + i), cvt_fp32_to_bf16(inout_val));
  }

  // process remaining part
  int i = (start_elements + main_elements) * element_size;
  while (remain_elements > 0) {
    float val = 0.0f;
    for (int j = 0; j < world_size; j++) {
      val += vcvtah_f32_bf16(*(bfloat16_t*)(buffers[j] + i));
    }
    *(bfloat16_t*)(to_buffer + i) = vcvth_bf16_f32(val);
    remain_elements--;
    i += element_size;
  }
}

inline void reduce_fp16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers, int world_size) {
  const int element_size = 2;
  const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
  int main_elements = num_elements - (num_elements % vector_length);
  int remain_elements = num_elements % vector_length;

  // process aligned part
#pragma omp parallel for
  for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
       i += VECTOR_LENGTH_IN_BYTES) {
    float16x8_t inout_val = vld1q_f16((const float16_t*)(buffers[0] + i));
    for (int j = 1; j < world_size; j++) {
      const float16x8_t in_val = vld1q_f16((const float16_t*)(buffers[j] + i));
      inout_val = vaddq_f16(inout_val, in_val);
    }
    vst1q_f16((float16_t*)(to_buffer + i), inout_val);
  }

  // process remaining part
  int i = (start_elements + main_elements) * element_size;
  while (remain_elements > 0) {
    float16_t val = 0.0f;
    for (int j = 0; j < world_size; j++) {
      val = vaddh_f16(val, *(float16_t*)(buffers[j] + i));
    }
    *(float16_t*)(to_buffer + i) = val;
    remain_elements--;
    i += element_size;
  }
}

inline void reduce_fp32_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers, int world_size) {
  const int element_size = 4;
  const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
  int main_elements = num_elements - (num_elements % vector_length);
  int remain_elements = num_elements % vector_length;

  // process aligned part
#pragma omp parallel for
  for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
       i += VECTOR_LENGTH_IN_BYTES) {
    float32x4_t inout_val = vld1q_f32((const float*)(buffers[0] + i));
    for (int j = 1; j < world_size; j++) {
      const float32x4_t in_val = vld1q_f32((const float*)(buffers[j] + i));
      inout_val = vaddq_f32(inout_val, in_val);
    }
    vst1q_f32((float32_t*)(to_buffer + i), inout_val);
  }

  // process remaining part
  int i = (start_elements + main_elements) * element_size;
  while (remain_elements > 0) {
    float val = 0.0f;
    for (int j = 0; j < world_size; j++) {
      val += *(float*)(buffers[j] + i);
    }
    *(float*)(to_buffer + i) = val;
    remain_elements--;
    i += element_size;
  }
}

inline void parallel_memcpy(void* to, void* from, size_t n_bytes) {
  auto aligned_bytes = n_bytes - (n_bytes % VECTOR_LENGTH_IN_BYTES);
  // process aligned part
#pragma omp parallel for
  for (size_t i = 0; i < aligned_bytes; i += VECTOR_LENGTH_IN_BYTES) {
    const uint8x16_t val = vld1q_u8((uint8_t*)from + i);
    vst1q_u8((uint8_t*)to + i, val);
  }

  // process remaining part
  for (size_t i = aligned_bytes; i < n_bytes; i++) {
    *((uint8_t*)to + i) = *((uint8_t*)from + i);
  }
}

#undef VECTOR_LENGTH_IN_BYTES
