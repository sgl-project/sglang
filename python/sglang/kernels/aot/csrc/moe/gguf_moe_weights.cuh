#pragma once

#include <cstdint>

template <typename output_t, typename weight_t, typename id_t>
__global__ void apply_gguf_moe_weights_kernel(
    output_t* __restrict__ outputs,
    const weight_t* __restrict__ weights,
    const id_t* __restrict__ ids,
    const int64_t rows,
    const int64_t hidden_size,
    const int64_t topk,
    const int64_t weight_stride0,
    const int64_t weight_stride1,
    const int64_t id_stride0,
    const int64_t id_stride1) {
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t token = row / topk;
    const int64_t slot = row % topk;
    const int64_t first_column = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    const int64_t column_step = static_cast<int64_t>(gridDim.y) * blockDim.x;
    if (ids[token * id_stride0 + slot * id_stride1] < 0) {
      for (int64_t column = first_column; column < hidden_size; column += column_step) {
        outputs[row * hidden_size + column] = static_cast<output_t>(0.0f);
      }
      continue;
    }
    const float weight = static_cast<float>(weights[token * weight_stride0 + slot * weight_stride1]);
    for (int64_t column = first_column; column < hidden_size; column += column_step) {
      const int64_t index = row * hidden_size + column;
      const float value = static_cast<float>(outputs[index]);
#ifdef USE_MUSA
      const float product = value * weight;
#else
      // SM90 enables fast math; preserve subnormal operands and products here.
      float product;
      asm("mul.rn.f32 %0, %1, %2;" : "=f"(product) : "f"(value), "f"(weight));
#endif
      outputs[index] = static_cast<output_t>(product);
    }
  }
}
