// Adapted from
// https://github.com/vllm-project/vllm/blob/eb59b5a6cba6727d3727c0372258db9002f687c1/csrc/quantization/awq/gemm_kernels.cu#L350
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/all.h>

template<bool norm = true>
inline __device__ uint4 dequantize_s4_to_bf16x2(uint32_t const& source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint4 result;

  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const& i4s = reinterpret_cast<uint32_t const&>(source);

  // 01234567 01234567
  // SEEEEEEE EMMMMMMM
  // 127 + 7 = 134 -> 0100 0011 0 -> 0x43
  static constexpr uint32_t immLut   = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

  const uint32_t  top_i4s = i4s >> 8;

  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));

  // For bottom 4 bits, need to subtract 2 ^ 7 = 128
  static constexpr uint32_t BF16_TOP_MAGIC_NUM = 0x43004300;
  // For top 4 bits, need to move 4 bits to the right, 2 ^ 4 = 16
  // This is the bf16x2 {1 / 16, 1 / 16} represented as an integer.
  // 0011 1101 1000 0000
  static constexpr uint32_t ONE_SIXTEENTH = 0x3d803d80;
  // 128 / 16 = 8 we need to subtract 8 from the above.
  // This is the bf16x2 {-8, -8} represented as an integer.
  static constexpr uint32_t NEG_8 = 0xc100c100;

  // Convert elt_01
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(BF16_TOP_MAGIC_NUM));
  // Convert elt_23
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_8));
  // Convert elt_45
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(BF16_TOP_MAGIC_NUM));
  // Convert elt_67
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_8));

  return result;
#else
  assert(false);
  return {};
#endif
}

__device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  uint4 result;

  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

  // 01234567 01234567
  // SEEEEEMM MMMMMMMM
  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

  // Note that the entire sequence only requires 1 shift instruction. This is
  // thanks to the register packing format and the fact that we force our
  // integers to be unsigned, and account for this in the fp16 subtractions. In
  // addition, I exploit the fact that sub and fma have the same throughput in
  // order to convert elt_23 and elt_67 to fp16 without having to shift them to
  // the bottom bits before hand.

  // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
  // dependency if we issue immediately before required.
  const uint32_t top_i4s = i4s >> 8;
  // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

  // This is the half2 {1024, 1024} represented as an integer.
  // 01234567 01234567
  // SEEEEEMM MMMMMMMM
  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
  // This is the half2 {1 / 16, 1 / 16} represented as an integer.
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
  // 1024 / 16 = 64 we need to subtract 64 from the above.
  // This is the half2 {-64, -64} represented as an integer.
  static constexpr uint32_t NEG_64 = 0xd400d400;

  // Finally, we construct the output numbers.
  // Convert elt_01
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_23
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
  // Convert elt_45
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_67
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

  return result;
#else
  assert(false);
  return {};
#endif
}

__global__ void __launch_bounds__(256) dequantize_weights(
    int* __restrict__ qweight,
    nv_bfloat16* __restrict__ scales,
    int* __restrict__ qzeros,
    nv_bfloat16* __restrict__ output,
    int group_size,
    int qweight_cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  uint4 zeros = dequantize_s4_to_bf16x2(qzeros[col + (row / group_size) * qweight_cols]);
  // uint4 zeros = dequantize_s4_to_fp16x2(qzeros[col + (row / group_size) * qweight_cols]);
  uint4 loaded_scale = *(uint4*)(scales + 8 * col + (row / group_size) * qweight_cols * 8);

  uint4 weight_fp16 = dequantize_s4_to_bf16x2(qweight[col + row * qweight_cols]);
  // uint4 weight_fp16 = dequantize_s4_to_fp16x2(qweight[col + row * qweight_cols]);

  // asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.x) : "r"(weight_fp16.x), "r"(zeros.x));
  // asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.x) : "r"(weight_fp16.x), "r"(loaded_scale.x));
  // asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.y) : "r"(weight_fp16.y), "r"(zeros.y));
  // asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.y) : "r"(weight_fp16.y), "r"(loaded_scale.y));
  // asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.z) : "r"(weight_fp16.z), "r"(zeros.z));
  // asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.z) : "r"(weight_fp16.z), "r"(loaded_scale.z));
  // asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.w) : "r"(weight_fp16.w), "r"(zeros.w));
  // asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.w) : "r"(weight_fp16.w), "r"(loaded_scale.w));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(weight_fp16.x) : "r"(weight_fp16.x), "r"(zeros.x));
  asm volatile("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(weight_fp16.x) : "r"(weight_fp16.x), "r"(loaded_scale.x));
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(weight_fp16.y) : "r"(weight_fp16.y), "r"(zeros.y));
  asm volatile("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(weight_fp16.y) : "r"(weight_fp16.y), "r"(loaded_scale.y));
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(weight_fp16.z) : "r"(weight_fp16.z), "r"(zeros.z));
  asm volatile("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(weight_fp16.z) : "r"(weight_fp16.z), "r"(loaded_scale.z));
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(weight_fp16.w) : "r"(weight_fp16.w), "r"(zeros.w));
  asm volatile("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(weight_fp16.w) : "r"(weight_fp16.w), "r"(loaded_scale.w));
#endif
  nv_bfloat16* output_ptr = output + 8 * col + 8 * row * qweight_cols;
  *(uint4*)output_ptr = weight_fp16;
}

torch::Tensor awq_dequantize(torch::Tensor qweight, torch::Tensor scales, torch::Tensor qzeros) {
  int qweight_rows = qweight.size(0);
  int qweight_cols = qweight.size(1);
  int group_size = qweight_rows / scales.size(0);

  int x_num_threads = 16;
  int y_num_threads = 16;
  int x_blocks = qweight_cols / x_num_threads;
  int y_blocks = qweight_rows / y_num_threads;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));

  auto output_tensor_options = torch::TensorOptions().dtype(scales.dtype()).device(scales.device());
  at::Tensor output = torch::empty({qweight_rows, qweight_cols * 8}, output_tensor_options);

  auto _qweight = reinterpret_cast<int*>(qweight.data_ptr<int>());
  auto _scales = reinterpret_cast<__nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
  auto _zeros = reinterpret_cast<int*>(qzeros.data_ptr<int>());
  auto _output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());

  dim3 num_blocks(x_blocks, y_blocks);
  dim3 threads_per_block(x_num_threads, y_num_threads);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dequantize_weights<<<num_blocks, threads_per_block, 0, stream>>>(
      _qweight, _scales, _zeros, _output, group_size, qweight_cols);

  return output;
}
