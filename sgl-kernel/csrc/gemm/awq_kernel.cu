// Adapted from
// https://github.com/vllm-project/vllm/blob/eb59b5a6cba6727d3727c0372258db9002f687c1/csrc/quantization/awq/gemm_kernels.cu#L350
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

__device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  uint4 result;

  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

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
  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
  // This is the half2 {1 / 16, 1 / 16} represented as an integer.
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
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

__device__ uint4 dequantize_s4_to_bf16x2(uint32_t const& source) {
#if CUDA_VERSION >= 12000
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint4 result;
  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const i4s = source;

  // Define masks and constants
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;
  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC300C300;

  int lo0 = lop3<(0xf0 & 0xcc) | 0xaa>(i4s, MASK, EX);
  int hi0 = lop3<(0xf0 & 0xcc) | 0xaa>(i4s >> 4, MASK, EX);
  int lo1 = lop3<(0xf0 & 0xcc) | 0xaa>(i4s >> 8, MASK, EX);
  int hi1 = lop3<(0xf0 & 0xcc) | 0xaa>(i4s >> 12, MASK, EX);

  nv_bfloat162* res = reinterpret_cast<nv_bfloat162*>(h);
  res[0] = __hfma2(
      *reinterpret_cast<nv_bfloat162*>(&lo0),
      *reinterpret_cast<const nv_bfloat162*>(&MUL),
      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  res[1] = __hfma2(
      *reinterpret_cast<nv_bfloat162*>(&hi0),
      *reinterpret_cast<const nv_bfloat162*>(&MUL),
      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  res[2] = __hfma2(
      *reinterpret_cast<nv_bfloat162*>(&lo1),
      *reinterpret_cast<const nv_bfloat162*>(&MUL),
      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  res[3] = __hfma2(
      *reinterpret_cast<nv_bfloat162*>(&hi1),
      *reinterpret_cast<const nv_bfloat162*>(&MUL),
      *reinterpret_cast<const nv_bfloat162*>(&ADD));

  return result;
#else
  assert(false);
  return {};
#endif
#endif
}

template <typename OutputT>
__global__ void __launch_bounds__(256) dequantize_weights(
    int* __restrict__ qweight,
    OutputT* __restrict__ scales,
    int* __restrict__ qzeros,
    OutputT* __restrict__ output,
    int group_size,
    int qweight_cols) {
#if CUDA_VERSION >= 12000
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int group_idx = row / group_size;
  int scale_offset = 8 * col + group_idx * qweight_cols * 8;
  uint4 loaded_scale = *(uint4*)(scales + scale_offset);

  // Handle different data types
  if constexpr (std::is_same<OutputT, half>::value) {
    // FP16 path
    uint4 zeros = dequantize_s4_to_fp16x2(qzeros[col + group_idx * qweight_cols]);
    uint4 weight_fp16 = dequantize_s4_to_fp16x2(qweight[col + row * qweight_cols]);

    // Use PTX assembly for FP16 operations
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.x) : "r"(weight_fp16.x), "r"(zeros.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.x) : "r"(weight_fp16.x), "r"(loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.y) : "r"(weight_fp16.y), "r"(zeros.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.y) : "r"(weight_fp16.y), "r"(loaded_scale.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.z) : "r"(weight_fp16.z), "r"(zeros.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.z) : "r"(weight_fp16.z), "r"(loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.w) : "r"(weight_fp16.w), "r"(zeros.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.w) : "r"(weight_fp16.w), "r"(loaded_scale.w));

    OutputT* output_ptr = output + 8 * col + 8 * row * qweight_cols;
    *(uint4*)output_ptr = weight_fp16;
  } else if constexpr (std::is_same<OutputT, __nv_bfloat16>::value) {
    uint4 weight_raw = dequantize_s4_to_bf16x2(qweight[col + row * qweight_cols]);
    uint4 zero_raw = dequantize_s4_to_bf16x2(qzeros[col + group_idx * qweight_cols]);
    uint4 scale_raw = *reinterpret_cast<uint4*>(scales + scale_offset);

    // Vectorized processing (each uint4 contains 4 nv_bfloat162)
    nv_bfloat162* weight_vec = reinterpret_cast<nv_bfloat162*>(&weight_raw);
    nv_bfloat162* zero_vec = reinterpret_cast<nv_bfloat162*>(&zero_raw);
    nv_bfloat162* scale_vec = reinterpret_cast<nv_bfloat162*>(&scale_raw);

// Single instruction dual-channel operation
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // uint4 = 4 * nv_bfloat162
      weight_vec[i] = __hmul2(__hsub2(weight_vec[i], zero_vec[i]), scale_vec[i]);
    }

    // Directly store to OutputT array (guaranteed contiguous memory)
    OutputT* output_ptr = output + 8 * col + row * qweight_cols * 8;
    static_assert(sizeof(uint4) == 8 * sizeof(OutputT), "Memory layout mismatch");
    *reinterpret_cast<uint4*>(output_ptr) = weight_raw;
  }
#endif
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
  auto _zeros = reinterpret_cast<int*>(qzeros.data_ptr<int>());

  dim3 num_blocks(x_blocks, y_blocks);
  dim3 threads_per_block(x_num_threads, y_num_threads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (scales.scalar_type() == at::ScalarType::Half) {
    auto _scales = reinterpret_cast<half*>(scales.data_ptr<at::Half>());
    auto _output = reinterpret_cast<half*>(output.data_ptr<at::Half>());
    dequantize_weights<half>
        <<<num_blocks, threads_per_block, 0, stream>>>(_qweight, _scales, _zeros, _output, group_size, qweight_cols);
  } else {
    auto _scales = reinterpret_cast<__nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
    auto _output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    dequantize_weights<__nv_bfloat16>
        <<<num_blocks, threads_per_block, 0, stream>>>(_qweight, _scales, _zeros, _output, group_size, qweight_cols);
  }

  return output;
}
