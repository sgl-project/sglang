// Adapted from
// https://github.com/vllm-project/vllm/blob/eb59b5a6cba6727d3727c0372258db9002f687c1/csrc/quantization/awq/gemm_kernels.cu#L350
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint4 result;
  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const i4s = source;

  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x44804480; // bf16(1024)

  const uint32_t top_i4s = i4s >> 8; // hidden latency

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

  static constexpr __nv_bfloat16 BF16_1024 = __float2bfloat16(1024.0f); // 0x4480
  static constexpr __nv_bfloat16 BF16_1_16 = __float2bfloat16(1.0f/16.0f); // 0x3D80
  static constexpr __nv_bfloat16 BF16_NEG64 = __float2bfloat16(-64.0f); // 0xC280

  // 拆分处理packed的两个bf16
  auto process_bf16_pair = [](uint32_t& packed_val, bool is_high_nibble) {
    __nv_bfloat162* pair = reinterpret_cast<__nv_bfloat162*>(&packed_val);
    
    __nv_bfloat16 val1 = pair->x;
    if(is_high_nibble) {
      val1 = __hfma(val1, BF16_1_16, BF16_NEG64);
    } else {
      val1 = __hsub(val1, BF16_1024);
    }

    __nv_bfloat16 val2 = pair->y;
    if(is_high_nibble) {
      val2 = __hfma(val2, BF16_1_16, BF16_NEG64);
    } else {
      val2 = __hsub(val2, BF16_1024);
    }

    pair->x = val1;
    pair->y = val2;
  };

  process_bf16_pair(h[0], false); // 0-1: low 4
  process_bf16_pair(h[1], true);  // 2-3: high 4
  process_bf16_pair(h[2], false);
  process_bf16_pair(h[3], true);

  return result;
#else
  assert(false);
  return {};
#endif
}

template <typename OutT>
__global__ void __launch_bounds__(256) dequantize_weights(
    int* __restrict__ qweight,
    OutT* __restrict__ scales,
    int* __restrict__ qzeros,
    OutT* __restrict__ output,
    int group_size,
    int qweight_cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // 根据输出类型选择解包函数
  uint4 zeros, loaded_scale, weight;
  if constexpr (std::is_same_v<OutT, __nv_bfloat16>) {
    zeros = dequantize_s4_to_bf16x2(qzeros[col + (row / group_size) * qweight_cols]);
    weight = dequantize_s4_to_bf16x2(qweight[col + row * qweight_cols]);
  } else {
    zeros = dequantize_s4_to_fp16x2(qzeros[col + (row / group_size) * qweight_cols]);
    weight = dequantize_s4_to_fp16x2(qweight[col + row * qweight_cols]);
  }

  loaded_scale = *(uint4*)(scales + 8 * col + (row / group_size) * qweight_cols * 8);

  if constexpr (std::is_same_v<OutT, __nv_bfloat16>) {
    // 
    float2* zeros_fp32 = reinterpret_cast<float2*>(&zeros);
    float2* scales_fp32 = reinterpret_cast<float2*>(&loaded_scale);
    float2* weight_fp32 = reinterpret_cast<float2*>(&weight);
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      weight_fp32[i].x = __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&weight.x)[2*i]);
      weight_fp32[i].x = (weight_fp32[i].x - __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&zeros.x)[2*i])) 
                          * __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&loaded_scale.x)[2*i]);
      
      weight_fp32[i].y = __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&weight.x)[2*i+1]);
      weight_fp32[i].y = (weight_fp32[i].y - __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&zeros.x)[2*i+1])) 
                          * __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&loaded_scale.x)[2*i+1]);
      
      reinterpret_cast<__nv_bfloat16*>(&weight.x)[2*i] = __float2bfloat16(weight_fp32[i].x);
      reinterpret_cast<__nv_bfloat16*>(&weight.x)[2*i+1] = __float2bfloat16(weight_fp32[i].y);
    }
  } else {
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight.x) : "r"(weight.x), "r"(zeros.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight.x) : "r"(weight.x), "r"(loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.y) : "r"(weight_fp16.y), "r"(zeros.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.y) : "r"(weight_fp16.y), "r"(loaded_scale.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.z) : "r"(weight_fp16.z), "r"(zeros.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.z) : "r"(weight_fp16.z), "r"(loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.w) : "r"(weight_fp16.w), "r"(zeros.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.w) : "r"(weight_fp16.w), "r"(loaded_scale.w));
  }

  // 存储结果（自动适配输出类型）
  OutT* output_ptr = output + 8 * col + 8 * row * qweight_cols;
  *(uint4*)output_ptr = weight;
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

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (scales.scalar_type() == at::ScalarType::Half) {
    auto _scales = reinterpret_cast<half*>(scales.data_ptr<at::Half>());
    auto _output = reinterpret_cast<half*>(output.data_ptr<at::Half>());
    dequantize_weights<half><<<num_blocks, threads_per_block, 0, stream>>>(
        _qweight, _scales, _zeros, _output, group_size, qweight_cols);
  } else if (scales.scalar_type() == at::ScalarType::BFloat16) {
    auto _scales = reinterpret_cast<__nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
    auto _output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    dequantize_weights<__nv_bfloat16><<<num_blocks, threads_per_block, 0, stream>>>(
        _qweight, _scales, _zeros, _output, group_size, qweight_cols);
  } else {
    AT_ERROR("awq_dequantize: Unsupported scale type");
  }

  return output;
}
