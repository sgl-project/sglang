#ifndef _data_types_cuh
#define _data_types_cuh

#include <sgl_kernel/marlin/scalar_type.hpp>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {
// Marlin params

static constexpr int default_threads = 256;
static constexpr int pipe_stages = 4;

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int max_thread_n = 256;

static constexpr int tile_size = 16;
static constexpr int max_par = 16;

// Repack params
static constexpr int repack_stages = 8;
static constexpr int repack_threads = 256;

static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

// Helpers
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using I4 = Vec<int, 4>;

constexpr int div_ceil(int a, int b) {
  return (a + b - 1) / b;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// No support for async
#else

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem),
      "l"(glob_ptr),
      "n"(BYTES));
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr),
      "n"(BYTES));
}

__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

#endif

// Dequant
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// Constructs destination register by taking bytes from 2 sources (based on
// mask)
template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(res) : "r"(a), "n"(start_byte), "n"(mask));
  return res;
}

template <typename scalar_t2, sglang::ScalarTypeId w_type_id, bool skip_flop = false>
__device__ inline void dequant(int q, scalar_t2* frag_b);

//
// Efficiently dequantize 4bit values packed in an int32 value into a full
// B-fragment of 4 fp16 values. We mostly follow the strategy in the link below,
// with some small changes:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L215-L287
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L327-L385
//
template <>
__device__ inline void dequant<half2, sglang::kU4B8.id(), true>(int q, half2* frag_b) {
  const int MASK = 0x000f000f;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  frag_b[0] = *reinterpret_cast<half2*>(&lo);
  frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, sglang::kU4B8.id(), false>(int q, half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // clang-format on
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(
      *reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<half2, sglang::kU4.id(), true>(int q, half2* frag_b) {
  dequant<half2, sglang::kU4B8.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<half2, sglang::kU4.id(), false>(int q, half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // clang-format on
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(
      *reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kU4B8.id(), true>(int q, nv_bfloat162* frag_b) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  // clang-format on

  frag_b[0] = *reinterpret_cast<nv_bfloat162*>(&lo);
  frag_b[1] = *reinterpret_cast<nv_bfloat162*>(&hi);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kU4B8.id(), false>(int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, sglang::kU4B8.id(), true>(q, frag_b);

  static constexpr uint32_t SUB = 0x43084308;

  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const nv_bfloat162*>(&SUB));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const nv_bfloat162*>(&SUB));
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kU4.id(), true>(int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, sglang::kU4B8.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kU4.id(), false>(int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, sglang::kU4.id(), true>(q, frag_b);

  static constexpr uint32_t SUB = 0x43004300;

  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const nv_bfloat162*>(&SUB));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const nv_bfloat162*>(&SUB));
}

//
// Fast Int8ToFp16/Int8ToBf16: Efficiently dequantize 8bit int values to fp16 or
// bf16 Reference:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L53-L85
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L125-L175
//
template <>
__device__ inline void dequant<half2, sglang::kU8B128.id(), true>(int q, half2* frag_b) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  frag_b[0] = *reinterpret_cast<half2*>(&lo);
  frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, sglang::kU8B128.id(), false>(int q, half2* frag_b) {
  dequant<half2, sglang::kU8B128.id(), true>(q, frag_b);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
}

template <>
__device__ inline void dequant<half2, sglang::kU8.id(), true>(int q, half2* frag_b) {
  dequant<half2, sglang::kU8B128.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<half2, sglang::kU8.id(), false>(int q, half2* frag_b) {
  dequant<half2, sglang::kU8.id(), true>(q, frag_b);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;
  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kU8B128.id(), false>(int q, nv_bfloat162* frag_b) {
  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388736.f;
  fp32_intermediates[1] -= 8388736.f;
  fp32_intermediates[2] -= 8388736.f;
  fp32_intermediates[3] -= 8388736.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0], fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2], fp32_intermediates_casted[3], 0x7632);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kU8.id(), false>(int q, nv_bfloat162* frag_b) {
  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388608.f;
  fp32_intermediates[1] -= 8388608.f;
  fp32_intermediates[2] -= 8388608.f;
  fp32_intermediates[3] -= 8388608.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0], fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2], fp32_intermediates_casted[3], 0x7632);
}

template <>
__device__ inline void dequant<half2, sglang::kFE4M3fn.id(), true>(int q, half2* frag_b) {
  // Constants for FP8 (E4M3) and FP16 formats
  constexpr int FP8_EXPONENT = 4, FP16_EXPONENT = 5;
  constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

  // Extract and shift FP8 values to FP16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
}

template <>
__device__ inline void dequant<half2, sglang::kFE4M3fn.id(), false>(int q, half2* frag_b) {
  dequant<half2, sglang::kFE4M3fn.id(), true>(q, frag_b);

  // Constants for FP8 (E4M3) and FP16 formats
  constexpr int FP8_EXPONENT = 4, FP16_EXPONENT = 5;

  // Construct and apply exponent bias
  constexpr int BIAS_OFFSET = (1 << (FP16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
  const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

  // Convert to half2 and apply bias
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kFE4M3fn.id(), true>(int q, nv_bfloat162* frag_b) {
  // Constants for FP8 (E4M3) and BF16 formats
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;

  constexpr int MASK = 0x7F007F00;

  // Extract and shift FP8 values to BF16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kFE4M3fn.id(), false>(int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, sglang::kFE4M3fn.id(), true>(q, frag_b);

  // Constants for FP8 (E4M3) and BF16 formats
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;

  // Construct and apply exponent bias
  constexpr int BIAS_OFFSET = (1 << (BF16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
  // Add 127 (float exponent bias) to BIAS_OFFSET and shift to float exponent
  // position
  constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
  const nv_bfloat162 bias_reg = __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));

  // Convert to bfloat162 and apply bias
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<half2, sglang::kFE2M1f.id(), true>(int q, half2* frag_b) {
  // Constants for FP4 (E2M1) and FP16 formats
  constexpr int FP4_EXPONENT = 2, FP16_EXPONENT = 5;
  constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70007000;

  // Extract and shift FP4 values to FP16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
}

template <>
__device__ inline void dequant<half2, sglang::kFE2M1f.id(), false>(int q, half2* frag_b) {
  dequant<half2, sglang::kFE2M1f.id(), true>(q, frag_b);

  // Constants for FP4 (E2M1) and FP16 formats
  constexpr int FP4_EXPONENT = 2, FP16_EXPONENT = 5;

  // Construct and apply exponent bias
  constexpr int BIAS_OFFSET = (1 << (FP16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
  const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

  // Convert to half2 and apply bias
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kFE2M1f.id(), true>(int q, nv_bfloat162* frag_b) {
  // Constants for FP4 (E2M1) and FP16 formats
  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70007000;

  // Extract and shift FP4 values to FP16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant<nv_bfloat162, sglang::kFE2M1f.id(), false>(int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, sglang::kFE2M1f.id(), true>(q, frag_b);

  // Constants for FP4 (E2M1) and BF16 formats
  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;

  // Construct and apply exponent bias
  constexpr int BIAS_OFFSET = (1 << (BF16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
  // Add 127 (float exponent bias) to BIAS_OFFSET and shift to float exponent
  // position
  constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
  const nv_bfloat162 bias_reg = __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));

  // Convert to half2 and apply bias
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <typename scalar_t2>
__device__ inline void dequant_fp8_scales(int q, scalar_t2* frag_b);

template <>
__device__ inline void dequant_fp8_scales<half2>(int q, half2* frag_b) {
  int Out1 = (q & 0xFF00FF00) >> 1;
  ;
  q <<= 8;
  int Out2 = (q & 0xFF00FF00) >> 1;

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
};

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162>(int q, nv_bfloat162* frag_b) {
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

  // Extract and shift FP8 values to BF16 format
  int Out1 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
};

// New version with s_type_id parameter for marlin_moe_wna16_v2
template <typename scalar_t2, sglang::ScalarTypeId s_type_id>
__device__ inline void dequant_fp8_scales(int q, scalar_t2* frag_b);

template <>
__device__ inline void dequant_fp8_scales<half2, sglang::kFE4M3fn.id()>(int q, half2* frag_b) {
  int Out1 = (q & 0xFF00FF00) >> 1;
  ;
  q <<= 8;
  int Out2 = (q & 0xFF00FF00) >> 1;

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
};

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, sglang::kFE4M3fn.id()>(int q, nv_bfloat162* frag_b) {
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

  // Extract and shift FP8 values to BF16 format
  int Out1 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, sglang::kFE8M0fnu.id()>(int q, nv_bfloat162* frag_b) {
  // In this conversion, 2 ** -127 in FP8E8M0 would become 0 in BF16,
  // but we assume that such a extreme value would not occur in real models.
  int Out1 = (q & 0xFF00FF00) >> 1;
  q <<= 7;
  int Out2 = q & 0x7F807F80;

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

#endif

// Dtype
template <typename scalar_t>
class ScalarType {};

template <>
class ScalarType<half> {
 public:
  using scalar_t = half;
  using scalar_t2 = half2;

  // Matrix fragments for tensor core instructions; their precise layout is
  // documented here:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  using FragA = Vec<half2, 4>;
  using FragB = Vec<half2, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<half2, 1>;
  using FragZP = Vec<half2, 4>;

  static __device__ float inline num2float(const half x) {
    return __half2float(x);
  }

  static __device__ half2 inline num2num2(const half x) {
    return __half2half2(x);
  }

  static __device__ half2 inline nums2num2(const half x1, const half x2) {
    return __halves2half2(x1, x2);
  }

  static __host__ __device__ half inline float2num(const float x) {
    return __float2half(x);
  }
};

template <>
class ScalarType<nv_bfloat16> {
 public:
  using scalar_t = nv_bfloat16;
  using scalar_t2 = nv_bfloat162;

  using FragA = Vec<nv_bfloat162, 4>;
  using FragB = Vec<nv_bfloat162, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<nv_bfloat162, 1>;
  using FragZP = Vec<nv_bfloat162, 4>;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  static __device__ float inline num2float(const nv_bfloat16 x) {
    return __bfloat162float(x);
  }

  static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x) {
    return __bfloat162bfloat162(x);
  }

  static __device__ nv_bfloat162 inline nums2num2(const nv_bfloat16 x1, const nv_bfloat16 x2) {
    return __halves2bfloat162(x1, x2);
  }

  static __host__ __device__ nv_bfloat16 inline float2num(const float x) {
    return __float2bfloat16(x);
  }
#endif
};

}  // namespace MARLIN_NAMESPACE_NAME

#endif
