// Activation quantizer for the vendored llama.cpp MMQ kernels: x [M, K] in fp32, fp16 or bf16 to the
// block_q8_1_mmq layout of ggml/src/ggml-cuda/mmq.cuh.
//
// Derived from quantize_mmq_q8_1 / quantize_mmq_q8_1_cuda in llama.cpp's ggml/src/ggml-cuda/quantize.cu
// (commit in ggml/LLAMA_CPP_COMMIT; MIT License, Copyright (c) 2023-2026 The ggml authors, see ggml/LICENSE).
// Changes from upstream:
//   * the input is read as fp32, fp16 or bf16 (upstream reads fp32 only), so half-precision activations
//     need no conversion pass; scalar loads are used when the rows are not 4-element aligned;
//   * no channels / samples and no PDL; the MoE gather / scatter variants take the row maps of mmq_moe.cu and
//     skip unused (masked) slots;
//   * the block sums (DS4 and D2S6 layouts) are d * sum(q), the sum of the values the kernel actually
//     multiplies, instead of the sum of the unquantized inputs. The kernels use the sum to apply a per-block
//     weight minimum or offset (Q4_1, Q5_1, Q4_K, Q5_K, Q2_K, IQ1_S, ...); with the quantized sum that term
//     cancels exactly and the result keeps only the q8_1 rounding of the products (SGLang's own q8_1
//     quantizer does the same, see gguf_kernel.cu);
//   * an all-zero block gets d = 0 and q = 0 explicitly.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "common.cuh"
#include "mmq.cuh"
#include "mmq_internal.cuh"
#include "sgl_llama_mmq.h"

namespace sgl_llama_mmq {

static constexpr int kQuantizeBlockSize = 128;  // CUDA_QUANTIZE_BLOCK_SIZE_MMQ in quantize.cuh
static_assert(MATRIX_ROW_PADDING % (4 * kQuantizeBlockSize) == 0, "Risk of out-of-bounds access.");

template <typename T>
struct load4;

template <>
struct load4<float> {
  static __device__ __forceinline__ float4 vec(const float* p) {
    return *(const float4*)p;
  }
};

template <>
struct load4<half> {
  static __device__ __forceinline__ float4 vec(const half* p) {
    const uint2 u = *(const uint2*)p;
    const float2 a = __half22float2(*(const half2*)&u.x);
    const float2 b = __half22float2(*(const half2*)&u.y);
    return make_float4(a.x, a.y, b.x, b.y);
  }
};

template <>
struct load4<nv_bfloat16> {
  static __device__ __forceinline__ float4 vec(const nv_bfloat16* p) {
    const uint2 u = *(const uint2*)p;
    // bf16 -> fp32 is a 16-bit shift.
    return make_float4(
        __uint_as_float(u.x << 16),
        __uint_as_float(u.x & 0xFFFF0000u),
        __uint_as_float(u.y << 16),
        __uint_as_float(u.y & 0xFFFF0000u));
  }
};

template <typename T>
static __device__ __forceinline__ float to_float(const T v) {
  return float(v);
}
template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(const nv_bfloat16 v) {
  return __bfloat162float(v);
}

// Row mapping of the MoE (MUL_MAT_ID) path, see mmq_moe.cu:
//   kPlain:   q8_1 column c = input row c;
//   kGather:  q8_1 column c = input row ids[c], for c < *n_valid (the compact, expert-sorted columns);
//   kScatter: input row t (a token) goes to q8_1 columns ids[t*n_used + s] for every slot s of the token
//             whose expert is valid (slot_expert != INT_MAX): each token is quantized once.
enum QuantizeMode { kPlain = 0, kGather = 1, kScatter = 2 };

struct RowMap {
  const int32_t* ids = nullptr;
  const int32_t* slot_expert = nullptr;
  const int32_t* n_valid = nullptr;
  int n_used = 1;
};

// Grid: (input rows, K_padded / (4 * kQuantizeBlockSize)); each thread quantizes 4 consecutive values.
// ncols: number of columns of the q8_1 buffer (its layout is [K_padded / 128][ncols] blocks).
template <typename T, mmq_q8_1_ds_layout ds_layout, bool vec_load, int mode>
static __global__ void quantize_mmq_q8_1_kernel(
    const T* __restrict__ x,
    void* __restrict__ vy,
    const int64_t K,
    const int64_t ldx,
    const int64_t K_padded,
    const int64_t ncols,
    const RowMap map) {
  constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
  constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;
  if (i0 >= K_padded) {
    return;
  }
  int64_t row = blockIdx.x;
  if constexpr (mode == kGather) {
    if (row >= *map.n_valid) {
      return;  // uniform over the block
    }
    row = map.ids[row];
  }
  const T* xr = x + row * ldx;

  float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (i0 < K) {  // K % 4 == 0 (K is a multiple of the weight block size)
    if constexpr (vec_load) {
      xi = load4<T>::vec(xr + i0);
    } else {
      xi = make_float4(to_float(xr[i0 + 0]), to_float(xr[i0 + 1]), to_float(xr[i0 + 2]), to_float(xr[i0 + 3]));
    }
  }

  float amax = fabsf(xi.x);
  amax = fmaxf(amax, fabsf(xi.y));
  amax = fmaxf(amax, fabsf(xi.z));
  amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
  for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
  }

  const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
  char4 q;
  q.x = roundf(xi.x * d_inv);
  q.y = roundf(xi.y * d_inv);
  q.z = roundf(xi.z * d_inv);
  q.w = roundf(xi.w * d_inv);

  float sum = 0.0f;
  if constexpr (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
    int sumq = (int)q.x + (int)q.y + (int)q.z + (int)q.w;
#pragma unroll
    for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
      sumq += __shfl_xor_sync(0xFFFFFFFF, sumq, offset, WARP_SIZE);
    }
    sum = d * (float)sumq;
  }

  const int64_t k_block = i0 / QK8_1_MMQ;  // block of 128 values along K
  const int64_t iqs = i0 % QK8_1_MMQ;      // value index in the block

  auto write = [&](const int64_t col) {
    block_q8_1_mmq* y = (block_q8_1_mmq*)vy + k_block * ncols + col;
    ((char4*)y->qs)[iqs / 4] = q;
    if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
      if (iqs % 16 == 0 && iqs < 96) {
        y->d2s6[2 + iqs / 16] = sum;
        if (iqs % 64 == 0) {
          y->d2s6[iqs / 64] = d;
        }
      }
    } else if (iqs % 32 == 0) {
      if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y->ds4[iqs / 32] = make_half2(d, sum);
      } else {
        y->d4[iqs / 32] = d;
      }
    }
  };

  if constexpr (mode == kScatter) {
    for (int s = 0; s < map.n_used; ++s) {
      const int64_t slot = row * map.n_used + s;
      if (map.slot_expert[slot] != INT_MAX) {
        write(map.ids[slot]);
      }
    }
  } else {
    write(blockIdx.x);
  }
}

template <typename T, mmq_q8_1_ds_layout ds_layout, int mode>
static void launch_quantize(
    const T* x,
    void* yq,
    int64_t grid_rows,
    int64_t K,
    int64_t ldx,
    int64_t K_padded,
    int64_t ncols,
    const RowMap& map,
    bool vec_load,
    cudaStream_t stream) {
  const dim3 num_blocks(grid_rows, K_padded / (4 * kQuantizeBlockSize), 1);
  const dim3 block_size(kQuantizeBlockSize, 1, 1);
  if (vec_load) {
    quantize_mmq_q8_1_kernel<T, ds_layout, true, mode>
        <<<num_blocks, block_size, 0, stream>>>(x, yq, K, ldx, K_padded, ncols, map);
  } else {
    quantize_mmq_q8_1_kernel<T, ds_layout, false, mode>
        <<<num_blocks, block_size, 0, stream>>>(x, yq, K, ldx, K_padded, ncols, map);
  }
}

template <typename T, int mode>
static void quantize_typed(
    const T* x,
    void* yq,
    int64_t grid_rows,
    int64_t K,
    int64_t ldx,
    ggml_type type,
    int64_t ncols,
    const RowMap& map,
    cudaStream_t stream) {
  const int64_t K_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
  constexpr int64_t vec_bytes = 4 * sizeof(T);
  const bool vec_load = (uintptr_t)x % vec_bytes == 0 && ldx % 4 == 0;
  switch (mmq_get_q8_1_ds_layout(type)) {
    case MMQ_Q8_1_DS_LAYOUT_D4:
      launch_quantize<T, MMQ_Q8_1_DS_LAYOUT_D4, mode>(x, yq, grid_rows, K, ldx, K_padded, ncols, map, vec_load, stream);
      break;
    case MMQ_Q8_1_DS_LAYOUT_DS4:
      launch_quantize<T, MMQ_Q8_1_DS_LAYOUT_DS4, mode>(
          x, yq, grid_rows, K, ldx, K_padded, ncols, map, vec_load, stream);
      break;
    case MMQ_Q8_1_DS_LAYOUT_D2S6:
      launch_quantize<T, MMQ_Q8_1_DS_LAYOUT_D2S6, mode>(
          x, yq, grid_rows, K, ldx, K_padded, ncols, map, vec_load, stream);
      break;
    default:
      GGML_ABORT("unexpected q8_1 layout");
  }
}

template <int mode>
static void quantize_dispatch(
    const void* x,
    XType xtype,
    int64_t grid_rows,
    int64_t K,
    int64_t ldx,
    int64_t type,
    void* yq,
    int64_t ncols,
    const RowMap& map,
    cudaStream_t stream) {
  GGML_ASSERT(K % 4 == 0);
  GGML_ASSERT(grid_rows <= INT_MAX && ncols <= INT_MAX);
  if (grid_rows == 0) {
    return;
  }
  switch (xtype) {
    case XType::F32:
      quantize_typed<float, mode>((const float*)x, yq, grid_rows, K, ldx, (ggml_type)type, ncols, map, stream);
      break;
    case XType::F16:
      quantize_typed<half, mode>((const half*)x, yq, grid_rows, K, ldx, (ggml_type)type, ncols, map, stream);
      break;
    case XType::BF16:
      quantize_typed<nv_bfloat16, mode>(
          (const nv_bfloat16*)x, yq, grid_rows, K, ldx, (ggml_type)type, ncols, map, stream);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

void quantize_q8_1_mmq(
    const void* x, XType xtype, int64_t M, int64_t K, int64_t ldx, int64_t type, void* yq, cudaStream_t stream) {
  quantize_dispatch<kPlain>(x, xtype, M, K, ldx, type, yq, M, RowMap{}, stream);
}

void quantize_q8_1_mmq_gather(
    const void* x,
    XType xtype,
    int64_t ncols,
    int64_t K,
    int64_t ldx,
    int64_t type,
    const int32_t* ids_src,
    const int32_t* n_valid,
    void* yq,
    cudaStream_t stream) {
  RowMap map;
  map.ids = ids_src;
  map.n_valid = n_valid;
  quantize_dispatch<kGather>(x, xtype, ncols, K, ldx, type, yq, ncols, map, stream);
}

void quantize_q8_1_mmq_scatter(
    const void* x,
    XType xtype,
    int64_t n_tokens,
    int n_used,
    int64_t K,
    int64_t ldx,
    int64_t type,
    const int32_t* slot_col,
    const int32_t* slot_expert,
    void* yq,
    cudaStream_t stream) {
  RowMap map;
  map.ids = slot_col;
  map.slot_expert = slot_expert;
  map.n_used = n_used;
  quantize_dispatch<kScatter>(x, xtype, n_tokens, K, ldx, type, yq, n_tokens * n_used, map, stream);
}

}  // namespace sgl_llama_mmq
