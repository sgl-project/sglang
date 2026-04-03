/*
 * Fused Add + RMSNorm + FP8 Per-Token Quantization CUDA Kernel
 *
 * Replaces three separate kernels:
 *   1. FusedAddRMSNorm(input, residual, weight)     — flashinfer kernel
 *   2. per_token_quant_fp8(normed, output, scale)    — sgl_kernel
 *
 * With a single kernel that:
 *   Phase 1: residual_add + sum_sq accumulation (registers)
 *   Phase 2: normalize + CTA absmax reduce → per-token scale + FP8 quantize
 *
 * Outputs:
 *   - residual: updated in-place (residual += input)
 *   - output_bf16: normed BF16 (for non-FP8 paths, e.g. gate/router)
 *   - output_fp8: normed FP8 (for FP8 GEMM)
 *   - output_scales: [num_tokens, 1] per-token scale
 *
 * Memory savings vs separate kernels:
 *   Separate: Read(inp) + Read(res) + Write(res) + Write(bf16)
 *             + Read(bf16) + Write(fp8) + Write(scale) = 4R + 3W
 *   Fused:   Read(inp) + Read(res) + Write(res)
 *             + Write(bf16) + Write(fp8) + Write(scale) = 2R + 3W (registers hold normed)
 *   Saves: 2 full reads of hidden_dim per token
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr float FP8_E4M3_MAX = 448.0f;

// Packed bf16 traits (same as fused_add_rmsnorm_quant.cuh)
template <typename T> struct VecTrait;
template <> struct VecTrait<__nv_bfloat16> {
  using fp16_t = __nv_bfloat16;
  using fp32x2_t = float2;
  template <typename T> using packed_t = __nv_bfloat162;
  using packed = packed_t<fp16_t>;
  using vec = device::AlignedVector<packed, 4>;
  static constexpr int kInnerLoop = 4;
  static constexpr int kElemsPerVec = 8;
};

/*
 * Fused kernel: residual_add + rmsnorm + per-token FP8 quantization
 *
 * Grid: (num_tokens,)
 * Block: vec_hidden_size threads (hidden_dim / 8)
 *
 * Per-token scale: one float per row, computed as max(abs(normed)) / 448.0
 */
template <typename T>
__global__ void fused_add_rmsnorm_per_token_quant_kernel(
    const T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    T* __restrict__ output_bf16,
    fp8_e4m3_t* __restrict__ output_fp8,
    float* __restrict__ output_scales,  // [num_tokens, 1]
    int vec_hidden_size,  // hidden_dim / 8
    float eps) {

  using Traits = VecTrait<T>;
  using vec_t = typename Traits::vec;
  using packed_t = typename Traits::packed;
  constexpr int kInner = Traits::kInnerLoop;
  constexpr int kElemsPerVec = Traits::kElemsPerVec;

  // Shared memory for CTA-wide reductions (sum_sq and absmax)
  __shared__ float smem[32];

  const int token_id = blockIdx.x;
  const int tid = threadIdx.x;

  // ================================================================
  // Phase 1: Vectorized load + residual add + sum_sq accumulation
  // ================================================================
  vec_t v_inp, v_res, v_weight;
  float vals[kElemsPerVec];  // Register storage for added values
  float acc_sq = 0.0f;

  if (tid < vec_hidden_size) {
    const vec_t* p_inp = reinterpret_cast<const vec_t*>(input) + token_id * vec_hidden_size;
    vec_t* p_res = reinterpret_cast<vec_t*>(residual) + token_id * vec_hidden_size;
    const vec_t* p_weight = reinterpret_cast<const vec_t*>(weight);

    v_inp = p_inp[tid];
    v_res = p_res[tid];
    v_weight = p_weight[tid];

    #pragma unroll
    for (int i = 0; i < kInner; i++) {
      float2 inp_f = device::cast<fp32x2_t, packed_t>(v_inp[i]);
      float2 res_f = device::cast<fp32x2_t, packed_t>(v_res[i]);
      float2 added = make_float2(inp_f.x + res_f.x, inp_f.y + res_f.y);
      acc_sq += added.x * added.x + added.y * added.y;
      vals[i * 2] = added.x;
      vals[i * 2 + 1] = added.y;
      v_res[i] = device::cast<packed_t, fp32x2_t>(added);
    }

    // Write updated residual
    vec_t* p_res_out = reinterpret_cast<vec_t*>(residual) + token_id * vec_hidden_size;
    p_res_out[tid] = v_res;
  }

  // ================================================================
  // CTA reduce for rstd
  // ================================================================
  auto cg_warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  float warp_sum = cooperative_groups::reduce(cg_warp, acc_sq, cooperative_groups::plus<float>());

  if (tid % 32 == 0) {
    smem[tid / 32] = warp_sum;
  }
  __syncthreads();

  if (tid < 32) {
    float cta_sum = cooperative_groups::reduce(
        cg_warp, (tid < blockDim.x / 32) ? smem[tid] : 0.0f, cooperative_groups::plus<float>());
    smem[tid] = rsqrtf(eps + cta_sum / static_cast<float>(vec_hidden_size * kElemsPerVec));
  }
  __syncthreads();

  float rstd = smem[0];  // Broadcast to all threads

  // ================================================================
  // Phase 2a: Normalize + compute local absmax
  // ================================================================
  float normed[kElemsPerVec];
  float local_absmax = 0.0f;

  if (tid < vec_hidden_size) {
    #pragma unroll
    for (int i = 0; i < kInner; i++) {
      float2 w_f = device::cast<fp32x2_t, packed_t>(v_weight[i]);
      normed[i * 2] = vals[i * 2] * rstd * w_f.x;
      normed[i * 2 + 1] = vals[i * 2 + 1] * rstd * w_f.y;
      local_absmax = fmaxf(local_absmax, fmaxf(fabsf(normed[i * 2]), fabsf(normed[i * 2 + 1])));
    }

    // Write BF16 normed output
    vec_t v_out;
    #pragma unroll
    for (int i = 0; i < kInner; i++) {
      v_out[i] = device::cast<packed_t, fp32x2_t>(make_float2(normed[i * 2], normed[i * 2 + 1]));
    }
    vec_t* p_bf16 = reinterpret_cast<vec_t*>(output_bf16) + token_id * vec_hidden_size;
    p_bf16[tid] = v_out;
  }

  // ================================================================
  // Phase 2b: CTA reduce for per-token absmax → scale → FP8 quantize
  // ================================================================
  __syncthreads();

  float warp_max = cooperative_groups::reduce(cg_warp, local_absmax, [](float a, float b) { return fmaxf(a, b); });

  if (tid % 32 == 0) {
    smem[tid / 32] = warp_max;
  }
  __syncthreads();

  if (tid < 32) {
    float cta_max = cooperative_groups::reduce(
        cg_warp, (tid < blockDim.x / 32) ? smem[tid] : 0.0f, [](float a, float b) { return fmaxf(a, b); });
    smem[0] = fmaxf(cta_max, 1e-10f) / FP8_E4M3_MAX;  // scale
  }
  __syncthreads();

  float scale = smem[0];
  float inv_scale = 1.0f / scale;

  // Write per-token scale
  if (tid == 0) {
    output_scales[token_id] = scale;
  }

  // Quantize to FP8
  if (tid < vec_hidden_size) {
    int base_idx = token_id * vec_hidden_size * kElemsPerVec + tid * kElemsPerVec;
    #pragma unroll
    for (int i = 0; i < kElemsPerVec; i++) {
      float q = fminf(fmaxf(normed[i] * inv_scale, -FP8_E4M3_MAX), FP8_E4M3_MAX);
      output_fp8[base_idx + i] = fp8_e4m3_t(q);
    }
  }
}

/*
 * RMSNorm-only variant (no residual add): norm + per-token FP8 quant
 * For layers that don't have residual connection (e.g., first layer input).
 */
template <typename T>
__global__ void fused_rmsnorm_per_token_quant_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output_bf16,
    fp8_e4m3_t* __restrict__ output_fp8,
    float* __restrict__ output_scales,
    int vec_hidden_size,
    float eps) {

  using Traits = VecTrait<T>;
  using vec_t = typename Traits::vec;
  using packed_t = typename Traits::packed;
  constexpr int kInner = Traits::kInnerLoop;
  constexpr int kElemsPerVec = Traits::kElemsPerVec;

  __shared__ float smem[32];

  const int token_id = blockIdx.x;
  const int tid = threadIdx.x;

  vec_t v_inp, v_weight;
  float vals[kElemsPerVec];
  float acc_sq = 0.0f;

  if (tid < vec_hidden_size) {
    const vec_t* p_inp = reinterpret_cast<const vec_t*>(input) + token_id * vec_hidden_size;
    const vec_t* p_weight = reinterpret_cast<const vec_t*>(weight);
    v_inp = p_inp[tid];
    v_weight = p_weight[tid];

    #pragma unroll
    for (int i = 0; i < kInner; i++) {
      float2 inp_f = device::cast<fp32x2_t, packed_t>(v_inp[i]);
      acc_sq += inp_f.x * inp_f.x + inp_f.y * inp_f.y;
      vals[i * 2] = inp_f.x;
      vals[i * 2 + 1] = inp_f.y;
    }
  }

  // CTA reduce for rstd
  auto cg_warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  float warp_sum = cooperative_groups::reduce(cg_warp, acc_sq, cooperative_groups::plus<float>());
  if (tid % 32 == 0) smem[tid / 32] = warp_sum;
  __syncthreads();
  if (tid < 32) {
    float cta_sum = cooperative_groups::reduce(
        cg_warp, (tid < blockDim.x / 32) ? smem[tid] : 0.0f, cooperative_groups::plus<float>());
    smem[tid] = rsqrtf(eps + cta_sum / static_cast<float>(vec_hidden_size * kElemsPerVec));
  }
  __syncthreads();
  float rstd = smem[0];

  // Normalize + absmax
  float normed[kElemsPerVec];
  float local_absmax = 0.0f;

  if (tid < vec_hidden_size) {
    #pragma unroll
    for (int i = 0; i < kInner; i++) {
      float2 w_f = device::cast<fp32x2_t, packed_t>(v_weight[i]);
      normed[i * 2] = vals[i * 2] * rstd * w_f.x;
      normed[i * 2 + 1] = vals[i * 2 + 1] * rstd * w_f.y;
      local_absmax = fmaxf(local_absmax, fmaxf(fabsf(normed[i * 2]), fabsf(normed[i * 2 + 1])));
    }

    vec_t v_out;
    #pragma unroll
    for (int i = 0; i < kInner; i++) {
      v_out[i] = device::cast<packed_t, fp32x2_t>(make_float2(normed[i * 2], normed[i * 2 + 1]));
    }
    vec_t* p_bf16 = reinterpret_cast<vec_t*>(output_bf16) + token_id * vec_hidden_size;
    p_bf16[tid] = v_out;
  }

  // CTA absmax reduce
  __syncthreads();
  float warp_max = cooperative_groups::reduce(cg_warp, local_absmax, [](float a, float b) { return fmaxf(a, b); });
  if (tid % 32 == 0) smem[tid / 32] = warp_max;
  __syncthreads();
  if (tid < 32) {
    float cta_max = cooperative_groups::reduce(
        cg_warp, (tid < blockDim.x / 32) ? smem[tid] : 0.0f, [](float a, float b) { return fmaxf(a, b); });
    smem[0] = fmaxf(cta_max, 1e-10f) / FP8_E4M3_MAX;
  }
  __syncthreads();

  float scale = smem[0];
  float inv_scale = 1.0f / scale;

  if (tid == 0) output_scales[token_id] = scale;

  if (tid < vec_hidden_size) {
    int base_idx = token_id * vec_hidden_size * kElemsPerVec + tid * kElemsPerVec;
    #pragma unroll
    for (int i = 0; i < kElemsPerVec; i++) {
      float q = fminf(fmaxf(normed[i] * inv_scale, -FP8_E4M3_MAX), FP8_E4M3_MAX);
      output_fp8[base_idx + i] = fp8_e4m3_t(q);
    }
  }
}

// ----------------------------------------------------------------
// Host wrappers
// ----------------------------------------------------------------
template <typename DType>
void fused_add_rmsnorm_per_token_quant(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView output_bf16,
    tvm::ffi::TensorView output_fp8,
    tvm::ffi::TensorView output_scales,
    double eps) {
  using namespace host;

  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"num_tokens"};
  auto D = SymbolicSize{"hidden_dim"};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, D}).with_dtype<DType>().with_device(device).verify(input);
  TensorMatcher({M, D}).with_dtype<DType>().with_device(device).verify(residual);
  TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(weight);
  TensorMatcher({M, D}).with_dtype<DType>().with_device(device).verify(output_bf16);
  TensorMatcher({M, D}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_fp8);

  const int64_t m = M.unwrap();
  const int64_t d = D.unwrap();

  RuntimeCheck(d % 8 == 0, "hidden_dim must be divisible by 8");

  const int vec_hidden_size = static_cast<int>(d / 8);

  if (m == 0) return;

  LaunchKernel(static_cast<uint32_t>(m), static_cast<uint32_t>(vec_hidden_size), device.unwrap())(
      fused_add_rmsnorm_per_token_quant_kernel<DType>,
      static_cast<const DType*>(input.data_ptr()),
      static_cast<DType*>(residual.data_ptr()),
      static_cast<const DType*>(weight.data_ptr()),
      static_cast<DType*>(output_bf16.data_ptr()),
      static_cast<fp8_e4m3_t*>(output_fp8.data_ptr()),
      static_cast<float*>(output_scales.data_ptr()),
      vec_hidden_size,
      static_cast<float>(eps));
}

template <typename DType>
void fused_rmsnorm_per_token_quant(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView output_bf16,
    tvm::ffi::TensorView output_fp8,
    tvm::ffi::TensorView output_scales,
    double eps) {
  using namespace host;

  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"num_tokens"};
  auto D = SymbolicSize{"hidden_dim"};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, D}).with_dtype<DType>().with_device(device).verify(input);
  TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(weight);
  TensorMatcher({M, D}).with_dtype<DType>().with_device(device).verify(output_bf16);
  TensorMatcher({M, D}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_fp8);

  const int64_t m = M.unwrap();
  const int64_t d = D.unwrap();

  RuntimeCheck(d % 8 == 0, "hidden_dim must be divisible by 8");

  const int vec_hidden_size = static_cast<int>(d / 8);

  if (m == 0) return;

  LaunchKernel(static_cast<uint32_t>(m), static_cast<uint32_t>(vec_hidden_size), device.unwrap())(
      fused_rmsnorm_per_token_quant_kernel<DType>,
      static_cast<const DType*>(input.data_ptr()),
      static_cast<const DType*>(weight.data_ptr()),
      static_cast<DType*>(output_bf16.data_ptr()),
      static_cast<fp8_e4m3_t*>(output_fp8.data_ptr()),
      static_cast<float*>(output_scales.data_ptr()),
      vec_hidden_size,
      static_cast<float>(eps));
}

}  // namespace
