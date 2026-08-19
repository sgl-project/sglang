/*
 * Fused Add + RMSNorm + per-token FP8 quantization.
 *
 * Collapses two launches
 *   1. fused_add_rmsnorm(input, residual, weight)   -> normed bf16
 *   2. per_token_quant_fp8(normed)                  -> (fp8, per-token scale)
 * into one, keeping the normed row in registers so the intermediate bf16 row is
 * never round-tripped through HBM.
 *
 *   separate: R(input) + R(residual) + W(residual) + W(bf16)
 *             + R(bf16) + W(fp8) + W(scale)
 *   fused:    R(input) + R(residual) + W(residual) + W(fp8) + W(scale)
 *
 * The normalized value is emitted only in fp8. Every caller feeds it straight
 * into an FP8 linear and takes the residual stream from the in-place `residual`
 * buffer, so the activation-dtype copy #22005 also wrote was pure overhead.
 *
 * Distinct from the existing quant-fusion paths:
 *   - flashinfer rmsnorm_quant / fused_add_rmsnorm_quant is *static per-tensor*
 *     (the scale is an input).
 *   - aiter fused AR+RMSNorm+quant is per-group 1x128 and ROCm-only.
 * This kernel computes a *dynamic per-token* scale (one float per row).
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cooperative_groups/reduce.h>
#include <tvm/ffi/container/tensor.h>

#include <cooperative_groups.h>
#include <cstdint>
#include <type_traits>

namespace sglang {

namespace cg = cooperative_groups;

template <typename T, int VEC_SIZE_IN_BYTE>
struct PerTokenQuantVecTrait;

template <>
struct PerTokenQuantVecTrait<__nv_bfloat16, 16> {
  using packed_t = __nv_bfloat162;
  using vec_t = device::AlignedVector<packed_t, 4>;
  static constexpr int kPacked = 4;
  static constexpr int kElems = 8;
};

template <>
struct PerTokenQuantVecTrait<half, 16> {
  using packed_t = half2;
  using vec_t = device::AlignedVector<packed_t, 4>;
  static constexpr int kPacked = 4;
  static constexpr int kElems = 8;
};

// Arch-aware FP8 e4m3 finite max (448 on CUDA / OCP e4m3fn, 224 on AMD
// e4m3fnuz) -- see sgl_kernel/math.cuh. Hardcoding 448 here would silently
// saturate wrong on e4m3fnuz. Note the constant lives in device::math, not
// math: at this (namespace sglang) scope `device` must be named explicitly.
using device::math::FP8_E4M3_MAX;
// Guards against a 1/0 scale on an all-zero row; mirrors the epsilon used by
// the standalone per-token quant kernel.
__device__ constexpr float kAbsMaxEps = 1e-10f;

/*
 * One CTA per token.
 *
 * The host wrapper rounds blockDim up to a whole number of warps, which is what
 * makes the warp count exact. #22005 launched blockDim.x == vec_hidden_size
 * verbatim, so for any hidden size whose vector width was not a multiple of 32
 * the reductions' `blockDim.x / 32` truncated away the tail warp (and for
 * vec_hidden_size < 32 it yielded zero warps, i.e. rstd == rsqrtf(eps)). With
 * the rounding in place blockDim.x % 32 == 0 always holds; the ceil-div below
 * is therefore belt-and-braces, not the fix, and it is a no-op today. Keep it
 * so the reductions stay correct if the rounding is ever relaxed.
 */
template <bool kHasResidual, bool kAddWeightOne, typename DType, int VEC_SIZE_IN_BYTE>
__device__ __forceinline__ void fused_add_rmsnorm_per_token_quant_body(
    const DType* __restrict__ input,
    DType* __restrict__ residual,
    const DType* __restrict__ weight,
    fp8_e4m3_t* __restrict__ out_fp8,
    float* __restrict__ out_scales,
    int vec_hidden_size,
    float eps) {
  using Traits = PerTokenQuantVecTrait<DType, VEC_SIZE_IN_BYTE>;
  using vec_t = typename Traits::vec_t;
  using packed_t = typename Traits::packed_t;
  constexpr int kPacked = Traits::kPacked;
  constexpr int kElems = Traits::kElems;

  // 32 slots is enough for 1024 threads (the launch cap).
  __shared__ float smem[32];

  const int token_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int num_warps = (blockDim.x + 31) / 32;
  const bool active = tid < vec_hidden_size;

  auto warp = cg::tiled_partition<32>(cg::this_thread_block());

  vec_t v_weight;
  float vals[kElems];
  float acc_sq = 0.0f;

  if (active) {
    const vec_t* p_in = reinterpret_cast<const vec_t*>(input) + token_id * vec_hidden_size;
    const vec_t* p_w = reinterpret_cast<const vec_t*>(weight);
    vec_t v_in = p_in[tid];
    v_weight = p_w[tid];

    if constexpr (kHasResidual) {
      vec_t* p_res = reinterpret_cast<vec_t*>(residual) + token_id * vec_hidden_size;
      vec_t v_res = p_res[tid];
#pragma unroll
      for (int i = 0; i < kPacked; ++i) {
        float2 a = device::cast<float2, packed_t>(v_in[i]);
        float2 b = device::cast<float2, packed_t>(v_res[i]);
        float2 s = make_float2(a.x + b.x, a.y + b.y);
        acc_sq += s.x * s.x + s.y * s.y;
        vals[i * 2] = s.x;
        vals[i * 2 + 1] = s.y;
        v_res[i] = device::cast<packed_t, float2>(s);
      }
      // residual += input, written back in place.
      p_res[tid] = v_res;
    } else {
#pragma unroll
      for (int i = 0; i < kPacked; ++i) {
        float2 a = device::cast<float2, packed_t>(v_in[i]);
        acc_sq += a.x * a.x + a.y * a.y;
        vals[i * 2] = a.x;
        vals[i * 2 + 1] = a.y;
      }
    }
  }

  // ---- CTA reduce: sum of squares -> rstd -------------------------------
  float warp_sum = cg::reduce(warp, acc_sq, cg::plus<float>());
  if (tid % 32 == 0) {
    smem[tid / 32] = warp_sum;
  }
  __syncthreads();
  if (tid < 32) {
    float part = (tid < num_warps) ? smem[tid] : 0.0f;
    float total = cg::reduce(warp, part, cg::plus<float>());
    smem[tid] = rsqrtf(eps + total / static_cast<float>(vec_hidden_size * kElems));
  }
  __syncthreads();
  const float rstd = smem[0];

  // ---- normalize, emit bf16, track per-thread absmax --------------------
  float normed[kElems];
  float local_absmax = 0.0f;

  if (active) {
#pragma unroll
    for (int i = 0; i < kPacked; ++i) {
      float2 w = device::cast<float2, packed_t>(v_weight[i]);
      float wx = kAddWeightOne ? (1.0f + w.x) : w.x;
      float wy = kAddWeightOne ? (1.0f + w.y) : w.y;
      normed[i * 2] = vals[i * 2] * rstd * wx;
      normed[i * 2 + 1] = vals[i * 2 + 1] * rstd * wy;
      local_absmax = fmaxf(local_absmax, fmaxf(fabsf(normed[i * 2]), fabsf(normed[i * 2 + 1])));
    }
    // The normed value is deliberately NOT written back in the activation
    // dtype. #22005 emitted it for "residual-stream consumers", but the only
    // callers are the FP8 fusion paths, which consume the fp8 output and take
    // the residual from the in-place `residual` buffer -- the extra [M, D]
    // store was allocated, filled and discarded. Dropping it is worth ~22-24%
    // at N=2048 (measured 8xH100, D=4096/8192).
  }

  // ---- CTA reduce: absmax -> per-token scale ----------------------------
  // smem is reused, so every thread must be past the rstd broadcast first.
  __syncthreads();
  float warp_max = cg::reduce(warp, local_absmax, cg::greater<float>());
  if (tid % 32 == 0) {
    smem[tid / 32] = warp_max;
  }
  __syncthreads();
  if (tid < 32) {
    float part = (tid < num_warps) ? smem[tid] : 0.0f;
    float total = cg::reduce(warp, part, cg::greater<float>());
    smem[tid] = fmaxf(total, kAbsMaxEps) / FP8_E4M3_MAX;
  }
  __syncthreads();

  const float scale = smem[0];
  const float inv_scale = 1.0f / scale;

  if (tid == 0) {
    out_scales[token_id] = scale;
  }

  if (active) {
    const int64_t base = static_cast<int64_t>(token_id) * vec_hidden_size * kElems + tid * kElems;
#pragma unroll
    for (int i = 0; i < kElems; ++i) {
      float q = fminf(fmaxf(normed[i] * inv_scale, -FP8_E4M3_MAX), FP8_E4M3_MAX);
      out_fp8[base + i] = fp8_e4m3_t(q);
    }
  }
}

template <bool kHasResidual, bool kAddWeightOne, typename DType, int VEC_SIZE_IN_BYTE>
__global__ void fused_add_rmsnorm_per_token_quant_kernel(
    const DType* __restrict__ input,
    DType* __restrict__ residual,
    const DType* __restrict__ weight,
    fp8_e4m3_t* __restrict__ out_fp8,
    float* __restrict__ out_scales,
    int vec_hidden_size,
    float eps) {
  fused_add_rmsnorm_per_token_quant_body<kHasResidual, kAddWeightOne, DType, VEC_SIZE_IN_BYTE>(
      input, residual, weight, out_fp8, out_scales, vec_hidden_size, eps);
}

template <bool kHasResidual, bool kAddWeightOne, typename DType>
struct FusedAddRMSNormPerTokenQuantKernel {
  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView out_fp8,
      tvm::ffi::TensorView out_scales,
      double eps) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_dtype<DType>().with_device(device).verify(input);
    if constexpr (kHasResidual) {
      TensorMatcher({N, D}).with_dtype<DType>().with_device(device).verify(residual);
    }
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(weight);
    TensorMatcher({N, D}).with_dtype<fp8_e4m3_t>().with_device(device).verify(out_fp8);
    TensorMatcher({N, 1}).with_dtype<float>().with_device(device).verify(out_scales);

    const int64_t num_tokens = N.unwrap();
    const int64_t hidden_size = D.unwrap();

    if (num_tokens == 0) {
      return;
    }

    // Only the 16B vector path is instantiated; kMaxVecBytes may be 32 on some
    // targets but the trait table above intentionally pins 16B so the register
    // array stays at 8 elements.
    constexpr int kVecBytes = 16;
    const int elements_in_vec = kVecBytes / sizeof(DType);

    RuntimeCheck(
        hidden_size % elements_in_vec == 0,
        "hidden_size ",
        hidden_size,
        " can not align to elements_in_vec ",
        elements_in_vec);

    const int vec_hidden_size = static_cast<int>(hidden_size / elements_in_vec);
    // Round up to whole warps; the CTA reductions ceil-div to match.
    const uint threads = (vec_hidden_size + 31) / 32 * 32;

    RuntimeCheck(
        threads <= 1024,
        "hidden_size ",
        hidden_size,
        " needs ",
        threads,
        " threads, exceeding the 1024-thread single-CTA-per-token limit");

    // Only dereference the residual view on the residual specialization; the
    // no-residual wrapper is called with a placeholder view.
    DType* residual_ptr = nullptr;
    if constexpr (kHasResidual) {
      residual_ptr = static_cast<DType*>(residual.data_ptr());
    }

    auto kernel = fused_add_rmsnorm_per_token_quant_kernel<kHasResidual, kAddWeightOne, DType, kVecBytes>;
    LaunchKernel(static_cast<uint>(num_tokens), threads, device.unwrap())
        .enable_pdl(false)(
            kernel,
            static_cast<const DType*>(input.data_ptr()),
            residual_ptr,
            static_cast<const DType*>(weight.data_ptr()),
            static_cast<fp8_e4m3_t*>(out_fp8.data_ptr()),
            static_cast<float*>(out_scales.data_ptr()),
            vec_hidden_size,
            static_cast<float>(eps));
  }
};

}  // namespace sglang
