#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <type_traits>

// ─── type aliases ─────────────────────────────────────────────────────────────
// We intentionally avoid pulling in cutlass headers so that this kernel can be
// compiled with the lightweight JIT path. Instead we use native CUDA types and
// a small compile-time AlignedArray helper.

namespace moe_fused_gate_detail {

// Maximum values-per-thread (VPT) the kernel supports.
// VPT = num_experts / num_expert_group; must be <= MAX_VPT.
static constexpr int WARP_SIZE_CONST = 32;
static constexpr int WARPS_PER_CTA = 6;
static constexpr int MAX_VPT = 32;

// ─── Minimal aligned array (replaces cutlass::AlignedArray) ──────────────────
template <typename T, int N>
struct alignas(sizeof(T) * N <= 16 ? sizeof(T) * N : 16) AlignedArray {
  T data[N];
  __device__ __forceinline__ T& operator[](int i) {
    return data[i];
  }
  __device__ __forceinline__ const T& operator[](int i) const {
    return data[i];
  }
};

// ─── Type trait helpers ───────────────────────────────────────────────────────
template <typename T>
__device__ __forceinline__ bool cmp_gt(const T& a, const T& b) {
  if constexpr (std::is_same<T, __half>::value) {
    return __half2float(a) > __half2float(b);
  } else {
    return a > b;
  }
}

template <typename T>
__device__ __forceinline__ bool cmp_eq(const T& a, const T& b) {
  if constexpr (std::is_same<T, __half>::value) {
    return __half2float(a) == __half2float(b);
  } else {
    return a == b;
  }
}

template <typename T>
__device__ __forceinline__ float to_float(const T& x) {
  if constexpr (std::is_same<T, __half>::value)
    return __half2float(x);
  else if constexpr (std::is_same<T, __nv_bfloat16>::value)
    return __bfloat162float(x);
  else
    return static_cast<float>(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float f) {
  if constexpr (std::is_same<T, __half>::value)
    return __float2half(f);
  else if constexpr (std::is_same<T, __nv_bfloat16>::value)
    return __float2bfloat16(f);
  else
    return static_cast<T>(f);
}

// ─── Kernel params structs ────────────────────────────────────────────────────

// Compile-time (templated) params
template <int VPT_, int NUM_EXPERTS_, int THREADS_PER_ROW_, int ROWS_PER_WARP_, int ROWS_PER_CTA_, int WARPS_PER_CTA_>
struct KernelParams {
  static constexpr int VPT = VPT_;
  static constexpr int NUM_EXPERTS = NUM_EXPERTS_;
  static constexpr int THREADS_PER_ROW = THREADS_PER_ROW_;
  static constexpr int ROWS_PER_WARP = ROWS_PER_WARP_;
  static constexpr int ROWS_PER_CTA = ROWS_PER_CTA_;
  static constexpr int WARPS_PER_CTA = WARPS_PER_CTA_;
};

// Runtime (dynamic) params
struct KernelParamsDynamic {
  int VPT;
  int NUM_EXPERTS;
  int THREADS_PER_ROW;
  int ROWS_PER_WARP;
  int ROWS_PER_CTA;
  int WARPS_PER_CTA;
};

// ─── Core device implementation ───────────────────────────────────────────────
template <typename T, typename Params>
__device__ void moe_fused_gate_impl(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output,
    Params params) {
  int tidx = threadIdx.x;
  int64_t thread_row =
      blockIdx.x * params.ROWS_PER_CTA + threadIdx.y * params.ROWS_PER_WARP + tidx / params.THREADS_PER_ROW;
  if (thread_row >= num_rows) return;

  int64_t topk_excl = topk - num_fused_shared_experts;

  auto* input_ptr = reinterpret_cast<T*>(input);
  auto* bias_ptr = reinterpret_cast<T*>(bias);
  auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;

  int thread_group_idx = tidx % params.THREADS_PER_ROW;
  int first_elt = thread_group_idx * params.VPT;
  T* thread_read_ptr = thread_row_ptr + first_elt;
  T* bias_thread_read_ptr = bias_ptr + first_elt;

  // Load row chunk + bias chunk into registers
  AlignedArray<T, MAX_VPT> row_chunk;
  AlignedArray<T, MAX_VPT> bias_chunk;
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = thread_read_ptr[ii];
    bias_chunk[ii] = bias_thread_read_ptr[ii];
  }

  __syncthreads();

  // Sigmoid
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    float v = to_float(row_chunk[ii]);
    row_chunk[ii] = from_float<T>(1.0f / (1.0f + expf(-v)));
  }
  __syncthreads();

  // Add bias
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    bias_chunk[ii] = from_float<T>(to_float(row_chunk[ii]) + to_float(bias_chunk[ii]));
  }

  // ── Exclude worst (THREADS_PER_ROW - topk_group) expert groups ───────────
#pragma unroll
  for (int k_idx = 0; k_idx < params.THREADS_PER_ROW - topk_group; ++k_idx) {
    int expert = first_elt;
    T max_val = from_float<T>(-FLT_MAX);
    T max_val_second = from_float<T>(-FLT_MAX);
#pragma unroll
    for (int ii = 0; ii < params.VPT; ++ii) {
      T val = bias_chunk[ii];
      if (cmp_gt(val, max_val)) {
        max_val_second = max_val;
        max_val = val;
      } else if (cmp_gt(val, max_val_second)) {
        max_val_second = val;
      }
    }
    float max_sum_f = to_float(max_val) + to_float(max_val_second);
    T max_sum = from_float<T>(max_sum_f);

    // argmin reduce across threads in the row
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_f = __shfl_xor_sync(0xFFFFFFFF, to_float(max_sum), mask, params.THREADS_PER_ROW);
      T other_sum = from_float<T>(other_f);
      int other_exp = __shfl_xor_sync(0xFFFFFFFF, expert, mask, params.THREADS_PER_ROW);
      if (cmp_gt(max_sum, other_sum) || (cmp_eq(other_sum, max_sum) && other_exp > expert)) {
        max_sum = other_sum;
        expert = other_exp;
      }
    }

    // Clear the losing group
    int thread_to_clear = expert / params.VPT;
    if (thread_group_idx == thread_to_clear) {
#pragma unroll
      for (int ii = 0; ii < params.VPT; ++ii)
        bias_chunk[ii] = from_float<T>(FLT_MAX);
    }
  }

  __syncthreads();

  // ── TopK selection ─────────────────────────────────────────────────────────
  float output_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk_excl; ++k_idx) {
    T max_val = bias_chunk[0];
    int expert = first_elt;

    if (!cmp_eq(max_val, from_float<T>(FLT_MAX))) {
#pragma unroll
      for (int ii = 1; ii < params.VPT; ++ii) {
        T val = bias_chunk[ii];
        if (cmp_gt(val, max_val)) {
          max_val = val;
          expert = first_elt + ii;
        }
      }
    } else {
      max_val = from_float<T>(-FLT_MAX);
    }

    // argmax reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_f = __shfl_xor_sync(0xFFFFFFFF, to_float(max_val), mask, params.THREADS_PER_ROW);
      T other_max = from_float<T>(other_f);
      int other_exp = __shfl_xor_sync(0xFFFFFFFF, expert, mask, params.THREADS_PER_ROW);
      if (cmp_gt(other_max, max_val) || (cmp_eq(other_max, max_val) && other_exp < expert)) {
        max_val = other_max;
        expert = other_exp;
      }
    }

    int thr_clear = expert / params.VPT;
    int64_t idx = topk * thread_row + k_idx;

    if (thread_group_idx == thr_clear) {
      int exp_local = expert % params.VPT;
      bias_chunk[exp_local] = from_float<T>(-FLT_MAX);
      output_ptr[idx] = to_float(row_chunk[exp_local]);
      indices_ptr[idx] = static_cast<int32_t>(expert);
    }

    if (thread_group_idx == 0) output_sum += output_ptr[idx];

    __syncthreads();
  }

  // ── Append fused shared expert slots ───────────────────────────────────────
  if (thread_group_idx == 0 && num_fused_shared_experts > 0) {
    int64_t last_idx = topk * thread_row + topk_excl;
    for (int i = 0; i < num_fused_shared_experts; ++i) {
      indices_ptr[last_idx + i] = static_cast<int32_t>(params.NUM_EXPERTS + i);
      output_ptr[last_idx + i] = output_sum / static_cast<float>(routed_scaling_factor);
    }
  }
  __syncthreads();

  // ── Rescale output ──────────────────────────────────────────────────────────
  if (thread_group_idx == 0) {
#pragma unroll
    for (int ii = 0; ii < topk; ++ii) {
      int64_t idx = topk * thread_row + ii;
      output_ptr[idx] /= output_sum;
      if (apply_routed_scaling_factor_on_output) output_ptr[idx] *= static_cast<float>(routed_scaling_factor);
    }
  }
}

// ─── Templated kernel (fast path for known configurations) ────────────────────
template <
    typename T,
    int VPT,
    int NUM_EXPERTS,
    int THREADS_PER_ROW,
    int ROWS_PER_WARP,
    int ROWS_PER_CTA,
    int WARPS_PER_CTA_V>
__global__ void moe_fused_gate_kernel(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  KernelParams<VPT, NUM_EXPERTS, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA_V> params;
  moe_fused_gate_impl<T>(
      input,
      bias,
      output_ptr,
      indices_ptr,
      num_rows,
      topk_group,
      topk,
      num_fused_shared_experts,
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output,
      params);
}

// ─── Dynamic kernel (fallback for arbitrary configurations) ───────────────────
template <typename T>
__global__ void moe_fused_gate_kernel_dynamic(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  KernelParamsDynamic params;
  params.NUM_EXPERTS = static_cast<int>(num_experts);
  params.VPT = static_cast<int>(num_experts / num_expert_group);
  params.THREADS_PER_ROW = static_cast<int>(num_expert_group);
  params.WARPS_PER_CTA = WARPS_PER_CTA;
  params.ROWS_PER_WARP = static_cast<int>(max(static_cast<int64_t>(1), (int64_t)WARP_SIZE_CONST / num_expert_group));
  params.ROWS_PER_CTA = params.WARPS_PER_CTA * params.ROWS_PER_WARP;

  moe_fused_gate_impl<T>(
      input,
      bias,
      output_ptr,
      indices_ptr,
      num_rows,
      topk_group,
      topk,
      num_fused_shared_experts,
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output,
      params);
}

// ─── Launch macro ─────────────────────────────────────────────────────────────
#define LAUNCH_MOE_GATE_CONFIG(                                                                        \
    T_TYPE, EXPERTS, EXPERT_GROUP, stream, inp, bia, out, idx, nr, tkg, tk, nfse, rsf, arsfo)          \
  do {                                                                                                 \
    constexpr int _VPT = (EXPERTS) / (EXPERT_GROUP);                                                   \
    constexpr int _RPW = ((EXPERT_GROUP) <= WARP_SIZE_CONST) ? (WARP_SIZE_CONST / (EXPERT_GROUP)) : 1; \
    constexpr int _RPCA = WARPS_PER_CTA * _RPW;                                                        \
    int64_t _nr = static_cast<int64_t>(nr);                                                            \
    int64_t _rpw = static_cast<int64_t>(_RPW);                                                         \
    int64_t _nw = (_nr + _rpw - 1) / _rpw;                                                             \
    int64_t _nb = (_nw + WARPS_PER_CTA - 1) / WARPS_PER_CTA;                                           \
    dim3 _blk(WARP_SIZE_CONST, WARPS_PER_CTA);                                                         \
    moe_fused_gate_kernel<T_TYPE, _VPT, (EXPERTS), (EXPERT_GROUP), _RPW, _RPCA, WARPS_PER_CTA>         \
        <<<_nb, _blk, 0, stream>>>(inp, bia, out, idx, _nr, tkg, tk, nfse, rsf, arsfo);                \
  } while (0)

// ─── Host launcher / TVM-FFI wrapper ─────────────────────────────────────────

struct MoeFusedGateKernel {
  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView bias,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView indices,
      int64_t num_expert_group,
      int64_t topk_group,
      int64_t topk,
      int64_t num_fused_shared_experts,
      double routed_scaling_factor,
      bool apply_routed_scaling_factor_on_output) {
    using namespace host;

    // ── Validate inputs ───────────────────────────────────────────────────────
    RuntimeCheck(input.ndim() == 2, "moe_fused_gate: input must be 2-D, got ndim=", input.ndim());
    RuntimeCheck(bias.ndim() == 1, "moe_fused_gate: bias must be 1-D, got ndim=", bias.ndim());
    RuntimeCheck(input.dtype() == bias.dtype(), "moe_fused_gate: input and bias must have the same dtype");

    int64_t num_rows = input.size(0);
    int64_t num_experts = input.size(1);

    RuntimeCheck(
        (num_experts & (num_experts - 1)) == 0, "moe_fused_gate: num_experts must be a power of 2, got ", num_experts);
    RuntimeCheck(
        num_experts % num_expert_group == 0,
        "moe_fused_gate: num_experts (",
        num_experts,
        ") must be divisible by num_expert_group (",
        num_expert_group,
        ")");
    int64_t vpt = num_experts / num_expert_group;
    RuntimeCheck(vpt <= MAX_VPT, "moe_fused_gate: num_experts/num_expert_group=", vpt, " exceeds MAX_VPT=", MAX_VPT);

    const cudaStream_t stream = LaunchKernel::resolve_device(input.device());

    void* inp = input.data_ptr();
    void* bia = bias.data_ptr();
    float* out = static_cast<float*>(output.data_ptr());
    int32_t* idx = static_cast<int32_t*>(indices.data_ptr());

    // ── Determine scalar type from DLDataType ─────────────────────────────────
    // DLDataType: {code, bits, lanes}
    // float16 => {2, 16, 1}, bfloat16 => {4, 16, 1} (kDLBfloat), float32 => {2, 32, 1}
    auto dtype = input.dtype();
    bool is_fp16 = (dtype.code == 2 && dtype.bits == 16);
    bool is_bf16 = (dtype.code == 4 && dtype.bits == 16);
    bool is_fp32 = (dtype.code == 2 && dtype.bits == 32);

    RuntimeCheck(
        is_fp16 || is_bf16 || is_fp32,
        "moe_fused_gate: unsupported dtype (code=",
        (int)dtype.code,
        ", bits=",
        (int)dtype.bits,
        ")");

    bool dispatched = false;

    // ── Fast path: known compile-time configurations ──────────────────────────
    auto dispatch_typed = [&](auto type_tag) {
      using T = typename decltype(type_tag)::type;
      switch (num_experts) {
        case 256:
          if (num_expert_group == 8) {
            LAUNCH_MOE_GATE_CONFIG(
                T,
                256,
                8,
                stream,
                inp,
                bia,
                out,
                idx,
                num_rows,
                topk_group,
                topk,
                num_fused_shared_experts,
                routed_scaling_factor,
                apply_routed_scaling_factor_on_output);
            dispatched = true;
          } else if (num_expert_group == 16) {
            LAUNCH_MOE_GATE_CONFIG(
                T,
                256,
                16,
                stream,
                inp,
                bia,
                out,
                idx,
                num_rows,
                topk_group,
                topk,
                num_fused_shared_experts,
                routed_scaling_factor,
                apply_routed_scaling_factor_on_output);
            dispatched = true;
          }
          break;
        case 128:
          if (num_expert_group == 4) {
            LAUNCH_MOE_GATE_CONFIG(
                T,
                128,
                4,
                stream,
                inp,
                bia,
                out,
                idx,
                num_rows,
                topk_group,
                topk,
                num_fused_shared_experts,
                routed_scaling_factor,
                apply_routed_scaling_factor_on_output);
            dispatched = true;
          } else if (num_expert_group == 8) {
            LAUNCH_MOE_GATE_CONFIG(
                T,
                128,
                8,
                stream,
                inp,
                bia,
                out,
                idx,
                num_rows,
                topk_group,
                topk,
                num_fused_shared_experts,
                routed_scaling_factor,
                apply_routed_scaling_factor_on_output);
            dispatched = true;
          }
          break;
        default:
          break;
      }
      if (!dispatched) {
        // dynamic fallback
        int64_t rows_per_warp = max(static_cast<int64_t>(1), (int64_t)WARP_SIZE_CONST / num_expert_group);
        int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
        int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
        dim3 blk(WARP_SIZE_CONST, WARPS_PER_CTA);
        moe_fused_gate_kernel_dynamic<T><<<num_blocks, blk, 0, stream>>>(
            inp,
            bia,
            out,
            idx,
            num_rows,
            num_experts,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output);
        dispatched = true;
      }
    };

    // type tag helper
    struct Fp16Tag {
      using type = __half;
    };
    struct Bf16Tag {
      using type = __nv_bfloat16;
    };
    struct Fp32Tag {
      using type = float;
    };

    if (is_fp16)
      dispatch_typed(Fp16Tag{});
    else if (is_bf16)
      dispatch_typed(Bf16Tag{});
    else
      dispatch_typed(Fp32Tag{});
  }
};

}  // namespace moe_fused_gate_detail

// ─── TVM-FFI export ───────────────────────────────────────────────────────────
// Exposed as: moe_fused_gate(input, bias, output, indices,
//                            num_expert_group, topk_group, topk,
//                            num_fused_shared_experts, routed_scaling_factor,
//                            apply_routed_scaling_factor_on_output)
