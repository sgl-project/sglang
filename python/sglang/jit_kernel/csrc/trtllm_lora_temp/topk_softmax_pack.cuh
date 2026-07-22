/*
 * Fused top-k gating softmax WITH routed-pack output.
 *
 * JIT port of the power-of-2 fast path of sgl-kernel's
 * csrc/moe/moe_topk_softmax_kernels.cu (`topkGatingSoftmax`, itself adapted from
 * vLLM v0.7.3 / TensorRT-LLM v0.7.1, Apache-2.0), extended with a third output:
 * the FlashInfer routed-MoE packed format
 *
 *     packed[idx] = (topk_id << 16) | bf16_bits(topk_weight)
 *
 * computed in the kernel epilogue AFTER renormalization — bit-identical to the
 * standalone `fused_pack_topk` triton kernel applied to the (post-processed)
 * topk_ids/topk_weights, including the padded-region mask: rows at or beyond
 * `num_token_non_padded` pack id = -1 (the `_mask_topk_ids_padded_region`
 * sentinel), matching what the separate pack would produce after the mask.
 * This removes the per-MoE-layer `_pack_topk_kernel` launch from the decode
 * critical path entirely (fusion instead of stream overlap).
 *
 * Scope intentionally narrowed vs the AOT kernel (callers fall back to the AOT
 * topk_softmax + separate pack otherwise):
 *   - power-of-2 num_experts in [1, 512] only (no cub workspace fallback)
 *   - no softcapping / correction bias (the Qwen3-MoE softmax path uses neither)
 */
#include <sgl_kernel/tensor.h>  // TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // RuntimeCheck

#include <sgl_kernel/utils.cuh>  // LaunchKernel, fp32_t/fp16_t/bf16_t, is_type

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

namespace {

static constexpr int WARP_SIZE = 32;

#define TSP_MAX(a, b) ((a) > (b) ? (a) : (b))
#define TSP_MIN(a, b) ((a) < (b) ? (a) : (b))

/// Aligned array type (mirrors the AOT kernel's CUTLASS-free aligned array)
template <typename T, int N, int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
  T data[N];
};

template <typename T>
__device__ float convert_to_float(T x) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(x);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(x);
  } else if constexpr (std::is_same_v<T, float>) {
    return x;
  } else {
    return static_cast<float>(x);
  }
}

// Reference pack (bit-identical to jit_kernel/flashinfer_trtllm_moe/topk_pack.py):
// low 16 bits = bf16(weight) bits (round-to-nearest-even, same as torch/triton
// `.to(bfloat16)`), high 16 bits = int16 expert id.
__device__ __forceinline__ int32_t pack_routed(int32_t id, float w) {
  const uint32_t wbits = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(w)));
  return static_cast<int32_t>((static_cast<uint32_t>(id) << 16) | wbits);
}

template <typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void topkGatingSoftmaxPack(
    const T* input,
    float* output,
    const int num_rows,
    int* indices,
    int* packed_output,
    const int32_t* num_token_non_padded,
    const int k,
    const bool renormalize) {
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

  static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;
  if (thread_row >= num_rows) {
    return;
  }

  const T* thread_row_ptr = input + static_cast<int64_t>(thread_row) * ELTS_PER_ROW;
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  using AccessType = AlignedArray<T, ELTS_PER_LDG>;

  T row_chunk_temp[VPT];
  AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk_temp);
  const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }

  float row_chunk[VPT];
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = convert_to_float<T>(row_chunk_temp[ii]);
  }

  float thread_max = row_chunk[0];
#pragma unroll
  for (int ii = 1; ii < VPT; ++ii) {
    thread_max = max(thread_max, row_chunk[ii]);
  }

#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max = max(thread_max, __shfl_xor_sync(0xffffffffu, thread_max, mask, THREADS_PER_ROW));
  }

  float row_sum = 0;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = expf(row_chunk[ii] - thread_max);
    row_sum += row_chunk[ii];
  }

#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    row_sum += __shfl_xor_sync(0xffffffffu, row_sum, mask, THREADS_PER_ROW);
  }

  const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
  }

  int start_col = first_elt_read_by_thread;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  float row_sum_for_renormalize = 0;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        float val = row_chunk[ldg * ELTS_PER_LDG + ii];
        if (val > max_val) {
          max_val = val;
          expert = col + ii;
        }
      }
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max = __shfl_xor_sync(0xffffffffu, max_val, mask, THREADS_PER_ROW);
      int other_expert = __shfl_xor_sync(0xffffffffu, expert, mask, THREADS_PER_ROW);
      if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    if (thread_group_idx == 0) {
      const int64_t idx = static_cast<int64_t>(k) * thread_row + k_idx;
      output[idx] = max_val;
      indices[idx] = expert;
      row_sum_for_renormalize += max_val;
    }

    if (k_idx + 1 < k) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
      }
    }
  }

  if (thread_group_idx == 0) {
    // Fused renormalization (same as the AOT kernel).
    if (renormalize) {
      float row_sum_for_renormalize_inv = 1.f / row_sum_for_renormalize;
#pragma unroll
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int64_t idx = static_cast<int64_t>(k) * thread_row + k_idx;
        output[idx] = output[idx] * row_sum_for_renormalize_inv;
      }
    }
    // Fused routed pack: pack the FINAL (post-renorm) weights. Padded rows
    // (>= *num_token_non_padded) pack id = -1, mirroring the in-place
    // `_mask_topk_ids_padded_region` sentinel that the separate pack kernel
    // would otherwise observe. The plain `indices` output is left unmasked
    // here exactly like the AOT kernel — the existing python post-process
    // masks it afterwards; only the packed tensor needs the mask baked in
    // because it is produced BEFORE that post-process runs.
    const bool row_padded = (num_token_non_padded != nullptr) && (thread_row >= *num_token_non_padded);
#pragma unroll
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int64_t idx = static_cast<int64_t>(k) * thread_row + k_idx;
      const int32_t id = row_padded ? -1 : indices[idx];
      packed_output[idx] = pack_routed(id, output[idx]);
    }
  }
}

namespace detail {
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
  static constexpr int VECs_PER_THREAD = TSP_MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
}  // namespace detail

template <typename T, int EXPERTS, int WARPS_PER_TB>
void launchTopkGatingSoftmaxPack(
    const T* input,
    float* output,
    int* indices,
    int* packed_output,
    const int32_t* num_token_non_padded,
    const int num_rows,
    const int k,
    const bool renormalize,
    DLDevice device) {
  static constexpr std::size_t MAX_BYTES_PER_LDG = 16;
  static constexpr int BYTES_PER_LDG = TSP_MIN(MAX_BYTES_PER_LDG, sizeof(T) * EXPERTS);
  using Constants = detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
  host::LaunchKernel(dim3(num_blocks), block_dim, device)(
      topkGatingSoftmaxPack<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG>,
      input,
      output,
      num_rows,
      indices,
      packed_output,
      num_token_non_padded,
      k,
      renormalize);
}

template <typename T>
void dispatchExperts(
    const T* input,
    float* output,
    int* indices,
    int* packed_output,
    const int32_t* num_token_non_padded,
    const int num_rows,
    const int num_experts,
    const int k,
    const bool renormalize,
    DLDevice device) {
  static constexpr int WARPS_PER_TB = 4;
#define TSP_LAUNCH(E)                              \
  launchTopkGatingSoftmaxPack<T, E, WARPS_PER_TB>( \
      input, output, indices, packed_output, num_token_non_padded, num_rows, k, renormalize, device)
  switch (num_experts) {
    case 1:
      TSP_LAUNCH(1);
      break;
    case 2:
      TSP_LAUNCH(2);
      break;
    case 4:
      TSP_LAUNCH(4);
      break;
    case 8:
      TSP_LAUNCH(8);
      break;
    case 16:
      TSP_LAUNCH(16);
      break;
    case 32:
      TSP_LAUNCH(32);
      break;
    case 64:
      TSP_LAUNCH(64);
      break;
    case 128:
      TSP_LAUNCH(128);
      break;
    case 256:
      TSP_LAUNCH(256);
      break;
    case 512:
      TSP_LAUNCH(512);
      break;
    default:
      host::RuntimeCheck(false, "topk_softmax_pack: num_experts must be a power of 2 in [1, 512], got ", num_experts);
  }
#undef TSP_LAUNCH
}

// ─────────────────────────────────────────────────────────────────────────────
// Launcher
// ─────────────────────────────────────────────────────────────────────────────
void topk_softmax_pack(
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView topk_indices,
    tvm::ffi::TensorView packed,
    tvm::ffi::TensorView gating_output,
    tvm::ffi::Optional<tvm::ffi::TensorView> num_token_non_padded,
    bool renormalize) {
  using namespace host;

  SymbolicSize N{"num_tokens"};
  SymbolicSize E{"num_experts"};
  SymbolicSize K{"topk"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N, E}).with_dtype<fp32_t, fp16_t, bf16_t>().with_device<kDLCUDA>(device_).verify(gating_output);
  TensorMatcher({N, K}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(topk_weights);
  TensorMatcher({N, K}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk_indices);
  TensorMatcher({N, K}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(packed);

  const int32_t* ntnp_ptr = nullptr;
  if (num_token_non_padded.has_value()) {
    SymbolicSize One{"ntnp_numel"};
    TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(num_token_non_padded.value());
    RuntimeCheck(One.unwrap() == 1, "num_token_non_padded must be a 1-element tensor");
    ntnp_ptr = static_cast<const int32_t*>(num_token_non_padded.value().data_ptr());
  }

  const int num_tokens = static_cast<int>(N.unwrap());
  const int num_experts = static_cast<int>(E.unwrap());
  const int topk = static_cast<int>(K.unwrap());
  DLDevice device = device_.unwrap();

  RuntimeCheck(topk <= num_experts, "topk must be <= num_experts");
  if (num_tokens == 0) return;

  auto* weights_ptr = static_cast<float*>(topk_weights.data_ptr());
  auto* indices_ptr = static_cast<int*>(topk_indices.data_ptr());
  auto* packed_ptr = static_cast<int*>(packed.data_ptr());

  if (is_type<fp32_t>(gating_output.dtype())) {
    dispatchExperts<float>(
        static_cast<const float*>(gating_output.data_ptr()),
        weights_ptr,
        indices_ptr,
        packed_ptr,
        ntnp_ptr,
        num_tokens,
        num_experts,
        topk,
        renormalize,
        device);
  } else if (is_type<fp16_t>(gating_output.dtype())) {
    dispatchExperts<__half>(
        static_cast<const __half*>(gating_output.data_ptr()),
        weights_ptr,
        indices_ptr,
        packed_ptr,
        ntnp_ptr,
        num_tokens,
        num_experts,
        topk,
        renormalize,
        device);
  } else {
    dispatchExperts<__nv_bfloat16>(
        static_cast<const __nv_bfloat16*>(gating_output.data_ptr()),
        weights_ptr,
        indices_ptr,
        packed_ptr,
        ntnp_ptr,
        num_tokens,
        num_experts,
        topk,
        renormalize,
        device);
  }
}

}  // namespace
