#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>    // For RuntimeCheck, div_ceil
#include <sgl_kernel/math.cuh>   // For device::math::max, abs
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, fp8_e4m3_t
#include <sgl_kernel/vec.cuh>    // For AlignedVector
#include <sgl_kernel/warp.cuh>   // For warp::reduce_max (unused but available)

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// ----------------------------------------------------------------
// Fused SiLU-and-Mul + FP8 per-group quantization kernel.
//
// Input:  [num_tokens, 2 * N]  (bf16) — gated activation input
//         First half: gate (w1), second half: up (w3)
// Output: [num_tokens, N]      (fp8_e4m3) — quantized activated output
// Scales: [num_tokens, N / group_size] (float32) — per-group scales
//
// For each element i in [0, N):
//   val = silu(input[i]) * input[i + N]
//   group_scale = max(abs(val) for val in group) / fp8_max
//   output[i] = fp8(val / group_scale)
//
// kThreadsPerGroup threads cooperatively process one group of
// `group_size` elements. Multiple groups may be processed per block.
// ----------------------------------------------------------------

constexpr int kThreadsPerGroup = 16;

__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = threadIdx.x % 32 >= 16 ? 0xffff0000 : 0x0000ffff;
  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

SGL_DEVICE float silu(float x) {
  return x / (1.0f + expf(-x));
}

template <typename T, bool kIsColumnMajor>
__global__ void fused_silu_mul_quant_kernel(
    const T* __restrict__ input,       // [M, 2*N]
    fp8_e4m3_t* __restrict__ output_q, // [M, N]
    float* __restrict__ output_s,      // [M, N/group_size] or column-major
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const int half_hidden,             // N (= 2*N / 2)
    const float eps,
    const float fp8_max,
    const int num_groups_per_row,
    const int scale_stride) {
  using namespace device;
  namespace math = device::math;

  const int local_group_id = static_cast<int>(threadIdx.x / kThreadsPerGroup);
  const int lane_id = threadIdx.x % kThreadsPerGroup;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;

  if (global_group_id >= num_groups) return;

  // Map global_group_id to (token_idx, group_in_token)
  const int token_idx = global_group_id / num_groups_per_row;
  const int group_in_token = global_group_id % num_groups_per_row;
  const int elem_offset = group_in_token * group_size;

  // Pointers into the gate (first half) and up (second half) of input
  const T* gate_ptr = input + token_idx * (2 * half_hidden) + elem_offset;
  const T* up_ptr = gate_ptr + half_hidden;
  fp8_e4m3_t* out_ptr = output_q + token_idx * half_hidden + elem_offset;

  // Scale output pointer
  float* scale_ptr;
  if constexpr (kIsColumnMajor) {
    scale_ptr = output_s + group_in_token * scale_stride + token_idx;
  } else {
    scale_ptr = output_s + global_group_id;
  }

  // Pass 1: compute silu(gate) * up and find absmax
  constexpr uint32_t kVecSize = 16 / sizeof(T);
  using vec_t = device::AlignedVector<T, kVecSize>;

  const int32_t num_vec_elems = group_size / kVecSize;
  float local_absmax = eps;

  for (int32_t i = lane_id; i < num_vec_elems; i += kThreadsPerGroup) {
    vec_t gate_vec, up_vec;
    gate_vec.load(gate_ptr, i);
    up_vec.load(up_ptr, i);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const float g = static_cast<float>(gate_vec[j]);
      const float u = static_cast<float>(up_vec[j]);
      const float val = silu(g) * u;
      local_absmax = math::max(local_absmax, math::abs(val));
    }
  }

  local_absmax = GroupReduceMax(local_absmax, lane_id);
  const float y_s = local_absmax / fp8_max;

  if (lane_id == 0) {
    *scale_ptr = y_s;
  }

  // Pass 2: quantize silu(gate) * up to fp8
  const float inv_s = 1.0f / y_s;
  for (int32_t i = lane_id; i < num_vec_elems; i += kThreadsPerGroup) {
    vec_t gate_vec, up_vec;
    gate_vec.load(gate_ptr, i);
    up_vec.load(up_ptr, i);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const float g = static_cast<float>(gate_vec[j]);
      const float u = static_cast<float>(up_vec[j]);
      const float val = silu(g) * u;
      const float q_val = fminf(fmaxf(val * inv_s, -fp8_max), fp8_max);
      out_ptr[i * kVecSize + j] = fp8_e4m3_t(q_val);
    }
  }
}

inline int compute_groups_per_block(int64_t num_groups) {
  if (num_groups % 16 == 0) return 16;
  if (num_groups % 8 == 0) return 8;
  if (num_groups % 4 == 0) return 4;
  if (num_groups % 2 == 0) return 2;
  return 1;
}

template <typename DType>
void fused_silu_mul_quant(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output_q,
    tvm::ffi::TensorView output_s,
    int64_t group_size,
    double eps,
    double fp8_max,
    bool column_major_scales) {
  using namespace host;

  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"num_tokens"};
  auto K2 = SymbolicSize{"hidden_dim_2x"};
  auto K = SymbolicSize{"hidden_dim"};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, K2}).with_dtype<DType>().with_device(device).verify(input);
  TensorMatcher({M, K}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_q);

  const int64_t m = M.unwrap();
  const int64_t k2 = K2.unwrap();
  const int64_t k = K.unwrap();

  RuntimeCheck(k2 == 2 * k, "input hidden dim must be 2x output hidden dim");
  RuntimeCheck(k % group_size == 0, "hidden_dim must be divisible by group_size");

  const int64_t num_groups_per_row = k / group_size;
  const int64_t total_groups = m * num_groups_per_row;

  // Scale stride for column-major layout
  int scale_stride = 0;
  if (column_major_scales) {
    // output_s shape: [num_groups_per_row, m] (column-major)
    scale_stride = static_cast<int>(m);
  }

  const int groups_per_block = compute_groups_per_block(total_groups);
  const int threads_per_block = groups_per_block * kThreadsPerGroup;
  const int num_blocks = div_ceil(static_cast<int>(total_groups), groups_per_block);

  if (column_major_scales) {
    LaunchKernel(num_blocks, threads_per_block, input.device())(
        fused_silu_mul_quant_kernel<DType, true>,
        static_cast<const DType*>(input.data_ptr()),
        static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        static_cast<int>(group_size),
        static_cast<int>(total_groups),
        groups_per_block,
        static_cast<int>(k),
        static_cast<float>(eps),
        static_cast<float>(fp8_max),
        static_cast<int>(num_groups_per_row),
        scale_stride);
  } else {
    LaunchKernel(num_blocks, threads_per_block, input.device())(
        fused_silu_mul_quant_kernel<DType, false>,
        static_cast<const DType*>(input.data_ptr()),
        static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        static_cast<int>(group_size),
        static_cast<int>(total_groups),
        groups_per_block,
        static_cast<int>(k),
        static_cast<float>(eps),
        static_cast<float>(fp8_max),
        static_cast<int>(num_groups_per_row),
        scale_stride);
  }
}

}  // namespace
