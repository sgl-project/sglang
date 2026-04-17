#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>    // For RuntimeCheck, div_ceil
#include <sgl_kernel/math.cuh>   // For device::math::max, abs
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, fp8_e4m3_t
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// ----------------------------------------------------------------
// Fused FP8 per-group quantization + scatter to 3D MoE input.
//
// Replaces two separate operations:
//   1. per_token_group_quant_fp8(hidden_states) → fp8_hidden + scale
//   2. fill_gateup_input(fp8_hidden, scale, src2dst → gateup_3d)
//
// with a single kernel that reads BF16 hidden_states once, quantizes
// each group to FP8 on the fly, and scatters directly to the 3D
// gateup_input buffer using src2dst mapping.
//
// Saves one intermediate FP8 tensor + one kernel launch.
// ----------------------------------------------------------------

constexpr int kThreadsPerGroup = 16;

__device__ __forceinline__ float GroupReduceMax(float val, const int /*tid*/) {
  unsigned mask = threadIdx.x % 32 >= 16 ? 0xffff0000 : 0x0000ffff;
  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

// Each thread group (16 threads) processes one group of `group_size` elements
// for one (src_token, topk_slot) pair.
//
// Grid: total_work_items / groups_per_block
// where total_work_items = total_topk_slots * groups_per_row
// total_topk_slots = num_tokens * topk

template <typename T>
__global__ void fused_quant_scatter_kernel(
    const T* __restrict__ input,          // [num_tokens, hidden_size] BF16
    fp8_e4m3_t* __restrict__ output,      // [num_experts, m_max, hidden_size] FP8
    float* __restrict__ output_scale,     // [num_experts, m_max, num_groups_per_row] FP32
    const int32_t* __restrict__ src2dst,  // [num_tokens * topk] → dst row index in 3D
    const int32_t* __restrict__ topk_ids, // [num_tokens * topk] → expert id
    int32_t hidden_size,
    int32_t group_size,
    int32_t num_groups_per_row,
    int32_t topk,
    int32_t total_topk_slots,
    float eps,
    float fp8_max) {
  using namespace device;
  namespace math = device::math;

  const int local_group_id = threadIdx.x / kThreadsPerGroup;
  const int lane_id = threadIdx.x % kThreadsPerGroup;
  const int groups_per_block = blockDim.x / kThreadsPerGroup;

  const int64_t global_work_id = (int64_t)blockIdx.x * groups_per_block + local_group_id;

  // global_work_id = slot_idx * num_groups_per_row + group_in_row
  const int32_t slot_idx = global_work_id / num_groups_per_row;
  const int32_t group_in_row = global_work_id % num_groups_per_row;

  if (slot_idx >= total_topk_slots) return;

  // Check expert_id >= 0 (filtered experts have -1)
  const int32_t expert_id = topk_ids[slot_idx];
  if (expert_id < 0) return;

  // Source: which token does this slot come from?
  const int32_t src_token = slot_idx / topk;
  const int32_t elem_offset = group_in_row * group_size;

  // Source pointer: input[src_token, elem_offset...]
  const T* src_ptr = input + (int64_t)src_token * hidden_size + elem_offset;

  // Destination: dst row in 3D output
  const int32_t dst_row = src2dst[slot_idx];
  fp8_e4m3_t* dst_ptr = output + (int64_t)dst_row * hidden_size + elem_offset;
  float* dst_scale_ptr = output_scale + (int64_t)dst_row * num_groups_per_row + group_in_row;

  // Pass 1: find absmax in group
  constexpr uint32_t kVecSize = 16 / sizeof(T);
  using vec_t = AlignedVector<T, kVecSize>;
  const int32_t num_vec_elems = group_size / kVecSize;

  float local_absmax = eps;
  for (int32_t i = lane_id; i < num_vec_elems; i += kThreadsPerGroup) {
    vec_t v;
    v.load(src_ptr, i);
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(v[j]);
      local_absmax = math::max(local_absmax, math::abs(val));
    }
  }

  local_absmax = GroupReduceMax(local_absmax, lane_id);
  const float y_s = local_absmax / fp8_max;

  if (lane_id == 0) {
    *dst_scale_ptr = y_s;
  }

  // Pass 2: quantize and scatter
  const float inv_s = 1.0f / y_s;
  for (int32_t i = lane_id; i < num_vec_elems; i += kThreadsPerGroup) {
    vec_t v;
    v.load(src_ptr, i);
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(v[j]);
      float q_val = fminf(fmaxf(val * inv_s, -fp8_max), fp8_max);
      dst_ptr[i * kVecSize + j] = fp8_e4m3_t(q_val);
    }
  }
}

template <typename DType>
void fused_quant_scatter(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView output_scale,
    tvm::ffi::TensorView src2dst,
    tvm::ffi::TensorView topk_ids,
    int64_t group_size,
    int64_t topk,
    double eps,
    double fp8_max) {
  using namespace host;

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  auto M = SymbolicSize{"num_tokens"};
  auto K = SymbolicSize{"hidden_size"};
  TensorMatcher({M, K}).with_dtype<DType>().with_device(device).verify(input);

  const int32_t num_tokens = static_cast<int32_t>(M.unwrap());
  const int32_t hidden_size = static_cast<int32_t>(K.unwrap());
  const int32_t num_groups_per_row = hidden_size / group_size;
  const int32_t total_topk_slots = num_tokens * topk;
  const int64_t total_work = (int64_t)total_topk_slots * num_groups_per_row;

  if (total_work == 0) return;

  const int groups_per_block = 16;  // 16 groups * 16 threads/group = 256 threads
  const int threads_per_block = groups_per_block * kThreadsPerGroup;
  const int num_blocks = div_ceil(static_cast<int>(total_work), groups_per_block);

  LaunchKernel(num_blocks, threads_per_block, input.device())(
      fused_quant_scatter_kernel<DType>,
      static_cast<const DType*>(input.data_ptr()),
      static_cast<fp8_e4m3_t*>(output.data_ptr()),
      static_cast<float*>(output_scale.data_ptr()),
      static_cast<const int32_t*>(src2dst.data_ptr()),
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      hidden_size,
      static_cast<int32_t>(group_size),
      num_groups_per_row,
      static_cast<int32_t>(topk),
      total_topk_slots,
      static_cast<float>(eps),
      static_cast<float>(fp8_max));
}

}  // namespace
