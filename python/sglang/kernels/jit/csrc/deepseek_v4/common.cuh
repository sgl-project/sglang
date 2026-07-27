#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/deepseek_v4/compress.cuh>

#include <dlpack/dlpack.h>

namespace host::compress {

using PlanResult = tvm::ffi::Tuple<uint32_t, uint32_t>;

struct CompressParams {
  PrefillPlan* __restrict__ compress_plan;
  PrefillPlan* __restrict__ write_plan;
  const int64_t* __restrict__ seq_lens;
  const int64_t* __restrict__ extend_lens;
  uint32_t batch_size;
  uint32_t num_tokens;
  uint32_t compress_ratio;
  bool is_overlap;
};

inline constexpr uint32_t kBlockSize = 1024;

#define PLAN_KERNEL __global__ __launch_bounds__(kBlockSize, 1) inline

PLAN_KERNEL void plan_prefill_cuda(const __grid_constant__ CompressParams params) {
  const auto &[
    compress_plan, write_plan, seq_lens, extend_lens, // pointers
    batch_size, num_tokens, compress_ratio, is_overlap // values
  ] = params;

  __shared__ uint32_t compress_counter;
  __shared__ uint32_t write_counter;

  uint32_t batch_id = 0;
  uint32_t counter = 0;
  uint32_t extend_len = extend_lens[0];

  const auto tid = threadIdx.x;
  if (tid == 0) {
    compress_counter = 0;
    write_counter = 0;
  }
  __syncthreads();

  for (uint32_t i = tid; i < num_tokens; i += blockDim.x) {
    const uint32_t ragged_id = i;
    uint32_t j = ragged_id - counter;
    while (j >= extend_len) {
      j -= extend_len;
      batch_id += 1;
      if (batch_id >= batch_size) [[unlikely]]
        break;
      counter += extend_len;
      extend_len = extend_lens[batch_id];
    }
    if (batch_id >= batch_size) [[unlikely]]
      break;
    const uint32_t seq_len = seq_lens[batch_id];
    const uint32_t extend_len = extend_lens[batch_id];
    const uint32_t prefix_len = seq_len - extend_len;
    const uint32_t ratio = compress_ratio * (1 + is_overlap);
    const uint32_t window_len = j + 1 < ratio ? ratio - (j + 1) : 0;
    const uint32_t position = prefix_len + j;
    const auto plan = PrefillPlan{
        .ragged_id = ragged_id,
        .batch_id = batch_id,
        .position = position,
        .window_len = window_len,
    };
    const uint32_t start_write_pos = [seq_len, compress_ratio, is_overlap] {
      const uint32_t pos = seq_len / compress_ratio * compress_ratio;
      if (!is_overlap) return pos;
      return pos >= compress_ratio ? pos - compress_ratio : 0;
    }();
    if ((position + 1) % compress_ratio == 0) {
      const auto write_pos = atomicAdd(&compress_counter, 1);
      compress_plan[write_pos] = plan;
    }
    if (position >= start_write_pos) {
      const auto write_pos = atomicAdd(&write_counter, 1);
      write_plan[write_pos] = plan;
    }
  }
  __syncthreads();
  constexpr auto kInvalid = static_cast<uint32_t>(-1);
  const auto kInvalidPlan = PrefillPlan{kInvalid, kInvalid, kInvalid, kInvalid};
  const auto compress_count = compress_counter;
  const auto write_count = write_counter;
  for (uint32_t i = compress_count + tid; i < num_tokens; i += blockDim.x) {
    compress_plan[i] = kInvalidPlan;
  }
  for (uint32_t i = write_count + tid; i < num_tokens; i += blockDim.x) {
    write_plan[i] = kInvalidPlan;
  }
}

inline PlanResult plan_prefill_host(const CompressParams& params, const bool use_cuda_graph) {
  const auto &[
    compress_ptr, write_ptr, seq_lens_ptr, extend_lens_ptr, // pointers
    batch_size, num_tokens, compress_ratio, is_overlap // values
  ] = params;

  uint32_t counter = 0;
  uint32_t compress_counter = 0;
  uint32_t write_counter = 0;
  const auto ratio = compress_ratio * (1 + is_overlap);
  for (const auto i : irange(batch_size)) {
    const uint32_t seq_len = seq_lens_ptr[i];
    const uint32_t extend_len = extend_lens_ptr[i];
    const uint32_t prefix_len = seq_len - extend_len;
    RuntimeCheck(0 < extend_len && extend_len <= seq_len);
    /// NOTE: `start_write_pos` must be a multiple of `compress_ratio`
    const uint32_t start_write_pos = [seq_len, compress_ratio, is_overlap] {
      const uint32_t pos = seq_len / compress_ratio * compress_ratio;
      if (!is_overlap) return pos;
      /// NOTE: to avoid unsigned integer underflow, don't use `pos - compress_ratio`
      return pos >= compress_ratio ? pos - compress_ratio : 0;
    }();
    /// NOTE: `position` is within [prefix_len, seq_len)
    for (const auto j : irange(extend_len)) {
      const uint32_t position = prefix_len + j;
      const auto plan = PrefillPlan{
          .ragged_id = counter + j,
          .batch_id = i,
          .position = position,
          .window_len = ratio - std::min(j + 1, ratio),
      };
      RuntimeCheck(plan.is_valid(compress_ratio, is_overlap), "Internal error!");
      if ((position + 1) % compress_ratio == 0) {
        compress_ptr[compress_counter++] = plan;
      }
      if (position >= start_write_pos) {
        write_ptr[write_counter++] = plan;
      }
    }
    counter += extend_len;
  }
  RuntimeCheck(counter == num_tokens, "input size ", counter, " != num_q_tokens ", num_tokens);
  if (!use_cuda_graph) return PlanResult{compress_counter, write_counter};
  constexpr auto kInvalid = static_cast<uint32_t>(-1);
  constexpr auto kInvalidPlan = PrefillPlan{kInvalid, kInvalid, kInvalid, kInvalid};
  for (const auto i : irange(compress_counter, num_tokens)) {
    compress_ptr[i] = kInvalidPlan;
  }
  for (const auto i : irange(write_counter, num_tokens)) {
    write_ptr[i] = kInvalidPlan;
  }
  return PlanResult{num_tokens, num_tokens};
}

inline PlanResult plan_prefill(
    const tvm::ffi::TensorView extend_lens,
    const tvm::ffi::TensorView seq_lens,
    const tvm::ffi::TensorView compress_plan,
    const tvm::ffi::TensorView write_plan,
    const uint32_t compress_ratio,
    const bool is_overlap,  // for overlap transform, we have to keep 1 more extra window
    const bool use_cuda_graph) {
  auto N = SymbolicSize{"batch_size"};
  auto M = SymbolicSize{"num_tokens"};
  auto device = SymbolicDevice{};
  const bool is_cuda = [&] {
    if (extend_lens.device().device_type == kDLCUDA) {
      device.set_options<kDLCUDA>();
      return true;
    } else {
      device.set_options<kDLCPU, kDLCUDAHost>();
      return false;
    }
  }();
  TensorMatcher({N})  // extend_lens and seq_lens
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(extend_lens)
      .verify(seq_lens);
  TensorMatcher({M, kPrefillPlanDim})  // compress_plan and write_plan
      .with_dtype<PrefillPlanTensorDtype>()
      .with_device(device)
      .verify(compress_plan)
      .verify(write_plan);

  const auto params = CompressParams{
      .compress_plan = static_cast<PrefillPlan*>(compress_plan.data_ptr()),
      .write_plan = static_cast<PrefillPlan*>(write_plan.data_ptr()),
      .seq_lens = static_cast<const int64_t*>(seq_lens.data_ptr()),
      .extend_lens = static_cast<const int64_t*>(extend_lens.data_ptr()),
      .batch_size = static_cast<uint32_t>(N.unwrap()),
      .num_tokens = static_cast<uint32_t>(M.unwrap()),
      .compress_ratio = compress_ratio,
      .is_overlap = is_overlap,
  };

  if (!is_cuda) return plan_prefill_host(params, use_cuda_graph);
  /// NOTE: cuda kernel plan is naturally compatible with cuda graph
  LaunchKernel(1, kBlockSize, device.unwrap())(plan_prefill_cuda, params);
  return PlanResult{params.num_tokens, params.num_tokens};
}

}  // namespace host::compress

namespace {

[[maybe_unused]]
constexpr auto& plan_compress_prefill = host::compress::plan_prefill;

}  // namespace
