// Partially migrated from AOT kernel:
// https://github.com/sgl-project/sglang/blob/v0.5.9/sgl-kernel/csrc/allreduce/custom_all_reduce.cu
// Which was originally adapted from:
// https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/custom_all_reduce.cu
// We redesign the controller interface to minimize control plane traffic,
// and fuse the reduce-scatter and broadcast in the 2-shot all reduce
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/common.cuh>
#include <sgl_kernel/distributed/custom_all_reduce.cuh>

#include <bit>
#include <cstdint>
#include <cstring>

namespace {

using device::distributed::PullController;
using host::distributed::AllReduceData;
using host::distributed::CustomAllReduceBase, host::distributed::CustomAllReduceRef;

struct AllReduceParams {
  void* __restrict__ output;
  uint32_t rank;
  uint32_t num_items;  // NOTE: support at most 4G, but that's too much
};

[[maybe_unused]]
SGL_DEVICE void prefetch_uniform_ptr(const void* ptr) {
  asm volatile("prefetchu.L1 [%0];" ::"l"(ptr) : "memory");
}

#define CUSTOM_AR_KERNEL __global__ __launch_bounds__(1024, 1)

template <bool kBroadcast, typename DType, uint32_t kNumGPU>
SGL_DEVICE void all_reduce_impl(const AllReduceParams& params, DType* (&input)[kNumGPU]) {
  using namespace device;

  constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  using DType2 = packed_t<DType>;
  using Storage = AlignedVector<DType2, kVecSize>;
  const auto& [output, rank, num_items] = params;

  for (auto i = blockIdx.x;; i += gridDim.x) {
    const auto offset = i * blockDim.x + threadIdx.x;
    if (offset * kVecSize * 2 >= num_items) break;
    Storage storage[kNumGPU];

#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i) {
      storage[i].load(input[i], offset);
    }
    const Storage result = distributed::reduce_impl(storage);
    if constexpr (kBroadcast) {
#pragma unroll
      for (uint32_t i = 0; i < kNumGPU; ++i) {
        result.store(input[i], offset);
      }
    } else {
      result.store(output, offset);
    }
  }
}

template <typename DType, uint32_t kNumGPU, bool kUsePDL>
CUSTOM_AR_KERNEL void all_reduce_one_shot_kernel(
    const AllReduceData* __restrict__ data,
    const AllReduceParams __grid_constant__ params,
    const PullController __grid_constant__ ctrl) {
  /// NOTE: we assume the data array is ready before the previous kernel
  DType* input[kNumGPU];
  prefetch_uniform_ptr(data);
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i)
    input[i] = static_cast<DType*>(data->input[i]);
  device::PDLWaitPrimary<kUsePDL>();

  ctrl.sync</*kFence=*/0, /*kStart=*/1>(params.rank, kNumGPU);
  all_reduce_impl</*kBroadcast=*/false>(params, input);

  device::PDLTriggerSecondary<kUsePDL>();
  ctrl.sync</*kFence=*/0, /*kStart=*/0>(params.rank, kNumGPU);
}

template <typename DType, uint32_t kNumGPU, bool kUsePDL>
CUSTOM_AR_KERNEL void all_reduce_two_shot_kernel(
    const AllReduceData* __restrict__ data,
    const AllReduceParams __grid_constant__ params,
    const PullController __grid_constant__ ctrl) {
  // get the range of this rank
  using device::kWarpThreads, device::div_ceil;

  prefetch_uniform_ptr(data);
  DType* input[kNumGPU];
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i)
    input[i] = static_cast<DType*>(data->input[i]);

  constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  const uint32_t num_items = params.num_items;
  const uint32_t total_vec = num_items / (kVecSize * 2);  // must be divisible here
  const uint32_t vec_per_rank = div_ceil(div_ceil(total_vec, kNumGPU), kWarpThreads) * kWarpThreads;
  const uint32_t local_vec_start = min(params.rank * vec_per_rank, total_vec);
  const uint32_t local_vec_finish = min(local_vec_start + vec_per_rank, total_vec);
  const uint32_t local_start = local_vec_start * kVecSize * 2;
  const uint32_t local_length = (local_vec_finish - local_vec_start) * kVecSize * 2;
  const auto local_params = AllReduceParams{
      .output = nullptr,  // this is not used for 2-shot all reduce
      .rank = params.rank,
      .num_items = local_length,
  };

#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i)
    input[i] += local_start;

  device::PDLWaitPrimary<kUsePDL>();

  ctrl.sync</*kFence=*/0, /*kStart=*/1>(params.rank, kNumGPU);
  all_reduce_impl</*kBroadcast=*/true>(local_params, input);

  device::PDLTriggerSecondary<kUsePDL>();
  ctrl.sync</*kFence=*/1, /*kStart=*/0>(params.rank, kNumGPU);
}

template <typename DType, uint32_t kNumGPU, bool kUsePDL>
struct CustomAllReducePull : public CustomAllReduceBase {
  static constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  static constexpr auto one_shot_kernel = all_reduce_one_shot_kernel<DType, kNumGPU, kUsePDL>;
  static constexpr auto two_shot_kernel = all_reduce_two_shot_kernel<DType, kNumGPU, kUsePDL>;
  static_assert(kNumGPU <= device::distributed::kMaxNumGPU, "kNumGPU exceeds the maximum supported GPUs");

  tvm::ffi::Tensor all_reduce(tvm::ffi::Tensor input, int shot) {
    using namespace host;
    const bool use_2shot = (shot == 2);
    const auto device = input.device();
    const auto input_ptr = input.data_ptr();
    const auto buffer_ptr = get_pull_buffer(m_storage);
    const auto num_items_int64 = input.numel();
    const auto num_items = static_cast<uint32_t>(num_items_int64);
    const auto items_per_block = m_cta_size * kVecSize * 2;
    const auto needed_blocks = div_ceil(num_items, items_per_block);
    const auto num_blocks = std::min(needed_blocks, m_num_cta);
    const auto kernel = use_2shot ? two_shot_kernel : one_shot_kernel;
    // only 1-shot + graph capture need extra output buffer
    const auto output = (m_is_graph_capturing && !use_2shot) ? ffi::empty_like(input) : input;
    const auto params = AllReduceParams{
        .output = use_2shot ? nullptr : output.data_ptr(),
        .rank = m_rank,
        .num_items = num_items,
    };

    RuntimeCheck(input.IsContiguous(), "Input tensor must be contiguous");
    RuntimeCheck(m_num_gpu == kNumGPU, "Mismatch GPU count");
    RuntimeCheck(shot == 1 || shot == 2, "Invalid shot count: ", shot);
    RuntimeCheck(device.device_type == kDLCUDA, "Only CUDA device is supported");
    RuntimeCheck(is_type<DType>(input.dtype()), "Input dtype mismatch");
    RuntimeCheck(std::bit_cast<intptr_t>(input_ptr) % 16 == 0, "Input pointer is not properly aligned");
    RuntimeCheck(m_pull_ctrl.has_value(), "Controller is not initialized");
    RuntimeCheck(static_cast<int64_t>(num_items) == num_items_int64, "Number of items exceeds 4G limit");

    const auto& ctrl = *m_pull_ctrl;
    const auto stream = LaunchKernel::resolve_device(device);
    auto launch = LaunchKernel{num_blocks, m_cta_size, stream};
    launch.enable_pdl(kUsePDL);
    const auto check_capturing = [&] {
      if (!m_is_graph_capturing) return false;  // override to avoid cudaRT call overhead
      cudaStreamCaptureStatus status;
      RuntimeDeviceCheck(cudaStreamIsCapturing(stream, &status));
      return status == cudaStreamCaptureStatusActive;
    };
    if (check_capturing()) {
      // no-op if not really capturing, we're in a dummy run
      const auto data_ptr = allocate_graph_capture_input(input_ptr);
      /// NOTE: we assume when the graph is replayed, the data_ptr should be ready
      launch(kernel, data_ptr, params, ctrl);
    } else {
      // 1.copy the input to the buffer
      const auto input_bytes = static_cast<int64_t>(sizeof(DType) * num_items);
      RuntimeCheck(input_bytes <= m_pull_buffer_bytes, "Input is too large, num items: ", num_items);
      RuntimeDeviceCheck(cudaMemcpyAsync(buffer_ptr, input_ptr, input_bytes, cudaMemcpyDeviceToDevice, stream));
      // 2. launch the all reduce kernel
      const auto data_ptr = get_data_ptr();  // use default buffer
      launch(kernel, data_ptr, params, ctrl);
      if (use_2shot) {  // 3. copy the reduced result back to the output, because 2-shot doesn't write to output
        RuntimeDeviceCheck(cudaMemcpyAsync(input_ptr, buffer_ptr, input_bytes, cudaMemcpyDeviceToDevice, stream));
      }
    }
    return output;
  }
};

template <typename DType, uint32_t kNumGPU, bool kUsePDL>
tvm::ffi::Tensor custom_all_reduce(CustomAllReduceRef obj, tvm::ffi::Tensor input, int shot) {
  using Impl = CustomAllReducePull<DType, kNumGPU, kUsePDL>;
  return static_cast<Impl&>(*obj.get()).all_reduce(input, shot);
}

}  // namespace
