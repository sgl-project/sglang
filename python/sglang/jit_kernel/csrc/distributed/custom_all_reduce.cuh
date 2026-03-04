#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/all_reduce.cuh>
#include <sgl_kernel/distributed/common.cuh>

#include <cstdint>
#include <cstring>

namespace {

using DistributedController = device::distributed::Controller;
using host::distributed::CustomAllReduceBase, host::distributed::CustomAllReduceRef, host::distributed::AllReduceData;

struct AllReduceParams {
  void* __restrict__ output;
  uint32_t rank;
  uint32_t num_items;  // NOTE: support at most 4G, but that's too much
};

template <bool kBroadcast, typename DType, uint32_t kNumGPU>
SGL_DEVICE void reduce_impl(const AllReduceParams& params, DType* (&input)[kNumGPU]) {
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
      storage[i] = load_as<Storage>(input[i], offset);
    }

    fp32x2_t acc[kVecSize] = {};
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const auto [x, y] = cast<fp32x2_t>(storage[i][j]);
        auto& [x_acc, y_acc] = acc[j];
        x_acc += x;
        y_acc += y;
      }
    }

    Storage result;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      result[j] = cast<DType2>(acc[j]);
    }

    if constexpr (kBroadcast) {
#pragma unroll
      for (uint32_t i = 0; i < kNumGPU; ++i) {
        store_as<Storage>(input[i], result, offset);
      }
    } else {
      store_as<Storage>(output, result, offset);
    }
  }
}

template <typename DType, uint32_t kNumGPU, bool kUsePDL>
__global__ void all_reduce_one_shot_kernel(
    const AllReduceData* __restrict__ data,
    const AllReduceParams __grid_constant__ params,
    const DistributedController __grid_constant__ ctrl) {
  /// NOTE: we assume the data array is ready before the previous kernel
  DType* input[kNumGPU];
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i)
    input[i] = static_cast<DType*>(data->input[i]);
  device::PDLWaitPrimary<kUsePDL>();

  ctrl.sync(params.rank, kNumGPU, /*stage=*/1);
  reduce_impl</*kBroadcast=*/false>(params, input);

  device::PDLTriggerSecondary<kUsePDL>();
  ctrl.reset(params.rank, kNumGPU, /*stage=*/1);
}

template <typename DType, uint32_t kNumGPU, bool kUsePDL>
__global__ void all_reduce_two_shot_kernel(
    const AllReduceData* __restrict__ data,
    const AllReduceParams __grid_constant__ params,
    const DistributedController __grid_constant__ ctrl) {
  // get the range of this rank
  using device::kWarpThreads, device::div_ceil;

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
      .output = nullptr,  // this is not used for 2 shot all reduce
      .rank = params.rank,
      .num_items = local_length,
  };

#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i)
    input[i] += local_start;

  device::PDLWaitPrimary<kUsePDL>();

  ctrl.sync(local_params.rank, kNumGPU, /*stage=*/1);
  reduce_impl</*kBroadcast=*/true>(local_params, input);
  ctrl.sync_release(local_params.rank, kNumGPU, /*stage=*/2);

  device::PDLTriggerSecondary<kUsePDL>();
  ctrl.reset(local_params.rank, kNumGPU, /*stage=*/2);
}

template <typename DType, uint32_t kNumGPU, bool kUsePDL>
struct CustomAllReduceImpl : public CustomAllReduceBase {
  static constexpr uint32_t kBlockSize = 256;
  static constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  static constexpr uint32_t kItemsPerBlock = kBlockSize * kVecSize * 2;
  static constexpr auto one_shot_kernel = all_reduce_one_shot_kernel<DType, kNumGPU, kUsePDL>;
  static constexpr auto two_shot_kernel = all_reduce_two_shot_kernel<DType, kNumGPU, kUsePDL>;
  static_assert(kNumGPU <= device::distributed::kMaxNumGPU, "kNumGPU exceeds the maximum supported GPUs");
  static_assert(kBlockSize >= kNumGPU * device::kWarpThreads, "kBlockSize must be at least kNumGPU warps");

  tvm::ffi::Tensor all_reduce(tvm::ffi::Tensor input, int shot) {
    using namespace host;
    RuntimeCheck(m_num_gpu == kNumGPU, "Mismatch handle");
    const bool use_2shot = (shot == 2);
    RuntimeCheck(shot == 1 || shot == 2, "Invalid shot count: ", shot);

    auto N = SymbolicSize{"batch_size"};
    auto M = SymbolicSize{"item_size"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, M})  // contiguous input
        .with_dtype<DType>(dtype)
        .with_device(device)
        .verify(input);

    const auto input_ptr = input.data_ptr();
    const auto buffer_ptr = pointer::offset(m_storage, m_payload_bytes());
    const auto num_items_int64 = M.unwrap() * N.unwrap();
    const auto num_items = static_cast<uint32_t>(num_items_int64);
    const auto needed_blocks = div_ceil(num_items, kItemsPerBlock);
    const auto num_blocks = std::min(needed_blocks, m_max_num_cta);
    const auto kernel = use_2shot ? two_shot_kernel : one_shot_kernel;
    tvm::ffi::Tensor output;
    if (m_is_graph_capturing) {
      output = use_2shot ? input : ffi::empty_like(input);
    } else {
      output = use_2shot ? ffi::from_blob_like(buffer_ptr, input) : input;
    }
    const auto params = AllReduceParams{
        .output = output.data_ptr(),
        .rank = m_rank,
        .num_items = num_items,
    };

    RuntimeCheck(M.unwrap() * sizeof(DType) % 16 == 0, "Input is not properly aligned to 16 bytes");
    RuntimeCheck(m_ctrl.has_value(), "Controller is not initialized");
    RuntimeCheck(static_cast<int64_t>(num_items) == num_items_int64, "Number of items exceeds 4G limit");

    const auto& ctrl = *m_ctrl;
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    auto launch = LaunchKernel{num_blocks, kBlockSize, stream};
    const auto check_capturing = [&] {
      if (!m_is_graph_capturing) return false;  // override to avoid cudaRT call overhead
      cudaStreamCaptureStatus status;
      RuntimeDeviceCheck(cudaStreamIsCapturing(stream, &status));
      return status == cudaStreamCaptureStatusActive;
    };
    if (check_capturing()) {
      // no-op if not really capturing, we're in a dummy run
      const auto data_ptr = m_allocate_graph_capture_input(input_ptr);
      /// NOTE: we assume when the graph is replayed, the data_ptr should be ready
      launch.enable_pdl(kUsePDL)(kernel, data_ptr, params, ctrl);
    } else {
      // 1.copy the input to the buffer
      const auto input_bytes = static_cast<int64_t>(sizeof(DType) * num_items);
      RuntimeCheck(input_bytes <= m_buffer_size_bytes, "Input is too large, num items: ", num_items);
      RuntimeDeviceCheck(cudaMemcpyAsync(buffer_ptr, input_ptr, input_bytes, cudaMemcpyDeviceToDevice, stream));
      // 2. launch the all reduce kernel
      launch.enable_pdl(kUsePDL)(kernel, m_get_data_ptr(), params, ctrl);
    }

    return output;
  }
};

template <typename DType, uint32_t kNumGPU, bool kUsePDL>
tvm::ffi::Tensor custom_all_reduce(CustomAllReduceRef obj, tvm::ffi::Tensor input, int shot) {
  using Impl = CustomAllReduceImpl<DType, kNumGPU, kUsePDL>;
  return static_cast<Impl&>(*obj.get()).all_reduce(input, shot);
}

template <typename T = void>
void register_custom_all_reduce() {
  namespace refl = tvm::ffi::reflection;
  using Class = CustomAllReduceBase;
  refl::ObjectDef<Class>()
      .def(refl::init<uint32_t, uint32_t, uint32_t, int64_t, int64_t>(), "__init__")
      .def("share_storage", &Class::share_storage)
      .def("share_graph_inputs", &Class::share_graph_inputs)
      .def("post_init", &Class::post_init)
      .def("register_inputs", &Class::register_inputs)
      .def("set_cuda_graph_capture", &Class::set_cuda_graph_capture)
      .def("free", &Class::free);
}

}  // namespace
