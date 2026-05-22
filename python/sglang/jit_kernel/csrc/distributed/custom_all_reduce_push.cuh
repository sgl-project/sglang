// Partially adapted from:
// https://github.com/flashinfer-ai/flashinfer/blob/v0.6.4/include/flashinfer/comm/trtllm_allreduce_fusion.cuh
// We simplify the lamport design and minimize the ring buffer count (from 3 -> 2)
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/common.cuh>
#include <sgl_kernel/distributed/custom_all_reduce.cuh>

#include <cstdint>
#include <cstring>

namespace {

using device::distributed::PushController;
using host::distributed::CustomAllReduceBase, host::distributed::CustomAllReduceRef;

struct AllReducePushData {
  void* __restrict__ buffer[device::distributed::kMaxNumGPU];
  const void* input;
  void* output;
  uint32_t rank;
  uint32_t num_items;
  uint32_t buffer_bytes;
  uint32_t epoch_bytes;
};

#define CUSTOM_AR_KERNEL __global__ __launch_bounds__(1024, 1)

template <typename T>
struct fp_trait {};

// TODO: support more dtypes
template <>
struct fp_trait<bf16_t> {
  using type = uint16_t;
  [[maybe_unused]]
  static constexpr uint16_t pos_zero = 0x0000u;
  [[maybe_unused]]
  static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<fp16_t> {
  using type = uint16_t;
  [[maybe_unused]]
  static constexpr uint16_t pos_zero = 0x0000u;
  [[maybe_unused]]
  static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<float> {
  using type = uint32_t;
  [[maybe_unused]]
  static constexpr uint32_t pos_zero = 0x00000000u;
  [[maybe_unused]]
  static constexpr uint32_t neg_zero = 0x80000000u;
};

template <typename DType>
SGL_DEVICE void clear_pos_zero(DType& val) {
  using Trait = fp_trait<DType>;
  const auto ptr = reinterpret_cast<typename Trait::type*>(&val);
  if (*ptr == Trait::pos_zero) *ptr = Trait::neg_zero;
}

template <typename DType>
SGL_DEVICE bool is_pos_zero(const DType& val) {
  using Trait = fp_trait<DType>;
  const auto ptr = reinterpret_cast<const typename Trait::type*>(&val);
  return *ptr == Trait::pos_zero;
}

template <typename DType>
SGL_DEVICE DType get_pos_zero() {
  using Trait = fp_trait<DType>;
  const auto value = Trait::pos_zero;
  return *reinterpret_cast<const DType*>(&value);
}

template <typename T>
SGL_DEVICE void ld_global_volatile_16B(T& x, const void* addr, int64_t offset) {
  static_assert(alignof(T) == 16 && sizeof(T) == 16);
  addr = device::pointer::offset<T>(addr, offset);
  uint4 val;
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  x = *reinterpret_cast<const T*>(&val);
}

template <typename T>
SGL_DEVICE void st_global_volatile_16B(const T& x, void* addr, int64_t offset) {
  static_assert(alignof(T) == 16 && sizeof(T) == 16);
  const uint4 val = *reinterpret_cast<const uint4*>(&x);
  addr = device::pointer::offset<T>(addr, offset);
  asm volatile(
      "st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}

template <typename DType, uint32_t kNumGPU>
SGL_DEVICE void push_impl(DType* (&push_buf)[kNumGPU], const void* data, uint32_t num_items) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  using Storage = AlignedVector<packed_t<DType>, kVecSize>;

  for (auto i = blockIdx.x;; i += gridDim.x) {
    const auto offset = i * blockDim.x + threadIdx.x;
    if (offset * kVecSize * 2 >= num_items) break;
    Storage vec;
    vec.load(data, offset);
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      clear_pos_zero(vec[j].x);
      clear_pos_zero(vec[j].y);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i) {
      st_global_volatile_16B(vec, push_buf[i], offset);
    }
  }
}

template <typename DType, uint32_t kNumGPU>
SGL_DEVICE void poll_impl(DType* (&poll_buf)[kNumGPU], void* data, uint32_t num_items) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  using Storage = AlignedVector<packed_t<DType>, kVecSize>;

  for (auto i = blockIdx.x;; i += gridDim.x) {
    const auto offset = i * blockDim.x + threadIdx.x;
    if (offset * kVecSize * 2 >= num_items) break;
    Storage storage[kNumGPU];

    while (true) {
      bool has_pos_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kNumGPU; ++i) {
        ld_global_volatile_16B(storage[i], poll_buf[i], offset);
#pragma unroll
        for (auto j = 0; j < kVecSize; ++j) {
          has_pos_zero |= is_pos_zero(storage[i][j].x);
          has_pos_zero |= is_pos_zero(storage[i][j].y);
        }
      }
      if (!has_pos_zero) break;
    }

    const Storage result = distributed::reduce_impl(storage);
    result.store(data, offset);

    Storage pos_zeros;
    pos_zeros.fill({get_pos_zero<DType>(), get_pos_zero<DType>()});
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i) {
      pos_zeros.store(poll_buf[i], offset);
    }
  }
}

template <typename DType, uint32_t kNumGPU, bool kUsePDL>
CUSTOM_AR_KERNEL void all_reduce_one_shot_push_kernel(
    const AllReducePushData __grid_constant__ params,  //
    const PushController __grid_constant__ ctrl) {
  using namespace device;

  const auto [buffer, input, output, rank, num_items, buffer_bytes, epoch_bytes] = params;

  PDLWaitPrimary<kUsePDL>();

  // Phase 1: Push data from input to all ranks' buffers
  const auto epoch_offset = ctrl.epoch() * epoch_bytes;
  DType* push_buf[kNumGPU];
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i) {
    push_buf[i] = static_cast<DType*>(pointer::offset(buffer[i], rank * buffer_bytes, epoch_offset));
  }
  push_impl(push_buf, input, num_items);

  PDLTriggerSecondary<kUsePDL>();

  // Phase 2: Poll local data
  DType* poll_buf[kNumGPU];
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i) {
    poll_buf[i] = static_cast<DType*>(pointer::offset(buffer[rank], i * buffer_bytes, epoch_offset));
  }
  poll_impl(poll_buf, output, num_items);
  ctrl.exit();
}

template <typename DType, uint32_t kNumGPU, bool kUsePDL>
struct CustomAllReducePush : public CustomAllReduceBase {
  static constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  static_assert(kNumGPU <= device::distributed::kMaxNumGPU, "kNumGPU exceeds the maximum supported GPUs");

  tvm::ffi::Tensor all_reduce(tvm::ffi::Tensor input, int shot) {
    using namespace host;
    const auto device = input.device();
    const auto input_ptr = input.data_ptr();
    const auto num_items_int64 = input.numel();
    const auto num_items = static_cast<uint32_t>(num_items_int64);
    const auto num_blocks = m_max_num_cta_push;  // must be constant to ensure correctness
    const auto num_threads = [&] {
      for (const auto t : {128u, 256u, 512u}) {
        if (t * num_blocks * 2 * kVecSize >= num_items) return t;
      }
      return 1024u;
    }();
    const auto output = input;
    AllReducePushData params;
    for (uint32_t i = 0; i < kNumGPU; ++i) {
      params.buffer[i] = get_push_buffer(m_peer_storage[i]);
    }
    params.input = input_ptr;
    params.output = input_ptr;
    params.rank = m_rank;
    params.num_items = num_items;
    params.buffer_bytes = m_push_buffer_bytes;
    params.epoch_bytes = kNumGPU * params.buffer_bytes;

    RuntimeCheck(input.IsContiguous(), "Input must be contiguous");
    RuntimeCheck(m_num_gpu == kNumGPU, "Number of GPUs mismatch");
    RuntimeCheck(device.device_type == kDLCUDA, "Only CUDA device is supported");
    RuntimeCheck(is_type<DType>(input.dtype()), "Input dtype mismatch");
    RuntimeCheck(std::bit_cast<intptr_t>(input_ptr) % 16 == 0, "Input pointer is not properly aligned");
    RuntimeCheck(m_push_ctrl.has_value(), "Controller is not initialized");
    RuntimeCheck(shot == 1, "Push all-reduce only supports 1-shot, got: ", shot);
    RuntimeCheck(static_cast<int64_t>(num_items) == num_items_int64, "Number of items exceeds 4G limit");

    const auto input_bytes = static_cast<int64_t>(sizeof(DType) * num_items_int64);
    RuntimeCheck(input_bytes <= m_push_buffer_bytes, "Input is too large, num items: ", num_items);

    const auto kernel = all_reduce_one_shot_push_kernel<DType, kNumGPU, kUsePDL>;
    LaunchKernel(num_blocks, num_threads, device)  //
        .enable_pdl(kUsePDL)(kernel, params, *m_push_ctrl);
    return output;
  }
};

template <typename DType, uint32_t kNumGPU, bool kUsePDL>
tvm::ffi::Tensor custom_all_reduce(CustomAllReduceRef obj, tvm::ffi::Tensor input, int shot) {
  using Impl = CustomAllReducePush<DType, kNumGPU, kUsePDL>;
  return static_cast<Impl&>(*obj.get()).all_reduce(input, shot);
}

}  // namespace
