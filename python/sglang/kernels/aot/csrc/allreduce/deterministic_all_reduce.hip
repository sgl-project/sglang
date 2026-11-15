// Deterministic All-Reduce for ROCm/HIP
//
// Every element is summed over the ranks in the fixed order 0..world_size-1 (no
// atomics), so each rank gets the same bits and an element's result does not depend
// on the input size. Two kernels do this:
//   - cross_device_reduce_1stage: each GPU reads all data from all GPUs and reduces
//     it locally. Best for small inputs (decode).
//   - cross_device_reduce_2stage<..., kFixedOrder=true>: reduce-scatter + all-gather
//     whose reduce-scatter keeps the same rank order. It gives the same bits as the
//     1-stage kernel and reads less data per GPU, so it is faster for large inputs.
// The size thresholds below are those CustomAllreduce::allreduce uses.

#include <ATen/hip/Exceptions.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/all.h>

#include "custom_all_reduce_hip.cuh"

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

// Helper function for weak contiguity check
bool _is_weak_contiguous_det(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() == t.numel() * t.element_size());
}

namespace {

template <typename T>
void launch_deterministic(
    sglang::CustomAllreduce* fa, sglang::RankData* ptrs, T* out, int size, bool allow_two_stage, hipStream_t stream) {
  using P = typename sglang::packed_t<T>::P;
  if (size % P::size != 0) {
    throw std::runtime_error("size must be multiple of " + std::to_string(P::size));
  }
  size /= P::size;
  const size_t bytes = static_cast<size_t>(size) * sizeof(P);
  const int ws = fa->world_size_;
  // 6 ranks keep the 1-stage kernel: the 2-stage path was only measured at 4 and 8.
  const bool two_stage = allow_two_stage && fa->full_nvlink_ &&
                         ((ws == 4 && bytes >= 512 * 1024) || (ws == 8 && bytes >= 256 * 1024));
  const int threads = 512;
  // 32 blocks measured fastest for the 2-stage kernel on MI350X; the 1-stage kernel keeps 16.
  const int blocks = std::min(two_stage ? 32 : 16, (size + threads - 1) / threads);
#define DET_AR_CASE(NG)                                                                                       \
  case NG:                                                                                                    \
    if (two_stage) {                                                                                          \
      hipLaunchKernelGGL(                                                                                     \
          (sglang::cross_device_reduce_2stage<T, NG, true>),                                                  \
          dim3(blocks),                                                                                       \
          dim3(threads),                                                                                      \
          0,                                                                                                  \
          stream,                                                                                             \
          ptrs,                                                                                               \
          fa->sg_,                                                                                            \
          fa->self_sg_,                                                                                       \
          out,                                                                                                \
          fa->rank_,                                                                                          \
          size);                                                                                              \
    } else {                                                                                                  \
      hipLaunchKernelGGL(                                                                                     \
          (sglang::cross_device_reduce_1stage<T, NG>),                                                        \
          dim3(blocks),                                                                                       \
          dim3(threads),                                                                                      \
          0,                                                                                                  \
          stream,                                                                                             \
          ptrs,                                                                                               \
          fa->sg_,                                                                                            \
          fa->self_sg_,                                                                                       \
          out,                                                                                                \
          fa->rank_,                                                                                          \
          size);                                                                                              \
    }                                                                                                         \
    break;
  switch (ws) {
    case 2:
      hipLaunchKernelGGL(
          (sglang::cross_device_reduce_1stage<T, 2>),
          dim3(blocks),
          dim3(threads),
          0,
          stream,
          ptrs,
          fa->sg_,
          fa->self_sg_,
          out,
          fa->rank_,
          size);
      break;
    DET_AR_CASE(4)
    DET_AR_CASE(6)
    DET_AR_CASE(8)
    default:
      throw std::runtime_error("world_size must be in (2,4,6,8)");
  }
#undef DET_AR_CASE
}

void deterministic_all_reduce_impl(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, bool allow_two_stage) {
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous_det(out));
  TORCH_CHECK(_is_weak_contiguous_det(inp));

  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);

  sglang::RankData* ptrs;
  hipStreamCaptureStatus status;
  AT_CUDA_CHECK(hipStreamIsCapturing(stream, &status));
  if (status == hipStreamCaptureStatusActive) {
    ptrs = fa->d_rank_data_base_ + fa->graph_unreg_buffers_.size();
    fa->graph_unreg_buffers_.push_back(inp.data_ptr());
  } else {
    auto it = fa->buffers_.find(inp.data_ptr());
    if (it == fa->buffers_.end()) {
      throw std::runtime_error("buffer not registered!");
    }
    ptrs = it->second;
  }

  const int size = out.numel();
  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      launch_deterministic<float>(fa, ptrs, reinterpret_cast<float*>(out.data_ptr()), size, allow_two_stage, stream);
      break;
    case at::ScalarType::Half:
      launch_deterministic<half>(fa, ptrs, reinterpret_cast<half*>(out.data_ptr()), size, allow_two_stage, stream);
      break;
#if (__HIP_ARCH__ >= 800 || !defined(__HIP_ARCH__))
    case at::ScalarType::BFloat16:
      launch_deterministic<nv_bfloat16>(fa, ptrs, reinterpret_cast<nv_bfloat16*>(out.data_ptr()), size, allow_two_stage, stream);
      break;
#endif
    default:
      throw std::runtime_error("deterministic allreduce only supports float32, float16 and bfloat16");
  }
}

}  // namespace

// Deterministic all-reduce for registered buffers (ROCm). Inputs here can exceed the
// IPC meta buffer, which the 2-stage kernel uses for 1/world_size of the input, so
// they always take the 1-stage kernel.
void deterministic_all_reduce_reg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out) {
  deterministic_all_reduce_impl(_fa, inp, out, /*allow_two_stage=*/false);
}

// Deterministic all-reduce for unregistered buffers (ROCm): the input is copied into
// reg_buffer, so it fits the meta buffer and may take the 2-stage kernel.
void deterministic_all_reduce_unreg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& reg_buffer, torch::Tensor& out) {
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();

  auto input_size = inp.numel() * inp.element_size();
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(input_size <= reg_buffer.numel() * reg_buffer.element_size(),
              "registered buffer is too small to contain the input");
  AT_CUDA_CHECK(hipMemcpyAsync(reg_buffer.data_ptr(), inp.data_ptr(),
                                input_size, hipMemcpyDeviceToDevice, stream));
  deterministic_all_reduce_impl(_fa, reg_buffer, out, /*allow_two_stage=*/true);
}
