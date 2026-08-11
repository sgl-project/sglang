// Deterministic All-Reduce for ROCm/HIP
//
// This is a wrapper that forces the use of the existing 1-stage all-reduce kernel
// (cross_device_reduce_1stage) which is inherently deterministic due to fixed
// accumulation ordering (no atomics, no race conditions).
//
// How the 1-stage kernel works:
//   - Each GPU reads ALL data from ALL other GPUs via direct memory access
//   - Each GPU reduces the data locally in a fixed order
//   - Result: every GPU has the complete reduced output
//
// This is NOT a reduce-scatter + all-gather (RS+AG) approach.
// The 2-stage kernel (cross_device_reduce_2stage) implements RS+AG but may have
// non-deterministic behavior, so we explicitly avoid it here.

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

// Deterministic all-reduce for registered buffers (ROCm)
// Uses the 1-stage kernel which is deterministic (fixed ordering)
void deterministic_all_reduce_reg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out) {
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous_det(out));
  TORCH_CHECK(_is_weak_contiguous_det(inp));

  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);

  // For ROCm, manually call the 1-stage kernel to ensure deterministic ordering
  // Get rank data pointer
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

  int size = out.numel();
  int threads = 512;

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      using T = float;
      using P = typename sglang::packed_t<T>::P;
      auto d = P::size;
      if (size % d != 0) {
        throw std::runtime_error("size must be multiple of " + std::to_string(d));
      }
      size /= d;
      int blocks = std::min(16, (size + threads - 1) / threads);
      // Always use 1-stage kernel for determinism
      switch (fa->world_size_) {
        case 2:
          hipLaunchKernelGGL((sglang::cross_device_reduce_1stage<T, 2>), dim3(blocks), dim3(threads), 0, stream,
                              ptrs, fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size);
          break;
        case 4:
          hipLaunchKernelGGL((sglang::cross_device_reduce_1stage<T, 4>), dim3(blocks), dim3(threads), 0, stream,
                              ptrs, fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size);
          break;
        case 6:
          hipLaunchKernelGGL((sglang::cross_device_reduce_1stage<T, 6>), dim3(blocks), dim3(threads), 0, stream,
                              ptrs, fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size);
          break;
        case 8:
          hipLaunchKernelGGL((sglang::cross_device_reduce_1stage<T, 8>), dim3(blocks), dim3(threads), 0, stream,
                              ptrs, fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size);
          break;
        default:
          throw std::runtime_error("world_size must be in (2,4,6,8)");
      }
      break;
    }
    case at::ScalarType::Half: {
      using T = half;
      using P = typename sglang::packed_t<T>::P;
      auto d = P::size;
      if (size % d != 0) {
        throw std::runtime_error("size must be multiple of " + std::to_string(d));
      }
      size /= d;
      int blocks = std::min(16, (size + threads - 1) / threads);
      switch (fa->world_size_) {
        case 2:
          hipLaunchKernelGGL((sglang::cross_device_reduce_1stage<T, 2>), dim3(blocks), dim3(threads), 0, stream,
                              ptrs, fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size);
          break;
        case 4:
          hipLaunchKernelGGL((sglang::cross_device_reduce_1stage<T, 4>), dim3(blocks), dim3(threads), 0, stream,
                              ptrs, fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size);
          break;
        case 6:
          hipLaunchKernelGGL((sglang::cross_device_reduce_1stage<T, 6>), dim3(blocks), dim3(threads), 0, stream,
                              ptrs, fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size);
          break;
        case 8:
          hipLaunchKernelGGL((sglang::cross_device_reduce_1stage<T, 8>), dim3(blocks), dim3(threads), 0, stream,
                              ptrs, fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size);
          break;
        default:
          throw std::runtime_error("world_size must be in (2,4,6,8)");
      }
      break;
    }
#if (__HIP_ARCH__ >= 800 || !defined(__HIP_ARCH__))
    case at::ScalarType::BFloat16: {
      using T = nv_bfloat16;
      using P = typename sglang::packed_t<T>::P;
      auto d = P::size;
      if (size % d != 0) {
        throw std::runtime_error("size must be multiple of " + std::to_string(d));
      }
      size /= d;
      int blocks = std::min(16, (size + threads - 1) / threads);
      switch (fa->world_size_) {
        case 2:
          hipLaunchKernelGGL((sglang::cross_device_reduce_1stage<T, 2>), dim3(blocks), dim3(threads), 0, stream,
                              ptrs, fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size);
          break;
        case 4:
          hipLaunchKernelGGL((sglang::cross_device_reduce_1stage<T, 4>), dim3(blocks), dim3(threads), 0, stream,
                              ptrs, fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size);
          break;
        case 6:
          hipLaunchKernelGGL((sglang::cross_device_reduce_1stage<T, 6>), dim3(blocks), dim3(threads), 0, stream,
                              ptrs, fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size);
          break;
        case 8:
          hipLaunchKernelGGL((sglang::cross_device_reduce_1stage<T, 8>), dim3(blocks), dim3(threads), 0, stream,
                              ptrs, fa->sg_, fa->self_sg_, reinterpret_cast<T*>(out.data_ptr()), fa->rank_, size);
          break;
        default:
          throw std::runtime_error("world_size must be in (2,4,6,8)");
      }
      break;
    }
#endif
    default:
      throw std::runtime_error("deterministic allreduce only supports float32, float16 and bfloat16");
  }
}

// Deterministic all-reduce for unregistered buffers (ROCm)
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
  deterministic_all_reduce_reg(_fa, reg_buffer, out);
}
