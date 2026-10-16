// CUDA fast path for the two-branch diffusion residual-gate update:
//   out = residual + gate * (update_a + update_b)
//
// Blocks that run attention and MLP off one modulated input (FLUX.3's mode and
// single-stream blocks) sum the two branches and then apply the gate, which is
// two full passes over the activation. This folds them into one.
//
// The arithmetic mirrors `torch.addcmul(residual, gate, update_a + update_b)`
// exactly: the branch sum rounds to T (as an eager bf16 add does) and the gated
// product stays in fp32 (as addcmul does), so the result is bit-identical.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace sglang {

namespace residual_gate_add2 {

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kMaxGrid = 65535;
constexpr uintptr_t kAlignment = 16;

template <typename T>
SGL_DEVICE T residual_gate_value2(T residual, T update_a, T update_b, T gate) {
  // Round the branch sum to T first: an eager `a + b` on T does the same.
  const T sum = device::cast<T>(device::cast<fp32_t>(update_a) + device::cast<fp32_t>(update_b));
  // Keep the gated product in fp32, as addcmul does.
  return device::cast<T>(device::cast<fp32_t>(residual) + device::cast<fp32_t>(gate) * device::cast<fp32_t>(sum));
}

template <typename T, int kVec, bool kBroadcastGate>
__global__ void residual_gate_add2_vec_kernel(
    T* __restrict__ out,
    const T* __restrict__ residual,
    const T* __restrict__ update_a,
    const T* __restrict__ update_b,
    const T* __restrict__ gate,
    int64_t num_vectors,
    int64_t row_vectors) {
  using Vec = device::AlignedVector<T, kVec>;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t vector = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; vector < num_vectors;
       vector += stride) {
    Vec residual_vec, a_vec, b_vec, gate_vec, out_vec;
    residual_vec.load(residual, vector);
    a_vec.load(update_a, vector);
    b_vec.load(update_b, vector);
    gate_vec.load(gate, kBroadcastGate ? vector % row_vectors : vector);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      out_vec[i] = residual_gate_value2(residual_vec[i], a_vec[i], b_vec[i], gate_vec[i]);
    }
    out_vec.store(out, vector);
  }
}

template <typename T, bool kBroadcastGate>
__global__ void residual_gate_add2_scalar_kernel(
    T* __restrict__ out,
    const T* __restrict__ residual,
    const T* __restrict__ update_a,
    const T* __restrict__ update_b,
    const T* __restrict__ gate,
    int64_t numel,
    int64_t hidden_size) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < numel; index += stride) {
    const T gate_value = kBroadcastGate ? SGLANG_LDG(gate + index % hidden_size) : gate[index];
    out[index] = residual_gate_value2(residual[index], update_a[index], update_b[index], gate_value);
  }
}

}  // namespace

/**
 * \brief Validate and launch a two-branch residual-gate update.
 *
 * Python flattens the tensors before dispatch. A broadcast gate contains one
 * hidden-size row; a full gate has the same number of elements as the inputs.
 */
template <typename T>
struct ResidualGateAdd2Kernel {
  static_assert(std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t> || std::is_same_v<T, fp32_t>);

  static void
  run(tvm::ffi::TensorView out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView update_a,
      tvm::ffi::TensorView update_b,
      tvm::ffi::TensorView gate,
      int64_t hidden_size,
      bool broadcast_gate) {
    using namespace host;

    auto N = SymbolicSize{"numel"};
    auto G = SymbolicSize{"gate_numel"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N}).with_dtype<T>().with_device(device).verify(out).verify(residual).verify(update_a).verify(
        update_b);
    TensorMatcher({G}).with_dtype<T>().with_device(device).verify(gate);

    const int64_t numel = N.unwrap();
    CHECK_HOST(hidden_size > 0 && numel % hidden_size == 0) << "hidden size must be positive and divide the input size";
    CHECK_HOST(G.unwrap() == (broadcast_gate ? hidden_size : numel)) << "gate size does not match its mode";
    if (numel == 0) {
      return;
    }

    auto* out_ptr = static_cast<T*>(out.data_ptr());
    const auto* residual_ptr = static_cast<const T*>(residual.data_ptr());
    const auto* a_ptr = static_cast<const T*>(update_a.data_ptr());
    const auto* b_ptr = static_cast<const T*>(update_b.data_ptr());
    const auto* gate_ptr = static_cast<const T*>(gate.data_ptr());
    CHECK_HOST(out_ptr != residual_ptr && out_ptr != a_ptr && out_ptr != b_ptr && out_ptr != gate_ptr)
        << "output must not alias an input";

    constexpr int kVec = kAlignment / sizeof(T);
    const bool aligned = reinterpret_cast<uintptr_t>(out_ptr) % kAlignment == 0 &&
                         reinterpret_cast<uintptr_t>(residual_ptr) % kAlignment == 0 &&
                         reinterpret_cast<uintptr_t>(a_ptr) % kAlignment == 0 &&
                         reinterpret_cast<uintptr_t>(b_ptr) % kAlignment == 0 &&
                         reinterpret_cast<uintptr_t>(gate_ptr) % kAlignment == 0;
    if (aligned && hidden_size % kVec == 0) {
      const int64_t num_vectors = numel / kVec;
      const int64_t row_vectors = hidden_size / kVec;
      const auto blocks =
          static_cast<uint32_t>(std::min<int64_t>(div_ceil(num_vectors, static_cast<int64_t>(kBlockSize)), kMaxGrid));
      if (broadcast_gate) {
        LaunchKernel(blocks, kBlockSize, device.unwrap())(
            residual_gate_add2_vec_kernel<T, kVec, true>,
            out_ptr,
            residual_ptr,
            a_ptr,
            b_ptr,
            gate_ptr,
            num_vectors,
            row_vectors);
      } else {
        LaunchKernel(blocks, kBlockSize, device.unwrap())(
            residual_gate_add2_vec_kernel<T, kVec, false>,
            out_ptr,
            residual_ptr,
            a_ptr,
            b_ptr,
            gate_ptr,
            num_vectors,
            row_vectors);
      }
      return;
    }

    const auto blocks =
        static_cast<uint32_t>(std::min<int64_t>(div_ceil(numel, static_cast<int64_t>(kBlockSize)), kMaxGrid));
    if (broadcast_gate) {
      LaunchKernel(blocks, kBlockSize, device.unwrap())(
          residual_gate_add2_scalar_kernel<T, true>, out_ptr, residual_ptr, a_ptr, b_ptr, gate_ptr, numel, hidden_size);
    } else {
      LaunchKernel(blocks, kBlockSize, device.unwrap())(
          residual_gate_add2_scalar_kernel<T, false>,
          out_ptr,
          residual_ptr,
          a_ptr,
          b_ptr,
          gate_ptr,
          numel,
          hidden_size);
    }
  }
};

}  // namespace residual_gate_add2

}  // namespace sglang
