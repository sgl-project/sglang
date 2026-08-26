// CUDA fast path for bit-exact diffusion residual-gate updates:
//   out = residual + update * gate

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

namespace residual_gate_add {

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kBroadcastRowsPerBlock = 4;
constexpr uint32_t kBroadcastColsPerBlock = 256;
constexpr uint32_t kMaxGrid = 65535;
constexpr uintptr_t kAlignment = 16;

enum class GateMode : int { kFull, kBroadcastRow };

template <typename T>
SGL_DEVICE T residual_gate_value(T residual, T update, T gate) {
  const T product = device::cast<T>(device::cast<fp32_t>(update) * device::cast<fp32_t>(gate));
  return device::cast<T>(device::cast<fp32_t>(residual) + device::cast<fp32_t>(product));
}

template <typename T, int kVec>
__global__ void residual_gate_add_vec_kernel(
    T* __restrict__ out,
    const T* __restrict__ residual,
    const T* __restrict__ update,
    const T* __restrict__ gate,
    int64_t num_vectors) {
  using Vec = device::AlignedVector<T, kVec>;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t vector = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; vector < num_vectors;
       vector += stride) {
    Vec residual_vec, update_vec, gate_vec, out_vec;
    residual_vec.load(residual, vector);
    update_vec.load(update, vector);
    gate_vec.load(gate, vector);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      out_vec[i] = residual_gate_value(residual_vec[i], update_vec[i], gate_vec[i]);
    }
    out_vec.store(out, vector);
  }
}

template <typename T, int kVec>
__global__ void residual_gate_add_broadcast_kernel(
    T* __restrict__ out,
    const T* __restrict__ residual,
    const T* __restrict__ update,
    const T* __restrict__ gate,
    int64_t rows,
    int64_t row_vectors) {
  using Vec = device::AlignedVector<T, kVec>;
  const int64_t column = static_cast<int64_t>(blockIdx.x) * kBroadcastColsPerBlock + threadIdx.x;
  if (column >= row_vectors) {
    return;
  }

  Vec gate_vec;
  gate_vec.load(gate, column);
  const int64_t row_stride = static_cast<int64_t>(gridDim.y) * kBroadcastRowsPerBlock;
  for (int64_t row_base = static_cast<int64_t>(blockIdx.y) * kBroadcastRowsPerBlock; row_base < rows;
       row_base += row_stride) {
#pragma unroll
    for (uint32_t row_offset = 0; row_offset < kBroadcastRowsPerBlock; ++row_offset) {
      const int64_t row = row_base + row_offset;
      if (row >= rows) {
        continue;
      }
      const int64_t vector = row * row_vectors + column;
      Vec residual_vec, update_vec, out_vec;
      residual_vec.load(residual, vector);
      update_vec.load(update, vector);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        out_vec[i] = residual_gate_value(residual_vec[i], update_vec[i], gate_vec[i]);
      }
      out_vec.store(out, vector);
    }
  }
}

template <typename T, GateMode kGateMode>
__global__ void residual_gate_add_scalar_kernel(
    T* __restrict__ out,
    const T* __restrict__ residual,
    const T* __restrict__ update,
    const T* __restrict__ gate,
    int64_t numel,
    int64_t hidden_size) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < numel; index += stride) {
    const T gate_value = kGateMode == GateMode::kFull ? gate[index] : SGLANG_LDG(gate + index % hidden_size);
    out[index] = residual_gate_value(residual[index], update[index], gate_value);
  }
}

}  // namespace

/**
 * \brief Validate and launch a bit-exact residual-gate update.
 *
 * Python flattens the tensors before dispatch. A broadcast gate contains one
 * hidden-size row; a full gate has the same number of elements as the inputs.
 */
template <typename T>
struct ResidualGateAddKernel {
  static_assert(std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t> || std::is_same_v<T, fp32_t>);

  static void
  run(tvm::ffi::TensorView out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView update,
      tvm::ffi::TensorView gate,
      int64_t hidden_size,
      bool broadcast_gate) {
    using namespace host;

    auto N = SymbolicSize{"numel"};
    auto G = SymbolicSize{"gate_numel"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N}).with_dtype<T>().with_device(device).verify(out).verify(residual).verify(update);
    TensorMatcher({G}).with_dtype<T>().with_device(device).verify(gate);

    const int64_t numel = N.unwrap();
    CHECK_HOST(hidden_size > 0 && numel % hidden_size == 0) << "hidden size must be positive and divide the input size";
    CHECK_HOST(G.unwrap() == (broadcast_gate ? hidden_size : numel)) << "gate size does not match its mode";
    if (numel == 0) {
      return;
    }

    auto* out_ptr = static_cast<T*>(out.data_ptr());
    const auto* residual_ptr = static_cast<const T*>(residual.data_ptr());
    const auto* update_ptr = static_cast<const T*>(update.data_ptr());
    const auto* gate_ptr = static_cast<const T*>(gate.data_ptr());
    CHECK_HOST(out_ptr != residual_ptr && out_ptr != update_ptr && out_ptr != gate_ptr)
        << "output must not alias an input";

    constexpr int kVec = kAlignment / sizeof(T);
    const bool aligned = reinterpret_cast<uintptr_t>(out_ptr) % kAlignment == 0 &&
                         reinterpret_cast<uintptr_t>(residual_ptr) % kAlignment == 0 &&
                         reinterpret_cast<uintptr_t>(update_ptr) % kAlignment == 0 &&
                         reinterpret_cast<uintptr_t>(gate_ptr) % kAlignment == 0;
    const bool vectorized = aligned && hidden_size % kVec == 0;
    if (vectorized) {
      const int64_t num_vectors = numel / kVec;
      if (!broadcast_gate) {
        const auto blocks =
            static_cast<uint32_t>(std::min<int64_t>(div_ceil(num_vectors, static_cast<int64_t>(kBlockSize)), kMaxGrid));
        LaunchKernel(blocks, kBlockSize, device.unwrap())(
            residual_gate_add_vec_kernel<T, kVec>, out_ptr, residual_ptr, update_ptr, gate_ptr, num_vectors);
        return;
      }

      const int64_t rows = numel / hidden_size;
      const int64_t row_vectors = hidden_size / kVec;
      const auto column_blocks =
          static_cast<uint32_t>(div_ceil(row_vectors, static_cast<int64_t>(kBroadcastColsPerBlock)));
      const int64_t row_tiles = div_ceil(rows, static_cast<int64_t>(kBroadcastRowsPerBlock));
      const auto row_blocks = static_cast<uint32_t>(std::min<int64_t>(row_tiles, kMaxGrid));
      LaunchKernel(dim3(column_blocks, row_blocks), kBroadcastColsPerBlock, device.unwrap())(
          residual_gate_add_broadcast_kernel<T, kVec>, out_ptr, residual_ptr, update_ptr, gate_ptr, rows, row_vectors);
      return;
    }

    const auto blocks =
        static_cast<uint32_t>(std::min<int64_t>(div_ceil(numel, static_cast<int64_t>(kBlockSize)), kMaxGrid));
    if (broadcast_gate) {
      LaunchKernel(blocks, kBlockSize, device.unwrap())(
          residual_gate_add_scalar_kernel<T, GateMode::kBroadcastRow>,
          out_ptr,
          residual_ptr,
          update_ptr,
          gate_ptr,
          numel,
          hidden_size);
    } else {
      LaunchKernel(blocks, kBlockSize, device.unwrap())(
          residual_gate_add_scalar_kernel<T, GateMode::kFull>,
          out_ptr,
          residual_ptr,
          update_ptr,
          gate_ptr,
          numel,
          hidden_size);
    }
  }
};

}  // namespace residual_gate_add

}  // namespace sglang
