// CUDA fast path for diffusion residual-gate elementwise updates.
//
// Implements:
//   out = residual + update * gate
//
// The production shapes come from LTX-2.3 HQ residual/gate updates.  This is
// intentionally narrow: contiguous residual/update/out tensors, with either a
// full contiguous gate or a row-broadcast [1, 1, D] gate.
//
// Developed with MIT HAN Lab Kernel Design Agents:
// https://github.com/mit-han-lab/kernel-design-agents

#pragma once

#include <sgl_kernel/tensor.h>  // For host dtype helpers and TensorView metadata
#include <sgl_kernel/utils.h>   // For RuntimeCheck and div_ceil

#include <sgl_kernel/type.cuh>   // For dtype_trait conversions
#include <sgl_kernel/utils.cuh>  // For LaunchKernel and CUDA dtype aliases
#include <sgl_kernel/vec.cuh>    // For device::AlignedVector

#include <cstdint>

namespace sglang_residual_gate_add {

namespace {

constexpr int kBlockSize = 256;
constexpr int kBcastRowsPerBlock = 4;
constexpr int kBcastColsVecPerBlock = 256;
constexpr int64_t kMaxGrid = 65535;

enum class GateMode : int { kFull = 0, kBcastRow = 1 };

inline const char* data_ptr(const tvm::ffi::TensorView& t) {
  return static_cast<const char*>(t.data_ptr()) + t.byte_offset();
}

inline char* mutable_data_ptr(const tvm::ffi::TensorView& t) {
  return static_cast<char*>(t.data_ptr()) + t.byte_offset();
}

inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

inline int64_t numel(const tvm::ffi::TensorView& t) {
  int64_t n = 1;
  for (int i = 0; i < t.ndim(); ++i) {
    n *= t.size(i);
  }
  return n;
}

inline int64_t grid_for(int64_t total) {
  int64_t grid = host::div_ceil(total, static_cast<int64_t>(kBlockSize));
  if (grid < 1) {
    grid = 1;
  }
  if (grid > kMaxGrid) {
    grid = kMaxGrid;
  }
  return grid;
}

inline bool is_dense_contiguous(const tvm::ffi::TensorView& t) {
  int64_t expected = 1;
  for (int i = t.ndim() - 1; i >= 0; --i) {
    if (t.size(i) == 1) {
      continue;
    }
    if (t.stride(i) != expected) {
      return false;
    }
    expected *= t.size(i);
  }
  return true;
}

template <typename T>
inline void check_dtype(const tvm::ffi::TensorView& t) {
  host::RuntimeCheck(host::is_type<T>(t.dtype()), "unexpected dtype for residual_gate_add");
}

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<fp16_t>(fp16_t v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<bf16_t>(bf16_t v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T residual_gate_value(T residual, T update, T gate) {
  const T product = dtype_trait<T>::from(to_float(update) * to_float(gate));
  return dtype_trait<T>::from(to_float(residual) + to_float(product));
}

template <typename T, int kVec>
__global__ void residual_gate_add_vec_kernel(
    const T* __restrict__ residual,
    const T* __restrict__ update,
    const T* __restrict__ gate,
    T* __restrict__ out,
    int64_t n_vec) {
  using Vec = device::AlignedVector<T, kVec>;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; v < n_vec; v += stride) {
    Vec r, u, g, o;
    r.load(residual, v);
    u.load(update, v);
    g.load(gate, v);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      o[i] = residual_gate_value(r[i], u[i], g[i]);
    }
    o.store(out, v);
  }
}

template <typename T, int kVec>
__global__ void residual_gate_add_bcast_row_tile_kernel(
    const T* __restrict__ residual,
    const T* __restrict__ update,
    const T* __restrict__ gate,
    T* __restrict__ out,
    int64_t rows,
    int64_t row_vec) {
  using Vec = device::AlignedVector<T, kVec>;
  const int64_t col_vec = static_cast<int64_t>(blockIdx.x) * kBcastColsVecPerBlock + threadIdx.x;
  if (col_vec >= row_vec) {
    return;
  }

  Vec g;
  g.load(gate, col_vec);

  // Grid-stride over row tiles so the launch stays valid even when the number
  // of row tiles exceeds the gridDim.y hardware limit.
  const int64_t row_tile_stride = static_cast<int64_t>(gridDim.y) * kBcastRowsPerBlock;
  for (int64_t row_base = static_cast<int64_t>(blockIdx.y) * kBcastRowsPerBlock; row_base < rows;
       row_base += row_tile_stride) {
#pragma unroll
    for (int row_offset = 0; row_offset < kBcastRowsPerBlock; ++row_offset) {
      const int64_t row = row_base + row_offset;
      if (row < rows) {
        const int64_t v = row * row_vec + col_vec;
        Vec r, u, o;
        r.load(residual, v);
        u.load(update, v);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          o[i] = residual_gate_value(r[i], u[i], g[i]);
        }
        o.store(out, v);
      }
    }
  }
}

template <typename T, GateMode kGate>
__global__ void residual_gate_add_scalar_kernel(
    const T* __restrict__ residual,
    const T* __restrict__ update,
    const T* __restrict__ gate,
    T* __restrict__ out,
    int64_t begin,
    int64_t total,
    int64_t D) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = begin + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
    const T gate_value = kGate == GateMode::kFull ? gate[i] : SGLANG_LDG(gate + (i % D));
    out[i] = residual_gate_value(residual[i], update[i], gate_value);
  }
}

template <typename T>
inline void launch_residual_gate_add(
    const tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& residual,
    const tvm::ffi::TensorView& update,
    const tvm::ffi::TensorView& gate,
    GateMode mode) {
  const int64_t total = numel(residual);
  if (total == 0) {
    return;
  }

  const int64_t D = residual.size(residual.ndim() - 1);
  const T* residual_ptr = reinterpret_cast<const T*>(data_ptr(residual));
  const T* update_ptr = reinterpret_cast<const T*>(data_ptr(update));
  const T* gate_ptr = reinterpret_cast<const T*>(data_ptr(gate));
  T* out_ptr = reinterpret_cast<T*>(mutable_data_ptr(out));
  constexpr int kVec = 16 / sizeof(T);

  const bool vec_ok = aligned16(residual_ptr) && aligned16(update_ptr) && aligned16(gate_ptr) && aligned16(out_ptr) &&
                      (D % kVec == 0) && (mode == GateMode::kBcastRow || total % kVec == 0);

  int64_t done = 0;
  if (vec_ok) {
    const int64_t n_vec = total / kVec;
    const int64_t row_vec = D / kVec;
    if (mode == GateMode::kFull) {
      host::LaunchKernel(static_cast<uint32_t>(grid_for(n_vec)), kBlockSize, out.device())(
          residual_gate_add_vec_kernel<T, kVec>, residual_ptr, update_ptr, gate_ptr, out_ptr, n_vec);
    } else {
      const int64_t rows = total / D;
      const int64_t col_blocks = host::div_ceil(row_vec, static_cast<int64_t>(kBcastColsVecPerBlock));
      const int64_t row_tiles = host::div_ceil(rows, static_cast<int64_t>(kBcastRowsPerBlock));
      const int64_t row_blocks = row_tiles > kMaxGrid ? kMaxGrid : row_tiles;
      host::LaunchKernel(
          dim3(static_cast<uint32_t>(col_blocks), static_cast<uint32_t>(row_blocks)),
          dim3(kBcastColsVecPerBlock),
          out.device())(
          residual_gate_add_bcast_row_tile_kernel<T, kVec>, residual_ptr, update_ptr, gate_ptr, out_ptr, rows, row_vec);
    }
    done = n_vec * kVec;
  }

  if (done < total) {
    if (mode == GateMode::kFull) {
      host::LaunchKernel(static_cast<uint32_t>(grid_for(total - done)), kBlockSize, out.device())(
          residual_gate_add_scalar_kernel<T, GateMode::kFull>,
          residual_ptr,
          update_ptr,
          gate_ptr,
          out_ptr,
          done,
          total,
          D);
    } else {
      host::LaunchKernel(static_cast<uint32_t>(grid_for(total - done)), kBlockSize, out.device())(
          residual_gate_add_scalar_kernel<T, GateMode::kBcastRow>,
          residual_ptr,
          update_ptr,
          gate_ptr,
          out_ptr,
          done,
          total,
          D);
    }
  }
}

template <typename T>
inline GateMode validate_residual_gate_add(
    const tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& residual,
    const tvm::ffi::TensorView& update,
    const tvm::ffi::TensorView& gate) {
  check_dtype<T>(out);
  check_dtype<T>(residual);
  check_dtype<T>(update);
  check_dtype<T>(gate);
  host::RuntimeCheck(residual.device().device_type == kDLCUDA, "residual must be CUDA");
  host::RuntimeCheck(update.device().device_type == kDLCUDA, "update must be CUDA");
  host::RuntimeCheck(gate.device().device_type == kDLCUDA, "gate must be CUDA");
  host::RuntimeCheck(out.device().device_type == kDLCUDA, "out must be CUDA");
  host::RuntimeCheck(
      residual.device().device_id == update.device().device_id &&
          residual.device().device_id == gate.device().device_id &&
          residual.device().device_id == out.device().device_id,
      "residual/update/gate/out must be on the same CUDA device");
  host::RuntimeCheck(residual.ndim() >= 2, "residual must be at least 2D");
  host::RuntimeCheck(update.ndim() == residual.ndim(), "update rank must match residual");
  host::RuntimeCheck(out.ndim() == residual.ndim(), "out rank must match residual");
  for (int i = 0; i < residual.ndim(); ++i) {
    host::RuntimeCheck(update.size(i) == residual.size(i), "update shape must match residual");
    host::RuntimeCheck(out.size(i) == residual.size(i), "out shape must match residual");
  }
  host::RuntimeCheck(is_dense_contiguous(residual), "residual must be contiguous");
  host::RuntimeCheck(is_dense_contiguous(update), "update must be contiguous");
  host::RuntimeCheck(is_dense_contiguous(out), "out must be contiguous");
  host::RuntimeCheck(is_dense_contiguous(gate), "gate must be contiguous");
  host::RuntimeCheck(data_ptr(out) != data_ptr(residual), "out must not alias residual");
  host::RuntimeCheck(data_ptr(out) != data_ptr(update), "out must not alias update");
  host::RuntimeCheck(data_ptr(out) != data_ptr(gate), "out must not alias gate");

  const int D_dim = residual.ndim() - 1;
  const int row_dim = residual.ndim() - 2;
  host::RuntimeCheck(gate.ndim() == residual.ndim(), "gate rank must match residual");
  host::RuntimeCheck(gate.size(D_dim) == residual.size(D_dim), "gate last dim must match residual");

  bool full_gate = true;
  for (int i = 0; i < residual.ndim(); ++i) {
    full_gate = full_gate && gate.size(i) == residual.size(i);
  }
  if (full_gate) {
    return GateMode::kFull;
  }

  host::RuntimeCheck(gate.size(row_dim) == 1, "broadcast gate row dim must be 1");
  for (int i = 0; i < D_dim; ++i) {
    host::RuntimeCheck(gate.size(i) == 1, "broadcast gate leading dims must be 1");
  }
  return GateMode::kBcastRow;
}

}  // namespace

template <typename T>
struct ResidualGateAddKernel {
  static void
  run(tvm::ffi::TensorView out, tvm::ffi::TensorView residual, tvm::ffi::TensorView update, tvm::ffi::TensorView gate) {
    const GateMode mode = validate_residual_gate_add<T>(out, residual, update, gate);
    launch_residual_gate_add<T>(out, residual, update, gate, mode);
  }
};

}  // namespace sglang_residual_gate_add
