// CUDA fast path for the Ulysses sequence-parallel output head merge.
//
//   usp_merge_heads:
//     x [W, S, B, h_local, D] (contiguous, the output all-to-all result)
//       -> out [B, S, W, h_local, D] (contiguous)
//     Replaces `x.permute(2, 1, 0, 3, 4).contiguous()` on the head_dim=2
//     output path of `_usp_output_all_to_all`.
//
// A pure copy (no arithmetic), so it is bit-exact with the eager permute by
// construction. It exists because ATen's generic permute-copy reaches well
// under half of HBM bandwidth on the packed-DiT shapes, while a single pass
// with coalesced vectorized stores runs near roofline.

#pragma once

#include <sgl_kernel/tensor.h>  // For host dtype helpers and TensorView metadata
#include <sgl_kernel/utils.h>   // For RuntimeCheck and div_ceil

#include <sgl_kernel/type.cuh>   // For CUDA dtype aliases
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For device::AlignedVector

#include <cstdint>

namespace sglang_usp_relayout {

namespace {

constexpr int kBlockSize = 256;
constexpr int64_t kMaxGrid = 65535;

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
  host::RuntimeCheck(host::is_type<T>(t.dtype()), "unexpected dtype for usp_merge_heads tensor");
}

// out[b, s, w, h, c] = x[w, s, b, h, c]
template <typename T, int kVec>
__global__ void usp_merge_heads_vec_kernel(
    T* __restrict__ out,
    const T* __restrict__ x,
    int64_t n_vec,
    int64_t d_vec,  // D / kVec
    int64_t h_local,
    int64_t batch,
    int64_t seq,
    int64_t world) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
    int64_t rest = i;
    const int64_t c_vec = rest % d_vec;
    rest /= d_vec;
    const int64_t h = rest % h_local;
    rest /= h_local;
    const int64_t w = rest % world;
    rest /= world;
    const int64_t s = rest % seq;
    const int64_t b = rest / seq;

    const int64_t src_vec = ((((w * seq + s) * batch + b) * h_local) + h) * d_vec + c_vec;
    device::AlignedVector<T, kVec> val;
    val.load(x, src_vec);
    val.store(out, i);
  }
}

template <typename T>
__global__ void usp_merge_heads_scalar_kernel(
    T* __restrict__ out,
    const T* __restrict__ x,
    int64_t total,
    int64_t head_dim,
    int64_t h_local,
    int64_t batch,
    int64_t seq,
    int64_t world) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
    int64_t rest = i;
    const int64_t c = rest % head_dim;
    rest /= head_dim;
    const int64_t h = rest % h_local;
    rest /= h_local;
    const int64_t w = rest % world;
    rest /= world;
    const int64_t s = rest % seq;
    const int64_t b = rest / seq;

    out[i] = x[((((w * seq + s) * batch + b) * h_local) + h) * head_dim + c];
  }
}

}  // namespace

template <typename T>
struct UspMergeHeadsKernel {
  static void run(tvm::ffi::TensorView out, tvm::ffi::TensorView x) {
    check_dtype<T>(out);
    check_dtype<T>(x);
    host::RuntimeCheck(x.ndim() == 5, "x must be [W, S, B, h_local, D]");
    host::RuntimeCheck(out.ndim() == 5, "out must be [B, S, W, h_local, D]");
    for (auto* t : {&x, &out}) {
      host::RuntimeCheck(t->device().device_type == kDLCUDA, "usp_merge_heads tensors must be CUDA");
      host::RuntimeCheck(is_dense_contiguous(*t), "usp_merge_heads tensors must be contiguous");
    }
    const int64_t world = x.size(0);
    const int64_t seq = x.size(1);
    const int64_t batch = x.size(2);
    const int64_t h_local = x.size(3);
    const int64_t head_dim = x.size(4);
    host::RuntimeCheck(
        out.size(0) == batch && out.size(1) == seq && out.size(2) == world && out.size(3) == h_local &&
            out.size(4) == head_dim,
        "out must be the [B, S, W, h_local, D] permutation of x");

    const int64_t total = numel(x);
    if (total == 0) {
      return;
    }

    T* out_ptr = reinterpret_cast<T*>(mutable_data_ptr(out));
    const T* x_ptr = reinterpret_cast<const T*>(data_ptr(x));

    constexpr int kVec = 16 / sizeof(T);
    const bool vec_ok = (head_dim % kVec == 0) && aligned16(out_ptr) && aligned16(x_ptr);
    if (vec_ok) {
      const int64_t n_vec = total / kVec;
      host::LaunchKernel(static_cast<uint32_t>(grid_for(n_vec)), kBlockSize, out.device())(
          usp_merge_heads_vec_kernel<T, kVec>, out_ptr, x_ptr, n_vec, head_dim / kVec, h_local, batch, seq, world);
    } else {
      host::LaunchKernel(static_cast<uint32_t>(grid_for(total)), kBlockSize, out.device())(
          usp_merge_heads_scalar_kernel<T>, out_ptr, x_ptr, total, head_dim, h_local, batch, seq, world);
    }
  }
};

}  // namespace sglang_usp_relayout
