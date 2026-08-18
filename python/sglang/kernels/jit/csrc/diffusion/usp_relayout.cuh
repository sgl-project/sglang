// CUDA fast path for the Ulysses sequence-parallel output-head merge:
//   [W, S, B, H, D] -> [B, S, W, H, D]

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

namespace usp_relayout {

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kMaxGrid = 65535;
constexpr uintptr_t kAlignment = 16;

// out[b, s, w, h, d] = x[w, s, b, h, d]
template <typename T, int kVec>
__global__ void usp_merge_heads_vec_kernel(
    T* __restrict__ out,
    const T* __restrict__ x,
    int64_t num_vectors,
    int64_t head_vectors,
    int64_t local_heads,
    int64_t batch,
    int64_t sequence_length,
    int64_t world_size) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < num_vectors;
       index += stride) {
    int64_t rest = index;
    const int64_t head_offset = rest % head_vectors;
    rest /= head_vectors;
    const int64_t head = rest % local_heads;
    rest /= local_heads;
    const int64_t rank = rest % world_size;
    rest /= world_size;
    const int64_t sequence = rest % sequence_length;
    const int64_t batch_index = rest / sequence_length;

    const int64_t source =
        ((((rank * sequence_length + sequence) * batch + batch_index) * local_heads) + head) * head_vectors +
        head_offset;
    device::AlignedVector<T, kVec> value;
    value.load(x, source);
    value.store(out, index);
  }
}

template <typename T>
__global__ void usp_merge_heads_scalar_kernel(
    T* __restrict__ out,
    const T* __restrict__ x,
    int64_t numel,
    int64_t head_dim,
    int64_t local_heads,
    int64_t batch,
    int64_t sequence_length,
    int64_t world_size) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < numel; index += stride) {
    int64_t rest = index;
    const int64_t head_offset = rest % head_dim;
    rest /= head_dim;
    const int64_t head = rest % local_heads;
    rest /= local_heads;
    const int64_t rank = rest % world_size;
    rest /= world_size;
    const int64_t sequence = rest % sequence_length;
    const int64_t batch_index = rest / sequence_length;

    out[index] =
        x[((((rank * sequence_length + sequence) * batch + batch_index) * local_heads) + head) * head_dim +
          head_offset];
  }
}

}  // namespace

/** \brief Merge Ulysses output heads with a bit-exact layout copy. */
template <typename T>
struct UspMergeHeadsKernel {
  static_assert(std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t> || std::is_same_v<T, fp32_t>);

  static void run(tvm::ffi::TensorView out, tvm::ffi::TensorView x) {
    using namespace host;

    auto W = SymbolicSize{"world_size"};
    auto S = SymbolicSize{"sequence_length"};
    auto B = SymbolicSize{"batch"};
    auto H = SymbolicSize{"local_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({W, S, B, H, D}).with_dtype<T>().with_device(device).verify(x);
    TensorMatcher({B, S, W, H, D}).with_dtype<T>().with_device(device).verify(out);

    const int64_t world_size = W.unwrap();
    const int64_t sequence_length = S.unwrap();
    const int64_t batch = B.unwrap();
    const int64_t local_heads = H.unwrap();
    const int64_t head_dim = D.unwrap();
    const int64_t numel = world_size * sequence_length * batch * local_heads * head_dim;
    if (numel == 0) {
      return;
    }

    auto* out_ptr = static_cast<T*>(out.data_ptr());
    const auto* x_ptr = static_cast<const T*>(x.data_ptr());
    CHECK_HOST(out_ptr != x_ptr) << "output must not alias the input";
    const auto launch = [&](auto kernel, int64_t work_items, auto... args) {
      const auto blocks =
          static_cast<uint32_t>(std::min<int64_t>(div_ceil(work_items, static_cast<int64_t>(kBlockSize)), kMaxGrid));
      LaunchKernel(blocks, kBlockSize, device.unwrap())(kernel, out_ptr, x_ptr, work_items, args...);
    };

    constexpr int kVec = kAlignment / sizeof(T);
    const bool vectorized = head_dim % kVec == 0 && reinterpret_cast<uintptr_t>(out_ptr) % kAlignment == 0 &&
                            reinterpret_cast<uintptr_t>(x_ptr) % kAlignment == 0;
    if (vectorized) {
      launch(
          usp_merge_heads_vec_kernel<T, kVec>,
          numel / kVec,
          head_dim / kVec,
          local_heads,
          batch,
          sequence_length,
          world_size);
    } else {
      launch(usp_merge_heads_scalar_kernel<T>, numel, head_dim, local_heads, batch, sequence_length, world_size);
    }
  }
};

}  // namespace usp_relayout

}  // namespace sglang
