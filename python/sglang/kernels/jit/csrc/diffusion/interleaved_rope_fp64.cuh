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

namespace interleaved_rope_fp64 {

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kMaxGrid = 65535;

template <typename T>
SGL_DEVICE device::AlignedVector<T, 2> rotate_pair(device::AlignedVector<T, 2> input, double cos, double sin) {
  static_assert(std::is_same_v<T, bf16_t>);
  const double x1 = static_cast<double>(input[0]);
  const double x2 = static_cast<double>(input[1]);
  const double x1_cos = __dmul_rn(x1, cos);
  const double x2_sin = __dmul_rn(x2, sin);
  const double x1_sin = __dmul_rn(x1, sin);
  const double x2_cos = __dmul_rn(x2, cos);
  device::AlignedVector<T, 2> output;
  // TensorIterator casts the fp64 expression through fp32 before its bf16
  // store. Preserving both conversions is required at bf16 tie boundaries.
  output[0] = static_cast<T>(static_cast<float>(__dsub_rn(x1_cos, x2_sin)));
  output[1] = static_cast<T>(static_cast<float>(__dadd_rn(x1_sin, x2_cos)));
  return output;
}

template <typename T>
__global__ void interleaved_rope_fp64_kernel(
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    const T* __restrict__ q,
    const T* __restrict__ k,
    const double* __restrict__ cos,
    const double* __restrict__ sin,
    int64_t num_pairs,
    int64_t seq_len,
    int64_t num_heads,
    int64_t pairs_per_head,
    int64_t head_dim) {
  using Pair = device::AlignedVector<T, 2>;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t pair_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pair_index < num_pairs;
       pair_index += stride) {
    const int64_t pair_in_head = pair_index % pairs_per_head;
    const int64_t row = pair_index / (num_heads * pairs_per_head);
    const int64_t token = row % seq_len;
    const int64_t table_index = token * head_dim + 2 * pair_in_head;
    const double cos_value = SGLANG_LDG(cos + table_index);
    const double sin_value = SGLANG_LDG(sin + table_index + 1);

    Pair q_pair;
    q_pair.load(q, pair_index);
    rotate_pair(q_pair, cos_value, sin_value).store(q_out, pair_index);

    Pair k_pair;
    k_pair.load(k, pair_index);
    rotate_pair(k_pair, cos_value, sin_value).store(k_out, pair_index);
  }
}

}  // namespace

/**
 * \brief Apply Diffusers-compatible interleaved RoPE to Q and K.
 *
 * The fp64 multiply and add/subtract roundings deliberately mirror the eager
 * PyTorch expression used by SANA-Video.
 *
 * \tparam T Activation type; currently bf16 only.
 */
template <typename T>
struct InterleavedRopeFP64Kernel {
  static_assert(std::is_same_v<T, bf16_t>);

  static void
  run(tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView cos,
      tvm::ffi::TensorView sin,
      int64_t batch_size,
      int64_t seq_len,
      int64_t num_heads,
      int64_t head_dim) {
    using namespace host;
    using Pair = device::AlignedVector<T, 2>;

    auto N = SymbolicSize{"activation_elements"};
    auto R = SymbolicSize{"table_elements"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N}).with_dtype<T>().with_device(device).verify(q_out).verify(k_out).verify(q).verify(k);
    TensorMatcher({R}).with_dtype<double>().with_device(device).verify(cos).verify(sin);

    CHECK_HOST(batch_size > 0 && seq_len > 0 && num_heads > 0 && head_dim > 0)
        << "interleaved_rope_fp64 dimensions must be positive";
    CHECK_HOST(head_dim % 2 == 0) << "interleaved_rope_fp64 head_dim must be even";
    CHECK_HOST(N.unwrap() == batch_size * seq_len * num_heads * head_dim)
        << "interleaved_rope_fp64 activation shape does not match dimensions";
    CHECK_HOST(R.unwrap() == seq_len * head_dim) << "interleaved_rope_fp64 table shape does not match dimensions";
    CHECK_HOST(
        q_out.data_ptr() != k_out.data_ptr() && q_out.data_ptr() != q.data_ptr() && q_out.data_ptr() != k.data_ptr() &&
        k_out.data_ptr() != q.data_ptr() && k_out.data_ptr() != k.data_ptr())
        << "interleaved_rope_fp64 outputs must not alias inputs";
    CHECK_HOST(
        reinterpret_cast<uintptr_t>(q_out.data_ptr()) % alignof(Pair) == 0 &&
        reinterpret_cast<uintptr_t>(k_out.data_ptr()) % alignof(Pair) == 0 &&
        reinterpret_cast<uintptr_t>(q.data_ptr()) % alignof(Pair) == 0 &&
        reinterpret_cast<uintptr_t>(k.data_ptr()) % alignof(Pair) == 0)
        << "interleaved_rope_fp64 activations must be aligned to rotation pairs";

    const int64_t num_pairs = N.unwrap() / 2;
    const int64_t pairs_per_head = head_dim / 2;
    const auto blocks =
        static_cast<uint32_t>(std::min<int64_t>(div_ceil(num_pairs, static_cast<int64_t>(kBlockSize)), kMaxGrid));
    LaunchKernel(blocks, kBlockSize, device.unwrap())(
        interleaved_rope_fp64_kernel<T>,
        static_cast<T*>(q_out.data_ptr()),
        static_cast<T*>(k_out.data_ptr()),
        static_cast<const T*>(q.data_ptr()),
        static_cast<const T*>(k.data_ptr()),
        static_cast<const double*>(cos.data_ptr()),
        static_cast<const double*>(sin.data_ptr()),
        num_pairs,
        seq_len,
        num_heads,
        pairs_per_head,
        head_dim);
  }
};

}  // namespace interleaved_rope_fp64

}  // namespace sglang
