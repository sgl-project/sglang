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

namespace ltx25_decoder_rope {

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kMaxGrid = 65535;

template <typename T>
SGL_DEVICE device::AlignedVector<T, 2> rotate_pair(device::AlignedVector<T, 2> input, float cos, float sin) {
  static_assert(std::is_same_v<T, bf16_t>);
  const float even = static_cast<float>(input[0]);
  const float odd = static_cast<float>(input[1]);
  const float even_cos = __fmul_rn(even, cos);
  const float odd_sin = __fmul_rn(odd, sin);
  const float even_sin = __fmul_rn(even, sin);
  const float odd_cos = __fmul_rn(odd, cos);
  device::AlignedVector<T, 2> output;
  output[0] = static_cast<T>(__fsub_rn(even_cos, odd_sin));
  output[1] = static_cast<T>(__fadd_rn(even_sin, odd_cos));
  return output;
}

template <typename T>
__global__ void ltx25_decoder_rope_kernel(
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    const T* __restrict__ q,
    const T* __restrict__ k,
    const float* __restrict__ cos_t,
    const float* __restrict__ sin_t,
    const float* __restrict__ cos_h,
    const float* __restrict__ sin_h,
    const float* __restrict__ cos_w,
    const float* __restrict__ sin_w,
    int64_t num_pairs,
    int64_t num_frames,
    int64_t height,
    int64_t width,
    int64_t num_heads,
    int64_t pairs_per_head,
    int64_t t_pairs,
    int64_t h_pairs) {
  using Pair = device::AlignedVector<T, 2>;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t pair_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pair_index < num_pairs;
       pair_index += stride) {
    const int64_t pair_in_head = pair_index % pairs_per_head;
    const int64_t row = pair_index / (num_heads * pairs_per_head);
    const int64_t token = row % (num_frames * height * width);
    const int64_t frame = token / (height * width);
    const int64_t spatial = token % (height * width);
    const int64_t y = spatial / width;
    const int64_t x = spatial % width;

    const float* cos_table;
    const float* sin_table;
    int64_t table_index;
    if (pair_in_head < t_pairs) {
      cos_table = cos_t;
      sin_table = sin_t;
      table_index = frame * t_pairs + pair_in_head;
    } else if (pair_in_head < t_pairs + h_pairs) {
      cos_table = cos_h;
      sin_table = sin_h;
      table_index = y * h_pairs + pair_in_head - t_pairs;
    } else {
      const int64_t w_pairs = pairs_per_head - t_pairs - h_pairs;
      cos_table = cos_w;
      sin_table = sin_w;
      table_index = x * w_pairs + pair_in_head - t_pairs - h_pairs;
    }
    const float cos_value = SGLANG_LDG(cos_table + table_index);
    const float sin_value = SGLANG_LDG(sin_table + table_index);

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
 * \brief Apply LTX-2.5 decoder 3D RoPE to Q and K from compact axis tables.
 *
 * The separate fp32 multiply and add/subtract roundings mirror the eager
 * PyTorch expression used by the decoder.
 *
 * \tparam T Activation type; currently bf16 only.
 */
template <typename T>
struct LTX25DecoderRopeKernel {
  static_assert(std::is_same_v<T, bf16_t>);

  static void
  run(tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView cos_t,
      tvm::ffi::TensorView sin_t,
      tvm::ffi::TensorView cos_h,
      tvm::ffi::TensorView sin_h,
      tvm::ffi::TensorView cos_w,
      tvm::ffi::TensorView sin_w,
      int64_t batch_size,
      int64_t num_frames,
      int64_t height,
      int64_t width,
      int64_t num_heads,
      int64_t head_dim,
      int64_t dim_t,
      int64_t dim_h) {
    using namespace host;
    using Pair = device::AlignedVector<T, 2>;

    auto N = SymbolicSize{"activation_elements"};
    auto RT = SymbolicSize{"temporal_table_elements"};
    auto RH = SymbolicSize{"height_table_elements"};
    auto RW = SymbolicSize{"width_table_elements"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N}).with_dtype<T>().with_device(device).verify(q_out).verify(k_out).verify(q).verify(k);
    TensorMatcher({RT}).with_dtype<float>().with_device(device).verify(cos_t).verify(sin_t);
    TensorMatcher({RH}).with_dtype<float>().with_device(device).verify(cos_h).verify(sin_h);
    TensorMatcher({RW}).with_dtype<float>().with_device(device).verify(cos_w).verify(sin_w);

    CHECK_HOST(batch_size > 0 && num_frames > 0 && height > 0 && width > 0 && num_heads > 0 && head_dim > 0)
        << "ltx25_decoder_rope dimensions must be positive";
    CHECK_HOST(head_dim % 2 == 0 && dim_t > 0 && dim_h > 0 && dim_t % 2 == 0 && dim_h % 2 == 0)
        << "ltx25_decoder_rope dimensions must split into positive rotation pairs";
    const int64_t dim_w = head_dim - dim_t - dim_h;
    CHECK_HOST(dim_w > 0 && dim_w % 2 == 0) << "ltx25_decoder_rope width dimension must be positive and even";
    CHECK_HOST(N.unwrap() == batch_size * num_frames * height * width * num_heads * head_dim)
        << "ltx25_decoder_rope activation shape does not match dimensions";
    CHECK_HOST(RT.unwrap() == num_frames * dim_t / 2) << "ltx25_decoder_rope temporal table shape mismatch";
    CHECK_HOST(RH.unwrap() == height * dim_h / 2) << "ltx25_decoder_rope height table shape mismatch";
    CHECK_HOST(RW.unwrap() == width * dim_w / 2) << "ltx25_decoder_rope width table shape mismatch";
    CHECK_HOST(
        q_out.data_ptr() != k_out.data_ptr() && q_out.data_ptr() != q.data_ptr() && q_out.data_ptr() != k.data_ptr() &&
        k_out.data_ptr() != q.data_ptr() && k_out.data_ptr() != k.data_ptr())
        << "ltx25_decoder_rope outputs must not alias inputs";
    CHECK_HOST(
        reinterpret_cast<uintptr_t>(q_out.data_ptr()) % alignof(Pair) == 0 &&
        reinterpret_cast<uintptr_t>(k_out.data_ptr()) % alignof(Pair) == 0 &&
        reinterpret_cast<uintptr_t>(q.data_ptr()) % alignof(Pair) == 0 &&
        reinterpret_cast<uintptr_t>(k.data_ptr()) % alignof(Pair) == 0)
        << "ltx25_decoder_rope activations must be aligned to rotation pairs";

    const int64_t num_pairs = N.unwrap() / 2;
    const int64_t pairs_per_head = head_dim / 2;
    const auto blocks =
        static_cast<uint32_t>(std::min<int64_t>(div_ceil(num_pairs, static_cast<int64_t>(kBlockSize)), kMaxGrid));
    LaunchKernel(blocks, kBlockSize, device.unwrap())(
        ltx25_decoder_rope_kernel<T>,
        static_cast<T*>(q_out.data_ptr()),
        static_cast<T*>(k_out.data_ptr()),
        static_cast<const T*>(q.data_ptr()),
        static_cast<const T*>(k.data_ptr()),
        static_cast<const float*>(cos_t.data_ptr()),
        static_cast<const float*>(sin_t.data_ptr()),
        static_cast<const float*>(cos_h.data_ptr()),
        static_cast<const float*>(sin_h.data_ptr()),
        static_cast<const float*>(cos_w.data_ptr()),
        static_cast<const float*>(sin_w.data_ptr()),
        num_pairs,
        num_frames,
        height,
        width,
        num_heads,
        pairs_per_head,
        dim_t / 2,
        dim_h / 2);
  }
};

}  // namespace ltx25_decoder_rope

}  // namespace sglang
