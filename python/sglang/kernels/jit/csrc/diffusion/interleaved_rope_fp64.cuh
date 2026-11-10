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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
  // Directed FP32 bounds enclose each separately rounded FP64 operation.
  // Matching finite BF16 endpoint bits certify the original two-stage rounding;
  // ambiguous boundaries and nonfinite values use the FP64 expression below.
  const float a = static_cast<float>(input[0]);
  const float b = static_cast<float>(input[1]);
  const float cl = __double2float_rd(cos), ch = __double2float_ru(cos);
  const float sl = __double2float_rd(sin), sh = __double2float_ru(sin);
  if (isfinite(a) && isfinite(b) && isfinite(cl) && isfinite(ch) && isfinite(sl) && isfinite(sh)) {
    device::AlignedVector<T, 2> output;
    if (a == 0.0f && b == 0.0f) {
      // Finite products of signed zeros preserve their signs at either precision.
      output[0] = static_cast<T>(__fsub_rn(__fmul_rn(a, cl), __fmul_rn(b, sl)));
      output[1] = static_cast<T>(__fadd_rn(__fmul_rn(a, sl), __fmul_rn(b, cl)));
      return output;
    }
    const float acl = __fmul_rd(a, a < 0 ? ch : cl);
    const float ach = __fmul_ru(a, a < 0 ? cl : ch);
    const float asl = __fmul_rd(a, a < 0 ? sh : sl);
    const float ash = __fmul_ru(a, a < 0 ? sl : sh);
    const float bcl = __fmul_rd(b, b < 0 ? ch : cl);
    const float bch = __fmul_ru(b, b < 0 ? cl : ch);
    const float bsl = __fmul_rd(b, b < 0 ? sh : sl);
    const float bsh = __fmul_ru(b, b < 0 ? sl : sh);
    const T lo0 = static_cast<T>(__fsub_rd(acl, bsh));
    const T hi0 = static_cast<T>(__fsub_ru(ach, bsl));
    const T lo1 = static_cast<T>(__fadd_rd(asl, bcl));
    const T hi1 = static_cast<T>(__fadd_ru(ash, bch));
    const auto bits0 = __bfloat16_as_ushort(lo0), bits1 = __bfloat16_as_ushort(lo1);
    if (bits0 == __bfloat16_as_ushort(hi0) && bits1 == __bfloat16_as_ushort(hi1) && (bits0 & 0x7f80) != 0x7f80 &&
        (bits1 & 0x7f80) != 0x7f80) {
      output[0] = lo0;
      output[1] = lo1;
      return output;
    }
  }
#endif
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
      tvm::ffi::TensorView sin) {
    using namespace host;
    using Pair = device::AlignedVector<T, 2>;

    auto B = SymbolicSize{"batch"};
    auto S = SymbolicSize{"sequence"};
    auto H = SymbolicSize{"heads"};
    auto D = SymbolicSize{"head_dim"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({B, S, H, D}).with_dtype<T>().with_device(device).verify(q_out).verify(k_out).verify(q).verify(k);
    TensorMatcher({1, S, 1, D}).with_dtype<double>().with_device(device).verify(cos).verify(sin);

    const int64_t batch_size = B.unwrap();
    const int64_t seq_len = S.unwrap();
    const int64_t num_heads = H.unwrap();
    const int64_t head_dim = D.unwrap();

    CHECK_HOST(batch_size > 0 && seq_len > 0 && num_heads > 0 && head_dim > 0)
        << "interleaved_rope_fp64 dimensions must be positive";
    CHECK_HOST(head_dim % 2 == 0) << "interleaved_rope_fp64 head_dim must be even";
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

    const int64_t num_pairs = q.numel() / 2;
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
