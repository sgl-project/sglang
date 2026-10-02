#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace sglang {

/**
 * \brief Apply Helios transposed RoPE to normalized Q/K in place.
 *
 * One thread owns one adjacent rotary pair. The explicit round-to-nearest
 * multiply and add/subtract operations preserve the separate eager FP32
 * intermediates before the result is rounded back to fp16/bf16.
 *
 * \tparam T Activation type: fp16_t or bf16_t
 * \param q Normalized query tensor, contiguous [tokens, heads, head_dim]
 * \param k Normalized key tensor, contiguous [tokens, heads, head_dim]
 * \param freqs Transposed Helios frequency tensor, contiguous
 *              [tokens, 2 * head_dim]
 * \param num_pairs Total adjacent Q/K pairs across tokens and heads
 * \param pairs_per_head Number of adjacent pairs in one attention head
 * \param num_heads Number of attention heads
 * \param freq_stride Last dimension of freqs, equal to 2 * head_dim
 */
template <typename T>
__global__ void helios_qk_rope_kernel(
    T* __restrict__ q,
    T* __restrict__ k,
    const float* __restrict__ freqs,
    uint32_t num_pairs,
    uint32_t pairs_per_head,
    uint32_t num_heads,
    uint32_t freq_stride) {
  static_assert(std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>);
  using Packed = packed_t<T>;

  auto* q_pairs = reinterpret_cast<Packed*>(q);
  auto* k_pairs = reinterpret_cast<Packed*>(k);
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t pair_index = blockIdx.x * blockDim.x + threadIdx.x; pair_index < num_pairs; pair_index += stride) {
    const uint32_t pair_in_head = pair_index % pairs_per_head;
    const uint32_t token_head = pair_index / pairs_per_head;
    const uint32_t token_index = token_head / num_heads;
    const uint32_t head_dim = pairs_per_head * 2;
    const uint32_t freq_base = token_index * freq_stride;
    const float cos = freqs[freq_base + pair_in_head * 2];
    const float sin = freqs[freq_base + head_dim + pair_in_head * 2 + 1];

    const auto q_value = device::cast<fp32x2_t, Packed>(q_pairs[pair_index]);
    const auto k_value = device::cast<fp32x2_t, Packed>(k_pairs[pair_index]);

    const float q_even = __fsub_rn(__fmul_rn(q_value.x, cos), __fmul_rn(q_value.y, sin));
    const float q_odd = __fadd_rn(__fmul_rn(q_value.x, sin), __fmul_rn(q_value.y, cos));
    const float k_even = __fsub_rn(__fmul_rn(k_value.x, cos), __fmul_rn(k_value.y, sin));
    const float k_odd = __fadd_rn(__fmul_rn(k_value.x, sin), __fmul_rn(k_value.y, cos));

    q_pairs[pair_index] = device::cast<Packed, fp32x2_t>(make_float2(q_even, q_odd));
    k_pairs[pair_index] = device::cast<Packed, fp32x2_t>(make_float2(k_even, k_odd));
  }
}

/** \brief Validate and launch the paired Helios Q/K RoPE kernel. */
template <typename DType>
struct HeliosQKRoPEKernel {
  static void run(const tvm::ffi::TensorView q, const tvm::ffi::TensorView k, const tvm::ffi::TensorView freqs) {
    using namespace host;

    auto N = SymbolicSize{"tokens"};
    auto H = SymbolicSize{"heads"};
    auto D = SymbolicSize{"head_dim"};
    auto F = SymbolicSize{"freq_dim"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, H, D}).with_dtype<DType>().with_device(device).verify(q).verify(k);
    TensorMatcher({N, F}).with_dtype<fp32_t>().with_device(device).verify(freqs);

    const int64_t tokens = N.unwrap();
    const int64_t heads = H.unwrap();
    const int64_t head_dim = D.unwrap();
    const int64_t freq_dim = F.unwrap();
    CHECK_HOST(tokens > 0 && heads > 0 && head_dim > 0)
        << "Helios QK RoPE expects positive dimensions, got tokens=" << tokens << ", heads=" << heads
        << ", head_dim=" << head_dim;
    CHECK_HOST(head_dim % 2 == 0) << "Helios QK RoPE head_dim must be even, got " << head_dim;
    CHECK_HOST(freq_dim == 2 * head_dim) << "Helios QK RoPE expects freq_dim=" << 2 * head_dim << ", got " << freq_dim;
    CHECK_HOST(reinterpret_cast<uintptr_t>(q.data_ptr()) % alignof(packed_t<DType>) == 0)
        << "Helios QK RoPE query pointer is not pair aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(k.data_ptr()) % alignof(packed_t<DType>) == 0)
        << "Helios QK RoPE key pointer is not pair aligned";

    const int64_t num_pairs_i64 = tokens * heads * (head_dim / 2);
    CHECK_HOST(num_pairs_i64 <= std::numeric_limits<uint32_t>::max())
        << "Helios QK RoPE pair count exceeds uint32: " << num_pairs_i64;

    const uint32_t num_pairs = static_cast<uint32_t>(num_pairs_i64);
    constexpr uint32_t kBlockSize = 256;
    const uint32_t grid = div_ceil(num_pairs, kBlockSize);
    LaunchKernel(grid, kBlockSize, device.unwrap())(
        helios_qk_rope_kernel<DType>,
        static_cast<DType*>(q.data_ptr()),
        static_cast<DType*>(k.data_ptr()),
        static_cast<const float*>(freqs.data_ptr()),
        num_pairs,
        static_cast<uint32_t>(head_dim / 2),
        static_cast<uint32_t>(heads),
        static_cast<uint32_t>(freq_dim));
  }
};

}  // namespace sglang
