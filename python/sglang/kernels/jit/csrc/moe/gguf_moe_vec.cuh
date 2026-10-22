#pragma once

#include <sgl_kernel/quantization/gguf/ggml-common.h>
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/quantization/gguf/moe_vec.cuh>
#include <sgl_kernel/quantization/gguf/quantize_q8_1.cuh>
#include <sgl_kernel/quantization/gguf/vecdotq.cuh>

#include <cstdint>
#include <limits>

namespace sglang {
namespace gguf {

template <int kQuantType>
struct MoeVecFormat;

#define GGUF_MOE_VEC_FORMAT(type, qk, block, launcher)                 \
  template <>                                                          \
  struct MoeVecFormat<type> {                                          \
    static constexpr int kBlockSize = qk;                              \
    static constexpr int kBlockBytes = sizeof(block);                  \
    static constexpr int kAlignment = alignof(block);                  \
    template <typename scalar_t>                                       \
    static void                                                        \
    run(const void* w,                                                 \
        const void* q,                                                 \
        scalar_t* y,                                                   \
        const int* ids,                                                \
        int topk,                                                      \
        int m,                                                         \
        int k,                                                         \
        int n,                                                         \
        int stride,                                                    \
        cudaStream_t stream) {                                         \
      launcher<scalar_t>(w, q, y, ids, topk, m, k, n, stride, stream); \
    }                                                                  \
  }

GGUF_MOE_VEC_FORMAT(2, QK4_0, block_q4_0, moe_vec_q4_0_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(3, QK4_0, block_q4_1, moe_vec_q4_1_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(6, QK5_0, block_q5_0, moe_vec_q5_0_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(7, QK5_1, block_q5_1, moe_vec_q5_1_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(8, QK8_0, block_q8_0, moe_vec_q8_0_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(10, QK_K, block_q2_K, moe_vec_q2_K_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(11, QK_K, block_q3_K, moe_vec_q3_K_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(12, QK_K, block_q4_K, moe_vec_q4_K_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(13, QK_K, block_q5_K, moe_vec_q5_K_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(14, QK_K, block_q6_K, moe_vec_q6_K_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(16, QK_K, block_iq2_xxs, moe_vec_iq2_xxs_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(17, QK_K, block_iq2_xs, moe_vec_iq2_xs_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(18, QK_K, block_iq3_xxs, moe_vec_iq3_xxs_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(19, QK_K, block_iq1_s, moe_vec_iq1_s_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(20, QK4_NL, block_iq4_nl, moe_vec_iq4_nl_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(21, QK_K, block_iq3_s, moe_vec_iq3_s_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(22, QK_K, block_iq2_s, moe_vec_iq2_s_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(23, QK_K, block_iq4_xs, moe_vec_iq4_xs_q8_1_cuda);
GGUF_MOE_VEC_FORMAT(29, QK_K, block_iq1_m, moe_vec_iq1_m_q8_1_cuda);

#undef GGUF_MOE_VEC_FORMAT

}  // namespace gguf

/// \brief Quantize input rows to Q8_1 and project compact GGUF expert routes.
/// \param expert_ids Contiguous int32 routes; negative IDs write zero without reading operands.
/// \param output Zero-initialized private output, matching the AOT vector projection lifetime.
/// \param quantized_input Private Q8_1 scratch, with the same padded-row format as the AOT ABI.
template <typename scalar_t, int kQuantType>
void gguf_moe_a8_vec(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView quantized_input,
    int64_t top_k,
    int64_t row,
    int64_t tokens) {
  using namespace host;
  using Format = gguf::MoeVecFormat<kQuantType>;
  constexpr int64_t kMaxIndex = std::numeric_limits<int32_t>::max();
  CHECK_HOST(tokens >= 0 && tokens <= kMaxIndex && top_k > 0 && top_k <= kMaxIndex);
  CHECK_HOST(tokens <= 65535 / top_k) << "GGUF vector routes exceed the CUDA z-grid limit";
  CHECK_HOST(row > 0 && row <= kMaxIndex);

  auto device = SymbolicDevice{};
  auto columns = SymbolicSize{"columns"};
  auto experts = SymbolicSize{"experts"};
  TensorMatcher({tokens, columns}).with_dtype<scalar_t>().template with_device<kDLCUDA>(device).verify(input);
  const int64_t k = columns.unwrap();
  CHECK_HOST(k > 0 && k <= kMaxIndex - 511 && k % Format::kBlockSize == 0);
  const int64_t packed_columns = k / Format::kBlockSize * Format::kBlockBytes;
  TensorMatcher({experts, row, packed_columns})
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(std::max(2, Format::kAlignment))
      .verify(weight);
  CHECK_HOST(experts.unwrap() > 0 && experts.unwrap() <= kMaxIndex / row / (k / Format::kBlockSize));
  TensorMatcher({-1, -1}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
  CHECK_HOST(expert_ids.numel() == tokens * top_k);
  TensorMatcher({tokens * top_k, row}).with_dtype<scalar_t>().with_device(device).verify(output);
  const int64_t padded = (k + 511) / 512 * 512;
  TensorMatcher({tokens, padded / 32 * 9}).with_dtype<int32_t>().with_device(device).verify(quantized_input);
  CHECK_HOST(tokens <= kMaxIndex / padded && tokens * top_k <= kMaxIndex / row);
  if (tokens == 0) return;

  const auto stream = LaunchKernel::resolve_device(device.unwrap());
  quantize_row_q8_1_cuda<scalar_t>(
      static_cast<const scalar_t*>(input.data_ptr()),
      quantized_input.data_ptr(),
      static_cast<int>(k),
      static_cast<int>(tokens),
      stream);
  Format::template run<scalar_t>(
      weight.data_ptr(),
      quantized_input.data_ptr(),
      static_cast<scalar_t*>(output.data_ptr()),
      static_cast<const int32_t*>(expert_ids.data_ptr()),
      static_cast<int>(top_k),
      static_cast<int>(tokens),
      static_cast<int>(k),
      static_cast<int>(row),
      static_cast<int>(padded / 32 * 9),
      stream);
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace sglang
