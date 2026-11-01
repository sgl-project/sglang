// SPDX-License-Identifier: Apache-2.0

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct SamRelPosBiasParams {
  bf16_t* __restrict__ output;
  const bf16_t* __restrict__ rel_h;
  const bf16_t* __restrict__ rel_w;
  int64_t rel_h_batch_stride;
  int64_t rel_h_head_stride;
  int64_t rel_h_query_stride;
  int64_t rel_h_key_stride;
  int64_t rel_w_batch_stride;
  int64_t rel_w_head_stride;
  int64_t rel_w_query_stride;
  int64_t rel_w_key_stride;
  uint32_t num_heads;
  uint32_t num_queries;
  uint32_t key_height;
  uint32_t key_width;
  uint32_t num_elements;
};

/**
 * \brief Materialize decomposed SAM relative-position bias.
 *
 * \tparam kUsePDL Whether to emit the PDL wait/trigger pair.
 */
template <bool kUsePDL>
__global__ void sam_rel_pos_bias_kernel(const __grid_constant__ SamRelPosBiasParams params) {
  using Vector = device::AlignedVector<bf16_t, 8>;
  constexpr uint32_t kVectorSize = 8;
  const uint32_t vector = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t first = vector * kVectorSize;
  if (first >= params.num_elements) {
    return;
  }
  device::PDLWaitPrimary<kUsePDL>();

  const uint32_t key_width_index = first % params.key_width;
  uint32_t remaining = first / params.key_width;
  const uint32_t key_height_index = remaining % params.key_height;
  remaining /= params.key_height;
  const uint32_t query = remaining % params.num_queries;
  remaining /= params.num_queries;
  const uint32_t head = remaining % params.num_heads;
  const uint32_t batch = remaining / params.num_heads;

  const int64_t rel_h_offset = static_cast<int64_t>(batch) * params.rel_h_batch_stride +
                               static_cast<int64_t>(head) * params.rel_h_head_stride +
                               static_cast<int64_t>(query) * params.rel_h_query_stride +
                               static_cast<int64_t>(key_height_index) * params.rel_h_key_stride;
  const int64_t rel_w_offset = static_cast<int64_t>(batch) * params.rel_w_batch_stride +
                               static_cast<int64_t>(head) * params.rel_w_head_stride +
                               static_cast<int64_t>(query) * params.rel_w_query_stride +
                               static_cast<int64_t>(key_width_index) * params.rel_w_key_stride;
  const bf16_t rel_h_value = params.rel_h[rel_h_offset];
  Vector rel_w_values;
  Vector output_values;
  rel_w_values.load(params.rel_w + rel_w_offset, 0);
#pragma unroll
  for (uint32_t i = 0; i < kVectorSize; ++i) {
    output_values[i] = rel_h_value + rel_w_values[i];
  }
  output_values.store(params.output + first, 0);
  device::PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Validate tensors and launch sam_rel_pos_bias_kernel.
 */
template <bool kUsePDL>
void sam_rel_pos_bias(tvm::ffi::TensorView output, tvm::ffi::TensorView rel_h, tvm::ffi::TensorView rel_w) {
  using namespace host;
  auto B = SymbolicSize{"batch"};
  auto H = SymbolicSize{"heads"};
  auto Q = SymbolicSize{"queries"};
  auto KH = SymbolicSize{"key_height"};
  auto KW = SymbolicSize{"key_width"};
  auto One = SymbolicSize{"one"};
  auto HSB = SymbolicSize{"rel_h_batch_stride"};
  auto HSH = SymbolicSize{"rel_h_head_stride"};
  auto HSQ = SymbolicSize{"rel_h_query_stride"};
  auto HSK = SymbolicSize{"rel_h_key_stride"};
  auto WSB = SymbolicSize{"rel_w_batch_stride"};
  auto WSH = SymbolicSize{"rel_w_head_stride"};
  auto WSQ = SymbolicSize{"rel_w_query_stride"};
  auto WSK = SymbolicSize{"rel_w_key_stride"};
  auto device = SymbolicDevice{};
  One.set_value(1);
  WSK.set_value(1);
  device.set_options<kDLCUDA>();

  TensorMatcher({B, H, Q, KH, One})
      .with_strides({HSB, HSH, HSQ, HSK, -1})
      .with_dtype<bf16_t>()
      .with_device(device)
      .verify(rel_h);
  // Each thread loads eight consecutive key-width values with one 16-byte access.
  TensorMatcher({B, H, Q, One, KW})
      .with_strides({WSB, WSH, WSQ, -1, WSK})
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(rel_w);
  TensorMatcher({B, H, Q, -1}).with_dtype<bf16_t>().with_device(device).verify(output);
  CHECK_HOST(output.size(3) == KH.unwrap() * KW.unwrap())
      << "Output key dimension does not match relative-position geometry";

  CHECK_HOST(KW.unwrap() % 8 == 0) << "Key width must be divisible by eight, got " << KW.unwrap();
  const auto num_elements = static_cast<uint32_t>(B.unwrap() * H.unwrap() * Q.unwrap() * KH.unwrap() * KW.unwrap());
  const auto params = SamRelPosBiasParams{
      .output = static_cast<bf16_t*>(output.data_ptr()),
      .rel_h = static_cast<const bf16_t*>(rel_h.data_ptr()),
      .rel_w = static_cast<const bf16_t*>(rel_w.data_ptr()),
      .rel_h_batch_stride = HSB.unwrap(),
      .rel_h_head_stride = HSH.unwrap(),
      .rel_h_query_stride = HSQ.unwrap(),
      .rel_h_key_stride = HSK.unwrap(),
      .rel_w_batch_stride = WSB.unwrap(),
      .rel_w_head_stride = WSH.unwrap(),
      .rel_w_query_stride = WSQ.unwrap(),
      .rel_w_key_stride = WSK.unwrap(),
      .num_heads = static_cast<uint32_t>(H.unwrap()),
      .num_queries = static_cast<uint32_t>(Q.unwrap()),
      .key_height = static_cast<uint32_t>(KH.unwrap()),
      .key_width = static_cast<uint32_t>(KW.unwrap()),
      .num_elements = num_elements,
  };
  constexpr uint32_t kBlockSize = 256;
  constexpr uint32_t kVectorSize = 8;
  const uint32_t num_blocks = div_ceil(num_elements, kBlockSize * kVectorSize);
  LaunchKernel(num_blocks, kBlockSize, device.unwrap()).enable_pdl(kUsePDL)(sam_rel_pos_bias_kernel<kUsePDL>, params);
}

}  // namespace sglang
