// Radix-select fast top-k with the AOT fast_topk_v2 semantics: for each row b,
// select the kTopK largest scores in [row_starts[b], row_starts[b] + lengths[b])
// and write their indices relative to row_starts[b]; ties use the smallest relative IDs, and emission follows a fixed warp/lane/word/bit order.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include "deterministic_topk.cuh"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

namespace sglang {

namespace fast_topk_detail {

constexpr uint32_t kThreadsPerBlock = 1024;
// Larger boundary buckets are rescanned without clipping candidates.
constexpr size_t kSmemBytes = 8 * 1024 * sizeof(uint32_t);  // 32KB

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, kTopK]
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
};

// When length <= kTopK, write the indices directly.
template <int kTopK>
SGL_DEVICE void naive_topk(int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < kTopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// Radix-select top-k. Assumes length > kTopK (checked by the caller).
template <int kTopK>
SGL_DEVICE void radix_select_topk(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  deterministic_topk_detail::radix_select<kTopK, kThreadsPerBlock, false, true>(input, index, row_start, length);
}

template <int kTopK, bool kUsePDL, bool kZeroStart = false>
__global__ __launch_bounds__(fast_topk_detail::kThreadsPerBlock) void fast_topk_kernel(
    const fast_topk_detail::FastTopKParams __grid_constant__ params) {
  using namespace fast_topk_detail;
  device::PDLWaitPrimary<kUsePDL>();

  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = kZeroStart ? 0 : params.row_starts[bid];
  const auto length = params.lengths[bid];
  const auto indice = params.indices + bid * kTopK;
  const auto score = params.input + bid * params.input_stride;
  if (length <= kTopK) {
    naive_topk<kTopK>(indice, length);
  } else {
    radix_select_topk<kTopK>(score, indice, row_start, length);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace fast_topk_detail

/**
 * \brief Per-row top-k selection over ragged rows of a fp32 score matrix.
 *
 * Row b selects the kTopK largest values in
 * score[b, row_starts[b] : row_starts[b] + lengths[b]) and writes their
 * indices (relative to row_starts[b]) into indices[b]. Unfilled slots are
 * -1 when lengths[b] < kTopK.
 */
template <int kTopK, bool kUsePDL, bool kZeroStart = false>
struct FastTopKKernel {
  static constexpr auto kernel = fast_topk_detail::fast_topk_kernel<kTopK, kUsePDL, kZeroStart>;

  static void
  run(const tvm::ffi::TensorView score,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView lengths) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto L = SymbolicSize{"length"};
    auto S = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, L})  // score
        .with_strides({S, 1})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(score);
    CHECK_HOST(row_starts.has_value() != kZeroStart);
    if (row_starts.has_value()) {
      TensorMatcher({B})  // row_starts
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(row_starts.value());
    }
    TensorMatcher({B, kTopK})  // indices
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(indices);
    TensorMatcher({B})  // lengths
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(lengths);

    const auto params = fast_topk_detail::FastTopKParams{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = kZeroStart ? nullptr : static_cast<const int32_t*>(row_starts.value().data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .input_stride = S.unwrap(),
    };

    const auto num_rows = static_cast<uint32_t>(B.unwrap());
    LaunchKernel(num_rows, fast_topk_detail::kThreadsPerBlock, device.unwrap(), fast_topk_detail::kSmemBytes)
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
