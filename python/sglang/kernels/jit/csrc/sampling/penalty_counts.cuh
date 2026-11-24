#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct AccumulatePenaltyCountsParams {
  int32_t* __restrict__ counts;
  int32_t* __restrict__ lens;
  const int64_t* __restrict__ batch_indices;
  const int64_t* __restrict__ penalty_indices;
  const void* __restrict__ token_ids;
  int64_t token_row_stride;
  int64_t token_col_stride;
  bool token_ids_are_int64;
  const uint8_t* __restrict__ accept_mask;  // null when every token counts
  int64_t mask_row_stride;
  int64_t mask_col_stride;
  int64_t num_requests;
  int64_t num_slots;
  int32_t vocab_size;
};

constexpr uint32_t kAccumulatePenaltyCountsThreads = 256;

/// One thread per (penalized request, token slot); integer atomics keep the
/// counts exact when a request's tokens repeat.
__global__ void accumulate_penalty_counts_kernel(const __grid_constant__ AccumulatePenaltyCountsParams p) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kAccumulatePenaltyCountsThreads + threadIdx.x;
  if (i >= p.num_requests * p.num_slots) return;
  const int64_t request = i / p.num_slots;
  const int64_t slot = i % p.num_slots;
  const int64_t batch_index = p.batch_indices[request];
  const int64_t offset = batch_index * p.token_row_stride + slot * p.token_col_stride;
  const int64_t token = p.token_ids_are_int64 ? static_cast<const int64_t*>(p.token_ids)[offset]
                                              : static_cast<const int32_t*>(p.token_ids)[offset];
  // A NaN-logit sample or a masked draft slot can hold an id outside the vocab.
  bool counted = 0 <= token && token < p.vocab_size;
  if (p.accept_mask != nullptr) {
    counted = counted && p.accept_mask[batch_index * p.mask_row_stride + slot * p.mask_col_stride] != 0;
  }
  if (!counted) return;
  const int64_t penalty_index = p.penalty_indices[request];
  atomicAdd(&p.counts[penalty_index * p.vocab_size + token], 1);
  atomicAdd(&p.lens[penalty_index], 1);
}

/// Count accepted in-vocab tokens into each penalized request's history row; see `penalty_counts.py`.
inline void accumulate_penalty_counts(
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView lens,
    tvm::ffi::TensorView batch_indices,
    tvm::ffi::TensorView penalty_indices,
    tvm::ffi::TensorView token_ids,
    tvm::ffi::Optional<tvm::ffi::TensorView> accept_mask) {
  using namespace host;
  SymbolicSize bs{"bs"}, k{"num_slots"}, vocab{"vocab"}, n{"num_penalized"}, num_rows{"num_history_rows"};
  SymbolicSize token_row_stride{"token_row_stride"}, token_col_stride{"token_col_stride"};
  SymbolicDevice device;
  device.set_options<kDLGPU>();

  TensorMatcher({num_rows, vocab})  //
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(counts);
  TensorMatcher({num_rows})  //
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(lens);
  TensorMatcher({n})  //
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(batch_indices)
      .verify(penalty_indices);
  TensorMatcher({bs, k})  //
      .with_strides({token_row_stride, token_col_stride})
      .with_dtype<int32_t, int64_t>()
      .with_device(device)
      .verify(token_ids);

  const uint8_t* accept_mask_ptr = nullptr;
  int64_t mask_row_stride = 0;
  int64_t mask_col_stride = 0;
  if (accept_mask.has_value()) {
    SymbolicSize row_stride{"mask_row_stride"}, col_stride{"mask_col_stride"};
    TensorMatcher({bs, k})  //
        .with_strides({row_stride, col_stride})
        .with_device(device)
        .verify(accept_mask.value());
    CHECK_HOST(accept_mask.value().dtype().bits == 8) << "accumulate_penalty_counts: accept_mask must be bool";
    accept_mask_ptr = static_cast<const uint8_t*>(accept_mask.value().data_ptr());
    mask_row_stride = row_stride.unwrap();
    mask_col_stride = col_stride.unwrap();
  }

  const int64_t num_threads = n.unwrap() * k.unwrap();
  if (num_threads == 0) return;
  CHECK_HOST(vocab.unwrap() < (int64_t{1} << 31)) << "accumulate_penalty_counts: vocab overflows int32";

  const auto params = AccumulatePenaltyCountsParams{
      .counts = static_cast<int32_t*>(counts.data_ptr()),
      .lens = static_cast<int32_t*>(lens.data_ptr()),
      .batch_indices = static_cast<const int64_t*>(batch_indices.data_ptr()),
      .penalty_indices = static_cast<const int64_t*>(penalty_indices.data_ptr()),
      .token_ids = token_ids.data_ptr(),
      .token_row_stride = token_row_stride.unwrap(),
      .token_col_stride = token_col_stride.unwrap(),
      .token_ids_are_int64 = token_ids.dtype().bits == 64,
      .accept_mask = accept_mask_ptr,
      .mask_row_stride = mask_row_stride,
      .mask_col_stride = mask_col_stride,
      .num_requests = n.unwrap(),
      .num_slots = k.unwrap(),
      .vocab_size = static_cast<int32_t>(vocab.unwrap()),
  };
  const auto num_blocks = static_cast<uint32_t>(div_ceil(num_threads, int64_t{kAccumulatePenaltyCountsThreads}));
  LaunchKernel(num_blocks, kAccumulatePenaltyCountsThreads, device.unwrap())(accumulate_penalty_counts_kernel, params);
}

}  // namespace sglang
