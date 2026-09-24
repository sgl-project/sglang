#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {
namespace prefill_candidate {
namespace impl = device::topk;

struct Params {
  const float* scores;
  const int32_t* lengths;
  const bool* candidates;
  const int32_t* offsets;
  int32_t* output;
  int64_t score_stride;
  int64_t candidate_stride;
  uint32_t block_size;
  uint32_t topk;
};

struct ScoreFilter {
  const bool* blocks;
  uint32_t block_size;

  SGL_DEVICE float operator()(float value, uint32_t index) const {
    return blocks[index / block_size] ? value : -impl::infinity_value();
  }
};

template <bool kPDL>
__global__ __launch_bounds__(impl::TopKConfig::kBlockSize, impl::TopKConfig::kOccupancy) void select(
    const __grid_constant__ Params params) {
  device::enable_smem_spilling();
  const auto row = blockIdx.x;
  const auto length = static_cast<uint32_t>(params.lengths[row]);
  const auto offset = params.offsets[row];
  const auto scores = params.scores + row * params.score_stride;
  const auto candidates = params.candidates + row * params.candidate_stride;
  const auto output = params.output + row * static_cast<int64_t>(params.topk);
  const ScoreFilter filter{candidates, params.block_size};

  if (length <= params.topk) {
    device::PDLWaitPrimary<kPDL>();
    for (uint32_t i = threadIdx.x; i < params.topk; i += blockDim.x) {
      const bool valid = i < length && candidates[i / params.block_size] && scores[i] > -impl::infinity_value();
      output[i] = valid ? static_cast<int32_t>(i) + offset : -1;
    }
    return;
  }

  using Register2 = impl::TopKRegister<2>;
  using Register4 = impl::TopKRegister<4>;
  using Streaming = impl::TopKStreaming;
  const impl::TopKProblem problem{
      .in = scores,
      .out = output,
      .topk = params.topk,
      .seq_len = length,
      .bias = impl::broadcast(offset),
  };
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  if (length <= Register2::kMaxSeqLen) {
    Register2::forward<kPDL>(problem, &smem, filter);
  } else if (length <= Register4::kMaxSeqLen) {
    Register4::forward<kPDL>(problem, &smem, filter);
  } else {
    Streaming::forward<kPDL>(problem, &smem, filter);
  }
  // Selection can underfill with -inf positions. Match mask_topk_scores in the
  // same CTA, after every thread has finished publishing its selected indices.
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < params.topk; i += blockDim.x) {
    const int64_t index = static_cast<int64_t>(output[i]) - offset;
    const bool valid = index >= 0 && index < length && candidates[index / params.block_size] &&
                       scores[index] > -impl::infinity_value();
    if (!valid) output[i] = -1;
  }
}

template <bool kPDL>
struct Kernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView candidates,
      const tvm::ffi::TensorView offsets,
      const tvm::ffi::TensorView output,
      int64_t block_size) {
    using namespace host;
    auto B = SymbolicSize{"rows"};
    auto W = SymbolicSize{"width"};
    auto S = SymbolicSize{"score_stride"};
    auto C = SymbolicSize{"candidate_width"};
    auto CS = SymbolicSize{"candidate_stride"};
    auto K = SymbolicSize{"topk"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLGPU>();
    TensorMatcher({B, W}).with_strides({S, 1}).with_dtype<float>().with_device(dev).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(lengths);
    TensorMatcher({B, C}).with_strides({CS, 1}).with_device(dev).verify(candidates);
    RuntimeCheck(
        candidates.dtype().code == kDLBool && candidates.dtype().bits == 8 && candidates.dtype().lanes == 1,
        "candidate mask must have dtype bool");
    TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(offsets);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(dev).verify(output);
    RuntimeCheck(block_size > 0, "candidate block size must be positive");
    RuntimeCheck(C.unwrap() >= (W.unwrap() + block_size - 1) / block_size, "candidate mask is too narrow");
    RuntimeCheck(S.unwrap() % 4 == 0, "score rows must be 16-byte aligned");
    RuntimeCheck(K.unwrap() > 0 && K.unwrap() <= impl::TopKConfig::kMaxTopK, "topk must be in (0, 2048]");
    if (B.unwrap() == 0) return;
    const Params params{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .candidates = static_cast<const bool*>(candidates.data_ptr()),
        .offsets = static_cast<const int32_t*>(offsets.data_ptr()),
        .output = static_cast<int32_t*>(output.data_ptr()),
        .score_stride = S.unwrap(),
        .candidate_stride = CS.unwrap(),
        .block_size = static_cast<uint32_t>(block_size),
        .topk = static_cast<uint32_t>(K.unwrap()),
    };
    LaunchKernel(static_cast<uint32_t>(B.unwrap()), impl::TopKConfig::kBlockSize, dev.unwrap())
        .config({.use_pdl = kPDL})
        .launch(select<kPDL>, params);
  }
};
}  // namespace prefill_candidate
}  // namespace sglang
