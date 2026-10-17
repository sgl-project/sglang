// Conv bias + residual add in one pass: out = bf16(bf16(y + bias[c]) + h).
//
// For a conv that ran without its bias, this replaces aten's separate bias pass
// (read + write of y) followed by the residual add with one elementwise kernel.
// aten adds bf16 tensors in fp32 and rounds once per op, and so does this kernel,
// so the result is bit-exact against `(y + bias.view(...)) + h`. Two layouts:
// channel innermost ([rows, C], channels_last) and channel outer ([N * C, S],
// NCHW / NCDHW contiguous); the vector never straddles a channel in either.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace bias_residual_add {

constexpr uint32_t kThreads = 256;
constexpr int kVec = 8;  // 16-byte bf16 vectors
constexpr uint32_t kMaxBlocks = 65535;

struct Params {
  bf16_t* out;
  const bf16_t* y;
  const bf16_t* h;
  const bf16_t* bias;
  int64_t numel;
  int64_t channels;
  int64_t spatial;  // elements per (n, c) row in the channel-outer layout
};

/// \tparam kChannelsInner channel is the innermost dim (else outer, [N * C, S])
template <bool kChannelsInner>
__global__ void kernel(const Params __grid_constant__ params) {
  using namespace device;
  using Vec = AlignedVector<bf16_t, kVec>;
  const int64_t vectors = params.numel / kVec;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; v < vectors; v += stride) {
    const int64_t e = v * kVec;
    Vec y, h, bias, out;
    y.load(params.y + e);
    h.load(params.h + e);
    if constexpr (kChannelsInner) {
      bias.load(params.bias + e % params.channels);
    } else {
      bias.fill(params.bias[(e / params.spatial) % params.channels]);
    }
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float biased = cast<fp32_t>(cast<bf16_t>(__fadd_rn(cast<fp32_t>(y[j]), cast<fp32_t>(bias[j]))));
      out[j] = cast<bf16_t>(__fadd_rn(biased, cast<fp32_t>(h[j])));
    }
    out.store(params.out + e);
  }
}

template <bool kChannelsInner>
struct Kernel {
  /// \brief out = bf16(bf16(y + bias[c]) + h) over flat contiguous bf16 tensors.
  /// \param channels C; \param spatial elements per (n, c) row (channel-outer layout only).
  static void
  run(tvm::ffi::TensorView out,
      tvm::ffi::TensorView y,
      tvm::ffi::TensorView h,
      tvm::ffi::TensorView bias,
      int64_t channels,
      int64_t spatial) {
    using namespace host;
    constexpr int64_t kAlign = kVec * static_cast<int64_t>(sizeof(bf16_t));
    auto N = SymbolicSize{"numel"};
    auto C = SymbolicSize{"channels"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N}).with_dtype<bf16_t>().with_device(device).ensure_alignment(kAlign).verify(out).verify(y).verify(
        h);
    TensorMatcher({C}).with_dtype<bf16_t>().with_device(device).ensure_alignment(kAlign).verify(bias);
    const int64_t numel = N.unwrap();
    CHECK_HOST(C.unwrap() == channels) << "bias has " << C.unwrap() << " entries, expected " << channels;
    CHECK_HOST(numel % kVec == 0) << "numel must be a multiple of " << kVec;
    if constexpr (kChannelsInner) {
      CHECK_HOST(channels % kVec == 0 && numel % channels == 0) << "channels must be a multiple of " << kVec;
    } else {
      CHECK_HOST(spatial % kVec == 0 && numel % (channels * spatial) == 0)
          << "spatial must be a multiple of " << kVec << " and divide numel";
    }
    if (numel == 0) return;
    const auto params = Params{
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .y = static_cast<const bf16_t*>(y.data_ptr()),
        .h = static_cast<const bf16_t*>(h.data_ptr()),
        .bias = static_cast<const bf16_t*>(bias.data_ptr()),
        .numel = numel,
        .channels = channels,
        .spatial = spatial,
    };
    const int64_t vectors = numel / kVec;
    const auto blocks =
        static_cast<uint32_t>(std::min<int64_t>(div_ceil(vectors, static_cast<int64_t>(kThreads)), kMaxBlocks));
    LaunchKernel(blocks, kThreads, device.unwrap())(kernel<kChannelsInner>, params);
  }
};

}  // namespace bias_residual_add

}  // namespace sglang
