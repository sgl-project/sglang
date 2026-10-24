// Channels-last channel RMSNorm + SiLU for the Qwen-Image 2.1 VAE (quality-gated).
//
// A group of kLanesPerRow lanes handles one pixel: its C channel values are
// contiguous in NHWC, so the group reduces the fp32 sum of squares itself instead
// of running aten's NCHW reduction plus the layout transposes cuDNN inserts around
// NHWC convs. Python picks the load width (kVec bf16 per lane), the lanes per pixel
// and the loads per lane from C, so a narrow channel count (C = 144 at 1024^2)
// still keeps several 16-byte loads in flight per lane instead of one 4-byte load.
// With kHasBias the preceding conv's per-channel bias is added first with aten's
// bf16 rounding, which lets the caller run that conv without its own bias pass.
// The pointwise tail matches the eager chain (bf16 after the divide, the scale,
// the gamma multiply and the +0.0 bias; aten's silu formula), but the reduction
// order differs from aten's, so the result is close, not bit-exact.
//   x'[p, c]  = kHasBias ? bf16(x[p, c] + bias[c]) : x[p, c]
//   out[p, c] = silu(bf16(bf16(bf16(bf16(x'[p, c] / max(||x'[p, :]||, 1e-12)) * scale) * gamma[c]) + 0))

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace channel_rmsnorm_silu_nhwc {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarpsPerBlock = kThreads / device::kWarpThreads;
constexpr uint32_t kMaxBlocks = 65535;
constexpr float kNormFloor = 1.0e-12f;

struct Params {
  bf16_t* out;
  const bf16_t* x;
  const bf16_t* gamma;
  const bf16_t* bias;  // read only when kHasBias
  int64_t rows;        // pixels (N * spatial)
  int64_t channels;    // C, a multiple of kVec
  float scale;
};

SGL_DEVICE float div_rn(float numerator, float denominator) {
  float out;
  asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(numerator), "f"(denominator));
  return out;
}

/// \tparam kVec bf16 elements per vector load (2, 4 or 8)
/// \tparam kLanesPerRow lanes cooperating on one pixel (power of two, <= 32)
/// \tparam kUnitsPerLane vector loads per lane, >= ceil(C / kVec / kLanesPerRow)
/// \tparam kHasBias add a per-channel bias (aten bf16 rounding) before the norm
template <int kVec, int kLanesPerRow, int kUnitsPerLane, bool kHasBias>
__global__ void kernel(const Params __grid_constant__ params) {
  using namespace device;
  static_assert(kVec == 2 || kVec == 4 || kVec == 8, "bf16 vector of 4, 8 or 16 bytes");
  static_assert(
      kLanesPerRow >= 1 && kLanesPerRow <= kWarpThreads && (kWarpThreads % kLanesPerRow) == 0,
      "lanes per row must divide the warp");
  static_assert(kUnitsPerLane >= 1 && kUnitsPerLane * kVec <= 64, "at most 64 channels per lane in registers");
  constexpr int kRowsPerWarp = kWarpThreads / kLanesPerRow;
  using Vec = AlignedVector<bf16_t, kVec>;
  const int lane = get_lane_id();
  const int sub = lane % kLanesPerRow;
  const int group = lane / kLanesPerRow;
  const int64_t channels = params.channels;
  const int units = static_cast<int>(channels / kVec);
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpThreads;
  const int64_t warp_stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
  for (int64_t row0 = warp * kRowsPerWarp; row0 < params.rows; row0 += warp_stride * kRowsPerWarp) {
    const int64_t row = row0 + group;
    // The loop bound is warp-uniform so every lane reaches the shuffles; lanes of
    // a row past the end compute on row 0 and skip the store.
    const bool active = row < params.rows;
    const int64_t base = (active ? row : 0) * channels;
    const bf16_t* __restrict__ x_row = params.x + base;
    Vec values[kUnitsPerLane];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kUnitsPerLane; ++i) {
      const int unit = sub + i * kLanesPerRow;
      if (active && unit < units) {
        values[i].load(x_row + unit * kVec);
        if constexpr (kHasBias) {
          Vec bias;
          bias.load(params.bias + unit * kVec);
#pragma unroll
          for (int j = 0; j < kVec; ++j) {
            values[i][j] = cast<bf16_t>(__fadd_rn(cast<fp32_t>(values[i][j]), cast<fp32_t>(bias[j])));
          }
        }
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
          const float v = cast<fp32_t>(values[i][j]);
          sum = __fmaf_rn(v, v, sum);
        }
      }
    }
#pragma unroll
    for (int offset = kLanesPerRow / 2; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    const float denominator = fmaxf(sqrtf(sum), kNormFloor);
    bf16_t* __restrict__ out_row = params.out + base;
#pragma unroll
    for (int i = 0; i < kUnitsPerLane; ++i) {
      const int unit = sub + i * kLanesPerRow;
      if (active && unit < units) {
        Vec gamma;
        gamma.load(params.gamma + unit * kVec);
        Vec result;
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float value = cast<fp32_t>(cast<bf16_t>(div_rn(cast<fp32_t>(values[i][j]), denominator)));
          value = cast<fp32_t>(cast<bf16_t>(__fmul_rn(value, params.scale)));
          value = cast<fp32_t>(cast<bf16_t>(__fmul_rn(value, cast<fp32_t>(gamma[j]))));
          const float normed = cast<fp32_t>(cast<bf16_t>(__fadd_rn(value, 0.0f)));
          result[j] = cast<bf16_t>(div_rn(normed, __fadd_rn(1.0f, expf(-normed))));
        }
        result.store(out_row + unit * kVec);
      }
    }
  }
}

template <int kVec, int kLanesPerRow, int kUnitsPerLane, bool kHasBias>
struct Kernel {
  /// \brief out[rows, C] = silu(rmsnorm(x[rows, C] (+ bias))) over channels-last pixel rows.
  /// \param bias per-channel bias [C]; validated but ignored unless kHasBias.
  static void
  run(tvm::ffi::TensorView out,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView gamma,
      tvm::ffi::TensorView bias,
      double scale) {
    using namespace host;
    constexpr int64_t kAlign = kVec * static_cast<int64_t>(sizeof(bf16_t));
    auto R = SymbolicSize{"rows"};
    auto C = SymbolicSize{"channels"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R, C}).with_dtype<bf16_t>().with_device(device).ensure_alignment(kAlign).verify(out).verify(x);
    TensorMatcher({C}).with_dtype<bf16_t>().with_device(device).ensure_alignment(kAlign).verify(gamma).verify(bias);
    const int64_t rows = R.unwrap(), channels = C.unwrap();
    CHECK_HOST(channels % kVec == 0) << "channels must be a multiple of " << kVec << ", got " << channels;
    CHECK_HOST(div_ceil(channels / kVec, static_cast<int64_t>(kLanesPerRow)) <= kUnitsPerLane)
        << "channels " << channels << " need more than " << kUnitsPerLane << " loads per lane";
    if (rows == 0 || channels == 0) return;
    const auto params = Params{
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .gamma = static_cast<const bf16_t*>(gamma.data_ptr()),
        .bias = static_cast<const bf16_t*>(bias.data_ptr()),
        .rows = rows,
        .channels = channels,
        .scale = static_cast<float>(scale),
    };
    constexpr int64_t kRowsPerBlock = static_cast<int64_t>(kWarpsPerBlock) * (device::kWarpThreads / kLanesPerRow);
    const auto blocks = static_cast<uint32_t>(std::min<int64_t>(div_ceil(rows, kRowsPerBlock), kMaxBlocks));
    LaunchKernel(blocks, kThreads, device.unwrap())(kernel<kVec, kLanesPerRow, kUnitsPerLane, kHasBias>, params);
  }
};

}  // namespace channel_rmsnorm_silu_nhwc

}  // namespace sglang
