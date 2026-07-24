#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <cstdint>
#include <cuda_fp8.h>
#include <type_traits>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::pack_fp8;

struct SituMulQuantVarlenParams {
  const bf16_t* __restrict__ input;
  fp8_e4m3_t* __restrict__ output;
  float* __restrict__ output_scale;
  const int32_t* __restrict__ masked_m;
  float beta;         // gate softcap (e.g. 4.0)
  float linear_beta;  // up softcap (e.g. 25.0)
  int64_t hidden_dim;
  uint32_t num_tokens;
  uint32_t num_experts;
};

constexpr uint32_t kMaxExperts = 256;

struct alignas(16) CTAWork {
  uint32_t expert_id;
  uint32_t expert_token_id;
  bool valid;
};

SGL_DEVICE uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
  static_assert(device::kWarpThreads == 32);
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

// SiTU (SoftCap-GLU) activation:
//   gate_out = beta * tanh(gate / beta) * sigmoid(gate)
//   up_out   = linear_beta * tanh(up / linear_beta)
//   output   = gate_out * up_out
// Unlike SiLU, no external swiglu_limit clamp is needed: the tanh softcap
// inherently bounds the output to |beta * linear_beta| (< FP8_E4M3_MAX).
template <bool kPrecise = true, typename DType2>
SGL_DEVICE fp32x2_t
situ_and_mul(DType2 gate, DType2 up, float beta, float inv_beta, float linear_beta, float inv_linear_beta) {
  using namespace device;
  const auto [g0, g1] = cast<fp32x2_t>(gate);
  const auto [u0, u1] = cast<fp32x2_t>(up);
  const float gate_out0 = beta * tanhf(g0 * inv_beta) * math::sigmoid_fast(g0);
  const float gate_out1 = beta * tanhf(g1 * inv_beta) * math::sigmoid_fast(g1);
  const float up_out0 = linear_beta * tanhf(u0 * inv_linear_beta);
  const float up_out1 = linear_beta * tanhf(u1 * inv_linear_beta);
  const float val0 = gate_out0 * up_out0;
  const float val1 = gate_out1 * up_out1;
  if constexpr (kPrecise) {
    return {val0, val1};
  } else {
    return cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{val0, val1}));
  }
}

[[maybe_unused]]
SGL_DEVICE CTAWork get_work(const SituMulQuantVarlenParams& params) {
  // Preconditions:
  // 1. blockDim.x >= params.num_experts
  // 2. params.num_experts <= kMaxExperts
  using namespace device;
  static_assert(kWarpThreads == 32);

  static __shared__ uint32_t s_warp_sum[32];
  static __shared__ CTAWork result;

  result.valid = false;

  const uint32_t tx = threadIdx.x;
  const uint32_t lane_id = tx % kWarpThreads;
  const uint32_t warp_id = tx / kWarpThreads;

  const uint32_t val = tx < params.num_experts ? params.masked_m[tx] : 0u;

  // Per-warp inclusive scan of masked_m.
  const uint32_t warp_inclusive = warp_inclusive_sum(lane_id, val);
  const uint32_t warp_exclusive = warp_inclusive - val;

  // Write each warp total.
  if (lane_id == kWarpThreads - 1) s_warp_sum[warp_id] = warp_inclusive;
  __syncthreads();
  const auto tmp_val = lane_id < warp_id ? s_warp_sum[lane_id] : 0u;
  const auto prefix_exclusive = warp::reduce_sum(tmp_val) + warp_exclusive;
  const auto bx = blockIdx.x;
  if (prefix_exclusive <= bx && bx < prefix_exclusive + val) {
    result = {tx, bx - prefix_exclusive, true};
  }
  __syncthreads();
  return result;
}

template <bool kScaleUE8M0, bool kTransposed, bool kSwizzle, bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void  // maximize occupancy
    situ_mul_quant_varlen_kernel(const SituMulQuantVarlenParams __grid_constant__ params) {
  using namespace device;

  constexpr uint32_t kGroupSize = 128u;
  constexpr uint32_t kWorkThreads = 16u;
  // each thread will handle 8 elements
  using InputVec = AlignedVector<bf16x2_t, 4>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, 4>;
  static_assert(8 * kWorkThreads == 128, "Invalid tiling");
  static_assert(!(kTransposed && !kScaleUE8M0), "transposed layout only supports ue8m0");

  const auto [expert_id, token_id, valid] = get_work(params);

  if (!valid) return;

  const auto work_id = threadIdx.x / kWorkThreads;

  const auto offset = expert_id * params.num_tokens + token_id;
  const auto input = params.input + offset * params.hidden_dim * 2;
  const auto output = params.output + offset * params.hidden_dim;
  [[maybe_unused]]
  const auto output_scale = [&] {
    const auto num_groups = params.hidden_dim / kGroupSize;
    if constexpr (kTransposed) {
      const auto base = reinterpret_cast<uint8_t*>(params.output_scale);
      // Physical layout is [E, G//4, N] int32.  Each int32 packs 4 consecutive
      // group scales for the same token, so the byte address is:
      //   expert_offset + (group/4)*N*4 + token*4 + group%4
      return base + expert_id * num_groups * params.num_tokens + (work_id / 4u) * (params.num_tokens * 4u) +
             token_id * 4u + (work_id % 4u);
    } else {
      return params.output_scale + offset * num_groups + work_id;
    }
  }();

  const float beta = params.beta;
  const float linear_beta = params.linear_beta;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = 1.0f / linear_beta;

  PDLWaitPrimary<kUsePDL>();

  InputVec gate_vec, up_vec;
  if constexpr (kSwizzle) {
    // gran=8 interleaved: every 16-element chunk on the N axis is
    // [gate[0..7], up[0..7]]. Each thread handles 8 consecutive output
    // elements, so its gate chunk lives at vec index 2*threadIdx.x and its
    // up chunk at 2*threadIdx.x+1.
    gate_vec.load(input, threadIdx.x * 2);
    up_vec.load(input, threadIdx.x * 2 + 1);
  } else {
    gate_vec.load(input, threadIdx.x);
    up_vec.load(input, threadIdx.x + blockDim.x);
  }

  float local_max = 0.0f;
  float results[8];

#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const auto [x, y] = situ_and_mul(gate_vec[i], up_vec[i], beta, inv_beta, linear_beta, inv_linear_beta);
    results[2 * i + 0] = x;
    results[2 * i + 1] = y;
    local_max = fmaxf(local_max, fmaxf(fabsf(x), fabsf(y)));
  }

  local_max = warp::reduce_max<kWorkThreads>(local_max);

  const float absmax = fmaxf(local_max, 1e-10f);
  float scale;
  uint32_t ue8m0_exp;

  if constexpr (kScaleUE8M0) {
    const float raw_scale = absmax / math::FP8_E4M3_MAX;
    ue8m0_exp = cast_to_ue8m0(raw_scale);
    scale = __uint_as_float(ue8m0_exp << 23);
  } else {
    scale = absmax / math::FP8_E4M3_MAX;
  }
  const auto inv_scale = 1.0f / scale;

  OutputVec out_vec;
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const float scaled_val0 = results[2 * i + 0] * inv_scale;
    const float scaled_val1 = results[2 * i + 1] * inv_scale;
    out_vec[i] = pack_fp8(scaled_val0, scaled_val1);
  }

  PDLTriggerSecondary<kUsePDL>();

  out_vec.store(output, threadIdx.x);
  if constexpr (kTransposed) {
    *output_scale = ue8m0_exp;
  } else {
    *output_scale = scale;
  }
}

// ---- Host wrapper
// ------------------------------------------------------------------------------------------------------------------------

template <int64_t kGroupSize, bool kScaleUE8M0, bool kSwizzle, bool kUsePDL>
struct SituAndMulMaskedPostQuantKernel {
  static_assert(kGroupSize == 128);
  static constexpr auto kernel_normal = situ_mul_quant_varlen_kernel<kScaleUE8M0, false, kSwizzle, kUsePDL>;
  static constexpr auto kernel_transposed = situ_mul_quant_varlen_kernel<true, true, kSwizzle, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const tvm::ffi::TensorView masked_m,
      const uint32_t topk,
      const bool transposed,
      const double beta,
      const double linear_beta) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto E = SymbolicSize{"num_experts"};
    auto T = SymbolicSize{"num_tokens_padded"};
    auto D = SymbolicSize{"hidden_dim x 2"};
    auto N = SymbolicSize{"hidden_dim"};
    auto G = SymbolicSize{"num_groups"};
    device.set_options<kDLCUDA>();

    TensorMatcher({E, T, D})  // input
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input);
    TensorMatcher({E, T, N})  // output
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(output);
    if (!transposed) {
      TensorMatcher({E, T, G})  //
          .with_dtype<fp32_t>()
          .with_device(device)
          .verify(output_scale);
    } else {
      RuntimeCheck(kScaleUE8M0, "transposed layout only supports scale_ue8m0=true");
      auto G_ = SymbolicSize{"G // 4"};
      TensorMatcher({E, G_, T})  //
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(output_scale);
      G.set_value(G_.unwrap() * 4);
    }
    TensorMatcher({E})  //
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(masked_m);

    const auto num_experts = static_cast<uint32_t>(E.unwrap());
    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    const auto num_groups = static_cast<uint32_t>(G.unwrap());
    const auto hidden_dim = N.unwrap();

    RuntimeCheck(D.unwrap() == 2 * hidden_dim, "invalid dimension");
    RuntimeCheck(hidden_dim % kGroupSize == 0);
    RuntimeCheck(num_experts <= kMaxExperts, "num_experts exceeds maximum (256)");
    RuntimeCheck(num_groups * kGroupSize == hidden_dim, "invalid num_groups");

    const auto params = SituMulQuantVarlenParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .output = static_cast<fp8_e4m3_t*>(output.data_ptr()),
        .output_scale = static_cast<float*>(output_scale.data_ptr()),
        .masked_m = static_cast<const int32_t*>(masked_m.data_ptr()),
        .beta = static_cast<float>(beta),
        .linear_beta = static_cast<float>(linear_beta),
        .hidden_dim = hidden_dim,
        .num_tokens = num_tokens,
        .num_experts = num_experts,
    };

    const auto num_threads = hidden_dim / 8;
    RuntimeCheck(num_threads % device::kWarpThreads == 0);
    RuntimeCheck(num_threads >= num_experts);
    const auto kernel = transposed ? kernel_transposed : kernel_normal;
    LaunchKernel(num_tokens * topk, num_threads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
