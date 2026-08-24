// Kimi K3 SiTU activation kernels: plain elementwise and varlen masked with a
// grouped-quant epilogue. The shared double-softcap activation is inlined below.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>   // For dtype_trait, bf16_t, fp32_t, cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, PDL helpers
#include <sgl_kernel/vec.cuh>    // For AlignedVector
#include <sgl_kernel/warp.cuh>   // For warp::copy_bytes, elect_one_lane, inclusive_sum

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>
#include <type_traits>
#ifndef USE_ROCM
#include <cuda_fp8.h>
#endif

namespace sglang {

namespace kimi_k3 {

/// One SiTU element. `sigmoid_fast` is `1/(1+expf(-x))` (math.cuh), i.e. the
/// same expression both call sites used before they were folded together.
template <bool kHasLinearBeta>
SGL_DEVICE float situ_activate(float g, float u, float beta, float inv_beta, float linear_beta, float inv_linear_beta) {
  const float gate_out = beta * tanhf(g * inv_beta) * device::math::sigmoid_fast(g);
  float up_out;
  if constexpr (kHasLinearBeta) {
    up_out = linear_beta * tanhf(u * inv_linear_beta);
  } else {
    up_out = u;
  }
  return gate_out * up_out;
}

}  // namespace kimi_k3

// SiTU (SoftCap-GLU) activation:
//   gate_out = beta * tanh(gate / beta) * sigmoid(gate)
//   up_out   = linear_beta * tanh(up / linear_beta)
//   output   = gate_out * up_out
//
// Input: bf16 tensor [N, 2*D] (gate = [:, :D], up = [:, D:])
// Output: bf16 tensor [N, D]

struct SituAndMulParams {
  const void* __restrict__ input;
  void* __restrict__ out;
  float beta;
  float inv_beta;
  float linear_beta;
  float inv_linear_beta;
  uint32_t hidden_dim;  // D (output width, half of input last dim)
  uint32_t num_tokens;
  uint32_t stride_in_vecs;  // input row stride in vector units (2*D/vec if dense)
};

template <typename TIn, typename TOut, bool kHasLinearBeta, bool kUsePDL>
__global__ void situ_and_mul_kernel(const __grid_constant__ SituAndMulParams params) {
  using namespace device;
  constexpr auto kWidest = sizeof(TIn) > sizeof(TOut) ? sizeof(TIn) : sizeof(TOut);
  constexpr auto kVecSize = kMaxVecBytes / kWidest;
  using vec_t = AlignedVector<TIn, kVecSize>;
  using out_vec_t = AlignedVector<TOut, kVecSize>;

  const auto num_vecs = params.hidden_dim / kVecSize;  // per token
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto token_id = tid / num_vecs;

  if (token_id >= params.num_tokens) return;

  const auto offset = tid % num_vecs;
  // Input rows may be strided (e.g. a slice of a wider fused-GEMM output);
  // within a row: gate = [0..D-1], up = [D..2D-1].
  const auto input_offset = static_cast<uint64_t>(token_id) * params.stride_in_vecs + offset;
  const auto output_offset = tid;

  PDLWaitPrimary<kUsePDL>();

  const auto gate = load_as<vec_t>(params.input, input_offset);
  const auto up = load_as<vec_t>(params.input, input_offset + num_vecs);

  PDLTriggerSecondary<kUsePDL>();

  const float beta = params.beta;
  const float inv_beta = params.inv_beta;
  const float linear_beta = params.linear_beta;
  const float inv_linear_beta = params.inv_linear_beta;

  out_vec_t out;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    const float g = cast<fp32_t>(gate[i]);
    const float u = cast<fp32_t>(up[i]);

    out[i] = cast<TOut>(kimi_k3::situ_activate<kHasLinearBeta>(g, u, beta, inv_beta, linear_beta, inv_linear_beta));
  }

  store_as<out_vec_t>(params.out, out, output_offset);
}

// Host launcher

template <typename TIn, typename TOut, bool kUsePDL>
struct SituAndMulKernel {
  static constexpr auto kWidest = sizeof(TIn) > sizeof(TOut) ? sizeof(TIn) : sizeof(TOut);
  static constexpr auto kVecSize = device::kMaxVecBytes / kWidest;
  static constexpr auto kBlockSize = 256u;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView out,
      const double beta,
      const double linear_beta,
      const bool has_linear_beta) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto D_in = SymbolicSize{"input_width"};
    auto D_out = SymbolicSize{"output_width"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D_out})  //
        .with_dtype<TOut>()
        .with_device(device_)
        .verify(out);
    TensorMatcher({N, D_in})  //
        .with_dtype<TIn>()
        .with_device(device_)
        .with_strides({-1, 1})
        .verify(input);

    const auto hidden_size = static_cast<uint32_t>(D_out.unwrap());
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto device = device_.unwrap();

    if (num_tokens == 0) return;
    RuntimeCheck(hidden_size * 2 == D_in.unwrap(), "invalid activation dimension: D_out * 2 != D_in");
    RuntimeCheck(hidden_size % kVecSize == 0, "hidden size must be divisible by vector size");
    RuntimeCheck(input.stride(0) % kVecSize == 0, "input row stride must be divisible by vector size");

    const auto num_total_items = num_tokens * (hidden_size / kVecSize);
    RuntimeCheck(num_total_items <= std::numeric_limits<uint32_t>::max(), "too many items for 32-bit indexing");

    const auto num_blocks = div_ceil(static_cast<uint32_t>(num_total_items), kBlockSize);
    const float beta_f = static_cast<float>(beta);
    const float linear_beta_f = static_cast<float>(linear_beta);

    const auto params = SituAndMulParams{
        .input = input.data_ptr(),
        .out = out.data_ptr(),
        .beta = beta_f,
        .inv_beta = 1.0f / beta_f,
        .linear_beta = linear_beta_f,
        .inv_linear_beta = linear_beta_f != 0.0f ? 1.0f / linear_beta_f : 0.0f,
        .hidden_dim = hidden_size,
        .num_tokens = num_tokens,
        .stride_in_vecs = static_cast<uint32_t>(input.stride(0) / kVecSize),
    };

    if (has_linear_beta) {
      LaunchKernel(num_blocks, kBlockSize, device)
          .enable_pdl(kUsePDL)(situ_and_mul_kernel<TIn, TOut, true, kUsePDL>, params);
    } else {
      LaunchKernel(num_blocks, kBlockSize, device)
          .enable_pdl(kUsePDL)(situ_and_mul_kernel<TIn, TOut, false, kUsePDL>, params);
    }
  }
};

// ---------------------------------------------------------------------------
// varlen masked variant with the grouped-quant epilogue. Same activation, a
// different kernel: __launch_bounds__(1024, 2) plus a per-group scale writeback.
// ---------------------------------------------------------------------------
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
  // kHasLinearBeta=true: this path always softcaps the up operand, as before.
  const float val0 = kimi_k3::situ_activate<true>(g0, u0, beta, inv_beta, linear_beta, inv_linear_beta);
  const float val1 = kimi_k3::situ_activate<true>(g1, u1, beta, inv_beta, linear_beta, inv_linear_beta);
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
  const uint32_t warp_inclusive = device::warp::inclusive_sum(lane_id, val);
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

}  // namespace sglang
