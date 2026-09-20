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

SGL_DEVICE float situ_tanhf(float x) {
#if defined(__HGGC_ARCH__)
  return __ppu_tanhf(x);
#else
  return tanhf(x);
#endif
}

SGL_DEVICE float situ_sigmoidf(float x) {
#if defined(__HGGC_ARCH__)
  return __ppu_sgmdf(x);
#else
  return device::math::sigmoid_fast(x);
#endif
}

/// One SiTU element. `sigmoid_fast` is `1/(1+expf(-x))` (math.cuh), i.e. the
/// same expression both call sites used before they were folded together.
template <bool kHasLinearBeta>
SGL_DEVICE float situ_activate(float g, float u, float beta, float inv_beta, float linear_beta, float inv_linear_beta) {
  const float gate_out = beta * situ_tanhf(g * inv_beta) * situ_sigmoidf(g);
  float up_out;
  if constexpr (kHasLinearBeta) {
    up_out = linear_beta * situ_tanhf(u * inv_linear_beta);
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

struct SituMulVarlenParams {
  const bf16_t* __restrict__ input;
  bf16_t* __restrict__ output;
  const int32_t* __restrict__ masked_m;
  float beta;
  float linear_beta;
  int64_t hidden_dim;
  uint32_t num_tokens;
  uint32_t num_experts;
};

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

dim3 get_masked_grid(uint32_t num_experts, uint32_t max_masked_m) {
  constexpr uint32_t kBlocksTarget = 2048;
  auto blocks_per_expert = (kBlocksTarget + num_experts - 1) / num_experts;
  if (max_masked_m > 0 && blocks_per_expert > max_masked_m) blocks_per_expert = max_masked_m;
  return dim3(blocks_per_expert, num_experts);
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

template <bool kHasLinearBeta, bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void situ_mul_varlen_kernel(const SituMulVarlenParams __grid_constant__ params) {
  using namespace device;

  constexpr uint32_t kValuesPerThread = 8u;
  using Vec = AlignedVector<bf16x2_t, kValuesPerThread / 2>;

  const auto expert_id = blockIdx.y;
  const auto num_tokens = static_cast<uint32_t>(params.masked_m[expert_id]);
  if (blockIdx.x >= num_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  const float beta = params.beta;
  const float linear_beta = params.linear_beta;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = kHasLinearBeta ? 1.0f / linear_beta : 0.0f;

  for (uint32_t token_id = blockIdx.x; token_id < num_tokens; token_id += gridDim.x) {
    const auto offset = expert_id * params.num_tokens + token_id;
    const auto input = params.input + offset * params.hidden_dim * 2;
    const auto output = params.output + offset * params.hidden_dim;

    Vec gate_vec, up_vec;
    gate_vec.load(input, threadIdx.x);
    up_vec.load(input, threadIdx.x + blockDim.x);

    if (token_id == blockIdx.x) PDLTriggerSecondary<kUsePDL>();

    Vec out_vec;

#pragma unroll
    for (uint32_t i = 0; i < kValuesPerThread / 2; ++i) {
      const auto [g0, g1] = cast<fp32x2_t>(gate_vec[i]);
      const auto [u0, u1] = cast<fp32x2_t>(up_vec[i]);
      out_vec[i] = cast<bf16x2_t>(fp32x2_t{
          sglang::kimi_k3::situ_activate<kHasLinearBeta>(g0, u0, beta, inv_beta, linear_beta, inv_linear_beta),
          sglang::kimi_k3::situ_activate<kHasLinearBeta>(g1, u1, beta, inv_beta, linear_beta, inv_linear_beta),
      });
    }

    out_vec.store(output, threadIdx.x);
  }
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

template <bool kUsePDL>
struct SituAndMulMaskedKernel {
  static constexpr auto kernel_with_linear_beta = situ_mul_varlen_kernel<true, kUsePDL>;
  static constexpr auto kernel_without_linear_beta = situ_mul_varlen_kernel<false, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView masked_m,
      const uint32_t topk,
      const uint32_t max_masked_m,
      const double beta,
      const double linear_beta,
      const bool has_linear_beta) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto E = SymbolicSize{"num_experts"};
    auto T = SymbolicSize{"num_tokens_padded"};
    auto D = SymbolicSize{"hidden_dim x 2"};
    auto N = SymbolicSize{"hidden_dim"};
    device.set_options<kDLCUDA>();

    TensorMatcher({E, T, D})  // input
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input);
    TensorMatcher({E, T, N})  // output
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output);
    TensorMatcher({E})  // masked_m
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(masked_m);

    const auto num_experts = static_cast<uint32_t>(E.unwrap());
    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    const auto hidden_dim = static_cast<uint32_t>(N.unwrap());

    RuntimeCheck(D.unwrap() == 2 * hidden_dim, "invalid dimension");
    RuntimeCheck(hidden_dim % 8 == 0, "hidden dimension must be divisible by 8");
    RuntimeCheck(num_experts <= kMaxExperts, "num_experts exceeds maximum (256)");
    if (num_tokens == 0) return;

    const auto params = SituMulVarlenParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .output = static_cast<bf16_t*>(output.data_ptr()),
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

    const auto kernel = has_linear_beta ? kernel_with_linear_beta : kernel_without_linear_beta;
    const auto grid_m = max_masked_m > 0 ? max_masked_m : num_tokens * topk;
    LaunchKernel(get_masked_grid(num_experts, grid_m), num_threads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

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

struct SituMulQuantMxfp4Params {
  const bf16_t* __restrict__ input;
  uint8_t* __restrict__ output;
  uint8_t* __restrict__ output_scale;
  const int32_t* __restrict__ masked_m;
  float beta;
  float linear_beta;
  int64_t hidden_dim;
  uint32_t num_tokens;
  uint32_t num_experts;
};

SGL_DEVICE uint32_t pack_mxfp4(float q0, float q1, float q2, float q3, float q4, float q5, float q6, float q7) {
  uint32_t packed;
  asm volatile(
      "{\n\t"
      ".reg .b8 r0, r1, r2, r3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r0, %2, %1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r1, %4, %3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r2, %6, %5;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r3, %8, %7;\n\t"
      "mov.b32 %0, {r0, r1, r2, r3};\n\t"
      "}\n"
      : "=r"(packed)
      : "f"(q0), "f"(q1), "f"(q2), "f"(q3), "f"(q4), "f"(q5), "f"(q6), "f"(q7));
  return packed;
}

template <bool kMasked, bool kUsePDL>
__global__
__launch_bounds__(1024, 2) void situ_mul_quant_mxfp4_kernel(const SituMulQuantMxfp4Params __grid_constant__ params) {
  using namespace device;

  constexpr uint32_t kGroupSize = 32u;
  constexpr uint32_t kWorkThreads = 4u;
  constexpr uint32_t kValuesPerThread = 8u;
  using InputVec = AlignedVector<bf16x2_t, kValuesPerThread / 2>;

  const auto expert_id = kMasked ? blockIdx.y : 0u;
  const auto num_tokens = kMasked ? static_cast<uint32_t>(params.masked_m[expert_id]) : params.num_tokens;
  if (blockIdx.x >= num_tokens) return;

  const float beta = params.beta;
  const float linear_beta = params.linear_beta;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = 1.0f / linear_beta;
  const auto group_id = threadIdx.x / kWorkThreads;

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t token_id = blockIdx.x; token_id < num_tokens; token_id += gridDim.x) {
    const auto offset = expert_id * params.num_tokens + token_id;
    const auto input = params.input + offset * params.hidden_dim * 2;
    const auto output = params.output + offset * params.hidden_dim / 2;
    const auto output_scale = [&] {
      const auto num_groups = params.hidden_dim / kGroupSize;
      const auto expert_offset = kMasked ? expert_id * num_groups * params.num_tokens : 0;
      return params.output_scale + expert_offset + (group_id / 2u) * (params.num_tokens * 2u) + token_id * 2u +
             group_id % 2u;
    }();

    InputVec gate_vec, up_vec;
    gate_vec.load(input, threadIdx.x);
    up_vec.load(input, threadIdx.x + blockDim.x);

    float local_max = 0.0f;
    float results[kValuesPerThread];
#pragma unroll
    for (uint32_t i = 0; i < kValuesPerThread / 2; ++i) {
      const auto [x, y] = situ_and_mul<false>(gate_vec[i], up_vec[i], beta, inv_beta, linear_beta, inv_linear_beta);
      results[2 * i + 0] = x;
      results[2 * i + 1] = y;
      local_max = fmaxf(local_max, fmaxf(fabsf(x), fabsf(y)));
    }

    local_max = warp::reduce_max<kWorkThreads>(local_max);
    const float absmax = fmaxf(local_max, 1e-10f);
    const uint32_t scale_bits = (__float_as_uint(absmax / 6.0f) + 0x007FFFFFu) & 0x7F800000u;
    const float inv_scale = __uint_as_float(0x7F000000u - scale_bits);

    if (token_id == blockIdx.x) PDLTriggerSecondary<kUsePDL>();

    reinterpret_cast<uint32_t*>(output)[threadIdx.x] = pack_mxfp4(
        results[0] * inv_scale,
        results[1] * inv_scale,
        results[2] * inv_scale,
        results[3] * inv_scale,
        results[4] * inv_scale,
        results[5] * inv_scale,
        results[6] * inv_scale,
        results[7] * inv_scale);
    if (threadIdx.x % kWorkThreads == 0) {
      *output_scale = static_cast<uint8_t>(scale_bits >> 23);
    }
  }
}

template <bool kUsePDL>
struct SituAndMulPostQuantMxfp4Kernel {
  static constexpr auto kernel = situ_mul_quant_mxfp4_kernel<false, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const double beta,
      const double linear_beta) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto T = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_dim x 2"};
    auto N = SymbolicSize{"hidden_dim"};
    auto P = SymbolicSize{"packed_hidden_dim"};
    auto S = SymbolicSize{"num_scale_pairs"};
    device.set_options<kDLCUDA>();

    TensorMatcher({T, D}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({T, P}).with_dtype<uint8_t>().with_device(device).verify(output);
    TensorMatcher({S, T}).with_dtype<uint16_t>().with_device(device).verify(output_scale);

    N.set_value(P.unwrap() * 2);
    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    const auto hidden_dim = static_cast<uint32_t>(N.unwrap());

    RuntimeCheck(D.unwrap() == 2 * hidden_dim, "invalid dimension");
    RuntimeCheck(hidden_dim % 32 == 0, "hidden dimension must be divisible by 32");
    RuntimeCheck(S.unwrap() * 64 == hidden_dim, "invalid scale dimension");
    if (num_tokens == 0) return;

    const auto params = SituMulQuantMxfp4Params{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .output = static_cast<uint8_t*>(output.data_ptr()),
        .output_scale = static_cast<uint8_t*>(output_scale.data_ptr()),
        .masked_m = nullptr,
        .beta = static_cast<float>(beta),
        .linear_beta = static_cast<float>(linear_beta),
        .hidden_dim = hidden_dim,
        .num_tokens = num_tokens,
        .num_experts = 1,
    };

    const auto num_threads = hidden_dim / 8;
    RuntimeCheck(num_threads % device::kWarpThreads == 0);
    LaunchKernel(num_tokens, num_threads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

template <bool kUsePDL>
struct SituAndMulMaskedPostQuantMxfp4Kernel {
  static constexpr auto kernel = situ_mul_quant_mxfp4_kernel<true, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const tvm::ffi::TensorView masked_m,
      const uint32_t topk,
      const uint32_t max_masked_m,
      const double beta,
      const double linear_beta) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto E = SymbolicSize{"num_experts"};
    auto T = SymbolicSize{"num_tokens_padded"};
    auto D = SymbolicSize{"hidden_dim x 2"};
    auto N = SymbolicSize{"hidden_dim"};
    auto P = SymbolicSize{"packed_hidden_dim"};
    auto S = SymbolicSize{"num_scale_pairs"};
    device.set_options<kDLCUDA>();

    TensorMatcher({E, T, D}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({E, T, P}).with_dtype<uint8_t>().with_device(device).verify(output);
    TensorMatcher({E, S, T}).with_dtype<uint16_t>().with_device(device).verify(output_scale);
    TensorMatcher({E}).with_dtype<int32_t>().with_device(device).verify(masked_m);

    N.set_value(P.unwrap() * 2);
    const auto num_experts = static_cast<uint32_t>(E.unwrap());
    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    const auto hidden_dim = static_cast<uint32_t>(N.unwrap());

    RuntimeCheck(D.unwrap() == 2 * hidden_dim, "invalid dimension");
    RuntimeCheck(hidden_dim % 32 == 0, "hidden dimension must be divisible by 32");
    RuntimeCheck(S.unwrap() * 64 == hidden_dim, "invalid scale dimension");
    RuntimeCheck(num_experts <= kMaxExperts, "num_experts exceeds maximum (256)");
    if (num_tokens == 0) return;

    const auto params = SituMulQuantMxfp4Params{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .output = static_cast<uint8_t*>(output.data_ptr()),
        .output_scale = static_cast<uint8_t*>(output_scale.data_ptr()),
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
    const auto grid_m = max_masked_m > 0 ? max_masked_m : num_tokens * topk;
    LaunchKernel(get_masked_grid(num_experts, grid_m), num_threads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
