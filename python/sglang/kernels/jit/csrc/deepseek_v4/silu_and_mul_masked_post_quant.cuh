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

struct SiluMulQuantVarlenParams {
  const bf16_t* __restrict__ input;
  fp8_e4m3_t* __restrict__ output;
  float* __restrict__ output_scale;
  const int32_t* __restrict__ masked_m;
  float swiglu_limit;  // only read when kApplySwigluLimit=true
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

template <bool kApplySwigluLimit, bool kPrecise = true, typename DType2>
SGL_DEVICE fp32x2_t silu_and_mul(DType2 gate, DType2 up, float limit) {
  using namespace device;
  // refer to as implementation. TL;DR: must clamp in bf16
  // https://github.com/deepseek-ai/DeepGEMM/blob/7f2a703ed51ac1f7af07f5e1453b2d3267d37d50/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L984-L997
  if constexpr (kApplySwigluLimit) {
    static_assert(std::is_same_v<DType2, bf16x2_t>);
    gate = __hmin2(gate, {limit, limit});
    up = __hmax2(up, {-limit, -limit});
    up = __hmin2(up, {limit, limit});
  }
  const auto [g0, g1] = cast<fp32x2_t>(gate);
  const auto [u0, u1] = cast<fp32x2_t>(up);
  const auto silu0 = g0 / (1.0f + __expf(-g0));
  const auto silu1 = g1 / (1.0f + __expf(-g1));
  const float val0 = silu0 * u0;
  const float val1 = silu1 * u1;
  if constexpr (kPrecise) {  // I don't know if we should enable this?
    return {val0, val1};
  } else {
    return cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{val0, val1}));
  }
}

[[maybe_unused]]
SGL_DEVICE CTAWork get_work(const SiluMulQuantVarlenParams& params) {
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

template <bool kScaleUE8M0, bool kTransposed, bool kSwizzle, bool kUsePDL, bool kApplySwigluLimit>
__global__ __launch_bounds__(1024, 2) void  // maximize occupancy
    silu_mul_quant_varlen_kernel(const SiluMulQuantVarlenParams __grid_constant__ params) {
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
    const auto [x, y] = silu_and_mul<kApplySwigluLimit>(gate_vec[i], up_vec[i], params.swiglu_limit);
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

struct SiluAndMulClampParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  float swiglu_limit;
};

template <typename DType, bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void  // maximize occupancy
    silu_mul_clamp_kernel(const SiluAndMulClampParams __grid_constant__ params) {
  using namespace device;
  static_assert(sizeof(DType) == 2, "only fp16/bf16 supported");
  using DType2 = packed_t<DType>;
  constexpr auto kVecSize = 16 / sizeof(DType);
  static_assert(kVecSize % 2 == 0 && kVecSize > 0);
  using Vec = AlignedVector<DType2, kVecSize / 2>;
  const auto bid = blockIdx.x;
  const auto tile = tile::Memory<Vec>::cta();
  const float limit = params.swiglu_limit;

  PDLWaitPrimary<kUsePDL>();
  const auto gate = tile.load(params.input, bid * 2 + 0);
  const auto up = tile.load(params.input, bid * 2 + 1);
  Vec out;

#pragma unroll
  for (uint32_t i = 0; i < kVecSize / 2; ++i) {
    out[i] = cast<DType2>(silu_and_mul<true>(cast<bf16x2_t>(gate[i]), cast<bf16x2_t>(up[i]), limit));
  }

  tile.store(params.output, out, bid);
  PDLTriggerSecondary<kUsePDL>();
}

// ---- Host wrapper
// ------------------------------------------------------------------------------------------------------------------------

template <int64_t kGroupSize, bool kScaleUE8M0, bool kSwizzle, bool kUsePDL, bool kApplySwigluLimit>
struct SiluAndMulMaskedPostQuantKernel {
  static_assert(kGroupSize == 128);
  static constexpr auto kernel_normal =
      silu_mul_quant_varlen_kernel<kScaleUE8M0, false, kSwizzle, kUsePDL, kApplySwigluLimit>;
  static constexpr auto kernel_transposed =
      silu_mul_quant_varlen_kernel<true, true, kSwizzle, kUsePDL, kApplySwigluLimit>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const tvm::ffi::TensorView masked_m,
      const uint32_t topk,
      const bool transposed,
      const double swiglu_limit) {
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

    const auto params = SiluMulQuantVarlenParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .output = static_cast<fp8_e4m3_t*>(output.data_ptr()),
        .output_scale = static_cast<float*>(output_scale.data_ptr()),
        .masked_m = static_cast<const int32_t*>(masked_m.data_ptr()),
        .swiglu_limit = static_cast<float>(swiglu_limit),
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

template <typename DType, bool kUsePDL>
struct SiluAndMulClampKernel {
  static constexpr auto kernel = silu_mul_clamp_kernel<DType, kUsePDL>;

  static void run(const tvm::ffi::TensorView input, const tvm::ffi::TensorView output, const double swiglu_limit) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"gate_up_dim"};  // 2 * out_dim
    auto H = SymbolicSize{"out_dim"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, D})  // input  (gate || up)
        .with_dtype<DType>()
        .with_device(device)
        .verify(input);
    TensorMatcher({M, H})  // output
        .with_dtype<DType>()
        .with_device(device)
        .verify(output);
    RuntimeCheck(D.unwrap() == 2 * H.unwrap(), "input last dim must be 2 * output last dim");

    constexpr uint32_t kVecSize = 16 / sizeof(DType);
    const auto out_dim = static_cast<uint32_t>(H.unwrap());
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    RuntimeCheck(out_dim % kVecSize == 0, "out_dim must be divisible by vector size");
    const auto num_threads = out_dim / kVecSize;
    RuntimeCheck(num_threads <= 1024, "out_dim too large for single-block-per-row launch");

    const auto params = SiluAndMulClampParams{
        .input = input.data_ptr(),
        .output = output.data_ptr(),
        .swiglu_limit = static_cast<float>(swiglu_limit),
    };
    LaunchKernel(num_tokens, num_threads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

struct SiluMulQuantContigParams {
  const bf16_t* __restrict__ input;
  fp8_e4m3_t* __restrict__ output;
  float* __restrict__ output_scale;
  float swiglu_limit;  // only read when kApplySwigluLimit=true
  int64_t hidden_dim;
  uint32_t num_tokens;
  uint32_t scale_row_stride_int32;  // only used when kTransposed=true
};

template <bool kScaleUE8M0, bool kTransposed, bool kSwizzle, bool kUsePDL, bool kApplySwigluLimit>
__global__ __launch_bounds__(1024, 2) void  // maximize occupancy
    silu_mul_quant_contig_kernel(const SiluMulQuantContigParams __grid_constant__ params) {
  using namespace device;

  constexpr uint32_t kGroupSize = 128u;
  constexpr uint32_t kWorkThreads = 16u;
  using InputVec = AlignedVector<bf16x2_t, 4>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, 4>;
  static_assert(8 * kWorkThreads == 128, "Invalid tiling");
  static_assert(!(kTransposed && !kScaleUE8M0), "transposed layout only supports ue8m0");

  const auto token_id = blockIdx.x;
  const auto work_id = threadIdx.x / kWorkThreads;

  const auto input = params.input + token_id * params.hidden_dim * 2;
  const auto output = params.output + token_id * params.hidden_dim;
  [[maybe_unused]]
  const auto output_scale = [&] {
    const auto num_groups = params.hidden_dim / kGroupSize;
    if constexpr (kTransposed) {
      // Physical layout is (G//4_pad, M_pad) int32; each int32 packs 4
      // consecutive UE8M0 exponents for the same token. Byte address:
      //   (work_id / 4) * M_pad * 4  +  token * 4  +  (work_id % 4).
      const auto base = reinterpret_cast<uint8_t*>(params.output_scale);
      return base + (work_id / 4u) * (params.scale_row_stride_int32 * 4u) + token_id * 4u + (work_id % 4u);
    } else {
      return params.output_scale + token_id * num_groups + work_id;
    }
  }();

  PDLWaitPrimary<kUsePDL>();

  InputVec gate_vec, up_vec;
  if constexpr (kSwizzle) {
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
    const auto [x, y] = silu_and_mul<kApplySwigluLimit>(gate_vec[i], up_vec[i], params.swiglu_limit);
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

template <int64_t kGroupSize, bool kScaleUE8M0, bool kSwizzle, bool kUsePDL, bool kApplySwigluLimit>
struct SiluAndMulContigPostQuantKernel {
  static_assert(kGroupSize == 128);
  static constexpr auto kernel_normal =
      silu_mul_quant_contig_kernel<kScaleUE8M0, false, kSwizzle, kUsePDL, kApplySwigluLimit>;
  static constexpr auto kernel_transposed =
      silu_mul_quant_contig_kernel<true, true, kSwizzle, kUsePDL, kApplySwigluLimit>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const bool transposed,
      const double swiglu_limit) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_dim x 2"};
    auto N = SymbolicSize{"hidden_dim"};
    auto G = SymbolicSize{"num_groups"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, D})  // input (gate/up, natural or gran=8 interleaved on last dim)
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input);
    TensorMatcher({M, N})  // fp8 output
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(output);

    const auto hidden_dim = N.unwrap();
    RuntimeCheck(D.unwrap() == 2 * hidden_dim, "invalid dimension");
    RuntimeCheck(hidden_dim % kGroupSize == 0);
    const auto num_groups = static_cast<uint32_t>(hidden_dim / kGroupSize);

    uint32_t scale_row_stride_int32 = 0;
    if (!transposed) {
      G.set_value(num_groups);
      TensorMatcher({M, G})  // (M, G) fp32 natural row-major
          .with_dtype<fp32_t>()
          .with_device(device)
          .verify(output_scale);
    } else {
      RuntimeCheck(kScaleUE8M0, "transposed layout only supports scale_ue8m0=true");
      RuntimeCheck(num_groups % 4 == 0, "transposed layout requires num_groups % 4 == 0");
      auto G_ = SymbolicSize{"G // 4"};
      G_.set_value(num_groups / 4);
      auto M_pad = SymbolicSize{"M padded"};
      TensorMatcher({M, G_})                  // `.transpose(-1,-2)[:M,:]` view of (G//4_pad, M_pad) int32
          .with_strides({int64_t{1}, M_pad})  // col-major transposed
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(output_scale);
      scale_row_stride_int32 = static_cast<uint32_t>(M_pad.unwrap());
    }

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());

    const auto params = SiluMulQuantContigParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .output = static_cast<fp8_e4m3_t*>(output.data_ptr()),
        .output_scale = static_cast<float*>(output_scale.data_ptr()),
        .swiglu_limit = static_cast<float>(swiglu_limit),
        .hidden_dim = hidden_dim,
        .num_tokens = num_tokens,
        .scale_row_stride_int32 = scale_row_stride_int32,
    };

    const auto num_threads = hidden_dim / 8;
    RuntimeCheck(num_threads % device::kWarpThreads == 0);
    const auto kernel = transposed ? kernel_transposed : kernel_normal;
    LaunchKernel(num_tokens, num_threads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
