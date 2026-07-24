// CUDA port of the triton fused_recurrent_kda_packed_decode_kernel for
// batched decode. The triton kernel holds a whole [BV, K] fp32 state tile in
// the registers of a single warp, which caps it at ~5 TB/s of the ~9.6 TB/s
// this in-place read+write stream can reach (probe: torch inplace mul_).
// This kernel streams the state row by row instead: one warp per V-row group,
// each row is a 512B float4 load -> warp-reduced dot -> decayed delta-rule
// update -> 512B store, so loads pipeline across rows and nothing holds a
// tile. Setup (l2norm'd q/k, per-K decay, beta) is computed redundantly per
// warp - the kernel has no __syncthreads at all.
//
// Math follows the triton kernel exactly (fp32 throughout, same op order):
//   h *= exp(g);  t = <h, k>;  v = (v - t) * sigmoid(b);  h += v * k;
//   o = <h, q>
// with g = -exp(A_log) * softplus(a + dt_bias) (K3: no lower bound) or
// lower_bound * sigmoid(exp(A_log) * (a + dt_bias)). Warp-shuffle reduction
// order differs from tl.sum, so outputs match to ULPs, not bits (validated
// against the triton kernel with tolerance + GSM8K).

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct KdaPackedDecodeParams {
  const bf16_t* __restrict__ mixed_qkv;  // [B, 2*H*K + HV*V]
  const bf16_t* __restrict__ a;          // [B, HV*K]
  const bf16_t* __restrict__ b;          // [B, HV]
  const fp32_t* __restrict__ A_log;      // [HV]
  const fp32_t* __restrict__ dt_bias;    // [HV*K]
  bf16_t* __restrict__ o;                // [B, HV*V] (contiguous view)
  fp32_t* __restrict__ state;            // pool, row stride = stride_state
  const int32_t* __restrict__ indices;   // [B]
  int64_t stride_mixed;
  int64_t stride_a;
  int64_t stride_b;
  int64_t stride_state;  // elements per pool slot
  uint32_t H;
  uint32_t HV;
  fp32_t scale;
  fp32_t lower_bound;
  int32_t use_lower_bound;
};

__device__ __forceinline__ float warp_allreduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

// K = V = 128 specialization: one lane owns 4 consecutive K-elements (16B).
template <int kWarps, bool kUsePDL>
__global__
__launch_bounds__(kWarps * 32) void kda_packed_decode_kernel(const KdaPackedDecodeParams __grid_constant__ params) {
  using namespace device;
  constexpr int K = 128;
  constexpr int V = 128;
  constexpr int kElems = 4;  // K / 32 lanes

  const uint32_t i_nh = blockIdx.x;
  const uint32_t n = i_nh / params.HV;
  const uint32_t hv = i_nh % params.HV;
  const uint32_t i_h = hv / (params.HV / params.H);
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;

  PDLWaitPrimary<kUsePDL>();

  bf16_t* o_ptr = params.o + (static_cast<int64_t>(n) * params.HV + hv) * V;
  const int64_t sidx = params.indices[n];
  if (sidx < 0) {
    // Padded cuda-graph slot: zero the output, leave the pool untouched.
    for (uint32_t i = threadIdx.x; i < V; i += kWarps * 32) {
      o_ptr[i] = cast<bf16_t>(0.0f);
    }
    PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // --- per-warp redundant setup (no cross-warp synchronization) ---
  const bf16_t* mixed = params.mixed_qkv + n * params.stride_mixed;
  const uint32_t e0 = lane * kElems;

  float q[kElems], k[kElems];
  float q_sq = 0.0f, k_sq = 0.0f;
#pragma unroll
  for (int e = 0; e < kElems; ++e) {
    q[e] = cast<fp32_t>(mixed[i_h * K + e0 + e]);
    k[e] = cast<fp32_t>(mixed[params.H * K + i_h * K + e0 + e]);
    q_sq += q[e] * q[e];
    k_sq += k[e] * k[e];
  }
  // tl: q / sqrt(sum(q*q) + 1e-6), then * scale
  const float q_inv = 1.0f / sqrtf(warp_allreduce_sum(q_sq) + 1e-6f);
  const float k_inv = 1.0f / sqrtf(warp_allreduce_sum(k_sq) + 1e-6f);
#pragma unroll
  for (int e = 0; e < kElems; ++e) {
    q[e] = q[e] * q_inv * params.scale;
    k[e] = k[e] * k_inv;
  }

  const float exp_A = expf(params.A_log[hv]);
  float decay[kElems];
#pragma unroll
  for (int e = 0; e < kElems; ++e) {
    const float x = cast<fp32_t>(params.a[n * params.stride_a + hv * K + e0 + e]) + params.dt_bias[hv * K + e0 + e];
    float g;
    if (params.use_lower_bound) {
      g = params.lower_bound / (1.0f + expf(-exp_A * x));
    } else {
      const float sp = (x <= 20.0f) ? logf(1.0f + expf(x)) : x;
      g = -exp_A * sp;
    }
    decay[e] = expf(g);
  }
  const float beta = 1.0f / (1.0f + expf(-cast<fp32_t>(params.b[n * params.stride_b + hv])));

  const bf16_t* v_ptr = mixed + 2 * params.H * K + hv * V;
  fp32_t* h_base = params.state + sidx * params.stride_state + static_cast<int64_t>(hv) * V * K;

  // --- stream this warp's V-rows: 512B load -> update -> 512B store ---
  constexpr int kRowsPerWarp = V / kWarps;
#pragma unroll 4
  for (int r = warp * kRowsPerWarp; r < (int)(warp + 1) * kRowsPerWarp; ++r) {
    float4 h4 = *reinterpret_cast<const float4*>(h_base + r * K + e0);
    float h[kElems] = {h4.x, h4.y, h4.z, h4.w};
    float t = 0.0f;
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      h[e] *= decay[e];
      t += h[e] * k[e];
    }
    t = warp_allreduce_sum(t);
    const float v_new = (cast<fp32_t>(v_ptr[r]) - t) * beta;
    float o_acc = 0.0f;
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      h[e] += v_new * k[e];
      o_acc += h[e] * q[e];
    }
    o_acc = warp_allreduce_sum(o_acc);
    *reinterpret_cast<float4*>(h_base + r * K + e0) = make_float4(h[0], h[1], h[2], h[3]);
    if (lane == 0) {
      o_ptr[r] = cast<bf16_t>(o_acc);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int kWarps, bool kUsePDL>
struct KdaPackedDecodeKernel {
  static constexpr auto kernel = kda_packed_decode_kernel<kWarps, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView mixed_qkv,
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView b,
      const tvm::ffi::TensorView A_log,
      const tvm::ffi::TensorView dt_bias,
      const tvm::ffi::TensorView o,
      const tvm::ffi::TensorView state,
      const tvm::ffi::TensorView indices,
      double scale,
      double lower_bound,
      bool use_lower_bound,
      int64_t num_q_heads) {
    using namespace host;

    auto B_ = SymbolicSize{"batch"};
    auto MixedDim_ = SymbolicSize{"mixed_dim"};
    auto ADim_ = SymbolicSize{"a_dim"};
    auto HV_ = SymbolicSize{"num_v_heads"};
    auto V_ = SymbolicSize{"head_v_dim"};
    auto K_ = SymbolicSize{"head_k_dim"};
    auto Slots_ = SymbolicSize{"pool_slots"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B_, MixedDim_}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(mixed_qkv);
    TensorMatcher({B_, ADim_}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(a);
    TensorMatcher({B_, HV_}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(b);
    TensorMatcher({HV_}).with_dtype<fp32_t>().with_device(device).verify(A_log);
    TensorMatcher({ADim_}).with_dtype<fp32_t>().with_device(device).verify(dt_bias);
    TensorMatcher({B_, HV_, V_}).with_dtype<bf16_t>().with_device(device).verify(o);
    TensorMatcher({Slots_, HV_, V_, K_})
        .with_dtype<fp32_t>()
        .with_device(device)
        .with_strides({-1, -1, -1, 1})
        .verify(state);
    TensorMatcher({B_}).with_dtype<int32_t>().with_device(device).verify(indices);

    const auto B = static_cast<uint32_t>(B_.unwrap());
    const auto HV = static_cast<uint32_t>(HV_.unwrap());
    const auto H = static_cast<uint32_t>(num_q_heads);
    RuntimeCheck(K_.unwrap() == 128 && V_.unwrap() == 128, "kda_packed_decode is specialized for K = V = 128");
    RuntimeCheck(
        ADim_.unwrap() == HV * 128 && H > 0 && HV % H == 0, "a/dt_bias must be [*, HV*K] and HV divisible by H");
    RuntimeCheck(MixedDim_.unwrap() == 2 * H * 128 + HV * 128, "mixed_qkv last dim must be 2*H*K + HV*V");
    RuntimeCheck(state.stride(1) == 128 * 128 && state.stride(2) == 128, "state inner layout must be dense [HV, V, K]");
    if (B == 0) return;

    const auto params = KdaPackedDecodeParams{
        .mixed_qkv = static_cast<const bf16_t*>(mixed_qkv.data_ptr()),
        .a = static_cast<const bf16_t*>(a.data_ptr()),
        .b = static_cast<const bf16_t*>(b.data_ptr()),
        .A_log = static_cast<const fp32_t*>(A_log.data_ptr()),
        .dt_bias = static_cast<const fp32_t*>(dt_bias.data_ptr()),
        .o = static_cast<bf16_t*>(o.data_ptr()),
        .state = static_cast<fp32_t*>(state.data_ptr()),
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .stride_mixed = mixed_qkv.stride(0),
        .stride_a = a.stride(0),
        .stride_b = b.stride(0),
        .stride_state = state.stride(0),
        .H = H,
        .HV = HV,
        .scale = static_cast<fp32_t>(scale),
        .lower_bound = static_cast<fp32_t>(lower_bound),
        .use_lower_bound = use_lower_bound ? 1 : 0,
    };

    LaunchKernel(B * HV, kWarps * 32, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
