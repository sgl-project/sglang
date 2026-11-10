// SM70 KDA recurrence. Volta has no bf16 tensor cores, and the Triton chunk
// kernel plus the packed-decode kernel are both bf16. This kernel is the
// fp16 path: one V-tile per block, K=V=128, fp32 state, fp32 math.
//
// Per token, per value head, state S is [V, K]:
//   q = l2(q) * scale,  k = l2(k)
//   g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))     (safe gate)
//     or -exp(A_log) * softplus(a + dt_bias)
//   S[:, k] *= exp(g[k])
//   v = beta * (v - S @ k)
//   S += v * k^T
//   o = S @ q
// beta = sigmoid(b) when raw_beta is set.
//
// V-rows are independent, so the grid is (V/8, sequences, heads). Each warp
// owns 2 rows and the whole K axis (4 elements per lane). State stays in
// shared memory across the tokens of one sequence.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cstdint>

namespace sglang {

// The token loop is serial, so prefill speed comes from blocks in flight:
// 8 rows give 128 blocks at 8 heads, against 32 with 32 rows (measured 2x).
constexpr int kKdaSm70RowsPerBlock = 8;
constexpr int kKdaSm70RowsPerWarp = kKdaSm70RowsPerBlock / 4;

struct KdaSm70Params {
  const fp16_t* __restrict__ q;  // [T, H, K]
  const fp16_t* __restrict__ k;
  const fp16_t* __restrict__ v;  // [T, HV, V]
  const fp16_t* __restrict__ a;  // [T, HV, K] raw gate
  const fp16_t* __restrict__ b;  // [T, HV]
  const fp32_t* __restrict__ A_log;
  const fp32_t* __restrict__ dt_bias;  // [HV, K]
  fp32_t* __restrict__ state;          // pool, slot stride = stride_state
  const int32_t* __restrict__ indices;
  const int32_t* __restrict__ cu;  // [N + 1]
  fp16_t* __restrict__ out;        // [T, HV, V]
  fp32_t* __restrict__ track;      // [N, HV, V, K], or nullptr
  const int32_t* __restrict__ track_chunk;
  fp32_t* __restrict__ inter;  // flat fp32 pool, or nullptr
  const int32_t* __restrict__ inter_indices;
  // Token strides in elements; q, k, v, a and b may be slices of one projection output.
  int64_t stride_q;
  int64_t stride_k;
  int64_t stride_v;
  int64_t stride_a;
  int64_t stride_b;
  int64_t stride_state;
  int64_t stride_track;
  int64_t inter_cache_steps;
  uint32_t H;
  uint32_t HV;
  fp32_t scale;
  fp32_t lower_bound;
  int32_t use_lower_bound;
  int32_t raw_beta;
  int32_t commit_state;
};

SGL_DEVICE float kda_sm70_sigmoid(float z) {
  if (z >= 0.f) {
    return 1.f / (1.f + expf(-z));
  }
  const float e = expf(z);
  return e / (1.f + e);
}

SGL_DEVICE float kda_sm70_gate(float x, float exp_a, float lower_bound, int use_lower_bound) {
  if (use_lower_bound) {
    return lower_bound * kda_sm70_sigmoid(exp_a * x);
  }
  // softplus, beta = 1, threshold = 20. Matches the Triton KDA kernel.
  const float sp = (x <= 20.f) ? logf(1.f + expf(x)) : x;
  return -exp_a * sp;
}

SGL_DEVICE float4 kda_sm70_ld4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}

SGL_DEVICE void kda_sm70_st4(float* p, float4 v) {
  *reinterpret_cast<float4*>(p) = v;
}

SGL_DEVICE void kda_sm70_snapshot(
    float* track, const float* s_state, int64_t stride_track, int seq, int hv, int v_base, int warp, int k0) {
  constexpr int K = 128;
  constexpr int V = 128;
  constexpr int kRowsPerWarp = kKdaSm70RowsPerWarp;
  if (track == nullptr) {
    return;
  }
  for (int r = 0; r < kRowsPerWarp; ++r) {
    const int lr = warp * kRowsPerWarp + r;
    const int global_v = v_base + lr;
    float* dst = track + seq * stride_track + (static_cast<int64_t>(hv) * V + global_v) * K + k0;
    kda_sm70_st4(dst, kda_sm70_ld4(s_state + lr * K + k0));
  }
}

__global__ void kda_sm70_kernel(const KdaSm70Params params) {
  constexpr int K = 128;
  constexpr int V = 128;
  constexpr int kRowsPerWarp = kKdaSm70RowsPerWarp;
  constexpr int kChunk = 64;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int seq = static_cast<int>(blockIdx.y);
  const int hv = static_cast<int>(blockIdx.z);
  const int v_base = static_cast<int>(blockIdx.x) * kKdaSm70RowsPerBlock;
  const uint32_t group = params.HV / params.H;
  const int k_head = hv / static_cast<int>(group);
  const int k0 = lane * 4;
  const int bos = params.cu[seq];
  const int eos = params.cu[seq + 1];
  const int64_t sidx = params.indices[seq];
  const float exp_a = expf(params.A_log[hv]);

  __shared__ float s_state[kKdaSm70RowsPerBlock * K];

  for (int r = 0; r < kRowsPerWarp; ++r) {
    const int lr = warp * kRowsPerWarp + r;
    const int global_v = v_base + lr;
    float* row = s_state + lr * K + k0;
    if (sidx >= 0) {
      const float* src = params.state + sidx * params.stride_state + (static_cast<int64_t>(hv) * V + global_v) * K + k0;
      kda_sm70_st4(row, kda_sm70_ld4(src));
    } else {
      kda_sm70_st4(row, make_float4(0.f, 0.f, 0.f, 0.f));
    }
  }

  const int chunk = params.track_chunk != nullptr ? params.track_chunk[seq] : -1;
  if (chunk == 0 && eos > bos) {
    kda_sm70_snapshot(params.track, s_state, params.stride_track, seq, hv, v_base, warp, k0);
  }

  // Token inputs do not depend on the state, so the next token's loads are
  // issued before this token's serial math and their latency overlaps it.
  struct TokenInputs {
    fp16_t q[4], k[4], a[4], b;
    fp16_t v[kRowsPerWarp];
  };
  const auto load_token = [&](int64_t tok, TokenInputs& in) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      in.q[e] = params.q[tok * params.stride_q + k_head * K + k0 + e];
      in.k[e] = params.k[tok * params.stride_k + k_head * K + k0 + e];
      in.a[e] = params.a[tok * params.stride_a + hv * K + k0 + e];
    }
    in.b = params.b[tok * params.stride_b + hv];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      in.v[r] = params.v[tok * params.stride_v + hv * V + v_base + warp * kRowsPerWarp + r];
    }
  };
  TokenInputs next;
  if (bos < eos) {
    load_token(bos, next);
  }

  for (int t = bos; t < eos; ++t) {
    const int64_t tok = t;
    const TokenInputs cur = next;
    if (t + 1 < eos) {
      load_token(tok + 1, next);
    }
    float qv[4], kv[4], decay[4];
    float q_sq = 0.f, k_sq = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      qv[e] = device::cast<float>(cur.q[e]);
      kv[e] = device::cast<float>(cur.k[e]);
      q_sq += qv[e] * qv[e];
      k_sq += kv[e] * kv[e];
      const float x = device::cast<float>(cur.a[e]) + params.dt_bias[hv * K + k0 + e];
      decay[e] = expf(kda_sm70_gate(x, exp_a, params.lower_bound, params.use_lower_bound));
    }
    q_sq = device::warp::reduce_sum<32>(q_sq);
    k_sq = device::warp::reduce_sum<32>(k_sq);
    const float q_inv = 1.f / sqrtf(q_sq + 1e-6f);
    const float k_inv = 1.f / sqrtf(k_sq + 1e-6f);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      qv[e] *= q_inv * params.scale;
      kv[e] *= k_inv;
    }
    const float b_raw = device::cast<float>(cur.b);
    const float beta = params.raw_beta ? kda_sm70_sigmoid(b_raw) : b_raw;

#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int lr = warp * kRowsPerWarp + r;
      const int global_v = v_base + lr;
      float* row = s_state + lr * K + k0;
      float4 h4 = kda_sm70_ld4(row);
      float h[4] = {h4.x, h4.y, h4.z, h4.w};
      float dot_k = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h[e] *= decay[e];
        dot_k += h[e] * kv[e];
      }
      dot_k = device::warp::reduce_sum<32>(dot_k);
      const float vv = device::cast<float>(cur.v[r]);
      const float v_new = (vv - dot_k) * beta;
      float dot_q = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h[e] += v_new * kv[e];
        dot_q += h[e] * qv[e];
      }
      dot_q = device::warp::reduce_sum<32>(dot_q);
      kda_sm70_st4(row, make_float4(h[0], h[1], h[2], h[3]));
      if (lane == 0) {
        params.out[(tok * params.HV + hv) * V + global_v] = device::cast<fp16_t>(dot_q);
      }
    }

    const int done = t - bos + 1;
    if (chunk > 0 && done == chunk * kChunk) {
      kda_sm70_snapshot(params.track, s_state, params.stride_track, seq, hv, v_base, warp, k0);
    }
    if (params.inter != nullptr && params.inter_indices != nullptr) {
      const int cache_idx = params.inter_indices[seq];
      if (cache_idx >= 0) {
        constexpr int64_t kInner = static_cast<int64_t>(V) * K;
        const int64_t step = t - bos;
        for (int r = 0; r < kRowsPerWarp; ++r) {
          const int lr = warp * kRowsPerWarp + r;
          const int global_v = v_base + lr;
          float* dst = params.inter +
                       (static_cast<int64_t>(cache_idx) * params.inter_cache_steps + step) * params.HV * kInner +
                       (static_cast<int64_t>(hv) * V + global_v) * K + k0;
          kda_sm70_st4(dst, kda_sm70_ld4(s_state + lr * K + k0));
        }
      }
    }
  }

  if (params.commit_state && sidx >= 0) {
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int lr = warp * kRowsPerWarp + r;
      const int global_v = v_base + lr;
      float* dst = params.state + sidx * params.stride_state + (static_cast<int64_t>(hv) * V + global_v) * K + k0;
      kda_sm70_st4(dst, kda_sm70_ld4(s_state + lr * K + k0));
    }
  }
}

template <typename T>
T* kda_sm70_opt_ptr(const tvm::ffi::Optional<tvm::ffi::TensorView>& opt) {
  if (!opt.has_value()) {
    return nullptr;
  }
  return static_cast<T*>(opt.value().data_ptr());
}

struct KdaSm70Kernel {
  static constexpr auto kernel = kda_sm70_kernel;

  /// \brief Recurrent KDA over packed sequences.
  /// \param q `[T, H, K]` fp16
  /// \param k `[T, H, K]` fp16
  /// \param v `[T, HV, V]` fp16
  /// \param a `[T, HV, K]` fp16 raw per-channel gate
  /// \param b `[T, HV]` fp16 beta logit, or beta when `raw_beta` is false
  /// \param state Pool `[slots, HV, V, K]` fp32, updated in place when `commit_state`
  /// \param cu `[N + 1]` int32 sequence offsets into `T`
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView b,
      const tvm::ffi::TensorView A_log,
      const tvm::ffi::TensorView dt_bias,
      const tvm::ffi::TensorView state,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView cu,
      const tvm::ffi::TensorView out,
      const tvm::ffi::Optional<tvm::ffi::TensorView> track_state,
      const tvm::ffi::Optional<tvm::ffi::TensorView> track_chunk,
      const tvm::ffi::Optional<tvm::ffi::TensorView> inter_state,
      const tvm::ffi::Optional<tvm::ffi::TensorView> inter_indices,
      int64_t inter_cache_steps,
      double scale,
      double lower_bound,
      bool use_lower_bound,
      bool raw_beta,
      bool commit_state) {
    using namespace host;

    auto T_ = SymbolicSize{"tokens"};
    auto H_ = SymbolicSize{"q_heads"};
    auto HV_ = SymbolicSize{"v_heads"};
    auto K_ = SymbolicSize{"head_k"};
    auto V_ = SymbolicSize{"head_v"};
    auto Slots_ = SymbolicSize{"slots"};
    auto N_ = SymbolicSize{"seqs"};
    auto Np1_ = SymbolicSize{"seqs_plus_one"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({T_, H_, K_}).with_dtype<fp16_t>().with_device(device).with_strides({-1, -1, 1}).verify(q);
    TensorMatcher({T_, H_, K_}).with_dtype<fp16_t>().with_device(device).with_strides({-1, -1, 1}).verify(k);
    TensorMatcher({T_, HV_, V_}).with_dtype<fp16_t>().with_device(device).with_strides({-1, -1, 1}).verify(v);
    TensorMatcher({T_, HV_, K_}).with_dtype<fp16_t>().with_device(device).with_strides({-1, -1, 1}).verify(a);
    TensorMatcher({T_, HV_}).with_dtype<fp16_t>().with_device(device).with_strides({-1, 1}).verify(b);
    TensorMatcher({HV_}).with_dtype<fp32_t>().with_device(device).verify(A_log);
    TensorMatcher({HV_, K_}).with_dtype<fp32_t>().with_device(device).with_strides({-1, 1}).verify(dt_bias);
    TensorMatcher({Slots_, HV_, V_, K_})
        .with_dtype<fp32_t>()
        .with_device(device)
        .with_strides({-1, -1, -1, 1})
        .verify(state);
    TensorMatcher({N_}).with_dtype<int32_t>().with_device(device).verify(indices);
    TensorMatcher({Np1_}).with_dtype<int32_t>().with_device(device).verify(cu);
    TensorMatcher({T_, HV_, V_}).with_dtype<fp16_t>().with_device(device).with_strides({-1, -1, 1}).verify(out);

    const auto T = T_.unwrap();
    const auto H = static_cast<uint32_t>(H_.unwrap());
    const auto HV = static_cast<uint32_t>(HV_.unwrap());
    const auto N = static_cast<uint32_t>(N_.unwrap());
    RuntimeCheck(K_.unwrap() == 128 && V_.unwrap() == 128, "SM70 KDA is specialized for K = V = 128");
    RuntimeCheck(H > 0 && HV % H == 0, "HV must be a positive multiple of H");
    RuntimeCheck(Np1_.unwrap() == static_cast<int64_t>(N) + 1, "cu_seqlens must have length N + 1");
    RuntimeCheck(state.stride(1) == 128 * 128 && state.stride(2) == 128, "state inner layout must be [HV, V, K]");
    RuntimeCheck(out.stride(1) == 128 && a.stride(1) == 128, "out and a must be dense in the last two dims");
    RuntimeCheck(
        q.stride(1) == 128 && k.stride(1) == 128 && v.stride(1) == 128, "q, k and v must be dense [H, 128] per token");
    RuntimeCheck(dt_bias.stride(0) == 128 && dt_bias.stride(1) == 1, "dt_bias must be contiguous [HV, 128]");
    RuntimeCheck(out.stride(0) == static_cast<int64_t>(HV) * 128, "out must be contiguous [T, HV, 128]");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 && state.stride(0) % 4 == 0,
        "state base and slot stride must be float4-aligned");
    (void)T;
    if (N == 0 || T == 0) {
      return;
    }

    int64_t stride_track = 0;
    if (track_state.has_value()) {
      RuntimeCheck(track_chunk.has_value(), "track_chunk_idx is required with track_state");
      auto NT_ = SymbolicSize{"track_rows"};
      auto HV2_ = SymbolicSize{"track_hv"};
      auto V2_ = SymbolicSize{"track_v"};
      auto K2_ = SymbolicSize{"track_k"};
      TensorMatcher({NT_, HV2_, V2_, K2_})
          .with_dtype<fp32_t>()
          .with_device(device)
          .with_strides({-1, -1, -1, 1})
          .verify(track_state.value());
      RuntimeCheck(
          NT_.unwrap() == N && HV2_.unwrap() == HV && V2_.unwrap() == 128 && K2_.unwrap() == 128,
          "track_state must be [N, HV, 128, 128]");
      RuntimeCheck(
          track_state.value().stride(1) == 128 * 128 && track_state.value().stride(2) == 128,
          "track_state inner layout must be [HV, V, K]");
      stride_track = track_state.value().stride(0);
      TensorMatcher({N_}).with_dtype<int32_t>().with_device(device).verify(track_chunk.value());
    }
    if (inter_state.has_value()) {
      RuntimeCheck(inter_indices.has_value(), "intermediate_state_indices is required");
      RuntimeCheck(inter_cache_steps > 0, "intermediate cache_steps must be positive");
      auto Flat_ = SymbolicSize{"inter_flat"};
      TensorMatcher({Flat_}).with_dtype<fp32_t>().with_device(device).verify(inter_state.value());
      TensorMatcher({N_}).with_dtype<int32_t>().with_device(device).verify(inter_indices.value());
    }

    const KdaSm70Params params{
        .q = static_cast<const fp16_t*>(q.data_ptr()),
        .k = static_cast<const fp16_t*>(k.data_ptr()),
        .v = static_cast<const fp16_t*>(v.data_ptr()),
        .a = static_cast<const fp16_t*>(a.data_ptr()),
        .b = static_cast<const fp16_t*>(b.data_ptr()),
        .A_log = static_cast<const fp32_t*>(A_log.data_ptr()),
        .dt_bias = static_cast<const fp32_t*>(dt_bias.data_ptr()),
        .state = static_cast<fp32_t*>(state.data_ptr()),
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .cu = static_cast<const int32_t*>(cu.data_ptr()),
        .out = static_cast<fp16_t*>(out.data_ptr()),
        .track = kda_sm70_opt_ptr<fp32_t>(track_state),
        .track_chunk = kda_sm70_opt_ptr<int32_t>(track_chunk),
        .inter = kda_sm70_opt_ptr<fp32_t>(inter_state),
        .inter_indices = kda_sm70_opt_ptr<int32_t>(inter_indices),
        .stride_q = q.stride(0),
        .stride_k = k.stride(0),
        .stride_v = v.stride(0),
        .stride_a = a.stride(0),
        .stride_b = b.stride(0),
        .stride_state = state.stride(0),
        .stride_track = stride_track,
        .inter_cache_steps = inter_cache_steps,
        .H = H,
        .HV = HV,
        .scale = static_cast<fp32_t>(scale),
        .lower_bound = static_cast<fp32_t>(lower_bound),
        .use_lower_bound = use_lower_bound ? 1 : 0,
        .raw_beta = raw_beta ? 1 : 0,
        .commit_state = commit_state ? 1 : 0,
    };

    dim3 grid(128 / kKdaSm70RowsPerBlock, N, HV);
    LaunchKernel(grid, 128, device.unwrap())(kernel, params);
  }
};

}  // namespace sglang
