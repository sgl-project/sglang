#pragma once

#include <sgl_kernel/sampling.cuh>

#include "softmax.cuh"

namespace sglang {

/// A warp's part of a row: its raw max logit (-inf when empty), the sum of ex2((x - max) * scale)
/// and its winner's key (greedy: the raw logit; sample: (x - max) * scale - lg2(E)) and element.
struct alignas(16) SampleStats {
  float max;
  float sum;
  float key;
  uint32_t index;
};

struct SoftmaxSampleParams {
  const float* input;
  float* output;
  const float* temperatures;  // nullptr, or one per row
  const int32_t* top_ks;      // top_k <= 1 takes the argmax of the raw logits
  // torch's Philox state: device pointers under graph capture, else nullptr and the values below
  const int64_t* seed_ptr;
  const int64_t* offset_ptr;
  float* topk_p;
  int64_t* topk_index;
  int64_t input_stride;
  int64_t output_stride;
  uint64_t seed;
  uint64_t offset;  // added to *offset_ptr when captured (the intragraph offset)
  uint32_t num_vecs;
};

// fp32 per dispatch vector (`_vec_size` in softmax_jit.py)
inline constexpr uint32_t kSampleVecSize = device::kMaxVecBytes / sizeof(float);

// torch's exponential_ on curand_uniform's u = w 2^-32 + 2^-33 for two words, in log2 units (E / ln 2,
// a common factor of every element); torch's eps rule (u >= 1 - 2^-24 gives 2^-24) as a clamp. Near
// u = 1 the approximate lg2 is as coarse as torch's __logf there.
SGL_DEVICE fp32x2_t exponential2(uint32_t w0, uint32_t w1) {
  constexpr float kTwoM32 = 2.3283064365386963e-10f;
  constexpr float kTwoM33 = 1.1641532182693481e-10f;
  constexpr float kEps = 5.9604644775390625e-8f * 1.4426950408889634f;  // 2^-24 log2(e)
  const auto u = device::math::fma_f32x2(
      fp32x2_t{__uint2float_rn(w0), __uint2float_rn(w1)}, fp32x2_t{kTwoM32, kTwoM32}, fp32x2_t{kTwoM33, kTwoM33});
  return {fmaxf(-device::ptx::lg2_approx_ftz(u.x), kEps), fmaxf(-device::ptx::lg2_approx_ftz(u.y), kEps)};
}

/// Two 256-thread CTAs per SM (128 registers): every config fits without spilling (up to 11 vectors
/// with 2 pre-drawn, ptxas 13.4), and a second CTA beats the registers in every measured cell.
template <uint32_t kThreads>
inline constexpr uint32_t kSoftmaxSampleBlocksPerSM = 512 / kThreads;

// Elements in pairs, for the packed fp32x2 instructions
inline constexpr uint32_t kSamplePairs = kSampleVecSize / 2;
using SampleVec = device::AlignedVector<fp32x2_t, kSamplePairs>;

/// E / ln 2 of one vector's elements, `group` being the Philox group of its first 4.
SGL_DEVICE void draw_vec(fp32x2_t (&out)[kSamplePairs], uint64_t seed, uint64_t group, uint64_t offset) {
#pragma unroll
  for (uint32_t g = 0; g < kSampleVecSize / 4; ++g) {
    const auto w = device::sampling::curand_philox4(seed, group + g, offset);
    out[g * 2 + 0] = exponential2(w.x, w.y);
    out[g * 2 + 1] = exponential2(w.z, w.w);
  }
}

/**
 * \brief Row softmax over fp32 logits (one row per cluster, blockIdx.y) that also picks one token per row.
 * \note top_k <= 1 rows take the lowest argmax of the raw logits. Other rows run an exponential race,
 *       argmax e / E, E ~ Exp(1) from torch's Philox state with curand's layout (counter (offset / 4,
 *       row * vocab / 4 + element / 4), word element % 4). Every logit is loaded into registers up
 *       front; the E of the first `kPreVecs` vectors are drawn while the loads are in flight, the
 *       rest where they are used. Each e replaces its logit and the probs are stored from registers.
 */
template <bool kUsePDL, uint32_t kThreads, uint32_t kVecsPerThread, uint32_t kPreVecs, uint32_t kClusterSize>
__global__ __launch_bounds__(kThreads, kSoftmaxSampleBlocksPerSM<kThreads>)  //
    void softmax_sample_kernel(const __grid_constant__ SoftmaxSampleParams params) {
  using namespace device;
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  constexpr float kLog2e = 1.4426950408889634f;
  constexpr uint32_t kVecSize = kSampleVecSize;
  constexpr uint32_t kGroupsPerVec = kVecSize / 4;  // one Philox call per 4 elements
  constexpr uint32_t kNumWarps = kThreads / kWarpSize;
  static_assert(kClusterSize >= 1 && kClusterSize <= kSoftmaxMaxCluster);
  static_assert(kClusterSize == 1 || SGL_ARCH_HOPPER_OR_GREATER);
  static_assert(kVecSize % 4 == 0 && kPreVecs <= kVecsPerThread);
  constexpr bool kUseCluster = kClusterSize > 1;
  constexpr uint32_t kPoolSize = kClusterSize * kNumWarps;
  constexpr uint32_t kPerLane = (kPoolSize + kWarpSize - 1) / kWarpSize;

  __shared__ SampleStats s_pool[kPoolSize];
  __shared__ uint64_t s_bar;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpSize;
  const auto lane_id = tx % kWarpSize;
  const auto row = blockIdx.y;
  const auto rank = kUseCluster ? ptx::cluster_ctarank() : 0u;

  if constexpr (kUseCluster) {
    if (tx == 0) {
      ptx::mbar_init(&s_bar, 1);
      ptx::mbar_arrive_expect_tx(&s_bar, kPoolSize * sizeof(SampleStats));
      ptx::fence_mbarrier_init_cluster();
    }
    ptx::barrier_cluster_arrive_relaxed();
  }

  // Everything but the logits was written before the previous kernel ran (top_ks / temperatures by
  // the host, the Philox state by torch before a replay), so it is read ahead of the PDL wait, which
  // also keeps these loads ahead of the logits loads
  const auto topk_p = assume_global(params.topk_p + row);
  const auto topk_index = assume_global(params.topk_index + row);
  const auto num_vecs = params.num_vecs;
  const auto top_k = params.top_ks[row];
  const auto temperature = params.temperatures == nullptr ? 1.0f : params.temperatures[row];
  // Approximate division: no slow-path branch that would wait for the load before the logits loads
  const auto log2_scale = __fdividef(kLog2e, temperature);
  const bool captured = params.seed_ptr != nullptr;
  const auto seed = captured ? static_cast<uint64_t>(*params.seed_ptr) : params.seed;
  const auto offset = params.offset + (captured ? static_cast<uint64_t>(*params.offset_ptr) : 0);
  // Vector n of the thread is the row's vector (n * kClusterSize + rank) * kThreads + tx: only the
  // last one can be past the row's end (the launcher checks the others)
  constexpr uint32_t kVecStride = kClusterSize * kThreads;
  const auto vec_base = rank * kThreads + tx;
  const auto vec_of = [&](uint32_t n) { return vec_base + n * kVecStride; };
  const auto group_base = static_cast<uint64_t>(row) * (num_vecs * kGroupsPerVec) + vec_base * kGroupsPerVec;
  const auto group_of = [&](uint32_t n) { return group_base + n * kVecStride * kGroupsPerVec; };
  constexpr uint32_t kLast = kVecsPerThread - 1;
  const bool last_valid = vec_of(kLast) < num_vecs;
  // The thread's vector 0; vector n is kVecStride * n vectors on, a constant offset
  const auto input = assume_global(params.input + params.input_stride * row + vec_base * kVecSize);
  const auto output = assume_global(params.output + params.output_stride * row + vec_base * kVecSize);

  PDLWaitPrimary<kUsePDL>();

  SampleVec x[kVecsPerThread];
#pragma unroll
  for (uint32_t n = 0; n < kLast; ++n) {
    x[n].load(input, n * kVecStride);
  }
  x[kLast].fill(fp32x2_t{kNegInf, kNegInf});
  if (last_valid) x[kLast].load(input, kLast * kVecStride);

  // Drawn whatever the row's top_k, so the draws need not wait for it. ptxas schedules within a block
  // and bottom-up would place each load only its modelled latency before its first use, i.e. among
  // the draws: a branch it cannot remove (num_vecs > 0 is checked by the launcher) ends the block
  if (num_vecs == 0) __trap();
  fp32x2_t pre[kPreVecs > 0 ? kPreVecs : 1][kSamplePairs];
#pragma unroll
  for (uint32_t n = 0; n < kPreVecs; ++n) {
    draw_vec(pre[n], seed, group_of(n), offset);
  }

  // Started from every pre-drawn E (E > 0, so -inf), the max cannot be hoisted above the draws,
  // where it would stall on the loads
  float thr_max = kNegInf;
  if constexpr (kPreVecs > 0) {
    float least = pre[0][0].x;
#pragma unroll
    for (uint32_t n = 0; n < kPreVecs; ++n) {
#pragma unroll
      for (uint32_t i = 0; i < kSamplePairs; ++i) {
        least = fminf(least, fminf(pre[n][i].x, pre[n][i].y));
      }
    }
    thr_max = least * kNegInf;
  }
  // The warp's raw max is every lane's shift: keys and sums of a warp share one frame
#pragma unroll
  for (uint32_t n = 0; n < kVecsPerThread; ++n) {
#pragma unroll
    for (uint32_t i = 0; i < kSamplePairs; ++i) {
      thr_max = fmaxf(thr_max, fmaxf(x[n][i].x, x[n][i].y));
    }
  }
  const auto warp_max = warp::reduce_max(thr_max);
  const bool warp_empty = warp_max == kNegInf;
  const auto shift = warp_empty ? 0.0f : warp_max;
  // e = ex2((x - shift) * scale), exact 1 at the max whatever its magnitude; two elements at a time
  const auto neg_shift2 = fp32x2_t{-shift, -shift};
  const auto log2_scale2 = fp32x2_t{log2_scale, log2_scale};
  const auto exp2_pair = [&](fp32x2_t v) {
    const auto t = math::mul_f32x2(math::add_f32x2(v, neg_shift2), log2_scale2);
    return fp32x2_t{ptx::ex2_approx(t.x), ptx::ex2_approx(t.y)};
  };

  auto sum2 = fp32x2_t{0.0f, 0.0f};
  float best_e = 0.0f;  // e of the thread's winner
  uint32_t best = ~0u;  // its slot, vector * kVecSize + element
  float thr_key;
  if (top_k <= 1) {
    // The lowest slot holding the warp max, by selects from the back; its e is the warp max's
#pragma unroll
    for (int32_t slot = kVecsPerThread * kVecSize - 1; slot >= 0; --slot) {
      const auto& pair = x[slot / kVecSize][slot % kVecSize / 2];
      best = (slot % 2 == 0 ? pair.x : pair.y) == warp_max ? slot : best;
    }
    best_e = ptx::ex2_approx(log2_scale2.x * (warp_max + neg_shift2.x));
#pragma unroll
    for (uint32_t n = 0; n < kVecsPerThread; ++n) {
#pragma unroll
      for (uint32_t i = 0; i < kSamplePairs; ++i) {
        x[n][i] = exp2_pair(x[n][i]);
        sum2 = math::add_f32x2(sum2, x[n][i]);
      }
    }
    thr_key = warp_max;
  } else {
    // Running best of e / E as (e, E), compared without a division; e = 0 never wins
    float best_den = 1.0f;
#pragma unroll
    for (uint32_t n = 0; n < kVecsPerThread; ++n) {
      fp32x2_t drawn[kSamplePairs];
      if (n >= kPreVecs) draw_vec(drawn, seed, group_of(n), offset);
#pragma unroll
      for (uint32_t i = 0; i < kSamplePairs; ++i) {
        const auto den = n < kPreVecs ? pre[n < kPreVecs ? n : 0][i] : drawn[i];
        const auto e = exp2_pair(x[n][i]);
        sum2 = math::add_f32x2(sum2, e);
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
          const auto e_h = h == 0 ? e.x : e.y;
          const auto den_h = h == 0 ? den.x : den.y;
          if (e_h * best_den > best_e * den_h) {
            best = n * kVecSize + i * 2 + h;
            best_e = e_h;
            best_den = den_h;
          }
        }
        x[n][i] = e;
      }
    }
    thr_key = best == ~0u ? kNegInf : ptx::lg2_approx_ftz(best_e) - ptx::lg2_approx_ftz(best_den);
  }
  const auto thr_sum = sum2.x + sum2.y;
  const auto thr_index = best == ~0u ? ~0u : vec_of(best / kVecSize) * kVecSize + best % kVecSize;

  // Exchange; keys are in the warp's frame (greedy: raw logits)
  const auto warp_key = warp::reduce_max(thr_key);
  const auto warp_index = warp::reduce_min(thr_key == warp_key ? thr_index : ~0u);
  const auto warp_stats = SampleStats{warp_max, warp::reduce_sum(thr_sum), warp_key, warp_index};
  if constexpr (kUseCluster) {
    ptx::barrier_cluster_wait();  // every peer's mbarrier is initialized past this point
    if (lane_id < kClusterSize) {
      const auto slot = ptx::to_shared(&s_pool[rank * kNumWarps + warp_id]);
      const auto bar = ptx::to_shared(&s_bar);
      ptx::st_async_v4(ptx::mapa_shared(slot, lane_id), warp_stats, ptx::mapa_shared(bar, lane_id));
    }
    ptx::mbar_wait_parity_acquire_cta(&s_bar, 0);
  } else {
    if (lane_id == 0) s_pool[warp_id] = warp_stats;
    __syncthreads();
  }

  // Row merge, in the same order in every warp
  SampleStats partials[kPerLane];
  float lane_max = kNegInf;
#pragma unroll
  for (uint32_t j = 0; j < kPerLane; ++j) {
    const auto slot = j * kWarpSize + lane_id;
    partials[j] = slot < kPoolSize ? s_pool[slot] : SampleStats{kNegInf, 0.0f, kNegInf, ~0u};
    lane_max = fmaxf(lane_max, partials[j].max);
  }
  const auto row_max = warp::reduce_max(lane_max);
  const bool empty = row_max == kNegInf;
  const auto row_shift = empty ? 0.0f : row_max;
  float lane_sum = 0.0f;
  float lane_key = kNegInf;
#pragma unroll
  for (uint32_t j = 0; j < kPerLane; ++j) {
    auto& p = partials[j];
    // An empty partial has max -inf: factor 0, key stays -inf
    const auto to_row = (p.max - row_shift) * log2_scale;
    lane_sum += p.sum * ptx::ex2_approx(to_row);
    if (top_k > 1) p.key += to_row;
    lane_key = fmaxf(lane_key, p.key);
  }
  const auto row_sum = warp::reduce_sum(lane_sum);
  const auto row_key = warp::reduce_max(lane_key);
  uint32_t lane_index = ~0u;
#pragma unroll
  for (uint32_t j = 0; j < kPerLane; ++j) {
    lane_index = min(lane_index, partials[j].key == row_key ? partials[j].index : ~0u);
  }
  const auto win_index = warp::reduce_min(lane_index);

  // p = e * 2^((shift - row_shift) * scale) / sum; an empty warp holds zeros and its factor may
  // overflow, so it scales by 0
  const auto scale =
      empty || warp_empty ? 0.0f : __fdividef(ptx::ex2_approx((shift - row_shift) * log2_scale), row_sum);
#pragma unroll
  for (uint32_t n = 0; n < kVecsPerThread; ++n) {
    if (n < kLast || last_valid) {
      SampleVec p;
#pragma unroll
      for (uint32_t i = 0; i < kSamplePairs; ++i) {
        // Scalar: FMUL writes the store's register quads directly, FMUL2 would need moves into them
        p[i] = fp32x2_t{x[n][i].x * scale, x[n][i].y * scale};
      }
      p.store(output, n * kVecStride);
    }
  }
  // The winner's owner writes it with the same multiply as its probs store; a row with nothing above
  // -inf (or only NaN) takes index 0, as torch.argmax, written by the owner of element 0
  if (win_index != ~0u ? thr_index == win_index : rank == 0 && tx == 0) {
    *topk_p = (win_index != ~0u ? best_e : x[0][0].x) * scale;
    *topk_index = win_index != ~0u ? win_index : 0u;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL, uint32_t kThreads, uint32_t kVecsPerThread, uint32_t kPreVecs, uint32_t kClusterSize>
struct SoftmaxSampleKernel {
  static constexpr auto kernel = softmax_sample_kernel<kUsePDL, kThreads, kVecsPerThread, kPreVecs, kClusterSize>;

  // Permission-only attributes, set once per device to their only value
  static void init_attributes(int device_id) {
    host::runtime::init_per_device(device_id, [] {
      if constexpr (kClusterSize > 8) {
        const auto kernel_ptr = reinterpret_cast<const void*>(kernel);
        CHECK_CUDA(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      }
    });
  }

  static uint32_t occupancy() {
    int device_id;
    CHECK_CUDA(cudaGetDevice(&device_id));
    init_attributes(device_id);
    return host::runtime::get_blocks_per_sm(kernel, kThreads);
  }

  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView output,
      tvm::ffi::Optional<tvm::ffi::TensorView> temperatures,
      tvm::ffi::TensorView top_ks,
      tvm::ffi::TensorView seed,
      tvm::ffi::TensorView offset,
      tvm::ffi::TensorView intragraph,
      tvm::ffi::TensorView topk_p,
      tvm::ffi::TensorView topk_index) {
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    auto V = SymbolicSize{"vocab_size"};
    auto device_ = SymbolicDevice{};
    auto state_device = SymbolicDevice{};
    TensorMatcher({N, V})  //
        .with_strides({-1, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<float>()
        .ensure_alignment(device::kMaxVecBytes)
        .verify(input);
    TensorMatcher({N, V})  //
        .with_strides({-1, 1})
        .with_device(device_)
        .with_dtype<float>()
        .ensure_alignment(device::kMaxVecBytes)
        .verify(output);
    TensorMatcher({N})  //
        .with_device(device_)
        .with_dtype<int32_t>()
        .verify(top_ks);
    TensorMatcher({1})  //
        .with_device<kDLCPU, kDLCUDA>(state_device)
        .with_dtype<int64_t>()
        .verify(seed)
        .verify(offset);
    TensorMatcher({1})  //
        .with_device<kDLCPU>()
        .with_dtype<int64_t>()
        .verify(intragraph);
    TensorMatcher({N, 1})  //
        .with_device(device_)
        .with_dtype<float>()
        .verify(topk_p);
    TensorMatcher({N, 1})  //
        .with_device(device_)
        .with_dtype<int64_t>()
        .verify(topk_index);
    auto temperature_ptr = static_cast<const float*>(nullptr);
    if (temperatures.has_value()) {
      TensorMatcher({N})  //
          .with_device(device_)
          .with_dtype<float>()
          .verify(temperatures.value());
      temperature_ptr = static_cast<const float*>(temperatures.value().data_ptr());
    }

    const auto num_rows = static_cast<uint32_t>(N.unwrap());
    const auto vocab_size = static_cast<uint32_t>(V.unwrap());
    constexpr uint32_t kMaxGridDimY = 65535;  // CUDA limit
    CHECK_HOST(num_rows <= kMaxGridDimY);
    if (num_rows == 0) return;
    CHECK_HOST(vocab_size > 0);
    CHECK_HOST(vocab_size % kSampleVecSize == 0) << "softmax_sample: vocab_size " << vocab_size;
    CHECK_HOST(vocab_size / kSampleVecSize <= kClusterSize * kVecsPerThread * kThreads)
        << "softmax_sample: vocab_size " << vocab_size << " exceeds the configured capacity";
    CHECK_HOST(vocab_size / kSampleVecSize > kClusterSize * (kVecsPerThread - 1) * kThreads)
        << "softmax_sample: vocab_size " << vocab_size << " leaves a thread's vector other than the last empty";

    const auto seed_data = static_cast<const int64_t*>(seed.data_ptr());
    const auto offset_data = static_cast<const int64_t*>(offset.data_ptr());
    const bool captured = state_device.unwrap().device_type == kDLCUDA;
    const auto intragraph_value = static_cast<uint64_t>(*static_cast<const int64_t*>(intragraph.data_ptr()));
    const auto params = SoftmaxSampleParams{
        .input = static_cast<const float*>(input.data_ptr()),
        .output = static_cast<float*>(output.data_ptr()),
        .temperatures = temperature_ptr,
        .top_ks = static_cast<const int32_t*>(top_ks.data_ptr()),
        .seed_ptr = captured ? seed_data : nullptr,
        .offset_ptr = captured ? offset_data : nullptr,
        .topk_p = static_cast<float*>(topk_p.data_ptr()),
        .topk_index = static_cast<int64_t*>(topk_index.data_ptr()),
        .input_stride = input.stride(0),
        .output_stride = output.stride(0),
        .seed = captured ? 0 : static_cast<uint64_t>(*seed_data),
        .offset = intragraph_value + (captured ? 0 : static_cast<uint64_t>(*offset_data)),
        .num_vecs = vocab_size / kSampleVecSize,
    };
    const auto device = device_.unwrap();
    init_attributes(device.device_id);
    auto launcher = LaunchKernel({kClusterSize, num_rows}, kThreads, device);
    if constexpr (kClusterSize > 1) {
      launcher.config({.use_pdl = kUsePDL, .cluster_dim = dim3{kClusterSize}})(kernel, params);
    } else {
      launcher.config({.use_pdl = kUsePDL})(kernel, params);
    }
  }
};

}  // namespace sglang
