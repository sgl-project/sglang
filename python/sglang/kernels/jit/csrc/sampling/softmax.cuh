#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/optional.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace sglang {

namespace device::ptx {

/// \brief Approximate 2^x, flush-to-zero; 2^-inf is 0.
SGL_DEVICE float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE uint32_t cluster_ctarank() {
#if SGL_ARCH_HOPPER_OR_GREATER
  uint32_t rank;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
#else
  return 0;
#endif
}

SGL_DEVICE uint32_t cluster_nctarank() {
#if SGL_ARCH_HOPPER_OR_GREATER
  uint32_t num;
  asm("mov.u32 %0, %%cluster_nctarank;" : "=r"(num));
  return num;
#else
  return 1;
#endif
}

/// \brief Address of the same shared variable in CTA `rank` of this cluster.
SGL_DEVICE uint32_t mapa_shared(uint32_t addr, uint32_t rank) {
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(result) : "r"(addr), "r"(rank));
  return result;
}

/// \brief Makes this thread's `mbarrier.init` visible to the cluster without a `MEMBAR.GPU`.
SGL_DEVICE void fence_mbarrier_init_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

SGL_DEVICE void barrier_cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

SGL_DEVICE void barrier_cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

/// \brief Async store of 8 bytes into a peer's shared memory, completing `remote_bar`'s tx count.
SGL_DEVICE void st_async_v2(uint32_t remote_addr, float x, float y, uint32_t remote_bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];" ::"r"(remote_addr),
               "f"(x),
               "f"(y),
               "r"(remote_bar)
               : "memory");
}

/// \brief CTA-scope acquire wait: same SASS as `.relaxed` (no `CCTL.IVALL`), and orders the
///        `st.async` data tracked by this mbarrier before later reads of it.
SGL_DEVICE void mbar_wait_parity_acquire_cta(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity)
      : "memory");
}

}  // namespace device::ptx

/// One warp's softmax statistics in the log2 frame: `max` of the scaled logits, and the
/// sum of `exp2(scaled - max)`. An empty warp holds `{-inf, 0}`.
struct alignas(8) SoftmaxStats {
  float max;
  float sum;
};

struct SoftmaxParams {
  const void* input;          // InT[rows][input_stride]
  float* output;              // may alias `input` when InT is float
  const float* temperatures;  // nullptr, or one per `rows_per_request` rows
  int64_t input_stride;
  int64_t output_stride;
  uint32_t num_vecs;  // vocab_size / kVecSize; exact, the launcher rejects a remainder
  device::math::fast_mod_div_u32_t rows_per_request;
};

inline constexpr uint32_t kSoftmaxMaxCluster = 16;
/// Up to this many cached fp32 per thread the kernel is held to 64 registers, else to 128.
inline constexpr uint32_t kSoftmaxPairedElems = 32;

/// CTAs of `kThreads` threads an SM holds, set by registers alone: 64 or 128 per thread.
template <uint32_t kThreads, uint32_t kVecsPerThread>
inline constexpr uint32_t kSoftmaxBlocksPerSM =
    (512 / kThreads) * (kVecsPerThread * device::kMaxVecBytes / sizeof(float) <= kSoftmaxPairedElems ? 2 : 1);

/**
 * \brief Row softmax with optional per-row temperature, one row per cluster of CTAs
 *        (blockIdx.y; the cluster size is a launch attribute, 1 to 16), every element held
 *        in registers between the read and the write, so each element is read once and
 *        written once.
 * \note The CTAs of a row exchange per-warp statistics by `st.async` into every peer's pool,
 *       tracked by the peer's mbarrier: no `barrier.cluster` per row, hence no `MEMBAR.GPU`.
 *       The one relaxed cluster barrier only publishes the mbarrier init and overlaps the loads.
 * \note `kMaxCluster` (a power of two >= the launch's cluster size) only bounds the unrolled pool
 *       merge. Rows must be whole, aligned vectors (vocab size, row stride and base); the
 *       launcher rejects anything else and `ops.sampling.softmax` falls back to torch.
 * \note Each CTA writes only the elements it read, so `output == input` is safe (fp32 input).
 * \note `InT` is the logits dtype (fp32, bf16 or fp16); statistics and output are fp32. A 16-bit
 *       input keeps the fp32 vector count per thread and loads it as 16 B, so the config is shared.
 */
template <bool kUsePDL, uint32_t kSoftmaxThreads, uint32_t kVecsPerThread, uint32_t kMaxCluster, typename InT>
__global__ __launch_bounds__(kSoftmaxThreads, kSoftmaxBlocksPerSM<kSoftmaxThreads, kVecsPerThread>)  //
    void softmax_kernel(const __grid_constant__ SoftmaxParams params) {
  using namespace device;
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  constexpr float kLog2e = 1.4426950408889634f;
  constexpr uint32_t kVecSize = kMaxVecBytes / sizeof(float);
  constexpr uint32_t kNumWarps = kSoftmaxThreads / kWarpSize;
  static_assert(kMaxCluster >= 1 && kMaxCluster <= kSoftmaxMaxCluster);
  static_assert(kMaxCluster == 1 || SGL_ARCH_HOPPER_OR_GREATER);
  constexpr bool kUseCluster = kMaxCluster > 1;
  constexpr uint32_t kMaxPoolSize = kMaxCluster * kNumWarps;
  constexpr uint32_t kPerLane = (kMaxPoolSize + kWarpSize - 1) / kWarpSize;
  constexpr uint32_t kChunkVecs = kVecsPerThread * kSoftmaxThreads;
  // 64-bit elements keep the 32 B store one `STG.E.ENL2.256`; a flat float[8] splits it.
  using vec_t = AlignedVector<fp32x2_t, kVecSize / 2>;
  // The same kVecSize elements per vector in the input dtype: a 16 B load for 16-bit inputs.
  using in_vec_t = AlignedVector<packed_t<InT>, kVecSize / 2>;

  __shared__ SoftmaxStats s_pool[kMaxPoolSize];
  __shared__ uint64_t s_bar;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpSize;
  const auto lane_id = tx % kWarpSize;
  const auto row = blockIdx.y;
  const auto cluster_size = kUseCluster ? ptx::cluster_nctarank() : 1u;
  const auto rank = kUseCluster ? ptx::cluster_ctarank() : 0u;
  const auto pool_size = cluster_size * kNumWarps;

  if constexpr (kUseCluster) {
    if (tx == 0) {
      ptx::mbar_init(&s_bar, 1);
      ptx::mbar_arrive_expect_tx(&s_bar, pool_size * sizeof(SoftmaxStats));
      ptx::fence_mbarrier_init_cluster();
    }
    ptx::barrier_cluster_arrive_relaxed();
  }

  const auto input = static_cast<const InT*>(params.input) + params.input_stride * row;
  const auto num_vecs = params.num_vecs;
  const auto vec_base = rank * kChunkVecs + tx;

  PDLWaitPrimary<kUsePDL>();

  vec_t vecs[kVecsPerThread];
  if constexpr (std::is_same_v<InT, float>) {
#pragma unroll
    for (uint32_t n = 0; n < kVecsPerThread; ++n) {
      const auto vid = vec_base + n * kSoftmaxThreads;
      vecs[n].fill(fp32x2_t{kNegInf, kNegInf});
      // nvcc 13.2-13.4 loses the global space across the PDL / cluster asm and emits 2x LD.E.128.
      __builtin_assume(__isGlobal(input));
      if (vid < num_vecs) vecs[n].load(input, vid);
    }
  } else {
    // Every load is issued before the first convert, which would otherwise wait on its load.
    in_vec_t raw[kVecsPerThread];
#pragma unroll
    for (uint32_t n = 0; n < kVecsPerThread; ++n) {
      __builtin_assume(__isGlobal(input));
      if (vec_base + n * kSoftmaxThreads < num_vecs) raw[n].load(input, vec_base + n * kSoftmaxThreads);
    }
#pragma unroll
    for (uint32_t n = 0; n < kVecsPerThread; ++n) {
      const bool valid = vec_base + n * kSoftmaxThreads < num_vecs;
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        vecs[n][i] = valid ? cast<fp32x2_t>(raw[n][i]) : fp32x2_t{kNegInf, kNegInf};
      }
    }
  }
  const auto log2_scale = params.temperatures == nullptr  //
                              ? kLog2e
                              : kLog2e / params.temperatures[row / params.rows_per_request];

  float thr_max = kNegInf;
#pragma unroll
  for (uint32_t n = 0; n < kVecsPerThread; ++n) {
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      thr_max = fmaxf(thr_max, fmaxf(vecs[n][i].x, vecs[n][i].y));
    }
  }
  // Max on raw logits (scaling by T > 0 is monotone). Exponents subtract the raw max before scaling, so
  // the max element is exactly ex2(0) even for huge finite logits (the NaN sanitizer's +-1e30).
  const auto raw_max = warp::reduce_max(thr_max);
  const bool empty = raw_max == kNegInf;
  const auto shift = empty ? 0.0f : raw_max;
  const auto base = shift * log2_scale;
  float thr_sum = 0.0f;
#pragma unroll
  for (uint32_t n = 0; n < kVecsPerThread; ++n) {
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      auto& [x, y] = vecs[n][i];
      x = ptx::ex2_approx((x - shift) * log2_scale);
      y = ptx::ex2_approx((y - shift) * log2_scale);
      thr_sum += x + y;
    }
  }
  const auto warp_stats = SoftmaxStats{empty ? kNegInf : base, warp::reduce_sum(thr_sum)};
  if constexpr (kUseCluster) {
    ptx::barrier_cluster_wait();  // every peer's mbarrier is initialized past this point
    if (lane_id < cluster_size) {
      const auto slot = ptx::to_shared(&s_pool[rank * kNumWarps + warp_id]);
      const auto bar = ptx::to_shared(&s_bar);
      const auto [v_lo, v_hi] = warp_stats;
      ptx::st_async_v2(ptx::mapa_shared(slot, lane_id), v_lo, v_hi, ptx::mapa_shared(bar, lane_id));
    }
    ptx::mbar_wait_parity_acquire_cta(&s_bar, 0);
  } else {
    if (lane_id == 0) s_pool[warp_id] = warp_stats;
    __syncthreads();
  }

  // Every warp reduces the whole pool in the same order, so all get bit-identical row_max / row_sum.
  SoftmaxStats partials[kPerLane];
  float lane_max = kNegInf;
#pragma unroll
  for (uint32_t j = 0; j < kPerLane; ++j) {
    const auto slot = j * kWarpSize + lane_id;
    partials[j] = slot < pool_size ? s_pool[slot] : SoftmaxStats{kNegInf, 0.0f};
    lane_max = fmaxf(lane_max, partials[j].max);
  }
  const auto row_max = warp::reduce_max(lane_max);
  float lane_sum = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < kPerLane; ++j) {
    lane_sum += partials[j].sum * ptx::ex2_approx(partials[j].max - row_max);
  }
  const auto row_sum = warp::reduce_sum(lane_sum);

  // `exp2(e - log2(sum))` would round the frame off the one the sum was taken in; keep the multiply.
  // An empty warp holds zeros and its 2^(0 - row_max) may overflow, so it scales by 0.
  const auto scale = empty ? 0.0f : ptx::ex2_approx(base - row_max) / row_sum;
  const auto output = params.output + params.output_stride * row;
#pragma unroll
  for (uint32_t n = 0; n < kVecsPerThread; ++n) {
    const auto vid = vec_base + n * kSoftmaxThreads;
    if (vid < num_vecs) {
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        vecs[n][i].x *= scale;
        vecs[n][i].y *= scale;
      }
      vecs[n].store(output, vid);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL, uint32_t kSoftmaxThreads, uint32_t kVecsPerThread, uint32_t kMaxCluster, typename InT>
struct SoftmaxKernel {
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(float);
  static constexpr auto kKernel = softmax_kernel<kUsePDL, kSoftmaxThreads, kVecsPerThread, kMaxCluster, InT>;

  /// \brief CTAs per SM the compiled kernel reaches; `__launch_bounds__` only sets a floor.
  static uint32_t occupancy() {
    return host::runtime::get_blocks_per_sm(kKernel, kSoftmaxThreads);
  }

  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView output,
      tvm::ffi::Optional<tvm::ffi::TensorView> temperatures,
      uint32_t rows_per_request,
      uint32_t cluster_size) {
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    auto V = SymbolicSize{"vocab_size"};
    auto device_ = SymbolicDevice{};
    TensorMatcher({N, V})  //
        .with_strides({-1, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<InT>()
        .ensure_alignment(kVecSize * sizeof(InT))
        .verify(input);
    TensorMatcher({N, V})  //
        .with_strides({-1, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<float>()
        .ensure_alignment(device::kMaxVecBytes)
        .verify(output);

    const auto num_rows = static_cast<uint32_t>(N.unwrap());
    const auto vocab_size = static_cast<uint32_t>(V.unwrap());
    constexpr uint32_t kMaxGridDimY = 65535;  // CUDA limit
    CHECK_HOST(num_rows <= kMaxGridDimY);
    if (num_rows == 0) return;
    CHECK_HOST(vocab_size > 0);
    CHECK_HOST(kMaxCluster == 1 ? cluster_size == 1 : (cluster_size > 1 && cluster_size <= kMaxCluster))
        << "softmax: cluster size " << cluster_size << " for a kernel built for " << kMaxCluster;
    CHECK_HOST(vocab_size % kVecSize == 0) << "softmax: vocab_size " << vocab_size;
    CHECK_HOST(vocab_size / kVecSize <= cluster_size * kVecsPerThread * kSoftmaxThreads)
        << "softmax: vocab_size " << vocab_size << " exceeds the configured capacity";

    CHECK_HOST(rows_per_request > 0);
    auto temperature_ptr = static_cast<const float*>(nullptr);
    if (temperatures.has_value()) {
      auto num_requests = SymbolicSize{"num_requests"};
      TensorMatcher({num_requests})  //
          .with_strides({1})
          .with_device<kDLCUDA>(device_)
          .with_dtype<float>()
          .verify(temperatures.value());
      CHECK_HOST(num_requests.unwrap() * rows_per_request == num_rows)
          << "softmax: " << num_requests.unwrap() << " temperatures x " << rows_per_request << " != " << num_rows
          << " rows";
      temperature_ptr = static_cast<const float*>(temperatures.value().data_ptr());
    }

    const auto device = device_.unwrap();
    const auto params = SoftmaxParams{
        .input = input.data_ptr(),
        .output = static_cast<float*>(output.data_ptr()),
        .temperatures = temperature_ptr,
        .input_stride = input.stride(0),
        .output_stride = output.stride(0),
        .num_vecs = vocab_size / kVecSize,
        .rows_per_request = static_cast<device::math::fast_mod_div_u32_t>(rows_per_request),
    };
    const auto cluster = static_cast<uint32_t>(cluster_size);
    auto launcher = LaunchKernel({cluster, num_rows}, kSoftmaxThreads, device);
    if (cluster > 1) {
      if constexpr (kMaxCluster > 8) {
        runtime::init_per_device(device.device_id, [] {
          const auto kernel_ptr = reinterpret_cast<const void*>(kKernel);
          CHECK_CUDA(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        });
      }
      launcher.config({.use_pdl = kUsePDL, .cluster_dim = dim3{cluster}})(kKernel, params);
    } else {
      launcher.config({.use_pdl = kUsePDL})(kKernel, params);
    }
  }
};

}  // namespace sglang
