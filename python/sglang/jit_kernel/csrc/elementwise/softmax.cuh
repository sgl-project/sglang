/// \file softmax.cuh
/// \brief Softmax kernels for LLM sampling with large vocabulary sizes.
///
/// Unified launcher `SoftmaxKernel<kUsePDL, DType>::run` dispatches between:
///   - **Fused** (num_splits is None): one CTA per row, online softmax
///     (single-pass max+sum, then normalize+write)
///   - **Split** (num_splits is set):  multi-CTA per row partial softmax +
///     warp-level merge correction.  num_splits must be <= 32.
///
/// Output dtype matches input dtype.  All internal computation is in fp32.

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cfloat>

namespace {

constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;  // 32

// Target occupancy is 2
#define SGL_SOFTMAX_KERNEL __global__ void __launch_bounds__(kBlockSize, 2)

// ============================================================================
// Params struct shared by all kernels
// ============================================================================

struct SoftmaxParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  void* __restrict__ workspace;  // partial (max, sum) pairs for split path
  const float* __restrict__ ts;  // per-row temperatures (null -> use scalar t)
  float t;                       // scalar temperature fallback
  uint32_t vocab_size;
  uint32_t num_splits;  // 0 for fused, <=32 for split

  SGL_DEVICE float get_temperature(uint32_t row) const {
    return ts ? ts[row] : t;
  }
};

// ============================================================================
// Online softmax helpers
// ============================================================================

/// Warp-level online softmax merge across all 32 lanes.
SGL_DEVICE fp32x2_t warp_online_softmax_merge(fp32x2_t ms) {
  namespace math = device::math;
  auto [m, s] = ms;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    float om = __shfl_xor_sync(0xffffffffu, m, mask);
    float os = __shfl_xor_sync(0xffffffffu, s, mask);
    float nm = math::max(m, om);
    s = s * math::exp(m - nm) + os * math::exp(om - nm);
    m = nm;
  }
  return fp32x2_t{m, s};
}

/// CTA-level online softmax merge.
/// smem must have at least 2 * kNumWarps floats.
/// After return, all threads see the final (max, sum) in (m, s).
[[maybe_unused]]
SGL_DEVICE fp32x2_t cta_online_softmax_merge(float m, float s) {
  __shared__ fp32x2_t smem_ms[kNumWarps];

  const uint32_t warp_id = threadIdx.x / device::kWarpThreads;
  const uint32_t lane_id = threadIdx.x % device::kWarpThreads;

  smem_ms[warp_id] = warp_online_softmax_merge({m, s});
  __syncthreads();
  if (warp_id == 0) {
    const auto ms = lane_id < kNumWarps ? smem_ms[lane_id] : fp32x2_t{-FLT_MAX, 0.0f};
    smem_ms[lane_id] = warp_online_softmax_merge(ms);
  }
  __syncthreads();
  return smem_ms[warp_id];
}

// ============================================================================
// Kernel A: Single-block fused softmax (online: 1 pass for max+sum)
// ============================================================================

template <typename DType, int kVecN, bool kUsePDL>
SGL_SOFTMAX_KERNEL softmax_fused_kernel(const __grid_constant__ SoftmaxParams params) {
  using namespace device;
  using vec_t = AlignedVector<DType, kVecN>;

  PDLWaitPrimary<kUsePDL>();

  const auto input = static_cast<const DType*>(params.input);
  const auto output = static_cast<DType*>(params.output);
  const auto vocab_size = params.vocab_size;

  const uint32_t row = blockIdx.x;
  const DType* row_in = input + static_cast<uint64_t>(row) * vocab_size;
  DType* row_out = output + static_cast<uint64_t>(row) * vocab_size;
  const float inv_temp = 1.0f / params.get_temperature(row);

  const uint32_t n_vecs = vocab_size / kVecN;

  // --- Pass 1: online softmax (single-pass max + sum) ---
  float m = -FLT_MAX;
  float s = 0.0f;
  for (uint32_t vi = threadIdx.x; vi < n_vecs; vi += kBlockSize) {
    vec_t v;
    v.load(row_in, vi);
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      float val = cast<fp32_t>(v[i]) * inv_temp;
      float old_m = m;
      m = math::max(m, val);
      s = s * math::exp(old_m - m) + math::exp(val - m);
    }
  }

  const auto [global_max, global_sum] = cta_online_softmax_merge(m, s);
  const float row_max = global_max;
  const float inv_sum = 1.0f / global_sum;

  // --- Pass 2: normalize & write (output dtype = input dtype) ---
  for (uint32_t vi = threadIdx.x; vi < n_vecs; vi += kBlockSize) {
    vec_t v_in;
    v_in.load(row_in, vi);
    vec_t v_out;
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      v_out[i] = cast<DType>(math::exp(cast<fp32_t>(v_in[i]) * inv_temp - row_max) * inv_sum);
    }
    v_out.store(row_out, vi);
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ============================================================================
// Kernel B1: Split partial softmax (map)
// ============================================================================
// Grid: (batch_size, num_splits).  Each block processes one chunk.
// Writes partial exp(x/T - local_max) as DType to output, plus
// (local_max, local_sum) to workspace as float2 pairs.

template <typename DType, int kVecN, bool kUsePDL>
SGL_SOFTMAX_KERNEL softmax_split_kernel(const __grid_constant__ SoftmaxParams params) {
  using namespace device;
  using vec_t = AlignedVector<DType, kVecN>;

  PDLWaitPrimary<kUsePDL>();

  const auto input = static_cast<const DType*>(params.input);
  const auto output = static_cast<DType*>(params.output);
  const auto partial_ms = static_cast<fp32x2_t*>(params.workspace);
  const auto vocab_size = params.vocab_size;
  const auto num_splits = params.num_splits;

  const uint32_t row = blockIdx.x;
  const uint32_t sid = blockIdx.y;
  const DType* row_in = input + static_cast<uint64_t>(row) * vocab_size;
  DType* row_out = output + static_cast<uint64_t>(row) * vocab_size;
  const float inv_temp = 1.0f / params.get_temperature(row);

  // Aligned chunk boundaries for vectorized loads
  const uint32_t chunk_raw = div_ceil(vocab_size, num_splits);
  const uint32_t chunk = div_ceil(chunk_raw, kVecN) * kVecN;
  const uint32_t c_start = sid * chunk;
  const uint32_t c_end = min(c_start + chunk, vocab_size);
  if (c_start >= vocab_size) return;
  const uint32_t c_len = c_end - c_start;

  __shared__ float smem_max[kNumWarps];

  const DType* c_in = row_in + c_start;
  DType* c_out = row_out + c_start;

  const uint32_t n_vecs = c_len / kVecN;

  // --- local max ---
  float tmax = -FLT_MAX;
  for (uint32_t vi = threadIdx.x; vi < n_vecs; vi += kBlockSize) {
    vec_t v;
    v.load(c_in, vi);
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      tmax = math::max(tmax, cast<fp32_t>(v[i]) * inv_temp);
    }
  }

  cta::reduce_max(tmax, smem_max, /*min_value=*/-FLT_MAX);
  __syncthreads();
  const float local_max = smem_max[0];

  // --- local sum + write partial exp (as DType) ---
  float tsum = 0.0f;
  for (uint32_t vi = threadIdx.x; vi < n_vecs; vi += kBlockSize) {
    vec_t v_in;
    v_in.load(c_in, vi);
    vec_t v_out;
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      float e = math::exp(cast<fp32_t>(v_in[i]) * inv_temp - local_max);
      v_out[i] = cast<DType>(e);
      tsum += e;
    }
    v_out.store(c_out, vi);
  }

  cta::reduce_sum(tsum, smem_max);
  __syncthreads();
  if (threadIdx.x == 0) {
    partial_ms[row * num_splits + sid] = fp32x2_t{local_max, smem_max[0]};
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ============================================================================
// Kernel B2: Split merge + correction (reduce)
// ============================================================================
// Grid: (batch_size, num_splits).  Each block corrects one chunk.
// Warp 0 in every block redundantly merges all partial (max, sum) - cheap
// since num_splits <= 32.  Then all threads apply the uniform per-chunk
// correction factor.

template <typename DType, int kVecN, bool kUsePDL>
SGL_SOFTMAX_KERNEL softmax_merge_kernel(const __grid_constant__ SoftmaxParams params) {
  using namespace device;
  using vec_t = AlignedVector<DType, kVecN>;

  PDLWaitPrimary<kUsePDL>();

  const auto output = static_cast<DType*>(params.output);
  const auto partial_ms = static_cast<const fp32x2_t*>(params.workspace);
  const auto vocab_size = params.vocab_size;
  const auto num_splits = params.num_splits;

  const uint32_t row = blockIdx.x;
  const uint32_t sid = blockIdx.y;
  DType* row_out = output + static_cast<uint64_t>(row) * vocab_size;

  // Compute this block's chunk range (same formula as split kernel)
  const uint32_t chunk_raw = div_ceil(vocab_size, num_splits);
  const uint32_t chunk = div_ceil(chunk_raw, kVecN) * kVecN;
  const uint32_t c_start = sid * chunk;
  const uint32_t c_end = min(c_start + chunk, vocab_size);
  if (c_start >= vocab_size) return;
  const uint32_t c_len = c_end - c_start;

  const uint32_t lane = threadIdx.x % kWarpThreads;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;

  // --- Warp 0: merge all partial (max, sum) -> correction for this chunk ---
  __shared__ float shared_corr;  // correction = exp(local_max - global_max) / global_sum
  const auto [local_max, local_sum] = partial_ms[row * num_splits + sid];
  if (warp_id == 0) {
    const auto ms = lane < num_splits ? partial_ms[row * num_splits + lane] : fp32x2_t{-FLT_MAX, 0.0f};
    const auto [global_max, global_sum] = warp_online_softmax_merge(ms);
    shared_corr = math::exp(local_max - global_max) / global_sum;
  }
  __syncthreads();

  // --- All threads: apply uniform correction to this chunk ---
  const float corr = shared_corr;
  DType* c_out = row_out + c_start;

  const uint32_t n_vecs = c_len / kVecN;

  for (uint32_t vi = threadIdx.x; vi < n_vecs; vi += kBlockSize) {
    vec_t v;
    v.load(c_out, vi);
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      v[i] = cast<DType>(cast<fp32_t>(v[i]) * corr);
    }
    v.store(c_out, vi);
  }

  PDLTriggerSecondary<kUsePDL>();
}

#undef SGL_SOFTMAX_KERNEL

// ============================================================================
// Unified host launcher
// ============================================================================

template <bool kUsePDL, typename DType>
struct SoftmaxKernel {
  static constexpr int kVecN = 16 / sizeof(DType);
  static constexpr auto fused_kernel = softmax_fused_kernel<DType, kVecN, kUsePDL>;
  static constexpr auto split_kernel = softmax_split_kernel<DType, kVecN, kUsePDL>;
  static constexpr auto merge_kernel = softmax_merge_kernel<DType, kVecN, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::Optional<tvm::ffi::TensorView> temperatures,
      const float temperature,
      uint32_t num_splits) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto V = SymbolicSize{"vocab_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, V})  //
        .with_dtype<DType>()
        .with_device(device_)
        .verify(input)
        .verify(output);
    if (temperatures.has_value()) {
      TensorMatcher({B})  //
          .with_dtype<fp32_t>()
          .with_device(device_)
          .verify(temperatures.value());
    }

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto vocab_size = static_cast<uint32_t>(V.unwrap());
    const auto device = device_.unwrap();

    if (batch_size == 0) return;
    RuntimeCheck(vocab_size % kVecN == 0, "vocab_size must be a multiple of ", kVecN, ", got ", vocab_size);

    if (num_splits == 0) {
      static const auto kNumSM = runtime::get_sm_count(device.device_id);
      // Find max splits (power-of-2) that can saturate 2 * kNumSM
      num_splits = 1;
      for (uint32_t s = 32; s > 1; s /= 2) {
        if (batch_size * s <= 2 * kNumSM) {
          num_splits = s;
          break;
        }
      }
    }

    auto params = SoftmaxParams{
        .input = input.data_ptr(),
        .output = output.data_ptr(),
        .workspace = nullptr,
        .ts = temperatures.has_value() ? static_cast<const float*>(temperatures.value().data_ptr()) : nullptr,
        .t = temperature,
        .vocab_size = vocab_size,
        .num_splits = num_splits,
    };

    if (num_splits <= 1) {
      // ---- Fused path: one CTA per row ----
      LaunchKernel(batch_size, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(fused_kernel, params);
    } else {
      // ---- Split path: multi-CTA per row ----
      RuntimeCheck(num_splits <= 32, "num_splits must be <= 32 for split softmax");

      // Allocate workspace: num_splits float2 pairs per row
      const int64_t ws_elems = static_cast<int64_t>(batch_size) * num_splits * 2;
      const auto workspace = ffi::empty({ws_elems}, get_dtype<float>(), device);
      params.workspace = workspace.data_ptr();

      // Kernel B1: partial softmax
      LaunchKernel({batch_size, num_splits}, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(split_kernel, params);

      // Kernel B2: merge + correction
      LaunchKernel({batch_size, num_splits}, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(merge_kernel, params);
    }
  }
};

}  // namespace
