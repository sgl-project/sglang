// Fused norm(x[, weight, bias]) * (1 + scale) + shift for SGLang diffusion
// models, plus the gated-residual variant that first computes
// res_out = residual + gate * x and normalizes res_out. H200 (sm_90a) build.
//
// Native-CUDA counterpart of the CuTe-DSL implementation in
// sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift.
// Numerics contract preserved from that implementation:
//   - the pre-norm value is rounded to the activation dtype before
//     normalization (res_out stores that rounded value, and the statistics
//     consume it);
//   - statistics accumulate in fp32; the normalized value is rounded to the
//     activation dtype before the (1 + scale) * y + shift epilogue;
//   - rms ignores bias; layer applies weight and bias when present.
// Layer variance uses the two-pass mean-then-variance form by default
// (kTwoPassVariance=true, matching the baseline contract); a single-round
// Welford/Chan path exists behind kTwoPassVariance=false as a measurable
// lever (it lost on B200; re-measure per-arch before adopting).
//
// One CTA per row; each thread owns one aligned vector of kVecBytes bytes.
// On Hopper device::kMaxVecBytes is 16, so a 32-byte kVecBytes is issued as
// two 128-bit transactions per thread (VecArray chunks). kVecBytes is a
// per-combo dispatch choice; blockDim.x = D / (kVecBytes / sizeof(DType)).

#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstdint>

namespace kda_norm_scale_shift {

using device::AlignedVector;
using device::kWarpThreads;

// Operand layout classes (values mirrored by the Python dispatcher).
inline constexpr int kOpAbsent = 0;  // scalar placeholder, no memory access
inline constexpr int kOpScalar = 1;  // [1] tensor, one broadcast value
inline constexpr int kOpRow = 2;     // [D] tensor, broadcast across rows
inline constexpr int kOpToken = 3;   // [num_rows, D] tensor, per-row values

inline constexpr int kMaxWarpsPerCta = 32;

struct NormScaleShiftParams {
  void* y;
  void* res_out;        // nullptr unless the residual variant
  const void* x;        // activations, [num_rows, D]
  const void* residual; // nullptr unless the residual variant
  const void* gate;
  const void* weight;
  const void* bias;
  const void* scale;
  const void* shift;
  int64_t num_rows;
  float eps;
  float inv_d;
};

// Per-thread aligned-vector view over kElems contiguous elements of T,
// split into <=kMaxVecBytes chunks so wide dtypes (fp32 operands paired with
// 16-element bf16 activations) still use maximal vector transactions.
template <typename T, int kElems>
struct VecArray {
  static constexpr int kChunkBytes =
      (kElems * sizeof(T) <= device::kMaxVecBytes) ? int(kElems * sizeof(T))
                                                   : int(device::kMaxVecBytes);
  static constexpr int kPerChunk = kChunkBytes / int(sizeof(T));
  static constexpr int kChunks = kElems / kPerChunk;
  static_assert(kChunks * kPerChunk == kElems);

  AlignedVector<T, kPerChunk> chunk[kChunks];

  // base: element pointer to this thread's first element (already offset).
  SGL_DEVICE void load(const void* base) {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      chunk[c].load(base, c);
    }
  }
  SGL_DEVICE void store(void* base) const {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      chunk[c].store(base, c);
    }
  }
  SGL_DEVICE T get(int i) const {
    return chunk[i / kPerChunk][i % kPerChunk];
  }
  SGL_DEVICE void set(int i, T v) {
    chunk[i / kPerChunk][i % kPerChunk] = v;
  }
};

// Load one operand (scale/shift/gate) into fp32 lanes according to its class.
template <typename T, int kElems, int kClass>
SGL_DEVICE void load_operand_f32(
    float (&dst)[kElems], const void* ptr, int64_t row, int64_t row_elems, int64_t thread_elem) {
  if constexpr (kClass == kOpScalar) {
    const float v = static_cast<float>(SGLANG_LDG(static_cast<const T*>(ptr)));
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      dst[e] = v;
    }
  } else if constexpr (kClass == kOpRow) {
    VecArray<T, kElems> vec;
    vec.load(static_cast<const T*>(ptr) + thread_elem);
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      dst[e] = static_cast<float>(vec.get(e));
    }
  } else if constexpr (kClass == kOpToken) {
    VecArray<T, kElems> vec;
    vec.load(static_cast<const T*>(ptr) + row * row_elems + thread_elem);
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      dst[e] = static_cast<float>(vec.get(e));
    }
  }
}

// Streaming mean/variance statistics combined with Chan's parallel merge:
// numerically robust (no E[x^2]-mean^2 cancellation) in a SINGLE reduction
// round. Counts are carried as floats; zero-count partners pass through.
struct WelfordStat {
  float n;
  float mean;
  float m2;

  SGL_DEVICE static WelfordStat merge(const WelfordStat& a, const WelfordStat& b) {
    if (b.n == 0.0f) return a;
    if (a.n == 0.0f) return b;
    const float n = a.n + b.n;
    const float delta = b.mean - a.mean;
    const float ratio = b.n / n;
    return WelfordStat{n, a.mean + delta * ratio, a.m2 + b.m2 + delta * delta * a.n * ratio};
  }
};

SGL_DEVICE WelfordStat warp_reduce_welford(WelfordStat s) {
#pragma unroll
  for (int mask = int(kWarpThreads) / 2; mask > 0; mask >>= 1) {
    const WelfordStat other{
        __shfl_xor_sync(device::kFullMask, s.n, mask, 32),
        __shfl_xor_sync(device::kFullMask, s.mean, mask, 32),
        __shfl_xor_sync(device::kFullMask, s.m2, mask, 32),
    };
    s = WelfordStat::merge(s, other);
  }
  return s;
}

// CTA-wide Welford merge; result broadcast to all threads via scratch rows
// (uses three scratch arrays of kMaxWarpsPerCta+1 floats).
SGL_DEVICE WelfordStat cta_reduce_welford(
    WelfordStat s, int warp, int lane, int num_warps, float* sn, float* smean, float* sm2) {
  s = warp_reduce_welford(s);
  if (lane == 0) {
    sn[warp] = s.n;
    smean[warp] = s.mean;
    sm2[warp] = s.m2;
  }
  __syncthreads();
  if (warp == 0) {
    WelfordStat acc = (lane < num_warps) ? WelfordStat{sn[lane], smean[lane], sm2[lane]}
                                         : WelfordStat{0.0f, 0.0f, 0.0f};
    acc = warp_reduce_welford(acc);
    if (lane == 0) {
      sn[kMaxWarpsPerCta] = acc.n;
      smean[kMaxWarpsPerCta] = acc.mean;
      sm2[kMaxWarpsPerCta] = acc.m2;
    }
  }
  __syncthreads();
  return WelfordStat{sn[kMaxWarpsPerCta], smean[kMaxWarpsPerCta], sm2[kMaxWarpsPerCta]};
}

// CTA-wide sum reduction of one fp32 value per thread; result broadcast.
// `slot` selects an independent shared-memory scratch row (0 or 1).
SGL_DEVICE float cta_reduce_sum(float v, int warp, int lane, int num_warps, float* scratch) {
  v = device::warp::reduce_sum(v);
  if (lane == 0) {
    scratch[warp] = v;
  }
  __syncthreads();
  if (warp == 0) {
    float a = (lane < num_warps) ? scratch[lane] : 0.0f;
    a = device::warp::reduce_sum(a);
    if (lane == 0) {
      scratch[kMaxWarpsPerCta] = a;
    }
  }
  __syncthreads();
  return scratch[kMaxWarpsPerCta];
}

template <
    typename DType,      // activation dtype (x / residual / y / res_out)
    typename ParamDType, // scale & shift dtype
    typename GateDType,  // gate dtype (ignored when kGateClass == kOpAbsent)
    typename WBDType,    // weight & bias dtype (ignored when !kHasWeightBias)
    bool kIsRms,
    int kScClass,        // kOpScalar / kOpRow / kOpToken
    int kGateClass,      // kOpAbsent / kOpRow / kOpToken
    bool kHasResidual,
    bool kHasWeightBias,
    bool kTwoPassVariance,
    bool kUsePDL,
    // Issue the scale/shift global loads before the statistics reductions and
    // hold the RAW (storage-dtype) vectors in registers until the epilogue.
    // NCU r1 on the fp32-row bucket showed nvcc otherwise sinks these loads to
    // the epilogue where their full latency is exposed after both reduction
    // barriers (short_scoreboard 2.4x the CuTe baseline at identical
    // geometry/regs/bytes). A zero-register prefetch.global.L1 variant was
    // measured first and rejected: every thread prefetching the shared row
    // operands flooded the LSU (620us vs 381us, barrier stalls 3x).
    bool kEarlyScaleShiftLoad,
    int kVecBytes>
__global__ void norm_scale_shift_kernel(const NormScaleShiftParams __grid_constant__ params) {
  using namespace device;
  constexpr int kElems = kVecBytes / int(sizeof(DType));
  static_assert(kElems > 0 && (kElems & (kElems - 1)) == 0);

  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & int(kWarpThreads - 1);
  const int warp = tid >> 5;
  const int num_warps = int(blockDim.x) >> 5;
  const int64_t D = int64_t(blockDim.x) * kElems;
  const int64_t base_elem = row * D + int64_t(tid) * kElems;
  const int64_t thread_elem = int64_t(tid) * kElems;

  __shared__ float s_scratch_a[kMaxWarpsPerCta + 1];
  __shared__ float s_scratch_b[kMaxWarpsPerCta + 1];
  __shared__ float s_scratch_c[kMaxWarpsPerCta + 1];

  PDLWaitPrimary<kUsePDL>();

  VecArray<DType, kElems> xv;
  xv.load(static_cast<const DType*>(params.x) + base_elem);

  // Early raw scale/shift loads (see kEarlyScaleShiftLoad). Held in storage
  // dtype so the live-range cost is the raw chunk registers only; expansion to
  // fp32 lanes happens at the epilogue.
  static_assert(!kEarlyScaleShiftLoad || kScClass == kOpRow || kScClass == kOpToken);
  VecArray<ParamDType, kElems> sc_raw, sh_raw;
  if constexpr (kEarlyScaleShiftLoad) {
    const int64_t ss_off = (kScClass == kOpToken) ? row * D + thread_elem : thread_elem;
    sc_raw.load(static_cast<const ParamDType*>(params.scale) + ss_off);
    sh_raw.load(static_cast<const ParamDType*>(params.shift) + ss_off);
  }

  float v[kElems];
#pragma unroll
  for (int e = 0; e < kElems; ++e) {
    v[e] = static_cast<float>(xv.get(e));
  }

  if constexpr (kHasResidual) {
    if constexpr (kGateClass != kOpAbsent) {
      float g[kElems];
      load_operand_f32<GateDType, kElems, kGateClass>(g, params.gate, row, D, thread_elem);
#pragma unroll
      for (int e = 0; e < kElems; ++e) {
        v[e] = g[e] * v[e];
      }
    }
    VecArray<DType, kElems> rv;
    rv.load(static_cast<const DType*>(params.residual) + base_elem);
    VecArray<DType, kElems> ro;
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      const DType rounded = static_cast<DType>(v[e] + static_cast<float>(rv.get(e)));
      ro.set(e, rounded);
      v[e] = static_cast<float>(rounded);  // the norm consumes the rounded value
    }
    ro.store(static_cast<DType*>(params.res_out) + base_elem);
  }

  // Fused statistics in fp32.
  float mean = 0.0f;
  float factor;
  if constexpr (kIsRms) {
    float sumsq = 0.0f;
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      sumsq += v[e] * v[e];
    }
    sumsq = cta_reduce_sum(sumsq, warp, lane, num_warps, s_scratch_a);
    factor = math::rsqrt(sumsq * params.inv_d + params.eps);
  } else if constexpr (kTwoPassVariance) {
    float sum = 0.0f;
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      sum += v[e];
    }
    mean = cta_reduce_sum(sum, warp, lane, num_warps, s_scratch_a) * params.inv_d;
    float acc = 0.0f;
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      const float d = v[e] - mean;
      acc += d * d;
    }
    const float var = cta_reduce_sum(acc, warp, lane, num_warps, s_scratch_b) * params.inv_d;
    factor = math::rsqrt(var + params.eps);
  } else {
    // Single-round robust statistics: thread-local mean + M2 over the kElems
    // register values, then a Chan/Welford parallel merge across the CTA.
    // Matches two-pass numerical quality without the second reduction round.
    float sum = 0.0f;
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      sum += v[e];
    }
    const float local_mean = sum * (1.0f / float(kElems));
    float local_m2 = 0.0f;
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      const float d = v[e] - local_mean;
      local_m2 += d * d;
    }
    const WelfordStat stat = cta_reduce_welford(
        WelfordStat{float(kElems), local_mean, local_m2},
        warp, lane, num_warps, s_scratch_a, s_scratch_b, s_scratch_c);
    mean = stat.mean;
    const float var = stat.m2 * params.inv_d;
    factor = math::rsqrt(var + params.eps);
  }

  // Normalize (+ optional affine), then round to the activation dtype before
  // the scale/shift epilogue (baseline contract).
  if constexpr (kHasWeightBias) {
    float w[kElems];
    load_operand_f32<WBDType, kElems, kOpRow>(w, params.weight, row, D, thread_elem);
    if constexpr (kIsRms) {
#pragma unroll
      for (int e = 0; e < kElems; ++e) {
        v[e] = v[e] * factor * w[e];  // rms ignores bias (baseline contract)
      }
    } else {
      float b[kElems];
      load_operand_f32<WBDType, kElems, kOpRow>(b, params.bias, row, D, thread_elem);
#pragma unroll
      for (int e = 0; e < kElems; ++e) {
        v[e] = (v[e] - mean) * factor * w[e] + b[e];
      }
    }
  } else {
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      v[e] = kIsRms ? v[e] * factor : (v[e] - mean) * factor;
    }
  }
#pragma unroll
  for (int e = 0; e < kElems; ++e) {
    v[e] = static_cast<float>(static_cast<DType>(v[e]));
  }

  float sc[kElems], sh[kElems];
  if constexpr (kEarlyScaleShiftLoad) {
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      sc[e] = static_cast<float>(sc_raw.get(e));
      sh[e] = static_cast<float>(sh_raw.get(e));
    }
  } else {
    load_operand_f32<ParamDType, kElems, kScClass>(sc, params.scale, row, D, thread_elem);
    load_operand_f32<ParamDType, kElems, kScClass>(sh, params.shift, row, D, thread_elem);
  }

  VecArray<DType, kElems> yv;
#pragma unroll
  for (int e = 0; e < kElems; ++e) {
    yv.set(e, static_cast<DType>(v[e] * (1.0f + sc[e]) + sh[e]));
  }
  yv.store(static_cast<DType*>(params.y) + base_elem);

  PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Host launchers (one struct per public-entry arity; the Python dispatcher
// flattens [B, S, D] activations to [num_rows, D] views and canonicalizes
// scale/shift/gate to [1], [D], or [num_rows, D] views before calling).
// ---------------------------------------------------------------------------

namespace detail {

template <typename DType, int kVecBytes>
inline auto launch_geometry(int64_t num_rows, int64_t hidden) {
  using namespace host;
  constexpr int kElems = kVecBytes / int(sizeof(DType));
  RuntimeCheck(hidden % 256 == 0 && hidden <= 8192, "hidden dim must be a multiple of 256 and <= 8192");
  RuntimeCheck(hidden % kElems == 0, "hidden dim must align with the vector width");
  const int64_t block = hidden / kElems;
  RuntimeCheck(block >= int64_t(device::kWarpThreads) && block <= 1024 && block % device::kWarpThreads == 0,
               "unsupported block size for hidden dim");
  RuntimeCheck(num_rows > 0 && num_rows <= int64_t(UINT32_MAX), "row count out of range");
  return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(num_rows), static_cast<uint32_t>(block)};
}

template <int kClass, typename T>
inline void verify_operand(
    const tvm::ffi::TensorView t, host::SymbolicSize& N, host::SymbolicSize& D, host::SymbolicDevice& device) {
  using namespace host;
  if constexpr (kClass == kOpScalar) {
    TensorMatcher({1}).with_dtype<T>().with_device(device).verify(t);
  } else if constexpr (kClass == kOpRow) {
    TensorMatcher({D}).with_dtype<T>().with_device(device).verify(t);
  } else {
    TensorMatcher({N, D}).with_dtype<T>().with_device(device).verify(t);
  }
}

}  // namespace detail

template <
    typename DType, typename ParamDType, int kScClass, bool kIsRms, bool kTwoPass, bool kUsePDL,
    bool kEarlyScaleShiftLoad, int kVecBytes>
struct NormScaleShiftKernel {
  static constexpr auto kernel = norm_scale_shift_kernel<
      DType, ParamDType, DType, DType, kIsRms, kScClass, kOpAbsent,
      /*kHasResidual=*/false, /*kHasWeightBias=*/false, kTwoPass, kUsePDL, kEarlyScaleShiftLoad, kVecBytes>;

  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      double eps) {
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    auto D = SymbolicSize{"hidden"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, D}).with_dtype<DType>().with_device(device).verify(x).verify(y);
    detail::verify_operand<kScClass, ParamDType>(scale, N, D, device);
    detail::verify_operand<kScClass, ParamDType>(shift, N, D, device);

    const auto [grid, block] = detail::launch_geometry<DType, kVecBytes>(N.unwrap(), D.unwrap());
    const auto params = NormScaleShiftParams{
        .y = y.data_ptr(),
        .res_out = nullptr,
        .x = x.data_ptr(),
        .residual = nullptr,
        .gate = nullptr,
        .weight = nullptr,
        .bias = nullptr,
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .num_rows = N.unwrap(),
        .eps = static_cast<float>(eps),
        .inv_d = 1.0f / static_cast<float>(D.unwrap()),
    };
    LaunchKernel(grid, block, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

template <
    typename DType, typename GateDType, typename ParamDType,
    int kGateClass,  // kOpAbsent for the gate-free wrapper arity below
    int kScClass, bool kIsRms, bool kTwoPass, bool kUsePDL, bool kEarlyScaleShiftLoad, int kVecBytes>
struct ScaleResidualNormScaleShiftKernel {
  static constexpr auto kernel = norm_scale_shift_kernel<
      DType, ParamDType, GateDType, DType, kIsRms, kScClass, kGateClass,
      /*kHasResidual=*/true, /*kHasWeightBias=*/false, kTwoPass, kUsePDL, kEarlyScaleShiftLoad, kVecBytes>;

  static void launch(
      tvm::ffi::TensorView y,
      tvm::ffi::TensorView res_out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView x,
      const void* gate_ptr,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      double eps) {
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    auto D = SymbolicSize{"hidden"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, D}).with_dtype<DType>().with_device(device).verify(x).verify(residual).verify(y).verify(
        res_out);
    detail::verify_operand<kScClass, ParamDType>(scale, N, D, device);
    detail::verify_operand<kScClass, ParamDType>(shift, N, D, device);

    const auto [grid, block] = detail::launch_geometry<DType, kVecBytes>(N.unwrap(), D.unwrap());
    const auto params = NormScaleShiftParams{
        .y = y.data_ptr(),
        .res_out = res_out.data_ptr(),
        .x = x.data_ptr(),
        .residual = residual.data_ptr(),
        .gate = gate_ptr,
        .weight = nullptr,
        .bias = nullptr,
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .num_rows = N.unwrap(),
        .eps = static_cast<float>(eps),
        .inv_d = 1.0f / static_cast<float>(D.unwrap()),
    };
    LaunchKernel(grid, block, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }

  // Gated arity (kGateClass must be kOpRow or kOpToken).
  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView res_out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView gate,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      double eps) {
    static_assert(kGateClass != kOpAbsent);
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    auto D = SymbolicSize{"hidden"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, D}).with_dtype<DType>().with_device(device).verify(x);
    detail::verify_operand<kGateClass, GateDType>(gate, N, D, device);
    launch(y, res_out, residual, x, gate.data_ptr(), scale, shift, eps);
  }

  // Gate-free arity (kGateClass must be kOpAbsent).
  static void
  run_nogate(
      tvm::ffi::TensorView y,
      tvm::ffi::TensorView res_out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      double eps) {
    static_assert(kGateClass == kOpAbsent);
    launch(y, res_out, residual, x, nullptr, scale, shift, eps);
  }
};

template <
    typename DType, typename GateDType, typename WBDType, typename ParamDType,
    int kGateClass, int kScClass, bool kIsRms, bool kTwoPass, bool kUsePDL, bool kEarlyScaleShiftLoad,
    int kVecBytes>
struct ScaleResidualNormScaleShiftAffineKernel {
  static constexpr auto kernel = norm_scale_shift_kernel<
      DType, ParamDType, GateDType, WBDType, kIsRms, kScClass, kGateClass,
      /*kHasResidual=*/true, /*kHasWeightBias=*/true, kTwoPass, kUsePDL, kEarlyScaleShiftLoad, kVecBytes>;

  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView res_out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView gate,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView bias,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      double eps) {
    static_assert(kGateClass != kOpAbsent);
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    auto D = SymbolicSize{"hidden"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, D}).with_dtype<DType>().with_device(device).verify(x).verify(residual).verify(y).verify(
        res_out);
    detail::verify_operand<kGateClass, GateDType>(gate, N, D, device);
    TensorMatcher({D}).with_dtype<WBDType>().with_device(device).verify(weight).verify(bias);
    detail::verify_operand<kScClass, ParamDType>(scale, N, D, device);
    detail::verify_operand<kScClass, ParamDType>(shift, N, D, device);

    const auto [grid, block] = detail::launch_geometry<DType, kVecBytes>(N.unwrap(), D.unwrap());
    const auto params = NormScaleShiftParams{
        .y = y.data_ptr(),
        .res_out = res_out.data_ptr(),
        .x = x.data_ptr(),
        .residual = residual.data_ptr(),
        .gate = gate.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = bias.data_ptr(),
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .num_rows = N.unwrap(),
        .eps = static_cast<float>(eps),
        .inv_d = 1.0f / static_cast<float>(D.unwrap()),
    };
    LaunchKernel(grid, block, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace kda_norm_scale_shift
