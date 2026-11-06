#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace sglang {

// Python passes the enumerator name as the template argument (`using enum` below)
enum class RMSNormMode {
  PLAIN,            // out = norm(x) * w
  FUSED_ADD,        // res = T(x + res); x = norm(res) * w
  QUANT,            // out8 = fp8(norm(x) * w / scale)
  FUSED_ADD_QUANT,  // both
};
using enum RMSNormMode;

// One row per `kThreadsPerRow` threads (whole warps, or a power-of-two slice of one warp for short
// rows), `kRowsPerCTA` rows per CTA, `kVecBytes` per access. The thread tile may over-cover the row;
// `in_bound` idles the surplus vectors.
template <typename T_, int64_t kDim_, uint32_t kVecBytes_, uint32_t kThreadsPerRow_, uint32_t kRowsPerCTA_>
struct RMSNormTile {
  using T = T_;
  static constexpr int64_t kDim = kDim_;
  static constexpr uint32_t kVecBytes = kVecBytes_;
  static constexpr uint32_t kVecElems = kVecBytes / sizeof(T);
  static constexpr uint32_t kPacks = kVecElems / 2;
  static_assert(kVecBytes == 16 || kVecBytes == 32);
  static_assert(kDim % kVecElems == 0, "hidden size must be a whole number of vectors");
  static constexpr uint32_t kNumVecs = kDim / kVecElems;

  static constexpr uint32_t kThreadsPerRow = kThreadsPerRow_;
  static_assert(
      kThreadsPerRow % device::kWarpThreads == 0 ||
          (device::kWarpThreads % kThreadsPerRow == 0 && host::is_pow2(kThreadsPerRow)),
      "a row takes whole warps or a power-of-two slice of one warp");
  static constexpr uint32_t kWarpsPerRow = host::div_ceil(kThreadsPerRow, device::kWarpThreads);
  static constexpr uint32_t kRowsPerCTA = kRowsPerCTA_;
  static constexpr uint32_t kNumThreads = kThreadsPerRow * kRowsPerCTA;
  static_assert(kNumThreads % device::kWarpThreads == 0 && kNumThreads <= 1024, "whole warps per CTA");

  static constexpr uint32_t kVecsPerThread = host::div_ceil(kNumVecs, kThreadsPerRow);
  static constexpr bool kAligned = (kVecsPerThread * kThreadsPerRow == kNumVecs);
  static_assert(kVecsPerThread <= 16, "too many vectors per thread; use more threads per row");

  SGL_DEVICE static constexpr bool in_bound(uint32_t i, uint32_t tid) {
    if constexpr (kAligned) return true;
    // only the last step can be partial
    if constexpr ((kVecsPerThread - 1) * kThreadsPerRow <= kNumVecs) {
      if (i + 1 < kVecsPerThread) return true;
    }
    return i * kThreadsPerRow + tid < kNumVecs;
  }
};

// Any inline asm earlier in the kernel (`griddepcontrol.wait` included) makes nvcc 13.2-13.4 lower
// the vector loads after it to generic `LD.E.128` pairs; stating the address space right before the
// access restores `LDG.E.256`.
template <typename T>
SGL_DEVICE T* assume_global(T* ptr) {
#ifndef USE_ROCM
  __builtin_assume(__isGlobal(ptr));
#endif
  return ptr;
}

struct RMSNormParams {
  const void* __restrict__ input;
  const void* __restrict__ weight;
  void* __restrict__ output;        // PLAIN / QUANT only; the fused modes rewrite `input`
  void* __restrict__ residual;      // fused modes: residual in, x + residual out
  const float* __restrict__ scale;  // quant modes: per-tensor fp8 scale, device scalar
  int64_t input_stride;
  int64_t output_stride;
  int64_t residual_stride;
  uint32_t num_tokens;
  float eps;
};

// One row tile per CTA. The row stays in registers between its two uses, the weight slice is loaded
// before `griddepcontrol.wait`, and rows spanning several warps reduce through shared memory with one
// barrier. Rows past `num_tokens` (the last packed CTA's surplus) load zeros, store nothing and still
// join the barrier.
template <typename Tile, RMSNormMode kMode, bool kCastXBeforeOutMul, bool kUsePDL>
__global__ __launch_bounds__(Tile::kNumThreads) void rmsnorm_kernel(const RMSNormParams __grid_constant__ params) {
  using namespace device;
  using T = typename Tile::T;
  using T2 = packed_t<T>;
  using vec_t = AlignedVector<T2, Tile::kPacks>;
  constexpr uint32_t kPacks = Tile::kPacks;
  constexpr uint32_t kVecsPerThread = Tile::kVecsPerThread;
  constexpr uint32_t kThreadsPerRow = Tile::kThreadsPerRow;
  constexpr uint32_t kWarpsPerRow = Tile::kWarpsPerRow;
  constexpr bool kFusedAdd = (kMode == RMSNormMode::FUSED_ADD || kMode == RMSNormMode::FUSED_ADD_QUANT);
  constexpr bool kQuant = (kMode == RMSNormMode::QUANT || kMode == RMSNormMode::FUSED_ADD_QUANT);
  constexpr float kInvDim = 1.0f / static_cast<float>(Tile::kDim);
  // `fma.rn.f32.bf16` is a Blackwell instruction; elsewhere it is a convert pair plus an FMA
  constexpr bool kMixedFMA = std::is_same_v<T, bf16_t> && SGL_ARCH_BLACKWELL_OR_GREATER;
  static_assert(sizeof(T) == 2, "fp16 / bf16 only");

  const auto tid = threadIdx.x % kThreadsPerRow;
  const auto row_in_cta = threadIdx.x / kThreadsPerRow;
  const auto warp_in_row = tid / kWarpThreads;
  const auto lane_id = tid % kWarpThreads;
  const auto row = blockIdx.x * Tile::kRowsPerCTA + row_in_cta;
  // with one row per CTA the grid is exactly num_tokens; only packed CTAs have surplus rows
  const bool valid = Tile::kRowsPerCTA == 1 || row < params.num_tokens;

  __shared__ float s_partial[Tile::kRowsPerCTA][kWarpsPerRow];

  vec_t weight[kVecsPerThread];
#pragma unroll
  for (uint32_t i = 0; i < kVecsPerThread; ++i) {
    if (Tile::in_bound(i, tid)) {
      weight[i].load(params.weight, i * kThreadsPerRow + tid);
    } else {
      weight[i].fill(T2{});
    }
  }
  [[maybe_unused]] float scale_inv = 1.0f;
  if constexpr (kQuant) {
    scale_inv = 1.0f / params.scale[0];
  }

  PDLWaitPrimary<kUsePDL>();

  const auto input_ptr = assume_global(pointer::offset<T>(params.input, row * params.input_stride));
  vec_t x[kVecsPerThread];
#pragma unroll
  for (uint32_t i = 0; i < kVecsPerThread; ++i) {
    if (valid && Tile::in_bound(i, tid)) {
      x[i].load(input_ptr, i * kThreadsPerRow + tid);
    } else {
      x[i].fill(T2{});
    }
  }

  if constexpr (kFusedAdd) {
    const auto residual_ptr = assume_global(pointer::offset<T>(params.residual, row * params.residual_stride));
    vec_t res[kVecsPerThread];
#pragma unroll
    for (uint32_t i = 0; i < kVecsPerThread; ++i) {
      if (valid && Tile::in_bound(i, tid)) {
        res[i].load(residual_ptr, i * kThreadsPerRow + tid);
      } else {
        res[i].fill(T2{});
      }
    }
    // the residual stream is 16-bit: the packed add rounds the exact sum once (the same value as
    // f32(x) + f32(res) rounded to T) and the rounded value is what gets normalized
#pragma unroll
    for (uint32_t i = 0; i < kVecsPerThread; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < kPacks; ++j) {
        x[i][j] = ReductionTrait<ReductionOp::SUM, T2>::reduce(x[i][j], res[i][j]);
      }
      if (valid && Tile::in_bound(i, tid)) {
        x[i].store(residual_ptr, i * kThreadsPerRow + tid);
      }
    }
  }

  // two accumulation chains; on Blackwell the mixed fma squares the 16-bit value without a convert
  float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kVecsPerThread; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < kPacks; ++j) {
      if constexpr (kMixedFMA) {
        const auto& e = unpack(x[i][j]);
        acc0 = math::fma_f32_bf16(e[0], e[0], acc0);
        acc1 = math::fma_f32_bf16(e[1], e[1], acc1);
      } else {
        const auto v = cast<fp32x2_t>(x[i][j]);
        acc0 = fmaf(v.x, v.x, acc0);
        acc1 = fmaf(v.y, v.y, acc1);
      }
    }
  }
  // rows shorter than a warp reduce over their own lane slice
  constexpr uint32_t kReduceLanes = std::min(kThreadsPerRow, kWarpThreads);
  float sum = warp::reduce_sum<kReduceLanes>(acc0 + acc1);
  if constexpr (kWarpsPerRow > 1) {
    if (lane_id == 0) s_partial[row_in_cta][warp_in_row] = sum;
    __syncthreads();
    sum = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kWarpsPerRow; ++w) {
      sum += s_partial[row_in_cta][w];
    }
  }
  const float r = math::rsqrt(sum * kInvDim + params.eps);

  // `factor` multiplies the normalized product: r for bf16 output, r / scale for fp8. x * w is
  // exact in fp32, so the product-then-scale form rounds once and needs no converts; the -0 addend
  // keeps the sign of a zero product. The HF form must round x * r to T before the weight multiply.
  const float factor = kQuant ? r * scale_inv : r;
  [[maybe_unused]] const float hf_post = kQuant ? scale_inv : 1.0f;
  const auto scaled = [&](uint32_t i, uint32_t j) -> fp32x2_t {
    if constexpr (kCastXBeforeOutMul) {
      auto v = cast<fp32x2_t>(x[i][j]);
      const auto w = cast<fp32x2_t>(weight[i][j]);
      v = cast<fp32x2_t>(cast<T2>(fp32x2_t{v.x * r, v.y * r}));
      return fp32x2_t{v.x * w.x * hf_post, v.y * w.y * hf_post};
    } else if constexpr (kMixedFMA) {
      const auto& e = unpack(x[i][j]);
      const auto& g = unpack(weight[i][j]);
      return fp32x2_t{math::fma_f32_bf16(e[0], g[0], -0.0f) * factor, math::fma_f32_bf16(e[1], g[1], -0.0f) * factor};
    } else {
      const auto v = cast<fp32x2_t>(x[i][j]);
      const auto w = cast<fp32x2_t>(weight[i][j]);
      return fp32x2_t{v.x * w.x * factor, v.y * w.y * factor};
    }
  };

  if constexpr (kQuant) {
    using out_vec_t = AlignedVector<fp8x2_e4m3_t, kPacks>;
    const auto output_ptr = assume_global(pointer::offset<fp8_e4m3_t>(params.output, row * params.output_stride));
#pragma unroll
    for (uint32_t i = 0; i < kVecsPerThread; ++i) {
      out_vec_t out;
#pragma unroll
      for (uint32_t j = 0; j < kPacks; ++j) {
        out[j] = cast<fp8x2_e4m3_t>(scaled(i, j));
      }
      if (valid && Tile::in_bound(i, tid)) {
        out.store(output_ptr, i * kThreadsPerRow + tid);
      }
    }
  } else {
    const auto out_base = kFusedAdd ? const_cast<void*>(params.input) : params.output;
    const auto out_stride = kFusedAdd ? params.input_stride : params.output_stride;
    const auto output_ptr = assume_global(pointer::offset<T>(out_base, row * out_stride));
#pragma unroll
    for (uint32_t i = 0; i < kVecsPerThread; ++i) {
      vec_t out;
#pragma unroll
      for (uint32_t j = 0; j < kPacks; ++j) {
        out[j] = cast<T2>(scaled(i, j));
      }
      if (valid && Tile::in_bound(i, tid)) {
        out.store(output_ptr, i * kThreadsPerRow + tid);
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

// `kVecBytes`, `kThreadsPerRow` and `kRowsPerCTA` are the schedule chosen in Python
// (`sglang.kernels.ops.layernorm.norm`), one JIT module per schedule. The fused modes rewrite
// `input` in place and ignore `output`; `scale` is the fp32 device scalar of the quant modes.
template <
    typename T,
    int64_t kDim,
    RMSNormMode kMode,
    bool kCastXBeforeOutMul,
    bool kUsePDL,
    uint32_t kVecBytes,
    uint32_t kThreadsPerRow,
    uint32_t kRowsPerCTA>
struct RMSNormKernel {
  static constexpr bool kFusedAdd = (kMode == RMSNormMode::FUSED_ADD || kMode == RMSNormMode::FUSED_ADD_QUANT);
  static constexpr bool kQuant = (kMode == RMSNormMode::QUANT || kMode == RMSNormMode::FUSED_ADD_QUANT);
#ifdef USE_ROCM
  static_assert(!kQuant, "fp8 quant modes are CUDA-only");
#endif
  static_assert(kVecBytes <= device::kMaxVecBytes, "vector width exceeds this arch");
  using Tile = RMSNormTile<T, kDim, kVecBytes, kThreadsPerRow, kRowsPerCTA>;
  static constexpr auto kernel = rmsnorm_kernel<Tile, kMode, kCastXBeforeOutMul, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> output,
      const tvm::ffi::Optional<tvm::ffi::TensorView> residual,
      const tvm::ffi::Optional<tvm::ffi::TensorView> scale,
      const float eps) {
    using namespace host;
    constexpr int64_t D = kDim;
    auto N = SymbolicSize{"num_tokens"};
    auto device_sym = SymbolicDevice{};
    device_sym.set_options<kDLGPU>();

    // the 32 B accesses need the base pointers and the row strides aligned
    TensorMatcher({N, D})  // input
        .with_strides({-1, 1})
        .with_dtype<T>()
        .with_device(device_sym)
        .ensure_alignment(kVecBytes)
        .verify(input);
    TensorMatcher({D})  // weight
        .with_dtype<T>()
        .with_device(device_sym)
        .ensure_alignment(kVecBytes)
        .verify(weight);

    RMSNormParams params{};
    params.input = input.data_ptr();
    params.weight = weight.data_ptr();
    params.input_stride = input.stride(0);
    params.eps = eps;
    if constexpr (kQuant) {
      CHECK_HOST(output.has_value() && scale.has_value()) << "quant modes need `output` and `scale`";
      TensorMatcher({N, D})  // fp8 output
          .with_strides({-1, 1})
          .with_dtype<fp8_e4m3_t>()
          .with_device(device_sym)
          .ensure_alignment(kVecBytes / 2)
          .verify(output.value());
      // Static FP8 loaders produce the per-tensor scale as either [1] or 0-D.
      const auto& scale_value = scale.value();
      CHECK_HOST(scale_value.numel() == 1 && is_type<fp32_t>(scale_value.dtype()))
          << "`scale` must hold a single fp32 element";
      CHECK_HOST(
          scale_value.device().device_type == input.device().device_type &&
          scale_value.device().device_id == input.device().device_id)
          << "`scale` must be on the same device as `input`";
      params.output = output.value().data_ptr();
      params.output_stride = output.value().stride(0);
      params.scale = static_cast<const float*>(scale_value.data_ptr());
    } else if constexpr (!kFusedAdd) {
      CHECK_HOST(output.has_value()) << "rmsnorm needs `output`";
      TensorMatcher({N, D})  // output
          .with_strides({-1, 1})
          .with_dtype<T>()
          .with_device(device_sym)
          .ensure_alignment(kVecBytes)
          .verify(output.value());
      params.output = output.value().data_ptr();
      params.output_stride = output.value().stride(0);
    }
    if constexpr (kFusedAdd) {
      CHECK_HOST(residual.has_value()) << "fused modes need `residual`";
      TensorMatcher({N, D})  // residual
          .with_strides({-1, 1})
          .with_dtype<T>()
          .with_device(device_sym)
          .ensure_alignment(kVecBytes)
          .verify(residual.value());
      params.residual = residual.value().data_ptr();
      params.residual_stride = residual.value().stride(0);
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    params.num_tokens = num_tokens;
    const auto num_blocks = div_ceil(num_tokens, kRowsPerCTA);
    LaunchKernel(num_blocks, Tile::kNumThreads, device_sym.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
