/**
 * \file rmsnorm.cuh
 * \brief RMSNorm and fused residual-add RMSNorm.
 *
 * The `CopyMode` staging paths take their idea from flashinfer's CuTe DSL
 * rmsnorm:
 *   https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15.post1/flashinfer/norm/kernels/rmsnorm.py
 * RMSNorm reads its row twice -- once for the sum of squares, once to scale --
 * so something has to survive the reduction. flashinfer parks the *input* tile
 * in shared memory with `cp.async` rather than holding it in registers, and
 * says why in its own comment: at large hidden sizes the live values spill to
 * local memory. We borrow the trick for the *weight* tile instead, since the
 * input already streams straight into the accumulator here.
 * Credit to the reference implementation.
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <bit>
#include <cstdint>
#include <type_traits>

namespace sglang {

template <typename T_, int64_t kDim_, uint32_t kNumThreads_, uint32_t kVecSize_>
struct RMSNormCTATrait {
  using T = T_;
  static constexpr int64_t kDim = kDim_;

  static constexpr uint32_t kVecSize = kVecSize_;
  static_assert(kDim % kVecSize == 0, "vector size must divide the hidden size");
  // vectors in one row; the thread tile is allowed to over-cover it
  static constexpr uint32_t kNumVecs = kDim / kVecSize;
  // one row per block, so the caller's thread count *is* the block
  static constexpr uint32_t kNumThreads = kNumThreads_;
  static constexpr uint32_t kBlockSize = kNumThreads;
  static_assert(kBlockSize % device::kWarpThreads == 0, "CTA tile needs a whole number of warps");
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  // the second reduction level is a single warp over `s_warp_sum`
  static_assert(kNumWarps <= device::kWarpThreads, "too many warps to reduce in one warp");
  // the tile may over-cover the row; `in_bound` idles the surplus lanes
  static constexpr uint32_t kUnroll = host::div_ceil(kNumVecs, kNumThreads);
  static constexpr bool kAligned = (kNumThreads * kUnroll == kNumVecs);

  static uint32_t get_num_blocks(uint32_t num_tokens) {
    return num_tokens;
  }
  SGL_DEVICE static uint32_t get_batch_id() {
    return blockIdx.x;
  }
  SGL_DEVICE static uint32_t get_offset() {
    return threadIdx.x;
  }
  SGL_DEVICE static bool in_bound(uint32_t i, uint32_t offset) {
    if constexpr (kAligned) return true;
    // the padding fits in one step, so only the last one can be partial
    if constexpr ((kUnroll - 1) * kNumThreads <= kNumVecs) {
      if (i < kUnroll - 1) return true;
    }
    return offset + i * kNumThreads < kNumVecs;
  }
};

enum class CopyMode {
  REG = 0,
  CP_ASYNC = 1,
  TMA = 2,
};

template <typename T_, int64_t kDim_, uint32_t kNumThreads_, uint32_t kVecSize_>
struct RMSNormWarpTrait {
  using T = T_;
  static constexpr int64_t kDim = kDim_;

  static constexpr uint32_t kVecSize = kVecSize_;
  static_assert(kDim % kVecSize == 0, "vector size must divide the hidden size");
  // vectors in one row; the thread tile is allowed to over-cover it
  static constexpr uint32_t kNumVecs = kDim / kVecSize;
  static constexpr uint32_t kBlockSize = 128;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  // several rows share a warp, so `warp::reduce` needs power-of-two sub-groups
  static constexpr uint32_t kNumThreads = kNumThreads_;
  static_assert(std::has_single_bit(kNumThreads), "warp tile needs a power-of-two thread count");
  static_assert(kNumThreads <= device::kWarpThreads, "work-set spans >1 warp, use the CTA impl");
  // the tile may over-cover the row; `in_bound` idles the surplus lanes
  static constexpr uint32_t kUnroll = host::div_ceil(kNumVecs, kNumThreads);
  static constexpr bool kAligned = (kNumThreads * kUnroll == kNumVecs);

  static uint32_t get_num_blocks(uint32_t num_tokens) {
    constexpr uint32_t kNumWorkers = kBlockSize / kNumThreads;
    return host::div_ceil(num_tokens, kNumWorkers);
  }
  SGL_DEVICE static uint32_t get_batch_id() {
    return blockIdx.x * (kBlockSize / kNumThreads) + threadIdx.x / kNumThreads;
  }
  SGL_DEVICE static uint32_t get_offset() {
    return threadIdx.x % kNumThreads;
  }
  SGL_DEVICE static bool in_bound(uint32_t i, uint32_t offset) {
    if constexpr (kAligned) return true;
    // the padding fits in one step, so only the last one can be partial
    if constexpr ((kUnroll - 1) * kNumThreads <= kNumVecs) {
      if (i < kUnroll - 1) return true;
    }
    return offset + i * kNumThreads < kNumVecs;
  }
};

template <typename T, int64_t kDim, uint32_t kNumThreads, uint32_t kVecSize>
using RMSNormTrait = std::conditional_t<
    // a row a warp can cover: pack several rows per block
    kNumThreads <= device::kWarpThreads,
    RMSNormWarpTrait<T, kDim, kNumThreads, kVecSize>,
    RMSNormCTATrait<T, kDim, kNumThreads, kVecSize> >;

#define RMSNORM_KERNEL __global__ __launch_bounds__(Trait::kBlockSize)

struct RMSNormParams {
  // fused-add: ptr_0 = input (x in, normalized out), ptr_1 = residual (in, x + res out)
  // plain:     ptr_0 = input (x in),                 ptr_1 = output (normalized out)
  void* __restrict__ ptr_0;
  void* __restrict__ ptr_1;
  const void* __restrict__ weight;
  int64_t stride_0;
  int64_t stride_1;
  uint32_t num_tokens;
  float eps;
};

namespace device::ptx {

/// \brief Raw async-copy PTX backing `CopyMode`. `cp.async` moves 16 bytes per
/// thread; `cp.async.bulk` moves a whole contiguous run from a single thread
/// and signals an mbarrier, so it needs no tensor-map descriptor.
SGL_DEVICE void cp_async_16B(void* smem_dst, const void* gmem_src) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(gmem_src));
}

SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

SGL_DEVICE void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

SGL_DEVICE void mbarrier_init(uint64_t* bar, uint32_t count) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr), "r"(count));
}

SGL_DEVICE void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(addr), "r"(bytes));
}

SGL_DEVICE void tma_bulk_g2s(void* smem, const void* gmem, uint32_t bytes, uint64_t* bar) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
               "l"(gmem),
               "r"(bytes),
               "r"(b)
               : "memory");
}

SGL_DEVICE void mbarrier_wait(uint64_t* bar, uint32_t phase) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "WAIT_LOOP:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
      "@P bra WAIT_DONE;\n"
      "bra WAIT_LOOP;\n"
      "WAIT_DONE:\n"
      "}\n" ::"r"(addr),
      "r"(phase));
}

}  // namespace device::ptx

template <typename Trait, bool kFusedAdd, bool kCastXBeforeOutMul, bool kUsePDL, int kCopyMode>
RMSNORM_KERNEL void rmsnorm_kernel(const RMSNormParams params) {
  using namespace device;
  constexpr uint32_t kDim = Trait::kDim;
  constexpr uint32_t kUnroll = Trait::kUnroll;
  constexpr uint32_t kVecSize = Trait::kVecSize;
  constexpr uint32_t kBlockSize = Trait::kBlockSize;
  constexpr uint32_t kNumThreads = Trait::kNumThreads;
  constexpr uint32_t kNumWarps = Trait::kNumWarps;
  constexpr CopyMode kMode = static_cast<CopyMode>(kCopyMode);
  using T = typename Trait::T;
  using T2 = packed_t<T>;
  using vec_t = AlignedVector<T2, kVecSize / 2>;
  static_assert(kVecSize % 2 == 0);
  static_assert(kCopyMode >= 0 && kCopyMode <= 2);

  // the staged tile is one (possibly over-covered) row of vectors, so the warp
  // trait's several workers share it rather than each keeping a private copy
  __shared__ alignas(128) vec_t s_stage[kMode == CopyMode::REG ? 1 : kNumThreads * kUnroll];
  __shared__ uint64_t s_barrier;  // mem barrier

  auto batch_id = Trait::get_batch_id();
  const auto offset = Trait::get_offset();
  if constexpr (Trait::kBlockSize != Trait::kNumThreads) {
    static_assert(Trait::kNumThreads <= kWarpThreads);
    if (batch_id >= params.num_tokens) {
      constexpr uint32_t kNumItemsPerWarp = kWarpThreads / Trait::kNumThreads;
      const auto masked_batch_id = batch_id / kNumItemsPerWarp * kNumItemsPerWarp;
      if (masked_batch_id >= params.num_tokens) return;
      batch_id = masked_batch_id;
    }
  }

  if constexpr (kMode == CopyMode::CP_ASYNC) {
    static_assert(kVecSize * sizeof(T) == 16, "cp.async & load smem only supports 16B");
  } else if constexpr (kMode == CopyMode::TMA) {
    static_assert(kVecSize * sizeof(T) == 16, "load smem only supports 16B");
    // one bulk copy serves the whole block, and the wait below is the only
    // block-wide sync the warp trait would have -- keep it to one row per block
    static_assert(kBlockSize == kNumThreads, "TMA staging is CTA-trait only");
    if (threadIdx.x < kWarpThreads) {
      if (warp::elect_one_lane()) {
        // only the electing lane arrives; the tx-byte count completes the rest
        device::ptx::mbarrier_init(&s_barrier, 1);
      }
      __syncwarp();
    }
  }

  vec_t inp[kUnroll];
  vec_t weight[kMode == CopyMode::REG ? kUnroll : 1];

  const auto mem = tile::Memory<vec_t>{offset, kNumThreads};
  const auto ptr_0 = pointer::offset<T>(params.ptr_0, batch_id * params.stride_0);
  const auto ptr_1 = pointer::offset<T>(params.ptr_1, batch_id * params.stride_1);

  PDLWaitPrimary<kUsePDL>();

  const auto load_weight = [&] {
    if constexpr (kMode == CopyMode::REG) {
#pragma unroll
      for (uint32_t i = 0; i < kUnroll; ++i) {
        if (Trait::in_bound(i, offset)) {
          weight[i] = mem.load(params.weight, i);
        }
      }
    } else if constexpr (kMode == CopyMode::CP_ASYNC) {
#pragma unroll
      for (uint32_t i = 0; i < kUnroll; ++i) {
        if (Trait::in_bound(i, offset)) {
          const auto pos = i * kNumThreads + offset;
          const auto ptr = pointer::offset<vec_t>(params.weight, pos);
          ptx::cp_async_16B(&s_stage[pos], ptr);
        }
      }
      // without a group to wait on, `cp.async.wait_group 0` is a no-op
      ptx::cp_async_commit();
    } else {
      if (threadIdx.x < kWarpThreads) {
        if (warp::elect_one_lane()) {
          constexpr uint32_t kNumBytes = kDim * sizeof(T);
          // the expected byte count has to be registered before the copy can
          // complete against it, so arrive first and issue second
          ptx::mbarrier_arrive_expect_tx(&s_barrier, kNumBytes);
          ptx::tma_bulk_g2s(&s_stage[0], params.weight, kNumBytes, &s_barrier);
        }
        __syncwarp();
      }
    }
  };

  if constexpr (kFusedAdd) {
    // first load inp & res interleave
    vec_t res[kUnroll];
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) {
      if (Trait::in_bound(i, offset)) {
        inp[i] = mem.load(ptr_0, i);
        res[i] = mem.load(ptr_1, i);
      }
    }

    load_weight();

    // fused residual add
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < kVecSize / 2; ++j) {
        using AddTrait = ReductionTrait<ReductionOp::SUM, T2>;
        inp[i][j] = AddTrait::reduce(inp[i][j], res[i][j]);
      }
      if (Trait::in_bound(i, offset)) {
        mem.store(ptr_1, inp[i], i);
      } else {
        inp[i].fill(T2{});
      }
    }
  } else {
    // first load inp
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) {
      if (Trait::in_bound(i, offset)) {
        inp[i] = mem.load(ptr_0, i);
      } else {
        inp[i].fill(T2{});
      }
    }

    // the staged modes already issued it above
    load_weight();
  }

  float local_sum = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kUnroll; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      const auto [x, y] = cast<fp32x2_t>(inp[i][j]);
      /// NOTE: in this form, the compiler can more easily generate FMA
      local_sum += x * x;
      local_sum += y * y;
    }
  }

  constexpr float kInvDim = 1.0f / kDim;
  float scale;

  // choose intra-block reduction / intra-warp reduction
  if constexpr (kBlockSize == kNumThreads) {
    const auto tx = threadIdx.x;
    const auto warp_id = tx / kWarpThreads;
    __shared__ float s_warp_sum[kNumWarps];
    s_warp_sum[warp_id] = warp::reduce_sum(local_sum);
    __syncthreads();

    __shared__ float s_scale;
    if (warp_id == 0) {
      const auto warp_sum = tx < kNumWarps ? s_warp_sum[tx] : 0.0f;
      const auto sum = warp::reduce_sum(warp_sum);
      s_scale = rsqrtf(sum * kInvDim + params.eps);
    } else {
      PDLTriggerSecondary<kUsePDL>();
    }
    __syncthreads();
    scale = s_scale;
  } else {
    const auto sum = warp::reduce_sum<kNumThreads>(local_sum);
    PDLTriggerSecondary<kUsePDL>();
    scale = rsqrtf(sum * kInvDim + params.eps);
  }

  // async copy barrier here
  if constexpr (kMode == CopyMode::CP_ASYNC) {
    ptx::cp_async_wait_all();
  } else if constexpr (kMode == CopyMode::TMA) {
    ptx::mbarrier_wait(&s_barrier, 0);
  }

  // the fused variant normalizes in place, the plain one writes to `ptr_1`
  const auto out_ptr = kFusedAdd ? ptr_0 : ptr_1;
#pragma unroll
  for (uint32_t i = 0; i < kUnroll; ++i) {
    if (Trait::in_bound(i, offset)) {
      vec_t out;
      vec_t w_vec;
      if constexpr (kMode == CopyMode::REG) {
        w_vec = weight[i];
      } else {
        w_vec = mem.load(s_stage, i);
      }

#pragma unroll
      for (uint32_t j = 0; j < kVecSize / 2; ++j) {
        auto v = cast<fp32x2_t>(inp[i][j]);
        const auto w = cast<fp32x2_t>(w_vec[j]);
        v.x *= scale;
        v.y *= scale;
        if constexpr (kCastXBeforeOutMul) {
          // HF semantics: round to the storage type before the weight multiply
          v = cast<fp32x2_t>(cast<T2>(v));
        }
        out[j] = cast<T2>(fp32x2_t{v.x * w.x, v.y * w.y});
      }
      mem.store(out_ptr, out, i);
    }
  }
}

/// \brief The vectorized loads need the row start on a `kVecBytes` boundary.
///
/// The tensor matcher constrains neither half of that: a column slice offsets
/// the base pointer, and `with_strides({-1, 1})` leaves the row stride free, so
/// a non-contiguous view can put every row after the first off the boundary.
template <uint32_t kVecBytes>
inline void check_ptr_alignment(const void* ptr, const char* name) {
  CHECK_HOST(reinterpret_cast<uintptr_t>(ptr) % kVecBytes == 0)
      << name << ": base pointer must be " << kVecBytes << "-byte aligned";
}

template <uint32_t kVecBytes, typename T>
inline void check_row_alignment(const tvm::ffi::TensorView view, const char* name) {
  check_ptr_alignment<kVecBytes>(view.data_ptr(), name);
  const auto stride_bytes = view.stride(0) * static_cast<int64_t>(sizeof(T));
  CHECK_HOST(stride_bytes % static_cast<int64_t>(kVecBytes) == 0)
      << name << ": row stride of " << stride_bytes << " bytes must be a multiple of " << kVecBytes;
}

/// \brief `kVecSize` / `kNumThreads` are the tuning knobs; they come from
/// `sglang.kernels.ops.layernorm.norm` so that retuning costs one JIT compile
/// per new configuration instead of invalidating every module built from this
/// file. The static_asserts in the traits are the contract they must meet.
template <
    typename T,
    int64_t kDim,
    bool kUsePDL,
    bool kCastXBeforeOutMul,
    uint32_t kVecSize,
    uint32_t kNumThreads,
    int kCopyMode>
struct FusedAddRMSNormKernel {
  static constexpr uint32_t kVecBytes = kVecSize * sizeof(T);
  using Trait = RMSNormTrait<T, kDim, kNumThreads, kVecSize>;
  static constexpr auto kernel = rmsnorm_kernel<Trait, true, kCastXBeforeOutMul, kUsePDL, kCopyMode>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView weight,
      const float eps) {
    using namespace host;
    constexpr int64_t D = kDim;
    auto N = SymbolicSize{"num_tokens"};
    auto device_sym = SymbolicDevice{};
    device_sym.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // input
        .with_strides({-1, 1})
        .with_dtype<T>()
        .with_device(device_sym)
        .verify(input);
    TensorMatcher({N, D})  // residual
        .with_strides({-1, 1})
        .with_dtype<T>()
        .with_device(device_sym)
        .verify(residual);
    TensorMatcher({D})  // weight
        .with_dtype<T>()
        .with_device(device_sym)
        .verify(weight);

    check_row_alignment<kVecBytes, T>(input, "input");
    check_row_alignment<kVecBytes, T>(residual, "residual");
    check_ptr_alignment<kVecBytes>(weight.data_ptr(), "weight");

    const auto device = device_sym.unwrap();
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = RMSNormParams{
        .ptr_0 = input.data_ptr(),
        .ptr_1 = residual.data_ptr(),
        .weight = weight.data_ptr(),
        .stride_0 = input.stride(0),
        .stride_1 = residual.stride(0),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    const auto num_blocks = Trait::get_num_blocks(num_tokens);
    LaunchKernel(num_blocks, Trait::kBlockSize, device)  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

/// \brief See `FusedAddRMSNormKernel` on where `kVecSize` / `kNumThreads`
/// come from. The caller also owns the latency-vs-throughput choice, so this
/// is instantiated once per schedule rather than branching on the token count.
template <
    typename T,
    int64_t kDim,
    bool kUsePDL,
    bool kCastXBeforeOutMul,
    uint32_t kVecSize,
    uint32_t kNumThreads,
    int kCopyMode>
struct RMSNormKernel {
  static constexpr uint32_t kVecBytes = kVecSize * sizeof(T);
  using Trait = RMSNormTrait<T, kDim, kNumThreads, kVecSize>;
  static constexpr auto kernel = rmsnorm_kernel<Trait, false, kCastXBeforeOutMul, kUsePDL, kCopyMode>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      const float eps) {
    using namespace host;
    constexpr int64_t D = kDim;
    auto N = SymbolicSize{"num_tokens"};
    auto device_sym = SymbolicDevice{};
    device_sym.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // input
        .with_strides({-1, 1})
        .with_dtype<T>()
        .with_device(device_sym)
        .verify(input);
    TensorMatcher({D})  // weight
        .with_dtype<T>()
        .with_device(device_sym)
        .verify(weight);
    TensorMatcher({N, D})  // output
        .with_strides({-1, 1})
        .with_dtype<T>()
        .with_device(device_sym)
        .verify(output);

    check_row_alignment<kVecBytes, T>(input, "input");
    check_row_alignment<kVecBytes, T>(output, "output");
    check_ptr_alignment<kVecBytes>(weight.data_ptr(), "weight");

    const auto device = device_sym.unwrap();
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = RMSNormParams{
        .ptr_0 = input.data_ptr(),
        .ptr_1 = output.data_ptr(),
        .weight = weight.data_ptr(),
        .stride_0 = input.stride(0),
        .stride_1 = output.stride(0),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    const auto num_blocks = Trait::get_num_blocks(num_tokens);
    LaunchKernel(num_blocks, Trait::kBlockSize, device)  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
