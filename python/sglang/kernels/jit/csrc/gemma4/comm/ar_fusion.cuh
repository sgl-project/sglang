// Gemma-4 fused TP all-reduce + layer epilogue + per-128 FP8 quant over NVLink
// multicast (bf16 in, block-FP8 out), for extend batches.
//
// Every dense Gemma-4 layer ends its attention and its MLP with
//
//   x = allreduce(partial); a = rmsnorm(x, w_a); r = (a + r) * s; y = rmsnorm(r, w_b);
//   q, scale = per_token_group_quant_fp8(y, 128)                 // feeds the next block-FP8 GEMM
//
// This kernel runs one such site in a single launch:
//   * token rows are sharded across the ranks: an owner reads its row already
//     summed over the team with `multimem.ld_reduce` (fp32 accumulation), so the
//     reduce-scatter happens in the switch;
//   * the residual r stays row-sharded for the whole forward -- only the owner
//     of a row ever reads or writes it;
//   * the FP8 row and its fp32 scales are `multimem.st`ored to every rank, which
//     is the all-gather (one egress copy, the switch fans out).
// kFinal (last layer) stops after the residual update and multicasts the bf16
// rows instead, so the caller's final norm sees every row.
//
// Barriers reuse the CustomAllReduceV2 pull plane: block b of every rank meets
// block b of every other rank through one `multimem.red` on the semaphore's
// multicast alias, twice per launch (all partials visible / all outputs
// landed). Every rank must launch the same grid, and the launch is clamped to
// the plane's block count.
//
// Rounding points match the unfused path (gemma4_fused_ops.py + the per-group
// quant), see `round_a` in Params. Hidden size and block width are template
// parameters: kHidden must be a multiple of 256 and kThreads a multiple of 32
// that divides kHidden / 8.
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace sglang {

namespace gemma4_ar {

using device::distributed::McBarrier;
using device::distributed::Semaphore;

struct Params {
  const uint8_t* partial_mc;  // [T, H] bf16, multicast VA: this rank's partial sums
  uint8_t* out_mc;            // fused: [T, H] fp8 multicast; final: [T, H] bf16 multicast (may alias partial_mc)
  float* scales_mc;           // [T, H / 128] fp32 multicast (fused only)
  __nv_bfloat16* residual;    // [rows, H] this rank's rows of the residual stream, updated in place
  const __nv_bfloat16* w_a;   // [H] norm before the residual add
  const __nv_bfloat16* w_b;   // [H] norm after it (fused only)
  const void* scalar;         // [1] layer scalar, fp32 or bf16 (scalar_bf16)
  Semaphore* sem_local;
  Semaphore* sem_mc;
  uint32_t rank;
  uint32_t world_size;
  uint32_t row0;  // first row this rank owns
  uint32_t rows;  // rows this rank owns
  float eps;
  bool scalar_bf16;
  // Attention site (true): the first norm's output is rounded to bf16 before the
  // add (unfused: a separate norm kernel) and the second norm reads the fp32 sum
  // (unfused: fused add + RMSNorm). MLP site (false): the add stays fp32
  // (gemma_rmsnorm_residual_scalar) and the next norm reads the bf16 stream.
  bool round_a;
};

SGL_DEVICE void unpack(const uint4& v, float* f) {
  const auto* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(p[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

SGL_DEVICE uint4 pack(const float* f) {
  uint4 v;
  auto* p = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return v;
}

SGL_DEVICE float bf16_round(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

// Shape constants and the per-row device code, for one hidden size and block width.
template <uint32_t kHidden, uint32_t kThreads>
struct RowEpilogue {
  static constexpr uint32_t kVec = 8;  // bf16 per 16-byte vector
  static constexpr uint32_t kNumVecs = kHidden / kVec;
  static constexpr uint32_t kPerThread = kNumVecs / kThreads;
  static constexpr uint32_t kGroup = 128;  // quant group (block_k of the FP8 weights)
  static constexpr uint32_t kGroupVecs = kGroup / kVec;
  static constexpr uint32_t kNumGroups = kHidden / kGroup;
  static constexpr uint32_t kNumWarps = kThreads / 32;
  static constexpr float kFp8Max = 448.f;
  static_assert(kHidden % 256 == 0, "hidden must be a multiple of 256");
  static_assert(kThreads % 32 == 0 && kThreads <= 1024, "block width");
  static_assert(kNumVecs % kThreads == 0, "kThreads must divide hidden / 8");
  static_assert(kThreads % kGroupVecs == 0, "a quant group never straddles a block pass");

  using vec_t = device::AlignedVector<bf16x2_t, 4>;

  SGL_DEVICE static float block_sum(float v, float* smem) {
    v = device::warp::reduce_sum(v);
    const auto warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();  // the previous reduction's readers are done with smem
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < kNumWarps; ++i)
      t += smem[i];
    return t;
  }

  SGL_DEVICE static void load_row(const Params& p, uint32_t lr, uint4* raw) {
    const int64_t vec0 = int64_t(p.row0 + lr) * kNumVecs;
#pragma unroll
    for (uint32_t i = 0; i < kPerThread; ++i) {
      vec_t v;
      device::ptx::ld_multimem_16B(v, p.partial_mc, vec0 + threadIdx.x + i * kThreads);
      raw[i] = *reinterpret_cast<const uint4*>(&v);
    }
  }

  SGL_DEVICE static void store_mc(uint8_t* mc, int64_t vec_offset, const uint4& v) {
    device::ptx::st_multimem_16B(*reinterpret_cast<const vec_t*>(&v), mc, vec_offset);
  }

  template <bool kFinal>
  SGL_DEVICE static void epilogue(const Params& p, uint32_t lr, const uint4* raw, float* smem) {
    const uint32_t t = threadIdx.x;
    const int64_t row = p.row0 + lr;
    float x[kPerThread][kVec];
    float ss = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < kPerThread; ++i) {
      unpack(raw[i], x[i]);
#pragma unroll
      for (uint32_t j = 0; j < kVec; ++j)
        ss += x[i][j] * x[i][j];
    }
    const float r1 = rsqrtf(block_sum(ss, smem) / kHidden + p.eps);
    const float s = p.scalar_bf16 ? __bfloat162float(*static_cast<const __nv_bfloat16*>(p.scalar))
                                  : *static_cast<const float*>(p.scalar);
    float ss2 = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < kPerThread; ++i) {
      const uint32_t c = t + i * kThreads;
      float w[kVec], r[kVec];
      unpack(*reinterpret_cast<const uint4*>(p.w_a + c * kVec), w);
      auto* rp = reinterpret_cast<uint4*>(p.residual + int64_t(lr) * kHidden + c * kVec);
      unpack(*rp, r);
#pragma unroll
      for (uint32_t j = 0; j < kVec; ++j) {
        const float n = x[i][j] * r1 * w[j];
        const float v = ((p.round_a ? bf16_round(n) : n) + r[j]) * s;
        x[i][j] = p.round_a ? v : bf16_round(v);
        ss2 += x[i][j] * x[i][j];
      }
      const uint4 packed = pack(x[i]);
      *rp = packed;
      if constexpr (kFinal) store_mc(p.out_mc, row * kNumVecs + c, packed);
    }
    if constexpr (kFinal) return;
    const float r2 = rsqrtf(block_sum(ss2, smem) / kHidden + p.eps);
#pragma unroll
    for (uint32_t i = 0; i < kPerThread; ++i) {
      const uint32_t c = t + i * kThreads;
      float w[kVec];
      unpack(*reinterpret_cast<const uint4*>(p.w_b + c * kVec), w);
      float amax = 0.f;
#pragma unroll
      for (uint32_t j = 0; j < kVec; ++j) {
        x[i][j] = bf16_round(x[i][j] * r2 * w[j]);
        amax = fmaxf(amax, fabsf(x[i][j]));
      }
      // the 16 lanes of a quant group are 16 consecutive threads
      amax = device::warp::reduce_max<kGroupVecs>(amax);
      const float scale = fmaxf(amax, 1e-10f) / kFp8Max;
      uint2 q;
      auto* qp = reinterpret_cast<__nv_fp8_e4m3*>(&q);
#pragma unroll
      for (uint32_t j = 0; j < kVec; ++j)
        qp[j] = __nv_fp8_e4m3(fminf(fmaxf(x[i][j] / scale, -kFp8Max), kFp8Max));
      // adjacent lanes hold adjacent vectors: the even lane issues both as one 16-byte multicast store
      const uint32_t px = __shfl_xor_sync(0xffffffff, q.x, 1), py = __shfl_xor_sync(0xffffffff, q.y, 1);
      if ((t & 1) == 0) store_mc(p.out_mc, (row * kHidden + c * kVec) / 16, make_uint4(q.x, q.y, px, py));
      if (c % kGroupVecs == 0) {
        device::ptx::multimem_store_relaxed(
            reinterpret_cast<uint32_t*>(p.scales_mc + row * kNumGroups + c / kGroupVecs), __float_as_uint(scale));
      }
    }
  }
};

template <uint32_t kHidden, uint32_t kThreads, bool kUsePDL, bool kFinal>
__global__ void __launch_bounds__(kThreads) gemma4_ar_fusion_kernel(const __grid_constant__ Params p) {
  using E = RowEpilogue<kHidden, kThreads>;
  __shared__ float smem[E::kNumWarps];
  // Reserve the window before the PDL wait, signal after it (see McBarrier).
  const auto barrier = McBarrier(p.sem_local, p.sem_mc, p.world_size, /*num_arrives=*/2);
  device::PDLWaitPrimary<kUsePDL>();
  barrier.arrive_relaxed(/*n=*/0);  // every rank's partials are written
  __syncthreads();
  uint4 raw[E::kPerThread];
  for (uint32_t lr = blockIdx.x; lr < p.rows; lr += gridDim.x) {
    E::load_row(p, lr, raw);
    E::template epilogue<kFinal>(p, lr, raw, smem);
  }
  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq(/*n=*/1);  // every rank's multicast stores have landed
}

template <uint32_t kHidden, uint32_t kThreads, bool kUsePDL>
struct Gemma4ArFusionKernel {
  static constexpr uint32_t kNumGroups = RowEpilogue<kHidden, kThreads>::kNumGroups;

  // ---- host -----------------------------------------------------------------

  using TensorView = tvm::ffi::TensorView;

  static Params make_params(
      const host::distributed::CommunicatorObj& comm,
      TensorView partial,
      int64_t partial_mc,
      TensorView residual,
      TensorView w_a,
      TensorView scalar,
      double eps,
      int64_t row0,
      int64_t rows) {
    using namespace host;
    const auto& pull = comm.get_pull_obj();
    SymbolicSize T = {"num_tokens"}, R = {"rows"};
    SymbolicDevice device;
    device.set_options<kDLGPU>();
    TensorMatcher({T, kHidden}).with_dtype<bf16_t>().with_device<kDLGPU>(device).verify(partial);
    TensorMatcher({R, kHidden}).with_dtype<bf16_t>().with_device<kDLGPU>(device).verify(residual);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device<kDLGPU>(device).verify(w_a);
    TensorMatcher({1}).with_dtype<fp32_t, bf16_t>().with_device<kDLGPU>(device).verify(scalar);
    CHECK_HOST(partial_mc != 0) << "the partial buffer needs a multicast address";
    CHECK_HOST(pull.mc_semaphore != nullptr) << "needs a multicast-capable pull plane";
    CHECK_HOST(0 <= row0 && 0 <= rows && row0 + rows <= T.unwrap()) << "row range escapes the batch";
    CHECK_HOST(rows == R.unwrap()) << "residual holds this rank's rows";
    Params p{};
    p.partial_mc = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(partial_mc));
    p.residual = static_cast<__nv_bfloat16*>(residual.data_ptr());
    p.w_a = static_cast<const __nv_bfloat16*>(w_a.data_ptr());
    p.scalar = scalar.data_ptr();
    p.scalar_bf16 = scalar.dtype().code == kDLBfloat;
    p.sem_local = pull.semaphores[pull.rank];
    p.sem_mc = pull.mc_semaphore;
    p.rank = pull.rank;
    p.world_size = pull.world_size;
    p.row0 = static_cast<uint32_t>(row0);
    p.rows = static_cast<uint32_t>(rows);
    p.eps = static_cast<float>(eps);
    return p;
  }

  template <bool kFinal>
  static void
  launch(const Params& p, int64_t num_blocks, const host::distributed::CommunicatorObj& comm, DLDevice device) {
    CHECK_HOST(num_blocks >= 1) << "invalid num_blocks: " << num_blocks;
    num_blocks = std::min<int64_t>(num_blocks, comm.get_pull_blocks());
    host::LaunchKernel(static_cast<uint32_t>(num_blocks), kThreads, device)
        .enable_pdl(kUsePDL)(gemma4_ar_fusion_kernel<kHidden, kThreads, kUsePDL, kFinal>, p);
  }

  /// One fused site. `partial` is this rank's [T, H] bf16 view of a symmetric
  /// buffer whose multicast VA is `partial_mc`; `out` / `scales` likewise for the
  /// [T, H] fp8 rows and [T, H / 128] fp32 scales that every rank receives.
  /// `residual` is this rank's [rows, H] slice of the stream, updated in place.
  /// num_blocks must be the same on every rank.
  static void site(
      host::distributed::CommunicatorRef ref,
      TensorView partial,
      int64_t partial_mc,
      TensorView out,
      int64_t out_mc,
      TensorView scales,
      int64_t scales_mc,
      TensorView residual,
      TensorView w_a,
      TensorView w_b,
      TensorView scalar,
      double eps,
      int64_t row0,
      int64_t rows,
      bool round_a,
      int64_t num_blocks) {
    using namespace host;
    const auto& comm = *ref.get();
    auto p = make_params(comm, partial, partial_mc, residual, w_a, scalar, eps, row0, rows);
    SymbolicSize T = {"num_tokens"};
    SymbolicDevice device;
    device.set_options<kDLGPU>();
    TensorMatcher({T, kHidden}).with_dtype<fp8_e4m3_t>().with_device<kDLGPU>(device).verify(out);
    TensorMatcher({T, kNumGroups}).with_dtype<fp32_t>().with_device<kDLGPU>(device).verify(scales);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device<kDLGPU>(device).verify(w_b);
    CHECK_HOST(T.unwrap() == partial.size(0)) << "out / scales and partial disagree on num_tokens";
    CHECK_HOST(out_mc != 0 && scales_mc != 0) << "the output buffers need multicast addresses";
    p.out_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(out_mc));
    p.scales_mc = reinterpret_cast<float*>(static_cast<uintptr_t>(scales_mc));
    p.w_b = static_cast<const __nv_bfloat16*>(w_b.data_ptr());
    p.round_a = round_a;
    launch<false>(p, num_blocks, comm, partial.device());
  }

  /// Last layer: allreduce -> norm(w_a) -> (+ residual) * scalar, every row as
  /// bf16 multicast into `out` (which may be the partial buffer itself: a row
  /// is only ever read and then written by its owner).
  static void final(
      host::distributed::CommunicatorRef ref,
      TensorView partial,
      int64_t partial_mc,
      TensorView out,
      int64_t out_mc,
      TensorView residual,
      TensorView w_a,
      TensorView scalar,
      double eps,
      int64_t row0,
      int64_t rows,
      int64_t num_blocks) {
    using namespace host;
    const auto& comm = *ref.get();
    auto p = make_params(comm, partial, partial_mc, residual, w_a, scalar, eps, row0, rows);
    SymbolicSize T = {"num_tokens"};
    SymbolicDevice device;
    device.set_options<kDLGPU>();
    TensorMatcher({T, kHidden}).with_dtype<bf16_t>().with_device<kDLGPU>(device).verify(out);
    CHECK_HOST(T.unwrap() == partial.size(0)) << "out and partial disagree on num_tokens";
    CHECK_HOST(out_mc != 0) << "the output buffer needs a multicast address";
    p.out_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(out_mc));
    p.round_a = false;
    launch<true>(p, num_blocks, comm, partial.device());
  }
};

}  // namespace gemma4_ar

template <uint32_t kHidden, uint32_t kThreads, bool kUsePDL>
using Gemma4ArFusionKernel = gemma4_ar::Gemma4ArFusionKernel<kHidden, kThreads, kUsePDL>;

}  // namespace sglang
