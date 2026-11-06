// bf16 all-gather of the [M, K / world] pre-expert projection slices, fused with the pre-expert RMS norm and the
// static-scale NVFP4 quantization of the gathered rows. The exchange is all_gather.cuh's (multicast push into the
// CustomAllReduceV2 push workspace, Lamport poll, same phase counters); each gathered row then goes through the
// row computation of norm_fp4_quant.cuh under its thread map, so fp4 / sf are bitwise those of all_gather followed by
// fused_norm_fp4_quant_static. The gathered bf16 rows are written only when kWriteRows.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>

#include "norm_fp4_quant.cuh"
#include <algorithm>
#include <cstdint>

namespace sglang {

namespace muse_ag_quant {

using device::distributed::Counter;

constexpr uint32_t kVec = 8;  // bf16 per 16-byte vector

struct Params {
  uint8_t* ws_local;      // local VA of the push workspace (poll + reset)
  uint8_t* ws_mc;         // multicast VA of the push workspace (push)
  Counter* counter;       // per-block phase counters
  uint32_t num_counters;  // PushPlane::num_blocks
  uint32_t half_bytes;    // world_size * slot_bytes
  uint32_t rank;
  const bf16_t* local;        // [M, K / world]
  const bf16_t* weight;       // [K] effective norm weight, or null
  const float* global_scale;  // [1] static encode scale
  float eps;
  bf16_t* out;    // [M, K] gathered rows
  uint64_t* fp4;  // [M, K / 16] (8 bytes per 16 elements)
  uint8_t* sf;    // [M, K / 16]
  uint32_t num_rows;
};

// Slot layout inside a phase half: dense [world][M][K / world] bf16, as in all_gather.cuh.
template <uint32_t kNLocal, int kNumThreads>
SGL_DEVICE void push_row(const Params& p, uint32_t half_off, uint32_t row) {
  // Compile-time trip count (kNLocal / kVec <= kNumThreads at H = 4096: one predicated vector per thread).
  for (uint32_t v = threadIdx.x; v < kNLocal / kVec; v += kNumThreads) {
    const uint32_t local_elem = row * kNLocal + v * kVec;
    uint4 raw = *reinterpret_cast<const uint4*>(p.local + local_elem);
    // +0.0 marks "not arrived"; -0.0 is numerically identical.
    uint32_t* w = reinterpret_cast<uint32_t*>(&raw);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      if ((w[j] & 0xffffu) == 0) w[j] |= 0x8000u;
      if ((w[j] >> 16) == 0) w[j] |= 0x80000000u;
    }
    const uint32_t byte_off = half_off + (p.rank * p.num_rows * kNLocal + local_elem) * sizeof(bf16_t);
    device::ptx::st_multimem_16B(raw, p.ws_mc + byte_off, 0);
  }
}

// A 16-element block as two 16-byte vectors (elements 0-7, 8-15).
using Block32B = uint4[2];

// True while any of the block's 16 bf16 lanes still holds the +0.0 "not arrived" marker.
SGL_DEVICE bool pending(const Block32B& b) {
  const uint32_t* w = reinterpret_cast<const uint32_t*>(&b);
  bool missing = false;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    missing |= ((w[i] & 0xffffu) == 0) | ((w[i] >> 16) == 0);
  return missing;
}

// One 32-byte relaxed gpu-scope load.
SGL_DEVICE void ld_relaxed_32B(Block32B& b, const bf16_t* src) {
  asm volatile(
      "ld.relaxed.gpu.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(b[0].x), "=r"(b[0].y), "=r"(b[0].z), "=r"(b[0].w), "=r"(b[1].x), "=r"(b[1].y), "=r"(b[1].z), "=r"(b[1].w)
      : "l"(src));
}

// One 32-byte store of zeros (STG.E.ENL2.256): re-arms a consumed slot as "not arrived".
SGL_DEVICE void st_zero_32B(bf16_t* dst) {
  asm volatile("st.global.v8.b32 [%0], {%1, %1, %1, %1, %1, %1, %1, %1};" ::"l"(dst), "r"(0u) : "memory");
}

// Spins until every lane of this thread's 16-element block has arrived, then re-arms its slot.
SGL_DEVICE void poll_block(Block32B& b, bf16_t* src) {
  do {
    ld_relaxed_32B(b, src);
  } while (pending(b));
  st_zero_32B(src);
}

// Element-wise a * b on fp32 pairs in one FMUL2 (mul.rn.f32x2): same rounding per lane as two FMULs.
SGL_DEVICE fp32x2_t mul2(fp32x2_t a, fp32x2_t b) {
  uint64_t r;
  asm("mul.rn.f32x2 %0, %1, %2;"
      : "=l"(r)
      : "l"(*reinterpret_cast<const uint64_t*>(&a)), "l"(*reinterpret_cast<const uint64_t*>(&b)));
  return *reinterpret_cast<const fp32x2_t*>(&r);
}

template <int64_t kHidden, uint32_t kWorld, int kNumThreads, bool kUsePDL, bool kHasWeight, bool kWriteRows>
__global__ void __launch_bounds__(kNumThreads) all_gather_norm_fp4_quant_kernel(const __grid_constant__ Params p) {
  using namespace device;
  using namespace device::norm_fp4_quant;
  constexpr uint32_t kNLocal = kHidden / kWorld;
  constexpr int kPacks = kHidden / (kNumThreads * kBlock);
  static_assert(kHidden % (kNumThreads * kBlock) == 0 && kNLocal % kBlock == 0, "a block must not cross a rank slice");
  using BF16Vec = AlignedVector<bf16x2_t, 4>;
  constexpr int kWarps = kNumThreads / kWarpThreads;
  static_assert(kWarps > 1 && kWarps <= kWarpThreads);
  // Double-buffered by row: a warp already in the next row cannot overwrite partials still being read.
  __shared__ float smem[2][kWarps];

  const uint32_t bx = blockIdx.x;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  // The norm weight is the same for every row (and not produced upstream): load it once, before the PDL wait.
  // One 32-byte vector = this thread's 16 weights.
  AlignedVector<bf16x2_t, kBlock / 2> w[kPacks];
  if constexpr (kHasWeight) {
#pragma unroll
    for (int pk = 0; pk < kPacks; ++pk)
      w[pk].load(p.weight, threadIdx.x + pk * kNumThreads);
  }
  const float encode_scale = p.global_scale[0];  // static, like the weight
  // Wait for the projection GEMM that writes `local`.
  PDLWaitPrimary<kUsePDL>();
  const uint32_t phase = p.counter[bx].get() & 1;
  const uint32_t half_off = phase * p.half_bytes;
  // Block 0 advances the counters no block of this grid owns. Every earlier push-plane kernel has finished once the
  // PDL wait returns, and the next one reads them only after its own wait, i.e. after this grid has finished.
  if (bx == 0) {
#pragma unroll 1
    for (uint32_t i = gridDim.x + threadIdx.x; i < p.num_counters; i += kNumThreads)
      p.counter[i].set(phase ^ 1);
  }
  const uint32_t work_blocks = gridDim.x;
  // Push every row this block owns before polling any, so no block waits on a row it has not pushed.
  for (uint32_t row = bx; row < p.num_rows; row += work_blocks)
    push_row<kNLocal, kNumThreads>(p, half_off, row);
  // PDL launch next kernel: Overlap more since we have completed loading rows from GMEM.
  PDLTriggerSecondary<kUsePDL>();

  const auto base = reinterpret_cast<bf16_t*>(p.ws_local + half_off);
  uint32_t buf = 0;
  for (uint32_t row = bx; row < p.num_rows; row += work_blocks, buf ^= 1) {
    // Gather this thread's 16-element blocks (norm_fp4_quant.cuh's thread map) and keep them in registers.
    fp32x2_t vals[kPacks][kBlock / 2];
    float sum_sq = 0.0f;
#pragma unroll
    for (int pk = 0; pk < kPacks; ++pk) {
      const uint32_t blk = threadIdx.x + pk * kNumThreads;
      // Block blk lives in the slot of the rank that owns its columns (a block never crosses a rank slice).
      const uint32_t col = blk * kBlock;
      bf16_t* src = base + ((col / kNLocal) * p.num_rows + row) * kNLocal + col % kNLocal;
      // Poll the whole 32-byte block in one load: a warp reads 1024 contiguous bytes.
      Block32B got;
      poll_block(got, src);
      // Optionally write the gathered bf16 row (unused by the experts path).
      if constexpr (kWriteRows) {
        reinterpret_cast<const BF16Vec*>(&got[0])->store(p.out + static_cast<int64_t>(row) * kHidden, blk * 2);
        reinterpret_cast<const BF16Vec*>(&got[1])->store(p.out + static_cast<int64_t>(row) * kHidden, blk * 2 + 1);
      }
      // bf16 -> fp32, and this thread's sum of squares (serial FMA chain: norm_fp4_quant's summation order).
      const auto& lo = *reinterpret_cast<const BF16Vec*>(&got[0]);
      const auto& hi = *reinterpret_cast<const BF16Vec*>(&got[1]);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        vals[pk][q] = cast<fp32x2_t>(lo[q]);
        vals[pk][q + 4] = cast<fp32x2_t>(hi[q]);
      }
#pragma unroll
      for (int q = 0; q < kBlock / 2; ++q) {
        sum_sq = fmaf(vals[pk][q].x, vals[pk][q].x, sum_sq);
        sum_sq = fmaf(vals[pk][q].y, vals[pk][q].y, sum_sq);
      }
    }
    // Row sum of squares: cta::reduce_sum's tree (warp butterfly, then the warps' partials), on this row's smem half.
    sum_sq = warp::reduce_sum(sum_sq);
    if (lane == 0) smem[buf][threadIdx.x / kWarpThreads] = sum_sq;
    __syncthreads();
    sum_sq = warp::reduce_sum(lane < kWarps ? smem[buf][lane] : 0.0f);
    // RMS norm and static-scale NVFP4, as muse_spark_norm_fp4_quant_kernel computes them.
    const float rstd = math::rsqrt(sum_sq / kHidden + p.eps);
    const float fp4_max_rcp = rcp_approx_ftz(6.0f);
#pragma unroll
    for (int pk = 0; pk < kPacks; ++pk) {
      const uint32_t blk = threadIdx.x + pk * kNumThreads;
      float f[kBlock];
      float block_max = 0.0f;
#pragma unroll
      for (int q = 0; q < kBlock / 2; ++q) {
        // x * rstd (* weight), two lanes per FMUL2.
        fp32x2_t v = mul2(vals[pk][q], fp32x2_t{rstd, rstd});
        if constexpr (kHasWeight) v = mul2(v, cast<fp32x2_t>(w[pk][q]));
        const auto r = cast<fp32x2_t>(cast<bf16x2_t>(v));  // the quantizer sees the bf16-rounded row
        f[2 * q] = r.x;
        f[2 * q + 1] = r.y;
        block_max = fmaxf(block_max, fmaxf(fabsf(r.x), fabsf(r.y)));
      }
      // Block scale factor (e4m3) and the matching e2m1 scale.
      const float scale_f = mul_f32(encode_scale, mul_f32(block_max, fp4_max_rcp));
      const uint32_t fp8 = cvt_f32_to_e4m3(scale_f) & 0xFFu;
      const float out_scale = output_scale_static(fp8, block_max, encode_scale);
      // Scale into the e2m1 range, two lanes per FMUL2 (mul_f32 is mul.rn.f32).
#pragma unroll
      for (int i = 0; i < kBlock; i += 2) {
        const fp32x2_t sc = mul2(fp32x2_t{f[i], f[i + 1]}, fp32x2_t{out_scale, out_scale});
        f[i] = sc.x;
        f[i + 1] = sc.y;
      }
      // 16 x e2m1 = 8 bytes (a warp writes 256 contiguous bytes), and the 1-byte scale factor.
      p.fp4[static_cast<int64_t>(row) * (kHidden / kBlock) + blk] =
          (static_cast<uint64_t>(cvt_e2m1x8(f + 8)) << 32) | static_cast<uint64_t>(cvt_e2m1x8(f));
      p.sf[static_cast<int64_t>(row) * (kHidden / kBlock) + blk] = static_cast<uint8_t>(fp8);
    }
  }
  // Every slot this block consumed is re-armed: publish the next phase.
  __syncthreads();
  if (threadIdx.x == 0) p.counter[bx].set(phase ^ 1);
}

}  // namespace muse_ag_quant

using host::distributed::CommunicatorRef;

template <int64_t kHidden, uint32_t kWorld, int kNumThreads, bool kUsePDL>
struct MuseSparkAllGatherNormFp4Quant {
  using TensorView = tvm::ffi::TensorView;
  static constexpr uint32_t kNLocal = kHidden / kWorld;

  static void
  run(CommunicatorRef ref,
      TensorView local,
      tvm::ffi::Optional<TensorView> weight,
      TensorView global_scale,
      double eps,
      tvm::ffi::Optional<TensorView> out,
      TensorView fp4,
      TensorView sf) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, kNLocal}).with_dtype<bf16_t>().with_device(device).verify(local);
    if (out.has_value()) TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device).verify(out.value());
    TensorMatcher({M, kHidden / 2}).with_dtype<uint8_t>().with_device(device).verify(fp4);
    TensorMatcher({M, kHidden / 16}).with_dtype<uint8_t>().with_device(device).verify(sf);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(global_scale);
    if (weight.has_value()) TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(weight.value());
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(num_tokens >= 1);
    CHECK_HOST(push.world_size == kWorld) << "the kernel is compiled for TP" << kWorld;
    CHECK_HOST(int64_t(num_tokens) * kHidden * 2 <= push.slot_bytes) << "gathered rows exceed the push slot";
    CHECK_HOST(push.mc_workspace != nullptr) << "requires a multicast-capable push plane";
    CHECK_HOST(push.num_blocks >= 1);
    const muse_ag_quant::Params params{
        .ws_local = push.workspaces[push.rank],
        .ws_mc = push.mc_workspace,
        .counter = push.counter,
        .num_counters = push.num_blocks,
        .half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size),
        .rank = push.rank,
        .local = static_cast<const bf16_t*>(local.data_ptr()),
        .weight = weight.has_value() ? static_cast<const bf16_t*>(weight.value().data_ptr()) : nullptr,
        .global_scale = static_cast<const float*>(global_scale.data_ptr()),
        .eps = static_cast<float>(eps),
        .out = out.has_value() ? static_cast<bf16_t*>(out.value().data_ptr()) : nullptr,
        .fp4 = static_cast<uint64_t*>(fp4.data_ptr()),
        .sf = static_cast<uint8_t*>(sf.data_ptr()),
        .num_rows = num_tokens,
    };
    const uint32_t num_blocks = std::min(num_tokens, push.num_blocks);
    using namespace muse_ag_quant;
    const auto kernel =
        weight.has_value()
            ? (out.has_value() ? all_gather_norm_fp4_quant_kernel<kHidden, kWorld, kNumThreads, kUsePDL, true, true>
                               : all_gather_norm_fp4_quant_kernel<kHidden, kWorld, kNumThreads, kUsePDL, true, false>)
            : (out.has_value() ? all_gather_norm_fp4_quant_kernel<kHidden, kWorld, kNumThreads, kUsePDL, false, true>
                               : all_gather_norm_fp4_quant_kernel<kHidden, kWorld, kNumThreads, kUsePDL, false, false>);
    LaunchKernel(num_blocks, kNumThreads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
