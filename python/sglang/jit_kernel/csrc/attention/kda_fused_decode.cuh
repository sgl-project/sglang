// KDA fused decode step for Kimi K3: causal conv1d update + delta-rule
// recurrence + gated RMSNorm in a single kernel (replaces the three-kernel
// causal_conv1d_update -> kda_packed_decode -> rms_norm_gated decode chain).
//
// Kernel body vendored from the NVIDIA x Moonshot Kimi K3 optimization
// package (KDA_decode/kda_decode_fusion_kernel.cu, many-heads variant).
// INTERNAL COLLABORATION CODE - not for open-source release without approval
// from both Moonshot and NVIDIA.
//
// Local integration changes vs. the NV source (assembled by script, each
// patch anchored on exact source text):
//   * explicit row strides for x/g/beta/onorm_g so the fused qkvg-projection
//     GEMM output slices feed the kernel without any .contiguous() copies;
//   * conv state addressed through cs_slot_stride/cs_w_stride so the kernel
//     updates the packed [slots, width, conv_dim] mamba pool in place
//     (the pool is natively transposed on this branch);
//   * ssm/temporal state addressed through state_slot_stride (state.stride(0))
//     so the kernel reads/writes envelope-strided [slots, HV, V, K] pools in
//     place: the unified / page-major layouts pitch one slot across ALL layers
//     (56,171,520 B on K3), NOT the dense HV*V*K pitch. int64 slot*stride math
//     avoids the envelope-pitch overflow (the exact chunk_delta_h bug pattern);
//   * the static-decode-layout path honors ssm_state_indices (the shipped
//     config hardwired slot = blockIdx.x, valid only for dense benches);
//   * padded cuda-graph slots (index < 0) zero the output row and skip all
//     pool updates, mirroring kda_packed_decode;
//   * tvm-ffi host wrapper (KdaFusedDecodeKernel) with TensorMatcher shape /
//     stride / dtype validation replaces the pybind binding;
//   * automatic 1D-TMA bulk state load for aligned recurrent-state layouts,
//     ported from the standalone KDA_decode/kda_decode_fusion_kernel.cu
//     experiment on chunan/kda (same mbarrier staging, same 3/4-stage
//     dispatch by grid size);
//   * PDL (kUsePDL, plumbed like kda_packed_decode): griddepcontrol wait at
//     kernel entry, launch_dependents at every exit.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t
#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kKernelWidth = 4;
constexpr int kConvStateWidth = kKernelWidth - 1;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;

__device__ __forceinline__ float bf16_load(const __nv_bfloat16* ptr, int idx) {
  return __bfloat162float(ptr[idx]);
}

__device__ __forceinline__ __nv_bfloat16 bf16_store(float value) {
  return __float2bfloat16(value);
}

template <bool kUseCacheGlobalStore>
__device__ __forceinline__ void store_state_float4(float* ptr, float4 value) {
  if constexpr (kUseCacheGlobalStore) {
    __stcg(reinterpret_cast<float4*>(ptr), value);
  } else {
    *reinterpret_cast<float4*>(ptr) = value;
  }
}

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_fast(float x) {
  return x * sigmoid_fast(x);
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ void cp_async_cg_16b(float* smem_ptr, const float* gmem_ptr) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group_0() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group_1() {
  asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group_2() {
  asm volatile("cp.async.wait_group 2;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_oldest(int outstanding_groups) {
  if (outstanding_groups >= 3) {
    cp_async_wait_group_2();
  } else if (outstanding_groups == 2) {
    cp_async_wait_group_1();
  } else {
    cp_async_wait_group_0();
  }
}

template <int kStageChunkV>
__device__ __forceinline__ void cp_async_state_chunk_stage(
    float* s_state, const float* state, int slot, int i_hv, int64_t state_slot_stride, int chunk, int stage) {
  constexpr int kFloat4PerChunk = kStageChunkV * kDimK / 4;
  const int tid = threadIdx.x;
  const int v_base = chunk * kStageChunkV;
  // Slot stride is the (possibly envelope) pitch supplied by the host, not
  // HV*V*K; the intra-slot offset i_hv*V*K + ... stays contiguous. int64
  // because slot*state_slot_stride overflows int32 on strided pools.
  const int64_t slot_base = static_cast<int64_t>(slot) * state_slot_stride;
  for (int linear4 = tid; linear4 < kFloat4PerChunk; linear4 += kThreads) {
    const int elem = linear4 * 4;
    const int row = elem / kDimK;
    const int k = elem - row * kDimK;
    float* dst = s_state + (stage * kStageChunkV + row) * kDimK + k;
    const float* src = state + slot_base + ((i_hv * kDimV + v_base + row) * kDimK + k);
    cp_async_cg_16b(dst, src);
  }
  cp_async_commit();
}

template <int kCopyThreads>
__device__ __forceinline__ void cp_async_state_chunk_for(
    float* s_state, const float* state, int slot, int i_hv, int64_t state_slot_stride, int chunk) {
  constexpr int kFloat4PerChunk = kChunkV * kDimK / 4;
  const int tid = threadIdx.x;
  const int stage = chunk & 1;
  const int v_base = chunk * kChunkV;
  const int64_t slot_base = static_cast<int64_t>(slot) * state_slot_stride;
  for (int linear4 = tid; linear4 < kFloat4PerChunk; linear4 += kCopyThreads) {
    const int elem = linear4 * 4;
    const int row = elem / kDimK;
    const int k = elem - row * kDimK;
    float* dst = s_state + (stage * kChunkV + row) * kDimK + k;
    const float* src = state + slot_base + ((i_hv * kDimV + v_base + row) * kDimK + k);
    cp_async_cg_16b(dst, src);
  }
  cp_async_commit();
}

__device__ __forceinline__ void cp_async_state_chunk(
    float* s_state, const float* state, int slot, int i_hv, int64_t state_slot_stride, int chunk) {
  cp_async_state_chunk_for<kThreads>(s_state, state, slot, i_hv, state_slot_stride, chunk);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define KDA_FUSED_DECODE_HAS_TMA 1
#else
#define KDA_FUSED_DECODE_HAS_TMA 0
#endif

__device__ __forceinline__ void mbarrier_init_one(uint64_t* bar) {
#if KDA_FUSED_DECODE_HAS_TMA
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(addr));
#endif
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar, uint32_t parity) {
#if KDA_FUSED_DECODE_HAS_TMA
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n"
        "selp.b32 %0, 1, 0, P;\n"
        "}\n"
        : "=r"(done)
        : "r"(addr), "r"(parity));
  }
#endif
}

// 1D TMA bulk copy gmem -> smem; the issuing thread arrives with expect-tx on
// the mbarrier, completion is observed via mbarrier_wait_parity.
__device__ __forceinline__ void tma_load_1d(float* smem_dst, const float* gmem_src, uint64_t* bar, uint32_t bytes) {
#if KDA_FUSED_DECODE_HAS_TMA
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  const uint32_t mbar = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("fence.proxy.async.shared::cta;" ::);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes));
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1], %2, [%3];" ::"r"(dst),
      "l"(gmem_src),
      "r"(bytes),
      "r"(mbar)
      : "memory");
#else
  __trap();
#endif
}

template <int kStageChunkV>
__device__ __forceinline__ void tma_state_chunk_stage(
    float* s_state, const float* state, int slot, int i_hv, int64_t state_slot_stride, int chunk, int stage,
    uint64_t* bar) {
  constexpr uint32_t kBytes = kStageChunkV * kDimK * sizeof(float);
  float* dst = s_state + stage * kStageChunkV * kDimK;
  // 16B-aligned for any slot iff state_slot_stride*sizeof(float) % 16 == 0
  // (host gates TMA off otherwise); the intra-slot chunk offset is a multiple
  // of kStageChunkV*kDimK*4, always 16B-aligned.
  const int64_t slot_base = static_cast<int64_t>(slot) * state_slot_stride;
  const float* src = state + slot_base + ((i_hv * kDimV + chunk * kStageChunkV) * kDimK);
  tma_load_1d(dst, src, bar, kBytes);
}

__device__ __forceinline__ float block_reduce_sum(float value, float* scratch) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  float warp_total = warp_reduce_sum(value);
  if (lane == 0) {
    scratch[warp] = warp_total;
  }
  __syncthreads();

  float block_total = 0.0f;
  if (warp == 0) {
    block_total = lane < kWarps ? scratch[lane] : 0.0f;
    block_total = warp_reduce_sum(block_total);
    if (lane == 0) {
      scratch[0] = block_total;
    }
  }
  __syncthreads();
  return scratch[0];
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

struct Sum4 {
  float a;
  float b;
  float c;
  float d;
};

__device__ __forceinline__ Sum4 warp_reduce_sum4(float a, float b, float c, float d) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    a += __shfl_xor_sync(0xffffffffu, a, offset);
    b += __shfl_xor_sync(0xffffffffu, b, offset);
    c += __shfl_xor_sync(0xffffffffu, c, offset);
    d += __shfl_xor_sync(0xffffffffu, d, offset);
  }
  return {a, b, c, d};
}

template <int kReduceWarps>
__device__ __forceinline__ Sum2 block_reduce_sum2_for(float x, float y, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const float warp_x = warp_reduce_sum(x);
  const float warp_y = warp_reduce_sum(y);
  if (lane == 0) {
    scratch[warp] = warp_x;
    scratch[kReduceWarps + warp] = warp_y;
  }
  __syncthreads();

  float block_x = 0.0f;
  float block_y = 0.0f;
  if (warp == 0) {
    block_x = lane < kReduceWarps ? scratch[lane] : 0.0f;
    block_y = lane < kReduceWarps ? scratch[kReduceWarps + lane] : 0.0f;
    block_x = warp_reduce_sum(block_x);
    block_y = warp_reduce_sum(block_y);
    if (lane == 0) {
      scratch[0] = block_x;
      scratch[1] = block_y;
    }
  }
  __syncthreads();
  return {scratch[0], scratch[1]};
}

__device__ __forceinline__ Sum2 block_reduce_sum2(float x, float y, float* scratch) {
  return block_reduce_sum2_for<kWarps>(x, y, scratch);
}

template <int kReduceWarps>
__device__ __forceinline__ float block_reduce_sum_active_for(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float warp_total = 0.0f;
  if (warp < kReduceWarps) {
    warp_total = warp_reduce_sum(value);
  }
  if (lane == 0 && warp < kReduceWarps) {
    scratch[warp] = warp_total;
  }
  __syncthreads();

  float block_total = 0.0f;
  if (warp == 0) {
    block_total = lane < kReduceWarps ? scratch[lane] : 0.0f;
    block_total = warp_reduce_sum(block_total);
    if (lane == 0) {
      scratch[0] = block_total;
    }
  }
  __syncthreads();
  return scratch[0];
}

template <int kReduceWarps>
__device__ __forceinline__ Sum2 block_reduce_sum2_active_for(float x, float y, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float warp_x = 0.0f;
  float warp_y = 0.0f;
  if (warp < kReduceWarps) {
    warp_x = warp_reduce_sum(x);
    warp_y = warp_reduce_sum(y);
  }
  if (lane == 0 && warp < kReduceWarps) {
    scratch[warp] = warp_x;
    scratch[kReduceWarps + warp] = warp_y;
  }
  __syncthreads();

  float block_x = 0.0f;
  float block_y = 0.0f;
  if (warp == 0) {
    block_x = lane < kReduceWarps ? scratch[lane] : 0.0f;
    block_y = lane < kReduceWarps ? scratch[kReduceWarps + lane] : 0.0f;
    block_x = warp_reduce_sum(block_x);
    block_y = warp_reduce_sum(block_y);
    if (lane == 0) {
      scratch[0] = block_x;
      scratch[1] = block_y;
    }
  }
  __syncthreads();
  return {scratch[0], scratch[1]};
}

__device__ __forceinline__ float fp32_at(const float* ptr, int64_t idx) {
  return ptr[idx];
}

template <
    bool kApplyOnorm,
    bool kUseStaticDecodeLayout = false,
    int kFixedHeads = 0,
    int kFixedValueHeads = 0,
    bool kUseHeadGrid = false,
    bool kAccumulateOnormSumsq = false,
    bool kUseActiveQkReduction = false,
    bool kUseCacheGlobalStore = false,
    bool kComputeOutputBeforeStore = false,
    bool kSkipWarpSync = false,
    bool kPreloadOnormParams = false,
    bool kPrefetchNextStateChunk = false,
    bool kUseActiveOnormReduction = false,
    bool kUpdateConvState = false,
    bool kUseLowerBound = false,
    bool kApplyBetaSigmoid = true,
    bool kUseTmaLoad = false,
    int kTmaStages = kNumChunks,
    bool kUsePDL = false>
__global__ __launch_bounds__(kThreads, 2) void kda_decode_fusion_many_heads_kernel(
    const __nv_bfloat16* __restrict__ x_q,
    const __nv_bfloat16* __restrict__ x_k,
    const __nv_bfloat16* __restrict__ x_v,
    const float* __restrict__ w_q_t,
    const float* __restrict__ w_k_t,
    const float* __restrict__ w_v_t,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    __nv_bfloat16* __restrict__ cs_q,
    __nv_bfloat16* __restrict__ cs_k,
    __nv_bfloat16* __restrict__ cs_v,
    const float* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ g,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ onorm_g,
    const float* __restrict__ onorm_weight,
    const int* __restrict__ ssm_state_indices,
    const int* __restrict__ cu_seqlens,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ out,
    int B,
    int H,
    int HV,
    float lower_bound,
    float scale,
    float onorm_eps,
    int64_t x_row_stride,
    int64_t g_row_stride,
    int64_t beta_row_stride,
    int64_t onormg_row_stride,
    int64_t cs_slot_stride,
    int64_t cs_w_stride,
    int64_t state_slot_stride) {
  device::PDLWaitPrimary<kUsePDL>();
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  int i_n;
  int i_hv;
  int i_h;
  int bos;
  int slot;
  if constexpr (kUseStaticDecodeLayout) {
    if constexpr (kUseHeadGrid) {
      i_n = blockIdx.x;
      i_hv = blockIdx.y;
    } else {
      const int nhv = blockIdx.x;
      i_n = nhv / kFixedValueHeads;
      i_hv = nhv - i_n * kFixedValueHeads;
    }
    i_h = i_hv;
    bos = i_n;
    slot = ssm_state_indices == nullptr ? i_n : ssm_state_indices[i_n];
  } else {
    const int nhv = blockIdx.x;
    i_n = nhv / HV;
    i_hv = nhv - i_n * HV;
    const int hv_per_h = HV / H;
    i_h = i_hv / hv_per_h;

    bos = cu_seqlens == nullptr ? i_n : cu_seqlens[i_n];
    const int eos = cu_seqlens == nullptr ? i_n + 1 : cu_seqlens[i_n + 1];
    if (eos <= bos) {
      device::PDLTriggerSecondary<kUsePDL>();
      return;
    }
    slot = ssm_state_indices == nullptr ? i_n : ssm_state_indices[i_n];
  }

  if (slot < 0) {
    // Padded cuda-graph slot: zero the output row, leave the pools untouched.
    const int hv_count_pad = kUseStaticDecodeLayout ? kFixedValueHeads : HV;
    if (tid < kDimV) {
      out[(i_n * hv_count_pad + i_hv) * kDimV + tid] = __float2bfloat16(0.0f);
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  const int hk_off = i_h * kDimK;
  const int hv_off = i_hv * kDimV;
  const int h_count = kUseStaticDecodeLayout ? kFixedHeads : H;
  const int hv_count = kUseStaticDecodeLayout ? kFixedValueHeads : HV;
  const int hkv_dim = h_count * kDimK;
  const int hvv_dim = hv_count * kDimV;

  // Dynamic size: 2 cp.async stages (32KB) or kTmaStages TMA stages (16KB per
  // stage; kTmaStages == kNumChunks means no buffer reuse).
  extern __shared__ __align__(16) float s_state[];
  __shared__ float s_q[kDimK];
  __shared__ float s_k[kDimK];
  __shared__ float s_decay[kDimK];
  __shared__ float s_v[kDimV];
  __shared__ float s_o[kDimV];
  __shared__ float s_reduce[kThreads];
  __shared__ float s_beta;
  __shared__ uint64_t s_tma_bar[kNumChunks];
  float pre_onorm_gate = 0.0f;
  float pre_onorm_weight = 0.0f;

  if constexpr (kUseTmaLoad) {
    // Chunk c lives in stage c % kTmaStages behind barrier c % kTmaStages
    // (wait parity (c / kTmaStages) & 1). Chunks 0/1 are issued here; chunk+2
    // is issued at the loop top while its stage is still fresh, and chunks
    // that reuse a stage are issued at the loop bottom behind a
    // __syncthreads(). Barrier-init visibility for waiting threads is
    // covered by the __syncthreads() before the chunk loop.
    if (tid == 0) {
#pragma unroll
      for (int c = 0; c < kTmaStages; ++c) {
        mbarrier_init_one(&s_tma_bar[c]);
      }
      tma_state_chunk_stage<kChunkV>(s_state, state, slot, i_hv, state_slot_stride, 0, 0, &s_tma_bar[0]);
      if (kTmaStages > 1 && kNumChunks > 1) {
        tma_state_chunk_stage<kChunkV>(s_state, state, slot, i_hv, state_slot_stride, 1, 1, &s_tma_bar[1]);
      }
    }
  } else {
    cp_async_state_chunk(s_state, state, slot, i_hv, state_slot_stride, 0);
  }

  if constexpr (kUpdateConvState) {
    if (tid < kDimK) {
      const int k = tid;
      const int hk = hk_off + k;
      const int64_t cs_base = slot * cs_slot_stride + hk;
      const int64_t xq_idx = bos * x_row_stride + hk;
      const float exp_a = __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

      float q_acc = bias_q[hk];
      float k_acc = bias_k[hk];
      __nv_bfloat16 q_shift0 = __float2bfloat16(0.0f);
      __nv_bfloat16 q_shift1 = __float2bfloat16(0.0f);
      __nv_bfloat16 k_shift0 = __float2bfloat16(0.0f);
      __nv_bfloat16 k_shift1 = __float2bfloat16(0.0f);
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const __nv_bfloat16 q_state = cs_q[cs_base + w * cs_w_stride];
        const __nv_bfloat16 k_state = cs_k[cs_base + w * cs_w_stride];
        q_acc += __bfloat162float(q_state) * fp32_at(w_q_t, w * hkv_dim + hk);
        k_acc += __bfloat162float(k_state) * fp32_at(w_k_t, w * hkv_dim + hk);
        if (w == 1) {
          q_shift0 = q_state;
          k_shift0 = k_state;
        } else if (w == 2) {
          q_shift1 = q_state;
          k_shift1 = k_state;
        }
      }
      const __nv_bfloat16 q_new = x_q[xq_idx];
      const __nv_bfloat16 k_new = x_k[xq_idx];
      q_acc += __bfloat162float(q_new) * fp32_at(w_q_t, (kKernelWidth - 1) * hkv_dim + hk);
      k_acc += __bfloat162float(k_new) * fp32_at(w_k_t, (kKernelWidth - 1) * hkv_dim + hk);

      cs_q[cs_base + 0] = q_shift0;
      cs_q[cs_base + cs_w_stride] = q_shift1;
      cs_q[cs_base + 2 * cs_w_stride] = q_new;
      cs_k[cs_base + 0] = k_shift0;
      cs_k[cs_base + cs_w_stride] = k_shift1;
      cs_k[cs_base + 2 * cs_w_stride] = k_new;

      s_q[k] = silu_fast(q_acc);
      s_k[k] = silu_fast(k_acc);

      const float g_raw = bf16_load(g, bos * g_row_stride + i_hv * kDimK + k) + dt_bias[hk];
      if constexpr (kUseLowerBound) {
        s_decay[k] = __expf(lower_bound * sigmoid_fast(exp_a * g_raw));
      } else {
        s_decay[k] = __expf(-exp_a * softplus_fast(g_raw));
      }
    }
  } else {
    if (tid < kDimK) {
      const int k = tid;
      const int hk = hk_off + k;
      const float exp_a = __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

      float q_acc = bias_q[hk];
      float k_acc = bias_k[hk];
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const int64_t cs_idx = slot * cs_slot_stride + hk + w * cs_w_stride;
        q_acc += bf16_load(cs_q, cs_idx) * fp32_at(w_q_t, w * hkv_dim + hk);
        k_acc += bf16_load(cs_k, cs_idx) * fp32_at(w_k_t, w * hkv_dim + hk);
      }
      q_acc += bf16_load(x_q, bos * x_row_stride + hk) * fp32_at(w_q_t, (kKernelWidth - 1) * hkv_dim + hk);
      k_acc += bf16_load(x_k, bos * x_row_stride + hk) * fp32_at(w_k_t, (kKernelWidth - 1) * hkv_dim + hk);

      s_q[k] = silu_fast(q_acc);
      s_k[k] = silu_fast(k_acc);

      const float g_raw = bf16_load(g, bos * g_row_stride + i_hv * kDimK + k) + dt_bias[hk];
      if constexpr (kUseLowerBound) {
        s_decay[k] = __expf(lower_bound * sigmoid_fast(exp_a * g_raw));
      } else {
        s_decay[k] = __expf(-exp_a * softplus_fast(g_raw));
      }
    }
  }

  if constexpr (kUpdateConvState) {
    if (tid < kDimV) {
      const int v = tid;
      const int hvv = hv_off + v;
      const int64_t cs_base = slot * cs_slot_stride + hvv;
      const int64_t xv_idx = bos * x_row_stride + hvv;

      float v_acc = bias_v[hvv];
      __nv_bfloat16 v_shift0 = __float2bfloat16(0.0f);
      __nv_bfloat16 v_shift1 = __float2bfloat16(0.0f);
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const __nv_bfloat16 v_state = cs_v[cs_base + w * cs_w_stride];
        v_acc += __bfloat162float(v_state) * fp32_at(w_v_t, w * hvv_dim + hvv);
        if (w == 1) {
          v_shift0 = v_state;
        } else if (w == 2) {
          v_shift1 = v_state;
        }
      }
      const __nv_bfloat16 v_new = x_v[xv_idx];
      v_acc += __bfloat162float(v_new) * fp32_at(w_v_t, (kKernelWidth - 1) * hvv_dim + hvv);
      cs_v[cs_base + 0] = v_shift0;
      cs_v[cs_base + cs_w_stride] = v_shift1;
      cs_v[cs_base + 2 * cs_w_stride] = v_new;
      s_v[v] = silu_fast(v_acc);

      if constexpr (kApplyOnorm && kPreloadOnormParams) {
        const int64_t onorm_idx = i_n * onormg_row_stride + i_hv * kDimV + v;
        pre_onorm_gate = sigmoid_fast(bf16_load(onorm_g, onorm_idx));
        pre_onorm_weight = onorm_weight[v];
      }
    }
  } else {
    if (tid < kDimV) {
      const int v = tid;
      const int hvv = hv_off + v;

      float v_acc = bias_v[hvv];
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const int64_t cs_idx = slot * cs_slot_stride + hvv + w * cs_w_stride;
        v_acc += bf16_load(cs_v, cs_idx) * fp32_at(w_v_t, w * hvv_dim + hvv);
      }
      v_acc += bf16_load(x_v, bos * x_row_stride + hvv) * fp32_at(w_v_t, (kKernelWidth - 1) * hvv_dim + hvv);
      s_v[v] = silu_fast(v_acc);

      if constexpr (kApplyOnorm && kPreloadOnormParams) {
        const int64_t onorm_idx = i_n * onormg_row_stride + i_hv * kDimV + v;
        pre_onorm_gate = sigmoid_fast(bf16_load(onorm_g, onorm_idx));
        pre_onorm_weight = onorm_weight[v];
      }
    }
  }

  if (tid == 0) {
    const float beta_raw = bf16_load(beta, bos * beta_row_stride + i_hv);
    if constexpr (kApplyBetaSigmoid) {
      s_beta = sigmoid_fast(beta_raw);
    } else {
      s_beta = beta_raw;
    }
  }
  __syncthreads();

  if constexpr (!kUseTmaLoad && kPrefetchNextStateChunk && kNumChunks > 1) {
    cp_async_state_chunk(s_state, state, slot, i_hv, state_slot_stride, 1);
  }

  const float q_sq = tid < kDimK ? s_q[tid] * s_q[tid] : 0.0f;
  const float k_sq = tid < kDimK ? s_k[tid] * s_k[tid] : 0.0f;
  Sum2 qk_sum;
  if constexpr (kUseActiveQkReduction) {
    qk_sum = block_reduce_sum2_active_for<kDimK / 32>(q_sq, k_sq, s_reduce);
  } else {
    qk_sum = block_reduce_sum2(q_sq, k_sq, s_reduce);
  }
  if (tid < kDimK) {
    s_q[tid] *= rsqrtf(qk_sum.x + 1.0e-6f) * scale;
    s_k[tid] *= rsqrtf(qk_sum.y + 1.0e-6f);
  }
  __syncthreads();

  const int k_base = lane * 4;
  const float4 q4 = *reinterpret_cast<const float4*>(s_q + k_base);
  const float4 k4 = *reinterpret_cast<const float4*>(s_k + k_base);
  const float4 decay4 = *reinterpret_cast<const float4*>(s_decay + k_base);
  float r_q[4] = {q4.x, q4.y, q4.z, q4.w};
  float r_k[4] = {k4.x, k4.y, k4.z, k4.w};
  float r_decay[4] = {decay4.x, decay4.y, decay4.z, decay4.w};
  float o_sumsq = 0.0f;

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    if constexpr (kUseTmaLoad) {
      if (tid == 0 && chunk + 2 < kNumChunks && chunk + 2 < kTmaStages) {
        tma_state_chunk_stage<kChunkV>(
            s_state, state, slot, i_hv, state_slot_stride, chunk + 2, chunk + 2, &s_tma_bar[chunk + 2]);
      }
      mbarrier_wait_parity(&s_tma_bar[chunk % kTmaStages], (chunk / kTmaStages) & 1);
    } else if constexpr (kPrefetchNextStateChunk && kNumChunks > 1) {
      if (chunk + 1 < kNumChunks) {
        cp_async_wait_group_1();
      } else {
        cp_async_wait_all();
      }
    } else {
      cp_async_wait_all();
    }
    if constexpr (!kUseTmaLoad && !kSkipWarpSync) {
      __syncwarp();
    }

    if constexpr (!kUseTmaLoad && !kPrefetchNextStateChunk) {
      if (chunk + 1 < kNumChunks) {
        cp_async_state_chunk(s_state, state, slot, i_hv, state_slot_stride, chunk + 1);
      }
    }

    const float* state_stage = s_state + (kUseTmaLoad ? (chunk % kTmaStages) : (chunk & 1)) * kChunkV * kDimK;

#pragma unroll
    for (int row = 0; row < kRowsPerWarp; row += 2) {
      const int v_row_a = warp + row * kWarps;
      const int v_row_b = warp + (row + 1) * kWarps;
      const int v0 = chunk * kChunkV + v_row_a;
      const int v1 = chunk * kChunkV + v_row_b;
      float h_a_vals[4];
      float h_b_vals[4];
      float dot_hk_a = 0.0f;
      float dot_hk_b = 0.0f;

      const float4 raw_h_a = *reinterpret_cast<const float4*>(state_stage + v_row_a * kDimK + k_base);
      const float4 raw_h_b = *reinterpret_cast<const float4*>(state_stage + v_row_b * kDimK + k_base);
      h_a_vals[0] = raw_h_a.x * r_decay[0];
      h_a_vals[1] = raw_h_a.y * r_decay[1];
      h_a_vals[2] = raw_h_a.z * r_decay[2];
      h_a_vals[3] = raw_h_a.w * r_decay[3];
      h_b_vals[0] = raw_h_b.x * r_decay[0];
      h_b_vals[1] = raw_h_b.y * r_decay[1];
      h_b_vals[2] = raw_h_b.z * r_decay[2];
      h_b_vals[3] = raw_h_b.w * r_decay[3];
      dot_hk_a = h_a_vals[0] * r_k[0] + h_a_vals[1] * r_k[1] + h_a_vals[2] * r_k[2] + h_a_vals[3] * r_k[3];
      dot_hk_b = h_b_vals[0] * r_k[0] + h_b_vals[1] * r_k[1] + h_b_vals[2] * r_k[2] + h_b_vals[3] * r_k[3];

      const Sum2 dot_hk = warp_reduce_sum_pair(dot_hk_a, dot_hk_b);
      const float v_new0 = (s_v[v0] - dot_hk.x) * s_beta;
      const float v_new1 = (s_v[v1] - dot_hk.y) * s_beta;

      float dot_hq_a = 0.0f;
      float dot_hq_b = 0.0f;
      // Writeback mirrors the load addressing: slot pitch from the host stride
      // (int64, envelope-safe), intra-slot offset i_hv*V*K + v*K + k contiguous.
      const int64_t slot_base_wb = static_cast<int64_t>(slot) * state_slot_stride;
      const int64_t state_idx_a = slot_base_wb + ((i_hv * kDimV + v0) * kDimK + k_base);
      const int64_t state_idx_b = slot_base_wb + ((i_hv * kDimV + v1) * kDimK + k_base);
      const float h_a_0 = h_a_vals[0] + r_k[0] * v_new0;
      const float h_a_1 = h_a_vals[1] + r_k[1] * v_new0;
      const float h_a_2 = h_a_vals[2] + r_k[2] * v_new0;
      const float h_a_3 = h_a_vals[3] + r_k[3] * v_new0;
      const float h_b_0 = h_b_vals[0] + r_k[0] * v_new1;
      const float h_b_1 = h_b_vals[1] + r_k[1] * v_new1;
      const float h_b_2 = h_b_vals[2] + r_k[2] * v_new1;
      const float h_b_3 = h_b_vals[3] + r_k[3] * v_new1;
      if constexpr (kComputeOutputBeforeStore) {
        dot_hq_a = h_a_0 * r_q[0] + h_a_1 * r_q[1] + h_a_2 * r_q[2] + h_a_3 * r_q[3];
        dot_hq_b = h_b_0 * r_q[0] + h_b_1 * r_q[1] + h_b_2 * r_q[2] + h_b_3 * r_q[3];
        store_state_float4<kUseCacheGlobalStore>(state + state_idx_a, make_float4(h_a_0, h_a_1, h_a_2, h_a_3));
        store_state_float4<kUseCacheGlobalStore>(state + state_idx_b, make_float4(h_b_0, h_b_1, h_b_2, h_b_3));
      } else {
        store_state_float4<kUseCacheGlobalStore>(state + state_idx_a, make_float4(h_a_0, h_a_1, h_a_2, h_a_3));
        store_state_float4<kUseCacheGlobalStore>(state + state_idx_b, make_float4(h_b_0, h_b_1, h_b_2, h_b_3));
        dot_hq_a = h_a_0 * r_q[0] + h_a_1 * r_q[1] + h_a_2 * r_q[2] + h_a_3 * r_q[3];
        dot_hq_b = h_b_0 * r_q[0] + h_b_1 * r_q[1] + h_b_2 * r_q[2] + h_b_3 * r_q[3];
      }

      const Sum2 dot_hq = warp_reduce_sum_pair(dot_hq_a, dot_hq_b);
      if (lane == 0) {
        s_o[v0] = dot_hq.x;
        s_o[v1] = dot_hq.y;
        if constexpr (kApplyOnorm && kAccumulateOnormSumsq) {
          o_sumsq += dot_hq.x * dot_hq.x + dot_hq.y * dot_hq.y;
        }
      }
    }

    if constexpr (kUseTmaLoad && kTmaStages < kNumChunks) {
      // chunk + kTmaStages reuses the stage this chunk just read; every warp
      // must be done with it before the single issuing thread overwrites it.
      if (chunk + kTmaStages < kNumChunks) {
        __syncthreads();
        if (tid == 0) {
          tma_state_chunk_stage<kChunkV>(
              s_state,
              state,
              slot,
              i_hv,
              state_slot_stride,
              chunk + kTmaStages,
              chunk % kTmaStages,
              &s_tma_bar[chunk % kTmaStages]);
        }
      }
    } else if constexpr (!kUseTmaLoad && kPrefetchNextStateChunk) {
      if (chunk + 2 < kNumChunks) {
        cp_async_state_chunk(s_state, state, slot, i_hv, state_slot_stride, chunk + 2);
      }
    }
  }
  __syncthreads();

  device::PDLTriggerSecondary<kUsePDL>();

  if constexpr (kApplyOnorm) {
    if constexpr (kAccumulateOnormSumsq) {
      if (lane == 0) {
        s_reduce[warp] = o_sumsq;
      }
      __syncthreads();

      float total_sumsq = 0.0f;
      if (warp == 0) {
        total_sumsq = lane < kWarps ? s_reduce[lane] : 0.0f;
        total_sumsq = warp_reduce_sum(total_sumsq);
        if (lane == 0) {
          s_reduce[0] = total_sumsq;
        }
      }
      __syncthreads();

      if (tid < kDimV) {
        const int out_idx = (i_n * hv_count + i_hv) * kDimV + tid;
        const float raw_o = s_o[tid];
        const float rstd = rsqrtf(s_reduce[0] / static_cast<float>(kDimV) + onorm_eps);
        float gate;
        float weight;
        if constexpr (kPreloadOnormParams) {
          gate = pre_onorm_gate;
          weight = pre_onorm_weight;
        } else {
          gate = sigmoid_fast(bf16_load(onorm_g, i_n * onormg_row_stride + i_hv * kDimV + tid));
          weight = onorm_weight[tid];
        }
        const float y = raw_o * rstd * weight * gate;
        out[out_idx] = bf16_store(y);
      }
    } else {
      const float raw_o = tid < kDimV ? s_o[tid] : 0.0f;
      const float o_sq = raw_o * raw_o;
      float sumsq;
      if constexpr (kUseActiveOnormReduction || kUseActiveQkReduction) {
        sumsq = block_reduce_sum_active_for<kDimV / 32>(o_sq, s_reduce);
      } else {
        sumsq = block_reduce_sum(o_sq, s_reduce);
      }

      if (tid < kDimV) {
        const int out_idx = (i_n * hv_count + i_hv) * kDimV + tid;
        const float rstd = rsqrtf(sumsq / static_cast<float>(kDimV) + onorm_eps);
        float gate;
        float weight;
        if constexpr (kPreloadOnormParams) {
          gate = pre_onorm_gate;
          weight = pre_onorm_weight;
        } else {
          gate = sigmoid_fast(bf16_load(onorm_g, i_n * onormg_row_stride + i_hv * kDimV + tid));
          weight = onorm_weight[tid];
        }
        const float y = raw_o * rstd * weight * gate;
        out[out_idx] = bf16_store(y);
      }
    }
  } else {
    if (tid < kDimV) {
      const int out_idx = (i_n * hv_count + i_hv) * kDimV + tid;
      out[out_idx] = bf16_store(s_o[tid]);
    }
  }
}

// K3 decode configuration of the many-heads kernel: onorm fused, static
// H = HV = 12 layout with a (B, HV) grid, onorm params preloaded, next state
// chunk prefetched, active onorm reduction, conv cache updated in place,
// beta sigmoid in-kernel. Both forget-gate variants are compiled (softplus
// and lower-bounded sigmoid) and selected at launch from the model config.
// kUseTmaLoad/kTmaStages select the 1D-TMA state-staging path in place of the
// cp.async fallback used for a misaligned recurrent-state slot stride.
template <bool kUseLowerBound, bool kUsePDL, bool kUseTmaLoad = false, int kTmaStages = kNumChunks>
constexpr auto kda_fused_decode_k3_kernel = kda_decode_fusion_many_heads_kernel<
    /*kApplyOnorm=*/true,
    /*kUseStaticDecodeLayout=*/true,
    /*kFixedHeads=*/12,
    /*kFixedValueHeads=*/12,
    /*kUseHeadGrid=*/true,
    /*kAccumulateOnormSumsq=*/false,
    /*kUseActiveQkReduction=*/false,
    /*kUseCacheGlobalStore=*/false,
    /*kComputeOutputBeforeStore=*/false,
    /*kSkipWarpSync=*/false,
    /*kPreloadOnormParams=*/true,
    /*kPrefetchNextStateChunk=*/true,
    /*kUseActiveOnormReduction=*/true,
    /*kUpdateConvState=*/true,
    kUseLowerBound,
    /*kApplyBetaSigmoid=*/true,
    kUseTmaLoad,
    kTmaStages,
    kUsePDL>;

template <bool kUsePDL>
struct KdaFusedDecodeKernel {
  static void
  run(const tvm::ffi::TensorView mixed_qkv,     // [B, 3*H*128] bf16, row-strided
      const tvm::ffi::TensorView a,             // [B, H*128] bf16 raw forget gate
      const tvm::ffi::TensorView b,             // [B, H] bf16 raw beta logits
      const tvm::ffi::TensorView conv_states,   // [slots, 3, conv_dim] bf16 pool
      const tvm::ffi::TensorView w_q_t,         // [4, H*128] fp32 dense
      const tvm::ffi::TensorView w_k_t,         // [4, H*128] fp32 dense
      const tvm::ffi::TensorView w_v_t,         // [4, H*128] fp32 dense
      const tvm::ffi::TensorView conv_bias,     // [3*H*128] fp32 (zeros if none)
      const tvm::ffi::TensorView A_log,         // [H] fp32
      const tvm::ffi::TensorView dt_bias,       // [H*128] fp32
      const tvm::ffi::TensorView onorm_g,       // [B, H*128] bf16, row-strided
      const tvm::ffi::TensorView onorm_weight,  // [128] fp32
      const tvm::ffi::TensorView state,         // [slots, H, 128, 128] fp32; inner-contiguous, any slot pitch
      const tvm::ffi::TensorView indices,       // [B] int32 (< 0 = padded slot)
      const tvm::ffi::TensorView out,           // [B, H*128] bf16 dense
      double scale,
      double onorm_eps,
      double lower_bound,
      bool use_lower_bound) {
    using namespace host;

    constexpr int64_t kH = 12;
    constexpr int64_t kSeg = kH * 128;  // 1536: q, k and v segment width

    auto B_ = SymbolicSize{"batch"};
    auto Slots_ = SymbolicSize{"pool_slots"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B_, 3 * kSeg}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(mixed_qkv);
    TensorMatcher({B_, kSeg}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(a);
    TensorMatcher({B_, kH}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(b);
    TensorMatcher({Slots_, 3, 3 * kSeg})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, -1, 1})
        .verify(conv_states);
    TensorMatcher({4, kSeg}).with_dtype<fp32_t>().with_device(device).with_strides({kSeg, 1}).verify(w_q_t);
    TensorMatcher({4, kSeg}).with_dtype<fp32_t>().with_device(device).with_strides({kSeg, 1}).verify(w_k_t);
    TensorMatcher({4, kSeg}).with_dtype<fp32_t>().with_device(device).with_strides({kSeg, 1}).verify(w_v_t);
    TensorMatcher({3 * kSeg}).with_dtype<fp32_t>().with_device(device).with_strides({1}).verify(conv_bias);
    TensorMatcher({kH}).with_dtype<fp32_t>().with_device(device).verify(A_log);
    TensorMatcher({kSeg}).with_dtype<fp32_t>().with_device(device).with_strides({1}).verify(dt_bias);
    TensorMatcher({B_, kSeg}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(onorm_g);
    TensorMatcher({128}).with_dtype<fp32_t>().with_device(device).with_strides({1}).verify(onorm_weight);
    // Slot stride (dim 0) is a wildcard: a locally-allocated pool pitches a
    // slot at the dense kH*128*128, but the unified / page-major pools pitch it
    // at the multi-layer envelope (56M elems on K3). The kernel reads the real
    // slot pitch from state.stride(0); only the inner [HV, V, K] contiguity
    // (strides {V*K, K, 1}) is load-bearing here.
    TensorMatcher({Slots_, kH, 128, 128})
        .with_dtype<fp32_t>()
        .with_device(device)
        .with_strides({-1, 128 * 128, 128, 1})
        .verify(state);
    TensorMatcher({B_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(indices);
    TensorMatcher({B_, kSeg}).with_dtype<bf16_t>().with_device(device).with_strides({kSeg, 1}).verify(out);

    const auto B = static_cast<int>(B_.unwrap());
    if (B == 0) return;

    const auto* mixed_ptr = static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr());
    auto* cs_ptr = static_cast<__nv_bfloat16*>(conv_states.data_ptr());
    const auto* bias_ptr = static_cast<const float*>(conv_bias.data_ptr());
    // Real per-slot pitch of the ssm/temporal pool (elements): dense kH*128*128
    // for a local pool, the multi-layer envelope for the unified / page-major
    // pools. Threaded into every ssm-state read/write; int64 avoids the
    // envelope-pitch overflow.
    const int64_t state_slot_stride = state.stride(0);
    auto kernel =
        use_lower_bound ? kda_fused_decode_k3_kernel<true, kUsePDL> : kda_fused_decode_k3_kernel<false, kUsePDL>;
    int tma_stages = 0;
    // TMA 1D-bulk needs the per-slot source address (state + slot*stride) 16B
    // aligned for every slot. state.data_ptr() is torch-aligned and each chunk
    // offset is a multiple of kChunkV*kDimK*4 B, so alignment holds iff the slot
    // pitch itself is 16B-aligned, i.e. state_slot_stride % 4 == 0 (fp32). The
    // K3 envelope pitch (14,042,880 elems, %4==0) and the dense pitch both
    // satisfy this; a pathological stride falls back to cp.async (still fully
    // fused, just no TMA) rather than silently mis-addressing the descriptor.
    const bool tma_slot_stride_aligned = (state_slot_stride % 4) == 0;
    if (tma_slot_stride_aligned) {
      // Full-state staging (4 stages, 64KB, sync-free) wins while the grid
      // is small enough that occupancy isn't the limiter; past that the
      // 48KB 3-stage variant (one sync for the single stage reuse) benches
      // fastest. Mirrors the KDA_decode standalone-kernel dispatch on
      // chunan/kda.
      tma_stages = B * static_cast<int>(kH) >= 1024 ? 3 : 4;
      if (tma_stages == 3) {
        kernel = use_lower_bound ? kda_fused_decode_k3_kernel<true, kUsePDL, true, 3>
                                 : kda_fused_decode_k3_kernel<false, kUsePDL, true, 3>;
      } else {
        tma_stages = 4;
        kernel = use_lower_bound ? kda_fused_decode_k3_kernel<true, kUsePDL, true, 4>
                                 : kda_fused_decode_k3_kernel<false, kUsePDL, true, 4>;
      }
    }
    const int smem_stages = tma_stages == 0 ? 2 : tma_stages;
    const size_t smem_bytes = static_cast<size_t>(smem_stages) * kChunkV * kDimK * sizeof(float);
    host::RuntimeDeviceCheck(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));

    LaunchKernel(dim3(B, kH), dim3(kThreads), device.unwrap(), smem_bytes).enable_pdl(kUsePDL)(
        kernel,
        /*x_q=*/mixed_ptr,
        /*x_k=*/mixed_ptr + kSeg,
        /*x_v=*/mixed_ptr + 2 * kSeg,
        static_cast<const float*>(w_q_t.data_ptr()),
        static_cast<const float*>(w_k_t.data_ptr()),
        static_cast<const float*>(w_v_t.data_ptr()),
        /*bias_q=*/bias_ptr,
        /*bias_k=*/bias_ptr + kSeg,
        /*bias_v=*/bias_ptr + 2 * kSeg,
        /*cs_q=*/cs_ptr,
        /*cs_k=*/cs_ptr + kSeg,
        /*cs_v=*/cs_ptr + 2 * kSeg,
        static_cast<const fp32_t*>(A_log.data_ptr()),
        /*g=*/static_cast<const __nv_bfloat16*>(a.data_ptr()),
        static_cast<const fp32_t*>(dt_bias.data_ptr()),
        /*beta=*/static_cast<const __nv_bfloat16*>(b.data_ptr()),
        static_cast<const __nv_bfloat16*>(onorm_g.data_ptr()),
        static_cast<const fp32_t*>(onorm_weight.data_ptr()),
        static_cast<const int32_t*>(indices.data_ptr()),
        /*cu_seqlens=*/static_cast<const int32_t*>(nullptr),
        static_cast<fp32_t*>(state.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()),
        B,
        /*H=*/static_cast<int>(kH),
        /*HV=*/static_cast<int>(kH),
        static_cast<float>(lower_bound),
        static_cast<float>(scale),
        static_cast<float>(onorm_eps),
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        onorm_g.stride(0),
        conv_states.stride(0),
        conv_states.stride(1),
        state_slot_stride);
  }
};

}  // namespace
