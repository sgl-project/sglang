/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// SM100 bring-up kernel for Q8KV8 sparse MLA prefill.
//
// It deliberately preserves the SM90 CUDA kernel's numerical dataflow:
//   QK: E4M3 x E4M3 -> FP32
//   softmax: FP32 online max/sum
//   PV: E4M3 probabilities x E4M3 values -> FP32
//   epilogue: FP32 normalization/descale -> BF16
//
// This first SM100 CUDA implementation is stage-synchronous. It establishes a
// native tcgen05 correctness/performance baseline before the KV gather and the
// QK/softmax/PV stages are overlapped with the three-buffer pipeline.

#pragma once

#include <cutlass/cuda_host_adapter.hpp>

#include "../sparse_mla_q8kv8_prefill_sm90/config.h"
#include "../sparse_mla_q8kv8_prefill_sm90/params.h"
#include "fp8_mma.cuh"
#include "helpers.cuh"
#include <cuda_fp8.h>
#include <math_constants.h>

namespace sglang::sm100_q8kv8 {

using namespace cute;
using fp8_t = cutlass::float_e4m3_t;
using bf16_t = cutlass::bfloat16_t;

constexpr int kBlockH = 32;
// Consume two logical 64-token sparse blocks in one tcgen05 frame.  Besides
// halving the softmax rendezvous count, N=128 maps one 32-column P fragment to
// each of WG0's four warps instead of leaving warp1/warp3 without P work.
constexpr int kBlockTopK = 128;
constexpr int kDv = 512;
// Short grids favor fewer resident warps; long grids amortize a wider producer
// and benefit from two additional TMA-issuing warps.
constexpr int kShortThreads = 384;
constexpr int kShortProducerWarps = 7;
constexpr int kLongThreads = 512;
constexpr int kLongProducerWarps = 11;
constexpr int kLongProducerMinSq = 4096;
constexpr int kMaxBlocks = 32;
// D512 (the DeepSeek-V4 production shape) fits three 128-row stages under the
// SM100 per-CTA shared-memory limit.  D576 retains two stages so its wider KV
// rows remain launchable.
template <int Dqk>
constexpr int kBuffersFor = Dqk == 512 ? 3 : 2;
constexpr float kMaxInit = -1.0e30f;
// Scaling the probabilities before their E4M3 cast retains weights near
// 1/topk. The inverse is folded into the final value scale.
constexpr float kProbFp8Scale = 256.0f;

namespace tmem_col {
constexpr int kO = 0;
constexpr int kP = 400;
}  // namespace tmem_col

template <int Rows, int Cols, bool UseSw128>
struct SmemLayoutQKSelector;

template <int Rows, int Cols>
struct SmemLayoutQKSelector<Rows, Cols, true> {
  using type = decltype(coalesce(
      tile_to_shape(UMMA::Layout_K_SW128_Atom<fp8_t>{}, Shape<Int<Rows>, Int<Cols>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));
};

template <int Rows, int Cols>
struct SmemLayoutQKSelector<Rows, Cols, false> {
  using type = decltype(coalesce(
      tile_to_shape(UMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<Rows>, Int<Cols>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));
};

template <int Dqk>
using SmemLayoutQK = typename SmemLayoutQKSelector<kBlockH, Dqk, Dqk == 512>::type;

template <int Dqk>
using SmemLayoutKV = typename SmemLayoutQKSelector<kBlockTopK, Dqk, Dqk == 512>::type;

// The first 512 bytes of every KV row serve as K for QK and as V for PV.
// Re-view the same physical K-major storage as a (K=64, N=512) B operand.
template <int Dqk>
using SmemLayoutV = decltype(coalesce(
    composition(SmemLayoutKV<Dqk>{}, Layout<Shape<Int<kDv>, Int<kBlockTopK>>, Stride<Int<kBlockTopK>, _1>>{}),
    Shape<_1, _1>{}));

using SmemLayoutS = decltype(coalesce(
    tile_to_shape(UMMA::Layout_K_INTER_Atom<fp8_t>{}, Shape<Int<kBlockH>, Int<kBlockTopK>>{}, Step<_1, _2>{}),
    Shape<_1, _1>{}));

// Stage half of the 512-wide output at a time. The 256-column buffer exactly
// aliases the FP8 Q allocation (32 * 256 * bf16 == 32 * 512 * fp8), allowing
// four 64-column TMA stores to be committed together instead of fencing and
// waiting after every individual store.
using SmemLayoutO = decltype(coalesce(
    tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16_t>{}, Shape<Int<kBlockH>, Int<256>>{}, Step<_1, _2>{}),
    Shape<_1, _1>{}));

using TiledMmaQK = decltype(make_tiled_mma(
    SM100_MMA_F8F6F4_WS_SS_SGL<fp8_t, fp8_t, float, kBlockH, kBlockTopK, UMMA::Major::K, UMMA::Major::K>{}));

using TiledMmaPV = decltype(make_tiled_mma(
    SM100_MMA_F8F6F4_WS_SS_SGL<fp8_t, fp8_t, float, kBlockH, 256, UMMA::Major::K, UMMA::Major::MN>{}));

template <int Dqk>
struct SharedStorage {
  union {
    array_aligned<fp8_t, cosize_v<SmemLayoutQK<Dqk>>> q;
    array_aligned<bf16_t, cosize_v<SmemLayoutO>> o;
  } q_o;
  // Dqk is 512 or 576. The first 512 columns use SmemLayoutKV, while the
  // optional final 64-column RoPE tail immediately follows it.
  array_aligned<fp8_t, kBlockTopK * Dqk> kv[kBuffersFor<Dqk>];
  array_aligned<fp8_t, cosize_v<SmemLayoutS>> s;
  bool valid[kBuffersFor<Dqk>][kBlockTopK];
  float exchange_max[128];
  float exchange_sum[128];
  array_aligned<uint32_t, 1> tmem_start;
  transac_bar_t qk_done[kBuffersFor<Dqk>];
  transac_bar_t pv_done[kBuffersFor<Dqk>];
  transac_bar_t kv_ready[kBuffersFor<Dqk>];
  transac_bar_t p_free;
  transac_bar_t s_ready;
  transac_bar_t q_ready;
};

struct TmaParams {
  CUtensorMap tensor_map_q;
  CUtensorMap tensor_map_kv;
  CUtensorMap tensor_map_o;
  int active_heads;
};

CUTE_DEVICE void tma_load_3d(void const* desc, transac_bar_t& barrier, void* smem_ptr, int col, int head, int query) {
  uint32_t const smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
  uint32_t const barrier_addr = cute::cast_smem_ptr_to_uint(&barrier);
  int64_t const cache_hint = int64_t(TMA::CacheHintSm90::EVICT_LAST);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile."
      "mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4}], [%5], %6;\n"
      :
      : "r"(smem_addr), "l"(desc), "r"(col), "r"(head), "r"(query), "r"(barrier_addr), "l"(cache_hint)
      : "memory");
}

enum NamedBarriers : int {
  kSoftmaxExchange = 0,
  kEpilogueExchange = 1,
};

CUTE_DEVICE void tma_gather4(void const* desc, transac_bar_t& barrier, void* smem_ptr, int col, int4 rows) {
  uint32_t const smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
  uint32_t const barrier_addr = cute::cast_smem_ptr_to_uint(&barrier);
  int64_t const cache_hint = int64_t(TMA::CacheHintSm90::EVICT_LAST);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
      "mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
      :
      : "r"(smem_addr),
        "l"(desc),
        "r"(col),
        "r"(rows.x),
        "r"(rows.y),
        "r"(rows.z),
        "r"(rows.w),
        "r"(barrier_addr),
        "l"(cache_hint)
      : "memory");
}

template <int Dqk, int ProducerWarps>
CUTE_DEVICE void load_kv(
    SharedStorage<Dqk>& smem,
    SparseMlaQ8Kv8PrefillParams const& params,
    TmaParams const& tma_params,
    int const* g_indices,
    int block_idx,
    int buffer_idx) {
  int const warp = threadIdx.x / 32;
  int const lane = threadIdx.x & 31;

  // WG1 owns the validity stores; one thread per row for N=128.
  if (threadIdx.x >= 128 && threadIdx.x < 256) {
    int const row = threadIdx.x - 128;
    int const index = __ldg(g_indices + block_idx * kBlockTopK + row);
    smem.valid[buffer_idx][row] = index >= 0 && index < params.s_kv;
  }

  // Producer ranks 0..3 are warp4..7; later ranks are warp9 and above. Warp8 is
  // reserved for the tcgen05 issuer.
  int const producer_rank = warp < 8 ? warp - 4 : warp - 5;
  if (lane == 0) {
    constexpr int kTmaWidth = Dqk == 512 ? 128 : 64;
    constexpr int kGatherGroups = kBlockTopK / 4;
    for (int group = producer_rank; group < kGatherGroups; group += ProducerWarps) {
      int4 const rows = __ldg(reinterpret_cast<int4 const*>(g_indices + block_idx * kBlockTopK) + group);
      CUTE_UNROLL
      for (int tile = 0; tile < Dqk / kTmaWidth; ++tile) {
        tma_gather4(
            &tma_params.tensor_map_kv,
            smem.kv_ready[buffer_idx],
            smem.kv[buffer_idx].data() + group * 4 * kTmaWidth + tile * kBlockTopK * kTmaWidth,
            tile * kTmaWidth,
            rows);
      }
    }
  }
}

template <int Dqk, bool StoreMeta, int Threads, int ProducerWarps>
__global__ void __launch_bounds__(Threads, 1) sparse_prefill_q8kv8_sm100_kernel(
    __grid_constant__ SparseMlaQ8Kv8PrefillParams const params, __grid_constant__ TmaParams const tma_params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
  int const s_q_idx = blockIdx.x / (params.h_q / kBlockH);
  int const q_h_idx = blockIdx.x % (params.h_q / kBlockH);
  int const warp_idx = cutlass::canonical_warp_idx_sync();
  int const wg_idx = threadIdx.x / 128;
  int const wg_tid = threadIdx.x & 127;

  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SharedStorage<Dqk>*>(smem_raw);

  int const* g_indices = params.indices + s_q_idx * params.stride_indices_s_q;
  int const topk_length = params.topk_length == nullptr ? params.topk : __ldg(params.topk_length + s_q_idx);
  int const num_blocks = max((topk_length + kBlockTopK - 1) / kBlockTopK, 1);
  int const active_tile_heads = max(min(tma_params.active_heads - q_h_idx * kBlockH, kBlockH), 0);
  constexpr int kBuffers = kBuffersFor<Dqk>;

  if (warp_idx == 0) {
    if (elect_one_sync()) {
      CUTE_UNROLL
      for (int i = 0; i < kBuffers; ++i) {
        smem.qk_done[i].init(1);
        smem.pv_done[i].init(1);
        smem.kv_ready[i].init(1);
      }
      smem.p_free.init(kBlockH * 4);
      smem.s_ready.init(kBlockH * 4);
      smem.q_ready.init(1);
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(512, smem.tmem_start.data());
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }

  __syncthreads();

  if (warp_idx == 4 && elect_one_sync()) {
    constexpr int kTmaWidth = Dqk == 512 ? 128 : 64;
    CUTE_UNROLL
    for (int tile = 0; tile < Dqk / kTmaWidth; ++tile) {
      tma_load_3d(
          &tma_params.tensor_map_q,
          smem.q_ready,
          smem.q_o.q.data() + tile * kBlockH * kTmaWidth,
          tile * kTmaWidth,
          q_h_idx * kBlockH,
          s_q_idx);
    }
  }

  TiledMmaQK tiled_qk;
  TiledMmaPV tiled_pv;
  Tensor t_p = partition_fragment_C(tiled_qk, Shape<Int<kBlockH>, Int<kBlockTopK>>{});
  Tensor t_o = partition_fragment_C(tiled_pv, Shape<Int<kBlockH>, Int<kDv>>{});
  t_p.data().get() = tmem_col::kP;
  t_o.data().get() = tmem_col::kO;

  float mi = kMaxInit;
  float li = 0.0f;
  float real_mi = -CUDART_INF_F;
  float const qk_scale_log2 = __ldg(params.q_scale_ptr) * __ldg(params.kv_scale_ptr) * params.sm_scale_div_log2;

  if (wg_idx == 1 || warp_idx > 8) {
    // Nine-warp KV producer. It can run up to three blocks ahead of the MMA
    // warp for D512; D576 uses two stages to remain within the smem limit.
    for (int block = 0; block < num_blocks; ++block) {
      int const buffer = block % kBuffers;
      if (block >= kBuffers) {
        smem.pv_done[buffer].wait(((block / kBuffers) & 1) ^ 1);
      }
      load_kv<Dqk, ProducerWarps>(smem, params, tma_params, g_indices, block, buffer);
      cutlass::arch::fence_view_async_shared();
    }
  } else if (wg_idx == 2) {
    // Batch two QK/PV blocks around one softmax synchronization frame.
    if (warp_idx == 8 && elect_one_sync()) {
      smem.q_ready.arrive_and_expect_tx(kBlockH * Dqk * sizeof(fp8_t));
      smem.q_ready.wait(0);
      tcgen05_after_thread_sync();
      auto s_q = make_tensor(make_smem_ptr(smem.q_o.q.data()), SmemLayoutQK<Dqk>{});
      auto s_s = make_tensor(make_smem_ptr(smem.s.data()), SmemLayoutS{});
      for (int step = 0; step < num_blocks + 1; ++step) {
        if (step < num_blocks) {
          int const buffer = step % kBuffers;
          smem.p_free.wait((step & 1) ^ 1);
          smem.kv_ready[buffer].arrive_and_expect_tx(kBlockTopK * Dqk * sizeof(fp8_t));
          smem.kv_ready[buffer].wait((step / kBuffers) & 1);
          tcgen05_after_thread_sync();
          auto s_k = make_tensor(make_smem_ptr(smem.kv[buffer].data()), SmemLayoutQK<Dqk>{});
          umma_ss(tiled_qk, s_q, s_k, t_p, true);
          umma_arrive(smem.qk_done[buffer]);
        }
        if (step > 0) {
          int const prev = step - 1;
          int const buffer = prev % kBuffers;
          smem.s_ready.wait(prev & 1);
          tcgen05_after_thread_sync();
          auto s_v = make_tensor(make_smem_ptr(smem.kv[buffer].data()), SmemLayoutV<Dqk>{});
          umma_ss(tiled_pv, s_s, s_v, t_o, prev == 0);
          umma_arrive(smem.pv_done[buffer]);
        }
      }
    }
  } else {
    // QK's N=128 fragment is split evenly across all four warps.  They keep a
    // common online-softmax state so each can independently rescale its
    // quarter of the N=512 output fragment when the running maximum changes.
    int const warp_in_wg = wg_tid / 32;
    int const lane = wg_tid & 31;
    bool const active_head = lane < active_tile_heads;
    int const col_base = warp_in_wg * 32;
    float p[32];
    for (int block = 0; block < num_blocks; ++block) {
      int const buffer = block % kBuffers;
      smem.qk_done[buffer].wait((block / kBuffers) & 1);
      tcgen05_after_thread_sync();
      tmem_load<32>(tmem_col::kP, p);
      cutlass::arch::fence_view_async_tmem_load();
      tcgen05_before_thread_sync();
      smem.p_free.arrive();

      float cur_max = -CUDART_INF_F;
      if (active_head) {
        CUTE_UNROLL
        for (int i = 0; i < 32; ++i) {
          int const col = col_base + i;
          bool const valid = col < topk_length - block * kBlockTopK && smem.valid[buffer][col];
          p[i] = valid ? p[i] : -CUDART_INF_F;
          cur_max = max(cur_max, p[i]);
        }
      }
      cur_max *= qk_scale_log2;
      smem.exchange_max[wg_tid] = cur_max;
      NamedBarrier::arrive_and_wait(kBlockH * 4, kSoftmaxExchange);
      cur_max =
          max(max(smem.exchange_max[lane], smem.exchange_max[32 + lane]),
              max(smem.exchange_max[64 + lane], smem.exchange_max[96 + lane]));
      if constexpr (StoreMeta) {
        real_mi = max(real_mi, cur_max);
      }

      // Keep the current softmax frame while the new block remains close.
      // This is algebraically valid (the block probabilities are evaluated in
      // the old frame) and avoids a full 64x512 TMEM read/scale/write for most
      // blocks. __any_sync makes the TMEM branch warp-uniform as required.
      bool const should_scale_o = block == 0 || __any_sync(0xffffffff, cur_max - mi > 6.0f);
      float const new_mi = should_scale_o ? max(mi, cur_max) : mi;
      float const alpha = block == 0 ? 0.0f : exp2f(mi - new_mi);
      if (block > 0 && should_scale_o) {
        int const prev_buffer = (block - 1) % kBuffers;
        smem.pv_done[prev_buffer].wait(((block - 1) / kBuffers) & 1);
        tcgen05_after_thread_sync();
        float o[32];
        CUTE_UNROLL
        for (int chunk = 0; chunk < (kDv / 4) / 32; ++chunk) {
          tmem_load<32>(tmem_col::kO + chunk * 32, o);
          cutlass::arch::fence_view_async_tmem_load();
          CUTE_UNROLL
          for (int i = 0; i < 32; ++i) {
            o[i] *= alpha;
          }
          tmem_store<32>(tmem_col::kO + chunk * 32, o);
          cutlass::arch::fence_view_async_tmem_store();
        }
        tcgen05_before_thread_sync();
      }

      float local_sum = 0.0f;
      auto s_s = make_tensor(make_smem_ptr(smem.s.data()), SmemLayoutS{});
      if (active_head) {
        CUTE_UNROLL
        for (int i = 0; i < 32; ++i) {
          float const prob = exp2f(p[i] * qk_scale_log2 - new_mi);
          local_sum += prob;
          s_s(lane, col_base + i) = fp8_t(prob * kProbFp8Scale);
        }
      }
      smem.exchange_sum[wg_tid] = local_sum;
      NamedBarrier::arrive_and_wait(kBlockH * 4, kSoftmaxExchange);
      li = li * alpha + smem.exchange_sum[lane] + smem.exchange_sum[32 + lane] + smem.exchange_sum[64 + lane] +
           smem.exchange_sum[96 + lane];
      mi = new_mi;
      cutlass::arch::fence_view_async_shared();
      smem.s_ready.arrive();
    }

    int const last = num_blocks - 1;
    smem.pv_done[last % kBuffers].wait((last / kBuffers) & 1);
    tcgen05_after_thread_sync();
  }

  if (wg_idx == 0) {
    int const warp_in_wg = wg_tid / 32;
    int const lane = wg_tid & 31;
    bool const active_head = lane < active_tile_heads;
    float const sink =
        params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + q_h_idx * kBlockH + lane) * CUDART_L2E_F;
    float const denom = li + exp2f(sink - mi);
    float const output_scale = li == 0.0f ? 0.0f : __ldg(params.kv_scale_ptr) / (kProbFp8Scale * denom);

    CUTE_UNROLL
    for (int wave = 0; wave < 2; ++wave) {
      float o[64];
      tmem_load<64>(tmem_col::kO + wave * 64, o);
      cutlass::arch::fence_view_async_tmem_load();
      if constexpr (!StoreMeta) {
        // The trusted DeepSeek-V4 backend supplies a compact active-head
        // output.  Each warp owns one 64-column TMEM tile, so it can write
        // that tile directly without the shared-memory/TMA epilogue's two
        // warpgroup rendezvous per wave.
        if (active_head) {
          bf16_t* g_o = params.out + (s_q_idx * tma_params.active_heads + q_h_idx * kBlockH + lane) * kDv +
                        (wave * 4 + warp_in_wg) * 64;
          CUTE_UNROLL
          for (int i = 0; i < 64; i += 8) {
            uint4 packed;
            auto* values = reinterpret_cast<bf16_t*>(&packed);
            CUTE_UNROLL
            for (int j = 0; j < 8; ++j) {
              values[j] = bf16_t(o[i + j] * output_scale);
            }
            *reinterpret_cast<uint4*>(g_o + i) = packed;
          }
        }
      } else {
        auto s_o = make_tensor(make_smem_ptr(smem.q_o.o.data()), SmemLayoutO{});
        if (active_head) {
          CUTE_UNROLL
          for (int i = 0; i < 64; i += 8) {
            uint4 packed;
            auto* values = reinterpret_cast<bf16_t*>(&packed);
            CUTE_UNROLL
            for (int j = 0; j < 8; ++j) {
              values[j] = bf16_t(o[i + j] * output_scale);
            }
            *reinterpret_cast<uint4*>(&s_o(lane, warp_in_wg * 64 + i)) = packed;
          }
        }
        cutlass::arch::fence_view_async_shared();
        NamedBarrier::arrive_and_wait(kBlockH * 4, kEpilogueExchange);

        if (wg_tid == 0) {
          CUTE_UNROLL
          for (int tile = 0; tile < 4; ++tile) {
            SM90_TMA_STORE_3D::copy(
                &tma_params.tensor_map_o,
                smem.q_o.o.data() + tile * kBlockH * 64,
                (wave * 4 + tile) * 64,
                q_h_idx * kBlockH,
                s_q_idx);
          }
          cute::tma_store_arrive();
          cute::tma_store_wait<0>();
        }
        NamedBarrier::arrive_and_wait(kBlockH * 4, kEpilogueExchange);
      }
    }

    if constexpr (StoreMeta) {
      if (warp_in_wg == 0 && lane < active_tile_heads) {
        int const meta_idx = s_q_idx * params.h_q + q_h_idx * kBlockH + lane;
        if (li == 0.0f) {
          params.max_logits[meta_idx] = -CUDART_INF_F;
          params.lse[meta_idx] = -CUDART_INF_F;
        } else {
          params.max_logits[meta_idx] = real_mi * CUDART_LN2_F;
          params.lse[meta_idx] = mi * CUDART_LN2_F + logf(li);
        }
      }
    }
  }

  __syncthreads();
  if (warp_idx == 0) {
    cute::TMEM::Allocator1Sm().free(0, 512);
  }
#else
  if (cute::thread0()) {
    CUTE_INVALID_CONTROL_PATH("SM100 Q8KV8 kernel requires sm_100a/f");
  }
#endif
}

template <int Dqk, bool StoreMeta = true>
inline void run_sparse_prefill_q8kv8_sm100(SparseMlaQ8Kv8PrefillParams const& params, int active_heads) {
  KU_ASSERT(params.h_kv == 1);
  KU_ASSERT(params.h_q > 0 && params.h_q % kBlockH == 0);
  KU_ASSERT(params.d_qk == Dqk);
  KU_ASSERT(params.d_v == kDv);
  KU_ASSERT(params.topk > 0 && params.topk % 128 == 0);
  KU_ASSERT(params.topk <= kMaxBlocks * kBlockTopK);
  KU_ASSERT(active_heads > 0 && active_heads <= params.h_q);

  CUtensorMap tensor_map_q;
  uint64_t q_size[3] = {Dqk, static_cast<uint64_t>(active_heads), static_cast<uint64_t>(params.s_q)};
  uint64_t q_stride[2] = {static_cast<uint64_t>(Dqk), static_cast<uint64_t>(active_heads) * Dqk};
  constexpr uint32_t kTmaWidth = Dqk == 512 ? 128 : 64;
  uint32_t q_box_size[3] = {kTmaWidth, kBlockH, 1};
  uint32_t q_elem_stride[3] = {1, 1, 1};
  CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &tensor_map_q,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
      3,
      const_cast<uint8_t*>(params.q),
      q_size,
      q_stride,
      q_box_size,
      q_elem_stride,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      Dqk == 512 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  KU_ASSERT(result == CUresult::CUDA_SUCCESS);

  CUtensorMap tensor_map_kv;
  uint64_t size[2] = {Dqk, static_cast<uint64_t>(params.s_kv)};
  uint64_t stride[1] = {static_cast<uint64_t>(params.stride_kv_s_kv) * sizeof(fp8_t)};
  uint32_t box_size[2] = {kTmaWidth, 1};
  uint32_t elem_stride[2] = {1, 1};
  result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &tensor_map_kv,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
      2,
      const_cast<uint8_t*>(params.kv),
      size,
      stride,
      box_size,
      elem_stride,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      Dqk == 512 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  KU_ASSERT(result == CUresult::CUDA_SUCCESS);

  CUtensorMap tensor_map_o;
  uint64_t o_size[3] = {kDv, static_cast<uint64_t>(active_heads), static_cast<uint64_t>(params.s_q)};
  uint64_t o_stride[2] = {
      static_cast<uint64_t>(kDv) * sizeof(bf16_t), static_cast<uint64_t>(active_heads) * kDv * sizeof(bf16_t)};
  uint32_t o_box_size[3] = {64, static_cast<uint32_t>(min(active_heads, kBlockH)), 1};
  uint32_t o_elem_stride[3] = {1, 1, 1};
  result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &tensor_map_o,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      3,
      params.out,
      o_size,
      o_stride,
      o_box_size,
      o_elem_stride,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  KU_ASSERT(result == CUresult::CUDA_SUCCESS);
  TmaParams tma_params{tensor_map_q, tensor_map_kv, tensor_map_o, active_heads};

  bool const use_long_producer = params.s_q >= kLongProducerMinSq;
  auto kernel = use_long_producer
                    ? &sparse_prefill_q8kv8_sm100_kernel<Dqk, StoreMeta, kLongThreads, kLongProducerWarps>
                    : &sparse_prefill_q8kv8_sm100_kernel<Dqk, StoreMeta, kShortThreads, kShortProducerWarps>;
  constexpr size_t smem_size = sizeof(SharedStorage<Dqk>);
  KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  int const grid = params.s_q * (params.h_q / kBlockH);
  int const threads = use_long_producer ? kLongThreads : kShortThreads;
  kernel<<<grid, threads, smem_size, params.stream>>>(params, tma_params);
  KU_CHECK_KERNEL_LAUNCH();
}

}  // namespace sglang::sm100_q8kv8
