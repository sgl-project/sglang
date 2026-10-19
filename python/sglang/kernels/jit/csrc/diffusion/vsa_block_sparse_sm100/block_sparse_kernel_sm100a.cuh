// Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
// (fastvideo-kernel/csrc/attention/block_sparse_kernel_sm100a.cuh, Apache-2.0). Inference-only
// forward; the sm_103a device pass is admitted alongside sm_100a.

// block_sparse_kernel_sm100a.cuh -- VSA block-sparse FMHA forward (per-q-block top-k), sm_100a.
// Warp-specialized: load / MMA (tcgen05) / softmax / correction / epilogue / scheduler.
// Writes O and, when asked, the log-sum-exp the backward consumes.
//
// Generated (comments stripped). Do not edit by hand.
#ifndef BLOCK_SPARSE_VSA_KERNEL_SM100A_CUH
#define BLOCK_SPARSE_VSA_KERNEL_SM100A_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include "primitives.cuh"

#ifndef VSA_BLK128
#define VSA_BLK128 false
#endif
#ifndef VSA_BHSD
#define VSA_BHSD false
#endif

// BLK128 is a file-scope constexpr, not a template parameter, so the blk64 and blk128 builds
// would instantiate the SAME kernel symbol with DIFFERENT bodies -- an ODR violation the linker
// resolves by silently keeping one. A config-named namespace keeps the two builds' symbols
// distinct so both can live in one extension.
#if VSA_BLK128
#define VSA_NAMESPACE vsa_blk128
#else
#define VSA_NAMESPACE vsa_blk64
#endif
namespace VSA_NAMESPACE {

constexpr bool BLK128 = VSA_BLK128;

#ifndef VSA_DEFER_ROWSUM
#define VSA_DEFER_ROWSUM (!VSA_BLK128)
#endif
constexpr bool DEFER_ROWSUM = VSA_DEFER_ROWSUM;

constexpr int BLOCK  = BLK128 ? 128 : 64;
constexpr int M_TILE = BLOCK;
constexpr int M_TILES_PER_CTA = 2;
constexpr int K_TILE = 256 / (BLK128 ? 2 : 1);
constexpr int BLOCKS_PER_KTILE = K_TILE / BLOCK;
constexpr int KV_HALF = K_TILE / 2;
constexpr int HEAD_DIM = 128;

constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
constexpr int Q_SUBTILES = HEAD_DIM / SUB_COLS_BF16;
constexpr int K_SUBTILES = HEAD_DIM / SUB_COLS_BF16;
constexpr int V_SUBTILES = (BLK128 ? K_TILE : KV_HALF) / SUB_COLS_BF16;
constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;
constexpr int K_SUB_COLS_BYTES = K_TILE * SUB_COLS_BYTES;
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_COLS_BYTES;

constexpr int KV_RING_SLOT_BYTES = 32 * 1024;
constexpr int SLOTS_PER_KV_TILE = BLK128 ? 1 : 2;

constexpr int NUM_KV_STAGES = BLK128 ? 3 : 4;

constexpr int V_BLK_BYTES = HEAD_DIM * SUB_COLS_BYTES;

constexpr int S_COLS = 128;
constexpr int K_ATOMS_PER_TILE = SUB_COLS_BF16 / 16;
constexpr int SPLIT_P_N    = S_COLS / 4 * 3;
constexpr int SPLIT_P_ATOM = SPLIT_P_N / 16;
constexpr int SPLIT_P_COL  = SPLIT_P_N / 2;
constexpr int EX2_FRG_PAIRS = 16;
constexpr int EX2_FRG_CNT   = S_COLS / 32;
constexpr int EX2_FREQ      = 16;
constexpr int EX2_RES       = 4;

constexpr int O_COLS = HEAD_DIM;
constexpr int TMEM_TOTAL = 512;

constexpr int STATS = BLK128 ? M_TILE : 2 * M_TILE;
constexpr int STAT_REGIONS = BLK128 ? 2 : 3;

constexpr int W_CORR0 = 8, W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 2;

extern __shared__ __align__(1024) uint8_t fmha_smem[];

union SmemDescPair {
  uint64_t u64;
  uint2 w;
};

__device__ __forceinline__ void desc_add_lo(SmemDescPair& d, uint32_t inc) {
  asm volatile(
      "{\n\t"
      ".reg .b32 lo, hi;\n\t"
      "mov.b64 {lo, hi}, %0;\n\t"
      "add.u32 lo, lo, %1;\n\t"
      "mov.b64 %0, {lo, hi};\n\t"
      "}"
      : "+l"(d.u64)
      : "r"(inc));
}

__device__ __forceinline__ void tcgen05_mma_ws_f16_ss_1sm_lead(uint32_t lead,
    uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "@q tcgen05.mma.ws.cta_group::1.kind::f16 [%1], %2, %3, %5, p, 0;\n\t"
    "}\n"
    :: "r"(lead), "r"(tmem_d), "l"(desc_a), "l"(desc_b),
       "r"(enable_input_d ? 1u : 0u), "r"(idesc));
}
__device__ __forceinline__ void tcgen05_mma_ws_f16_ts_1sm_lead(uint32_t lead,
    uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "@q tcgen05.mma.ws.cta_group::1.kind::f16 [%1], [%2], %3, %5, p, 0;\n\t"
    "}\n"
    :: "r"(lead), "r"(tmem_d), "r"(tmem_a), "l"(desc_b),
       "r"(enable_input_d ? 1u : 0u), "r"(idesc));
}

struct WorkItem {
  int sample;
  int head;
  int mtile0;
  int mtile1;
  int global_mtile0;
  int global_mtile1;
  // Shared trip count for the CTA's q-tile pair: max of the two tiles' own
  // counts, floored at 1. Every warp role derives its loop bounds from this
  // one value, so the producer/consumer mbarrier pairing stays symmetric for
  // ANY per-tile counts (including 0): a tile shorter than the pair max runs
  // its extra iterations against clamped (valid, in-bounds) KV blocks and
  // masks them to -inf via the per-tile count below, contributing nothing.
  int num_kv_blocks;
  // Each q-tile's OWN q2k_num. Used for (a) the q2k_idx window clamp in the
  // load warp and (b) the vbs-threshold masking in the softmax warps. The
  // pre-fix kernel used q2k_num[global_mtile0] for BOTH tiles, silently
  // corrupting one tile of any pair whose rows have different counts.
  int num_kv_blocks_mt[2];
};
template <bool Q_RASTER>
__device__ __forceinline__ WorkItem decode_workitem(
    int workitem_id, int num_heads, int num_blocks, int packed_mtiles_per_seq,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2, const int* q2k_num) {
  WorkItem it;
  const int per_sample = num_heads * packed_mtiles_per_seq;
  it.sample = (int)fdiv((unsigned)workitem_id, magic0);
  const int rem = workitem_id - it.sample * per_sample;
  int p;
  if constexpr (Q_RASTER) {
    it.head = (int)fdiv((unsigned)rem, magic2);
    p       = rem - it.head * packed_mtiles_per_seq;
  } else {
    p       = (int)fdiv((unsigned)rem, magic1);
    it.head = rem - p * num_heads;
  }
  it.mtile0 = 2 * p;
  it.mtile1 = 2 * p + 1;
  it.global_mtile0  = (it.sample * num_heads + it.head) * num_blocks + it.mtile0;
  it.global_mtile1  = it.global_mtile0 + 1;
  it.num_kv_blocks_mt[0] = q2k_num[it.global_mtile0];
  it.num_kv_blocks_mt[1] = q2k_num[it.global_mtile1];
  // Floor of 1: a pair whose rows are BOTH empty still walks one K-tile so no
  // mbarrier wait is left without its arrive (the load/MMA/softmax/correction
  // pipeline has a fixed per-iteration handshake); the per-tile counts mask
  // that K-tile completely, so such rows produce exactly-zero output.
  it.num_kv_blocks = max(max(it.num_kv_blocks_mt[0], it.num_kv_blocks_mt[1]), 1);
  return it;
}

template <int S_LD_COLS = 32, bool FULL_NAMED_BAR = false, bool EX2_EMU = false, bool SPLIT_P = false,
          bool SOFTMAX_THROTTLE = false, bool USE_CLC = true, bool Q_RASTER = true, bool MHA = true,
          int RESCALE_THRESHOLD = 8,

          bool BHSD = false>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
fmha_context_bf16_gen_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v_t,
      const __grid_constant__ CUtensorMap tmap_v,
    const __grid_constant__ CUtensorMap tmap_o, int seqlen,
    int num_heads, float scale_log2, int num_samples, int num_blocks,
    int packed_mtiles_per_seq, int max_kv,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    const int* __restrict__ q2k_idx, const int* __restrict__ q2k_num,
    const int* __restrict__ variable_block_sizes,
      float* __restrict__ lse_out) {
// Multi-arch builds: torch's cmake appends -gencode for EVERY entry of
// TORCH_CUDA_ARCH_LIST to this TU on top of the pinned compute_100a pass, and
// tcgen05/setmaxnreg do not exist outside sm_100a -- ptxas rejects the sm_120a
// (or plain sm_100) pass outright. Keep the body only where it can compile:
// the host pass (no __CUDA_ARCH__, needed for launch plumbing) and the
// sm_100a device pass (arch 1000 WITH the family-specific feature set that the
// "a" suffix defines). Every other device pass gets an empty stub; the Python
// is_supported() / host launcher never dispatch here off sm_100, so the stub
// is unreachable at runtime.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL)) || \
    (__CUDA_ARCH__ == 1030 && defined(__CUDA_ARCH_FEAT_SM103_ALL))

  const int total_workitems = num_samples * num_heads * packed_mtiles_per_seq;

  uint8_t* sQ0 = fmha_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sQ[2] = { sQ0, sQ1 };
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO0 = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * KV_RING_SLOT_BYTES);
  __nv_bfloat16* sO1 = sO0 + M_TILE * HEAD_DIM;
  __nv_bfloat16* sO_bufs[2] = { sO0, sO1 };
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sO1) + M_TILE * HEAD_DIM * sizeof(__nv_bfloat16));
  uint64_t* empty_bar= full_bar + NUM_KV_STAGES;
  uint64_t* full_bar_q  = empty_bar + NUM_KV_STAGES;
  uint64_t* empty_bar_q   = full_bar_q + 2;
  uint64_t* full_bar_spo  = empty_bar_q + 2;
  uint64_t* empty_bar_spo = full_bar_spo + 2;
  uint64_t* full_bar_o_acc   = empty_bar_spo + 2;
  uint64_t* full_bar_alpha = full_bar_o_acc + 2;
  uint64_t* full_bar_l   = full_bar_alpha + 2;
  uint64_t* full_bar_p_last    = full_bar_l + 2;
  uint64_t* empty_bar_alpha_and_l = full_bar_p_last + 2;
  uint64_t* full_bar_o_epi  = empty_bar_alpha_and_l + 2;
  uint64_t* empty_bar_o_epi = full_bar_o_epi + 2;
  uint64_t* clc_full  = empty_bar_o_epi + 2;
  uint64_t* clc_empty = clc_full + CLC_STAGES;
  uint32_t* clc_response = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(clc_empty + CLC_STAGES) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;
  float* alpha_and_l_smem = reinterpret_cast<float*>(tmem_slot + 2);

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  const uint32_t tmem_base = *tmem_slot;

  if (tid == 0) {
    #pragma unroll
    for (int s = 0; s < NUM_KV_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar[s]), 1);
    }
    for (int i = 0; i < 2; ++i) {
      mbarrier_init(smem_ptr_u32(&full_bar_q[i]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_q[i]), 1);
      mbarrier_init(smem_ptr_u32(&full_bar_l[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_spo[i]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_spo[i]), 256);
      mbarrier_init(smem_ptr_u32(&full_bar_o_acc[i]), 1);
      mbarrier_init(smem_ptr_u32(&full_bar_alpha[i]), 128);
      mbarrier_init(smem_ptr_u32(&empty_bar_alpha_and_l[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_p_last[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_o_epi[i]), 128);
      mbarrier_init(smem_ptr_u32(&empty_bar_o_epi[i]), 1);
    }
    if constexpr (USE_CLC) {
      #pragma unroll
      for (int s = 0; s < CLC_STAGES; ++s) {
        mbarrier_init(smem_ptr_u32(&clc_full[s]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[s]), N_WARPS);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i)
        clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();

  if (warp_id == W_LOAD) {
    setmaxnreg_dec<48>();

    EmptyPhaseTracker<NUM_KV_STAGES> kv_empty_ph;
    EmptyPhaseTracker<1> q_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int q_start = it.sample * seqlen;
      const int k_start = it.sample * seqlen;
      const int global_mtile[2]  = { it.global_mtile0, it.global_mtile1 };
      const int mtile[2] = { it.mtile0, it.mtile1 };
      const int num_k_tiles = (it.num_kv_blocks + BLOCKS_PER_KTILE - 1) / BLOCKS_PER_KTILE;

      int window_start[2] = { -(1 << 30), -(1 << 30) };
      int kv_block_id_cache[2] = { 0, 0 };
      auto get_kv_block_id = [&](int mtile_idx, int j) -> int {
        if (j >= window_start[mtile_idx] + 32) {
          window_start[mtile_idx] = j & ~31;
          // Clamp into THIS tile's own row of q2k_idx (the pair may run more
          // iterations than this tile has blocks -- see WorkItem). Positions
          // past the tile's count re-read its last valid entry; an empty tile
          // never dereferences its row at all and loads block 0 instead. Both
          // are fully masked by the per-tile vbs thresholds, so only the
          // load address safety matters here.
          const int cnt = it.num_kv_blocks_mt[mtile_idx];
          const int idx = max(0, min(window_start[mtile_idx] + lane, cnt - 1));
          kv_block_id_cache[mtile_idx] =
              (cnt > 0) ? q2k_idx[global_mtile[mtile_idx] * max_kv + idx] : 0;
        }
        return __shfl_sync(0xffffffffu, kv_block_id_cache[mtile_idx], j & 31);
      };

      auto load_k_oneslot = [&](int mtile_idx, int ktile_idx, int s) {
        const int kv_stage = kv_empty_ph.get_stage();

        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();

        int kv_tok[BLOCKS_PER_KTILE];
        #pragma unroll
        for (int blk = 0; blk < BLOCKS_PER_KTILE; ++blk)
          kv_tok[blk] = k_start + get_kv_block_id(mtile_idx, ktile_idx * BLOCKS_PER_KTILE + blk) * BLOCK;
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), KV_RING_SLOT_BYTES);
          if constexpr (BLK128) {

            if constexpr (BHSD)
                  tma_load_4d(smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES)), &tmap_k, smem_ptr_u32(&full_bar[kv_stage]),
                              0, kv_tok[0] - k_start, 0, it.sample * num_heads + it.head);
                else
                  tma_load_3d(smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES)), &tmap_k, smem_ptr_u32(&full_bar[kv_stage]),
                              0, kv_tok[0], it.head * K_SUBTILES);
          } else {
            #pragma unroll
            for (int blk = 0; blk < BLOCKS_PER_KTILE; ++blk)
              if constexpr (BHSD)
                    tma_load_4d(smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES) + blk * BLOCK * SUB_COLS_BYTES),
                                &tmap_k, smem_ptr_u32(&full_bar[kv_stage]),
                                0, kv_tok[blk] - k_start, s, it.sample * num_heads + it.head);
                  else
                    tma_load_3d(smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES) + blk * BLOCK * SUB_COLS_BYTES),
                                &tmap_k, smem_ptr_u32(&full_bar[kv_stage]),
                                0, kv_tok[blk], it.head * K_SUBTILES + s);
          }
        }
      };
      auto load_v_oneslot = [&](int mtile_idx, int ktile_idx, int p) {
        const int kv_stage = kv_empty_ph.get_stage();

        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();

        if constexpr (BLK128) {

          const int kv_tok0 = k_start + get_kv_block_id(mtile_idx, ktile_idx) * BLOCK;
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), KV_RING_SLOT_BYTES);

              if constexpr (BHSD)
                tma_load_4d(smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES)), &tmap_v, smem_ptr_u32(&full_bar[kv_stage]),
                            0, kv_tok0 - k_start, 0, it.sample * num_heads + it.head);
              else
                tma_load_3d(smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES)), &tmap_v, smem_ptr_u32(&full_bar[kv_stage]),
                            0, kv_tok0, it.head * K_SUBTILES);
          }
        } else {
          int kv_tok[2];
          #pragma unroll
          for (int h = 0; h < 2; ++h)
            kv_tok[h] = k_start + get_kv_block_id(mtile_idx, ktile_idx * BLOCKS_PER_KTILE + 2 * h + p) * BLOCK;
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), KV_RING_SLOT_BYTES);

              #pragma unroll
              for (int h = 0; h < 2; ++h) {
                #pragma unroll
                for (int s2 = 0; s2 < K_SUBTILES; ++s2) {
                  const uint32_t dst = smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES)
                                                    + h * V_BLK_BYTES + s2 * (BLOCK * SUB_COLS_BYTES));
                  if constexpr (BHSD)
                    tma_load_4d(dst, &tmap_v, smem_ptr_u32(&full_bar[kv_stage]),
                                0, kv_tok[h] - k_start, s2, it.sample * num_heads + it.head);
                  else
                    tma_load_3d(dst, &tmap_v, smem_ptr_u32(&full_bar[kv_stage]),
                                0, kv_tok[h], it.head * K_SUBTILES + s2);
                }
              }
          }
        }
      };
      auto load_k = [&](int mtile_idx, int ktile_idx) {
        load_k_oneslot(mtile_idx, ktile_idx, 0);
        if constexpr (!BLK128) load_k_oneslot(mtile_idx, ktile_idx, 1);
      };
      auto load_v = [&](int mtile_idx, int ktile_idx) {
        load_v_oneslot(mtile_idx, ktile_idx, 0);
        if constexpr (!BLK128) load_v_oneslot(mtile_idx, ktile_idx, 1);
      };

      load_k(0, 0); load_k(1, 0);

      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
        const int tok0 = q_start + mtile[m] * BLOCK;
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), Q_TILE_BYTES);

          tma_load_4d(smem_ptr_u32(sQ[m]), &tmap_q, smem_ptr_u32(&full_bar_q[m]),
                        0, BHSD ? it.sample * num_heads + it.head : it.head,
                        BHSD ? tok0 - q_start : tok0, 0);
        }
      }
      q_empty_ph.advance();
      for (int ktile_idx = 0; ktile_idx + 1 < num_k_tiles; ++ktile_idx) {
        load_v(0, ktile_idx); load_k(0, ktile_idx + 1);
        load_v(1, ktile_idx); load_k(1, ktile_idx + 1);
      }
      load_v(0, num_k_tiles - 1); load_v(1, num_k_tiles - 1);

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_MMA) {
    setmaxnreg_dec<48>();

    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;

    const uint32_t idesc_qk = make_idesc_bf16_f32(M_TILE, K_TILE, false, false);
    const uint32_t idesc_pv = make_idesc_bf16_f32(M_TILE, BLK128 ? HEAD_DIM : 2 * HEAD_DIM,
                                                 false, true);

    constexpr uint32_t DESC_LBO_MN = (uint32_t)((BLK128 ? K_TILE : BLOCK) * SUB_COLS_BF16 * 2);
    const uint64_t desc_v0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO_MN,
                                                       SmemSwizzleBlackwell::B128);
    constexpr uint32_t PV_DESC_STEP = (uint32_t)((16 * SUB_COLS_BF16 * 2) >> 4);
    const uint64_t desc_q0  = build_smem_desc_blackwell(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_kv0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);

    constexpr uint64_t KV_DESC_DELTA    = KV_RING_SLOT_BYTES >> 4;
    constexpr uint64_t Q_SUB_DELTA        = Q_SUB_COLS_BYTES >> 4;
    constexpr uint64_t Q_MTILE_DESC_DELTA = Q_TILE_BYTES >> 4;

    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<1> q_ph;
    PhaseTracker<1> spo_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int num_k_tiles = (it.num_kv_blocks + BLOCKS_PER_KTILE - 1) / BLOCKS_PER_KTILE;

      auto bmm1 = [&](int i) {
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint64_t da_base = desc_q0 + (uint64_t)i * Q_MTILE_DESC_DELTA;

        int slot = 0;
        #pragma unroll
        for (int s = 0; s < Q_SUBTILES; ++s) {
          if (!BLK128 || s == 0) {
            slot = kv_ph.get_stage();
            mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
            kv_ph.advance();
          }

          const uint64_t da = da_base + (uint64_t)s * Q_SUB_DELTA;
          const uint64_t db = desc_kv0 + (uint64_t)slot * KV_DESC_DELTA
                            + (BLK128 ? (uint64_t)s * (K_SUB_COLS_BYTES >> 4) : 0u);
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const bool enable_d = (s != 0) || (ki != 0);
            if constexpr (BLK128)
              tcgen05_mma_f16_ss_lead(lead, s_tmem_addr, da + 2 * ki, db + 2 * ki, idesc_qk, enable_d);
            else
              tcgen05_mma_ws_f16_ss_1sm_lead(lead, s_tmem_addr, da + 2 * ki, db + 2 * ki, idesc_qk, enable_d);
          }
          if (!BLK128 || s == Q_SUBTILES - 1)
            tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[slot]));
        }

        tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_spo[i]));
      };

      auto bmm2 = [&](int i, bool first_ktile, auto last_c) {
        constexpr bool last_ktile = decltype(last_c)::value;
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);

        int slot = 0;
        #pragma unroll
        for (int p = 0; p < V_SUBTILES; ++p) {
        SmemDescPair desc_bv;
          if (!BLK128 || p == 0) {
            slot = kv_ph.get_stage();
            mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
            kv_ph.advance();
          }
            if (!BLK128 || p == 0) {
              desc_bv.u64 = desc_v0;
              desc_bv.w.x += (uint32_t)(slot * (int)KV_DESC_DELTA);
            }
          const uint64_t dbV = desc_kv0 + (uint64_t)slot * KV_DESC_DELTA
                             + (BLK128 ? (uint64_t)p * (V_BLK_BYTES >> 4) : 0u);
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const int a = p * K_ATOMS_PER_TILE + ki;
            if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
              mbarrier_wait_parity(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
            }
            const bool accumulate = (!first_ktile) || (a != 0);
            if constexpr (BLK128) {
                tcgen05_mma_f16_ts_1sm_lead(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), desc_bv.u64, idesc_pv, accumulate);
                desc_add_lo(desc_bv, PV_DESC_STEP);
              } else {
                tcgen05_mma_ws_f16_ts_1sm_lead(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), desc_bv.u64, idesc_pv, accumulate);
                desc_add_lo(desc_bv, PV_DESC_STEP);
              }
          }

          if (!BLK128 || p == V_SUBTILES - 1) {
            tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[slot]));
          }
        }

        if constexpr (last_ktile) {
          tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_o_acc[i]));
        }
      };

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());
        bmm1(i);
      }

      for (int k = 0; k + 1 < num_k_tiles; ++k) {
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
          bmm2(i, (k == 0), std::false_type{});
          bmm1(i);
        }

        spo_ph.advance();
      }

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i)
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar_q[i]));
      q_ph.advance();

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
        bmm2(i, (num_k_tiles == 1), std::true_type{});
      }

      spo_ph.advance();

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_EPI) {
    setmaxnreg_dec<48>();

    PhaseTracker<1> full_o_ph;

    if (elect_one_sync()) {
      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m)
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[m]));
    }
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int q_start = it.sample * seqlen;
      const int mtile[2] = { it.mtile0, it.mtile1 };

      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_epi[m]), full_o_ph.get_phase());

        if (elect_one_sync()) {
          const int tok0 = q_start + mtile[m] * BLOCK;

          tma_store_4d(&tmap_o, 0, BHSD ? it.sample * num_heads + it.head : it.head,
                         BHSD ? tok0 - q_start : tok0, 0,
                       smem_ptr_u32(reinterpret_cast<const uint8_t*>(sO_bufs[m])));
          cp_async_bulk_commit_group();
        }
      }

      if (elect_one_sync()) {
        cp_async_bulk_wait_group_read<1>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[0]));
        cp_async_bulk_wait_group_read<0>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[1]));
      }

      full_o_ph.advance();
      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_SCHED) {
    setmaxnreg_dec<48>();

    if constexpr (USE_CLC) {
      int prod_stage = 0; uint32_t prod_phase = 1;
      int cons_stage = 0; uint32_t cons_phase = 0;
      while (true) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        clc_arrive_expect_tx_cta(smem_ptr_u32(&clc_full[prod_stage]), 16);
        if (lane == 0)
          clc_try_cancel_async(smem_ptr_u32(&clc_response[prod_stage * 4]),
                               smem_ptr_u32(&clc_full[prod_stage]));
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, cons_stage, cons_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(cons_stage, cons_phase);
        if (!next.valid) break;
      }

      for (int s = 0; s < CLC_STAGES; ++s) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      }
    }
  }
  else if (warp_id >= W_CORR0 && warp_id < W_MMA) {
    setmaxnreg_dec<80>();

    const int corr_warp_id = warp_id - W_CORR0;
    const int corr_tid = corr_warp_id * 32 + lane;
    const int corr_row = BLK128 ? corr_tid : (corr_tid & 63);
    const bool kv_half0 = BLK128 || corr_tid < 64;
    [[maybe_unused]] PhaseTracker<1> alpha_ph;
    PhaseTracker<1> o_acc_ph;
    PhaseTracker<1> o_epi_empty_ph;

    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
      mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
    }

    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int num_k_tiles = (it.num_kv_blocks + BLOCKS_PER_KTILE - 1) / BLOCKS_PER_KTILE;

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
      }
      if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();

      for (int k = 1; k < num_k_tiles; ++k) {
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
          else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());

          const float alpha = alpha_and_l_smem[(i * STAT_REGIONS + 0) * STATS + corr_tid];
          if constexpr (!SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));

          bool skip = __all_sync(0xffffffffu, alpha == 1.0f);
          if (!skip) {

            const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);

            const float2 alpha2 = f32x2_splat(alpha);

            #pragma unroll
            for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
              uint32_t o_regs[16];
              tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
              float2* o2 = reinterpret_cast<float2*>(o_regs);
              #pragma unroll
              for (int e = 0; e < 8; ++e) o2[e] = fmul2(o2[e], alpha2);
              tcgen05_st_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
            }
            tcgen05_wait_st();

            tcgen05_fence_before_thread_sync();
          }
          if constexpr (SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
        }
        if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();
      }

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());

        float scale_own;
        if constexpr (BLK128) {

          const float l = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + corr_tid];
          mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
          scale_own = (l > 0.f) ? rcp_approx_ftz_f32(l) : 0.f;
        } else {
          const float l_own = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + corr_tid];
          const float m_own = alpha_and_l_smem[(i * STAT_REGIONS + 2) * STATS + corr_tid];
          const float l_par = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + (corr_tid ^ 64)];
          const float m_par = alpha_and_l_smem[(i * STAT_REGIONS + 2) * STATS + (corr_tid ^ 64)];
          mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
          const float d = (m_own - m_par) * scale_log2;
          const float beta_lo = ex2_approx_f32(-fabsf(d));
          const float beta_own = (d >= 0.f) ? 1.f : beta_lo;
          const float beta_par = (d >= 0.f) ? beta_lo : 1.f;
          const float l_tot = beta_own * l_own + beta_par * l_par;
          scale_own = (l_tot > 0.f) ? beta_own * rcp_approx_ftz_f32(l_tot) : 0.f;

            if (lse_out != nullptr && (corr_tid >> 6) == 0) {
              const float m_tot = (d >= 0.f) ? m_own : m_par;
              const int q_row = (i == 0 ? it.mtile0 : it.mtile1) * BLOCK + (corr_tid & 63);
              const float l_safe = (l_tot > 0.f) ? l_tot : 1.0f;
              lse_out[((long)it.sample * num_heads + it.head) * (long)seqlen + q_row] =
                  m_tot * scale_log2 + __log2f(l_safe);
            }
        }
        const float2 scale2 = f32x2_splat(scale_own);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
        if constexpr (BLK128) {

          #pragma unroll
          for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
            uint32_t o_regs[16];
            tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
            if (c0 == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
            const float2* o2 = reinterpret_cast<const float2*>(o_regs);
            const int s = c0 / SUB_COLS_BF16;
            const int v_base = (c0 % SUB_COLS_BF16) / 8;
            __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
            #pragma unroll
            for (int vv = 0; vv < 2; ++vv) {
              const int v = v_base + vv;
              const float2 r0 = fmul2(o2[vv * 4 + 0], scale2);
              const float2 r1 = fmul2(o2[vv * 4 + 1], scale2);
              const float2 r2 = fmul2(o2[vv * 4 + 2], scale2);
              const float2 r3 = fmul2(o2[vv * 4 + 3], scale2);
              uint4 packed;
              packed.x = cvt_f32x2_to_bf16x2(r0.x, r0.y);
              packed.y = cvt_f32x2_to_bf16x2(r1.x, r1.y);
              packed.z = cvt_f32x2_to_bf16x2(r2.x, r2.y);
              packed.w = cvt_f32x2_to_bf16x2(r3.x, r3.y);
              *reinterpret_cast<uint4*>(&so_sub[corr_row * SUB_COLS_BF16 + (v ^ (corr_row & 7)) * 8]) = packed;
            }
          }
        } else {

          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
          if (!kv_half0) {
            #pragma unroll
            for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
              uint32_t o_regs[16];
              tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
              float2* o2 = reinterpret_cast<float2*>(o_regs);
              #pragma unroll
              for (int e = 0; e < 8; ++e) o2[e] = fmul2(o2[e], scale2);
              const int s = c0 / SUB_COLS_BF16;
              const int v_base = (c0 % SUB_COLS_BF16) / 8;
              __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
              #pragma unroll
              for (int vv = 0; vv < 2; ++vv) {
                const int v = v_base + vv;
                uint4 packed;
                packed.x = cvt_f32x2_to_bf16x2(o2[vv * 4 + 0].x, o2[vv * 4 + 0].y);
                packed.y = cvt_f32x2_to_bf16x2(o2[vv * 4 + 1].x, o2[vv * 4 + 1].y);
                packed.z = cvt_f32x2_to_bf16x2(o2[vv * 4 + 2].x, o2[vv * 4 + 2].y);
                packed.w = cvt_f32x2_to_bf16x2(o2[vv * 4 + 3].x, o2[vv * 4 + 3].y);
                *reinterpret_cast<uint4*>(&so_sub[corr_row * SUB_COLS_BF16 + (v ^ (corr_row & 7)) * 8]) = packed;
              }
            }
          }
          bar_sync<9>(128);
          if (kv_half0) {
            #pragma unroll
            for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
              uint32_t o_regs[16];
              tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
              float2* o2 = reinterpret_cast<float2*>(o_regs);
              #pragma unroll
              for (int e = 0; e < 8; ++e) o2[e] = fmul2(o2[e], scale2);
              const int s = c0 / SUB_COLS_BF16;
              const int v_base = (c0 % SUB_COLS_BF16) / 8;
              __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
              #pragma unroll
              for (int vv = 0; vv < 2; ++vv) {
                const int v = v_base + vv;
                __nv_bfloat16* dst = &so_sub[corr_row * SUB_COLS_BF16 + (v ^ (corr_row & 7)) * 8];
                uint4 h1 = *reinterpret_cast<uint4*>(dst);
                o2[vv * 4 + 0] = fadd2(o2[vv * 4 + 0], __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&h1.x)));
                o2[vv * 4 + 1] = fadd2(o2[vv * 4 + 1], __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&h1.y)));
                o2[vv * 4 + 2] = fadd2(o2[vv * 4 + 2], __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&h1.z)));
                o2[vv * 4 + 3] = fadd2(o2[vv * 4 + 3], __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&h1.w)));
                uint4 packed;
                packed.x = cvt_f32x2_to_bf16x2(o2[vv * 4 + 0].x, o2[vv * 4 + 0].y);
                packed.y = cvt_f32x2_to_bf16x2(o2[vv * 4 + 1].x, o2[vv * 4 + 1].y);
                packed.z = cvt_f32x2_to_bf16x2(o2[vv * 4 + 2].x, o2[vv * 4 + 2].y);
                packed.w = cvt_f32x2_to_bf16x2(o2[vv * 4 + 3].x, o2[vv * 4 + 3].y);
                *reinterpret_cast<uint4*>(dst) = packed;
              }
            }
          }
          bar_sync<9>(128);
        }

        tcgen05_fence_before_thread_sync();

        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));

        fence_proxy_async_shared();

        mbarrier_arrive(smem_ptr_u32(&full_bar_o_epi[i]));
      }
      o_acc_ph.advance();
      o_epi_empty_ph.advance();

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else {
    setmaxnreg_inc<192>();

    const int warp_id_u = __shfl_sync(0xffffffffu, warp_id, 0);
    const int m_tile = warp_id_u < 4 ? 0 : 1;
    const int warp_in_group = warp_id_u & 3;

    const int sm_tid = warp_in_group * 32 + lane;
    const uint32_t alpha_slot_u32 = smem_ptr_u32(&alpha_and_l_smem[(m_tile * STAT_REGIONS + 0) * STATS + sm_tid]);
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int num_k_tiles = (it.num_kv_blocks + BLOCKS_PER_KTILE - 1) / BLOCKS_PER_KTILE;

      float m_run = -INFINITY, l_run = 0.f;
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
      scale_empty_ph.advance();

      int thr_window = -(1 << 30);
      int thr_cache0 = BLOCK, thr_cache1 = BLOCK;
      auto get_vbs_thresholds = [&](int k, int gqb_mt, int half, int& t0, int& t1) {
        if (k >= thr_window + 32) {
          thr_window = k & ~31;
          const int kk = thr_window + lane;
          // THIS tile's own count (this warp group serves exactly one m_tile).
          // Positions at or past it -- the pair-max padding, and everything in
          // an empty row -- get threshold 0, i.e. the whole 64/128-token block
          // masks to -inf, regardless of which KV block the load warp fetched.
          const int cnt = it.num_kv_blocks_mt[m_tile];
          if constexpr (BLK128) {
            thr_cache0 = (kk < cnt)
                       ? variable_block_sizes[q2k_idx[gqb_mt * max_kv + kk]] : 0;
          } else {
            const int b0 = kk * BLOCKS_PER_KTILE + 2 * half;
            thr_cache0 = (b0     < cnt)
                       ? variable_block_sizes[q2k_idx[gqb_mt * max_kv + b0]]     : 0;
            thr_cache1 = (b0 + 1 < cnt)
                       ? variable_block_sizes[q2k_idx[gqb_mt * max_kv + b0 + 1]] : 0;
          }
        }
        t0 = __shfl_sync(0xffffffffu, thr_cache0, k & 31);
        t1 = BLK128 ? 0 : __shfl_sync(0xffffffffu, thr_cache1, k & 31);
      };
      auto softmax_step = [&](auto is_first_c, int k) {
        constexpr bool IS_FIRST = decltype(is_first_c)::value;
        const int gqb_mt_ = (m_tile == 0) ? it.global_mtile0 : it.global_mtile1;
        int vbs_thr0_, vbs_thr1_;
        get_vbs_thresholds(k, gqb_mt_, warp_in_group >> 1, vbs_thr0_, vbs_thr1_);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());

        uint32_t s_regs[S_COLS];
        #pragma unroll
        for (int c0 = 0; c0 < S_COLS; c0 += S_LD_COLS) {
          const uint32_t taddr = s_tmem_addr + (uint32_t)c0;
          if      constexpr (S_LD_COLS == 32)  tcgen05_ld_32x32b_x32 (taddr, *reinterpret_cast<uint32_t(*)[32]>(&s_regs[c0]));
          else if constexpr (S_LD_COLS == 64)  tcgen05_ld_32x32b_x64 (taddr, *reinterpret_cast<uint32_t(*)[64]>(&s_regs[c0]));
          else if constexpr (S_LD_COLS == 128) tcgen05_ld_32x32b_x128(taddr, *reinterpret_cast<uint32_t(*)[128]>(&s_regs[c0]));
        }

        float* scores = reinterpret_cast<float*>(s_regs);
        float2* scores2 = reinterpret_cast<float2*>(s_regs);

        tcgen05_fence_before_thread_sync();

        if constexpr (BLK128) {
          if (vbs_thr0_ < S_COLS) mask_s_row_r2p<false, S_COLS>(scores, 0, 0, vbs_thr0_);
        } else {
          if (vbs_thr0_ < 64) mask_s_row_r2p<false, 64>(scores,      0, 0, vbs_thr0_);
          if (vbs_thr1_ < 64) mask_s_row_r2p<false, 64>(scores + 64, 0, 0, vbs_thr1_);
        }

        float rmax0 = m_run, rmax1 = -INFINITY, rmax2 = -INFINITY, rmax3 = -INFINITY;
        #pragma unroll
        for (int j = 0; j < S_COLS; j += 8) {
          rmax0 = fmaxf(fmaxf(rmax0, scores[j + 0]), scores[j + 1]);
          rmax1 = fmaxf(fmaxf(rmax1, scores[j + 2]), scores[j + 3]);
          rmax2 = fmaxf(fmaxf(rmax2, scores[j + 4]), scores[j + 5]);
          rmax3 = fmaxf(fmaxf(rmax3, scores[j + 6]), scores[j + 7]);
        }
        float new_m = fmaxf(fmaxf(rmax0, rmax1), fmaxf(rmax2, rmax3));

          new_m = fmaxf(new_m, -FLT_MAX);
        float alpha = 0.0f;
        if constexpr (!IS_FIRST) {

          const float acc_scale_ = (m_run - new_m) * scale_log2;
          if (acc_scale_ >= -(float)RESCALE_THRESHOLD) { new_m = m_run; alpha = 1.0f; }
          else                     { alpha = ex2_approx_f32(acc_scale_); }

          sts_f32(alpha_slot_u32, alpha);
        }
        if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
        else mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));

        const float2 scale2 = f32x2_splat(scale_log2);
        const float2 neg_m_scaled2 = f32x2_splat(-new_m * scale_log2);
        uint32_t p_regs[S_COLS / 2];
        [[maybe_unused]] float2 lt2_live = make_float2(IS_FIRST ? 0.0f : l_run * alpha, 0.0f);
        #pragma unroll
        for (int c = 0; c < S_COLS / 2; ++c) {
          const float2 a2 = ffma2(scores2[c], scale2, neg_m_scaled2);

          if constexpr (EX2_EMU) {
            const int jj = c / EX2_FRG_PAIRS;
            const int kk = 2 * (c % EX2_FRG_PAIRS);
            const bool use_hw = (kk % EX2_FREQ < EX2_FREQ - EX2_RES) || (jj >= EX2_FRG_CNT - 1);
            scores2[c] = use_hw ? make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y))
                                : ex2_emu_f32x2(a2.x, a2.y);
          } else {
            scores2[c] = make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y));
          }
          if constexpr (!DEFER_ROWSUM) lt2_live = fadd2(lt2_live, scores2[c]);
          p_regs[c] = cvt_f32x2_to_bf16x2(scores2[c].x, scores2[c].y);
        }
        const uint32_t p_tmem_addr = s_tmem_addr;

        if constexpr (SPLIT_P) {
          tcgen05_st_32x32b_x32(p_tmem_addr,      *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x16(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[32]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
          tcgen05_st_32x32b_x16(p_tmem_addr + SPLIT_P_COL, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[SPLIT_P_COL]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&full_bar_p_last[m_tile]));
        } else {
          tcgen05_st_32x32b_x32(p_tmem_addr,      *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x32(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[32]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
        }

        spo_ph.advance();
        float2 lt2 = lt2_live;
        if constexpr (DEFER_ROWSUM) {

          float2 lt2a = make_float2(IS_FIRST ? 0.0f : l_run * alpha, 0.0f), lt2b = make_float2(0.f, 0.f);
          float2 lt2c = make_float2(0.f, 0.f), lt2d = make_float2(0.f, 0.f);
          #pragma unroll
          for (int c = 0; c < S_COLS / 2; c += 4) {
            lt2a = fadd2(lt2a, scores2[c + 0]);
            lt2b = fadd2(lt2b, scores2[c + 1]);
            lt2c = fadd2(lt2c, scores2[c + 2]);
            lt2d = fadd2(lt2d, scores2[c + 3]);
          }
          lt2 = fadd2(fadd2(lt2a, lt2b), fadd2(lt2c, lt2d));
        }
        l_run = lt2.x + lt2.y; m_run = new_m;
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
        scale_empty_ph.advance();
      };
      if (num_k_tiles > 0) softmax_step(std::true_type{}, 0);
      for (int k = 1; k < num_k_tiles; ++k) softmax_step(std::false_type{}, k);

        if constexpr (BLK128) {
          if (lse_out != nullptr) {
            const int q_row = (m_tile == 0 ? it.mtile0 : it.mtile1) * BLOCK + sm_tid;
            const float l_safe = (l_run > 0.f) ? l_run : 1.0f;
            lse_out[((long)it.sample * num_heads + it.head) * (long)seqlen + q_row] =
                m_run * scale_log2 + __log2f(l_safe);
          }
        }

      alpha_and_l_smem[(m_tile * STAT_REGIONS + 1) * STATS + sm_tid] = l_run;
      if constexpr (!BLK128)
        alpha_and_l_smem[(m_tile * STAT_REGIONS + 2) * STATS + sm_tid] = m_run;
      if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
      else mbarrier_arrive(smem_ptr_u32(&full_bar_l[m_tile]));

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
#endif  // host pass or sm_100a device pass (multi-arch guard; see note at the top of the body)
}

}  // namespace VSA_NAMESPACE

#endif
