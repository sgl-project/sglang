#pragma once

// [Route H step2a] fold-rotation PERF PROBE toggle.
//   When defined, the packed producer skips the per-K-block R@X wgmma
//   reconstruction and writes raw unpacked X straight to sK nope. Output
//   is intentionally salad; this only measures the full-load decode tps
//   ceiling to confirm the reconstruction chain is the 1-block/SM
//   latency-bound bottleneck. Comment out for the byte-correct step1 path.
//
// [Route H step2a RESULT] Probe measured full-load 32-req decode tps = 19.54,
//   IDENTICAL to step1 byte-correct (19.53). Skipping the entire producer R@X
//   wgmma reconstruction gave ZERO tps gain -> the "per-K-block R@X rebuild is
//   the main bottleneck" hypothesis is DISPROVEN. Bottleneck is NOT producer
//   compute (consumer WG QK/PV chain + 1-block/SM low-occupancy + barrier sync
//   dominate). Probe DISABLED; default is the byte-correct step1 #else path.
//   Keep the guarded probe branch for future A/B comparison.
// #define FMLA_FOLD_ROT_PROBE 1

// [Route H step3b] producer NULL-WORK PERF PROBE toggle.
//   When defined, the packed producer skips ALL nope reconstruction
//   (bit-unpack + affine + R@X wgmma/legacy + staging->sK). Only the
//   rope direct-copy and the buffer handshake (bar_k_avail wait /
//   bar_k_local_ready arrive / is_kv_valid write) survive; sK nope stays
//   uninitialized so the output is intentionally salad.
//
//   Rationale (step3 static analysis): step2a only removed the producer
//   wgmma MATH yet kept the byte-unpack + 14 NamedBarriers + staging->sK,
//   and step1-vs-step2a shows the producer GLOBAL-LOAD volume (2464 vs
//   224 loads/thread) also does not move tps. Neither is a controlled
//   variable experiment for the producer's FIXED STRUCTURAL cost (barrier
//   handshake + 7-dim_block serial skeleton + staging). This probe zeroes
//   that entire cost in one cut:
//     tps stays ~19.5 -> producer is NOT the bottleneck at all
//       (consumer WG QK/PV chain or wall+drop_shadow multi-pool schedule);
//     tps jumps       -> the producer's fixed skeleton IS the bottleneck.
//   Occupancy lever is physically ruled out: SmemPlan ~180KB/block, 2
//   blocks need 360KB > 228KB SM90 dyn-smem cap, and __launch_bounds__
//   pins minBlocksPerSM=1. So this bisection is the only remaining cut.
//
// [Route H step3b RESULT] Probe measured full-load 32-req decode tps =
//   19.54/19.54/19.53, IDENTICAL to step2a (19.54) and step1 (19.53).
//   Zeroing the ENTIRE producer nope reconstruction (unpack + 14 barriers
//   + staging->sK + wgmma) gave ZERO tps gain. Combined with step1 (load
//   dedup, no effect) and step2a (wgmma-skip, no effect), the producer WG
//   is FULLY EXONERATED across three independent negative experiments.
//   Bottleneck is definitively NOT the producer. Pivot: consumer WG QK/PV
//   chain (inside flashmla) vs wall+drop_shadow multi-pool schedule
//   (outside flashmla) -- isolated next by a baseline template-A native
//   FP8 canary on the identical workload. Probe DISABLED.
//
// [Route H step5b] RE-ENABLE the producer-null probe on the CORRECT
//   fa68162 consumer at the ~944 tps cgon baseline. The step3b result
//   above is VOID: it was measured on the corrupted d557790 consumer
//   (19.5 tps floor). On the correct consumer, PROBE2 (step4a) only
//   removed the R-side (fill_sR+wgmma+scatter) while KEEPING fill_sX
//   decode, and step5a only nulled fill_sX LOADS while KEEPING the
//   fill_sX decode instruction海 (shift4/mask/fmaf). NCU_ROOTCAUSE_2606
//   identified the root cause as bit-unpack INSTRUCTION膨胀 (inst 6.85x,
//   shared_ld 62.7x) -- NOT the loads. No experiment has ever zeroed the
//   fill_sX DECODE instructions on the correct consumer. This probe does
//   exactly that: skips the ENTIRE nope rebuild (unpack decode + affine +
//   wgmma + staging->sK + producer barriers), keeping only rope-copy +
//   handshake, so sK nope stays uninitialized (salad). PERF-only.
//     tps JUMPS   -> the decode instruction膨胀 IS the wall; the next lever
//                    is producer decode simplification (LUT / warp-coop
//                    unpack), NOT the 500-line smem occupancy surgery.
//     tps NEUTRAL -> the producer (incl. decode膨胀) is fully hidden behind
//                    the 1-block/SM latency wall; occupancy is the only
//                    remaining lever, justifying the smem-reduction surgery.
//
// [Route H step5b RESULT] Probe measured int4 cgon decode bench tps = 940.93
//   (TPOT 27.89ms), vs step4d byte-correct baseline 944.72 (TPOT 27.14ms):
//   -0.40% tps / +2.8% TPOT = NEUTRAL, within run noise. Zeroing the ENTIRE
//   producer nope rebuild -- INCLUDING the fill_sX bit-unpack DECODE
//   instruction膨胀 (shift4/mask/fmaf) that NCU_ROOTCAUSE_2606 flagged as
//   inst 6.85x / shared_ld 62.7x -- on the CORRECT fa68162 consumer gave
//   ZERO tps gain. This is the FIRST experiment ever to zero the decode
//   instruction膨胀 on the correct consumer, and it is DECISIVE: producer
//   decode simplification (LUT / warp-coop unpack) buys NOTHING end-to-end
//   because the producer (loads + decode + wgmma + barriers) is fully hidden
//   behind the 1-block/SM latency wall. The consumer sits idle on the buffer
//   handshake regardless of how cheap the producer is. Combined with step2a /
//   step3b / step4a-PROBE2 / step5a, the producer is now exonerated across
//   FIVE independent negative experiments. OCCUPANCY is the ONLY remaining
//   lever: enabling >1 block/SM co-residency (currently smem-blocked by
//   SmemPlan 225280 B > ~116KB needed for 2 blocks under the 232448 B dyn-smem
// cap) so block A's idle consumer overlaps block B's producer. Next: reduce
//   SharedMemoryPlan to <=116KB/block. Probe DISABLED (byte-correct).
// [Route H step7] RE-ENABLE producer-null probe as the BASELINE for the
//   rope-null bisection below. Nope rebuild stays compiled out; only rope
//   direct-copy + handshake survive. per-call = 329.7us (8 rank, 1290 calls)
//   vs byte-correct 322.9us (NEUTRAL, +2% noise) -- FIRST kernel-duration
//   granularity proof that producer nope compute has ZERO per-call cost.
// [Route H step7 done] Probe DISABLED (byte-correct). Both producer-null and
//   the rope-null probe below were measured NEUTRAL at per-call granularity;
//   see the rope-null RESULT block for the decisive datum. Production build
//   carries no probe.
// #define FMLA_PRODUCER_NULL_PROBE 1

// [Route H step7] rope-gather NULL PROBE toggle (layered on producer-null).
//   With FMLA_PRODUCER_NULL_PROBE the nope rebuild is already compiled out,
//   so the producer's ONLY surviving global-memory activity is the per-token
//   rope direct-copy scattered gather (pk_row[nope_bytes..], L849-892). This
//   probe zeroes that gather too: force every rope token down the write-zeros
//   branch (skip pk_base rope_bf16 reads), leaving the producer as a PURE
//   handshake (bar_k_avail.wait -> zero-fill sK -> bar_k_local_ready.arrive).
//   Output is intentionally salad; PERF probe gated by per-call + decode tps.
//
//   Bisection of the 329.7us producer-null per-call floor:
//     per-call DROPS toward native ~39us -> the rope scattered gather is the
//       memory main-line wall; next lever = cp.async/TMA bulk rope gather.
//     per-call NEUTRAL (~329us)          -> even rope loads are hidden; the
//       329us is 100% the 1-block/SM handshake+schedule structure; the ONLY
//       lever is occupancy (>1 block/SM co-residency), justifying the smem
//       surgery. Producer memory main-line is fully exonerated.
// [Route H step7 RESULT] Probe measured rope-null per-call = 329.6us (8 rank,
//   1290 calls: 329.79/329.47/329.37/329.43/330.02/331.25/329.73/330.02us) vs
//   producer-null 329.7us -- NEUTRAL, within noise. Zeroing the producer's
//   ONLY surviving global gather (per-token rope scatter) on top of the
//   already-nulled nope rebuild moved the per-call floor by 0. The ENTIRE
//   producer memory main-line (nope loads + nope compute + rope loads + all
//   producer barriers) has ZERO per-call cost; the 329us floor is 100%
//   consumer QK/PV chain + 1-block/SM barrier handshake structure. 8th
//   independent negative experiment, 2nd at kernel-duration granularity.
//   Cross-checked with STEP6 NCU (occupancy dead-end: reg+smem double-lock
//   1-block/SM). FlashMLA kernel internals are exhaustively sealed; the e2e
//   lever is OUTSIDE the kernel (store-side + max_running_requests). Probe
//   DISABLED (byte-correct).
// #define FMLA_ROPE_NULL_PROBE 1

// [Route H step3k] in-kernel clock64 SEGMENT PROFILE toggle.
//   When defined, one representative thread per block accumulates clock64()
//   deltas for the packed producer + consumer critical-path segments into a
//   __device__ global counter array. Host run() throttled-prints the mean
//   cycles/block/segment to stderr. Byte-correct (only adds clock64 reads +
//   atomicAdds; the compute path is untouched). Purpose: split the 3.4x
//   use_packed inner-loop (step3j) into producer bar-wait / rope-copy / nope
//   rebuild vs consumer bar-wait / QK+softmax so we know which segment to
//   optimize next. Comment out for the byte-correct production build (adds no
//   counters).
// [2026-07-08] Temporarily ENABLED to re-measure segment cycles on the
//   CORRECT fa68162 consumer (the prior Route H "producer exonerated"
//   conclusion was measured at the corrupted d557790 19.5 tps floor, which
//   is void). MUST run cgoff (the readback cudaMemcpyFromSymbol is a stream
//   sync illegal under cgon capture). Revert after localizing the segment.
// [2026-07-08 step3m done] Reverted to // #define after the grouped kt-outer
//   fill_sX redundancy fix was verified at cgoff (fill_sX 1501K->690K cyc,
//   nope_rebuild 1880K->1465K, byte-correct). Production build carries no
//   clock64/atomicAdd counters.
// [2026-07-08 step3o measure] Temporarily ENABLED again to re-profile the
//   segment split AFTER step3n (RC_GROUP=4, redundancy 3x->2x). Need to know
//   the new dominant sub-segment before choosing the next target (fill_sX
//   should have dropped ~another 33%; is fill_sR __ldg or barrier now the
//   leader?). MUST run cgoff. Revert after localizing.
// [2026-07-08 step3o done] Reverted to // #define after the loop-invariant
//   hoist (fill_sX bit/byte offset + group header offset) and R-address
//   strength-reduction were verified at cgoff: nope_rebuild 875K->683K cyc
//   (-22%), fill_sR 338K->196K (-42%, no per-element int64 multiply), fill_sX
//   456K->412K (-10%). Byte-correct. Production build carries no counters.
// [2026-07-08 step3r measure] Temporarily ENABLED again to re-profile the
//   segment split AFTER step3q (fill_sX per-(token,group) header division
//   hoisted into s_hdr pre-divided table). Need the new dominant sub-segment
//   before choosing step3r target: fill_sX should drop further; is fill_sR,
//   the wgmma+barrier chunk, or the consumer bar_ready empty-wait (355K, the
//   largest single segment at step3o) now the leader? MUST run cgoff (the
//   readback cudaMemcpyFromSymbol is a stream sync illegal under cgon
//   capture). Revert after localizing.
// [2026-07-08 step3r done] Reverted to // #define after the R-prestore-bf16
//   change was measured at cgoff: fill_sR 185K->180.5K cyc (-2.4%),
//   nope_rebuild 523K->516K (-1.3%). Smaller than hoped: fill_sR is bound by
//   the 49-tile x 32 strided __ldg LATENCY, not L2 bandwidth or the bf16
//   convert, so halving the load width barely moves it. Kept because it is
//   net-positive + value-identical (bf16 RNE == the kernel's prior
//   bf16(fp32) truncation) and halves R's L2/mem footprint. Next lever is
//   fill_sX (256K, still #1). Production build carries no counters.
// [2026-07-09 step3t measure] ENABLED to split seg-7 (fill_sX 256K) into
//   pure-unpack (7) vs fence+producer-barrier empty-wait (11).
// [2026-07-09 step3t RESULT] pure_unpack=258K sX_barrier_wait=5.5K -> the 256K
//   is 98% pure unpack, NOT the producer barrier empty-wait (2%). step3s
//   working hypothesis DISPROVEN. fill_sX is a latency-bound scattered
//   per-token global read wall (~575 cyc/element), consistent with step3s
//   __ldg null-effect. Optimization must attack the unpack schedule, not the
//   barrier.
// [2026-07-09 step3u RESULT] fill_sX load/compute split (fill a 32-word reg
//   array in a pure load phase, then a pure decode phase) measured
//   pure_unpack 258K->231K cyc/block (-10.3%), value-identical (curl no salad).
//   The prior fused loop's null-check + store dependency serialized the 32
//   independent scattered loads; splitting exposes memory-level parallelism.
//   KEPT (net-positive, byte-correct). Production build disables the counters.
// [2026-07-09 step3v measure] ENABLED to re-profile the FULL segment landscape
//   AFTER step3u (fill_sX 258K->231K). Need the post-step3u balance between
//   producer nope_rebuild (~516K, fill_sX now 231K + fill_sR ~180K + wgmma +
//   scatter) and consumer bar_ready empty-wait (~272K) to decide the next
//   architectural lever: producer/consumer overlap (deepen the k-buffer
//   pipeline to hide the consumer bar_ready wait) vs attack fill_sR. MUST run
//   cgoff (the readback cudaMemcpyFromSymbol is a stream sync illegal under
//   cgon capture). Revert after localizing.
// [2026-07-09 step3v RESULT] Measured post-step3u: nope_rebuild=477K/block is
//   the producer critical path (fill_sX=231K 48%, fill_sR=168K 35%, wgmma=49K,
//   scatter=9K); CONS bar_ready=160K is the consumer IDLE-waiting for the
//   producer, while consumer QK_softmax is only 6.5K. So (1) the producer is
//   unambiguously the long pole and (2) producer/consumer overlap CANNOT help
//   -- the consumer has almost no work to overlap with (6.5K vs 160K idle).
//   Tried fill_sR load/compute split (the step3u lever): NULL effect (fill_sR
//   stayed ~168K) because fill_sR has no null-check dependency and its R reads
//   are already coalesced across the warpgroup, so the compiler already
//   pipelines them. Producer inner-loop micro-opt is EXHAUSTED. The remaining
//   levers are occupancy (2 blocks/SM is smem-blocked: SmemPlan ~180KB x2 >
//   228KB cap) and the memory main-line. Counters DISABLED for production; the
//   fill_sR split was reverted so this production build is code-identical to
//   step3u (130c8ef) except comments.
// [2026-07-13 step4b] DISABLED for the production step4b build: the clock64
//   counters inflate Duration/inst and the cudaMemcpyFromSymbol readback is a
//   stream sync illegal under cgon capture. Re-enable only for segment
//   profiling at cgoff.
// #define FMLA_CLK_PROFILE 1

// [Route H step4a] fold-rotation EXECUTION-PATH PROBE toggle.
//   When defined, the packed producer keeps the fill_sX unpack (x = code*step
//   + min) but SKIPS fill_sR + the R@X wgmma + the rC->staging->sK scatter,
//   and writes the unpacked x STRAIGHT into sK nope columns instead. This is
//   the exact producer path that the full fold-rotation design (Q_folded =
//   Q_nope @ R done once per block in the consumer WG, K = x) would run, so it
//   measures the decode-tps CEILING achievable by removing R@X from the KV
//   side. Output is intentionally salad (Q is NOT folded here) -- this is a
//   PERF probe only, gated by end-to-end decode tps, NOT correctness.
//     tps jumps  -> R@X removal is the lever; implement the full Q@R fold.
//     tps flat   -> producer is a pure fill_sX memory-latency wall; the R@X
//                   fold buys nothing and the 500-line surgery is skipped.
//   NOTE: unlike the void step2a probe (measured on the corrupted d557790
//   19.5tps floor, and which only skipped the wgmma MATH while keeping
//   fill_sR + staging), this probe removes fill_sR (35%) + wgmma (10%) +
//   scatter (2%) = the entire R-related producer cost, on the CORRECT
//   fa68162 consumer / 165tps baseline. Comment out for production.
// #define FMLA_FOLD_ROT_PROBE2 1

// [Route H step5a] fill_sX SCATTERED-GATHER NULL PROBE toggle.
//   When defined, the int4 producer load phase skips the actual per-token
//   packed-row global byte read (pk_row[byte_off4]) and substitutes a
//   constant `loaded = 0`. Everything else -- the lane-paired xor shuffle,
//   the code>>shift4 decode, the s_hdr affine fmaf, the bf16 sX_tile store,
//   the fill_sR loads, the R@X wgmma, the scatter->sK, and ALL producer
//   NamedBarriers -- runs byte-for-byte unchanged. Output is intentionally
//   salad (every code decodes to 0 -> x = fmin), so this is a PERF probe
//   only, gated by end-to-end decode tps, NOT correctness.
//
//   Rationale: step3v localized fill_sX = 231K cyc/block (48% of the 477K
//   producer nope_rebuild) as a latency-bound scattered per-token global
//   read wall (~575 cyc/element; __ldg null-effect at step3s). The two
//   negative producer experiments that "exonerated" the producer either
//   (a) ran on the VOID d557790 consumer (step2a/step3b) or (b) kept
//   fill_sX and only removed the R-side (PROBE2/step4a = fill_sR+wgmma+
//   scatter). NO experiment has zeroed fill_sX's GLOBAL LOADS on the
//   CORRECT fa68162 consumer. This probe is that missing controlled cut:
//     tps JUMPS   -> the scattered packed-KV gather IS the wall; the next
//                    lever is a cp.async / TMA bulk gather + prefetch of the
//                    224 B packed rows (memory main-line), NOT occupancy.
//     tps NEUTRAL -> the producer is fully exonerated even for its loads;
//                    the wall is 1-block/SM occupancy + barrier schedule,
//                    and the memory main-line surgery is skipped.
//   Comment out for the byte-correct production build.
// [Route H step5a RESULT] Probe measured int4 cgon decode tps = 940.21
//   (TPOT 27.95ms), vs step4d byte-correct baseline 944.72 (TPOT 27.14ms):
//   -0.48% = NEUTRAL, within run noise. Zeroing the fill_sX scattered
//   packed-KV global byte loads on the CORRECT fa68162 consumer gave ZERO
//   tps gain. This closes the last producer hypothesis: the ~231K cyc/block
//   scattered gather (step3v) is FULLY HIDDEN behind the 1-block/SM latency
//   wall -- the consumer idles waiting on the buffer handshake regardless of
//   how fast the loads complete. The producer is now exonerated across FOUR
//   independent negative cuts (step2a wgmma-skip, step3b full-rebuild-null,
//   step4a/PROBE2 R-side-remove, step5a load-null). The wall is NOT the
//   memory main-line; a cp.async/TMA bulk gather buys nothing. The ONLY
//   remaining lever is OCCUPANCY: get >1 block/SM co-resident (currently
//   smem-blocked at SmemPlan 225KB > 116KB needed for 2 blocks under the
//   232KB dyn-smem cap) so the idle consumer of block A overlaps the
//   producer of block B. Probe DISABLED; production is byte-correct.
// #define FMLA_FILL_SX_NULL_PROBE 1

#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>
#include <math_constants.h>

#include <cstdio>
#include <cstdlib>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

#include "components/dequant.h"
#include "components/helpers.h"
#include "config.h"
#include "flashmla_utils.h"
#include "splitkv_mla.h"
using namespace cute;

namespace sm90::decode::sparse_fp8 {

#ifdef FMLA_CLK_PROFILE
// [Route H step3k] segment cycle counters (device global).
//   Index layout:
//     0: producer bar_k_avail.wait   (empty-wait for buffer free)
//     1: producer rope-copy          (per-token scattered 268B rope read)
//     2: producer nope rebuild       (bit-unpack + affine + R@X wgmma/legacy)
//     3: consumer bar_k_local_ready.wait (empty-wait for producer)
//     4: consumer QK + softmax       (wgmma QK + scale_softmax + save)
//   [5]: number of accumulated (block) samples (only slot used for both WGs;
//        producer counts into 5, consumer into 6, so we can normalize each).
//   [step3l] nope_rebuild sub-segment split (accumulated over all 49
//     inner iterations per producer block, normalized by np):
//     7: fill_sX_tile  (bit-unpack + per-group affine + bf16 store)
//     8: fill_sR_tile  (R matrix __ldg + bf16 store)
//     9: wgmma chunk   (2x NamedBarrier(128) + gemm + warpgroup_wait<0>)
//    10: scatter_rC_to_sK (fence + 2x NamedBarrier + staging->sK stores)
//   Slots 11-15 unused / padding.
//   NOTE: static (not inline) __device__ -> each instantiation TU gets its
//   own copy. Safe because the kernel and its host run() readback live in the
//   same TU per (MODEL_TYPE, NUM_HEADS) instantiation. inline __device__ is
//   rejected under whole-program mode (-rdc=false).
static __device__ unsigned long long g_fmla_clk[16];

static __forceinline__ __device__ void fmla_clk_add(int seg, unsigned long long dt) {
  // Only one representative lane per warpgroup logs, to avoid 128x inflation.
  atomicAdd(&g_fmla_clk[seg], dt);
}
#endif

static constexpr float MAX_INIT_VAL = -1e30;  // Prevent (-inf) - (-inf) = nan
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using fp8_e8m0 = __nv_fp8_e8m0;

template <typename Tensor0, typename Tensor1, typename Tensor2>
__forceinline__ __device__ void scale_softmax(
    Tensor0& rP,
    Tensor1& rS,
    Tensor2& rO,
    float scale_softmax_log2,
    float sScale[],
    float rM[2],
    float rL[2],
    bool is_kv_valid[],
    int block_idx,
    int idx_in_warpgroup) {
  float scale_for_olds[2];
  CUTE_UNROLL
  for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
    Tensor cur_rP = flatten(rP(make_coord(_, local_row_idx, _), _, _));
    Tensor cur_rS = flatten(rS(make_coord(_, local_row_idx, _), _, _));
    Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));

    float cur_max = -INFINITY;
    CUTE_UNROLL
    for (int i = 0; i < size(cur_rP); ++i) {
      if (!is_kv_valid[(i & 1) + (i / 2) * 8 + (idx_in_warpgroup % 4) * 2]) cur_rP(i) = -INFINITY;
      cur_max = max(cur_max, cur_rP(i));
    }
    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

    cur_max *= scale_softmax_log2;
    float old_max = rM[local_row_idx];
    rM[local_row_idx] = max(cur_max, old_max);
    float scale_for_old = exp2f(old_max - rM[local_row_idx]);
    scale_for_olds[local_row_idx] = scale_for_old;

    CUTE_UNROLL
    for (int i = 0; i < size(cur_rO); ++i) {
      cur_rO(i) *= scale_for_old;
    }

    float cur_sum = 0;
    CUTE_UNROLL
    for (int i = 0; i < size(cur_rP); ++i) {
      cur_rP(i) = exp2f(cur_rP(i) * scale_softmax_log2 - rM[local_row_idx]);
      cur_rS(i) = (bf16)cur_rP(i);
      cur_sum += cur_rP(i);
    }

    rL[local_row_idx] = rL[local_row_idx] * scale_for_old + cur_sum;
  }
  if (idx_in_warpgroup % 4 == 0) *(float2*)(sScale + 2 * (idx_in_warpgroup / 4)) = *(float2*)(scale_for_olds);
}

template <ModelType MODEL_TYPE, int NUM_HEADS>
template <typename TMAParams>
__device__ void
KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(const SparseAttnDecodeParams& params, const TMAParams& tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
  const int head_block_idx = NUM_M_BLOCKS == 1 ? 0 : blockIdx.x;
  const int s_q_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int idx_in_cluster = CLUSTER_SIZE == 1 ? 0 : head_block_idx % 2;
  const int warpgroup_idx = cutlass::canonical_warp_group_idx();
  const int idx_in_warpgroup = threadIdx.x % 128;
  const int warp_idx = cutlass::canonical_warp_idx_sync();

  // Define shared tensors
  extern __shared__ char wksp_buf[];
  SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
  Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
  Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
  Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
  Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
  float* sM = plan.sM;
  float* sL = plan.sL;
  float* sScale = plan.sScale;

  // Prefetch TMA descriptors
  if (warp_idx == 0 && elect_one_sync()) {
    cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
  }

  // Initialize TMA barriers
  if (warp_idx == 0 && elect_one_sync()) {
    plan.bar_q.init(1);
    if constexpr (CLUSTER_SIZE == 2) {
      CUTE_UNROLL
      for (int i = 0; i < NUM_K_BUFS; ++i) {
        plan.bar_k_local_ready[i].init(128);
        plan.bar_k_remote_ready[i].init(1);
        plan.bar_k_avail[i].init(4);
      }
    } else {
      CUTE_UNROLL
      for (int i = 0; i < NUM_K_BUFS; ++i) {
        plan.bar_k_local_ready[i].init(128);
        plan.bar_k_avail[i].init(256);
      }
    }
    cutlass::arch::fence_barrier_init();
  }
  ku::barrier_cluster_arrive_relaxed();

  int bar_phase_k = 0;  // Don't use array here to prevent using local memory

  // Programmatic Dependent Launch: Wait for the previous kernel to finish
  // Don't use PDL because of compiler bugs!
  // cudaGridDependencySynchronize();

  DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];

  if (sched_meta.begin_req_idx >= params.b) return;

  if (warp_idx == 0 && elect_one_sync()) {
    Tensor gQ = flat_divide(
        tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, sched_meta.begin_req_idx),
        Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{})(_, _, head_block_idx, _0{});
    launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
    plan.bar_q.arrive_and_expect_tx(BLOCK_M * HEAD_DIM_K * sizeof(bf16));
  }

  ku::barrier_cluster_wait_acquire();

  struct MainloopArgs {
    int start_block_idx, end_block_idx;
    bool is_no_split;

    // The following fields are only valid for MODEL1
    int topk_length, extra_topk_length, num_orig_kv_blocks;
  };
  auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
    MainloopArgs args;
    int total_topk_padded;
    if constexpr (MODEL_TYPE == ModelType::V32) {
      total_topk_padded = params.topk;
    } else {
      int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
      int orig_topk_padded = max(ku::ceil(topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
      int extra_topk_length =
          params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
      total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)TOPK_BLOCK_SIZE);
      args.topk_length = topk_length;
      args.extra_topk_length = extra_topk_length;
      args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;
    }

    args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
    args.end_block_idx =
        batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / TOPK_BLOCK_SIZE;
    args.is_no_split = batch_idx == sched_meta.begin_req_idx
                           ? !sched_meta.is_first_req_splitted
                           : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

    return args;
  };

  if (warpgroup_idx == 0) {
    cutlass::arch::warpgroup_reg_alloc<192>();

    TiledMMA tiled_mma_QK = TiledMMA_QK{};
    ThrMMA thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
    TiledMMA tiled_mma_PV = TiledMMA_PV_LocalP{};
    ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

    float rL[2], rM[2];
    Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V / 2>>{});
    Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
    Tensor rS = make_tensor<bf16>(partition_shape_A(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

    float rAttn_sink[2] = {-CUDART_INF_F, -CUDART_INF_F};
    if (params.attn_sink != nullptr) {
      for (int i = 0; i < 2; ++i) {
        int head_idx = head_block_idx * BLOCK_M + get_AorC_row_idx(i, idx_in_warpgroup);
        rAttn_sink[i] = __ldg((float*)params.attn_sink + head_idx) * CUDART_L2E_F;
      }
    }

#pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
      MainloopArgs args = get_cur_req_info(batch_idx);

      rL[0] = rL[1] = 0.0f;
      rM[0] = rM[1] = MAX_INIT_VAL;
      cute::fill(rO, 0.);

      // Wait for Q
      plan.bar_q.wait((sched_meta.begin_req_idx - batch_idx) & 1);

      CUTE_NO_UNROLL
      for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
        int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;
        Tensor sK = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutHalfV{});

        // Wait, issue WGMMA
#ifdef FMLA_CLK_PROFILE
        unsigned long long _clk_c0 = clock64();
#endif
        plan.bar_k_local_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
        if constexpr (CLUSTER_SIZE == 2) {
          plan.bar_k_remote_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
        }
#ifdef FMLA_CLK_PROFILE
        unsigned long long _clk_c1 = clock64();
        if (idx_in_warpgroup == 0) fmla_clk_add(3, _clk_c1 - _clk_c0);
#endif

        gemm<true, -1>(tiled_mma_QK, thr_mma_QK.partition_fragment_A(sQ), thr_mma_QK.partition_fragment_B(sK), rP);

        bar_phase_k ^= 1 << buf_idx;

        // [micro-opt B] Move sScale/sS free-barrier wait into QK wgmma
        // async window (between commit_batch and wait<0>). Barrier
        // arrive+wait is a non-matrix-pipe instruction, safe to issue
        // while QK wgmma is still in flight, so the spin-wait overlaps
        // with tensor-core compute instead of stalling after it.
        if (block_idx != args.start_block_idx) NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_free);

        cute::warpgroup_wait<0>();

        // Since in our case TOPK_BLOCK_SIZE == BLOCK_M, so we only need to do OOB checking for the last 2 blocks
        scale_softmax(
            rP,
            rS,
            rO,
            params.sm_scale_div_log2,
            sScale,
            rM,
            rL,
            plan.is_kv_valid[buf_idx],
            block_idx,
            idx_in_warpgroup);

        // Store S into shared, inform warpgroup 1
        save_rPb_to_sP(rS, sS, idx_in_warpgroup);
        fence_view_async_shared();
#ifdef FMLA_CLK_PROFILE
        {
          unsigned long long _clk_c2 = clock64();
          if (idx_in_warpgroup == 0) {
            fmla_clk_add(4, _clk_c2 - _clk_c1);
            fmla_clk_add(6, 1ull);  // consumer block sample count
          }
        }
#endif

        // Issue O += S @ V
        gemm<false, -1>(tiled_mma_PV, rS, thr_mma_PV.partition_fragment_B(sV), rO);

        NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_ready);

        cute::warpgroup_wait<0>();

        if constexpr (CLUSTER_SIZE == 2) {
          plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
          plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
        } else {
          plan.bar_k_avail[buf_idx].arrive();
        }
      }

      // Copy the next q
      if (threadIdx.x / 32 == 0 && elect_one_sync()) {
        if (batch_idx != sched_meta.end_req_idx) {
          Tensor gQ = flat_divide(
              tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, batch_idx + 1),
              Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{})(_, _, head_block_idx, _0{});
          launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
          plan.bar_q.arrive_and_expect_tx(BLOCK_M * HEAD_DIM_K * sizeof(bf16));
        } else {
          // This kernel is followed by the combine kernel, so we signal PDL here
          cudaTriggerProgrammaticLaunchCompletion();
        }
      }

      // Synchronize L and M across warpgroups
      rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
      rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
      rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
      rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

      if (idx_in_warpgroup % 4 == 0) {
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
          int row = get_AorC_row_idx(i, idx_in_warpgroup);
          sL[row] = rL[i];
          sM[row] = rM[i];
        }
      }

      float o_scales[2];
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        if (args.is_no_split) {
          o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i] + exp2f(rAttn_sink[i] - rM[i]));
        } else {
          o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i]);
        }
        if (idx_in_warpgroup % 4 == 0) {
          int row = get_AorC_row_idx(i, idx_in_warpgroup);
          plan.sOScale[row] = o_scales[i];
        }
      }

      // This is a synchronization point for warpgroup 0/1.
      // Warpgroup 0 should wait wg 1 for oBuf/oAccumBuf (overlapped with k) to be free
      // Warpgroup 1 should wait wg 0 for sL to be ready
      NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

      CUTE_UNROLL
      for (int i = 0; i < 2; ++i)
        rL[i] = rL[i] == 0.0f ? 1.0f : rL[i];

      int start_head_idx = head_block_idx * BLOCK_M;
      int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);
      if (args.is_no_split) {
        bf16* o_ptr = (bf16*)params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q +
                      start_head_idx * params.stride_o_h_q;  // (BLOCK_M, HEAD_DIM_V) : (params.stride_o_h_q, 1)
        Tensor gO = make_tensor(
            make_gmem_ptr(o_ptr),
            make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_h_q, _1{})));
        float* gSoftmaxLse = (float*)params.lse + batch_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q +
                             start_head_idx;  // (BLOCK_M) : (1)

        store_o<true>(
            rO,
            gO,
            sOBuf,
            sOAccumBuf,
            plan,
            o_scales,
            tma_params,
            batch_idx,
            s_q_idx,
            head_block_idx,
            num_valid_seq_q,
            warpgroup_idx,
            idx_in_warpgroup);

        int i = threadIdx.x;
        if (i < num_valid_seq_q) {
          float cur_L = sL[i];
          gSoftmaxLse[i] = cur_L == 0.0f ? INFINITY : logf(cur_L) + sM[i] / (float)M_LOG2E;
        }

        cute::tma_store_wait<0>();
      } else {
        int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
        float* oaccum_ptr =
            (float*)params.o_accum + split_idx * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q +
            start_head_idx * params.stride_o_accum_h_q;  // (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
        float* gSoftmaxLseAccum = (float*)params.lse_accum + split_idx * params.stride_lse_accum_split +
                                  s_q_idx * params.stride_lse_accum_s_q + start_head_idx;  // (BLOCK_M) : (1)
        Tensor gOAccum = make_tensor(
            make_gmem_ptr(oaccum_ptr),
            make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_accum_h_q, _1{})));
        store_o<false>(
            rO,
            gOAccum,
            sOBuf,
            sOAccumBuf,
            plan,
            o_scales,
            tma_params,
            batch_idx,
            s_q_idx,
            head_block_idx,
            num_valid_seq_q,
            warpgroup_idx,
            idx_in_warpgroup);

        int i = threadIdx.x;
        if (i < num_valid_seq_q) {
          float cur_L = sL[i];
          gSoftmaxLseAccum[i] = cur_L == 0.0f ? -INFINITY : log2f(cur_L) + sM[i];
        }

        cute::tma_store_wait<0>();
      }

      sync_all_threads_in_cluster();
    }
  } else if (warpgroup_idx == 1) {
    cutlass::arch::warpgroup_reg_dealloc<160>();

    TiledMMA tiled_mma_PV = TiledMMA_PV_RemoteP{};
    ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
    Tensor rO = partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V / 2>>{});

#pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
      MainloopArgs args = get_cur_req_info(batch_idx);
      cute::fill(rO, 0.);

      CUTE_NO_UNROLL
      for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
        int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;
        Tensor sV =
            make_tensor(make_smem_ptr(plan.u.k[buf_idx].data() + (SmemLayoutV{})(_256{}, _0{})), SmemLayoutHalfV{});

        // Wait for S and sScale
        NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_ready);

        // Scale O
        float cur_scales[2];
        *(float2*)cur_scales = *(float2*)(sScale + (idx_in_warpgroup / 4) * 2);
        CUTE_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
          Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));
          CUTE_UNROLL
          for (int i = 0; i < size(cur_rO); ++i) {
            cur_rO(i) *= cur_scales[local_row_idx];
          }
        }

        // Issue O += S @ V, and wait
        gemm<false, -1>(tiled_mma_PV, thr_mma_PV.partition_fragment_A(sS), thr_mma_PV.partition_fragment_B(sV), rO);
        cute::warpgroup_wait<0>();

        if constexpr (CLUSTER_SIZE == 2) {
          plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
          plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
        } else {
          plan.bar_k_avail[buf_idx].arrive();
        }

        if (block_idx != args.end_block_idx - 1)
          NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_free);  // Tell WG0 that sScale and sS are available
      }

      NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

      float o_scales[2];
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        int row = get_AorC_row_idx(i, idx_in_warpgroup);
        o_scales[i] = plan.sOScale[row];
      }

      int start_head_idx = head_block_idx * BLOCK_M;
      int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);
      if (args.is_no_split) {
        bf16* o_ptr = (bf16*)params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q +
                      start_head_idx * params.stride_o_h_q;  // (BLOCK_M, HEAD_DIM_V) : (params.stride_o_h_q, 1)
        Tensor gO = make_tensor(
            make_gmem_ptr(o_ptr),
            make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_h_q, _1{})));

        store_o<true>(
            rO,
            gO,
            sOBuf,
            sOAccumBuf,
            plan,
            o_scales,
            tma_params,
            batch_idx,
            s_q_idx,
            head_block_idx,
            num_valid_seq_q,
            warpgroup_idx,
            idx_in_warpgroup);

        cute::tma_store_wait<0>();
      } else {
        int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
        float* oaccum_ptr =
            (float*)params.o_accum + split_idx * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q +
            start_head_idx * params.stride_o_accum_h_q;  // (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
        Tensor gOAccum = make_tensor(
            make_gmem_ptr(oaccum_ptr),
            make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_accum_h_q, _1{})));
        store_o<false>(
            rO,
            gOAccum,
            sOBuf,
            sOAccumBuf,
            plan,
            o_scales,
            tma_params,
            batch_idx,
            s_q_idx,
            head_block_idx,
            num_valid_seq_q,
            warpgroup_idx,
            idx_in_warpgroup);

        cute::tma_store_wait<0>();
      }

      sync_all_threads_in_cluster();
    }
  } else {
    // Producer warpgroup
    cutlass::arch::warpgroup_reg_dealloc<152>();

    static_assert(CLUSTER_SIZE == 1 || CLUSTER_SIZE == 2);
    static constexpr int NUM_TOKENS_PER_THREAD = CLUSTER_SIZE == 1 ? 2 : 1;
    static constexpr int NUM_TOKENS_PER_ROUND =
        32;  // If head is 128, each CTA is responsible for dequantizing 32 tokens (1 rounds); if head is 64, each CTA
             // is responsible for dequantizing 64 tokens (2 rounds)
    int warp_idx = __shfl_sync(0xffffffff, idx_in_warpgroup / 32, 0);
    int lane_idx = idx_in_warpgroup % 32;
    int my_token_idx_base = warp_idx * 8 + lane_idx % 8;

    CUTE_NO_UNROLL
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
      MainloopArgs args = get_cur_req_info(batch_idx);
      int* gIndices =
          params.indices + batch_idx * params.stride_indices_b + s_q_idx * params.stride_indices_s_q;  // (topk) : (1)
      int* gExtraIndices = params.extra_indices + batch_idx * params.stride_extra_indices_b +
                           s_q_idx * params.stride_extra_indices_s_q;  // (extra_topk) : (1)

      int nxt_token_indexs[NUM_TOKENS_PER_THREAD];
      CUTE_UNROLL
      for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
        if (MODEL_TYPE == ModelType::V32 || args.start_block_idx < args.num_orig_kv_blocks)
          nxt_token_indexs[round] = __ldg(
              gIndices + args.start_block_idx * TOPK_BLOCK_SIZE + idx_in_cluster * (TOPK_BLOCK_SIZE / 2) +
              round * NUM_TOKENS_PER_ROUND + my_token_idx_base);
      }

      struct IsOrigBlock {};
      struct IsExtraBlock {};

      struct IsFirstExtraBlock {};
      struct IsNotFirstExtraBlock {};
      auto process_one_block = [&](int block_idx, auto is_extra_block_t, auto is_first_extra_block_t) {
        static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
        static constexpr bool IS_FIRST_EXTRA_BLOCK =
            std::is_same_v<decltype(is_first_extra_block_t), IsFirstExtraBlock>;
        int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;

        int* indices_base;
        int page_block_size;
        int64_t k_block_stride, k_row_stride;
        fp8* k_ptr;
        if constexpr (!IS_EXTRA_BLOCK) {
          indices_base = gIndices + (block_idx)*TOPK_BLOCK_SIZE;
          page_block_size = params.page_block_size;
          k_block_stride = params.stride_kv_block;
          k_row_stride = params.stride_kv_row;
          k_ptr = (fp8*)params.kv;
        } else {
          indices_base = gExtraIndices + (block_idx - args.num_orig_kv_blocks) * TOPK_BLOCK_SIZE;
          page_block_size = params.extra_page_block_size;
          k_block_stride = params.stride_extra_kv_block;
          k_row_stride = params.stride_extra_kv_row;
          k_ptr = (fp8*)params.extra_kv;
        }
        [[maybe_unused]] int topk_length = IS_EXTRA_BLOCK ? args.extra_topk_length : args.topk_length;
        [[maybe_unused]] int rel_block_idx = IS_EXTRA_BLOCK ? (block_idx - args.num_orig_kv_blocks) : block_idx;
        transac_bar_t* peer_bar_k_remote_ready = get_peer_addr(&(plan.bar_k_remote_ready[buf_idx]));

        // [M3.c.4 Stage-2] Packed-FP8 fused-dequant path.
        // When packed_kcache_ptr is set, we read packed INT-N rows,
        // bit-unpack + affine + R@x on the fly, and write BF16 to sK.
        // [c4c128-packed] Extra KV blocks (c4/c128 sink) now ALSO
        // support the packed path when extra_packed_kcache_ptr is set:
        // they share SWA's calib (R/scale/zero/bit_uniform/row layout),
        // so only the packed byte buffer + its per-page stride switch.
        // [DEBUG L2] packed real path re-enabled, setmaxnreg still off
        if constexpr (PACKED_INT4) {
          // ---- Packed FP8 K-load path (S2-S2 fused dequant) ----
          //
          // Process in 7 dim-blocks (448 / 64 = 7).
          // Per block (64 dims):
          //   1. compute dequant for all 64 tokens -> staging (8KB smem in union)
          //   2. each thread reads its own token's 64 dims into registers
          //   3. named barrier sync (staging no longer needed)
          //   4. each thread writes regs to GMMA-layout sK
          // This avoids smem overwrite since staging is fully read
          // before any sK writes happen.

          const int qk_nope = params.qk_nope_head_dim;
          const int row_bits = params.row_bits;
          const int packed_row_bytes = params.packed_row_bytes;
          const int nope_bytes = packed_row_bytes - 128;  // rope = 64 bf16 = 128 bytes

          // [c4c128-packed] Select the packed byte buffer + per-page
          // stride per pool. All other calib pointers are shared.
          const uint8_t* pk_base = IS_EXTRA_BLOCK ? reinterpret_cast<const uint8_t*>(params.extra_packed_kcache_ptr)
                                                  : reinterpret_cast<const uint8_t*>(params.packed_kcache_ptr);
          const float* sk_base = params.scale_kcache_ptr;
          const float* R_base = params.R_matrix_ptr;
          // [step3r] BF16-prestored R for the uniform-bit fill_sR
          //   path (set only when bit_uniform>0). Halves the R L2
          //   load width + removes per-element fp32->bf16 convert;
          //   value-identical (kernel already truncated R to bf16).
          const bf16* R_bf16_base = reinterpret_cast<const bf16*>(params.R_matrix_bf16_ptr);
          const float* zp_base = params.zero_point_ptr;
          const int* dob_base = params.dim_of_bit_ptr;
          const int* bpd_base = params.bitpos_in_dim_ptr;
          const int64_t pk_block_stride =
              IS_EXTRA_BLOCK ? params.extra_packed_kv_block_stride : params.packed_kv_block_stride;

          // PACKED_INT4 decodes directly to sK. Reuse the current sK backing
          // for the discarded generic R@X branch so its two legacy 8 KiB
          // scratch tiles do not inflate SharedMemoryPlan.
          bf16* staging = plan.u.k[buf_idx].data();

          // Wait for the nope buffer to be available
#ifdef FMLA_CLK_PROFILE
          unsigned long long _clk_p0 = clock64();
#endif
          plan.bar_k_avail[buf_idx].wait((bar_phase_k >> buf_idx & 1) ^ 1);
#ifdef FMLA_CLK_PROFILE
          unsigned long long _clk_p1 = clock64();
          if (idx_in_warpgroup == 0) fmla_clk_add(0, _clk_p1 - _clk_p0);
#endif

          if (CLUSTER_SIZE == 2 && idx_in_warpgroup == 0) {
            plan.bar_k_remote_ready[buf_idx].arrive_and_expect_tx(
                (TOPK_BLOCK_SIZE / 2) * (HEAD_DIM_NOPE + HEAD_DIM_ROPE) * sizeof(bf16));
          }

          // ---- First, copy rope half directly (no staging needed) ----
          CUTE_UNROLL
          for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
            int my_token_idx = my_token_idx_base + round * NUM_TOKENS_PER_ROUND;
            bf16* sK_rope_base = plan.u.k[buf_idx].data() +
                                 (idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx) * 8 +
                                 ((lane_idx / 8) * 8) * TOPK_BLOCK_SIZE;
            bf16* sK_rope_peer_base = get_peer_addr(sK_rope_base);

            const int token_idx_abs = idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx;
            const int token_index = __ldg(indices_base + token_idx_abs);
            [[maybe_unused]] const bool token_is_in_topk =
                rel_block_idx * TOPK_BLOCK_SIZE + token_idx_abs < topk_length;

#ifdef FMLA_ROPE_NULL_PROBE
            // [Route H step7] force write-zeros: skip the per-token
            //   packed-row rope global gather to bisect the 329us
            //   producer-null floor into memory-main-line vs
            //   handshake structure. Output is salad (PERF probe).
            if (false) {
#else
            if (token_is_in_topk && token_index != -1) {
#endif
              const int block_index = (int)((uint32_t)token_index / (uint32_t)page_block_size);
              const int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;
              const uint8_t* pk_row = pk_base + block_index * pk_block_stride + rel_idx_in_block * packed_row_bytes;
              const bf16* rope_bf16 = reinterpret_cast<const bf16*>(pk_row + nope_bytes);

              CUTE_UNROLL
              for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE / 32; dim_idx += 1) {
                bf16x8 val = *reinterpret_cast<const bf16x8*>(&rope_bf16[(lane_idx / 8) * 8 + dim_idx * 32]);
                int smem_offset = (HEAD_DIM_NOPE + dim_idx * 32) * TOPK_BLOCK_SIZE;
                *(__int128_t*)(sK_rope_base + smem_offset) = *(__int128_t*)&val;
                if constexpr (CLUSTER_SIZE == 2) {
                  st_async_128b(sK_rope_peer_base + smem_offset, val, peer_bar_k_remote_ready);
                }
              }
            } else {
              CUTE_UNROLL
              for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE / 32; dim_idx += 1) {
                bf16x8 val;
                *(uint128_t*)&val = uint128_t();
                int smem_offset = (HEAD_DIM_NOPE + dim_idx * 32) * TOPK_BLOCK_SIZE;
                *(__int128_t*)(sK_rope_base + smem_offset) = *(__int128_t*)&val;
                if constexpr (CLUSTER_SIZE == 2) {
                  st_async_128b(sK_rope_peer_base + smem_offset, val, peer_bar_k_remote_ready);
                }
              }
            }
          }
#ifdef FMLA_CLK_PROFILE
          unsigned long long _clk_p2 = clock64();
          if (idx_in_warpgroup == 0) fmla_clk_add(1, _clk_p2 - _clk_p1);
#endif

          // ==========================================================
          // [M3.c.4 Stage-5 Route G step 4+5] wgmma R@X uniform-bit
          // path (MODEL1 + CLUSTER_SIZE==1 + bu > 0 only).
          //
          // Structural rewrite that replaces the per-token 4-barrier
          // storm of the legacy inner loop with a cooperative
          // 128-thread fill + tensor-core reduction:
          //
          //   for dim_block in 0..HEAD_DIM_NOPE/64:            (7)
          //     rC[64,64] = 0                                 (fp32)
          //     for kt in 0..qk_nope/64:                        (7)
          //       128 threads cooperatively fill:
          //         sX_tile[t=0..63, d=0..63] bf16              // unpack + affine
          //         sR_tile[j=0..63, d=0..63] bf16              // R[dim_base+j, kt*64+d]
          //       fence + NamedBarrier(128)
          //       wgmma MMA_64x64x16_F32BF16BF16_SS<K,K>:
          //         rC += sX_tile @ sR_tile^T                  // 4 issues of k16 per tile
          //       warpgroup_wait<0>
          //       NamedBarrier(128)                             // release sX/sR for kt+1
          //     scatter bf16(rC) -> staging via partition_C
          //     NamedBarrier(128)
          //     staging -> sK[dim_block tile] via 128-bit stores  (reused legacy path)
          //     NamedBarrier(128)
          //
          // Barrier count per TOPK_BLOCK: 7 * (7*2 + 2) = 112,
          //   vs legacy uniform 896 (~8x), vs var-bit 1792 (~16x).
          // R@X FLOPs stay identical but come from tensor cores
          //   (wgmma m64n64k16) instead of 128 lanes x 224 FMA.
          //
          // The generic R@X path below is compile-time discarded for the
          // dedicated PACKED_INT4 specialization. Its tensor declarations
          // alias the current sK buffer instead of reserving dead scratch.
          // ==========================================================
          // Bit-uniform parameters hoisted here so they are in scope
          // for BOTH the wgmma_uniform_supported path below AND the
          // legacy fallback block that follows.
#ifndef FMLA_PRODUCER_NULL_PROBE
          constexpr int bu = 4;
          constexpr int u_groups = 14;
          constexpr int u_hdr_bytes = 16;
          constexpr int u_group_size = 32;
          constexpr float u_step_denom = 7.0f;

          constexpr bool wgmma_uniform_supported = (MODEL_TYPE == ModelType::MODEL1) && (CLUSTER_SIZE == 1);

          if constexpr (wgmma_uniform_supported) {
            if constexpr (bu > 0) {
              // [step3k smem-fit revert] Producer R@X wgmma loop,
              //   single-buffer, dim-block-by-1. Uses only
              //   legacy sX/staging and sR tiles.
              //
              // Why: packed_x_alt_tile + packed_r_alt_tile (+16 KB)
              //   were REMOVED from SharedMemoryPlan to fit under
              //   the H20 SM90 opt-in dyn-smem cap of 232448 B
              //   (MODEL1 plan was 241664 B -> cudaFuncSetAttribute
              //   invalid argument, kernel never launched). The
              //   stale Jun-30 binary masked this.
              //
              // Route H step3b (producer-null probe) PROVED the
              //   producer is NOT the decode bottleneck (zeroing
              //   the ENTIRE nope rebuild gave 0 tps change at
              //   19.53/19.54 tps). So the speculative sX-double-
              //   buffer + dim-block-by-2 pipeline those alt tiles
              //   enabled (Route G step8 + Route H step1) is dead
              //   weight -- it only traded smem for cycles the
              //   consumer WG was already waiting on.
              //
              // Byte-correctness: same R@X = X @ R.T math, same
              //   tensor shape, same staging->sK copy layout. Only
              //   the in-loop schedule changes (1-at-a-time instead
              //   of 2-at-a-time, no fill-wgmma overlap).
              Tensor sX_tile = make_tensor(make_smem_ptr(staging), SmemLayoutXTile{});
              Tensor sR_tile = make_tensor(make_smem_ptr(staging), SmemLayoutKTile{});
              Tensor sStaging =
                  make_tensor(make_smem_ptr(staging), Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, _1>>{});

              TiledMMA tiled_mma_wg = TiledMMA_QK{};
              ThrMMA thr_mma_wg = tiled_mma_wg.get_slice(idx_in_warpgroup);

              const int k_tiles = qk_nope / 64;

              // [step3p] Hoist the per-token packed-row base pointer
              //   out of fill_sX_tile into a shared table filled ONCE.
              //   pk_row = pk_base + block_index*pk_block_stride +
              //   rel_idx*packed_row_bytes depends ONLY on the token
              //   (indices_base[t] -> uint32 div + mod + 2 muls), NOT
              //   on k_base or d. The step3o fill_sX_tile recomputed
              //   that addressing (+ __ldg(index)) for all 32 elements
              //   on EVERY (group,kt) call = 14x per producer block,
              //   and identically across the 64 fx_d threads sharing
              //   an fx_th. Precompute the 64-token pointer table with
              //   64 threads, sync once, then fill_sX_tile just indexes
              //   s_pk_row[t]. Invalid tokens store nullptr (validity
              //   folded in). Byte-identical addressing + validity vs
              //   step3o; only the redundant recompute is removed.
              __shared__ const uint8_t* s_pk_row[TOPK_BLOCK_SIZE];
              if (idx_in_warpgroup < TOPK_BLOCK_SIZE) {
                const int t = idx_in_warpgroup;
                const int token_index = __ldg(indices_base + t);
                bool out_of_range = false;
                if constexpr (MODEL_TYPE == ModelType::MODEL1) {
                  if (rel_block_idx * TOPK_BLOCK_SIZE + t >= topk_length) {
                    out_of_range = true;
                  }
                }
                const bool invalid = (token_index == -1) || out_of_range;
                const uint8_t* row = nullptr;
                if (!invalid) {
                  const int block_index = (int)((uint32_t)token_index / (uint32_t)page_block_size);
                  const int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;
                  row = pk_base + block_index * pk_block_stride + rel_idx_in_block * packed_row_bytes;
                }
                s_pk_row[t] = row;
              }
              NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);

              // [step3q] Hoist the per-(token,group) affine header
              //   OUT of fill_sX_tile into a shared table filled ONCE
              //   per producer block.
              //   In fill_sX_tile the group index
              //   g = d_global / u_group_size = (k_base + fx_d) / 64
              //     = k_base >> 6  (fx_d in [0,63] -> no carry),
              //   so ALL 64 fx_d threads of a given fx_th recompute the
              //   SAME 64 tokens' (fmin, fstep) -- two __half2float
              //   widens PLUS the frange/denom DIVISION -- 64x-
              //   redundantly across fx_d, on every fill_sX_tile call.
              //   Precompute all (HEAD_DIM_NOPE/64) groups x 64 tokens
              //   ONCE with the full 128-thread WG. Store fmin and the
              //   PRE-DIVIDED fstep = frange/denom as a __half2 (4 B/
              //   entry, 1792 B) so the hot loop drops the division
              //   entirely (just two __half2float widens + fmaf). A
              //   fp32 float2 table (3584 B) overflows the H20 SM90
              //   static-smem headroom (233984 > 232448 cap); the
              //   half fstep has 10-bit mantissa vs the bf16(x_val)
              //   output's 7-bit, so the extra rounding is below output
              //   granularity -> bf16-output-identical to step3p.
              //   Invalid tokens (s_pk_row[t]==nullptr) store {0,0};
              //   the loop still gates on s_pk_row[t].
              __shared__ __half2 s_hdr[(HEAD_DIM_NOPE / 64) * TOPK_BLOCK_SIZE];
              if constexpr (!PACKED_INT4) {
                const int n_groups = k_tiles;
                const float inv_denom = 1.0f / u_step_denom;
                for (int idx = idx_in_warpgroup; idx < n_groups * TOPK_BLOCK_SIZE; idx += 128) {
                  const int t = idx & (TOPK_BLOCK_SIZE - 1);
                  const int g = idx / TOPK_BLOCK_SIZE;
                  const uint8_t* pk_row = s_pk_row[t];
                  const __half step =
                      pk_row != nullptr ? reinterpret_cast<const __half*>(pk_row + 224)[g] : __float2half(0.0f);
                  s_hdr[idx] = __half2(__float2half(0.0f), step);
                }
                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
              }

              // [step3o] Loop-invariant hoist. For a given thread the
              //   32 elements share the SAME d = lin & 63 (lin =
              //   e*128+idx, and 128 is a multiple of 64 so the low 6
              //   bits are idx's). Hence d_global and every quantity
              //   derived from it (bit/byte offset, shift, mask, group
              //   header offset) are per-thread constants and are
              //   computed ONCE outside the e-loop instead of 32x.
              //   Byte-identical: same sX_tile(t,d) mapping, same math.
              const int fx_d = idx_in_warpgroup & 63;
              const int fx_th = idx_in_warpgroup >> 6;  // 0 or 1
              auto fill_sX_tile = [&](int k_base, bool direct_staging = false, bool direct_sK = false) {
#if !defined(FMLA_ENABLE_U32_LOAD_ORACLE)
                if (bu == 4) {
                  // Map two producer threads to each token. Each
                  // thread decodes one contiguous 32-dim half-row
                  // from four aligned u32 loads instead of issuing
                  // 32 scattered byte loads across different rows.
                  const int t = idx_in_warpgroup >> 1;
                  const int d_base = (idx_in_warpgroup & 1) * 32;
                  const uint8_t* pk_row = s_pk_row[t];
                  uint32_t packed_words[4] = {};
                  if (pk_row != nullptr) {
                    const int byte_base = (k_base + d_base) >> 1;
                    CUTE_UNROLL
                    for (int w = 0; w < 4; ++w) {
                      packed_words[w] = __ldg(reinterpret_cast<const uint32_t*>(pk_row + byte_base + w * 4));
                    }
                  }

                  const int hdr_base = (k_base / u_group_size) * TOPK_BLOCK_SIZE;
                  const __half2 hdr = s_hdr[hdr_base + t];
                  const float fmin = __half2float(__low2half(hdr));
                  const float fstep = __half2float(__high2half(hdr));
                  CUTE_UNROLL
                  for (int i = 0; i < 32; ++i) {
                    const int d = d_base + i;
                    const int code = static_cast<int>((packed_words[i >> 3] >> ((i & 7) * 4)) & 0xFu);
                    const float x_val = pk_row != nullptr ? fmaf(static_cast<float>(code), fstep, fmin) : 0.0f;
                    if (direct_sK) {
                      const int dim_group = d >> 4;
                      const int dim_half = d & 8;
                      const int dim_sub = d & 7;
                      bf16* sK_nope_base = plan.u.k[buf_idx].data() + t * 8 + dim_group * 16 * TOPK_BLOCK_SIZE;
                      sK_nope_base[(k_base + dim_half) * TOPK_BLOCK_SIZE + dim_sub] = bf16(x_val);
                    } else if (direct_staging) {
                      staging[t * 64 + d] = bf16(x_val);
                    } else {
                      sX_tile(t, d) = bf16(x_val);
                    }
                  }
                  return;
                }
#endif
                const int d = fx_d;
                const int d_global = k_base + d;
                const int bit_off_global = d_global * bu;
                const int byte_off = bit_off_global >> 3;
                const int shift = bit_off_global & 7;
#if defined(FMLA_ENABLE_U32_LOAD_ORACLE)
                const int word_byte_off = byte_off & ~3;
                const int word_shift = shift + ((byte_off & 3) << 3);
                const bool use_u32_load = params.debug_u32_packed_load && bu <= 3;
                const int decode_shift = use_u32_load ? word_shift : shift;
#else
                const int decode_shift = shift;
#endif
                const uint32_t mask = (1u << bu) - 1u;
                const int g = d_global / u_group_size;
                const int hdr_base = g * TOPK_BLOCK_SIZE;  // [step3q] s_hdr row for this group

                // [step4b] bu==4 nibble-aligned compile-time-constant
                //   fast path. When bit_uniform==4 every 4-bit code
                //   is a nibble whose bit offset is d_global*4, so
                //   byte_off4 = d_global>>1 and shift4 = (d_global&1)*4
                //   in {0,4}: the code NEVER crosses a byte boundary
                //   (shift4+4 <= 8), so a SINGLE 1-byte load holds the
                //   whole code and the mask is the literal 0xF. This
                //   differs from the rejected step4a byte-align: the
                //   bu==4 vs generic split is done ONCE per 32-element
                //   phase (bu is warp-uniform -> no divergence, no
                //   per-element runtime branch), so it removes the
                //   generic path's 2nd byte OR + runtime (1<<bu)-1 mask
                //   arithmetic without the +727K per-element branch
                //   instructions that killed step4a. Value-identical
                //   to generic for bu==4: the generic word's low byte
                //   is exactly pk_row[byte_off], and shift4==shift,
                //   mask 0xF == (1<<4)-1.
                const bool int4 = (bu == 4)
#if defined(FMLA_ENABLE_U32_LOAD_ORACLE)
                                  // The debug oracle forces the generic path.
                                  && !params.debug_u32_packed_load
#endif
                    ;
                const int byte_off4 = d_global >> 1;     // (d_global*4)>>3
                const int shift4 = (d_global & 1) << 2;  // (d_global*4)&7 in {0,4}

                // [step3u] load/compute split. fill_sX is
                //   latency-bound scattered per-token global reads
                //   (step3t: pure_unpack 258K cyc/block = 98% of
                //   seg-7, ~575 cyc/element -> memory-latency wall,
                //   __ldg was null-effect at step3s). The prior
                //   fused loop chained (2x 1-byte global load ->
                //   decode -> bf16 store) per e, so the null-check
                //   control dependency + store dependency blocked
                //   the compiler from issuing the 32 INDEPENDENT
                //   scattered loads back-to-back. Split into a pure
                //   load phase (fills a 32-word register array,
                //   loads fire in parallel -> max memory-level
                //   parallelism hides latency) then a pure decode
                //   phase. Value-identical: same word bytes, same
                //   fmaf, same bf16 store, only the schedule moves.
                uint32_t words[32];
                if (int4) {
                  // [step4d] lane-paired nibble load.
                  //   byte_off4 == byte_off for bu==4; the code is
                  //   wholly inside this one byte (shift4+4<=8), so
                  //   adjacent d lanes (2m, 2m+1) need the same byte
                  //   and select low/high nibble via shift4. Load it
                  //   once in the even lane, then broadcast to the
                  //   odd lane with xor-1 shuffle. This preserves the
                  //   step4b decode exactly while halving the int4
                  //   scattered per-token byte loads.
                  CUTE_UNROLL
                  for (int e = 0; e < 32; ++e) {
                    const int t = e * 2 + fx_th;
                    const uint8_t* pk_row = s_pk_row[t];
#ifdef FMLA_FILL_SX_NULL_PROBE
                    // [step5a] Null the scattered per-token
                    //   global byte read to isolate its latency.
                    //   Shuffle/decode/store schedule unchanged.
                    const uint32_t loaded = 0u;
                    (void)pk_row;
#else
                    const uint32_t loaded = ((d & 1) == 0 && pk_row != nullptr) ? (uint32_t)pk_row[byte_off4] : 0u;
#endif
                    const uint32_t pair_loaded = __shfl_xor_sync(0xffffffffu, loaded, 1);
                    words[e] = ((d & 1) == 0) ? loaded : pair_loaded;
                  }
                } else {
                  CUTE_UNROLL
                  for (int e = 0; e < 32; ++e) {
                    const int t = e * 2 + fx_th;
                    const uint8_t* pk_row = s_pk_row[t];
                    uint32_t word = 0u;
                    if (pk_row != nullptr) {
#if defined(FMLA_ENABLE_U32_LOAD_ORACLE)
                      if (use_u32_load) {
                        // Debug/probe-only oracle path for the
                        // rejected u32-load experiment. Keep it
                        // runtime-gated so the default byte-load
                        // production path is unchanged.
                        word = __ldg(reinterpret_cast<const uint32_t*>(pk_row + word_byte_off));
                      } else {
                        word = (uint32_t)pk_row[byte_off];
                        word |= ((uint32_t)pk_row[byte_off + 1]) << 8;
                        if (bu > 8) {
                          word |= ((uint32_t)pk_row[byte_off + 2]) << 16;
                        }
                      }
#else
                      word = (uint32_t)pk_row[byte_off];
                      word |= ((uint32_t)pk_row[byte_off + 1]) << 8;
                      if (bu > 8) {
                        word |= ((uint32_t)pk_row[byte_off + 2]) << 16;
                      }
#endif
                    }
                    words[e] = word;
                  }
                }
                if (int4) {
                  // [step4b] constant shift4/0xF decode. Value-
                  //   identical to generic for bu==4 (shift4==shift,
                  //   0xF==(1<<4)-1) but the compiler folds the
                  //   literal mask + shift, dropping the runtime
                  //   (1<<bu)-1 and variable-shift arithmetic.
                  CUTE_UNROLL
                  for (int e = 0; e < 32; ++e) {
                    const int t = e * 2 + fx_th;
                    float x_val = 0.0f;
                    if (s_pk_row[t] != nullptr) {
                      const int code = (int)((words[e] >> shift4) & 0xFu);
                      const __half2 hdr = s_hdr[hdr_base + t];
                      const float fmin = __half2float(__low2half(hdr));
                      const float fstep = __half2float(__high2half(hdr));
                      x_val = fmaf((float)code, fstep, fmin);
                    }
                    if (direct_sK) {
                      // Identity-tail blocks do not need R@X. Store the
                      // decoded scalar directly into the final sK layout
                      // and avoid the row-major staging round trip.
                      const int dim_group = d >> 4;
                      const int dim_half = d & 8;
                      const int dim_sub = d & 7;
                      bf16* sK_nope_base = plan.u.k[buf_idx].data() + t * 8 + dim_group * 16 * TOPK_BLOCK_SIZE;
                      sK_nope_base[(k_base + dim_half) * TOPK_BLOCK_SIZE + dim_sub] = bf16(x_val);
                    } else if (direct_staging) {
                      staging[t * 64 + d] = bf16(x_val);
                    } else {
                      sX_tile(t, d) = bf16(x_val);
                    }
                  }
                } else {
                  CUTE_UNROLL
                  for (int e = 0; e < 32; ++e) {
                    const int t = e * 2 + fx_th;
                    float x_val = 0.0f;
                    if (s_pk_row[t] != nullptr) {
                      const int code = (int)((words[e] >> decode_shift) & mask);
                      // [step3q] (fmin, fstep) pre-divided + cached
                      //   in s_hdr; hot loop just widens + fmaf,
                      //   no per-element frange/denom division.
                      const __half2 hdr = s_hdr[hdr_base + t];
                      const float fmin = __half2float(__low2half(hdr));
                      const float fstep = __half2float(__high2half(hdr));
                      x_val = fmaf((float)code, fstep, fmin);
                    }
                    if (direct_sK) {
                      // Identity-tail blocks do not need R@X. Store the
                      // decoded scalar directly into the final sK layout
                      // and avoid the row-major staging round trip.
                      const int dim_group = d >> 4;
                      const int dim_half = d & 8;
                      const int dim_sub = d & 7;
                      bf16* sK_nope_base = plan.u.k[buf_idx].data() + t * 8 + dim_group * 16 * TOPK_BLOCK_SIZE;
                      sK_nope_base[(k_base + dim_half) * TOPK_BLOCK_SIZE + dim_sub] = bf16(x_val);
                    } else if (direct_staging) {
                      // Direct paths do not feed wgmma; keep
                      // staging row-major so the existing
                      // staging->sK vector store layout is reused.
                      staging[t * 64 + d] = bf16(x_val);
                    } else {
                      sX_tile(t, d) = bf16(x_val);
                    }
                  }
                }
              };

              auto run_warp_int4_g32_decode = [&]() {
                const int warp = idx_in_warpgroup >> 5;
                const int lane = idx_in_warpgroup & 31;
                CUTE_UNROLL
                for (int round = 0; round < 16; ++round) {
                  const int t = warp * 16 + round;
                  const uint8_t* pk_row = s_pk_row[t];
                  uint32_t word = 0;
                  if (pk_row != nullptr) {
                    word = __ldg(reinterpret_cast<const uint32_t*>(pk_row + lane * 4));
                  }
                  const int group = lane >> 2;
                  float scale = 0.0f;
                  if (pk_row != nullptr) {
                    scale = static_cast<float>(reinterpret_cast<const __nv_fp8_e4m3*>(pk_row + 224)[group]);
                  }

                  bf16x8 prefix_out;
                  bf16* prefix_elem = reinterpret_cast<bf16*>(&prefix_out);
                  CUTE_UNROLL
                  for (int j = 0; j < 8; ++j) {
                    const int nibble = static_cast<int>((word >> (j * 4)) & 0xFu);
                    const int code = (nibble ^ 8) - 8;
                    prefix_elem[j] = bf16(pk_row != nullptr ? static_cast<float>(code) * scale : 0.0f);
                  }
                  const int prefix_dim = lane * 8;
                  const int prefix_group = prefix_dim >> 4;
                  const int prefix_half = prefix_dim & 8;
                  bf16* prefix_sK = plan.u.k[buf_idx].data() + t * 8 + prefix_group * 16 * TOPK_BLOCK_SIZE;

                  bf16x8 tail_out;
                  if (lane < 24) {
                    const int tail_dim = 256 + lane * 8;
                    uint32_t tail_word = 0;
                    if (pk_row != nullptr) {
                      tail_word = __ldg(reinterpret_cast<const uint32_t*>(pk_row + (tail_dim >> 1)));
                    }
                    const int tail_hdr_group = tail_dim >> 5;
                    float tail_scale = 0.0f;
                    if (pk_row != nullptr) {
                      tail_scale =
                          static_cast<float>(reinterpret_cast<const __nv_fp8_e4m3*>(pk_row + 224)[tail_hdr_group]);
                    }
                    bf16* tail_elem = reinterpret_cast<bf16*>(&tail_out);
                    CUTE_UNROLL
                    for (int j = 0; j < 8; ++j) {
                      const int nibble = static_cast<int>((tail_word >> (j * 4)) & 0xFu);
                      const int code = (nibble ^ 8) - 8;
                      tail_elem[j] = bf16(pk_row != nullptr ? static_cast<float>(code) * tail_scale : 0.0f);
                    }
                  }

                  // Keep the logical K layout unchanged, but issue
                  // 128-bit stores in two lane phases to reduce
                  // same-cycle conflicts in the interleaved layout.
                  CUTE_UNROLL
                  for (int store_phase = 0; store_phase < 2; ++store_phase) {
                    if ((lane >> 4) == store_phase) {
                      *reinterpret_cast<__int128_t*>(prefix_sK + prefix_half * TOPK_BLOCK_SIZE) =
                          *reinterpret_cast<__int128_t*>(&prefix_out);
                    }
                    if (lane < 24 && (lane >> 4) == store_phase) {
                      const int tail_dim = 256 + lane * 8;
                      const int tail_group = tail_dim >> 4;
                      const int tail_half = tail_dim & 8;
                      bf16* tail_sK = plan.u.k[buf_idx].data() + t * 8 + tail_group * 16 * TOPK_BLOCK_SIZE;
                      *reinterpret_cast<__int128_t*>(tail_sK + tail_half * TOPK_BLOCK_SIZE) =
                          *reinterpret_cast<__int128_t*>(&tail_out);
                    }
                  }
                }
                cutlass::arch::fence_view_async_shared();
                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
              };

              // [step3o] Same hoist + strength-reduction for fill_sR.
              //   d is a per-thread constant; j advances by 2 per e, so
              //   the R address advances by a constant 2*qk_nope stride
              //   (no int64 multiply per element).
              // [step3r] R is now prestored bf16: __ldg reads 2 B and
              //   stores straight to sR_tile with no fp32->bf16 convert.
              //   Value-identical to the old fp32 __ldg + bf16() cast
              //   (bf16 store was already the gemm input precision).
              // [step3v] load/compute split (fill a 32-word reg array
              //   then store) was MEASURED here as a NULL effect:
              //   fill_sR stayed ~168K cyc/block. Unlike fill_sX
              //   (step3u), fill_sR has no null-check control
              //   dependency and its R-matrix reads are already
              //   COALESCED across the warpgroup (64 threads read 64
              //   consecutive columns = one cache line), so the
              //   compiler already pipelines the loads -> no artificial
              //   serialization to remove. Consistent with step3r
              //   (halving load width only -2.4%): fill_sR is a
              //   fundamental L2-latency wall for the constant R
              //   matrix, already at floor. REVERTED to the step3r
              //   fused form. Producer inner-loop micro-opt is now
              //   exhausted (fill_sX 231K + fill_sR 168K = 84% of
              //   nope_rebuild, both at floor); next lever is
              //   architectural (producer/consumer overlap / memory
              //   main-line).
              auto fill_sR_tile = [&](int dim_base, int k_base) {
                const int d = fx_d;
                const int d_global = k_base + d;
                const int r_stride = 2 * qk_nope;
                const bf16* r_ptr = R_bf16_base + (int64_t)(dim_base + fx_th) * (int64_t)qk_nope + (int64_t)d_global;
                CUTE_UNROLL
                for (int e = 0; e < 32; ++e) {
                  const int j = e * 2 + fx_th;
                  // Raw 16-bit read-only cached load; bf16 is a
                  //   16-bit POD so the reinterpret is exact.
                  const uint16_t rbits = __ldg(reinterpret_cast<const uint16_t*>(r_ptr));
                  sR_tile(j, d) = reinterpret_cast<const bf16&>(rbits);
                  r_ptr += r_stride;
                }
              };

              auto scatter_rC_to_sK = [&](auto& rC_frag, int dim_base) {
                Tensor tC_sStaging = thr_mma_wg.partition_C(sStaging);
                CUTE_UNROLL
                for (int i = 0; i < size(rC_frag); ++i) {
                  tC_sStaging(i) = bf16(rC_frag(i));
                }
                cutlass::arch::fence_view_async_shared();
                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);

                CUTE_UNROLL
                for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                  int my_token_idx = my_token_idx_base + round * NUM_TOKENS_PER_ROUND;
                  const int abs_token = idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx;
                  const int dim_in_block = (lane_idx / 8) * 16;

                  bf16x8 val_lo = *reinterpret_cast<bf16x8*>(staging + abs_token * 64 + dim_in_block + 0);
                  bf16x8 val_hi = *reinterpret_cast<bf16x8*>(staging + abs_token * 64 + dim_in_block + 8);

                  bf16* sK_nope_base =
                      plan.u.k[buf_idx].data() + abs_token * 8 + ((lane_idx / 8) * 16) * TOPK_BLOCK_SIZE;

                  int smem_offset_lo = (dim_base + 0) * TOPK_BLOCK_SIZE;
                  int smem_offset_hi = (dim_base + 8) * TOPK_BLOCK_SIZE;
                  *(__int128_t*)(sK_nope_base + smem_offset_lo) = *(__int128_t*)&val_lo;
                  *(__int128_t*)(sK_nope_base + smem_offset_hi) = *(__int128_t*)&val_hi;
                }

                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
              };

              auto write_staging_tile_to_sK = [&](int dim_base) {
                cutlass::arch::fence_view_async_shared();
                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
                CUTE_UNROLL
                for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                  int my_token_idx = my_token_idx_base + round * NUM_TOKENS_PER_ROUND;
                  const int abs_token = idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx;
                  const int dim_in_block = (lane_idx / 8) * 16;
                  bf16x8 val_lo = *reinterpret_cast<bf16x8*>(staging + abs_token * 64 + dim_in_block + 0);
                  bf16x8 val_hi = *reinterpret_cast<bf16x8*>(staging + abs_token * 64 + dim_in_block + 8);
                  bf16* sK_nope_base =
                      plan.u.k[buf_idx].data() + abs_token * 8 + ((lane_idx / 8) * 16) * TOPK_BLOCK_SIZE;
                  *(__int128_t*)(sK_nope_base + (dim_base + 0) * TOPK_BLOCK_SIZE) = *(__int128_t*)&val_lo;
                  *(__int128_t*)(sK_nope_base + (dim_base + 8) * TOPK_BLOCK_SIZE) = *(__int128_t*)&val_hi;
                }
                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
              };

              // [step3m] fill_sX redundancy elimination via GROUPED
              //   kt-outer / dim-inner reorder.
              //
              //   step3l localized the bottleneck: fill_sX_tile
              //   (bit-unpack + per-group affine of the packed X row)
              //   was ~78% of the whole attention call, because the
              //   original dim-outer/kt-inner loop re-unpacked each of
              //   the 7 unique X k-tiles once per dim_block -> 7x7=49
              //   fills for only 7 distinct tiles (7x redundant).
              //
              //   fill_sX_tile(k_base) depends ONLY on kt (k_base =
              //   kt*64), NOT on dim_block. So we hoist it: for each
              //   GROUP of RC_GROUP=3 dim_blocks we keep 3 rC
              //   accumulators live and, per kt, unpack sX ONCE then
              //   feed all 3 dim_blocks (each needs its own sR + gemm).
              //   fill count drops 49 -> ceil(7/3)*7 = 21 (3x instead
              //   of 7x). 3 rC = 96 regs/thread, fits the producer
              //   historical producer register budget (7 rC = 224 would
              //   not). A full kt-outer (all 7 rC) is register-
              //   infeasible; smem-caching 7 X tiles (56 KB) also
              //   overflows the ~7 KB dyn-smem headroom.
              //
              //   Byte-correct: identical R@X = X @ R.T math, identical
              //   sR load + gemm + scatter->sK per dim_block. Only the
              //   iteration order + how often sX is (re)filled changes.
              //   packed_kv_producer_sync is producer-WG-internal (128
              //   threads), so the changed barrier count is self-
              //   consistent and does not touch the consumer WG.
              constexpr int DIM_BLOCKS = HEAD_DIM_NOPE / 64;  // 7
              // [step3n] RC_GROUP raised 3 -> 4: fill_sX redundancy
              //   3x (21 fills) -> 2x (ceil(7/4)=2 groups x 7 kt = 14
              //   fills). 4 rC accumulators = 4*32 = 128 regs/thread,
              //   still under the historical producer register budget
              //   budget (5 would be 160 -> overflow). If the extra
              //   accumulator spills, the end-to-end cgon tps will
              //   regress vs step3m (259 tps) and we revert to 3.
              constexpr int RC_GROUP = 4;

              auto do_one_dim = [&](auto& rC_ref, int dim_base, int k_base) {
#ifdef FMLA_CLK_PROFILE
                unsigned long long _clk_r00 = clock64();
#endif
                fill_sR_tile(dim_base, k_base);
#ifdef FMLA_CLK_PROFILE
                unsigned long long _clk_r0 = clock64();
                if (idx_in_warpgroup == 0) fmla_clk_add(8, _clk_r0 - _clk_r00);
#endif
                cutlass::arch::fence_view_async_shared();
                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);

                gemm<false, -1>(
                    tiled_mma_wg,
                    thr_mma_wg.partition_fragment_A(sX_tile),
                    thr_mma_wg.partition_fragment_B(sR_tile),
                    rC_ref);
                cute::warpgroup_wait<0>();
                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
#ifdef FMLA_CLK_PROFILE
                unsigned long long _clk_r1 = clock64();
                if (idx_in_warpgroup == 0) fmla_clk_add(9, _clk_r1 - _clk_r0);
#endif
              };

#ifdef FMLA_FOLD_ROT_PROBE2
              // [Route H step4a] fold-rotation EXECUTION-PATH probe.
              //   Unpack each of the 7 x k-tiles and write it STRAIGHT
              //   to sK nope (direct_staging + write_staging_tile_to_sK),
              //   skipping fill_sR + the R@X wgmma + scatter_rC_to_sK.
              //   This is the exact producer path the full Q@R fold
              //   design (Q_folded = Q_nope @ R in consumer, K = x) will
              //   run, so it measures the decode-tps CEILING of removing
              //   R@X from the KV side. Output is intentionally salad
              //   (Q is NOT folded here). PERF probe only, gated by
              //   end-to-end decode tps, NOT correctness.
              CUTE_NO_UNROLL
              for (int kt = 0; kt < k_tiles; ++kt) {
                const int k_base = kt * 64;
                fill_sX_tile(k_base, true);
                write_staging_tile_to_sK(k_base);
              }
#else
              if constexpr (PACKED_INT4) {
                run_warp_int4_g32_decode();
              } else {
                CUTE_NO_UNROLL
                for (int grp0 = 0; grp0 < DIM_BLOCKS; grp0 += RC_GROUP) {
                  const int rem = DIM_BLOCKS - grp0;
                  const int G = rem < RC_GROUP ? rem : RC_GROUP;
                  if (params.identity_tail_bypass && grp0 >= 4) {
                    // build_hadamard(448) is block-diagonal:
                    // a 256-dim Hadamard prefix plus a 192-dim
                    // identity tail. For dim blocks 4..6, R@X is
                    // exactly X, so keep K-side math/rounding for
                    // the Hadamard prefix and only bypass the
                    // identity tail.
                    for (int kt = grp0; kt < grp0 + G; ++kt) {
                      const int k_base = kt * 64;
                      fill_sX_tile(k_base, false, true);
                      cutlass::arch::fence_view_async_shared();
                      NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
                    }
                    continue;
                  }

                  Tensor rC0 = partition_fragment_C(tiled_mma_wg, Shape<Int<64>, Int<64>>{});
                  Tensor rC1 = partition_fragment_C(tiled_mma_wg, Shape<Int<64>, Int<64>>{});
                  Tensor rC2 = partition_fragment_C(tiled_mma_wg, Shape<Int<64>, Int<64>>{});
                  Tensor rC3 = partition_fragment_C(tiled_mma_wg, Shape<Int<64>, Int<64>>{});
                  clear(rC0);
                  clear(rC1);
                  clear(rC2);
                  clear(rC3);

                  const int kt_end = (params.identity_tail_bypass && grp0 == 0)
                                         ? 4  // block-diagonal R: H_256 output only needs 256 input dims
                                         : k_tiles;
                  for (int kt = 0; kt < kt_end; ++kt) {
                    const int k_base = kt * 64;
#ifdef FMLA_CLK_PROFILE
                    unsigned long long _clk_s0 = clock64();
#endif
                    // Unpack the X k-tile ONCE for the whole group.
                    fill_sX_tile(k_base);
#ifdef FMLA_CLK_PROFILE
                    // [step3t] split seg-7 into pure-unpack (7) vs
                    // fence+producer-barrier empty-wait (11) to see
                    // which half of the 256K is the real cost.
                    unsigned long long _clk_sM = clock64();
                    if (idx_in_warpgroup == 0) fmla_clk_add(7, _clk_sM - _clk_s0);
#endif
                    cutlass::arch::fence_view_async_shared();
                    NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
#ifdef FMLA_CLK_PROFILE
                    unsigned long long _clk_s1 = clock64();
                    if (idx_in_warpgroup == 0) fmla_clk_add(11, _clk_s1 - _clk_sM);
#endif
                    // Feed sX to each dim_block in the group; sX
                    // stays resident, only sR is (re)loaded per dim.
                    do_one_dim(rC0, (grp0 + 0) * 64, k_base);
                    if (G > 1) do_one_dim(rC1, (grp0 + 1) * 64, k_base);
                    if (G > 2) do_one_dim(rC2, (grp0 + 2) * 64, k_base);
                    if (G > 3) do_one_dim(rC3, (grp0 + 3) * 64, k_base);
                  }

#ifdef FMLA_CLK_PROFILE
                  unsigned long long _clk_s4 = clock64();
#endif
                  scatter_rC_to_sK(rC0, (grp0 + 0) * 64);
                  if (G > 1) scatter_rC_to_sK(rC1, (grp0 + 1) * 64);
                  if (G > 2) scatter_rC_to_sK(rC2, (grp0 + 2) * 64);
                  if (G > 3) scatter_rC_to_sK(rC3, (grp0 + 3) * 64);
#ifdef FMLA_CLK_PROFILE
                  unsigned long long _clk_s5 = clock64();
                  if (idx_in_warpgroup == 0) fmla_clk_add(10, _clk_s5 - _clk_s4);
#endif
                }
              }
#endif  // FMLA_FOLD_ROT_PROBE2

              cutlass::arch::fence_view_async_shared();
              // Fall through to shared bar_k_local_ready arrive +
              // is_kv_valid write below (outside the packed branch).
            }  // end if (bu > 0) inside wgmma_uniform_supported
          }

          // [c4c128-packed] Compile the legacy scalar path only for
          // non-wgmma targets. MODEL1/CLUSTER_SIZE==1 int4 uses the
          // wgmma uniform path above; keeping the bu==0 scalar
          // fallback compiled in permanently allocates its
          // __shared__ scratch (s_codes/s_x) and pushes total shared
          // memory over the H20 launch cap. This bring-up path runs
          // with SGLANG_RQ_BIT_UNIFORM=3, so disabling MODEL1 bu==0
          // fallback is intentional until it is rewritten without
          // static shared memory.
          if constexpr (!wgmma_uniform_supported) {
            // [M3.c.4 Stage-5 Bug-3 fix] Per-token full unpack + affine
            // + R@x dequant, with **unified barrier sequence** for both
            // valid and invalid tokens.
            //
            // Calibration convention (build_rotated_kv_calib.py +
            // rotated_quant_dsv4_kernels.py):
            //   store:   K_rot = nope @ R; codes = round((K_rot - zero) / scale)
            //   load:    nope  = (codes * scale + zero) @ R.t()
            // With R row-major in memory, the inverse rotation produces
            //   result[j] = sum_d R[j, d] * x[d]
            // where x[d] = codes[d] * scale[d] + zero[d]. This mirrors
            // dense_fp8 fork's flash_fwd_mla_kernel.h prologue + prefetch.
            //
            // Why unified barriers: the previous revision had the
            // invalid path skip all 4 NamedBarriers in the per-token
            // loop while the valid path did them. Although `invalid`
            // is uniform across the producer warpgroup's 128 threads
            // per token, mixing barrier-bearing and barrier-free
            // iterations of the SAME loop creates a fragile contract
            // with the consumer warpgroups' wait on
            // bar_k_local_ready[buf_idx] (arrived after the dim_block
            // outer loop). Forcing both branches through the exact
            // same 4-barrier sequence makes the producer's smem
            // ordering provably consistent with the dense fork's
            // gold reference (which has no invalid branching at all).
            //
            // s_codes / s_x are token-scoped scratchpads shared by the
            // 128 producer-WG threads. qk_nope <= 512 (V32: 512, MODEL1:
            // 448); we size to 576 to stay above HEAD_DIM_K.
            __shared__ int s_codes[576];
            __shared__ float s_x[576];
            // [Stage-5 Route G step6.4] revert step6.3 header smem cache:
            // the extra NamedBarrier::sync(128,...) needed to publish
            // s_hdr across the 128 producer threads costs ~100 cyc/token
            // but empirically dropped 32-req steady-state gen tps from
            // 12.19 -> 6.76 (measured on fp8-dsv4 canary 09:05:35 UTC).
            // Producer WG has only 128 threads and is Q/K-bound, not
            // header-LDG bound, so per-thread ldg header (already
            // L2-hot after step6.1 warm-up) is cheaper than a
            // whole-warp synchronization. Route to reclaim tps.

            // [Stage-5 Route G step 5] uniform-bit fast path.
            //
            // Selected when params.bit_uniform > 0. Each nope dim
            // uses `bu` contiguous bits, so a single thread can
            // locate its own dim's code via byte shift + mask
            // (no atomicOr scatter). Per-token affine lives in a
            // 28 B (for 7 groups) header right after the code
            // bytes and right before the rope BF16 tail:
            //     [code_bytes][28 B header][128 B rope]
            // header[g] = (fp16 min, fp16 range), 4 B per group.
            // s_x[d] = code * (range / ((1<<bu)-1)) + min.
            //
            // Barrier count per token drops from 4 to 2 (one
            // after we fill s_x, one after R@x staging write
            // before reusing s_x for the next t).
            // (bu/u_groups/u_hdr_bytes/u_group_size/u_step_denom
            // are declared above, before the wgmma_uniform_supported
            // branch, so they are in scope here.)

            CUTE_UNROLL
            for (int dim_block = 0; dim_block < HEAD_DIM_NOPE / 64; ++dim_block) {
              const int dim_base = dim_block * 64;

              // ---- Step 1: per-token unpack + affine + R@x ----
              // Variable-width bit layout described by row_bits global bit slots:
              //   bit i lives at byte (i/8), bit (i%8) of a packed row, and
              //   contributes value (1 << bitpos_in_dim[i]) to dim_of_bit[i].
              //
              // For each token t we (always 4 NamedBarriers, both paths):
              //   (a) s_codes[d] = sum_{i : dim_of_bit[i] == d} bit(i) << bitpos_in_dim[i]
              //   (b) s_x[d]     = s_codes[d] * scale[d] + zero[d]
              //   (c) staging[t, d_in_block] = sum_d R[(dim_base+d_in_block), d] * s_x[d]
              for (int t = 0; t < TOPK_BLOCK_SIZE; ++t) {
                int token_index = __ldg(indices_base + t);
                bool out_of_range = false;
                if constexpr (MODEL_TYPE == ModelType::MODEL1) {
                  if (rel_block_idx * TOPK_BLOCK_SIZE + t >= topk_length) {
                    out_of_range = true;
                  }
                }
                const bool invalid = (token_index == -1) || out_of_range;

                // Compute pk_row pointer up-front (only used when valid).
                const uint8_t* pk_row = nullptr;
                if (!invalid) {
                  const int block_index = (int)((uint32_t)token_index / (uint32_t)page_block_size);
                  const int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;
                  pk_row = pk_base + block_index * pk_block_stride + rel_idx_in_block * packed_row_bytes;
                }

                if (bu > 0) {
                  // ---- Uniform-bit fast path (Stage-5 Route G step6.4). ----
                  // Same as step6.1: per-thread ldg header (L2-hot
                  // after 7-group warm-up on the first few tokens)
                  // + 1 FMA per dim. No smem cache (step6.3 attempt
                  // regressed 12.19 -> 6.76 tps due to the extra
                  // NamedBarrier::sync(128,...) needed to publish
                  // s_hdr across the 128 producer-WG threads).
                  const uint8_t* hdr_base = invalid ? nullptr : (pk_row + nope_bytes - u_hdr_bytes);

                  if (!invalid) {
                    for (int d = idx_in_warpgroup; d < qk_nope; d += 128) {
                      const int bit_off_global = d * bu;
                      const int byte_off = bit_off_global >> 3;
                      const int shift = bit_off_global & 7;
                      uint32_t word = (uint32_t)pk_row[byte_off];
                      word |= ((uint32_t)pk_row[byte_off + 1]) << 8;
                      if (bu > 8) {
                        word |= ((uint32_t)pk_row[byte_off + 2]) << 16;
                      }
                      const uint32_t mask = (1u << bu) - 1u;
                      const int code = (int)((word >> shift) & mask);
                      const int g = d / u_group_size;
                      const __half* hdr_h = reinterpret_cast<const __half*>(hdr_base + g * 4);
                      const float fmin = __half2float(hdr_h[0]);
                      const float frange = __half2float(hdr_h[1]);
                      const float fstep = frange * (1.0f / u_step_denom);
                      s_x[d] = fmaf((float)code, fstep, fmin);
                    }
                  } else {
                    for (int d = idx_in_warpgroup; d < qk_nope; d += 128) {
                      s_x[d] = 0.0f;
                    }
                  }
                  NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
                } else {
                  // ---- Legacy variable-bit path (path-disjoint). ----
                  // (a-1) init s_codes (always)
                  for (int d = idx_in_warpgroup; d < qk_nope; d += 128) {
                    s_codes[d] = 0;
                  }
                  NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);

                  // (a-2) atomicOr each bit slot into its dim (skip for invalid;
                  // codes remain 0 from init).
                  if (!invalid) {
                    for (int bit_idx = idx_in_warpgroup; bit_idx < row_bits; bit_idx += 128) {
                      const int d = __ldg(dob_base + bit_idx);
                      const int bpos = __ldg(bpd_base + bit_idx);
                      const int byte_off = bit_idx >> 3;
                      const int bit_off = bit_idx & 7;
                      const int bit_v = (pk_row[byte_off] >> bit_off) & 1;
                      if (bit_v) {
                        atomicOr(&s_codes[d], 1 << bpos);
                      }
                    }
                  }
                  NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);

                  // (b) affine dequant: s_x[d] = codes*scale + zero  for valid
                  //                     s_x[d] = 0                    for invalid
                  // sk_base / zp_base are per-dim length-qk_nope.
                  if (!invalid) {
                    for (int d = idx_in_warpgroup; d < qk_nope; d += 128) {
                      s_x[d] = (float)s_codes[d] * sk_base[d] + zp_base[d];
                    }
                  } else {
                    for (int d = idx_in_warpgroup; d < qk_nope; d += 128) {
                      s_x[d] = 0.0f;
                    }
                  }
                  NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
                }

                // (c) R @ s_x for this dim_block's 64 outputs.
                // [Stage-5 Route G step6.1] 双 lane 协作 + float4 向量化.
                //   - 128 lanes 全部激活: pair (l, l^1) 同 warp，
                //     每 pair 计算 1 个 output j = l/2 (lane 0/1
                //     -> j=0, lane 2/3 -> j=1, ..., lane 62/63 -> j=31;
                //     lane 64/65 -> j=32, ..., lane 126/127 -> j=63)。
                //   - 每 lane 累加 half dims (224)，从 half-offset
                //     开始，用 float4 一次 load 4 个 (R, s_x) 做 4× FMA。
                //   - qk_nope=448 -> 224/4 = 56 iter/lane。
                //   - 用 __shfl_xor_sync(mask, sum, 1) 在 pair 内合并。
                //   - 相比旧 kernel (64 lane 各 448 标量 MADD)：
                //     lane 利用率 64->128 (×2)，
                //     每 lane MADD 数 448->56*4=224 (× 0.5 计算量)，
                //     LDG/LDS 从 float 变 float4 (×4 带宽利用)。
                //     综合 ~4× 加速 R@x 阶段。
                //   - 对 s_x 语义无假设：invalid token s_x 全 0 时
                //     sum 天然为 0，barrier 序列与 legacy 完全一致。
                {
                  const int lane = idx_in_warpgroup;
                  const int pair_id = lane >> 1;  // 0..63
                  const int half = lane & 1;      // 0 or 1
                  const int j = dim_base + pair_id;
                  const float* R_row = R_base + (int64_t)j * (int64_t)qk_nope;
                  const int qk_half = qk_nope >> 1;  // 224
                  const int d_start = half * qk_half;
                  float sum = 0.0f;
#pragma unroll 1
                  for (int d = 0; d < qk_half; d += 4) {
                    const int gd = d_start + d;
                    const float4 r4 = *reinterpret_cast<const float4*>(R_row + gd);
                    const float4 x4 = *reinterpret_cast<const float4*>(&s_x[gd]);
                    sum += r4.x * x4.x;
                    sum += r4.y * x4.y;
                    sum += r4.z * x4.z;
                    sum += r4.w * x4.w;
                  }
                  // Merge lane pair (l, l^1) within warp.
                  sum += __shfl_xor_sync(0xffffffff, sum, 1);
                  if (half == 0) {
                    staging[t * 64 + pair_id] = bf16(sum);
                  }
                }
                // Sync before reusing s_codes/s_x for the next token.
                NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
              }
              // [Stage-5 Route G step7-cleanup] Removed a redundant
              // NamedBarrier here. The barrier inside the for-t loop
              // (post-R@x, pre next-iter s_x rewrite) already fires
              // on the t=63 iteration and synchronizes all 128
              // producer threads. All 64 staging[] entries are
              // guaranteed visible + globally consistent at loop
              // exit, so the extra sync before staging->sK read
              // was pure overhead (~50-100 cyc * 7 dim_blocks =
              // ~350-700 cyc per TOPK_BLOCK saved).

              // ---- Step 2 + 3: per round, read staging to regs, write sK ----
              CUTE_UNROLL
              for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                int my_token_idx = my_token_idx_base + round * NUM_TOKENS_PER_ROUND;
                const int abs_token = idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx;
                const int dim_in_block = (lane_idx / 8) * 16;

                // Read this thread's 16 dims (= 2 x bf16x8) into registers
                bf16x8 val_lo = *reinterpret_cast<bf16x8*>(&staging[abs_token * 64 + dim_in_block + 0]);
                bf16x8 val_hi = *reinterpret_cast<bf16x8*>(&staging[abs_token * 64 + dim_in_block + 8]);

                // Write registers to GMMA-layout sK
                bf16* sK_nope_base = plan.u.k[buf_idx].data() + abs_token * 8 + ((lane_idx / 8) * 16) * TOPK_BLOCK_SIZE;
                bf16* sK_nope_peer_base = get_peer_addr(sK_nope_base);

                int smem_offset_lo = (dim_base + 0) * TOPK_BLOCK_SIZE;
                int smem_offset_hi = (dim_base + 8) * TOPK_BLOCK_SIZE;
                *(__int128_t*)(sK_nope_base + smem_offset_lo) = *(__int128_t*)&val_lo;
                *(__int128_t*)(sK_nope_base + smem_offset_hi) = *(__int128_t*)&val_hi;
                if constexpr (CLUSTER_SIZE == 2) {
                  st_async_128b(sK_nope_peer_base + smem_offset_lo, val_lo, peer_bar_k_remote_ready);
                  st_async_128b(sK_nope_peer_base + smem_offset_hi, val_hi, peer_bar_k_remote_ready);
                }
              }

              // All threads done reading staging; safe to refill next dim-block
              NamedBarrier::sync(128, NamedBarriers::packed_kv_producer_sync);
            }

            fence_view_async_shared();
          }  // end if constexpr (!wgmma_uniform_supported) legacy path
#else
          // [Route H step3b] producer null-work: nope reconstruction
          // skipped entirely; only rope-copy above + handshake below
          // survive. Keep the async-proxy fence so consumer wgmma
          // reads of sK are ordered after the rope stores.
          fence_view_async_shared();
#endif  // FMLA_PRODUCER_NULL_PROBE (skips all nope reconstruction)
#ifdef FMLA_CLK_PROFILE
          unsigned long long _clk_p3 = clock64();
          if (idx_in_warpgroup == 0) {
            fmla_clk_add(2, _clk_p3 - _clk_p2);
            fmla_clk_add(5, 1ull);  // producer block sample count
          }
#endif
        } else {
          // ---- Original dense FP8 K-load path ----
          CUTE_UNROLL
          for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
            int my_token_idx = my_token_idx_base + round * NUM_TOKENS_PER_ROUND;
            bf16* sK_nope_base = plan.u.k[buf_idx].data() +
                                 (idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx) * 8 +
                                 ((lane_idx / 8) * 16) * TOPK_BLOCK_SIZE;
            bf16* sK_nope_peer_base = get_peer_addr(sK_nope_base);

            // Get prefetched token index
            int token_index;
            if constexpr (!IS_EXTRA_BLOCK) {
              token_index = nxt_token_indexs[round];
              if (block_idx + 1 != (MODEL_TYPE == ModelType::V32 ? args.end_block_idx : args.num_orig_kv_blocks))
                nxt_token_indexs[round] = __ldg(
                    gIndices + (block_idx + 1) * TOPK_BLOCK_SIZE + idx_in_cluster * (TOPK_BLOCK_SIZE / 2) +
                    my_token_idx);
            } else {
              if constexpr (IS_FIRST_EXTRA_BLOCK) {
                token_index = __ldg(
                    gExtraIndices + (block_idx - args.num_orig_kv_blocks) * TOPK_BLOCK_SIZE +
                    idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx);
              } else {
                token_index = nxt_token_indexs[round];
              }
              if (block_idx + 1 != args.end_block_idx)
                nxt_token_indexs[round] = __ldg(
                    gExtraIndices + (block_idx + 1 - args.num_orig_kv_blocks) * TOPK_BLOCK_SIZE +
                    idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx);
            }

            if constexpr (MODEL_TYPE == ModelType::MODEL1) {
              // For MODEL1, we need to check whether the token_index is within topk_length
              if (rel_block_idx * TOPK_BLOCK_SIZE + idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx >=
                  topk_length) {
                token_index =
                    -1;  // To prevent IMA when we have invalid (e.g. INT_MAX) topk indexes outside topk_length
              }
            }

            int block_index =
                token_index == -1
                    ? 0
                    : (int)((uint32_t)token_index /
                            (uint32_t)page_block_size);  // Use uint32_t division and mod to improve performance
            int rel_idx_in_block =
                (uint32_t)token_index %
                (uint32_t)page_block_size;  // NOTE When token_index is -1 (UINT_MAX), UINT_MAX%page_block_size <
                                            // page_block_size, so there will be no illegal-memory-access error

            fp8* gK_base;
            bf16 scales[NUM_SCALES];
            if constexpr (MODEL_TYPE == ModelType::V32) {
              static_assert(NUM_SCALES == 4);
              gK_base = k_ptr + block_index * k_block_stride + rel_idx_in_block * k_row_stride;
              float scales_float[NUM_SCALES];
              *(float4*)(scales_float) = load_128b_from_gmem<float4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(
                  (float*)(gK_base + HEAD_DIM_NOPE));
              CUTE_UNROLL
              for (int i = 0; i < NUM_SCALES; ++i) {
                scales[i] = (bf16)scales_float[i];
              }
            } else {
              static_assert(NUM_SCALES == 8);
              gK_base = k_ptr + block_index * k_block_stride +
                        rel_idx_in_block * (HEAD_DIM_NOPE + HEAD_DIM_ROPE * sizeof(bf16));
              fp8_e8m0* gK_scales_base = (fp8_e8m0*)(k_ptr + block_index * k_block_stride +
                                                     page_block_size * (HEAD_DIM_NOPE + HEAD_DIM_ROPE * sizeof(bf16)) +
                                                     rel_idx_in_block * NUM_SCALES * sizeof(fp8_e8m0));
              fp8_e8m0 scales_e8m0[NUM_SCALES];
              *(int64_t*)scales_e8m0 = __ldg((int64_t*)gK_scales_base);
              CUTE_UNROLL
              for (int i = 0; i < NUM_SCALES; i += 2) {
                *(__nv_bfloat162_raw*)(scales + i) =
                    __nv_cvt_e8m0x2_to_bf162raw(*(__nv_fp8x2_storage_t*)(scales_e8m0 + i));
              }
            }

            // Wait for the nope buffer to be available
            if (round == 0) {
              plan.bar_k_avail[buf_idx].wait((bar_phase_k >> buf_idx & 1) ^ 1);
            }

            if (CLUSTER_SIZE == 2 && round == 0 && idx_in_warpgroup == 0) {
              plan.bar_k_remote_ready[buf_idx].arrive_and_expect_tx(
                  (TOPK_BLOCK_SIZE / 2) * (HEAD_DIM_NOPE + HEAD_DIM_ROPE) * sizeof(bf16));
            }

            // Collectively copy from global memory and dequant
            // For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py

            fp8* gK_nope = gK_base + (lane_idx / 8) * 16;
            if (token_index == -1) {
              CUTE_UNROLL
              for (int i = 0; i < NUM_SCALES; ++i)
                scales[i] = (bf16)0.0f;
            }
            CUTE_UNROLL
            for (int dim_idx = 0; dim_idx < HEAD_DIM_NOPE / 64; dim_idx += 1) {
              fp8x16 cur_fp8x16 = load_128b_from_gmem<fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(
                  gK_nope + dim_idx * 64);  // We use EVICT_LAST here since gK_base may not be aligned to 32B (for V3.2)
                                            // and the performance is the best among all cache hints (for MODEL1)
              bf16 scale = scales[MODEL_TYPE == ModelType::V32 ? dim_idx / 2 : dim_idx];
              auto dequant_and_save_bf16x8 = [&](const fp8x8& data, int offset) {
                int smem_offset = (dim_idx * 64 + offset) * TOPK_BLOCK_SIZE;
                bf16x8 cur_bf16x8 = cvt_fp8x8_bf16x8(data, __bfloat162bfloat162(*(__nv_bfloat16*)(&scale)));
                *(__int128_t*)(sK_nope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
                if constexpr (CLUSTER_SIZE == 2) {
                  st_async_128b(sK_nope_peer_base + smem_offset, cur_bf16x8, peer_bar_k_remote_ready);
                }
              };
              if (token_index == -1) *(uint128_t*)(&cur_fp8x16) = uint128_t();
              dequant_and_save_bf16x8(cur_fp8x16.lo, 0);
              dequant_and_save_bf16x8(cur_fp8x16.hi, 8);
            }

            bf16* gK_rope;
            if constexpr (MODEL_TYPE == ModelType::V32) {
              gK_rope = (bf16*)(gK_base + HEAD_DIM_NOPE + NUM_SCALES * sizeof(float)) + (lane_idx / 8) * 8;
            } else {
              gK_rope = (bf16*)(gK_base + HEAD_DIM_NOPE) + (lane_idx / 8) * 8;
            }
            bf16* sK_rope_base = plan.u.k[buf_idx].data() +
                                 (idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx) * 8 +
                                 ((lane_idx / 8) * 8) * TOPK_BLOCK_SIZE;
            bf16* sK_rope_peer_base = get_peer_addr(sK_rope_base);

            CUTE_UNROLL
            for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE / 32; dim_idx += 1) {
              bf16x8 cur_bf16x8 =
                  load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_rope + dim_idx * 32);
              if constexpr (MODEL_TYPE == ModelType::V32) {
                // NOTE We do not need to mask the RoPE part for V3.2 since it isn't involved in the SV gemm
              } else {
                if (token_index == -1) *(uint128_t*)(&cur_bf16x8) = uint128_t();
              }
              int smem_offset = (HEAD_DIM_NOPE + dim_idx * 32) * TOPK_BLOCK_SIZE;
              *(__int128_t*)(sK_rope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
              if constexpr (CLUSTER_SIZE == 2) {
                st_async_128b(sK_rope_peer_base + smem_offset, cur_bf16x8, peer_bar_k_remote_ready);
              }
            }
          }

          fence_view_async_shared();
        }  // end if (use_packed) / else

        if (idx_in_warpgroup < 32) {
          // We put this after fence_view_async_shared() since this won't be read by async proxy
          auto is_index_valid = [&](int index, int offset_within_thread) -> bool {
            if constexpr (MODEL_TYPE == ModelType::V32) {
              return index != -1;
            } else {
              return index != -1 && rel_block_idx * TOPK_BLOCK_SIZE + lane_idx * 2 + offset_within_thread < topk_length;
            }
          };
          int2 indices = __ldg((int2*)(indices_base + lane_idx * 2));
          *(char2*)(&plan.is_kv_valid[buf_idx][lane_idx * 2]) = {
              is_index_valid(indices.x, 0), is_index_valid(indices.y, 1)};
        }

        // Signal the barrier
        plan.bar_k_local_ready[buf_idx].arrive();
        bar_phase_k ^= 1 << buf_idx;
      };

      if constexpr (MODEL_TYPE == ModelType::V32) {
        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
          process_one_block(block_idx, IsOrigBlock{}, IsNotFirstExtraBlock{});
        }
      } else {
        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < min(args.num_orig_kv_blocks, args.end_block_idx);
             ++block_idx) {
          process_one_block(block_idx, IsOrigBlock{}, IsNotFirstExtraBlock{});
        }

        if (args.num_orig_kv_blocks < args.end_block_idx) {
          process_one_block(max(args.start_block_idx, args.num_orig_kv_blocks), IsExtraBlock{}, IsFirstExtraBlock{});
        }
        CUTE_NO_UNROLL
        for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks) + 1; block_idx < args.end_block_idx;
             ++block_idx) {
          process_one_block(block_idx, IsExtraBlock{}, IsNotFirstExtraBlock{});
        }
      }

      sync_all_threads_in_cluster();
    }
  }
#else
  if (cute::thread0()) {
    CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
  }
#endif
}

template <typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE) flash_fwd_splitkv_mla_fp8_sparse_kernel(
    __grid_constant__ const SparseAttnDecodeParams params, __grid_constant__ const TMAParams tma_params) {
  Kernel::template devfunc<TMAParams>(params, tma_params);
}

template <ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run_impl(const SparseAttnDecodeParams& params) {
  KU_ASSERT(params.h_kv == 1);
  KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
  KU_ASSERT(params.d_qk == HEAD_DIM_K);
  KU_ASSERT(params.d_v == HEAD_DIM_V);
  KU_ASSERT(params.h_q % BLOCK_M == 0);
  if constexpr (MODEL_TYPE == ModelType::MODEL1) {
    constexpr int BYTES_PER_TOKEN = HEAD_DIM_NOPE + 2 * HEAD_DIM_ROPE + 8;
    // [M3.c.4 Stage-5] When use_packed=true (packed_kcache_ptr non-null),
    // kv tensor carries packed-FP8 bytes_per_token (e.g. 268 for b=2.5)
    // and the kernel ignores stride_kv_row in favor of packed_row_bytes.
    // Native FP8 path (packed_kcache_ptr == nullptr) still enforces 584.
    if (params.packed_kcache_ptr == nullptr) {
      KU_ASSERT(
          params.stride_kv_row == BYTES_PER_TOKEN,
          "Each page block in KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each
                                                                                                                 // block
                                                                                                                 // must
                                                                                                                 // be
                                                                                                                 // contiguous
      if (params.extra_kv != nullptr && params.extra_packed_kcache_ptr == nullptr) {
        KU_ASSERT(
            params.stride_extra_kv_row == BYTES_PER_TOKEN,
            "Each page block in extra KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each block must be contiguous
      }
    }
  } else {
    KU_ASSERT(params.extra_kv == nullptr, "V3.2 does not support extra KV cache");
    KU_ASSERT(params.topk_length == nullptr, "V3.2 does not support dynamic topk length");
    KU_ASSERT(params.stride_kv_row == 656);  // number of bytes per token (512 fp8 + 4 float32 + 64 bfloat16)
  }

  auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q, params.b);
  auto tma_Q = cute::make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(
          make_gmem_ptr((bf16*)params.q),
          make_layout(shape_Q, make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b))),
      SmemLayoutQ{});

  CUtensorMap tensor_map_o;
  {
    // Here we manually construct TMA descriptor to store O, in order to leverage 5D TMA
    uint64_t size[5] = {
        OBUF_SW, (unsigned long)params.h_q, HEAD_DIM_V / OBUF_SW, (unsigned long)params.s_q, (unsigned long)params.b};
    uint64_t stride[4] = {
        params.stride_o_h_q * sizeof(bf16),
        OBUF_SW * sizeof(bf16),
        params.stride_o_s_q * sizeof(bf16),
        params.stride_o_b * sizeof(bf16)};
    uint32_t box_size[5] = {OBUF_SW, BLOCK_M, HEAD_DIM_V / OBUF_SW, 1, 1};
    uint32_t elem_stride[5] = {1, 1, 1, 1, 1};
    CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tensor_map_o,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        5,
        params.out,
        size,
        stride,
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        OBUF_SW == 64   ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
        : OBUF_SW == 32 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
        : OBUF_SW == 16 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B
                        : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    KU_ASSERT(res == CUresult::CUDA_SUCCESS);
  }

  TmaParams<decltype(shape_Q), decltype(tma_Q)> tma_params = {shape_Q, tma_Q, tensor_map_o};
  auto mla_kernel =
      &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<MODEL_TYPE, NUM_HEADS>, decltype(tma_params)>;

  constexpr size_t smem_size = sizeof(SharedMemoryPlan);
  // [c4c128-packed debug] One-shot host-side launch diagnostics for H20
  // cudaFuncSetAttribute invalid-argument triage. This prints the actual
  // dynamic smem request and device opt-in cap before the failing call.
  static bool smem_diag_printed = false;
  const bool smem_diag_enabled = std::getenv("FMLA_SMEM_DIAG") != nullptr;
  if (smem_diag_enabled && !smem_diag_printed) {
    smem_diag_printed = true;
    int dev = -1;
    int optin_smem = -1;
    int smem_per_sm = -1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    fprintf(
        stderr,
        "[FMLA_SMEM_DIAG] model=%d heads=%d cluster=%d smem_size=%zu "
        "optin=%d smem_per_sm=%d dev=%d\n",
        static_cast<int>(MODEL_TYPE),
        NUM_HEADS,
        CLUSTER_SIZE,
        smem_size,
        optin_smem,
        smem_per_sm,
        dev);
  }
  KU_CUDA_CHECK(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  // NOTE Don't use PDL because of potential compiler bugs!
  // cudaLaunchAttribute mla_kernel_attributes[1];
  // mla_kernel_attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  // mla_kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
  // cudaLaunchConfig_t mla_kernel_config = {
  //     dim3(num_m_block, params.h_k, params.num_sm_parts),
  //     dim3(NUM_THREADS, 1, 1),
  //     smem_size,
  //     stream,
  //     mla_kernel_attributes,
  //     1
  // };
  // cudaLaunchKernelEx(&mla_kernel_config, mla_kernel, params, tma_params);
  cutlass::ClusterLaunchParams launch_params = {
      dim3(NUM_M_BLOCKS, params.s_q, params.num_sm_parts),
      dim3(NUM_THREADS, 1, 1),
      dim3(CLUSTER_SIZE, 1, 1),
      smem_size,
      params.stream};
  cutlass::launch_kernel_on_cluster(launch_params, (void*)mla_kernel, params, tma_params);
  KU_CHECK_KERNEL_LAUNCH();

#ifdef FMLA_CLK_PROFILE
  // [Route H step3k] throttled readback of the segment cycle counters.
  //   Print mean cycles/block/segment every N launches, then zero the
  //   accumulators. N chosen so decode-loop noise averages out while
  //   staying human-readable in the server log.
  {
    static thread_local unsigned long long _fmla_launch_ctr = 0;
    constexpr unsigned long long PRINT_EVERY = 10ull;
    if ((++_fmla_launch_ctr % PRINT_EVERY) == 0) {
      unsigned long long h[16] = {0};
      cudaMemcpyFromSymbol(h, g_fmla_clk, sizeof(h));
      unsigned long long np = h[5] ? h[5] : 1ull;  // producer samples
      unsigned long long nc = h[6] ? h[6] : 1ull;  // consumer samples
      double p0 = (double)h[0] / np, p1 = (double)h[1] / np, p2 = (double)h[2] / np;
      double c3 = (double)h[3] / nc, c4 = (double)h[4] / nc;
      double s7 = (double)h[7] / np, s8 = (double)h[8] / np;
      double s9 = (double)h[9] / np, s10 = (double)h[10] / np;
      double s11 = (double)h[11] / np;
      fprintf(
          stderr,
          "[FMLA_CLK step3k] launch#%llu np=%llu nc=%llu | "
          "PROD bar_avail=%.0f rope=%.0f nope_rebuild=%.0f | "
          "CONS bar_ready=%.0f QK_softmax=%.0f (cyc/block)\n"
          "                  [step3l nope sub] fill_sX=%.0f fill_sR=%.0f "
          "wgmma+bar=%.0f scatter=%.0f (cyc/block, sum over 49 iters)\n"
          "                  [step3t sX split] pure_unpack=%.0f sX_barrier_wait=%.0f (cyc/block)\n",
          _fmla_launch_ctr,
          np,
          nc,
          p0,
          p1,
          p2,
          c3,
          c4,
          s7,
          s8,
          s9,
          s10,
          s7,
          s11);
      fflush(stderr);
      unsigned long long z[16] = {0};
      cudaMemcpyToSymbol(g_fmla_clk, z, sizeof(z));
    }
  }
#endif
}

template <ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run_int4(const SparseAttnDecodeParams& params) {
  run_impl(params);
}

template <ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_int4_sparse_kernel(const SparseAttnDecodeParams& params) {
  KernelTemplate<MODEL_TYPE, NUM_HEADS>::run_int4(params);
}

}  // namespace sm90::decode::sparse_fp8
