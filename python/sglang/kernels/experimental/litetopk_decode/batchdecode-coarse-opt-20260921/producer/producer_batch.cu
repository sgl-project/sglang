// Batched derivative of decode_topk_final/producer/producer.cu.
// The frozen B=1 producer is NOT modified; this is a second translation unit with its
// own torch namespace so both can be loaded at once.
//
// What changes versus the B=1 artifact, and nothing else:
//   * num_q_tokens_total = B in the scheduler construction (was the literal 1), which
//     re-enables the DeepGEMM prefix-sum/binary-search work balancer across B rows.
//   * The emitter honours the row index the frozen core already hands it
//     (native_tma_core.cuh: emit(scheduler.get_logits_row(q_block_idx, i), ...)) instead
//     of dropping every row but 0, and addresses dense[row] and lengths[row].
//   * One shared 2048-bin histogram is flushed into global hist[row] at each row
//     transition. Transitions are CTA-uniform (kNextN==1 => one q token per request =>
//     num_q_blocks==1 => the row is constant for a whole next_q_block iteration) and,
//     because the metadata kernel snaps SM starts to request boundaries, each of the B-1
//     boundaries is interior to at most one CTA: at most B-1 extra flushes in the grid.
//   * The B=1 closed-form schedule guard becomes a device-side transcription of the
//     general rule in deep_gemm/scheduler/sm100_paged_mqa_logits.cuh:127-177.
// The shared native core, scheduler, numeric reduction and shared-memory budget are
// unchanged. See ../../decode_topk_final/README.md and ../hist_fused/LICENSE.*.
// DeepGEMM-derived host/scheduler code: Copyright (c) 2025 DeepSeek, MIT.
#include "producer_batch.h"
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include "temporal_decode/fused/native_tma_core.cuh"
#include <climits>
#include <type_traits>
#include <cmath>

#ifndef LITETOPK_COARSE_BINS
#error "LITETOPK_COARSE_BINS must be 512 or 1024"
#endif
#ifndef LITETOPK_PRODUCER_NAMESPACE
#error "LITETOPK_PRODUCER_NAMESPACE must name the isolated Torch namespace"
#endif

namespace LITETOPK_PRODUCER_NAMESPACE {
namespace {
constexpr int Ctas = 148, Threads = 384, MathThreads = 256;
constexpr int Bins = LITETOPK_COARSE_BINS;
static_assert(Bins == 512 || Bins == 1024);
constexpr int CoarseShift = Bins == 1024 ? 6 : 7;
// Keep the original shared-memory carve-out so this experiment changes only
// histogram work and global traffic, not scorer occupancy or launch resources.
constexpr int AllocatedBins = 2048;
// Mode 24 only: a 2049th bin that absorbs out-of-range columns. Never flushed, never read.
constexpr int ScratchBin = Bins;
constexpr int MaxRows = 32;   // GVR2 split family is B <= 32; see selector/build.py
constexpr unsigned Full = 0xffffffffu;
// Hardware named barrier ids used by this kernel, after CUTLASS's +8 reserved offset:
//   logical 0 -> hw 8   frozen core, NamedBarrier(kNumMathThreads, 0) at native_tma_core.cuh:507
//   logical 1 -> hw 9   final publish barrier (all 384 threads), as in the B=1 producer
//   logical 2 -> hw 10  row-transition flush (math threads only) -- verified free by
//                       cuobjdump -sass of the B=1 build: only 0x0/0x8/0x9 are emitted.
constexpr int RowFlushBarrier = 2;
using Tensor = at::Tensor;
template <int QStages, int KVStages>
using StageStorage = deep_gemm::layout::MQALogitsSharedStorage<
    32, 128, false, 4, 256, QStages, KVStages, 3, cutlass::float_e4m3_t, float>;
static_assert(sizeof(StageStorage<1, 6>) == 220160);
// The installed upstream scorer is NOT our instantiation: the cached DeepGEMM cubin is
// sm100_paged_mqa_logits<1,32,128,...,3,5,...> i.e. Q3/KV5, and it is launched with PDL,
// while we compile Q1/KV6 and launch bare. That makes `bprod_nohist - scorer_only` a
// comparison between two different kernels, not a measurement of what our emitter adds --
// it reached -16 us (NEGATIVE) at 1M B=16. Q3/KV5 happens to need exactly the same shared
// bytes, so a stage=3 build is a drop-in control that isolates the emitter at last.
// stage 3 is a MEASUREMENT configuration only; production stays Q1/KV6 (hist_stage/RESULTS.md
// screened Q3/KV5 at 46.851/46.575 against Q1/KV6 at 46.476/46.041 and Q1/KV6 was chosen).
static_assert(sizeof(StageStorage<3, 5>) == 220160);

// Native scheduler with an active physical-page guard, not a layout change.
// Identical to the B=1 producer except that num_q_tokens_total is the batch size.
template <int Page, bool Fold>
struct SafeScheduler : deep_gemm::sched::SM100PagedMQALogitsScheduler<
    1, true, true, 32, 256, Page, 16> {
  using Base = deep_gemm::sched::SM100PagedMQALogitsScheduler<
      1, true, true, 32, 256, Page, 16>;
  int pages;
  int* diag;
  __device__ SafeScheduler(unsigned sm, const unsigned* lengths,
      const unsigned* schedule, const unsigned* indices, const unsigned* table,
      unsigned stride, int num_pages, int* diagnostics, unsigned rows)
      : Base(sm, lengths, schedule, indices, table, stride, rows),
        pages(num_pages), diag(diagnostics) {}
  __device__ __forceinline__ unsigned get_kv_page_coord_by_page_offset(
      const unsigned& offset) const {
    if (offset >= this->cur_request_num_kv_pages) return 0;
    const unsigned page = Base::get_kv_page_coord_by_page_offset(offset);
    if (page >= static_cast<unsigned>(pages)) {
      atomicOr(diag, 16);
      return 0;
    }
    return page;
  }
  // W6. dispatch_num_block_tokens<BLOCK_Q=4> (native_tma_core.cuh:45-53) recurses
  // 4 -> 3 -> 2 -> 1 and instantiates the emit body once per candidate, so the shipped
  // kernel carries 4+3+2+1 = 10 copies of it while kNextN==1 executes only the Int<1>
  // one. batched_boundaries_warp rejects adjacent-equal indices, so every logical
  // request is exactly one q token => num_q_tokens==1 => num_q_blocks==1 => the block
  // span is 1 token, i.e. the base accessor can only ever return 1 here. Returning the
  // literal lets the three comparisons fold and the other nine bodies never exist.
  // The precondition is enforced fail-closed before the core is entered.
  __device__ __forceinline__ unsigned get_num_block_tokens(const unsigned& q_block_idx) const {
    if constexpr (Fold) {
      // Fail closed rather than silently dropping scores: this fires once per q block,
      // not per score, so it costs nothing in the hot loop. qualify_producer additionally
      // asserts histogram_total == n and bit-exact dense, which would also catch it.
      if (Base::get_num_block_tokens(q_block_idx) != 1u) atomicOr(diag, 16);
      return 1u;
    } else {
      return Base::get_num_block_tokens(q_block_idx);
    }
  }
};

// Two modes only. 8 is production; 0 is the same kernel with the coarse histogram
// compiled out, kept because it is the only way to price the histogram at all.
// The measured attribution that selected this configuration (runs-v2-attrib-r1, disjoint,
// two layers, --cycles 6, summed over 8 cells): W1 -68.61 us (91%, 8/8 cells negative),
// W6 -8.07 us (11%, 8/8 negative), W2 +1.09 us (-1%, 6/8 positive -- kept only because
// dropping it would invalidate the SASS these numbers were taken against, and its effect
// is below noise). The intermediate modes 2/3/4/5/7 that produced that attribution are
// deleted; the run JSONs hold the numbers.
// TEMPORARY diagnostic modes; they produce a WRONG histogram and must never enter a chain.
//   10  key chain kept live, per-score ATOMS deleted, zeroing+flush KEPT.
//       CAUTION: (8 - 10) is CONFOUNDED. With no per-score atomic every bin stays zero, so
//       flush_bins' `count != 0` skips every global atomicAdd -- the subtraction therefore
//       carries the flush's global traffic as well as the per-score shared atomic, and that
//       traffic grows with the number of NONZERO bins per CTA, which is exactly what differs
//       between 1M B=32 (226,714 scores/CTA) and the other cells (56,678).
//   12  per-score ATOMS kept, zeroing and flush removed  -> (8 - 12) prices the fixed term
//   15  like 12 but the per-score ATOMS also deleted     -> (12 - 15) is the CLEAN per-score
//                                                           atomic: neither side flushes
//   16  like 12 but the ATOMS replaced by a plain shared STORE to the same address
//       -> (12 - 16) separates atomicity from the shared-memory access itself
// The non-histogram half of the tax (Mode 0 - upstream scorer) is about TWICE the whole
// histogram at 1M B=32 and has never been decomposed. These three price its parts, each by
// changing exactly one thing while holding the instruction count fixed where it matters:
//   17  dense[col] -> dense[col & 1023]: SAME STG.E, same address arithmetic, but a 4 KiB
//       per-row working set instead of 134 MB per layer, so (0 - 17) is the DRAM/L2 cost of
//       the dense store alone. Dense output is deliberately wrong.
//   18  the per-score NaN guard deleted. (0 - 18) prices FSETP.NAN + the speculative HFMA2
//       + the @P BRA + the IMAD.MOV restore IN THE NO-HISTOGRAM CONFIGURATION ONLY.
//       An earlier revision of this comment called it "the calibration constant for every
//       remove-N-instructions estimate" -- that is WRONG and is retracted: the same deletion
//       measured -10.59 us here (Mode 0 - Mode 18) and *positive* in the histogram
//       configuration (Mode 21 - Mode 8), so the effect is not additive and does not
//       transfer. Use the 2x2 {0,18} x {8,21} factorial and its interaction term instead.
//   19  the row-transition test hoisted out of the per-score path (fail-closed if a second
//       row is ever seen) -> (0 - 19) prices the ISETP.NE + not-taken BRA + the phi copy.
//   20  PRODUCTION CANDIDATE = Mode 8 with the NaN accumulator folded into the cold fault
//       path. The FSETP.NAN predicate is needed anyway to gate the histogram atomic, but
//       `if (nan) nan_acc = 1u` costs a speculative set plus a restore on the taken path
//       (SASS: HFMA2 at /*49e0*/ and IMAD.MOV at /*4a20*/), and finish() then spends an
//       __any_sync on it. Raising diag directly from the NaN branch keeps the diagnostic
//       EXACTLY as strong -- same bit, same meaning -- while moving all of it onto a path
//       that real indexer scores never take. Measured cost of the NaN guard as a whole:
//       10.59 us at 1M B=32 shared (runs-emit-decomp-r2, Mode 0 - Mode 18).
//   21  PRODUCTION CANDIDATE = Mode 8 with ALL NaN logic removed, on the operator owner's
//       decision. Rationale as given: NaN is outside the operator contract (the frozen
//       selector says so itself, topk_filtered_boundary.cuh:26-27), the real captures span
//       bins [400,1644] with no NaN and no inf, and neither competing kernel pays for it --
//       SGLang's deepseek_v4_topk.cu contains no NaN handling at all, and GVR2's "NaN-safe
//       degeneracy guard" protects its SAMPLED bracket [GM,GX], a failure mode an exact
//       histogram certificate does not have.
//       WHAT IS GIVEN UP, recorded so it is not rediscovered the hard way: (a) a positive
//       NaN maps to bin 15, ahead of +inf, so it would take a top-k slot from a real element
//       AND the dense rescan would not find it (NaN >= thr is false), leaving the quota
//       short -- a silent wrong answer, not a degraded one; (b) diag bit 4 was the only
//       fail-loud signal that upstream handed us garbage (an uninitialised FP8 cache reads
//       0x7F/0xFF, which are e4m3 NaNs). Mode 8 and Mode 20 remain in the TU as the guarded
//       variants if that trade is ever revisited.
//   22, 23  FAILED EXPERIMENTS. Kept only as measurement controls; do not read anything
//       about scheduling into them.
//       INTENT: removing the NaN branch measured SLOWER in 3 of 6 cells across two runs, even
//       though it deletes 24 instructions and the branch body never runs on real scores. The
//       static SASS moves what is between the dense store and the shared atomic --
//           Mode 8 : STG.E ... 9 instructions (NaN control + key chain) ... ATOMS.POPC.INC
//           Mode 21: key chain ... STG.E, ATOMS.POPC.INC  (adjacent)
//       -- so the hypothesis was that the six ALU instructions are free SPACING in Mode 8 and
//       that Mode 22 could buy it back deliberately.
//       WHAT ACTUALLY HAPPENED: the experiment was never applied. `asm volatile("" : "+r"(x)
//       ::: "memory")` constrains PTX generation but emits no machine instruction and does
//       not bind ptxas's scheduler; Mode 22's SASS came out instruction-for-instruction
//       IDENTICAL to Mode 21's (1472, same gap of 1). Mode 23's eight `add.u32 k,k,0` reached
//       PTX and were folded away before SASS. So Mode 22 == Mode 21 says nothing about
//       spacing either way -- it is a pure same-machine-code control, and as such it does
//       give the harness's reproducibility on these cells (+-1.2 us).
//       STATUS OF THE HYPOTHESIS: UNTESTED. Instruction ADDRESS distance is not issue-cycle
//       distance; scheduling control bits, other warps' issue, and queue occupancy all sit in
//       between, and Mode 20 -- whose gap is 17, wider than Mode 8's 9 -- was also slightly
//       slower, which the simple story does not explain. A real test must keep the same
//       effective operations, move only the order of the key/address computation relative to
//       STG and ATOMS, and be accepted on its SASS (registers, spills, control bits) BEFORE
//       it is timed.
//   24  EMIT EXPERIMENT: branchless bounds handling. The guarded body of Mode 8 is a full
//       reconvergence region -- ISETP.GE, BSSY.RECONVERGENT, @P BRA, 16 instructions, BSYNC --
//       entered once per score. Both of its side effects can be made harmless out of range
//       instead of skipped:
//         * the dense store: the scheduler covers ceil(n/256) splits, so the largest column a
//           CTA can emit is ceil(n/256)*256-1, which is <= the row stride (1,048,575 against a
//           1,048,576 stride at n=1,048,321, and exactly n-1 at 262,144). Those columns live in
//           the row's own padding and the selector only ever reads [0, n), so writing them is
//           invisible. validate_buffers already requires stride(0) >= width.
//         * the histogram: out-of-range keys are redirected to a 2049th SCRATCH bin. flush_bins
//           and the entry zeroing both loop `bin < Bins`, so the scratch bin is never published,
//           never flushed and never read; qualify_producer's histogram_total == n check sums
//           bins 0..2047 and therefore still holds.
//       Costs 4 more shared bytes (228,352 -> 228,356 against a 232,448 limit; the sm_100 carve-
//       out bucket is 228 KiB either way, so occupancy is untouched). This is a STRUCTURAL
//       change, not an instruction-count one: the branch and its reconvergence disappear and a
//       SEL takes their place. Zero-length rows (zero_gap / vllm_tail) simply send every column
//       to the scratch bin, which is the correct behaviour.
template <int Mode> inline constexpr bool kDiag    = (Mode == 10 || Mode == 12 || Mode == 15 ||
                                                      Mode == 16 || Mode == 17 || Mode == 18 ||
                                                      Mode == 19);
template <int Mode> inline constexpr bool kProd    = (Mode == 8 || Mode == 20 || Mode == 21 ||
                                                      Mode == 22 || Mode == 23 || Mode == 24);
template <int Mode> inline constexpr bool kDenseLocal = (Mode == 17);
template <int Mode> inline constexpr bool kNoNan      = (Mode == 18 || Mode == 21 || Mode == 22);
template <int Mode> inline constexpr bool kOneRow     = (Mode == 19);
template <int Mode> inline constexpr bool kColdNan    = (Mode == 20);
template <int Mode> inline constexpr bool kSpaceStore = (Mode == 22 || Mode == 23);
template <int Mode> inline constexpr bool kBranchless = (Mode == 24);
// Extra dependent no-op ALU steps inserted between the dense store and the shared atomic.
// Each is an empty asm that ptxas cannot fold away but that emits no instruction of its own;
// they exist to pin the ORDER. Mode 23 additionally widens the real gap.
template <int Mode> inline constexpr int  kExtraSpace = (Mode == 23) ? 8 : 0;
template <int Mode> inline constexpr bool kHasBins = (kProd<Mode> || (kDiag<Mode> && Mode < 17));
template <int Mode> inline constexpr bool kFlushes = (kProd<Mode> || Mode == 10);
template <int Mode> inline constexpr bool kAtomic  = (kProd<Mode> || Mode == 12);
template <int Mode> inline constexpr bool kPlainStore = (Mode == 16);
// Both modes fold: the precondition is the same, and a control that does not share the
// production kernel's codegen shape is not a control.
template <int Mode> inline constexpr bool kFold    = (Mode != 99);
template <int Mode> inline constexpr bool kValidMode = (Mode == 0 || kProd<Mode> || kDiag<Mode>);

// W2. Coarse-key projection: fp32 -> fp16 -> descending bin index.
// Non-negative values are inverted into [0, 0x7FFF] so the largest magnitude lands
// in bin 0; negative values keep their natural bit order, which places every one of
// them above every non-negative key. The shift then folds the low mantissa bits away.
// Verified exhaustively over all 65,536 fp16 bit patterns for shifts 5/6/7:
// +-0 -> 1023/1024, +-inf -> 31/2016 (at shift 5), and every NaN encoding.
__device__ __forceinline__ unsigned coarse_key_u32(float x) {
  const unsigned bits = __half_as_ushort(__float2half_rn(x));
  const unsigned rank = (bits & 0x8000u) ? bits : (~bits & 0x7FFFu);
  return rank >> CoarseShift;
}

// W1. The V1 flush is a __noinline__ MEMBER, so `this` is address-taken and ptxas gives
// the whole emitter an 80-byte ABI frame; the live loop then executes STL.U8 on EVERY
// in-bounds score (it spills `nan_seen`), riding the same LSU as the STG and the ATOMS,
// and the callee reaches shared memory through generic LD/ST instead of LDS/STS.
// Taking every operand by value keeps BatchEmitter in registers. Still out of line for
// the original reason: inlining puts a 2048-iteration loop at each emit site.
template <int Mode>
__device__ __noinline__ void flush_bins(int* local_histogram, int* global_histogram, int cur_row) {
  static_assert(kFlushes<Mode>, "call sites are guarded; there is no histogram-free flush");
  cutlass::arch::NamedBarrier(MathThreads, RowFlushBarrier).sync();
  if (cur_row >= 0) {
    int* dst = global_histogram + static_cast<size_t>(cur_row) * Bins;
    for (unsigned bin = threadIdx.x; bin < Bins; bin += MathThreads) {
      const int count = local_histogram[bin];
      if (count != 0) {
        atomicAdd(dst + bin, count);
        local_histogram[bin] = 0;
      }
    }
  }
  cutlass::arch::NamedBarrier(MathThreads, RowFlushBarrier).sync();
}

// The batched emitter. Nothing takes its address, so it stays in registers: the flush is
// a free function taking every operand by value.
template <int Mode>
struct BatchEmitter {
  float* dense;
  const int* lengths;
  int* local_histogram;
  int* global_histogram;
  int* diag;
  unsigned dense_stride;
  float* row_dense = nullptr;
  int cur_row = -1;
  int cur_len = 0;
  unsigned key_sink = 0u;   // Mode 10 only: keeps the key chain live with no atomic.
  unsigned nan_acc = 0u;    // W1: a word, not a byte-packed bool. As a bool inside an
                            // address-taken object this cost SEL+LOP3+SEL+PRMT+STL.U8 per
                            // score; that store was 91% of the whole measured win.

  __device__ __forceinline__ void operator()(unsigned row, unsigned col, float score) {
    // CTA-uniform across the 256 math threads: `row` depends only on the scheduler task.
    if constexpr (kOneRow<Mode>) {
      // Mode 19: bind once, then never test again. Fail closed if the assumption breaks.
      if (cur_row < 0) {
        cur_row = static_cast<int>(row);
        row_dense = dense + static_cast<size_t>(row) * dense_stride;
        cur_len = lengths[row];
      } else if (static_cast<int>(row) != cur_row) {
        atomicOr(diag, 16);
      }
    } else if (static_cast<int>(row) != cur_row) {
      // Guarded at the call site, not inside the callee: an empty __noinline__ callee is
      // still an ABI call, and its mere presence gives the measurement arm a stack frame.
      if constexpr (kFlushes<Mode>) flush_bins<Mode>(local_histogram, global_histogram, cur_row);
      cur_row = static_cast<int>(row);
      row_dense = dense + static_cast<size_t>(row) * dense_stride;
      cur_len = lengths[row];
    }
    if constexpr (kBranchless<Mode>) {
      // No guard: the store lands in the row's padding and the count lands in the scratch bin.
      row_dense[col] = score;
      const bool nan = isnan(score);
      if (nan) atomicOr(diag, 4);
      else {
        const unsigned key = (col < static_cast<unsigned>(cur_len))
                                 ? coarse_key_u32(score) : static_cast<unsigned>(ScratchBin);
        atomicAdd(local_histogram + key, 1);
      }
      return;
    }
    if (col < static_cast<unsigned>(cur_len)) {
      // Mode 17: identical STG.E and address arithmetic, 4 KiB working set instead of 134 MB.
      row_dense[kDenseLocal<Mode> ? (col & 1023u) : col] = score;
      const bool nan = kNoNan<Mode> ? false : isnan(score);
      if constexpr (!kNoNan<Mode> && !kColdNan<Mode>) { if (nan) nan_acc = 1u; }
      // Mode 20: the diagnostic lives on the branch a NaN would already take, so the
      // non-NaN path (every real score) carries no bookkeeping at all.
      if constexpr (kColdNan<Mode>) { if (nan) atomicOr(diag, 4); }
      if constexpr (Mode == 10 || Mode == 15) {
        if (!nan) key_sink ^= coarse_key_u32(score);          // same chain, no ATOMS
      } else if constexpr (kPlainStore<Mode>) {
        if (!nan) local_histogram[coarse_key_u32(score)] = 1; // same address, no RMW
      } else if constexpr (kSpaceStore<Mode>) {
        // A bare memory fence is NOT enough: the key chain's input (score) is available long
        // before the store, so ptxas hoists the whole chain above it and the fence anchors
        // nothing (measured: Mode 22 with only a fence produced SASS byte-identical to the
        // unfenced Mode 21). Routing the DATA through the opaque asm is what pins the order --
        // the chain now consumes a value that does not exist until after the store.
        unsigned bits = __float_as_uint(score);
        asm volatile("" : "+r"(bits) :: "memory");
        unsigned key = coarse_key_u32(__uint_as_float(bits));
        // Mode 23 widens the gap with real dependent IADDs (an empty asm emits nothing).
        #pragma unroll
        for (int sp = 0; sp < kExtraSpace<Mode>; ++sp)
          asm volatile("add.u32 %0, %0, 0;" : "+r"(key));
        if (!nan) atomicAdd(local_histogram + key, 1);
      } else if constexpr (kAtomic<Mode>) {
        if (!nan) atomicAdd(local_histogram + coarse_key_u32(score), 1);
      }
    }
  }

  __device__ __forceinline__ void finish() {
    if constexpr (Mode == 10 || Mode == 15) {                  // always false: keeps the chain
      if (key_sink == 0xdeadbeefu) atomicOr(diag, 32);
    }
    if constexpr (!kNoNan<Mode> && !kColdNan<Mode>) {
      const bool any_nan = __any_sync(Full, nan_acc != 0u);
      if ((threadIdx.x & 31) == 0 && any_nan) atomicOr(diag, 4);
    }
  }
};

// Device transcription of the general metadata rule, specialised to what we compile
// (kNextN=1, varlen, SPLIT_KV=256): logical request r is q token r, its work is
// ceil(len_r/256), SM start w = sm*q + min(sm, rem), boundary = first r with
// inclusive prefix > w, else the one-past-the-end sentinel (rows, 0).
// Mirrors sm100_paged_mqa_logits.cuh:128-177 including zero-length rows, which never
// advance the prefix and so can never be an SM start.
// One warp derives everything, once per CTA, and publishes 6 words. Doing it
// redundantly in all 384 threads costs 4*B dependent global loads per thread at kernel
// entry, before the first TMA is issued -- measured at up to +18.9 us at B=32/1M with
// the histogram compiled out, i.e. pure loss against the identical upstream scorer.
// MaxRows <= 32 means one lane per row: the O(B) loop collapses to warp primitives.
// Publishes [start.x, start.y, end.x, end.y, bad, total].
__device__ __forceinline__ void batched_boundaries_warp(
    const int* lengths, const int* indices, int rows, int width, unsigned sm, int* out) {
  const unsigned lane = threadIdx.x & 31u;
  const bool live = lane < static_cast<unsigned>(rows);
  const int n = live ? lengths[lane] : 0;
  const int idx = live ? indices[lane] : -1;
  // What the metadata kernel actually needs is that every q token is its own logical
  // request, i.e. that no two ADJACENT indices are equal (a logical request is a maximal
  // run of equal indices, sm100_paged_mqa_logits.cuh:105-113). It does NOT need the labels
  // to be 0..B-1: vLLM pads a short decode batch with `num_decodes + arange(pad)`
  // (mla/indexer.py:833-835), so a tail step with one live request out of two scheduled
  // legitimately carries indices [0, 2]. Requiring idx == lane rejected every such step.
  const int prev = __shfl_up_sync(Full, idx, 1);
  const bool lane_bad = live && (n < 0 || n > width ||
                                 (lane > 0u && idx == prev));
  const unsigned work = live ? (static_cast<unsigned>(n) + 255u) / 256u : 0u;
  unsigned incl = work;                       // inclusive scan == prefix_work[] of the
  #pragma unroll                              // metadata kernel, one lane per request
  for (int off = 1; off < 32; off <<= 1) {
    const unsigned v = __shfl_up_sync(Full, incl, off);
    if (lane >= static_cast<unsigned>(off)) incl += v;
  }
  const unsigned excl = incl - work;
  const unsigned total = __shfl_sync(Full, incl, 31);
  const unsigned bad = __any_sync(Full, lane_bad) ? 1u : 0u;
  const unsigned quotient = total / Ctas, remainder = total % Ctas;
  #pragma unroll
  for (int which = 0; which < 2; ++which) {
    const unsigned s = sm + static_cast<unsigned>(which);
    const unsigned w = s * quotient + min(s, remainder);
    // First request whose inclusive prefix exceeds w; == the metadata binary search.
    const unsigned mask = __ballot_sync(Full, live && incl > w);
    unsigned bx = static_cast<unsigned>(rows), by = 0u;   // one-past-the-end sentinel
    if (mask != 0u) {
      const int r = __ffs(static_cast<int>(mask)) - 1;
      bx = static_cast<unsigned>(r);
      by = w - __shfl_sync(Full, excl, r);
    }
    if (lane == 0) {
      out[which * 2] = static_cast<int>(bx);
      out[which * 2 + 1] = static_cast<int>(by);
    }
  }
  if (lane == 0) {
    out[4] = static_cast<int>(bad);
    out[5] = static_cast<int>(total);
  }
}

template <int Page, int QStages, int KVStages, int Mode>
__global__ __launch_bounds__(Threads, 1) void native_tma_histogram_batch(
    const int* lengths, const int* table, int table_stride, int num_pages,
    const int* schedule, const int* indices, float* dense, int dense_stride,
    int* histogram, int* diag, int width, int rows,
    const __grid_constant__ cute::TmaDescriptor map_q,
    const __grid_constant__ cute::TmaDescriptor map_sf_q,
    const __grid_constant__ cute::TmaDescriptor map_kv,
    const __grid_constant__ cute::TmaDescriptor map_sf_kv,
    const __grid_constant__ cute::TmaDescriptor map_weights) {
  using Storage = StageStorage<QStages, KVStages>;
  static_assert((QStages == 1 && KVStages == 6) || (QStages == 3 && KVStages == 5));
  static_assert(kValidMode<Mode>);
  // Frozen per-CTA diagnostic ownership and guarded scheduler; no inter-CTA counter.
  diag += blockIdx.x;
  if (threadIdx.x == 0) diag[0] = 0;
  // The first six words of the (not yet zeroed) local histogram are the publication
  // slot for the warp-derived boundaries; the zero loop below runs after everyone
  // has read them.
  extern __shared__ __align__(Storage::kSwizzleAlignment) uint8_t hist_smem[];
  int* local_histogram = reinterpret_cast<int*>(hist_smem + sizeof(Storage));
  if (rows < 1 || rows > MaxRows) {
    if (threadIdx.x == 0) diag[0] = 16;
    return;
  }
  if (threadIdx.x < 32)
    batched_boundaries_warp(lengths, indices, rows, width, blockIdx.x, local_histogram);
  __syncthreads();
  const uint2 expected_start = make_uint2(static_cast<unsigned>(local_histogram[0]),
                                          static_cast<unsigned>(local_histogram[1]));
  const uint2 expected_end = make_uint2(static_cast<unsigned>(local_histogram[2]),
                                        static_cast<unsigned>(local_histogram[3]));
  const bool bad = local_histogram[4] != 0;
  const unsigned total = static_cast<unsigned>(local_histogram[5]);
  // Every branch below is CTA-uniform: all threads read the same six published words.
  if (bad) {
    if (threadIdx.x == 0) diag[0] = 16;
    return;
  }
  if (!total) return;
  // Check official varlen prefix-sum metadata before any scheduler access.
  const uint2 start = reinterpret_cast<const uint2*>(schedule)[blockIdx.x];
  const uint2 end = reinterpret_cast<const uint2*>(schedule)[blockIdx.x + 1];
  if (start.x != expected_start.x || start.y != expected_start.y ||
      end.x != expected_end.x || end.y != expected_end.y) {
    if (threadIdx.x == 0) diag[0] = 16;
    return;
  }
  if (start.x >= static_cast<unsigned>(rows) || (start.x == end.x && start.y == end.y)) return;

  __syncthreads();  // boundaries consumed before the publication slot is overwritten
  if constexpr (kHasBins<Mode>) {
    for (unsigned bin = threadIdx.x; bin < Bins; bin += Threads)
      local_histogram[bin] = 0;
  }
  // The native core's unconditional entry __syncthreads publishes both local
  // zeros and this CTA's global diag reset before any fault or final emit.
  BatchEmitter<Mode> emit{dense, lengths, local_histogram, histogram, diag,
                          static_cast<unsigned>(dense_stride)};
  const auto make_scheduler = [&](unsigned sm, unsigned*, unsigned*) {
    return SafeScheduler<Page, kFold<Mode>>(sm, reinterpret_cast<const unsigned*>(lengths),
        reinterpret_cast<const unsigned*>(schedule), reinterpret_cast<const unsigned*>(indices),
        reinterpret_cast<const unsigned*>(table), table_stride, num_pages, diag,
        static_cast<unsigned>(rows));
  };
  deep_gemm::sm100_mqa_logits_filter_core_impl<
      32, 128, false, false, 4, 256, QStages, KVStages, 0, 128, 256,
      cutlass::float_e4m3_t, float, float>(width, dense,
          map_q, map_sf_q, map_kv, map_sf_kv, map_weights, make_scheduler, emit);
  emit.finish();
  // Same frozen all-thread barrier; the tail flush belongs to the math threads, which
  // are the only ones that know which row this CTA finished on.
  if constexpr (kFlushes<Mode>) {
    cutlass::arch::NamedBarrier(Threads, 1).sync();
    if (threadIdx.x < MathThreads && emit.cur_row >= 0) {
      int* dst = histogram + static_cast<size_t>(emit.cur_row) * Bins;
      for (unsigned bin = threadIdx.x; bin < Bins; bin += MathThreads) {
        const int count = local_histogram[bin];
        if (count != 0) atomicAdd(dst + bin, count);
      }
    }
  }
}

void driver_check(CUresult code, const char* operation) {
  if (code == CUDA_SUCCESS) return;
  const char* message = nullptr;
  cuGetErrorString(code, &message);
  TORCH_CHECK(false, operation, ": ", message ? message : "unknown CUDA driver error");
}

void check_tensor(const Tensor& tensor, at::ScalarType dtype,
                  const c10::Device& device, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device &&
      tensor.scalar_type() == dtype && tensor.is_contiguous(), name,
      " must be contiguous CUDA tensor of required dtype on the handle device");
}

bool overlaps(const Tensor& a, const Tensor& b) {
  const auto a0 = reinterpret_cast<uintptr_t>(a.const_data_ptr());
  const auto b0 = reinterpret_cast<uintptr_t>(b.const_data_ptr());
  return a0 < b0 + b.nbytes() && b0 < a0 + a.nbytes();
}

template <int Page, int QStages, int KVStages, int Mode>
int configure_resources(int requested, int maximum, int& registers, int& local_bytes) {
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, native_tma_histogram_batch<Page, QStages, KVStages, Mode>));
  TORCH_CHECK(static_cast<size_t>(requested) + attributes.sharedSizeBytes <=
      static_cast<size_t>(maximum), "Batched TMA histogram Q", QStages, "/KV", KVStages,
      " requires ", requested, " dynamic + ", attributes.sharedSizeBytes,
      " static shared bytes, exceeding device opt-in limit ", maximum,
      "; R1 split256/math256/TMEM3 remain fixed");
  C10_CUDA_CHECK(cudaFuncSetAttribute(native_tma_histogram_batch<Page, QStages, KVStages, Mode>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, requested));
  registers = attributes.numRegs;
  local_bytes = static_cast<int>(attributes.localSizeBytes);
  return static_cast<int>(attributes.sharedSizeBytes);
}
}  // namespace

struct BatchProducerHandle::Impl {
  Tensor q, cache, weights;
  int width, rows, stage, mode, q_stages, kv_stages, page, pages, shared_bytes,
      static_shared_bytes, device_shared_limit, registers, local_bytes;
  cute::TmaDescriptor map_q, map_kv, map_sf_kv, map_weights;

  Impl(Tensor query, Tensor kv, Tensor head_weights, int64_t max_seq_len, int64_t rows_arg,
       int64_t stage_arg, int64_t mode_arg)
      : q(query), cache(kv), weights(head_weights) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    check_tensor(q, at::ScalarType::Float8_e4m3fn, q.device(), "q");
    check_tensor(cache, at::kByte, q.device(), "cache");
    check_tensor(weights, at::kFloat, q.device(), "weights");
    TORCH_CHECK(rows_arg >= 1 && rows_arg <= MaxRows, "rows must be in 1..32 (GVR2 split family)");
    rows = static_cast<int>(rows_arg);
    TORCH_CHECK((q.dim() == 3 && q.size(0) == rows && q.size(1) == 32 && q.size(2) == 128) ||
        (q.dim() == 4 && q.size(0) == rows && q.size(1) == 1 && q.size(2) == 32 && q.size(3) == 128),
        "q must be R1 H32 D128 [B,32,128] or [B,1,32,128]");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == rows && weights.size(1) == 32,
        "weights must be FP32 [B,32]");
    TORCH_CHECK(cache.dim() == 4 && cache.size(0) >= 1 && cache.size(2) == 1 &&
        cache.size(3) == 132 && (cache.size(1) == 64 || cache.size(1) == 128),
        "cache must be native SoA [pages,64|128,1,132]");
    TORCH_CHECK(max_seq_len > 0 && max_seq_len <= 1048576, "max_seq_len outside 1..1M");
    TORCH_CHECK(stage_arg == 2 || stage_arg == 3,
        "stage must be 2 (Q1/KV6, production) or 3 (Q3/KV5, the upstream-shaped control)");
    TORCH_CHECK(mode_arg == 8 || mode_arg == 0 || mode_arg == 10 || mode_arg == 12 ||
        mode_arg == 15 || mode_arg == 16 || mode_arg == 17 || mode_arg == 18 ||
        mode_arg == 19 || mode_arg == 20 || mode_arg == 21 || mode_arg == 22 ||
        mode_arg == 23 || mode_arg == 24,
        "mode must be 8 (production), 0 (scores only; measurement arm), or the temporary "
        "diagnostics 10 (no per-score atomic) / 12 (no fixed per-CTA term) whose histogram "
        "is WRONG by construction");
    TORCH_CHECK(cache.size(0) <= INT_MAX, "too many physical pages");
    width = max_seq_len; stage = stage_arg; mode = mode_arg; page = cache.size(1); pages = cache.size(0);
    q_stages = stage == 3 ? 3 : 1; kv_stages = stage == 3 ? 5 : 6;
    c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(cache.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(weights.data_ptr()) % 16 == 0,
        "TMA input pointers require 16-byte alignment");
    // Only the row extent of the Q and weights descriptors differs from the B=1 build.
    const cuuint64_t q_dims[2] = {128, static_cast<unsigned>(32 * rows)}, q_strides[1] = {128};
    const cuuint32_t q_box[2] = {128, 128}, unit2[2] = {1, 1};
    driver_check(cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap*>(&map_q),
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, q.data_ptr(), q_dims, q_strides, q_box, unit2,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "encode Q");
    const cuuint64_t kv_dims[3] = {128, static_cast<unsigned>(page), static_cast<unsigned>(pages)};
    const cuuint64_t kv_strides[2] = {128, static_cast<unsigned>(page * 132)};
    const cuuint32_t kv_box[3] = {128, static_cast<unsigned>(page), 1}, unit3[3] = {1, 1, 1};
    driver_check(cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap*>(&map_kv),
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, cache.data_ptr(), kv_dims, kv_strides, kv_box, unit3,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "encode KV");
    const cuuint64_t sf_dims[2] = {static_cast<unsigned>(page), static_cast<unsigned>(pages)};
    const cuuint64_t sf_strides[1] = {static_cast<unsigned>(page * 132)};
    const cuuint32_t sf_box[2] = {static_cast<unsigned>(page), 1};
    driver_check(cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap*>(&map_sf_kv),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, cache.data_ptr<uint8_t>() + page * 128,
        sf_dims, sf_strides, sf_box, unit2, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "encode KV scales");
    const cuuint64_t w_dims[2] = {32, static_cast<unsigned>(rows)}, w_strides[1] = {128};
    const cuuint32_t w_box[2] = {32, 4};
    driver_check(cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap*>(&map_weights),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, weights.data_ptr(), w_dims, w_strides, w_box, unit2,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "encode weights");
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&device_shared_limit,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, q.get_device()));
    // Deliberately retain the production carve-out. The effective zero/flush loops
    // still use Bins, and Mode 24's scratch bin is inside this reserved tail.
    shared_bytes = sizeof(StageStorage<1, 6>) + AllocatedBins * sizeof(int);
    #define CONFIG_S(Q, KV, MODE) (page == 64 \
        ? configure_resources<64, Q, KV, MODE>(shared_bytes, device_shared_limit, registers, local_bytes) \
        : configure_resources<128, Q, KV, MODE>(shared_bytes, device_shared_limit, registers, local_bytes))
    #define CONFIG(MODE) (stage == 3 ? CONFIG_S(3, 5, MODE) : CONFIG_S(1, 6, MODE))
    switch (mode) {
      case 8:  static_shared_bytes = CONFIG(8); break;
      case 10: static_shared_bytes = CONFIG(10); break;
      case 12: static_shared_bytes = CONFIG(12); break;
      case 15: static_shared_bytes = CONFIG(15); break;
      case 16: static_shared_bytes = CONFIG(16); break;
      case 17: static_shared_bytes = CONFIG(17); break;
      case 18: static_shared_bytes = CONFIG(18); break;
      case 19: static_shared_bytes = CONFIG(19); break;
      case 20: static_shared_bytes = CONFIG(20); break;
      case 21: static_shared_bytes = CONFIG(21); break;
      case 22: static_shared_bytes = CONFIG(22); break;
      case 23: static_shared_bytes = CONFIG(23); break;
      case 24: static_shared_bytes = CONFIG(24); break;
      default: static_shared_bytes = CONFIG(0); break;
    }
    #undef CONFIG
  }
};

BatchProducerHandle::BatchProducerHandle(Tensor q, Tensor cache, Tensor weights,
    int64_t max_seq_len, int64_t rows, int64_t stage, int64_t mode)
    : impl(std::make_unique<Impl>(q, cache, weights, max_seq_len, rows, stage, mode)) {}
BatchProducerHandle::~BatchProducerHandle() = default;

c10::intrusive_ptr<BatchProducerHandle> make_handle(Tensor q, Tensor cache,
    Tensor weights, int64_t max_seq_len, int64_t rows, int64_t stage, int64_t mode) {
  return c10::make_intrusive<BatchProducerHandle>(q, cache, weights, max_seq_len, rows, stage, mode);
}

namespace {
void validate_buffers(const BatchProducerHandle::Impl& h,
    const Tensor& table, const Tensor& lengths, const Tensor& schedule,
    const Tensor& indices, const Tensor& dense, const Tensor& hist, const Tensor& diag) {
  for (const auto& tensor : {table, lengths, schedule, indices, hist, diag})
    check_tensor(tensor, at::kInt, h.q.device(), "int32 argument");
  // dense is [B, width] carved out of a padded [B, stride] allocation, so it is a row
  // view rather than contiguous; every other requirement is the B=1 one.
  TORCH_CHECK(dense.is_cuda() && dense.device() == h.q.device() &&
      dense.scalar_type() == at::kFloat, "dense must be FP32 CUDA on the handle device");
  TORCH_CHECK(table.dim() == 2 && table.size(0) == h.rows &&
      table.size(1) >= (h.width + h.page - 1) / h.page && table.size(1) <= INT_MAX,
      "table must be int32[B, pages] covering max_seq_len");
  TORCH_CHECK(lengths.numel() == h.rows && indices.numel() == h.rows,
      "lengths[B] and indices[B] required");
  TORCH_CHECK(schedule.dim() == 2 && schedule.size(0) == Ctas + 1 && schedule.size(1) == 2 &&
      reinterpret_cast<uintptr_t>(schedule.data_ptr()) % alignof(uint2) == 0,
      "schedule must be aligned native int32[149,2]");
  TORCH_CHECK(dense.dim() == 2 && dense.size(0) == h.rows && dense.size(1) == h.width &&
      dense.stride(1) == 1 && dense.stride(0) >= h.width && dense.stride(0) % 4 == 0 &&
      dense.stride(0) <= INT_MAX && reinterpret_cast<uintptr_t>(dense.data_ptr()) % 16 == 0,
      "dense must be FP32[B,max_seq_len] with unit column stride and 4-divisible row stride");
  TORCH_CHECK(hist.dim() == 2 && hist.size(0) == h.rows && hist.size(1) == Bins,
      "hist must be the configured int32[B,bins] global histogram (zero at entry; consumer restores; "
      "mode 0 leaves it untouched)");
  TORCH_CHECK(diag.dim() == 1 && diag.numel() == Ctas, "diag must be int32[148]");
  for (const auto& output : {dense, hist, diag}) {
    for (const auto& input : {h.q, h.cache, h.weights, table, lengths, schedule, indices})
      TORCH_CHECK(!overlaps(output, input), "producer input/output storage must not overlap");
  }
  TORCH_CHECK(!overlaps(dense, hist) && !overlaps(dense, diag) && !overlaps(hist, diag),
      "producer output buffers must not overlap");
}

void validate_rebound_inputs(const BatchProducerHandle::Impl& h, const Tensor& q,
                             const Tensor& weights) {
  check_tensor(q, at::ScalarType::Float8_e4m3fn, h.q.device(), "rebound q");
  check_tensor(weights, at::kFloat, h.q.device(), "rebound weights");
  TORCH_CHECK((q.dim() == 3 && q.size(0) == h.rows && q.size(1) == 32 && q.size(2) == 128) ||
      (q.dim() == 4 && q.size(0) == h.rows && q.size(1) == 1 && q.size(2) == 32 && q.size(3) == 128),
      "rebound q must be R1 H32 D128 [B,32,128] or [B,1,32,128]");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == h.rows && weights.size(1) == 32,
      "rebound weights must be FP32 [B,32]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(weights.data_ptr()) % 16 == 0,
      "rebound Q/weights require 16-byte-aligned TMA addresses");
}

void launch_produce(const BatchProducerHandle::Impl& h,
    const Tensor& table, const Tensor& lengths, const Tensor& schedule,
    const Tensor& indices, const Tensor& dense, const Tensor& hist, const Tensor& diag,
    const cute::TmaDescriptor& map_q, const cute::TmaDescriptor& map_weights,
    cudaStream_t stream) {
  // One same-stream kernel for the whole batch; the grid is the machine, not the batch.
  #define LAUNCH_STAGE(PAGE, Q, KV, MODE) native_tma_histogram_batch<PAGE, Q, KV, MODE><<<Ctas, Threads, h.shared_bytes, stream>>>( \
      lengths.data_ptr<int>(), table.data_ptr<int>(), table.size(1), h.pages, \
      schedule.data_ptr<int>(), indices.data_ptr<int>(), dense.data_ptr<float>(), \
      static_cast<int>(dense.stride(0)), hist.data_ptr<int>(), diag.data_ptr<int>(), \
      h.width, h.rows, map_q, map_weights, h.map_kv, h.map_sf_kv, map_weights)
  #define LAUNCH_HIST(PAGE, MODE) do { \
      if (h.stage == 3) { LAUNCH_STAGE(PAGE, 3, 5, MODE); } else { LAUNCH_STAGE(PAGE, 1, 6, MODE); } } while (0)
  #define DISPATCH(M) do { if (h.page == 64) { LAUNCH_HIST(64, M); } else { LAUNCH_HIST(128, M); } } while (0)
  switch (h.mode) {
    case 8:  DISPATCH(8);  break;
    case 10: DISPATCH(10); break;
    case 12: DISPATCH(12); break;
    case 15: DISPATCH(15); break;
    case 16: DISPATCH(16); break;
    case 17: DISPATCH(17); break;
    case 18: DISPATCH(18); break;
    case 19: DISPATCH(19); break;
    case 20: DISPATCH(20); break;
    case 21: DISPATCH(21); break;
    case 22: DISPATCH(22); break;
    case 23: DISPATCH(23); break;
    case 24: DISPATCH(24); break;
    default: DISPATCH(0);  break;
  }
  #undef DISPATCH
  #undef LAUNCH_HIST
  #undef LAUNCH_STAGE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

void produce(const c10::intrusive_ptr<BatchProducerHandle>& handle,
    const Tensor& table, const Tensor& lengths, const Tensor& schedule,
    const Tensor& indices, const Tensor& dense, const Tensor& hist, const Tensor& diag) {
  TORCH_CHECK(handle && handle->impl, "invalid producer handle");
  const auto& h = *handle->impl;
  validate_buffers(h, table, lengths, schedule, indices, dense, hist, diag);
  c10::cuda::CUDAGuard guard(h.q.device());
  launch_produce(h, table, lengths, schedule, indices, dense, hist, diag,
      h.map_q, h.map_weights, c10::cuda::getCurrentCUDAStream(h.q.get_device()).stream());
}

void produce_rebound(const c10::intrusive_ptr<BatchProducerHandle>& handle,
    const Tensor& q, const Tensor& weights,
    const Tensor& table, const Tensor& lengths, const Tensor& schedule,
    const Tensor& indices, const Tensor& dense, const Tensor& hist, const Tensor& diag) {
  TORCH_CHECK(handle && handle->impl, "invalid producer handle");
  const auto& h = *handle->impl;
  validate_rebound_inputs(h, q, weights);
  validate_buffers(h, table, lengths, schedule, indices, dense, hist, diag);
  for (const auto& output : {dense, hist, diag})
    TORCH_CHECK(!overlaps(output, q) && !overlaps(output, weights),
        "current rebound Q/weights must not overlap producer outputs");
  c10::cuda::CUDAGuard guard(h.q.device());
  static_assert(alignof(cute::TmaDescriptor) >= 64);
  cute::TmaDescriptor current_q_map = h.map_q;
  cute::TmaDescriptor current_weights_map = h.map_weights;
  driver_check(cuTensorMapReplaceAddress(
      reinterpret_cast<CUtensorMap*>(&current_q_map), q.data_ptr()), "replace Q address");
  driver_check(cuTensorMapReplaceAddress(
      reinterpret_cast<CUtensorMap*>(&current_weights_map), weights.data_ptr()), "replace weights address");
  launch_produce(h, table, lengths, schedule, indices, dense, hist, diag,
      current_q_map, current_weights_map,
      c10::cuda::getCurrentCUDAStream(h.q.get_device()).stream());
}

c10::Dict<std::string, int64_t> producer_info(const c10::intrusive_ptr<BatchProducerHandle>& handle) {
  TORCH_CHECK(handle && handle->impl, "invalid producer handle");
  const auto& h = *handle->impl;
  c10::Dict<std::string, int64_t> info;
  info.insert("mode", h.mode);
  info.insert("stage", h.stage);
  info.insert("ctas", Ctas);
  info.insert("threads", Threads);
  info.insert("math_threads", MathThreads);
  info.insert("bins", Bins);
  info.insert("rows", h.rows);
  info.insert("hist_rows", h.rows);
  info.insert("workspace_words", Bins * h.rows);
  info.insert("max_rows", MaxRows);
  info.insert("row_flush_barrier", RowFlushBarrier);
  info.insert("folded_block_tokens", h.mode == 8 ? 1 : 0);
  info.insert("publishes_histogram", (h.mode == 8 || h.mode >= 20) ? 1 : 0);
  info.insert("nan_guarded", (h.mode == 8 || h.mode == 20) ? 1 : 0);
  info.insert("completion_counter_in_kernel", 0);
  info.insert("certificate_in_producer", 0);
  info.insert("launches_per_complete_producer_call", 1);
  info.insert("diag_elements", Ctas);
  info.insert("producer_launches", 1);
  info.insert("global_reset_launches", 0);
  info.insert("supports_produce_rebound", 1);
  info.insert("page", h.page);
  info.insert("max_seq_len", h.width);
  info.insert("native_shared_bytes", sizeof(StageStorage<1, 6>));
  info.insert("registers_per_thread", h.registers);
  info.insert("local_bytes_per_thread", h.local_bytes);
  info.insert("split_kv", 256);
  info.insert("extra_shared_bytes", AllocatedBins * sizeof(int));
  info.insert("effective_histogram_shared_words", Bins);
  info.insert("shared_bytes", h.shared_bytes);
  info.insert("static_shared_bytes", h.static_shared_bytes);
  info.insert("device_optin_shared_bytes", h.device_shared_limit);
  info.insert("stages_q", h.q_stages);
  info.insert("stages_kv", h.kv_stages);
  info.insert("stages_tmem", 3);
  return info;
}
}  // namespace LITETOPK_PRODUCER_NAMESPACE

#define LITETOPK_STRINGIFY_INNER(value) #value
#define LITETOPK_STRINGIFY(value) LITETOPK_STRINGIFY_INNER(value)
TORCH_LIBRARY(LITETOPK_PRODUCER_NAMESPACE, m) {
  using namespace LITETOPK_PRODUCER_NAMESPACE;
  m.class_<BatchProducerHandle>("BatchProducerHandle");
  m.def("make_handle(Tensor q, Tensor cache, Tensor weights, int max_seq_len, int rows, int stage, int mode) -> __torch__.torch.classes."
        LITETOPK_STRINGIFY(LITETOPK_PRODUCER_NAMESPACE) ".BatchProducerHandle", &make_handle);
  m.def("produce(__torch__.torch.classes." LITETOPK_STRINGIFY(LITETOPK_PRODUCER_NAMESPACE)
        ".BatchProducerHandle handle, Tensor table, Tensor lengths, Tensor schedule, Tensor indices, Tensor(a!) dense, Tensor(b!) hist, Tensor(c!) diag) -> ()", &produce);
  m.def("produce_rebound(__torch__.torch.classes." LITETOPK_STRINGIFY(LITETOPK_PRODUCER_NAMESPACE)
        ".BatchProducerHandle handle, Tensor q, Tensor weights, Tensor table, Tensor lengths, Tensor schedule, Tensor indices, Tensor(a!) dense, Tensor(b!) hist, Tensor(c!) diag) -> ()", &produce_rebound);
  m.def("producer_info(__torch__.torch.classes." LITETOPK_STRINGIFY(LITETOPK_PRODUCER_NAMESPACE)
        ".BatchProducerHandle handle) -> Dict(str, int)", &producer_info);
}
#undef LITETOPK_STRINGIFY
#undef LITETOPK_STRINGIFY_INNER
