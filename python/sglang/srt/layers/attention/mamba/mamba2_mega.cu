// Nemotron 3 Super Mamba-2 decoder layer as one SM120 cooperative kernel.
//
// One launch covers: residual add + RMSNorm, mixed-precision 4096->18560 input
// projection, gate/x/B/C/dt split, width-4 depthwise causal convolution with
// the committed-window update, 128-head selective state update, gated RMSNorm,
// and the mixed-precision 8192->4096 output projection written straight into
// the caller-owned destinations.
//
// Phase DAG (2 grid barriers):
//   P0  every CTA redundantly builds the normalized activation (4096 wide, L2
//       resident) so no barrier is needed to publish it; CTA 0 writes
//       out_residual and clears the gated-norm group accumulators.
//   P1  in_proj rows are grid-strided across warps.  Rows in the conv range get
//       convolution + silu + committed-window update fused into their epilogue,
//       dt rows get softplus/dA fused, gate rows are staged as-is.
//   -- grid.sync --
//   P2  selective state update over (head, head_dim) rows, 128 states per row,
//       fp32 read/modify/write into the committed slot.  x*silu(gate) is
//       staged; each CTA first reduces its group square sums in shared memory,
//       then atomically accumulates one value per non-empty group into global
//       scratch.
//   -- grid.sync --
//   P3  gated RMSNorm epilogue + out_proj rows grid-strided across warps,
//       writing out_hidden directly.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// Launch-geometry knobs (see tools/tune_geometry.py).
#ifndef NWARPS
#define NWARPS 8
#endif
#ifndef P1_ROWS
#define P1_ROWS 1
#endif
#ifndef P3_ROWS
#define P3_ROWS 1
#endif
// Weight-load flavour.  A demand stream tops out well below what the same DRAM
// delivers when prefetches are added on top of it, and prefetches do not
// consume L1 miss slots -- so the L1 path is worth probing directly:
//   0 = default caching, 1 = ld.global.cs (evict-first), 2 = ld.global.cg
//   (bypass L1, cache in L2), 3 = ld.global.nc (read-only path), 4 =
//   ld.global.lu
#ifndef WSTREAM
#define WSTREAM 1
#endif
// 1 = prefetch the next grid-stride panel into L2 (measured: no gain, off).
#ifndef PREFETCH
#define PREFETCH 0
#endif
// Pull the out_proj weights into L2 ahead of P3, so P3 reads them at L2 speed.
//   1 = spread the hints across P1.  Works, but P1's demand stream already has
//       DRAM saturated, so the traffic just moves: P3 drops by as much as P1
//       gains (measured, see profiles/01).
//   2 = issue them only in the two grid-barrier windows.  The hint is not free
//       -- it is serviced while the grid drains, so it lands on whatever DRAM
//       slack that phase has -- but P3 then reads out_proj from L2 at 4.0-4.5
//       TB/s instead of DRAM at 1.2.  One instruction per warp covers a 4 KiB
//       page, so a sweep across a 658-CTA grid is ~21.6 MB.
//       Re-measured at grid 658 (see PF_OUT_FP8): worth 1.9-2.0 us on the two
//       BF16-in rows, where PF_OUT=0 costs 187.74/166.59 us against 185.85/
//       164.58.  PF_OUT=1 remains far worse than either, at 191.31/166.75.
#ifndef PF_OUT
#define PF_OUT 2
#endif
// Number of 4 KiB pages of the in_proj panel each warp pulls into L2 during P0.
// Measured dead end, kept as the knob that proves it -- see profiles/04.
//
// The premise was that P0 has spare bandwidth: it moves only 24 KB of DRAM
// traffic (hidden, residual, norm_w; everything after the first CTA is an L2
// hit) yet costs 4.2 us cold against 1.7 us warm.  The hint does exactly what
// it is supposed to -- P1 rises from 1128 to 1376 GB/s on m1-l0 at PF_IN=4 --
// and P0 lengthens by precisely what P1 saves (4.4 -> 16.3 us), leaving CUPTI
// flat at 118.0 / 118.3 / 118.1 / 118.9 us for PF_IN = 0 / 1 / 2 / 4.  So
// that 2.5 us is not idle bandwidth, it is the memory system already working:
// the harness leaves L2 full of dirty lines and their writeback is in flight
// from the kernel's first instruction.  DRAM is saturated from t=0, and there
// is no window anywhere in this kernel that a prefetch can spend for free.
#ifndef PF_IN
#define PF_IN 0
#endif
// 1 = P2 folds gnorm_w into the y_pre it stores, so P3's staging does not read
// it.  P3 is L2-bandwidth-bound once PF_OUT lands the out_proj panel there:
// each participating CTA stages 32 KB of y_pre plus 16 KB of gnorm_w, and at
// 512 CTAs that is 24.6 MB on top of the panel's own 33.6 MB.  Folding moves
// gnorm_w's share to one 2-byte load per row on P2's epilogue lane -- 16 KB for
// the entire grid instead of 16 KB per CTA.
//
// This reassociates two fp32 multiplies: (y*gw)*rstd instead of (y*rstd)*gw.
// Both round to the same bf16 for all but ~1 element in 65536, which is far
// inside the tolerance of the production output; measured, all four rows come
// out bit-identical.
//
// Worth -0.65% to -1.24% per row, +0.89% geomean (m1-l0 113.44 -> 112.03,
// m1-l6 136.70 -> 135.66, m1-l22 185.28 -> 184.08, m1-l38 164.02 -> 162.59).
//
// profiles/01 measured this same fold as "net zero" and rejected it.  That was
// correct then and is worth understanding rather than just overwriting: at the
// time P3 read out_proj from DRAM and was DRAM-bound, so trading L2 traffic for
// DRAM latency bought nothing.  PF_OUT now lands the panel in L2 and P3 is
// L2-bandwidth-bound, which is exactly the regime where the trade pays.
#ifndef GNORM_P2
#define GNORM_P2 1
#endif
// Pages of the out_proj panel each warp hints per barrier window.  Coverage is
// PF_OUT_PAGES * 2 * ngwarp pages; the BF16 panel is 16384 pages, so at grid
// 658 one page per warp leaves it only 64% covered -- which looked like the
// obvious thing to fix, and is not.  Measured 1 / 2 / 3: m1-l0
// 113.38/115.23/115.02, m1-l6 136.54/138.21/137.06, m1-l22
// 185.94/186.86/186.88, m1-l38 164.57/165.86/165.50.  One page per warp is best
// on every row.
//
// So the hint's value is not coverage of the panel, it is how much DRAM work
// the barrier window can absorb, and one sweep already fills it.  Asking for
// more just takes bandwidth from the P1 tail that is still draining into the
// same barrier -- the same conservation PF_IN ran into.  PF_OUT_PAGES=1
// reproduces the previous code exactly (pages gwarp and gwarp + ngwarp).
//
// Re-tested once PF_OUT_COV existed, because the two knobs together can express
// what neither could alone: PAGES=2 reaches 21056 pages of the BF16 panel and
// COV=13 caps that at 13312, i.e. f = 0.81 -- the balance point the FP8-out
// rows were just shown to want, and which PAGES alone could only overshoot
// to 1.0. It is still worse: m1-l6 135.10 -> 136.59, m1-l22 184.40 -> 185.23.
// So what limits the BF16-out rows is not the coverage fraction but how much
// DRAM work a barrier window can absorb, exactly as the 1/2/3 sweep concluded.
// The FP8-out rows are the ones with a coverage problem, and only downward.
#ifndef PF_OUT_PAGES
#define PF_OUT_PAGES 1
#endif
// Sixteenths of the out_proj panel the hints are allowed to cover.  16 = the
// whole panel (previous behaviour); the guard is a prefix, so at COV=12 rows
// 0..3071 are hinted and rows 3072..4095 are left for P3 to demand-read.
//
// The direction PF_OUT_PAGES never explored is *down*.  On the two FP8-out rows
// one sweep per window already covers 100% of the 8192-page panel, so P3 reads
// every byte from L2 at 4.1-5.0 TB/s and DRAM is idle for the whole phase --
// 8.2 us on m1-l0, 6.8 us on m1-l38.  Leaving a slice unhinted puts that idle
// DRAM back to work: P1 gets shorter by the bytes it no longer pulls, and P3
// pays nothing extra as long as its DRAM stream finishes inside the L2 stream
// it runs alongside.  The optimum is where the two streams balance, f = R_l2 /
// (R_l2 + R_dram) ~ 0.8, not at full coverage.
//
// Measured (tools/define_ab.py, 5 interleaved repeats, authoritative CUPTI),
// COV = 16/14/13/12/11/10/8:
//   m1-l0  112.05 / 112.06 / 110.43 / 109.38 / 111.46 / 113.04 / 112.51
//   m1-l38 162.80 / 161.90 / 161.58 / 161.84 / 163.12 / 163.81 / 163.44
// A clear interior optimum at 12, i.e. f = 0.75, worth -2.4% on m1-l0 and
// -0.6..-1.2% on m1-l38.  The turn below 11 is sharp for a structural reason:
// round 0 reaches page `ngwarp` = 5264 of the FP8 panel's 8192, so COV >= 11
// trims only the *second* window's hints, and COV <= 10 starts cutting the
// first window's coverage -- which is the coverage that actually lands in bar1.
//
// The two BF16-out rows are untouched at 12: their panel is 16384 pages and the
// two windows only reach 10528 of them, so f = 0.64 already sits below the
// balance and the cap does not bind (135.63 -> 135.45, 184.97 -> 184.86, both
// inside the spread).  Raising *their* coverage to the predicted 0.81 was tried
// and refuted -- see PF_OUT_PAGES.
#ifndef PF_OUT_COV
#define PF_OUT_COV 12
#endif
// The same cap for a BF16 out_proj, in sixteenths of its 16384-page panel.
//
// It needs its own knob because `PF_OUT_COV` is a fraction of the panel and the
// BF16 panel is twice the size, so the cap that binds an FP8 one at f = 0.75 is
// inert here: the two windows reach 10528 pages, f = 0.64, already below 12/16.
// Reading that as "these rows have no coverage problem" conflates the two
// mechanisms profiles/06 separated.  The FP8 win at COV=12 was only *half*
// about where the panel is served from; the other half -- the larger half on
// m1-l38 -- was that issuing 5264 hints immediately before a barrier delays it,
// and bar2 fell from 5.9 to 2.1 us when the second window was trimmed.  The
// BF16-out rows never had their second window trimmed at all, and their bar2
// still costs 5.9 (m1-l6) and 6.8 us (m1-l22).
//
// The two windows reach page 10527 of 16384, so anything from 11 up is the same
// uncapped kernel; 12 is the value the un-split `PF_OUT_COV` was applying here,
// and picking it makes this split a no-op -- all four specializations compile
// to SASS byte-identical to the pre-split source, checked with `cuobjdump
// -sass`. 6..10 trims only the second window (round 0 ends at page ngwarp =
// 5264, i.e. 5.14/16); 5 and below start cutting the first window, which is the
// coverage that lands inside bar1.
//
// Measured (tools/define_ab.py, 9 interleaved repeats, authoritative CUPTI),
// COV_BF16 = 16/8/6/4:
//   m1-l6  135.13 / 134.93 / 134.19 / 135.97
//   m1-l22 185.57 / 185.89 / 185.34 / 187.54
// The same interior-optimum shape as the FP8 sweep, and it lands on the
// smallest value that still leaves window 0 whole -- 5.14/16 rounded up.  4
// cuts into window 0 and is the worst point on both rows, which is the FP8
// panel's sharp turn below 11 reproduced on a panel of a different size.
//
// That is `tools/define_ab.py` talking, and the harness disagrees, so the knob
// stays at 16.  Two full runs at 6 against two at 16, candidate us:
//   m1-l6  135.31 / 135.25   against  135.06 / 135.06
//   m1-l22 185.04 / 185.04   against  184.75 / 184.57
// against a four-run control of the byte-identical cubin that spans
// 135.06-135.39 and 184.57-184.96.  So COV=6 does not separate from doing
// nothing on either row: it buys nothing, which is why it goes back.  What it
// does refute, and not marginally, is define_ab's -0.70% on m1-l6 -- that
// predicts 134.19 us, and no full run of any variant has ever put the row below
// 135.06.
//
// The harness's per-row floor is ~+-0.25%, not the +-0.07% the two FP8-out rows
// first suggested; profiles/07 has the samples and what the bar means for any
// future knob.
//
// This is the second time define_ab has been overruled here, and together the
// two instances say something the first could not.  profiles/06 explained the
// PF_P3 reversal as interleaving leaving the previous variant's out_proj panel
// resident, which would bias *toward* configurations that want it in L2 -- but
// this knob takes the panel *out* of L2 and was over-credited just the same.  A
// bias with no sign is not that mechanism.  What both knobs have in common is
// only that they redistribute traffic between phases without changing the byte
// count, and that is precisely the class of change whose measured value depends
// on machine state a same-process interleave holds fixed and the harness does
// not.  The rule stands and is now unconditional: **a PF_* knob is settled by a
// full run, in either direction.**  Everything below is kept as the evidence
// for that.
//
// It is worth much less than the FP8 cap was (-0.70% on m1-l6, -0.12% on
// m1-l22, ~0.2% of geomean), and the phase spans say why.  `tools/ab_sweep.sh
// PF_OUT_COV_BF16 16 6` on m1-l22 gives P2 9.3 -> 4.4, bar2 6.5 -> 2.3 and P3
// 13.7 -> 15.1, i.e. `body` 171.5 -> 164.2 for a CUPTI that barely moves: the
// 9 us the hints cost CTA 0 comes back as post-CTA-0 tail when the uncovered
// slice is demand-read instead.  (P2's own span shrinking is not a compiler
// reordering -- the hints are issued after `phase_mark(4)`.  CTAs leave bar1
// spread out, so the CTAs that finish P2 early are hinting while CTA 0 is still
// in its P2 body.)  This is the conservation profiles/03 named, measured from
// both ends; the residue is what the hint costs beyond its bytes.
#ifndef PF_OUT_COV_BF16
#define PF_OUT_COV_BF16 12
#endif
// Where the second window's hints are issued: 0 = immediately before barrier2,
// 1 = at the top of P2, before the state recurrence.  See the block comment at
// the use site for the reasoning; it did not survive contact.
//
// tools/define_ab.py PF_OUT_EARLY 0 1, five interleaved repeats:
//   m1-l0  109.23 -> 112.89   m1-l6  135.68 -> 139.65
//   m1-l22 184.62 -> 188.53   m1-l38 160.94 -> 163.07
// Worse on every row, by more than trimming the same hints away entirely.  P2
// is latency-bound, not idle: a warp's single float4 of state is the whole
// phase, and 168k prefetch requests queued ahead of it delay every CTA's
// recurrence rather than filling a gap.  So P2 joins P0 (profiles/04) -- a
// phase can look like it has spare DRAM and have no room for a hint, and the
// two now differ in mechanism but not in verdict.  Kept at 0 as the evidence.
#ifndef PF_OUT_EARLY
#define PF_OUT_EARLY 0
#endif
// 0/1 = off (PF_OUT_COV's prefix).  N > 1 = hint every row except every Nth,
// packing the hint slots onto the rows that remain, so coverage is (N-1)/N.
//
// PF_OUT_COV leaves the unhinted slice as a *suffix* of the panel, and a suffix
// of rows is a suffix of warps: at COV=8 the 2048 DRAM-read rows all live in
// CTAs 256..511, and that stream measures ~1.0 TB/s where P1's reaches 1.44.
// SKIP spreads the same slice over the whole grid instead -- at SKIP=4 every
// CTA has exactly two DRAM-reading warps out of eight.
//
// SKIP=4 on an FP8 panel is a controlled comparison against COV=12: 6144 hints
// either way, 5264 in the first window and 880 in the second either way, 3/4 of
// the panel covered either way.  Only the addresses differ.
#ifndef PF_OUT_SKIP
#define PF_OUT_SKIP 0
#endif
// 4 KiB pages of its *own* out_proj row a P3 warp hints before the gated-norm
// staging.  A row is 2 pages of an FP8 panel and 4 of a BF16 one.
//
// The staging loop is ~3.4 us of pure L2 traffic (32 KB of y_pre per CTA, 512
// CTAs) that runs before the GEMV issues a single weight load, so any row still
// coming from DRAM can have its fetch hidden underneath it for free. PF_OUT_COV
// is what created those rows.
//
// Measured 0/1/2/4:
//   m1-l0  109.66 / 110.29 / 112.11 / 113.71   (FP8 out)
//   m1-l6  135.81 / 134.75 / 134.53 / 135.17   (BF16 out)
//   m1-l22 185.10 / 184.35 / 184.16 / 185.18   (BF16 out)
//   m1-l38 159.90 / 160.45 / 162.74 / 164.51   (FP8 out)
// and 2 re-confirmed at --repeat 9: m1-l6 135.12 -> 134.18, m1-l22 184.45 ->
// 183.92.  That is a consistent win on both BF16-out rows across two
// independent interleaved sweeps, so it was gated on out_proj precision and
// recorded.
//
// **The benchmark harness disagrees, and the harness is authoritative.**  Two
// full runs at PF_P3=2 put m1-l22 at 186.10 and 186.18 us against 184.88 at
// PF_P3=0, i.e. a reproducible +0.7%, while m1-l6 gained only 0.2%.  Those runs
// also bound this harness's own repeatability at +-0.1% per row (m1-l0 and
// m1-l38 compile byte-identically across the commit and moved 0.04-0.13%), so
// the contradiction is far outside the noise on both sides.  Net geomean effect
// is about -0.13%.  Left off.
//
// The methodological lesson is the durable part, and it is specific to this
// family of change: tools/define_ab.py runs its variants interleaved on the
// same bound layer, so each launch inherits an L2 that the *previous variant's*
// launch just populated with the same out_proj panel.  For a knob whose whole
// mechanism is what happens to be resident in L2, that is not an independent
// sample.  The cold-L2 flush makes the variants comparable to each other but
// not to the harness.  profiles/03 recorded the same shape of trap for `body`
// vs `cupti`; this is the same error one level out -- confirm PF_* knobs with a
// full run.
#ifndef PF_P3
#define PF_P3 0
#endif
// 1 = also issue the P3 self-row hints on an FP8 out_proj panel.  Off: those
// rows are already in L2, so the hint only competes with the demand stream --
// m1-l0 109.66 -> 112.11 and m1-l38 159.90 -> 162.74 at PF_P3=2.
#ifndef PF_P3_FP8
#define PF_P3_FP8 0
#endif
// 1 = also issue the PF_OUT hints when in_proj is FP8.
//
// PF_OUT used to be gated off for an FP8 in_proj, on the reasoning that its P1
// is half as long and already at the FP8 ceiling, so it has no bandwidth margin
// to absorb the hint.  That was true and correctly measured -- at grid 940.  It
// stopped being true when the grid dropped to 658: the gate is now worth
// -1.1% on both FP8-in rows (m1-l0 114.85 -> 113.58, m1-l6 137.42 -> 135.95)
// and is exactly neutral on the two BF16-in rows, which do not read it.
//
// Worth remembering as a class of bug rather than a one-off: a knob whose
// justification is a bandwidth *margin* is a knob that silently expires when
// any other change moves that margin.  This one had been carrying a stale gate
// for exactly one commit.
#ifndef PF_OUT_FP8
#define PF_OUT_FP8 1
#endif
// 1 = CTAs that own no out_proj row return immediately after the second grid
// barrier instead of staging an activation they will never read.
#ifndef P3_TRIM
#define P3_TRIM 1
#endif
// 1 = stage P2's B/C/D through shared memory, 0 = read them from scratch.
// Measured (tools/define_ab.py, 5 interleaved repeats): staging costs 1.4-2.0%
// on every row -- 116.77/139.55/189.76/167.01 us with, 114.88/137.65/185.95/
// 163.73 us without.  Staging is a reflex that only pays when a CTA reuses what
// it stages, and this one never does: a warp handles one or two of the 8192
// rows, so the whole grid's demand for B and C is 8192 x 32 B = 256 KB, while
// copying them in costs 8.5 KB per CTA, i.e. 5.6 MB across a 658-CTA grid.
#ifndef P2_STAGE
#define P2_STAGE 0
#endif
// Debug only: replace P1's conv/dt epilogue with a single store, to separate
// the epilogue's cost from the weight stream's.  Produces wrong results.
#ifndef NO_EPI
#define NO_EPI 0
#endif
// Debug: stop after this many phases (6 = full kernel).  Lets a profile or an
// intermediate comparison isolate one phase without splitting the launch.
#ifndef MAX_PHASE
#define MAX_PHASE 6
#endif

#define HSZ 4096
#define ISZ 8192
#define NHEAD 128
#define HDIM 64
#define NGROUP 8
#define DSTATE 128
#define CONVW 4
#define CONVDIM 10240
#define GSSZ 1024
#define PROJW 18560
#define HPG (NHEAD / NGROUP)
#define GNORM (ISZ / NGROUP)

// scratch layout, in floats
#define SC_GATE 0
#define SC_X 8192
#define SC_B 16384
#define SC_C 17408
#define SC_DT 18432
#define SC_DA 18560
#define SC_YPRE 18688
#define SC_GSUM 26880
#define SC_TIME 26896 /* 8 u64 phase timestamps from CTA 0, ns */
#define SC_TOTAL 26912

// Dynamic shared-memory plan, in bytes.  The three big consumers have disjoint
// lifetimes -- the in_proj activation dies at the first grid barrier, the
// post-conv B/C staging dies at the second, and the out_proj activation is only
// written after that -- so they all alias one region.  Keeping the total under
// 102400/5 - 1024 is what buys 5 resident CTAs per SM instead of 3.
template <bool IN_FP8, bool OUT_FP8> struct SmemPlan {
  static constexpr int kAct = IN_FP8 ? HSZ : 2 * HSZ;
  static constexpr int kY = OUT_FP8 ? ISZ : 2 * ISZ;
  static constexpr int kBC = P2_STAGE ? (2 * GSSZ + NHEAD) * 4 : 0;
  static constexpr int kMain = kAct > kY ? (kAct > kBC ? kAct : kBC)
                                         : (kY > kBC ? kY : kBC);
  static constexpr int kRed = kMain;      /* P0 warp partials */
  static constexpr int kGsum = kRed + 64; /* P2 CTA-local group square sums */
  static constexpr int kTotal = kGsum + 64;
};

struct MegaParams {
  const __nv_bfloat16 *hidden;
  const __nv_bfloat16 *residual;
  __nv_bfloat16 *out_hidden;
  __nv_bfloat16 *out_residual;
  const __nv_bfloat16 *norm_w;
  const void *in_w;
  const __nv_bfloat16 *conv_w;
  const __nv_bfloat16 *conv_b;
  const float *A;
  const __nv_bfloat16 *D;
  const __nv_bfloat16 *dt_bias;
  const __nv_bfloat16 *gnorm_w;
  const void *out_w;
  __nv_bfloat16 *conv_state;
  float *ssm_state;
  const int *cache_idx;
  float *scratch;
  float in_alpha;
  float in_inv_scale;
  float out_alpha;
  float out_inv_scale;
  float eps;
  int idx_offset;
};

// ---------------------------------------------------------------- dot helpers

__device__ __forceinline__ float dot16_fp8(const float4 wv, const float4 xv) {
  const __nv_fp8x2_e4m3 *w2 = reinterpret_cast<const __nv_fp8x2_e4m3 *>(&wv);
  const __nv_fp8x2_e4m3 *x2 = reinterpret_cast<const __nv_fp8x2_e4m3 *>(&xv);
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 a = static_cast<float2>(w2[i]);
    const float2 b = static_cast<float2>(x2[i]);
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
  }
  return acc;
}

__device__ __forceinline__ float dot8_bf16(const float4 wv, const float4 xv) {
  const __nv_bfloat162 *w2 = reinterpret_cast<const __nv_bfloat162 *>(&wv);
  const __nv_bfloat162 *x2 = reinterpret_cast<const __nv_bfloat162 *>(&xv);
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 a = __bfloat1622float2(w2[i]);
    const float2 b = __bfloat1622float2(x2[i]);
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
  }
  return acc;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// Pull the panel this warp will stream on its *next* grid-stride iteration into
// L2.  One warp covers 32*128B = 4KiB per instruction, and because every warp
// advances by exactly `ngwarp` units the grid prefetches the contiguous block
// it is about to demand-read -- the idle-bandwidth window the SM120 guide calls
// for.
template <int BYTES>
__device__ __forceinline__ void prefetch_l2(const char *__restrict__ base,
                                            int lane) {
#if PREFETCH
#pragma unroll
  for (int i = 0; i < (BYTES + 4095) / 4096; ++i)
    asm volatile(
        "prefetch.global.L2 [%0];" ::"l"(base + (size_t)i * 4096 + lane * 128));
#endif
}

// Prefetch the slice of `bytes` that belongs to grid-stride unit `u` of
// `nunit`, so the whole panel is pulled evenly across the phase instead of in a
// burst.
__device__ __forceinline__ void prefetch_slice(const char *__restrict__ base,
                                               int pages, int u, int nunit,
                                               int lane) {
#if PF_OUT == 1
  // pages <= 16384 and u < 18560, so the product stays in 32 bits and this is a
  // 32-bit divide rather than the ~100-cycle 64-bit one.
  const int lo = (pages * u) / nunit;
  const int hi = (pages * (u + 1)) / nunit;
  for (int q = lo; q < hi; ++q)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(base + ((size_t)q << 12) +
                                                  lane * 128));
#endif
}

// PF_OUT_PAGES 4 KiB pages of the out_proj panel per warp, issued immediately
// before a grid barrier so the hints are serviced while the grid drains.  A
// warp covers a page with 32 lanes x 128 B, so one instruction per warp sweeps
// ngwarp * 4 KiB -- 21.6 MB across a 658-CTA grid.  `round` is 0 before the
// first barrier and 1 before the second, so the two windows cover disjoint
// slices and PF_OUT_PAGES scales total coverage independently of the grid.
template <int PPR, int COV>
__device__ __forceinline__ void prefetch_pages(const char *__restrict__ base,
                                               int pages, int gwarp, int ngwarp,
                                               int round, int lane) {
#if PF_OUT == 2
#if PF_OUT_SKIP > 1
  // SKIP already fixes the coverage fraction, so the only bound left is the
  // panel itself.  The two knobs are alternative spellings of one quantity and
  // are not meant to compose.
  const int covered = pages;
#else
  // Sixteenths, so COV=16 recovers `pages` exactly and no divide is emitted.
  const int covered = (pages * COV) >> 4;
#endif
#pragma unroll
  for (int k = 0; k < PF_OUT_PAGES; ++k) {
    const int slot = gwarp + (round * PF_OUT_PAGES + k) * ngwarp;
#if PF_OUT_SKIP > 1
    // Spread the unhinted slice over the panel instead of leaving it as a
    // suffix: skip every PF_OUT_SKIP-th row and pack the hint slots onto the
    // rows that remain, so slot count and coverage are unchanged and only the
    // addresses move.  PPR is the panel's pages per row, so `sub` walks a row
    // and `c` indexes the hinted rows.
    constexpr int kKeep = PF_OUT_SKIP - 1;
    const int c = slot / PPR;
    const int row = (c / kKeep) * PF_OUT_SKIP + (c % kKeep);
    const int page = row * PPR + (slot % PPR);
#else
    const int page = slot;
#endif
    if (page < covered)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(
          base + ((size_t)page << 12) + lane * 128));
  }
#else
  (void)base;
  (void)pages;
  (void)gwarp;
  (void)ngwarp;
  (void)round;
  (void)lane;
#endif
}

// The first PF_IN pages of the in_proj panel this warp will demand-read,
// walking the same grid-stride sequence P1 uses so the hint and the demand
// agree.  A warp covers one 4 KiB page per instruction (32 lanes x 128 B),
// which is exactly one FP8 row or half a BF16 row.
template <bool IN_FP8>
__device__ __forceinline__ void prefetch_in(const char *__restrict__ base,
                                            int gwarp, int ngwarp, int lane) {
#if PF_IN
  constexpr int kUnitBytes = P1_ROWS * (IN_FP8 ? HSZ : 2 * HSZ);
  constexpr int kPages = kUnitBytes >> 12;
  static_assert(kPages >= 1, "a grid-stride unit must cover at least one page");
#pragma unroll
  for (int i = 0; i < PF_IN; ++i) {
    const int u = gwarp + (i / kPages) * ngwarp;
    if (u < PROJW / P1_ROWS)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(
          base + (size_t)u * kUnitBytes + (size_t)(i % kPages) * 4096 +
          lane * 128));
  }
#else
  (void)base;
  (void)gwarp;
  (void)ngwarp;
  (void)lane;
#endif
}

__device__ __forceinline__ float4 load_w(const float4 *p) {
#if WSTREAM == 1
  return __ldcs(p);
#elif WSTREAM == 2
  return __ldcg(p);
#elif WSTREAM == 3
  return __ldg(p);
#elif WSTREAM == 4
  return __ldlu(p);
#else
  return *p;
#endif
}

// One warp streams R consecutive weight rows against the staged activation.
// Larger R raises the bytes a warp has outstanding at once and amortizes the
// per-row warp reduction and epilogue; the activation tile is read once per R.
template <bool FP8, int K, int R>
__device__ __forceinline__ void gemv_rows(const void *__restrict__ wbase,
                                          const void *__restrict__ sx, int lane,
                                          float (&acc)[R]) {
  // Row-outer / k-inner: the R*K weight bytes are then walked strictly
  // linearly, so the only pipeline bubble (warp reduction + epilogue) lands
  // once per R*K bytes instead of once per row.  Measured bandwidth tracks
  // exactly this contiguous burst length, which is why interleaving the rows
  // was slower.
  if (FP8) {
    const char *w = reinterpret_cast<const char *>(wbase);
    const char *x = reinterpret_cast<const char *>(sx);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float a = 0.0f;
#pragma unroll
      for (int j = 0; j < K / (32 * 16); ++j) {
        const int off = j * (32 * 16) + lane * 16;
        const float4 wv =
            load_w(reinterpret_cast<const float4 *>(w + (size_t)r * K + off));
        const float4 xv = *reinterpret_cast<const float4 *>(x + off);
        a += dot16_fp8(wv, xv);
      }
      acc[r] = a;
    }
  } else {
    const __nv_bfloat16 *w = reinterpret_cast<const __nv_bfloat16 *>(wbase);
    const __nv_bfloat16 *x = reinterpret_cast<const __nv_bfloat16 *>(sx);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float a = 0.0f;
#pragma unroll
      for (int j = 0; j < K / (32 * 8); ++j) {
        const int off = j * (32 * 8) + lane * 8;
        const float4 wv =
            load_w(reinterpret_cast<const float4 *>(w + (size_t)r * K + off));
        const float4 xv = *reinterpret_cast<const float4 *>(x + off);
        a += dot8_bf16(wv, xv);
      }
      acc[r] = a;
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
    acc[r] = warp_sum(acc[r]);
}

// The reference conv kernel writes `acc / (1 + exp(-acc))`; the gated norm
// writes `x * sigmoid(z)`.  Keep both spellings so the fp32 rounding matches
// each.
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

__device__ __forceinline__ float silu_mul(float v) {
  return v * (1.0f / (1.0f + expf(-v)));
}

__device__ __forceinline__ float softplus_ref(float v) {
  return v <= 20.0f ? logf(expf(v) + 1.0f) : v;
}

__device__ __forceinline__ float to_bf16f(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

// Phase timestamps from one representative CTA: ~5 instructions total, kept in
// the production path so a profile never needs a different binary.
__device__ __forceinline__ void phase_mark(float *sc, int slot, int tid) {
  if (blockIdx.x == 0 && tid == 0) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    reinterpret_cast<unsigned long long *>(sc + SC_TIME)[slot] = t;
  }
}

// ---------------------------------------------------------------- the kernel

template <bool IN_FP8, bool OUT_FP8, int NWARP>
__global__ void __launch_bounds__(NWARP * 32) mamba2_mega_kernel(MegaParams p) {
  extern __shared__ __align__(16) char smem[];
  constexpr int NTHREAD = NWARP * 32;
  using Sm = SmemPlan<IN_FP8, OUT_FP8>;
  char *const s_main = smem;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int gwarp = blockIdx.x * NWARP + warp;
  const int ngwarp = gridDim.x * NWARP;

  float *sc = p.scratch;
  const int slot = p.cache_idx[p.idx_offset];
  phase_mark(sc, 0, tid);

  // ---------------------------------------------------------------- P0
  {
    // Every CTA rebuilds the 4096-wide normalized activation so no barrier is
    // needed to publish it.  The pre-norm values stay in registers between the
    // square-sum pass and the scaling pass; re-reading hidden/residual instead
    // costs another 16 KB of L2 traffic per CTA (~9 MB over the grid).
    constexpr int NPV = (HSZ / 8) / NTHREAD;
    static_assert(NPV * NTHREAD * 8 == HSZ,
                  "P0 needs HSZ/8 divisible by the CTA size");
    float2 a[NPV][4];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < NPV; ++i) {
      const int v = tid + i * NTHREAD;
      const float4 hv = *reinterpret_cast<const float4 *>(p.hidden + v * 8);
      const __nv_bfloat162 *h2 = reinterpret_cast<const __nv_bfloat162 *>(&hv);
#pragma unroll
      for (int k = 0; k < 4; ++k)
        a[i][k] = __bfloat1622float2(h2[k]);
      if (p.residual != nullptr) {
        const float4 rv = *reinterpret_cast<const float4 *>(p.residual + v * 8);
        const __nv_bfloat162 *r2 =
            reinterpret_cast<const __nv_bfloat162 *>(&rv);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 t = __bfloat1622float2(r2[k]);
          a[i][k].x += t.x;
          a[i][k].y += t.y;
        }
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        ss = fmaf(a[i][k].x, a[i][k].x, ss);
        ss = fmaf(a[i][k].y, a[i][k].y, ss);
      }
      if (blockIdx.x == 0) {
        float4 out;
        __nv_bfloat162 *o2 = reinterpret_cast<__nv_bfloat162 *>(&out);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          o2[k] = __float22bfloat162_rn(a[i][k]);
        *reinterpret_cast<float4 *>(p.out_residual + v * 8) = out;
      }
    }
    // Issued here, not at entry: hidden and residual have been consumed by this
    // point, so the panel hint cannot queue ahead of the two cold loads P0 is
    // actually waiting on.
    prefetch_in<IN_FP8>(reinterpret_cast<const char *>(p.in_w), gwarp, ngwarp,
                        lane);
    ss = warp_sum(ss);
    float *red = reinterpret_cast<float *>(smem + Sm::kRed);
    if (lane == 0)
      red[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int i = 0; i < NWARP; ++i)
      tot += red[i];
    const float rstd = rsqrtf(tot * (1.0f / (float)HSZ) + p.eps);

#pragma unroll
    for (int i = 0; i < NPV; ++i) {
      const int v = tid + i * NTHREAD;
      const float4 wv = *reinterpret_cast<const float4 *>(p.norm_w + v * 8);
      const __nv_bfloat162 *w2 = reinterpret_cast<const __nv_bfloat162 *>(&wv);
      float q[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 wf = __bfloat1622float2(w2[k]);
        q[k * 2 + 0] = to_bf16f(a[i][k].x * rstd * wf.x);
        q[k * 2 + 1] = to_bf16f(a[i][k].y * rstd * wf.y);
      }
      if (IN_FP8) {
        const __nv_fp8x4_e4m3 lo(
            make_float4(q[0] * p.in_inv_scale, q[1] * p.in_inv_scale,
                        q[2] * p.in_inv_scale, q[3] * p.in_inv_scale));
        const __nv_fp8x4_e4m3 hi(
            make_float4(q[4] * p.in_inv_scale, q[5] * p.in_inv_scale,
                        q[6] * p.in_inv_scale, q[7] * p.in_inv_scale));
        uint2 packed;
        packed.x = (unsigned)lo.__x;
        packed.y = (unsigned)hi.__x;
        *reinterpret_cast<uint2 *>(s_main + v * 8) = packed;
      } else {
        float4 packed;
        __nv_bfloat162 *b2 = reinterpret_cast<__nv_bfloat162 *>(&packed);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          b2[k] = __floats2bfloat162_rn(q[k * 2 + 0], q[k * 2 + 1]);
        *reinterpret_cast<float4 *>(s_main + v * 16) = packed;
      }
    }
    if (blockIdx.x == 0 && tid < NGROUP)
      sc[SC_GSUM + tid] = 0.0f;
  }
  __syncthreads();
  phase_mark(sc, 1, tid);

  // ---------------------------------------------------------------- P1
  {
    constexpr int R = P1_ROWS;
    static_assert(PROJW % R == 0, "P1_ROWS must divide the projection width");
    const void *w_base = p.in_w;
    for (int u = gwarp; u < PROJW / R; u += ngwarp) {
      const int base = u * R;
      // Each of the first R lanes owns one row's epilogue.  Issuing the
      // committed-window and conv-weight loads here lets them overlap the GEMV.
      const int row = base + lane;
      const int c = row - ISZ;
      const bool mine = lane < R;
      const bool is_conv = mine && row >= ISZ && row < ISZ + CONVDIM;
      __nv_bfloat16 *st = p.conv_state + (size_t)slot * CONVDIM * (CONVW - 1) +
                          (size_t)(is_conv ? c : 0) * 3;
      __nv_bfloat16 s0{}, s1{}, s2{};
      float cb = 0.f;
      __nv_bfloat162 w01{}, w23{};
      if (is_conv) {
        s0 = st[0];
        s1 = st[1];
        s2 = st[2];
        cb = __bfloat162float(p.conv_b[c]);
        // conv_w rows are 8 bytes; only float2 (8B) alignment is guaranteed.
        const float2 cw4 =
            *reinterpret_cast<const float2 *>(p.conv_w + (size_t)c * 4);
        const __nv_bfloat162 *cw2 =
            reinterpret_cast<const __nv_bfloat162 *>(&cw4);
        w01 = cw2[0];
        w23 = cw2[1];
      }
      float acc[R];
      const int nxt = (u + ngwarp) * R;
      prefetch_slice(reinterpret_cast<const char *>(p.out_w),
                     (HSZ * ISZ * (OUT_FP8 ? 1 : 2)) >> 12, u, PROJW / R, lane);
      if (IN_FP8) {
        if (nxt < PROJW)
          prefetch_l2<R * HSZ>(
              reinterpret_cast<const char *>(w_base) + (size_t)nxt * HSZ, lane);
        gemv_rows<true, HSZ, R>(reinterpret_cast<const char *>(w_base) +
                                    (size_t)base * HSZ,
                                s_main, lane, acc);
#pragma unroll
        for (int r = 0; r < R; ++r)
          acc[r] *= p.in_alpha;
      } else {
        if (nxt < PROJW)
          prefetch_l2<R * 2 * HSZ>(reinterpret_cast<const char *>(w_base) +
                                       (size_t)nxt * 2 * HSZ,
                                   lane);
        gemv_rows<false, HSZ, R>(
            reinterpret_cast<const __nv_bfloat16 *>(w_base) +
                (size_t)base * HSZ,
            s_main, lane, acc);
      }
      if (mine) {
        float mine_acc = acc[0];
#pragma unroll
        for (int r = 1; r < R; ++r)
          if (lane == r)
            mine_acc = acc[r];
        const float v = to_bf16f(mine_acc);
#if NO_EPI
        sc[SC_GATE + (row & (ISZ - 1))] = v;
#else
        if (row < ISZ) {
          sc[SC_GATE + row] = v;
        } else if (is_conv) {
          // The reference Triton kernel multiplies the bf16 tap and weight in
          // bf16, then accumulates into an fp32 bias; reproduce that rounding.
          const __nv_bfloat16 vb = __float2bfloat16_rn(v);
          float o = cb;
          o += __bfloat162float(__hmul(s0, w01.x));
          o += __bfloat162float(__hmul(s1, w01.y));
          o += __bfloat162float(__hmul(s2, w23.x));
          o += __bfloat162float(__hmul(vb, w23.y));
          st[0] = s1;
          st[1] = s2;
          st[2] = vb;
          const float y = to_bf16f(silu(o));
          if (c < ISZ) {
            sc[SC_X + c] = y;
          } else if (c < ISZ + GSSZ) {
            sc[SC_B + (c - ISZ)] = y;
          } else {
            sc[SC_C + (c - ISZ - GSSZ)] = y;
          }
        } else {
          const int h = row - ISZ - CONVDIM;
          const float dt = softplus_ref(v + __bfloat162float(p.dt_bias[h]));
          sc[SC_DT + h] = dt;
          sc[SC_DA + h] = expf(p.A[h] * dt);
        }
#endif
      }
    }
  }

  phase_mark(sc, 2, tid);
#if MAX_PHASE <= 2
  return;
#endif
  constexpr int kOutPages = (HSZ * ISZ * (OUT_FP8 ? 1 : 2)) >> 12;
  constexpr int kOutPPR =
      (OUT_FP8 ? ISZ : 2 * ISZ) >> 12; // pages per row: 2 or 4
  // A fraction of the panel, so the two precisions need separate caps: the same
  // sixteenth is twice as many pages on a BF16 panel.
  constexpr int kOutCov = OUT_FP8 ? PF_OUT_COV : PF_OUT_COV_BF16;
  if constexpr (!IN_FP8 || PF_OUT_FP8)
    prefetch_pages<kOutPPR, kOutCov>(reinterpret_cast<const char *>(p.out_w),
                                     kOutPages, gwarp, ngwarp, 0, lane);
  cg::this_grid().sync();
  phase_mark(sc, 3, tid);

  // ---------------------------------------------------------------- P2
  {
    float *sg = reinterpret_cast<float *>(smem + Sm::kGsum);
    if (tid < NGROUP)
      sg[tid] = 0.0f;
    __syncthreads();
#if P2_STAGE
    float *sB = reinterpret_cast<float *>(s_main);
    float *sC = sB + GSSZ;
    float *sD = sB + 2 * GSSZ;
    for (int i = tid; i < 2 * GSSZ; i += NTHREAD)
      sB[i] = sc[SC_B + i];
    for (int i = tid; i < NHEAD; i += NTHREAD)
      sD[i] = __bfloat162float(p.D[i]);
#else
    // Read B/C straight out of the scratch buffer P1 just wrote.  Staging them
    // costs 8.5 KB of L2 traffic per CTA -- 5.6 MB over a 658-CTA grid -- to
    // serve reads that add up to 8192 rows x 32 B = 256 KB taken directly, and
    // every one of those is an L2 hit on a line P1 wrote moments earlier.  A
    // warp handles one or two rows here, so there is no reuse for smem to buy.
    const float *sB = sc + SC_B;
    const float *sC = sc + SC_C;
    static_assert(SC_C == SC_B + GSSZ, "P2 indexes C as B + GSSZ");
#endif
#if PF_OUT_EARLY
    // The second window's hints, issued at the *top* of P2 instead of at its
    // bottom.  Two things are wrong with the bottom, and this fixes both:
    //
    // - DRAM is idle here.  P2 moves only the 8.4 MB of committed state and
    // runs
    //   it at 874-1074 GB/s on the BF16-out rows, well under the ~1500 the same
    //   DRAM delivers in P1, because a warp's single float4 makes the phase
    //   latency- rather than bandwidth-bound.  profiles/04 retired "find the
    //   idle phase and prefetch into it" for P0, where the gap was the
    //   harness's own writeback already in flight; P2's gap is a different
    //   animal and the hint has 4-10 us of real slack to be serviced in.
    // - A warp-wide `prefetch` is 32 independent 128 B requests, so one sweep
    //   queues ~168k of them.  profiles/06 measured what that costs when the
    //   next instruction is a grid barrier: trimming the second window took
    //   bar2 from 5.9 to 2.1 us on the FP8-out rows.  The BF16-out rows never
    //   had theirs trimmed and still pay 6.1 (m1-l22) and 7.8 us (m1-l6).
    //   Moving the sweep ahead of P2's body lets it drain under the state
    //   recurrence rather than under the barrier.
    if constexpr (!IN_FP8 || PF_OUT_FP8)
      prefetch_pages<kOutPPR, kOutCov>(reinterpret_cast<const char *>(p.out_w),
                                       kOutPages, gwarp, ngwarp, 1, lane);
#endif

    const size_t sbase = (size_t)slot * NHEAD * HDIM * DSTATE;
    for (int r = gwarp; r < NHEAD * HDIM; r += ngwarp) {
      const int h = r >> 6;
      const int g = h / HPG;
#if GNORM_P2
      // Issued before the state load so the 2-byte fetch hides behind it.
      const float gwr = __bfloat162float(p.gnorm_w[r]);
#endif
      const float dt = sc[SC_DT + h];
      const float dA = sc[SC_DA + h];
      const float x = sc[SC_X + r];
      const float dtx = dt * x;
      float *st = p.ssm_state + sbase + (size_t)r * DSTATE;
      float4 sv = *reinterpret_cast<const float4 *>(st + lane * 4);
      const float4 bv =
          *reinterpret_cast<const float4 *>(sB + g * DSTATE + lane * 4);
      const float4 cv =
          *reinterpret_cast<const float4 *>(sC + g * DSTATE + lane * 4);
      sv.x = fmaf(sv.x, dA, dtx * bv.x);
      sv.y = fmaf(sv.y, dA, dtx * bv.y);
      sv.z = fmaf(sv.z, dA, dtx * bv.z);
      sv.w = fmaf(sv.w, dA, dtx * bv.w);
      *reinterpret_cast<float4 *>(st + lane * 4) = sv;
      float o = sv.x * cv.x;
      o = fmaf(sv.y, cv.y, o);
      o = fmaf(sv.z, cv.z, o);
      o = fmaf(sv.w, cv.w, o);
      o = warp_sum(o);
      if (lane == 0) {
#if P2_STAGE
        const float dval = sD[h];
#else
        const float dval = __bfloat162float(p.D[h]);
#endif
        // `preallocated_ssm_out` is a bf16 buffer: the recurrence output is
        // rounded before the gated norm consumes it.
        o = to_bf16f(fmaf(dval, x, o));
        const float yp = o * silu_mul(sc[SC_GATE + r]);
#if GNORM_P2
        sc[SC_YPRE + r] = yp * gwr;
#else
        sc[SC_YPRE + r] = yp;
#endif
        // The RMS denominator is the mean square of y_pre *before* the
        // gated-norm weight, so this stays on the unfolded value either way.
        atomicAdd(&sg[g], yp * yp);
      }
    }
    __syncthreads();
    if (tid < NGROUP && sg[tid] != 0.0f)
      atomicAdd(&sc[SC_GSUM + tid], sg[tid]);
  }

  phase_mark(sc, 4, tid);
#if MAX_PHASE <= 4
  return;
#endif
#if !PF_OUT_EARLY
  if constexpr (!IN_FP8 || PF_OUT_FP8)
    prefetch_pages<kOutPPR, kOutCov>(reinterpret_cast<const char *>(p.out_w),
                                     kOutPages, gwarp, ngwarp, 1, lane);
#endif
  cg::this_grid().sync();
  phase_mark(sc, 5, tid);

  // ---------------------------------------------------------------- P3
  // P3 has only HSZ=4096 out_proj rows to hand to ngwarp=5264 warps, so the
  // last 146 CTAs own no row at all -- and every one of them was still staging
  // the full gated-norm activation (32 KB of y_pre plus 16 KB of gnorm_w per
  // CTA) for a GEMV it never runs.  No barrier follows, so they can simply
  // leave.
#if P3_TRIM
  if (blockIdx.x * NWARP >= HSZ / P3_ROWS)
    return;
#endif
  {
    // Hint this warp's own out_proj row before the gated-norm staging rather
    // than after it.  Now that PF_OUT_COV leaves a quarter of the panel
    // unhinted, the warps that own those rows are the only ones that still have
    // to reach DRAM, and the staging loop below is ~3.4 us of pure L2 traffic
    // (32 KB of y_pre per CTA, 512 CTAs) that their fetch can hide behind.  The
    // GEMV's existing `prefetch_l2` cannot do this: it targets iteration u +
    // ngwarp, and with ngwarp = 5264 > HSZ = 4096 there is never a second
    // iteration, so at P3_ROWS=1 that hint is dead code.
    constexpr int kPfP3 = (OUT_FP8 && !PF_P3_FP8) ? 0 : PF_P3;
    if constexpr (kPfP3 > 0) {
      if (gwarp < HSZ / P3_ROWS) {
        const size_t row_bytes = (size_t)P3_ROWS * (OUT_FP8 ? ISZ : 2 * ISZ);
        const char *mine =
            reinterpret_cast<const char *>(p.out_w) + (size_t)gwarp * row_bytes;
#pragma unroll
        for (int i = 0; i < kPfP3; ++i)
          asm volatile("prefetch.global.L2 [%0];" ::"l"(
              mine + (size_t)i * 4096 + lane * 128));
      }
    }
    float *rstd = reinterpret_cast<float *>(smem + Sm::kRed);
    if (tid < NGROUP)
      rstd[tid] = rsqrtf(sc[SC_GSUM + tid] * (1.0f / (float)GNORM) + p.eps);
    __syncthreads();
    if (OUT_FP8) {
      for (int v = tid; v < ISZ / 4; v += NTHREAD) {
        const int base = v * 4;
        const float r = rstd[base / GNORM];
        float q[4];
#if GNORM_P2
#pragma unroll
        for (int i = 0; i < 4; ++i)
          q[i] = to_bf16f(sc[SC_YPRE + base + i] * r) * p.out_inv_scale;
#else
#pragma unroll
        for (int i = 0; i < 4; ++i)
          q[i] = to_bf16f(sc[SC_YPRE + base + i] * r *
                          __bfloat162float(p.gnorm_w[base + i])) *
                 p.out_inv_scale;
#endif
        reinterpret_cast<__nv_fp8x4_e4m3 *>(s_main)[v] =
            __nv_fp8x4_e4m3(make_float4(q[0], q[1], q[2], q[3]));
      }
    } else {
      __nv_bfloat16 *yb = reinterpret_cast<__nv_bfloat16 *>(s_main);
      for (int v = tid; v < ISZ / 8; v += NTHREAD) {
        const int base = v * 8;
        const float r = rstd[base / GNORM];
        float4 packed;
        __nv_bfloat162 *b2 = reinterpret_cast<__nv_bfloat162 *>(&packed);
#if GNORM_P2
#pragma unroll
        for (int i = 0; i < 4; ++i)
          b2[i] = __floats2bfloat162_rn(sc[SC_YPRE + base + i * 2 + 0] * r,
                                        sc[SC_YPRE + base + i * 2 + 1] * r);
#else
        const float4 gw = *reinterpret_cast<const float4 *>(p.gnorm_w + base);
        const __nv_bfloat162 *g2 =
            reinterpret_cast<const __nv_bfloat162 *>(&gw);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 gf = __bfloat1622float2(g2[i]);
          b2[i] =
              __floats2bfloat162_rn(sc[SC_YPRE + base + i * 2 + 0] * r * gf.x,
                                    sc[SC_YPRE + base + i * 2 + 1] * r * gf.y);
        }
#endif
        *reinterpret_cast<float4 *>(yb + base) = packed;
      }
    }
    __syncthreads();
    constexpr int R3 = P3_ROWS;
    static_assert(HSZ % R3 == 0, "P3_ROWS must divide the hidden size");
    for (int u = gwarp; u < HSZ / R3; u += ngwarp) {
      const int base = u * R3;
      float acc[R3];
      const int nxt = (u + ngwarp) * R3;
      if (OUT_FP8) {
        if (nxt < HSZ)
          prefetch_l2<R3 * ISZ>(reinterpret_cast<const char *>(p.out_w) +
                                    (size_t)nxt * ISZ,
                                lane);
        gemv_rows<true, ISZ, R3>(reinterpret_cast<const char *>(p.out_w) +
                                     (size_t)base * ISZ,
                                 s_main, lane, acc);
#pragma unroll
        for (int r = 0; r < R3; ++r)
          acc[r] *= p.out_alpha;
      } else {
        if (nxt < HSZ)
          prefetch_l2<R3 * 2 * ISZ>(reinterpret_cast<const char *>(p.out_w) +
                                        (size_t)nxt * 2 * ISZ,
                                    lane);
        gemv_rows<false, ISZ, R3>(
            reinterpret_cast<const __nv_bfloat16 *>(p.out_w) +
                (size_t)base * ISZ,
            s_main, lane, acc);
      }
      if (lane < R3) {
        float mine = acc[0];
#pragma unroll
        for (int r = 1; r < R3; ++r)
          if (lane == r)
            mine = acc[r];
        p.out_hidden[base + lane] = __float2bfloat16_rn(mine);
      }
    }
  }
  phase_mark(sc, 6, tid);
}

// ---------------------------------------------------------------- host launch

static const void *pick_kernel(int in_fp8, int out_fp8) {
  if (in_fp8 && out_fp8)
    return (const void *)mamba2_mega_kernel<true, true, NWARPS>;
  if (in_fp8)
    return (const void *)mamba2_mega_kernel<true, false, NWARPS>;
  if (out_fp8)
    return (const void *)mamba2_mega_kernel<false, true, NWARPS>;
  return (const void *)mamba2_mega_kernel<false, false, NWARPS>;
}

static int pick_smem(int in_fp8, int out_fp8) {
  if (in_fp8 && out_fp8)
    return SmemPlan<true, true>::kTotal;
  if (in_fp8)
    return SmemPlan<true, false>::kTotal;
  if (out_fp8)
    return SmemPlan<false, true>::kTotal;
  return SmemPlan<false, false>::kTotal;
}

extern "C" {

int mamba2_mega_launch(MegaParams *p, int in_fp8, int out_fp8, int grid,
                       cudaStream_t stream) {
  void *args[] = {p};
  return (int)cudaLaunchCooperativeKernel(pick_kernel(in_fp8, out_fp8),
                                          dim3(grid), dim3(NWARPS * 32), args,
                                          pick_smem(in_fp8, out_fp8), stream);
}

int mamba2_mega_config(int in_fp8, int out_fp8, int *max_blocks, int *regs,
                       int *threads, int *smem, int *scratch_floats) {
  const void *fn = pick_kernel(in_fp8, out_fp8);
  int occ = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, fn, NWARPS * 32, pick_smem(in_fp8, out_fp8));
  if (err != cudaSuccess)
    return (int)err;
  cudaFuncAttributes attr{};
  cudaFuncGetAttributes(&attr, fn);
  *max_blocks = occ;
  *regs = attr.numRegs;
  *threads = NWARPS * 32;
  *smem = pick_smem(in_fp8, out_fp8);
  *scratch_floats = SC_TOTAL;
  return 0;
}
}
