# SPDX-License-Identifier: Apache-2.0
"""GDN ReplaySSM MTP decode step (1 <= T <= 8) on the UMMA / TMEM design: Rubin (sm_107a) build.

A separate copy of ``gdn_replay_mtp_umma.py`` tuned on Rubin (sm_107a), public entry
``gated_delta_rule_mtp_ucache_flush_umma_rubin``; same tensors, ring / flush contract and
results (bit-identical outputs, state and rings in round-to-nearest; the stochastic-rounding
builds draw the same noise per element). Validated on sm_107a and sm_100a; other archs raise.
Differences from the base kernel:

* named barriers 1-3 only (Rubin has 16 per SM, which capped the base kernel at 2 CTAs / SM),
  non-``.aligned`` syncs where a barrier has more than one sync site, and a proxy fence before
  the D-build buffer release;
* T >= 5 verify rows read their v rows through SMEM (``vsmem``) and split the warpgroup
  registers before any global load (``kernel_early``);
* less code per CTA against SM instruction-cache thrash when flush and verify rows share an SM:
  the math warps' per-head code as a rolled loop at T <= 4, and the flush RMW as loops
  (stochastic rounding everywhere, round to nearest on sm_107a at most T, with packed
  fp16 x fp32 FMAs);
* a persistent kernel (``kernel_persist``, sm_107a; chosen automatically at T = 2-8 when
  ``num_sms`` is below the device's SM count, e.g. one die's green context, or forced with
  ``persistent``), with
  ``num_sms`` / ``grid_override`` / ``persistent_counter`` / ``flush_hint``;
* on sm_107a the automatic heads per CTA of non-persistent launches also depend on T,
  ``flush_hint``, the rounding mode and whether the launch covers the whole device
  (``auto_heads_per_cta``: all value heads of a key head per CTA from 1.5-3.6 waves of CTA
  slots with flush rows and from 0.8-1 without, not 8, except Philox above 6 rounds at T <= 4
  in random order on a partial-device launch);
* the fp32 copies of bf16 ``A_log`` / ``dt_bias`` are safe across streams and refreshed in place
  after an in-place update (same address, so graphs captured earlier read the new values).

Same contract as the base kernel (fp16-state arm: bf16 q/k/v/a/b and
outputs, fp16 state pool, bf16 k ring, bf16 or fp16 u ring, fp32 g ring), same 32-slot
ring contract; an fp16 k ring is also taken (see ``k_f16`` below):

* live window ``[cache_base, cache_base + hist_len) mod 32``, P = hist_len <= 16,
  g ring = window-relative cumulative log-decay, ``w_j = e^{G_P - g_j}``;
* every row appends (k̂_s, u_s, g) at ``(base + P + s) & 31``; g restarts at 0 on flush rows;
* rows with ``P >= flush_min`` also write ``S0 <- e^{G_P} S0 + sum_{j<P} w_j u_j k̂_j^T``
  (the T current tokens are not folded);
* padded rows (state index < 0) exit and write nothing; cursor commits are host-side;
* optional stochastic rounding of the flushed fp16 state (``stochastic_rounding``), off by
  default (round-to-nearest). "philox": FlashInfer's Mamba scheme, one Philox4x32 draw (key =
  seed, counter = flat state-pool offset of the 8-element chunk) feeds 4 ``cvt.rs.f16x2``
  pairs. "lcg": one LCG step per pair (s ^ (s >> 16) dither) from a hashed per-row seed
  (``rand_seed`` or %clock).

Math per (request, value head) with ``X = [k̂_0..k̂_{T-1}; s q̂_0..s q̂_{T-1}]``:

    S_h x = e^{G_P} S0 x + sum_j (w_j u_j)(k̂_j . x)                  (x in X)
    u_t   = b_t v_t - b_t e^{g_t} S_h k̂_t - sum_{s<t} b_t e^{g_t-g_s}(k̂_s.k̂_t) u_s
    y_t   = e^{g_t} S_h q̂_t + sum_{s<=t} e^{g_t-g_s}(k̂_s.q̂_t) u_s

CTA = (key head, request): the key head's value heads in sequence (all HV / H of them, 1 to
4, or a slice of 1 / 2 for small batches, ``heads_per_cta``); grid (H x (HV / H) /
heads_per_cta, B), key head fastest; 4 CTAs / SM
(SMEM ~56 KB, 80 registers, 128 TMEM columns); 192 threads in three roles -- 256 with the
register split of the builds that would otherwise spill (``_use_regsplit``), where the
math warps are warps 4-7 and warps 2-3 idle (setmaxnreg 24 / 104):
  warp 0   : TMA producer — each value head's fp16 state as two 16 KB 128B-swizzled K halves
             through two stages; per-head u windows bulk-copied into a two-head SMEM double
             buffer (T >= 5 verify rows: with the head's v rows in the window's never-live
             slots); flush rows TMA-store the folded halves.
  warp 1   : allocates TMEM at CTA entry (the next CTA on the SM is not launched before the
             allocation permit is relinquished), only UMMA issuer.
  warps 2-5: the math warps ("consumers", warps 4-7 with the register split): 128 lanes =
             128 V rows (TMEM lane = row); prologue + per-row math. Warps cw0 / cw0 + 1 are
             the "ring-row" warps (Gram rows of the k̂ ring), cw0 + 2 / cw0 + 3 the "X-tile"
             warps (X x X Gram, X tile, gates, coefficient tables).
UMMAs (kind::f16, fp32 accumulate in TMEM):
  readout  R = S0 . X^T            M128 N16 K128  A: state stage   B: X tile (K-major)
  history  Y = D . C^T             M128 N16 K16   A: D (TMEM)      B: C tile
  fold     F = D . K_hist (halves) M128 N64 K16   A: D (TMEM)      B: K_hist tile, MN-major
  with D[v, j] = w_j u_j[v], C[x, j] = k̂_j . x, K_hist[j] = k̂_j (j < P).
The per-CTA Gram [k̂_ring; X] . X^T (all dots + the norms) runs on mma.sync with fragments
loaded straight from global memory.  The intra-step causal solve is a T-step forward
substitution per V row on CUDA cores.  Flush rows issue fold half 0 before the state
lands, so each half's RMW follows its readout and the epilogue overlaps the next head's
state load.

Persistent variant (sm_107a only, any T, all 4 value heads of a key head per CTA; automatic at
T = 2-8 on die-localized launches): ``kernel_persist``, at most 4 CTAs per SM of the launch's
stream that loop over the (key head, row) items -- verify, flush and padded rows in any mix --
with ``kernel_early``'s per-row steps; warp 2 is its scheduler (see the comment above
``call_persist``). It only measured faster on die-localized launches (see
``gated_delta_rule_mtp_ucache_flush_umma_rubin``).

Specialised for K == V == 128, HV % H == 0, HV // H == 4 (Qwen3.5 GDN geometry).

fp16 k ring (``k_f16``, chosen by the k ring's dtype; the bf16 build is unchanged): the appended
k̂_t is the X tile's own fp16 value (the k̂ the readout and so u_t used), the fold's K_hist tile
copies the ring as is, and the ring rows of the Gram run f16 mma.sync against the fp16 X̂ rows,
so C[x, j] = k̂_j . x̂ reads the same k̂_j the fold folds and the same x̂ the readout reads (the
bf16 build rounds k̂_j to bf16 in C and in the fold but keeps it fp16 in u_t).
"""

from __future__ import annotations

import functools
import math
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.nvgpu import cpasync

try:
    from . import _umma_helpers_rubin as U
    from .device_target import gdn_compile_options, gdn_device_target
except (
    ImportError
):  # tests and benchmarks may load this file by path, outside the package
    from flashinfer.gdn_kernels import _umma_helpers_rubin as U
    from flashinfer.gdn_kernels.device_target import (
        gdn_compile_options,
        gdn_device_target,
    )

RING_SLOTS = 32
RING_MASK = RING_SLOTS - 1
W_RING = 16
K_DIM = 128
V_DIM = 128
HPK = 4  # most value heads a CTA runs: per-head SMEM / TMEM / barrier arrays are sized for 4
# The value heads per key head, HV / H, is 1..4 per kernel instance (``self.hpk``); a CTA
# runs ``heads_per_cta`` of them (a divisor of HV / H) with the per-head arrays sized for 4.
_EPS = 1e-6
_TMEM_COLS = 128
_COEF = 192  # fp32 coefficient words per head (see _CO_*)
# per-head coefficient table offsets (words): intra-step solve A[t][s] / output Bq[t][s]
# (8 x 8 each), beta_t, beta_t e^{g_t}, e^{g_t}, g_t, window weights w_j (16), e^{G_P}
_CO_A, _CO_BQ, _CO_BETA, _CO_BG, _CO_EG, _CO_GAM, _CO_W, _CO_EGP = (
    0,
    64,
    128,
    136,
    144,
    152,
    160,
    176,
)
_MIN_BLOCKS_PER_SM = (
    4  # 128 of the 512 TMEM columns, ~56 KB SMEM and 80 registers per CTA
)
# Register split (setmaxnreg dec / inc): 256 threads, warps 0-3 (TMA producer, UMMA issuer,
# two idle warps -- the persistent kernel's scheduler on warp 2) at 24 registers, the math
# warps 4-7 (one warpgroup) at 104; the 64-register launch budget keeps 4 CTAs / SM. The two
# extra warps cost 1-2 % where nothing spills, so builds take it only where the 192-thread,
# 80-register layout spills on the math warps' path (``_use_regsplit``). The 192-thread layout
# itself cannot split: setmaxnreg is warpgroup-wide and a 2-warp warpgroup deadlocks.
_REGSPLIT = (24, 104)
_FHFMA2_CC = (
    (10, 7),
)  # archs whose ptxas has fma.rn.f32x2.f16x2.f32x2.f32x2 (sm_107a)
# Round-to-nearest builds that run the flush RMW as the stochastic-rounding builds' loop (with
# FHFMA2, so sm_107a only), 4 heads per CTA: the T where it measured faster (see rmw_loop).
_RN_RMW_LOOP_T = (1, 2, 3, 5, 6, 8)
# Persistent kernel: mailbox words per entry (6 used), mailbox readers (warps 0, 1 and the four
# math warps: one arrive each), the per-CTA init barrier (warps 0-2 + the math warps) and
# barrier 3 per row (the X-tile warps' arrive + the ring-row warps' sync)
_PZ_RI = 8
_PZ_READERS = 6
_PZ_INIT = 224
_PZ_B3 = 128
# archs the persistent kernel is enabled on: sm_107a only (spill-free at T = 2-8 in every
# rounding mode, tested and measured there). It compiles spill-free for sm_100a (B200) too, but
# has never run there, and its speed-up comes from die-localized launches, which B200 lacks.
_PZ_CC = ((10, 7),)
# Automatic choice (persistent=None, the default): the persistent kernel where it measured faster
# -- T in _PZ_AUTO_T (at T = 1 the prologue reorder already starts the state stream early), all
# 4 value heads per CTA, a die-localized launch (``num_sms`` below the device's SMs; one
# whole-GPU launch over die-interleaved memory measured slower) and at least
# _PZ_AUTO_MIN_ITEMS[hint] (key head, row) items per CTA slot (H x B / (4 x num_sms)).
_PZ_AUTO_T = (2, 3, 4, 5, 6, 7, 8)
# flush_hint "none": by T, where the persistent kernel overtook the non-persistent one at 4 heads
# per CTA (B >= 53 x items per 106-SM die: 186 / 128 / 128 / 128 / 265 / 292 / 292 at T = 2-8);
# with flush rows possible (None / "some" / "flush_first") from 8 items, where it overtook with
# 30 % flush rows. Measurements: PR #6791.
_PZ_AUTO_MIN_ITEMS: dict[str, float | dict[int, float]] = {
    "none": {2: 3.5, 3: 2.4, 4: 2.4, 5: 2.4, 6: 5.0, 7: 5.5, 8: 5.5},
    "some": 8,
}


def _use_regsplit(T: int, hpc: int, sr: str) -> bool:
    """Builds that take the register split (_REGSPLIT): the ones whose 192-thread layout spills
    on the math warps' path (sm_100a / sm_107a, static STL / LDL). Round to nearest: 4 heads
    per CTA at T >= 5 (T = 8: 20 STL, +8 % kernel time), 3 heads at T = 5 / 6 (the split 1-8 %
    faster; at T = 7 / 8 it spills lightly and the split measured up to 2.5 % slower).
    Stochastic rounding (RMW loop): Philox at T >= 5 with 4 heads and at T = 8 with 2; never
    LCG, whose 64-register prologue spills in the split."""
    if sr != "":
        return sr == "philox" and ((hpc == HPK and T >= 5) or (hpc == 2 and T == 8))
    if hpc == 3:
        return T in (5, 6)
    return hpc == HPK and T >= 5


def _persist_ok(T: int, hpk: int, hpc: int, sr: str) -> bool:
    """Persistent kernel scope: 1 <= T <= 8, 4 value heads per key head and per CTA;
    round-to-nearest and every stochastic-rounding mode (the persistent build always takes the
    register split; at T <= 4 it uses the T >= 5 row structure: v rows through SMEM on verify rows,
    the D-build masks, no prologue reorder)."""
    return (
        1 <= T <= 8
        and hpk == HPK
        and hpc == HPK
        and sr in ("", "philox", "lcg", "lcg_clock")
    )


def _sel(cond, a, b, ty):
    """Branch-free typed select (arith.select) for use outside AST-preprocessed scopes."""
    return ty(cutlass.select_(cond, ty(a), ty(b)))


class GdnReplayMtpUmmaRubin:
    def __init__(
        self,
        T: int,
        h: int,
        hv: int,
        u_bf16: bool = True,
        use_row_order: bool = False,
        hpc: Optional[int] = None,
        sr: str = "",
        sr_rounds: int = 0,
        mixed_fma: bool = False,
        persistent: bool = False,
        h_io: Optional[int] = None,
        hv_io: Optional[int] = None,
        min_blocks: int = _MIN_BLOCKS_PER_SM,
        small_b: bool = False,
        parts: int = 1,
        k_f16: bool = False,
        qk_io: Optional[int] = None,
        v_io: Optional[int] = None,
        ab_dyn: bool = False,
        pdl: bool = False,
        ring_rows: bool = False,
        fused_conv: bool = False,
        fused_norm: bool = False,
    ):
        assert 1 <= T <= 8
        assert hv % h == 0 and 1 <= hv // h <= HPK, (h, hv)
        self.hpk = hv // h  # value heads per key head
        # hpc = value heads per CTA (hpk: one CTA per key head; a smaller divisor of hpk splits
        # a key head over hpk / hpc CTAs, which repeat the per-key-head prologue but shorten
        # the serial head chain)
        hpc = self.hpk if hpc is None else int(hpc)
        assert 1 <= hpc <= self.hpk and self.hpk % hpc == 0, (hpc, self.hpk)
        self.hpc = hpc
        self.nsub = self.hpk // hpc
        self.T = T
        self.h = h
        self.hv = hv
        # Head counts of the q / k and v / a / b / out tensors' token rows (their strides) and of
        # the logical state pool (the stochastic-rounding ids): above h / hv when the launch runs
        # a slice of the heads (uGPU head split: the inputs are head-slice views of the full
        # tensors, the state and rings the slice's own part, hv_base its first value head).
        self.h_io = h if h_io is None else h_io
        self.hv_io = hv if hv_io is None else hv_io
        assert self.h_io >= h and self.hv_io >= hv and self.hv_io * h == self.h_io * hv
        # parts > 1: the state and rings are uGPU buffers [pool, parts, heads / parts, ...]
        # (ugpu_ucache_buffers) and this one launch runs every part's heads: the kernel views
        # each buffer's (part, head) modes as one head mode of shape (heads / parts, parts), so
        # head j addresses part j // (heads / parts) -- the same bytes as the per-part launches
        # (non-persistent kernel only)
        self.parts = int(parts)
        assert self.parts >= 1 and h % self.parts == 0 and hv % self.parts == 0
        assert self.parts == 1 or (not persistent and self.h_io == h)
        # u ring dtype (bf16 / fp16): read and appended through dtype-aware conversions.
        # k ring dtype: bf16, or fp16 with k_f16 (_ring_gram, _k_append_words, _kt_words).
        self.u_bf16 = u_bf16
        self.k_f16 = bool(k_f16)
        self.use_row_order = use_row_order
        # stochastic rounding of the flush store: "" (round-to-nearest), "philox" (sr_rounds
        # rounds), "lcg" (seeded by rand_seed) or "lcg_clock" (seeded by %clock)
        assert sr in ("", "philox", "lcg", "lcg_clock")
        self.sr = sr
        self.sr_rounds = sr_rounds if sr == "philox" else 0
        self.sr_lcg = sr in ("lcg", "lcg_clock")
        # the persistent kernel always splits (warp 2 is its scheduler; the math warps run at 104
        # registers from entry, so LCG's 64-register prologue spills do not arise there)
        self.split = persistent or _use_regsplit(T, hpc, self.sr)  # 256-thread layout
        self.regsplit = _REGSPLIT  # (dec, inc) register counts, used when self.split
        self.cw0 = 4 if self.split else 2  # first consumer warp
        self.num_threads = 256 if self.split else 192
        # Row structure, by T (each measured faster only on its side of T: Rubin, localized,
        # B = 512):
        # * T <= 4: the prologue reorder (``pro_reorder``): entry loads first, cursors last,
        #   warp 0's mbarrier inits around head 0's state (T = 1: 80.7 -> 79.3 us).
        # * T >= 5 (and the persistent kernel at every T): v rows through SMEM (``vsmem``) and
        #   the D-build masks (``d_mask``). vsmem: a verify row has P < flush_min <= W_RING - T
        #   + 1 live window slots, so slots wv = W_RING - T .. 15 of its u-window buffer are
        #   never read as u; warp 0 bulk-copies the head's T v rows there, with its u window and
        #   on the same u_full mbarrier, and the epilogue reads v from SMEM instead of global
        #   memory (at T = 8 the consumer chain waited on the global v loads). The buffer then
        #   stays in use until the head's epilogue has read its v rows, so u_free is arrived
        #   there and head h + 2's window goes behind its own state's second half (see
        #   _u_window_head). Flush rows keep 16-slot windows and v from global memory.
        # * small_b (the build options of _SMALL_B_BUILD_CC, see compile): the prologue reorder at
        #   T >= 5 too with 1 head per CTA (latency-bound CTAs, B <= 13 per die: T = 6 / 8
        #   1.03-1.15x vs vsmem).
        self.small_b = small_b
        self.pro_reorder = (T <= 4 or (small_b and hpc == 1)) and not persistent
        self.vsmem = (T >= 5 or persistent) and not self.pro_reorder
        self.wv = W_RING - T  # u slots copied on verify rows (vsmem)
        # D build: clear the slots j >= P with uniform word masks instead of per-value selects
        # (_d_build). Not in the 2- and 3-head builds (192 threads, 80 registers at their limit),
        # where the masks' uniform registers pushed ptxas into spills.
        self.d_mask = self.vsmem and hpc in (1, HPK)
        # Less code per CTA against SM instruction-cache thrash (flush and verify CTAs share an
        # SM; the unrolled T = 1, 4-head kernel is ~93 KB):
        # * ``roll``: the math warps' per-head code (D build, waits, epilogue) as a rolled loop
        #   over the heads (dynamic head index; the mbarriers of each kind in one array) at
        #   T <= 4 (Rubin, T = 1, 20 % flush: ICC miss 17.5 -> 11.9 %; T = 4 1.02-1.03x). Not
        #   at T >= 5: the register-split builds' 64-register prologue spills with it.
        # * ``rmw_loop``: the flush RMW as dynamic loops over (half, column group), one copy of
        #   its body per CTA (``roll_f``) or per head. Every stochastic-rounding build (its
        #   unrolled RMW is ~2x the round-to-nearest code: T = 1 / 4 / 8, 30 % flush, 1.33-1.46x
        #   with the loop; the RNG itself was not the cost), the round-to-nearest builds with 4
        #   heads per CTA at _RN_RMW_LOOP_T on sm_107a (20-30 % flush 1.02-1.08x; at T = 4 / 7
        #   the verify-only step measured 1-4 % slower) and the persistent kernel. Same bits.
        #   small_b: also the round-to-nearest builds with 1 or 2 heads per CTA at every T (few
        #   CTAs: the shorter flush code wins over the unrolled RMW, flush rows 1.01-1.09x).
        # * small_b: no ``roll`` in the round-to-nearest builds with 2 heads per CTA (few CTAs
        #   per SM, so less i-cache pressure than the rolled loop's dynamic head index costs:
        #   up to 1.06x unrolled at T <= 4; the stochastic-rounding builds measured 0.96-1.02x,
        #   so they keep it).
        self.rmw_loop = (
            persistent
            or sr != ""
            or (mixed_fma and self.hpc == HPK and T in _RN_RMW_LOOP_T)
            or (mixed_fma and small_b and self.hpc < HPK)
        )
        self.roll = (
            T <= 4 and self.hpc > 1 and not (small_b and self.hpc == 2 and sr == "")
        )
        # the flush head loop rolled as well in the register-split RMW-loop builds with 4 heads
        # per CTA (T >= 5: Philox and round to nearest, 30 % flush 1.03-1.07x vs 1.00-1.03x with
        # one RMW loop per head); not in the 80-register LCG builds (spills)
        self.roll_f = self.roll or (self.rmw_loop and self.split and self.hpc == HPK)
        # the RMW's e^{G_P} S0 + F as packed fp16 x fp32 FMAs (FHFMA2, sm_107a), in the RMW loop
        # only (bit-identical to the fp32 FMAs, see U.ffma_f16x2_f32; without it the RN loop
        # spills 2-4 STL at T >= 4)
        self.mixed_fma = mixed_fma and self.rmw_loop
        # RMW TMEM columns per load (and Philox streams per call = its 8-element chunks): 32 for
        # RN and for Philox with 4 heads per CTA (many CTAs, issue throughput); 16 for Philox
        # with fewer heads per CTA (few CTAs: the RMW latency is the kernel time and the
        # smaller live set halves it) and for LCG always (spills at 32)
        self.ldn = 16 if (self.sr_rounds > 0 and self.hpc < HPK) or self.sr_lcg else 32
        self.ngrp = 64 // self.ldn
        # vsmem's prologue copies are in the default (not pro_reorder) prologue order
        assert not (self.vsmem and self.pro_reorder)
        # early_split: the register-split builds with 4 heads per CTA (T >= 5) run kernel_early:
        # the warpgroup register split before any global load, so the math warps issue their
        # entry loads at 104 registers in the prologue reorder's order and run its consumer
        # structure (x_ready not held to the ring rows)
        self.early_split = self.split and self.hpc == HPK and not self.pro_reorder
        # persistent: kernel_persist, see call_persist
        self.persistent = persistent
        # CTAs per SM the non-persistent launch bound promises (registers per thread = 64K /
        # (threads x this)): _MIN_BLOCKS_PER_SM, or _SMALL_GRID_CTAS_PER_SM for small grids
        # (_launch). Not in the register-split builds, whose setmaxnreg counts assume 64.
        assert min_blocks == _MIN_BLOCKS_PER_SM or (
            min_blocks == _SMALL_GRID_CTAS_PER_SM and not self.split and not persistent
        ), (min_blocks, self.split, persistent)
        self.min_blocks = min_blocks
        if persistent:
            assert _persist_ok(T, self.hpk, hpc, self.sr), (T, self.hpk, hpc, self.sr)
            assert self.split and self.vsmem and self.nsub == 1
        # Strided inputs (local, SGLang): q / k and v as head slices of a wider token row (the
        # packed post-conv [q | k | v] rows; qk_io / v_io = that row's width in 128-element heads,
        # i.e. their per-token stride; default h_io / hv_io) and a / b with a runtime per-token
        # stride (ab_dyn: views of the projection's [b | a] columns). Only the layouts change:
        # every q / k / v / a / b address goes through them, so the values read are the same.
        self.qk_io = self.h_io if qk_io is None else int(qk_io)
        self.v_io = self.hv_io if v_io is None else int(v_io)
        self.ab_dyn = bool(ab_dyn)
        # pdl (local, SGLang): programmatic dependent launch. The CTA may become resident while
        # the previous kernel (the conv that writes q / k / v) is still running: TMEM is allocated
        # (and its permit relinquished, so the SM's other CTA slots fill too), every thread that
        # reads q / k / v / a / b waits (griddepcontrol.wait) first (warp 0's L2 prefetch of v
        # before it is only a hint), and the CTA lets the next kernel's launch begin
        # (launch_dependents; that kernel waits for this grid itself). The caller guarantees that
        # only the kernel just before (and nothing earlier) is still in flight and that it writes
        # none of state, rings, cursors, row order or A_log / dt_bias: its writes are visible only
        # after the wait, which warps 0 / 1 skip.
        self.pdl = bool(pdl)
        # ring_rows (local, SGLang): the k / u / g rings get their own row index per batch row
        # (``ring_idx``, e.g. the request's row in the request table) instead of the state slot,
        # so they can be sized by live requests while the state pool keeps every cached slot.
        # Only the ring addresses change (state reads and writes keep the slot), so the values
        # read and written are the same. Non-persistent, 4-D, non-early-split builds only.
        self.ring_rows = bool(ring_rows)
        if self.ring_rows:
            assert not persistent and self.parts == 1 and not self.early_split, (
                persistent,
                self.parts,
                self.early_split,
            )
        # fused_conv (local, SGLang, SGLANG_ENABLE_GDN_UCACHE_FUSED_CONV): q / k / v are the
        # target-verify conv's output rows, and this kernel computes them: each CTA's math warps
        # convolve the CTA's own channels (q and k of its key head, v of its value heads) from the
        # raw projection rows (``conv_x_row`` elements a token, a launch argument), store the
        # outputs, the new conv state and the step windows as the separate conv would, and sync
        # before any q / k / v read. One CTA per key head and row (hpc == hpk) is the unique
        # writer of those channels.
        self.fused_conv = bool(fused_conv)
        # fused_norm (local, SGLang, SGLANG_ENABLE_GDN_UCACHE_FUSED_NORM): the epilogue stores
        # rmsnorm(y) * w * sigmoid(z) per (token, value head) instead of y, the gated norm the model
        # applies next (16-lane x 8-element partial sums over the 4 math warps' 128 V rows).
        self.fused_norm = bool(fused_norm)
        if self.fused_norm:
            assert (
                self.pro_reorder and not persistent and not self.split and self.parts == 1
                and self.sr == "" and not self.vsmem
            ), (self.pro_reorder, persistent, self.split, self.parts, self.sr)
        if self.fused_conv:
            assert (
                self.pdl
                and self.pro_reorder
                and not persistent
                and not self.split
                and self.nsub == 1
                and self.parts == 1
                and self.sr == ""
                and self.qk_io == self.v_io
                and 2 * h * K_DIM + hv * V_DIM == self.qk_io * K_DIM
            ), (self.pdl, self.pro_reorder, persistent, self.split, self.nsub, self.parts)
        if (
            self.pdl
            or self.ab_dyn
            or self.qk_io != self.h_io
            or self.v_io != self.hv_io
        ):
            # the non-persistent bodies (kernel, kernel_early) over the whole 4-D pool only
            assert self.qk_io >= h and self.v_io >= hv, (self.qk_io, self.v_io)
            assert not persistent and self.parts == 1 and self.h_io == h, (
                persistent,
                self.parts,
                self.h_io,
            )

    @cute.jit
    def _fuse_parts(self, x: cute.Tensor):
        """[pool, parts, heads / parts, ...] -> [pool, (heads / parts, parts), ...]: one head
        mode, head j -> (j % (heads / parts), j // (heads / parts)) (parts > 1 only)."""
        sh, st = x.layout.shape, x.layout.stride
        lay = cute.make_layout(
            (sh[0], (sh[2], sh[1]), *sh[3:]),
            stride=(st[0], (st[2], st[1]), *st[3:]),
        )
        return cute.make_tensor(x.iterator, lay)

    @cute.jit
    def _make_state_tma(self, state: cute.Tensor, op):
        """TMA atom for one 16 KB state stage: a value head's 128 V rows x one 64-element K half,
        128B-swizzled (the readout UMMA's A operand)."""
        elems = 64
        smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (1, 1, V_DIM, (elems, 1), 1),
                stride=(0, 0, elems, (1, V_DIM * elems), V_DIM * elems),
            ),
        )
        if cutlass.const_expr(self.parts > 1):
            # (part, head) of the fused head mode take two TMA dimensions: K stays one
            # dimension (the half is a coordinate offset) to keep the descriptor at 5
            info = cpasync.make_tiled_tma_atom(
                op, state, smem_layout, cta_tiler=(1, 1, V_DIM, elems)
            )
            return cpasync.TmaInfo(
                info.atom,
                cute.logical_divide(info.tma_tensor, (None, None, None, elems)),
                info.smem_layout,
            )
        return cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(state, (None, None, None, elems)),
            smem_layout,
            cta_tiler=(1, 1, V_DIM, elems),
        )

    @cute.jit
    def __call__(
        self,
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state,
        state_idx,
        k_cache,
        u_cache,
        g_cache,
        hist_len,
        cache_base,
        row_order,
        out,
        rand_seed,
        ring_idx,
        scale: Float32,
        flush_min: Int32,
        hv_base: Int32,
        conv_x: Int64,
        conv_w: Int64,
        conv_s: Int64,
        conv_win: Int64,
        conv_widx: Int64,
        conv_x_row: Int32,
        norm_z: Int64,
        norm_z_row: Int32,
        norm_w: Int64,
        norm_eps: Float32,
        stream,
    ):
        """Non-persistent launch: one CTA per (key head x head slice, batch row). ``conv_*``:
        the fused conv's byte addresses and x row stride (fused_conv builds; 0 and never read
        otherwise)."""
        if cutlass.const_expr(self.parts > 1):
            state = self._fuse_parts(state)
            k_cache = self._fuse_parts(k_cache)
            u_cache = self._fuse_parts(u_cache)
            g_cache = self._fuse_parts(g_cache)
        tma_ld = self._make_state_tma(state, cpasync.CopyBulkTensorTileG2SOp())
        tma_st = self._make_state_tma(state, cpasync.CopyBulkTensorTileS2GOp())
        kern = (
            self.kernel_early if cutlass.const_expr(self.early_split) else self.kernel
        )
        kern(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state_idx,
            k_cache,
            u_cache,
            g_cache,
            hist_len,
            cache_base,
            row_order,
            out,
            rand_seed,
            scale,
            flush_min,
            hv_base,
            tma_ld,
            tma_st,
            ring_idx,
            conv_x,
            conv_w,
            conv_s,
            conv_win,
            conv_widx,
            conv_x_row,
            norm_z,
            norm_z_row,
            norm_w,
            norm_eps,
        ).launch(
            grid=(self.h * self.nsub, q.shape[0], 1),
            block=(self.num_threads, 1, 1),
            min_blocks_per_mp=self.min_blocks,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        state_idx_t: cute.Tensor,
        k_cache: cute.Tensor,
        u_cache: cute.Tensor,
        g_cache: cute.Tensor,
        hist_len: cute.Tensor,
        cache_base: cute.Tensor,
        row_order: cute.Tensor,
        out: cute.Tensor,
        rand_seed: cute.Tensor,
        scale: Float32,
        flush_min: Int32,
        hv_base: Int32,
        tma_ld: cpasync.TmaInfo,
        tma_st: cpasync.TmaInfo,
        ring_idx_t: cute.Tensor,
        conv_x: Int64,
        conv_w: Int64,
        conv_s: Int64,
        conv_win: Int64,
        conv_widx: Int64,
        conv_x_row: Int32,
        norm_z: Int64,
        norm_z_row: Int32,
        norm_w: Int64,
        norm_eps: Float32,
    ):
        """One (key head, row) CTA: the builds without ``early_split`` (T <= 4, and the T >= 5
        builds with fewer heads per CTA or without the register split)."""
        T = self.T
        T2 = 2 * T

        tid, _, _ = cute.arch.thread_idx()
        bx, blk_row, _ = cute.arch.block_idx()
        HPC = self.hpc
        if cutlass.const_expr(self.nsub == 1):
            kh = bx  # key head
            sub = Int32(0)
        else:
            kh = bx // self.nsub  # key head
            sub = bx % self.nsub  # which hpc-head slice of it
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        if cutlass.const_expr(self.use_row_order):
            brow = Int32(row_order[blk_row])
        else:
            brow = blk_row
        hv0 = kh * self.hpk + sub * HPC

        # ------------------------------------------------------------------ SMEM
        smem = cutlass.utils.SmemAllocator()
        st_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((V_DIM, (64, 2)), stride=(64, (1, V_DIM * 64))),
        )
        s_state = smem.allocate_tensor(
            Float16, st_layout.outer, byte_alignment=1024, swizzle=st_layout.inner
        )
        s_x = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * 128), byte_alignment=1024
        )
        # K_hist tile: window keys k̂_j (j < P, else 0) row-major [16 slots x 128] fp16 in the
        # X-tile SW128 layout; the fold reads it as an MN-major B operand (N = k index)
        s_kt = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * 128), byte_alignment=1024
        )
        s_c = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * 16), byte_alignment=256
        )
        s_gram = smem.allocate_array(Float32, 32 * 16)
        s_coef = smem.allocate_array(Float32, HPK * _COEF)
        s_rn = smem.allocate_array(Float32, 16)
        # fused_norm: [head parity][token][16 thread partials] of the row sums of squares
        s_nrm = smem.allocate_array(Float32, 2 * 8 * 16) if self.fused_norm else None
        # u windows of two heads: [buf][logical slot j][V] (ring dtype), bulk-copied by warp 0
        s_uwin = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(2 * W_RING * V_DIM), byte_alignment=128
        )
        u_full = self._mbar_list(smem, HPK)
        u_free = self._mbar_list(smem, 2)
        st_full = self._mbar_list(smem, 2 * HPK)
        mma_slot = self._mbar_list(smem, 2 * HPK)
        if cutlass.const_expr(self.rmw_loop):
            # RMW loop: half g waits on [rd0 | mma_done][g HPK + hh] and
            # [f_mma_a | f_mma_b][g HPK + hh] (one array each)
            rdm = self._mbar_list(smem, 2 * HPK)
            rd0, mma_done = rdm[:HPK], rdm[HPK:]
        else:
            mma_done = self._mbar_list(smem, HPK)
        d_ready = self._mbar_list(smem, HPK)
        epi_done = self._mbar_list(smem, HPK)
        if cutlass.const_expr(self.rmw_loop):
            fab = self._mbar_list(smem, 2 * HPK)
            f_mma_a, f_mma_b = fab[:HPK], fab[HPK:]
        else:
            f_mma_a = self._mbar_list(smem, HPK)
            f_mma_b = self._mbar_list(smem, HPK)
        f_epi_a = self._mbar_list(smem, HPK)
        if cutlass.const_expr(not self.rmw_loop):
            rd0 = self._mbar_list(smem, HPK)
        f_gdone = self._mbar_list(smem, 2 * HPK)
        x_ready = smem.allocate_array(cutlass.Int64, 1)
        tmem_done = smem.allocate_array(cutlass.Int64, 1)
        tmem_addr = smem.allocate(Int32, 4)

        # TMEM first: the next CTA on this SM is not launched before this one relinquishes
        # its TMEM allocation permit (measured: allocating after the prologue spaced the
        # CTAs of one SM ~2.5 us apart and left a quarter of the CTA slots empty)
        if warp == 1:
            U.tmem_alloc(tmem_addr, _TMEM_COLS)
            cute.arch.relinquish_tmem_alloc_permit()
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            # Only q / k / v / a / b come from the kernels still in flight (the conv): warps 0 / 1 read
            # the cursors, row order, state and rings, which earlier launches wrote, and go ahead (head
            # 0's state and u windows load during the conv); the math warps, and warp 0 where it copies
            # v rows (vsmem), wait for it.
            if cutlass.const_expr(self.vsmem):
                cute.arch.griddepcontrol_wait()
            else:
                if warp >= self.cw0:
                    if cutlass.const_expr(self.fused_conv):
                        # fused_conv: the kernel in flight is the projection GEMM; the math warps
                        # wait for it inside the conv prologue, which ends with their barrier 1
                        self._conv_prephase(
                            q,
                            state_idx_t,
                            brow,
                            kh,
                            hv0,
                            tid - 32 * self.cw0,
                            conv_x,
                            conv_w,
                            conv_s,
                            conv_win,
                            conv_widx,
                            conv_x_row,
                        )
                    else:
                        cute.arch.griddepcontrol_wait()

        # Prologue, two orders. Without pro_reorder: the cursors first (one global round trip,
        # issued by every thread), then every mbarrier while that round trip is in flight,
        # then the entry loads. With pro_reorder: the entry loads first and the cursors last
        # (see below), and warp 0's inits once the cursors are known, around head 0's state.
        # Warp 0 publishes the inits through one named barrier per role (2: warp 1 + the X-tile
        # warps; 3: the ring-row warps), each synced before that role's first mbarrier use --
        # no CTA-wide sync, which a warp only reaches once its outstanding loads have landed.
        # Warp 1 shares the X-tile warps' barrier at no cost: its next step waits for x_ready,
        # which they arrive on right after it. Named barriers 1-3 only (0 stays the CTA-wide
        # one, unused): SM107 (Rubin) has 16 per SM, so ids up to 5 would cap it at 2 CTAs/SM
        # (ncu "Block Limit Barriers") below the 4 that TMEM / SMEM / registers allow.
        # A barrier instance whose participants reach it at more than one instruction (2/128:
        # warp 0's arrive + warp 1's and the X-tile warps' syncs; 3/160 and 3/96: arrives + the
        # ring-row warps' sync; 1/128 without pro_reorder: the X-tile and ring-row warps' syncs)
        # uses the non-aligned forms (U.bar_sync_ua, barrier_arrive(aligned=False)): bar.sync /
        # bar.arrive are .aligned, which requires all participants to execute the same
        # instruction. The single-site instances (1/128 after the role split, 2/64) stay aligned.
        mbar = dict(
            st_full=st_full,
            mma_slot=mma_slot,
            f_gdone=f_gdone,
            u_full=u_full,
            mma_done=mma_done,
            d_ready=d_ready,
            epi_done=epi_done,
            f_mma_a=f_mma_a,
            f_mma_b=f_mma_b,
            f_epi_a=f_epi_a,
            rd0=rd0,
            u_free=u_free,
            x_ready=x_ready,
            tmem_done=tmem_done,
        )
        if cutlass.const_expr(self.pro_reorder):
            if warp == 0:
                # the state descriptor's fetch overlaps the cursor round trip
                cpasync.prefetch_descriptor(tma_ld.atom)
        else:
            sidx = Int32(state_idx_t[brow])
            P = Int32(hist_len[brow])
            base = Int32(cache_base[brow])
            if cutlass.const_expr(self.ring_rows):
                ridx = Int32(ring_idx_t[brow])
            else:
                ridx = sidx
            if warp == 0:
                with cute.arch.elect_one():
                    self._init_mbarriers(mbar, 0)
                cute.arch.fence_acq_rel_cta()
                cute.arch.barrier_arrive(
                    barrier_id=2, number_of_threads=128, aligned=False
                )
                cute.arch.barrier_arrive(
                    barrier_id=3, number_of_threads=96, aligned=False
                )

        # Entry loads: inputs that depend on the batch row only, issued by EVERY thread (warps
        # that do not use a value compute an in-range dummy address and skip the load) ahead of
        # the padded-row exit, so their latency overlaps the cursors'. With pro_reorder the
        # cursor loads follow them with no branch in between: ptxas converts the cursors to
        # uniform registers at the end of the basic block that loads them, and every warp
        # waits for them there (warp 0's mbarrier inits branching right after the cursor loads
        # put the X-row / gate loads, the inits and warp 0's first TMA a cursor round trip
        # late); issued last, the cursors get scoreboards of their own and are converted only
        # after every entry load has issued.
        T_ = self.T
        grp_e = lane >> 2
        c4_e = lane & 3
        CW0 = self.cw0
        cw_e = (warp - CW0) & 3
        nt_e = cw_e & 1
        q_g = q.iterator.toint()
        k_g = k.iterator.toint()
        if cutlass.const_expr(self.fused_conv):
            # this CTA wrote q / k in its conv prologue: X-row loads after its barrier
            q_g = U.opaque_i64(q_g)
            k_g = U.opaque_i64(k_g)
        xr_b = nt_e * 8 + grp_e
        if cutlass.const_expr(not self.pro_reorder):
            is_cons = Int32(warp >= CW0)
        is_xt = Int32(
            warp >= CW0 + 2
        )  # X-tile warps (mt == 1); the ring-row warps never read wo
        if cutlass.const_expr(self.pro_reorder):
            # X rows: the X-tile warps' here; the ring-row warps load theirs with the ring rows,
            # which read the same (tied) registers -- and so would wait for these loads' round
            # trip on top of the cursors' if the ring-row warps issued them here. Gate inputs:
            # the gate lanes' warp (the first X-tile warp) only.
            x_pred = is_xt
            gate_pred = Int32(warp == CW0 + 2)
        else:
            # X rows and gate inputs: every consumer warp
            x_pred = is_cons
            gate_pred = is_cons
        x_own = self._x_row_addr(k, q, k_g, q_g, brow, kh, xr_b)
        x_oth = self._x_row_addr(k, q, k_g, q_g, brow, kh, (1 - nt_e) * 8 + grp_e)
        wb = []
        wo = []
        for p in cutlass.range_constexpr(4):
            koff = Int64((32 * p + 8 * c4_e) * 2)
            wb.append(U.ldg_v4_if(x_pred, x_own + koff))
            wo.append(U.ldg_v4_if(is_xt, x_oth + koff))
        ct_e = (tid - 32 * CW0) & 127
        gh = (ct_e >> 3) & 3
        ghc = gh % HPC  # in-range head for loads
        gt = ct_e & 7
        tt = _sel(gt < T_, gt, 0, Int32)
        ab_off = Int64(cute.crd2idx((brow, tt, hv0 + ghc), a.layout)) * 2
        # never in warps 0..CW0-1: with the register split, they cannot finish setmaxnreg.dec
        # while loads they issued are outstanding, and they never use these values
        av_raw = U.ldg_u16_if(gate_pred, a.iterator.toint() + ab_off)
        bv_raw = U.ldg_u16_if(gate_pred, b.iterator.toint() + ab_off)
        # pro_reorder: the gate warp's per-head constants too (a dependent round trip after
        # x_ready otherwise); not in the split builds with fewer heads per CTA, whose
        # 64-register prologue they push into spills (sm_107a)
        pre_gate = self.pro_reorder and (not self.split or HPC == HPK)
        if cutlass.const_expr(self.pro_reorder):
            alog_e = U.ldg_f32_if(
                gate_pred if pre_gate else Int32(0),
                a_log.iterator.toint()
                + Int64(cute.crd2idx((hv0 + ghc,), a_log.layout)) * 4,
            )
            dtb_e = U.ldg_f32_if(
                gate_pred if pre_gate else Int32(0),
                dt_bias.iterator.toint()
                + Int64(cute.crd2idx((hv0 + ghc,), dt_bias.layout)) * 4,
            )
            sidx = Int32(state_idx_t[brow])
            P = Int32(hist_len[brow])
            base = Int32(cache_base[brow])
            if cutlass.const_expr(self.ring_rows):
                ridx = Int32(ring_idx_t[brow])
            else:
                ridx = sidx

        if warp == 1:
            if sidx < 0:
                # padded row: hand the TMEM columns back before the CTA exits
                cute.arch.sync_warp()
                U.tmem_dealloc(tmem_addr[0], _TMEM_COLS)
        U.exit_if_negative(sidx)
        if cutlass.const_expr(not self.pro_reorder):
            is_flush = flush_min <= P
            # Cursors known: the Gram's ring rows (ring-row warps; row grp goes into the
            # registers that hold the other X half in the X-tile warps) and the gate values, so
            # their round trips overlap the entry loads and warp 0's window / state issue.
            # ring-row warps (mt == 0)
            is_rr = Int32(warp == CW0) + Int32(warp == CW0 + 1)
            kc_g = k_cache.iterator.toint()
            wr1_e = self._ring_row_loads(
                k_cache, kc_g, is_rr, ridx, kh, base, grp_e, c4_e, wo
            )
            # window g values (gate threads: G_P of their head; weight threads: g_j)
            win_e = self._window_g_loads(
                g_cache, is_cons, ct_e - 32, ridx, hv0, ghc, base, P
            )

        st_addr = s_state.iterator.toint()
        x_addr = s_x.iterator.toint()
        kt_addr = s_kt.iterator.toint()
        c_addr = s_c.iterator.toint()
        uw_addr = s_uwin.iterator.toint()
        u_g = u_cache.iterator.toint()

        if warp == 0:
            if cutlass.const_expr(self.pro_reorder):
                # Cursors known: head 0's state goes out first (it starts the CTA's serial state
                # stream), behind only the inits it needs (its two stages and the first two u
                # windows'); then the other mbarrier inits (barrier 3 also hands the ring-row
                # warps the X-tile warps' X x X Gram), the u windows and the v prefetch.
                with cute.arch.elect_one():
                    for s_ in cutlass.range_constexpr(2):
                        cute.arch.mbarrier_init(st_full[s_], 1)
                        cute.arch.mbarrier_init(u_full[s_], 1)
                    cute.arch.mbarrier_init_fence()
            else:
                cpasync.prefetch_descriptor(tma_ld.atom)
                # the two u windows (4 KB) ahead of the two state halves (32 KB): head 0's D
                # build needs its window long before the readout needs the state
                for hh in cutlass.range_constexpr(min(2, HPC)):
                    if cutlass.const_expr(self.vsmem):
                        # verify rows: wv u slots + the v rows (issued behind the state)
                        with cute.arch.elect_one():
                            self._u_window_copy(
                                u_cache,
                                u_g,
                                uw_addr,
                                u_full[hh],
                                ridx,
                                hv0 + hh,
                                base,
                                hh,
                                _sel(is_flush, W_RING, self.wv, Int32),
                                _sel(is_flush, 0, T * V_DIM * 2, Int32),
                            )
                    else:
                        with cute.arch.elect_one():
                            self._u_window_copy(
                                u_cache,
                                u_g,
                                uw_addr,
                                u_full[hh],
                                ridx,
                                hv0 + hh,
                                base,
                                hh,
                            )
            for g in cutlass.range_constexpr(2):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(st_full[g], 16384)
                U.simple_tma_copy(
                    tma_ld.atom,
                    tma_ld.tma_tensor[sidx, hv0, None, (None, g)],
                    s_state[None, (None, g)],
                    st_full[g],
                )
            if cutlass.const_expr(self.vsmem):
                # verify rows: heads 0 / 1's v rows into their windows' free slots (their
                # u_full expect-tx above already counts them)
                if not is_flush:
                    with cute.arch.elect_one():
                        for hh in cutlass.range_constexpr(min(2, HPC)):
                            self._v_rows_copy(
                                v, uw_addr, u_full[hh], brow, hv0 + hh, hh
                            )
            if cutlass.const_expr(self.pro_reorder):
                with cute.arch.elect_one():
                    self._init_mbarriers(mbar, 2)
                cute.arch.fence_acq_rel_cta()
                cute.arch.barrier_arrive(
                    barrier_id=2, number_of_threads=128, aligned=False
                )
                cute.arch.barrier_arrive(
                    barrier_id=3, number_of_threads=160, aligned=False
                )
                for hh in cutlass.range_constexpr(min(2, HPC)):
                    with cute.arch.elect_one():
                        self._u_window_copy(
                            u_cache, u_g, uw_addr, u_full[hh], ridx, hv0 + hh, base, hh
                        )
            # Every head's v rows into L2 (evict_last), behind the state: the math warps load
            # head h's v at the start of its D build and need it in its epilogue; from DRAM
            # that round trip was exposed there (Rubin: the bf16 conversions of v held 14-25 %
            # of the stall samples at T = 1-8). One bulk prefetch per token covers the CTA's
            # heads (contiguous in [B, T, HV, V]); no registers, no scoreboard. A register
            # double buffer (head h + 1's loads issued in head h) does not work here: ptxas
            # puts every v load on one scoreboard, so head h's first use also waits for head
            # h + 1's loads. (vsmem verify rows: it feeds warp 0's v-row copies of heads 2 / 3.)
            v_g = v.iterator.toint()
            with cute.arch.elect_one():
                # fused_conv: v is written by this CTA's conv prologue, not yet in memory
                for t in cutlass.range_constexpr(0 if self.fused_conv else T):
                    U.bulk_prefetch_l2(
                        v_g + Int64(cute.crd2idx((brow, t, hv0, 0), v.layout)) * 2,
                        HPC * V_DIM * 2,
                    )

        if cutlass.const_expr(self.pro_reorder):
            is_flush = flush_min <= P
            # Cursors known: the ring-row warps' ring rows (row grp goes into the registers that
            # hold the other X half in the X-tile warps) and own X rows, so their round trip
            # overlaps the X-tile warps' prologue and warp 0's state / window issue.
            # ring-row warps (mt == 0)
            is_rr = Int32(warp == CW0) + Int32(warp == CW0 + 1)
            kc_g = k_cache.iterator.toint()
            wr1_e = self._ring_row_loads(
                k_cache, kc_g, is_rr, ridx, kh, base, grp_e, c4_e, wo, wb, x_own
            )

        # Warpgroup-level split first: with the register split each setmaxnreg must
        # dominate its warpgroup's code (ptxas uses the minimum after a merge). This shape
        # (and row from warp & 3) also lets ptxas fit Philox-5 at 4 heads per CTA in 80
        # registers, where a flat if / elif / else over the three roles spilled 6 values
        # whose reloads (16 % L1 misses: 233 KB of the SM's 256 KB is shared memory) sat on
        # the flush critical path (T = 8, B = 512, every row flushing: 451 vs 495 us).
        # Allocation-sensitive: re-check STL / LDL over the (T, heads per CTA, rounding)
        # cells (CUTE_DSL_KEEP=sass) after edits to the math warps' path.
        if warp < CW0:
            if cutlass.const_expr(self.split):
                cute.arch.setmaxregister_decrease(self.regsplit[0])
            if warp == 0:
                # ============================================================ producer
                if is_flush:
                    cpasync.prefetch_descriptor(tma_st.atom)
                    for slot in cutlass.range_constexpr(2, 2 * HPC):
                        prev = slot - 2
                        cute.arch.mbarrier_wait(f_gdone[prev], 0)
                        U.simple_tma_copy(
                            tma_st.atom,
                            s_state[None, (None, prev % 2)],
                            tma_st.tma_tensor[
                                sidx, hv0 + prev // 2, None, (None, prev % 2)
                            ],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                st_full[slot], 16384
                            )
                        U.simple_tma_copy(
                            tma_ld.atom,
                            tma_ld.tma_tensor[
                                sidx, hv0 + slot // 2, None, (None, slot % 2)
                            ],
                            s_state[None, (None, slot % 2)],
                            st_full[slot],
                        )
                        if cutlass.const_expr(slot % 2 == 0 and slot // 2 >= 2):
                            hh = slot // 2
                            cute.arch.mbarrier_wait(u_free[hh - 2], 0)
                            with cute.arch.elect_one():
                                self._u_window_copy(
                                    u_cache,
                                    u_g,
                                    uw_addr,
                                    u_full[hh],
                                    ridx,
                                    hv0 + hh,
                                    base,
                                    hh % 2,
                                )
                    for prev in cutlass.range_constexpr(2 * HPC - 2, 2 * HPC):
                        cute.arch.mbarrier_wait(f_gdone[prev], 0)
                        U.simple_tma_copy(
                            tma_st.atom,
                            s_state[None, (None, prev % 2)],
                            tma_st.tma_tensor[
                                sidx, hv0 + prev // 2, None, (None, prev % 2)
                            ],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)
                else:
                    for slot in cutlass.range_constexpr(2, 2 * HPC):
                        cute.arch.mbarrier_wait(mma_slot[slot - 2], 0)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                st_full[slot], 16384
                            )
                        U.simple_tma_copy(
                            tma_ld.atom,
                            tma_ld.tma_tensor[
                                sidx, hv0 + slot // 2, None, (None, slot % 2)
                            ],
                            s_state[None, (None, slot % 2)],
                            st_full[slot],
                        )
                        # u window behind this state slot, if any (see _u_window_head)
                        hh = self._u_window_head(slot)
                        if cutlass.const_expr(hh >= 0):
                            cute.arch.mbarrier_wait(u_free[hh - 2], 0)
                            if cutlass.const_expr(self.vsmem):
                                with cute.arch.elect_one():
                                    self._u_window_copy(
                                        u_cache,
                                        u_g,
                                        uw_addr,
                                        u_full[hh],
                                        ridx,
                                        hv0 + hh,
                                        base,
                                        hh % 2,
                                        self.wv,
                                        T * V_DIM * 2,
                                    )
                                    self._v_rows_copy(
                                        v, uw_addr, u_full[hh], brow, hv0 + hh, hh % 2
                                    )
                            else:
                                with cute.arch.elect_one():
                                    self._u_window_copy(
                                        u_cache,
                                        u_g,
                                        uw_addr,
                                        u_full[hh],
                                        ridx,
                                        hv0 + hh,
                                        base,
                                        hh % 2,
                                    )
            elif warp == 1:
                # ============================================================ UMMA issuer
                tbase = tmem_addr[0]
                sd = U.sdesc_sw128(V_DIM * 128)
                x_base = sd | Uint64(x_addr >> 4)
                stage_base = [
                    sd | Uint64(st_addr >> 4),
                    sd | Uint64((st_addr + 16384) >> 4),
                ]
                c_desc = U.sdesc_sw32(Uint64(c_addr >> 4))
                # MN-major SW128: SBO = 1024 B between 8-slot K atoms, N halves 2048 B apart
                kt_desc = U.sdesc_mn_sw128(2048, 1024) | Uint64(kt_addr >> 4)
                idesc16 = U.f16_idesc(128, 16)
                idesc64mn = U.f16_idesc(128, 64, b_mn_major=True)
                U.bar_sync_ua(2, 128)  # warp 0's inits
                cute.arch.mbarrier_wait(x_ready, 0)
                U.fence_after_sync()
                if is_flush:
                    # per head: fold half 0 (needs only D, Kt) -> readout half 0 (rd0: the
                    # consumers' RMW of stage 0 may start) -> readout half 1 + history
                    # (mma_done) -> fold half 1 once the RMW has read fold half 0 (f_epi_a).
                    # R/Y (cols 0..31) and F (64..127) of head h + 1 wait for head h's epilogue.
                    for hh in cutlass.range_constexpr(HPC):
                        cute.arch.mbarrier_wait(d_ready[hh], 0)
                        if cutlass.const_expr(hh > 0):
                            cute.arch.mbarrier_wait(epi_done[hh - 1], 0)
                        U.fence_after_sync()
                        U.umma_ts(tbase + 64, tbase + 32, kt_desc, idesc64mn, False)
                        U.umma_commit(f_mma_a[hh])
                        for g in cutlass.range_constexpr(2):
                            slot = 2 * hh + g
                            cute.arch.mbarrier_wait(st_full[slot], 0)
                            U.fence_after_sync()
                            for ka in cutlass.range_constexpr(4):
                                U.umma_ss(
                                    tbase + 0,
                                    stage_base[g] | Uint64((ka * 32) >> 4),
                                    x_base | Uint64((g * 2048 + ka * 32) >> 4),
                                    idesc16,
                                    (g > 0) or (ka > 0),
                                )
                            if cutlass.const_expr(g == 0):
                                U.umma_commit(rd0[hh])
                        U.umma_ts(tbase + 16, tbase + 32, c_desc, idesc16, False)
                        U.umma_commit(mma_done[hh])
                        cute.arch.mbarrier_wait(f_epi_a[hh], 0)
                        U.fence_after_sync()
                        U.umma_ts(
                            tbase + 64,
                            tbase + 32,
                            kt_desc + Uint64(2048 >> 4),
                            idesc64mn,
                            False,
                        )
                        U.umma_commit(f_mma_b[hh])
                else:
                    for hh in cutlass.range_constexpr(HPC):
                        rcol = (hh % 2) * 32
                        if cutlass.const_expr(hh >= 2):
                            cute.arch.mbarrier_wait(epi_done[hh - 2], 0)
                            U.fence_after_sync()
                        for g in cutlass.range_constexpr(2):
                            slot = 2 * hh + g
                            cute.arch.mbarrier_wait(st_full[slot], 0)
                            U.fence_after_sync()
                            for ka in cutlass.range_constexpr(4):
                                U.umma_ss(
                                    tbase + rcol,
                                    stage_base[g] | Uint64((ka * 32) >> 4),
                                    x_base | Uint64((g * 2048 + ka * 32) >> 4),
                                    idesc16,
                                    (g > 0) or (ka > 0),
                                )
                            U.umma_commit(mma_slot[slot])
                        cute.arch.mbarrier_wait(d_ready[hh], 0)
                        U.fence_after_sync()
                        U.umma_ts(
                            tbase + rcol + 16,
                            tbase + 64 + (hh % 2) * 8,
                            c_desc,
                            idesc16,
                            False,
                        )
                        U.umma_commit(mma_done[hh])
                cute.arch.mbarrier_wait(tmem_done, 0)
                U.fence_after_sync()
                U.tmem_dealloc(tbase, _TMEM_COLS)
        else:
            if cutlass.const_expr(self.pro_reorder):
                # window g values, ring-row warps only (lanes 0-31: G_P of head gh for the g
                # appends; lanes 0-63: G_P and g_j of the window weights). ptxas tracks the
                # prologue's loads on one scoreboard: in the X-tile warps these would hold the
                # Gram -- and x_ready -- to their round trip, and scheduled ahead of the ring
                # rows (which read tied registers) they would hold the ring rows to it; here,
                # past the role split, they follow the ring rows. Ahead of setmaxnreg.inc,
                # which waits for warp 0's prologue.
                wh, wj, gp_g, gp_w, gj_w = self._window_g_loads(
                    g_cache, is_rr, ct_e, ridx, hv0, ghc, base, P
                )
            else:
                # the gate / weight lanes' values, loaded at entry
                wh, wj, gp_g, gp_w, gj_w = win_e
            if cutlass.const_expr(self.split):
                cute.arch.setmaxregister_increase(self.regsplit[1])
            # ============================================================ consumers
            cw = warp - CW0
            ct = cw * 32 + lane  # TMEM lane address / k index for staging
            row = ((warp & 3) << 5) | lane  # V row held by this thread (= TMEM lane)
            grp = lane >> 2
            c4 = lane & 3

            # -------- Gram on mma.sync: rows [k̂_ring(16); X(16)] x cols X(16), K = 128.
            # Fragments come straight from global memory: per 32-wide k block a lane loads 16 B
            # (8 contiguous k) of each of its rows; the k order inside the m16n8k16 fragment is
            # permuted identically for A and B, which a dot product does not see.
            # Warps mt == 1 own the X x X tiles (norms, intra-step dots): they depend on the
            # batch row only, so the X tile and x_ready follow ONE global round trip.  Warps
            # mt == 0 own the ring x X tiles from the rows issued once the cursors landed.
            mt = cw >> 1
            nt = cw & 1
            gc = nt * 8 + 2 * c4
            if mt == 1:
                # A rows grp / grp + 8 of the X x X tile: this lane's own row (wb) and the
                # other half's (wo), both loaded at entry
                accx = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                for p in cutlass.range_constexpr(4):
                    for hs in cutlass.range_constexpr(2):
                        w0a = _sel(nt == 0, wb[p][2 * hs], wo[p][2 * hs], Uint32)
                        w1a = _sel(nt == 0, wo[p][2 * hs], wb[p][2 * hs], Uint32)
                        w0b = _sel(
                            nt == 0, wb[p][2 * hs + 1], wo[p][2 * hs + 1], Uint32
                        )
                        w1b = _sel(
                            nt == 0, wo[p][2 * hs + 1], wb[p][2 * hs + 1], Uint32
                        )
                        accx = U.mma_bf16_16816(
                            w0a, w1a, w0b, w1b, wb[p][2 * hs], wb[p][2 * hs + 1], accx
                        )
                s_gram[(16 + grp) * 16 + gc] = accx[0]
                s_gram[(16 + grp) * 16 + gc + 1] = accx[1]
                s_gram[(24 + grp) * 16 + gc] = accx[2]
                s_gram[(24 + grp) * 16 + gc + 1] = accx[3]
                if cutlass.const_expr(self.pro_reorder):
                    # The ring-row warps read the X x X diagonal (their rows' norms): arrive
                    # only. A sync here waited for them, and they reach any barrier only once
                    # their ring rows have landed (ptxas moves those registers into the mma
                    # operands first), which put x_ready -- and the first readout -- a round
                    # trip late. This warp's own diagonal (rn_x) needs a warp sync only.
                    cute.arch.barrier_arrive(
                        barrier_id=3, number_of_threads=160, aligned=False
                    )
                    cute.arch.sync_warp()
                else:
                    U.bar_sync_ua(1, 128)  # + the ring-row warps' (two sites)
                # X tile (fp16, SW128) and x_ready from these warps' own rows: they hold the
                # same rows as the ring-row warps and have nothing in flight, so the readout
                # can start once the X x X Gram is done instead of after the ring rows
                rn_x = cute.math.rsqrt(s_gram[(16 + xr_b) * 16 + xr_b] + Float32(_EPS))
                rn_x = _sel(xr_b >= T, rn_x * scale, rn_x, Float32)
                for p in cutlass.range_constexpr(4):
                    kc = 4 * p + c4
                    xw = []
                    for i in cutlass.range_constexpr(4):
                        xw.append(
                            U.pack_f16x2(
                                U.bf16x2_lo_f32(wb[p][i]) * rn_x,
                                U.bf16x2_hi_f32(wb[p][i]) * rn_x,
                            )
                        )
                    xb = (kc // 8) * 2048 + xr_b * 128 + (((kc % 8) ^ (xr_b % 8)) * 16)
                    U.sts_v4(x_addr + xb, xw[0], xw[1], xw[2], xw[3])
                U.bar_sync_ua(2, 128)  # warp 0's inits (+ warp 1's sync: two sites)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(x_ready)
                # Everything below needs only the X x X Gram and the gate inputs (one round
                # trip), so it overlaps warps mt == 0 waiting for the ring rows.
                cl = ct - 64
                if cl < T2:
                    rn_own = cute.math.rsqrt(
                        s_gram[(16 + cl) * 16 + cl] + Float32(_EPS)
                    )
                    if cl >= T:
                        rn_own = rn_own * scale
                    s_rn[cl] = rn_own
                if cl < 32:
                    # gates: lane = head gh x token gt
                    av = U.u16_as_f32(av_raw, True)
                    bv = U.u16_as_f32(bv_raw, True)
                    if cutlass.const_expr(pre_gate):
                        alog = alog_e
                        dtb = dtb_e
                    else:
                        alog = Float32(a_log[hv0 + ghc])
                        dtb = Float32(dt_bias[hv0 + ghc])
                    xx = av + dtb
                    sp = xx
                    if xx <= Float32(20.0):
                        sp = cute.math.log(
                            Float32(1.0) + cute.math.exp(xx, fastmath=True),
                            fastmath=True,
                        )
                    la = -cute.math.exp(alog, fastmath=True) * sp
                    if gt >= T:
                        la = Float32(0.0)
                    beta = Float32(1.0) / (
                        Float32(1.0) + cute.math.exp(-bv, fastmath=True)
                    )
                    gam = la
                    for off in (1, 2, 4):
                        lower = cute.arch.shuffle_sync_up(gam, off, mask_and_clamp=0)
                        if gt >= off:
                            gam = gam + lower
                    eg = cute.math.exp(gam, fastmath=True)
                    cbg = gh * _COEF
                    s_coef[cbg + _CO_BETA + gt] = beta
                    s_coef[cbg + _CO_BG + gt] = beta * eg
                    s_coef[cbg + _CO_EG + gt] = eg
                    s_coef[cbg + _CO_GAM + gt] = gam
                # id 2 again, now among the 64 X-tile threads only: both passed its first
                # (128-thread) instance above, so that instance has completed
                cute.arch.barrier(barrier_id=2, number_of_threads=64)
                # intra-step coefficient tables A[t][s], Bq[t][s] per head (64 threads)
                NT = HPC * T * T
                for it in cutlass.range_constexpr((NT + 63) // 64):
                    idx = cl + it * 64
                    if idx < NT:
                        ch = idx // (T * T)
                        t = (idx // T) % T
                        s = idx % T
                        cb = ch * _COEF
                        dec = cute.math.exp(
                            s_coef[cb + _CO_GAM + t] - s_coef[cb + _CO_GAM + s],
                            fastmath=True,
                        )
                        kk = s_gram[(16 + s) * 16 + t] * s_rn[s] * s_rn[t]
                        qk = s_gram[(16 + s) * 16 + T + t] * s_rn[s] * s_rn[T + t]
                        av_ = Float32(0.0)
                        if s < t:
                            av_ = s_coef[cb + _CO_BETA + t] * dec * kk
                        bq_ = Float32(0.0)
                        if s <= t:
                            bq_ = dec * qk
                        s_coef[cb + _CO_A + t * 8 + s] = av_
                        s_coef[cb + _CO_BQ + t * 8 + s] = bq_
            else:
                wr0 = wo  # ring rows grp / grp + 8, issued when the cursors landed
                wr1 = wr1_e
                if cutlass.const_expr(self.pro_reorder):
                    # the X-tile warps' X x X Gram and warp 0's inits (before this role's first
                    # mbarrier use)
                    U.bar_sync_ua(3, 160)
                else:
                    U.bar_sync_ua(1, 128)  # + the X-tile warps' (two sites)
                # norm of this lane's X row (for the k̂ ring append below)
                rn_b = cute.math.rsqrt(s_gram[(16 + xr_b) * 16 + xr_b] + Float32(_EPS))
                rn_b = _sel(xr_b >= T, rn_b * scale, rn_b, Float32)
                # ring x X tiles
                accr = self._ring_gram(wr0, wr1, wb, rn_b)
                s_gram[grp * 16 + gc] = accr[0]
                s_gram[grp * 16 + gc + 1] = accr[1]
                s_gram[(grp + 8) * 16 + gc] = accr[2]
                s_gram[(grp + 8) * 16 + gc + 1] = accr[3]
                if cutlass.const_expr(not self.pro_reorder):
                    # warp 0's inits, before this role's first mbarrier use (the coefficients)
                    U.bar_sync_ua(3, 96)
                # k̂ ring appends (ring dtype, 16 B per lane)
                if xr_b < T:
                    if sub == 0:
                        kdst = (
                            kc_g
                            + Int64(
                                cute.crd2idx(
                                    (ridx, kh, (base + P + xr_b) & RING_MASK, 0),
                                    k_cache.layout,
                                )
                            )
                            * 2
                        )
                        for p in cutlass.range_constexpr(4):
                            kw4 = self._k_append_words(wb[p], rn_b)
                            U.stg_v4(
                                kdst + Int64((32 * p + 8 * c4) * 2),
                                kw4[0],
                                kw4[1],
                                kw4[2],
                                kw4[3],
                            )
            cute.arch.barrier(barrier_id=1, number_of_threads=128)

            # -------- after the ring tiles: C tile (warps mt == 1), window weights, g-ring
            # appends, Kt (flush rows)
            if mt == 1:
                cl = ct - 64
                for i in cutlass.range_constexpr(4):
                    e = cl * 4 + i
                    xc = e >> 4
                    j = e & 15
                    cval = Float32(0.0)
                    if j < P:
                        if xc < T2:
                            if cutlass.const_expr(self.k_f16):
                                cval = s_gram[j * 16 + xc]  # ring x X̂: normalized already
                            else:
                                rnx = cute.math.rsqrt(
                                    s_gram[(16 + xc) * 16 + xc] + Float32(_EPS)
                                )
                                if xc >= T:
                                    rnx = rnx * scale
                                cval = s_gram[j * 16 + xc] * rnx
                    cbyte = xc * 32 + (((j // 8) ^ ((xc >> 2) & 1)) * 16) + (j % 8) * 2
                    U.sts_u16(c_addr + cbyte, U.f32_to_u16(cval, False))
            if is_flush:
                # K_hist tile (fold B operand): thread -> (slot row jr = ct / 8, 16 B chunk c8) of both
                # K groups; ring rows were just read by the Gram (L1/L2 hits); slots j >= P zeroed
                jr = ct >> 3
                c8 = ct & 7
                krow = (
                    kc_g
                    + Int64(
                        cute.crd2idx(
                            (ridx, kh, (base + jr) & RING_MASK, c8 * 8), k_cache.layout
                        )
                    )
                    * 2
                )
                kr = [U.ldg_v4(krow), U.ldg_v4(krow + Int64(128))]
                for gk in cutlass.range_constexpr(2):
                    kw4 = self._kt_words(kr[gk], jr < P)
                    U.sts_v4(
                        kt_addr + gk * 2048 + jr * 128 + ((c8 ^ (jr % 8)) * 16),
                        kw4[0],
                        kw4[1],
                        kw4[2],
                        kw4[3],
                    )
            # C / Kt are read by the history / fold UMMAs, which warp 1 issues only after
            # d_ready, arrived by these same threads after this fence
            if cutlass.const_expr(self.pro_reorder):
                # Only their writers fence: the X-tile warps (C) and, on flush rows, everyone
                # (Kt). The fence is also a CTA memory barrier (MEMBAR.ALL.CTA), which would hold
                # the ring-row warps until their k-ring appends are acknowledged.
                if mt == 1:
                    cute.arch.fence_view_async_shared()
                else:
                    if is_flush:
                        cute.arch.fence_view_async_shared()
            else:
                cute.arch.fence_view_async_shared()

            # g appends (head gh x token gt) and window weights (head wh x slot wj): with
            # pro_reorder on the ring-row lanes, which load their g values past the role split;
            # else on the gate lanes / lanes 32-95, from the entry loads
            if cutlass.const_expr(self.pro_reorder):
                if ct < 32:
                    if gt < T:
                        if gh < HPC:
                            gnew = s_coef[gh * _COEF + _CO_GAM + gt]
                            if not is_flush:
                                if P > 0:
                                    gnew = gnew + gp_g
                            g_cache[ridx, hv0 + gh, (base + P + gt) & RING_MASK] = gnew
                if ct < 64:
                    w = Float32(0.0)
                    if wj < P:
                        w = cute.math.exp(gp_w - gj_w, fastmath=True)
                    s_coef[wh * _COEF + _CO_W + wj] = w
                    if wj == 0:
                        egp = Float32(1.0)
                        if P > 0:
                            egp = cute.math.exp(gp_w, fastmath=True)
                        s_coef[wh * _COEF + _CO_EGP] = egp
            else:
                if ct >= 64:
                    if ct < 96:
                        # G_P needs the cursors: second round trip
                        if gt < T:
                            if gh < HPC:
                                gnew = s_coef[gh * _COEF + _CO_GAM + gt]
                                if not is_flush:
                                    if P > 0:
                                        gnew = gnew + gp_g
                                g_cache[ridx, hv0 + gh, (base + P + gt) & RING_MASK] = (
                                    gnew
                                )
                if ct >= 32:
                    if ct < 96:
                        w = Float32(0.0)
                        if wj < P:
                            w = cute.math.exp(gp_w - gj_w, fastmath=True)
                        s_coef[wh * _COEF + _CO_W + wj] = w
                        if wj == 0:
                            egp = Float32(1.0)
                            if P > 0:
                                egp = cute.math.exp(gp_w, fastmath=True)
                            s_coef[wh * _COEF + _CO_EGP] = egp
            cute.arch.barrier(barrier_id=1, number_of_threads=128)

            tbase = tmem_addr[0]
            ctx = dict(
                sidx=sidx,
                ridx=ridx,
                base=base,
                P=P,
                brow=brow,
                hv0=hv0,
                ct=ct,
                row=row,
                tbase=tbase,
                st_addr=st_addr,
                u_cache=u_cache,
                v=v,
                out=out,
                s_coef=s_coef,
                d_ready=d_ready,
                mma_done=mma_done,
                epi_done=epi_done,
                uw_addr=uw_addr,
                u_full=u_full,
                u_free=u_free,
                f_mma_a=f_mma_a,
                f_mma_b=f_mma_b,
                f_epi_a=f_epi_a,
                f_gdone=f_gdone,
                rd0=rd0,
                hv_base=hv_base,
                s_nrm=s_nrm,
                norm=(norm_z, norm_z_row, norm_w, norm_eps),
            )
            if is_flush:
                self._load_seed(ctx, rand_seed)
                if cutlass.const_expr(self.rmw_loop and self.roll_f):
                    # RMW-loop builds: heads and the RMW as loops (one copy of the RMW
                    # body); same steps as _head (flush)
                    for hh in cutlass.range(HPC, unroll=1):
                        vv = self._d_build(hh, 32, ctx, flush=True)
                        for g in cutlass.range(2, unroll=1):
                            egp, srow, r0, r1, r2, r3 = self._rmw_setup(hh, g, ctx)
                            self._rmw_wait(hh, g, ctx)
                            for gq in cutlass.range(self.ngrp, unroll=1):
                                r0, r1, r2, r3 = self._rmw_group(
                                    hh, g, gq, ctx, egp, srow, r0, r1, r2, r3
                                )
                            cute.arch.fence_view_async_shared()
                            cute.arch.mbarrier_arrive(
                                self._mb(ctx, "f_gdone", 2 * hh + g)
                            )
                        self._epilogue(hh, 0, ctx, vv, False)
                elif cutlass.const_expr(self.rmw_loop):
                    # RMW-loop builds, heads unrolled: one copy of the RMW loop per head
                    for hh in cutlass.range_constexpr(HPC):
                        vv = self._d_build(hh, 32, ctx, flush=True)
                        for g in cutlass.range(2, unroll=1):
                            egp, srow, r0, r1, r2, r3 = self._rmw_setup(hh, g, ctx)
                            self._rmw_wait(hh, g, ctx)
                            for gq in cutlass.range(self.ngrp, unroll=1):
                                r0, r1, r2, r3 = self._rmw_group(
                                    hh, g, gq, ctx, egp, srow, r0, r1, r2, r3
                                )
                            cute.arch.fence_view_async_shared()
                            cute.arch.mbarrier_arrive(
                                self._mb(ctx, "f_gdone", 2 * hh + g)
                            )
                        self._epilogue(hh, 0, ctx, vv, False)
                elif cutlass.const_expr(self.roll):
                    for hh in cutlass.range(HPC, unroll=1):
                        self._head(hh, True, ctx)
                else:
                    for hh in cutlass.range_constexpr(HPC):
                        self._head(hh, True, ctx)
            else:
                if cutlass.const_expr(self.roll):
                    for hh in cutlass.range(HPC, unroll=1):
                        self._head(hh, False, ctx)
                else:
                    for hh in cutlass.range_constexpr(HPC):
                        self._head(hh, False, ctx)
            U.fence_before_sync()
            cute.arch.mbarrier_arrive(tmem_done)

    @cute.kernel
    def kernel_early(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        state_idx_t: cute.Tensor,
        k_cache: cute.Tensor,
        u_cache: cute.Tensor,
        g_cache: cute.Tensor,
        hist_len: cute.Tensor,
        cache_base: cute.Tensor,
        row_order: cute.Tensor,
        out: cute.Tensor,
        rand_seed: cute.Tensor,
        scale: Float32,
        flush_min: Int32,
        hv_base: Int32,
        tma_ld: cpasync.TmaInfo,
        tma_st: cpasync.TmaInfo,
        ring_idx_t: cute.Tensor,
        conv_x: Int64,
        conv_w: Int64,
        conv_s: Int64,
        conv_win: Int64,
        conv_widx: Int64,
        conv_x_row: Int32,
        norm_z: Int64,
        norm_z_row: Int32,
        norm_w: Int64,
        norm_eps: Float32,
    ):
        """One (key head, row) CTA, ``early_split`` builds (register split, 4 heads per CTA,
        T >= 5): the same steps as ``kernel``, with warps 0 / 1 reading the cursors themselves
        and the math warps issuing their entry loads after setmaxnreg.inc."""
        T = self.T
        T2 = 2 * T

        tid, _, _ = cute.arch.thread_idx()
        bx, blk_row, _ = cute.arch.block_idx()
        HPC = self.hpc
        if cutlass.const_expr(self.nsub == 1):
            kh = bx  # key head
            sub = Int32(0)
        else:
            kh = bx // self.nsub  # key head
            sub = bx % self.nsub  # which hpc-head slice of it
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        if cutlass.const_expr(self.use_row_order):
            brow = Int32(row_order[blk_row])
        else:
            brow = blk_row
        hv0 = kh * self.hpk + sub * HPC

        # ------------------------------------------------------------------ SMEM
        smem = cutlass.utils.SmemAllocator()
        st_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((V_DIM, (64, 2)), stride=(64, (1, V_DIM * 64))),
        )
        s_state = smem.allocate_tensor(
            Float16, st_layout.outer, byte_alignment=1024, swizzle=st_layout.inner
        )
        s_x = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * 128), byte_alignment=1024
        )
        # K_hist tile: window keys k̂_j (j < P, else 0) row-major [16 slots x 128] fp16 in the
        # X-tile SW128 layout; the fold reads it as an MN-major B operand (N = k index)
        s_kt = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * 128), byte_alignment=1024
        )
        s_c = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * 16), byte_alignment=256
        )
        s_gram = smem.allocate_array(Float32, 32 * 16)
        s_coef = smem.allocate_array(Float32, HPK * _COEF)
        s_rn = smem.allocate_array(Float32, 16)
        # u windows of two heads: [buf][logical slot j][V] (ring dtype), bulk-copied by warp 0
        s_uwin = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(2 * W_RING * V_DIM), byte_alignment=128
        )
        u_full = self._mbar_list(smem, HPK)
        u_free = self._mbar_list(smem, 2)
        st_full = self._mbar_list(smem, 2 * HPK)
        mma_slot = self._mbar_list(smem, 2 * HPK)
        if cutlass.const_expr(self.rmw_loop):
            # RMW loop: half g waits on [rd0 | mma_done][g HPK + hh] and
            # [f_mma_a | f_mma_b][g HPK + hh] (one array each)
            rdm = self._mbar_list(smem, 2 * HPK)
            rd0, mma_done = rdm[:HPK], rdm[HPK:]
        else:
            mma_done = self._mbar_list(smem, HPK)
        d_ready = self._mbar_list(smem, HPK)
        epi_done = self._mbar_list(smem, HPK)
        if cutlass.const_expr(self.rmw_loop):
            fab = self._mbar_list(smem, 2 * HPK)
            f_mma_a, f_mma_b = fab[:HPK], fab[HPK:]
        else:
            f_mma_a = self._mbar_list(smem, HPK)
            f_mma_b = self._mbar_list(smem, HPK)
        f_epi_a = self._mbar_list(smem, HPK)
        if cutlass.const_expr(not self.rmw_loop):
            rd0 = self._mbar_list(smem, HPK)
        f_gdone = self._mbar_list(smem, 2 * HPK)
        x_ready = smem.allocate_array(cutlass.Int64, 1)
        tmem_done = smem.allocate_array(cutlass.Int64, 1)
        tmem_addr = smem.allocate(Int32, 4)

        # TMEM first: the next CTA on this SM is not launched before this one relinquishes
        # its TMEM allocation permit (measured: allocating after the prologue spaced the
        # CTAs of one SM ~2.5 us apart and left a quarter of the CTA slots empty)
        if warp == 1:
            U.tmem_alloc(tmem_addr, _TMEM_COLS)
            cute.arch.relinquish_tmem_alloc_permit()
        if cutlass.const_expr(self.pdl):
            # Every global read is below; q / k / v / a / b come from the kernel still in flight.
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # Prologue, two orders. Without pro_reorder: the cursors first (one global round trip,
        # issued by every thread), then every mbarrier while that round trip is in flight,
        # then the entry loads. With pro_reorder: the entry loads first and the cursors last
        # (see below), and warp 0's inits once the cursors are known, around head 0's state.
        # Warp 0 publishes the inits through one named barrier per role (2: warp 1 + the X-tile
        # warps; 3: the ring-row warps), each synced before that role's first mbarrier use --
        # no CTA-wide sync, which a warp only reaches once its outstanding loads have landed.
        # Warp 1 shares the X-tile warps' barrier at no cost: its next step waits for x_ready,
        # which they arrive on right after it. Named barriers 1-3 only (0 stays the CTA-wide
        # one, unused): SM107 (Rubin) has 16 per SM, so ids up to 5 would cap it at 2 CTAs/SM
        # (ncu "Block Limit Barriers") below the 4 that TMEM / SMEM / registers allow.
        # A barrier instance whose participants reach it at more than one instruction (2/128:
        # warp 0's arrive + warp 1's and the X-tile warps' syncs; 3/160 and 3/96: arrives + the
        # ring-row warps' sync; 1/128 without pro_reorder: the X-tile and ring-row warps' syncs)
        # uses the non-aligned forms (U.bar_sync_ua, barrier_arrive(aligned=False)): bar.sync /
        # bar.arrive are .aligned, which requires all participants to execute the same
        # instruction. The single-site instances (1/128 after the role split, 2/64) stay aligned.
        mbar = dict(
            st_full=st_full,
            mma_slot=mma_slot,
            f_gdone=f_gdone,
            u_full=u_full,
            mma_done=mma_done,
            d_ready=d_ready,
            epi_done=epi_done,
            f_mma_a=f_mma_a,
            f_mma_b=f_mma_b,
            f_epi_a=f_epi_a,
            rd0=rd0,
            u_free=u_free,
            x_ready=x_ready,
            tmem_done=tmem_done,
        )
        T_ = self.T
        CW0 = self.cw0
        st_addr = s_state.iterator.toint()
        x_addr = s_x.iterator.toint()
        kt_addr = s_kt.iterator.toint()
        c_addr = s_c.iterator.toint()
        uw_addr = s_uwin.iterator.toint()
        u_g = u_cache.iterator.toint()

        if warp < CW0:
            cute.arch.setmaxregister_decrease(self.regsplit[0])
            if warp == 0:
                # early_split: warp 0's prologue at 24 registers (no load outstanding at its
                # setmaxnreg.dec, so the math warps' .inc does not wait for a round trip)
                cpasync.prefetch_descriptor(tma_ld.atom)
                sidx = Int32(state_idx_t[brow])
                P = Int32(hist_len[brow])
                base = Int32(cache_base[brow])
                with cute.arch.elect_one():
                    self._init_mbarriers(mbar, 0)
                cute.arch.fence_acq_rel_cta()
                U.exit_if_negative(sidx)
                cute.arch.barrier_arrive(
                    barrier_id=2, number_of_threads=128, aligned=False
                )
                cute.arch.barrier_arrive(
                    barrier_id=3, number_of_threads=160, aligned=False
                )
                is_flush = flush_min <= P
                # the two u windows (4 KB) ahead of the two state halves (32 KB): head 0's D
                # build needs its window long before the readout needs the state
                for hh in cutlass.range_constexpr(min(2, HPC)):
                    if cutlass.const_expr(self.vsmem):
                        # verify rows: wv u slots + the v rows (issued behind the state)
                        with cute.arch.elect_one():
                            self._u_window_copy(
                                u_cache,
                                u_g,
                                uw_addr,
                                u_full[hh],
                                sidx,
                                hv0 + hh,
                                base,
                                hh,
                                _sel(is_flush, W_RING, self.wv, Int32),
                                _sel(is_flush, 0, T * V_DIM * 2, Int32),
                            )
                    else:
                        with cute.arch.elect_one():
                            self._u_window_copy(
                                u_cache,
                                u_g,
                                uw_addr,
                                u_full[hh],
                                sidx,
                                hv0 + hh,
                                base,
                                hh,
                            )
                for g in cutlass.range_constexpr(2):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(st_full[g], 16384)
                    U.simple_tma_copy(
                        tma_ld.atom,
                        tma_ld.tma_tensor[sidx, hv0, None, (None, g)],
                        s_state[None, (None, g)],
                        st_full[g],
                    )
                if cutlass.const_expr(self.vsmem):
                    # verify rows: heads 0 / 1's v rows into their windows' free slots (their
                    # u_full expect-tx above already counts them)
                    if not is_flush:
                        with cute.arch.elect_one():
                            for hh in cutlass.range_constexpr(min(2, HPC)):
                                self._v_rows_copy(
                                    v, uw_addr, u_full[hh], brow, hv0 + hh, hh
                                )
                # Every head's v rows into L2 (evict_last), behind the state: the math warps load
                # head h's v at the start of its D build and need it in its epilogue; from DRAM
                # that round trip was exposed there (Rubin: the bf16 conversions of v held 14-25 %
                # of the stall samples at T = 1-8). One bulk prefetch per token covers the CTA's
                # heads (contiguous in [B, T, HV, V]); no registers, no scoreboard. A register
                # double buffer (head h + 1's loads issued in head h) does not work here: ptxas
                # puts every v load on one scoreboard, so head h's first use also waits for head
                # h + 1's loads. (vsmem verify rows: it feeds warp 0's v-row copies of heads 2 / 3.)
                v_g = v.iterator.toint()
                with cute.arch.elect_one():
                    for t in cutlass.range_constexpr(T):
                        U.bulk_prefetch_l2(
                            v_g + Int64(cute.crd2idx((brow, t, hv0, 0), v.layout)) * 2,
                            HPC * V_DIM * 2,
                        )
                # ============================================================ producer
                if is_flush:
                    cpasync.prefetch_descriptor(tma_st.atom)
                    for slot in cutlass.range_constexpr(2, 2 * HPC):
                        prev = slot - 2
                        cute.arch.mbarrier_wait(f_gdone[prev], 0)
                        U.simple_tma_copy(
                            tma_st.atom,
                            s_state[None, (None, prev % 2)],
                            tma_st.tma_tensor[
                                sidx, hv0 + prev // 2, None, (None, prev % 2)
                            ],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                st_full[slot], 16384
                            )
                        U.simple_tma_copy(
                            tma_ld.atom,
                            tma_ld.tma_tensor[
                                sidx, hv0 + slot // 2, None, (None, slot % 2)
                            ],
                            s_state[None, (None, slot % 2)],
                            st_full[slot],
                        )
                        if cutlass.const_expr(slot % 2 == 0 and slot // 2 >= 2):
                            hh = slot // 2
                            cute.arch.mbarrier_wait(u_free[hh - 2], 0)
                            with cute.arch.elect_one():
                                self._u_window_copy(
                                    u_cache,
                                    u_g,
                                    uw_addr,
                                    u_full[hh],
                                    sidx,
                                    hv0 + hh,
                                    base,
                                    hh % 2,
                                )
                    for prev in cutlass.range_constexpr(2 * HPC - 2, 2 * HPC):
                        cute.arch.mbarrier_wait(f_gdone[prev], 0)
                        U.simple_tma_copy(
                            tma_st.atom,
                            s_state[None, (None, prev % 2)],
                            tma_st.tma_tensor[
                                sidx, hv0 + prev // 2, None, (None, prev % 2)
                            ],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)
                else:
                    for slot in cutlass.range_constexpr(2, 2 * HPC):
                        cute.arch.mbarrier_wait(mma_slot[slot - 2], 0)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                st_full[slot], 16384
                            )
                        U.simple_tma_copy(
                            tma_ld.atom,
                            tma_ld.tma_tensor[
                                sidx, hv0 + slot // 2, None, (None, slot % 2)
                            ],
                            s_state[None, (None, slot % 2)],
                            st_full[slot],
                        )
                        # u window behind this state slot, if any (see _u_window_head)
                        hh = self._u_window_head(slot)
                        if cutlass.const_expr(hh >= 0):
                            cute.arch.mbarrier_wait(u_free[hh - 2], 0)
                            if cutlass.const_expr(self.vsmem):
                                with cute.arch.elect_one():
                                    self._u_window_copy(
                                        u_cache,
                                        u_g,
                                        uw_addr,
                                        u_full[hh],
                                        sidx,
                                        hv0 + hh,
                                        base,
                                        hh % 2,
                                        self.wv,
                                        T * V_DIM * 2,
                                    )
                                    self._v_rows_copy(
                                        v, uw_addr, u_full[hh], brow, hv0 + hh, hh % 2
                                    )
                            else:
                                with cute.arch.elect_one():
                                    self._u_window_copy(
                                        u_cache,
                                        u_g,
                                        uw_addr,
                                        u_full[hh],
                                        sidx,
                                        hv0 + hh,
                                        base,
                                        hh % 2,
                                    )
            elif warp == 1:
                sidx = Int32(state_idx_t[brow])
                P = Int32(hist_len[brow])
                if sidx < 0:
                    # padded row: hand the TMEM columns back before the CTA exits
                    cute.arch.sync_warp()
                    U.tmem_dealloc(tmem_addr[0], _TMEM_COLS)
                U.exit_if_negative(sidx)
                is_flush = flush_min <= P
                # ============================================================ UMMA issuer
                tbase = tmem_addr[0]
                sd = U.sdesc_sw128(V_DIM * 128)
                x_base = sd | Uint64(x_addr >> 4)
                stage_base = [
                    sd | Uint64(st_addr >> 4),
                    sd | Uint64((st_addr + 16384) >> 4),
                ]
                c_desc = U.sdesc_sw32(Uint64(c_addr >> 4))
                # MN-major SW128: SBO = 1024 B between 8-slot K atoms, N halves 2048 B apart
                kt_desc = U.sdesc_mn_sw128(2048, 1024) | Uint64(kt_addr >> 4)
                idesc16 = U.f16_idesc(128, 16)
                idesc64mn = U.f16_idesc(128, 64, b_mn_major=True)
                U.bar_sync_ua(2, 128)  # warp 0's inits
                cute.arch.mbarrier_wait(x_ready, 0)
                U.fence_after_sync()
                if is_flush:
                    # per head: fold half 0 (needs only D, Kt) -> readout half 0 (rd0: the
                    # consumers' RMW of stage 0 may start) -> readout half 1 + history
                    # (mma_done) -> fold half 1 once the RMW has read fold half 0 (f_epi_a).
                    # R/Y (cols 0..31) and F (64..127) of head h + 1 wait for head h's epilogue.
                    for hh in cutlass.range_constexpr(HPC):
                        cute.arch.mbarrier_wait(d_ready[hh], 0)
                        if cutlass.const_expr(hh > 0):
                            cute.arch.mbarrier_wait(epi_done[hh - 1], 0)
                        U.fence_after_sync()
                        U.umma_ts(tbase + 64, tbase + 32, kt_desc, idesc64mn, False)
                        U.umma_commit(f_mma_a[hh])
                        for g in cutlass.range_constexpr(2):
                            slot = 2 * hh + g
                            cute.arch.mbarrier_wait(st_full[slot], 0)
                            U.fence_after_sync()
                            for ka in cutlass.range_constexpr(4):
                                U.umma_ss(
                                    tbase + 0,
                                    stage_base[g] | Uint64((ka * 32) >> 4),
                                    x_base | Uint64((g * 2048 + ka * 32) >> 4),
                                    idesc16,
                                    (g > 0) or (ka > 0),
                                )
                            if cutlass.const_expr(g == 0):
                                U.umma_commit(rd0[hh])
                        U.umma_ts(tbase + 16, tbase + 32, c_desc, idesc16, False)
                        U.umma_commit(mma_done[hh])
                        cute.arch.mbarrier_wait(f_epi_a[hh], 0)
                        U.fence_after_sync()
                        U.umma_ts(
                            tbase + 64,
                            tbase + 32,
                            kt_desc + Uint64(2048 >> 4),
                            idesc64mn,
                            False,
                        )
                        U.umma_commit(f_mma_b[hh])
                else:
                    for hh in cutlass.range_constexpr(HPC):
                        rcol = (hh % 2) * 32
                        if cutlass.const_expr(hh >= 2):
                            cute.arch.mbarrier_wait(epi_done[hh - 2], 0)
                            U.fence_after_sync()
                        for g in cutlass.range_constexpr(2):
                            slot = 2 * hh + g
                            cute.arch.mbarrier_wait(st_full[slot], 0)
                            U.fence_after_sync()
                            for ka in cutlass.range_constexpr(4):
                                U.umma_ss(
                                    tbase + rcol,
                                    stage_base[g] | Uint64((ka * 32) >> 4),
                                    x_base | Uint64((g * 2048 + ka * 32) >> 4),
                                    idesc16,
                                    (g > 0) or (ka > 0),
                                )
                            U.umma_commit(mma_slot[slot])
                        cute.arch.mbarrier_wait(d_ready[hh], 0)
                        U.fence_after_sync()
                        U.umma_ts(
                            tbase + rcol + 16,
                            tbase + 64 + (hh % 2) * 8,
                            c_desc,
                            idesc16,
                            False,
                        )
                        U.umma_commit(mma_done[hh])
                cute.arch.mbarrier_wait(tmem_done, 0)
                U.fence_after_sync()
                U.tmem_dealloc(tbase, _TMEM_COLS)
        else:
            cute.arch.setmaxregister_increase(self.regsplit[1])
            # entry loads at 104 registers in the prologue reorder's order: X rows (X-tile warps), gate
            # inputs and gate constants (gate warp), cursors last; the ring-row warps' X rows with their
            # ring rows once the cursors are known
            grp_e = lane >> 2
            c4_e = lane & 3
            cw_e = (warp - CW0) & 3
            nt_e = cw_e & 1
            q_g = q.iterator.toint()
            k_g = k.iterator.toint()
            xr_b = nt_e * 8 + grp_e
            is_xt = Int32(warp >= CW0 + 2)
            gate_pred = Int32(warp == CW0 + 2)
            x_own = self._x_row_addr(k, q, k_g, q_g, brow, kh, xr_b)
            x_oth = self._x_row_addr(k, q, k_g, q_g, brow, kh, (1 - nt_e) * 8 + grp_e)
            wb = []
            wo = []
            for p in cutlass.range_constexpr(4):
                koff = Int64((32 * p + 8 * c4_e) * 2)
                wb.append(U.ldg_v4_if(is_xt, x_own + koff))
                wo.append(U.ldg_v4_if(is_xt, x_oth + koff))
            ct_e = (tid - 32 * CW0) & 127
            gh = (ct_e >> 3) & 3
            ghc = gh % HPC
            gt = ct_e & 7
            tt = _sel(gt < T_, gt, 0, Int32)
            ab_off = Int64(cute.crd2idx((brow, tt, hv0 + ghc), a.layout)) * 2
            av_raw = U.ldg_u16_if(gate_pred, a.iterator.toint() + ab_off)
            bv_raw = U.ldg_u16_if(gate_pred, b.iterator.toint() + ab_off)
            alog_e = U.ldg_f32_if(
                gate_pred,
                a_log.iterator.toint()
                + Int64(cute.crd2idx((hv0 + ghc,), a_log.layout)) * 4,
            )
            dtb_e = U.ldg_f32_if(
                gate_pred,
                dt_bias.iterator.toint()
                + Int64(cute.crd2idx((hv0 + ghc,), dt_bias.layout)) * 4,
            )
            sidx = Int32(state_idx_t[brow])
            P = Int32(hist_len[brow])
            base = Int32(cache_base[brow])
            U.exit_if_negative(sidx)
            is_flush = flush_min <= P
            is_rr = Int32(warp == CW0) + Int32(warp == CW0 + 1)
            kc_g = k_cache.iterator.toint()
            wr1_e = self._ring_row_loads(
                k_cache, kc_g, is_rr, sidx, kh, base, grp_e, c4_e, wo, wb, x_own
            )
            wh, wj, gp_g, gp_w, gj_w = self._window_g_loads(
                g_cache, is_rr, ct_e, sidx, hv0, ghc, base, P
            )
            # ============================================================ consumers
            cw = warp - CW0
            ct = cw * 32 + lane  # TMEM lane address / k index for staging
            row = ((warp & 3) << 5) | lane  # V row held by this thread (= TMEM lane)
            grp = lane >> 2
            c4 = lane & 3

            # -------- Gram on mma.sync: rows [k̂_ring(16); X(16)] x cols X(16), K = 128.
            # Fragments come straight from global memory: per 32-wide k block a lane loads 16 B
            # (8 contiguous k) of each of its rows; the k order inside the m16n8k16 fragment is
            # permuted identically for A and B, which a dot product does not see.
            # Warps mt == 1 own the X x X tiles (norms, intra-step dots): they depend on the
            # batch row only, so the X tile and x_ready follow ONE global round trip.  Warps
            # mt == 0 own the ring x X tiles from the rows issued once the cursors landed.
            mt = cw >> 1
            nt = cw & 1
            gc = nt * 8 + 2 * c4
            if mt == 1:
                # A rows grp / grp + 8 of the X x X tile: this lane's own row (wb) and the
                # other half's (wo), both loaded at entry
                accx = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                for p in cutlass.range_constexpr(4):
                    for hs in cutlass.range_constexpr(2):
                        w0a = _sel(nt == 0, wb[p][2 * hs], wo[p][2 * hs], Uint32)
                        w1a = _sel(nt == 0, wo[p][2 * hs], wb[p][2 * hs], Uint32)
                        w0b = _sel(
                            nt == 0, wb[p][2 * hs + 1], wo[p][2 * hs + 1], Uint32
                        )
                        w1b = _sel(
                            nt == 0, wo[p][2 * hs + 1], wb[p][2 * hs + 1], Uint32
                        )
                        accx = U.mma_bf16_16816(
                            w0a, w1a, w0b, w1b, wb[p][2 * hs], wb[p][2 * hs + 1], accx
                        )
                s_gram[(16 + grp) * 16 + gc] = accx[0]
                s_gram[(16 + grp) * 16 + gc + 1] = accx[1]
                s_gram[(24 + grp) * 16 + gc] = accx[2]
                s_gram[(24 + grp) * 16 + gc + 1] = accx[3]
                # The ring-row warps read the X x X diagonal (their rows' norms): arrive
                # only. A sync here waited for them, and they reach any barrier only once
                # their ring rows have landed (ptxas moves those registers into the mma
                # operands first), which put x_ready -- and the first readout -- a round
                # trip late. This warp's own diagonal (rn_x) needs a warp sync only.
                cute.arch.barrier_arrive(
                    barrier_id=3, number_of_threads=160, aligned=False
                )
                cute.arch.sync_warp()
                # X tile (fp16, SW128) and x_ready from these warps' own rows: they hold the
                # same rows as the ring-row warps and have nothing in flight, so the readout
                # can start once the X x X Gram is done instead of after the ring rows
                rn_x = cute.math.rsqrt(s_gram[(16 + xr_b) * 16 + xr_b] + Float32(_EPS))
                rn_x = _sel(xr_b >= T, rn_x * scale, rn_x, Float32)
                for p in cutlass.range_constexpr(4):
                    kc = 4 * p + c4
                    xw = []
                    for i in cutlass.range_constexpr(4):
                        xw.append(
                            U.pack_f16x2(
                                U.bf16x2_lo_f32(wb[p][i]) * rn_x,
                                U.bf16x2_hi_f32(wb[p][i]) * rn_x,
                            )
                        )
                    xb = (kc // 8) * 2048 + xr_b * 128 + (((kc % 8) ^ (xr_b % 8)) * 16)
                    U.sts_v4(x_addr + xb, xw[0], xw[1], xw[2], xw[3])
                U.bar_sync_ua(2, 128)  # warp 0's inits (+ warp 1's sync: two sites)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(x_ready)
                # Everything below needs only the X x X Gram and the gate inputs (one round
                # trip), so it overlaps warps mt == 0 waiting for the ring rows.
                cl = ct - 64
                if cl < T2:
                    rn_own = cute.math.rsqrt(
                        s_gram[(16 + cl) * 16 + cl] + Float32(_EPS)
                    )
                    if cl >= T:
                        rn_own = rn_own * scale
                    s_rn[cl] = rn_own
                if cl < 32:
                    # gates: lane = head gh x token gt
                    av = U.u16_as_f32(av_raw, True)
                    bv = U.u16_as_f32(bv_raw, True)
                    alog = alog_e
                    dtb = dtb_e
                    xx = av + dtb
                    sp = xx
                    if xx <= Float32(20.0):
                        sp = cute.math.log(
                            Float32(1.0) + cute.math.exp(xx, fastmath=True),
                            fastmath=True,
                        )
                    la = -cute.math.exp(alog, fastmath=True) * sp
                    if gt >= T:
                        la = Float32(0.0)
                    beta = Float32(1.0) / (
                        Float32(1.0) + cute.math.exp(-bv, fastmath=True)
                    )
                    gam = la
                    for off in (1, 2, 4):
                        lower = cute.arch.shuffle_sync_up(gam, off, mask_and_clamp=0)
                        if gt >= off:
                            gam = gam + lower
                    eg = cute.math.exp(gam, fastmath=True)
                    cbg = gh * _COEF
                    s_coef[cbg + _CO_BETA + gt] = beta
                    s_coef[cbg + _CO_BG + gt] = beta * eg
                    s_coef[cbg + _CO_EG + gt] = eg
                    s_coef[cbg + _CO_GAM + gt] = gam
                # id 2 again, now among the 64 X-tile threads only: both passed its first
                # (128-thread) instance above, so that instance has completed
                cute.arch.barrier(barrier_id=2, number_of_threads=64)
                # intra-step coefficient tables A[t][s], Bq[t][s] per head (64 threads)
                NT = HPC * T * T
                for it in cutlass.range_constexpr((NT + 63) // 64):
                    idx = cl + it * 64
                    if idx < NT:
                        ch = idx // (T * T)
                        t = (idx // T) % T
                        s = idx % T
                        cb = ch * _COEF
                        dec = cute.math.exp(
                            s_coef[cb + _CO_GAM + t] - s_coef[cb + _CO_GAM + s],
                            fastmath=True,
                        )
                        kk = s_gram[(16 + s) * 16 + t] * s_rn[s] * s_rn[t]
                        qk = s_gram[(16 + s) * 16 + T + t] * s_rn[s] * s_rn[T + t]
                        av_ = Float32(0.0)
                        if s < t:
                            av_ = s_coef[cb + _CO_BETA + t] * dec * kk
                        bq_ = Float32(0.0)
                        if s <= t:
                            bq_ = dec * qk
                        s_coef[cb + _CO_A + t * 8 + s] = av_
                        s_coef[cb + _CO_BQ + t * 8 + s] = bq_
            else:
                wr0 = wo  # ring rows grp / grp + 8, issued when the cursors landed
                wr1 = wr1_e
                # the X-tile warps' X x X Gram and warp 0's inits (before this role's first
                # mbarrier use)
                U.bar_sync_ua(3, 160)
                # norm of this lane's X row (for the k̂ ring append below)
                rn_b = cute.math.rsqrt(s_gram[(16 + xr_b) * 16 + xr_b] + Float32(_EPS))
                rn_b = _sel(xr_b >= T, rn_b * scale, rn_b, Float32)
                # ring x X tiles
                accr = self._ring_gram(wr0, wr1, wb, rn_b)
                s_gram[grp * 16 + gc] = accr[0]
                s_gram[grp * 16 + gc + 1] = accr[1]
                s_gram[(grp + 8) * 16 + gc] = accr[2]
                s_gram[(grp + 8) * 16 + gc + 1] = accr[3]
                # k̂ ring appends (ring dtype, 16 B per lane)
                if xr_b < T:
                    if sub == 0:
                        kdst = (
                            kc_g
                            + Int64(
                                cute.crd2idx(
                                    (sidx, kh, (base + P + xr_b) & RING_MASK, 0),
                                    k_cache.layout,
                                )
                            )
                            * 2
                        )
                        for p in cutlass.range_constexpr(4):
                            kw4 = self._k_append_words(wb[p], rn_b)
                            U.stg_v4(
                                kdst + Int64((32 * p + 8 * c4) * 2),
                                kw4[0],
                                kw4[1],
                                kw4[2],
                                kw4[3],
                            )
            cute.arch.barrier(barrier_id=1, number_of_threads=128)

            # -------- after the ring tiles: C tile (warps mt == 1), window weights, g-ring
            # appends, Kt (flush rows)
            if mt == 1:
                cl = ct - 64
                for i in cutlass.range_constexpr(4):
                    e = cl * 4 + i
                    xc = e >> 4
                    j = e & 15
                    cval = Float32(0.0)
                    if j < P:
                        if xc < T2:
                            if cutlass.const_expr(self.k_f16):
                                cval = s_gram[j * 16 + xc]  # ring x X̂: normalized already
                            else:
                                rnx = cute.math.rsqrt(
                                    s_gram[(16 + xc) * 16 + xc] + Float32(_EPS)
                                )
                                if xc >= T:
                                    rnx = rnx * scale
                                cval = s_gram[j * 16 + xc] * rnx
                    cbyte = xc * 32 + (((j // 8) ^ ((xc >> 2) & 1)) * 16) + (j % 8) * 2
                    U.sts_u16(c_addr + cbyte, U.f32_to_u16(cval, False))
            if is_flush:
                # K_hist tile (fold B operand): thread -> (slot row jr = ct / 8, 16 B chunk c8) of both
                # K groups; ring rows were just read by the Gram (L1/L2 hits); slots j >= P zeroed
                jr = ct >> 3
                c8 = ct & 7
                krow = (
                    kc_g
                    + Int64(
                        cute.crd2idx(
                            (sidx, kh, (base + jr) & RING_MASK, c8 * 8), k_cache.layout
                        )
                    )
                    * 2
                )
                kr = [U.ldg_v4(krow), U.ldg_v4(krow + Int64(128))]
                for gk in cutlass.range_constexpr(2):
                    kw4 = self._kt_words(kr[gk], jr < P)
                    U.sts_v4(
                        kt_addr + gk * 2048 + jr * 128 + ((c8 ^ (jr % 8)) * 16),
                        kw4[0],
                        kw4[1],
                        kw4[2],
                        kw4[3],
                    )
            # C / Kt are read by the history / fold UMMAs, which warp 1 issues only after
            # d_ready, arrived by these same threads after this fence
            # Only their writers fence: the X-tile warps (C) and, on flush rows, everyone
            # (Kt). The fence is also a CTA memory barrier (MEMBAR.ALL.CTA), which would hold
            # the ring-row warps until their k-ring appends are acknowledged.
            if mt == 1:
                cute.arch.fence_view_async_shared()
            else:
                if is_flush:
                    cute.arch.fence_view_async_shared()

            # g appends (head gh x token gt) and window weights (head wh x slot wj) on the
            # ring-row lanes, which loaded their g values past the role split
            if ct < 32:
                if gt < T:
                    if gh < HPC:
                        gnew = s_coef[gh * _COEF + _CO_GAM + gt]
                        if not is_flush:
                            if P > 0:
                                gnew = gnew + gp_g
                        g_cache[sidx, hv0 + gh, (base + P + gt) & RING_MASK] = gnew
            if ct < 64:
                w = Float32(0.0)
                if wj < P:
                    w = cute.math.exp(gp_w - gj_w, fastmath=True)
                s_coef[wh * _COEF + _CO_W + wj] = w
                if wj == 0:
                    egp = Float32(1.0)
                    if P > 0:
                        egp = cute.math.exp(gp_w, fastmath=True)
                    s_coef[wh * _COEF + _CO_EGP] = egp
            cute.arch.barrier(barrier_id=1, number_of_threads=128)

            tbase = tmem_addr[0]
            ctx = dict(
                sidx=sidx,
                base=base,
                P=P,
                brow=brow,
                hv0=hv0,
                ct=ct,
                row=row,
                tbase=tbase,
                st_addr=st_addr,
                u_cache=u_cache,
                v=v,
                out=out,
                s_coef=s_coef,
                d_ready=d_ready,
                mma_done=mma_done,
                epi_done=epi_done,
                uw_addr=uw_addr,
                u_full=u_full,
                u_free=u_free,
                f_mma_a=f_mma_a,
                f_mma_b=f_mma_b,
                f_epi_a=f_epi_a,
                f_gdone=f_gdone,
                rd0=rd0,
                hv_base=hv_base,
            )
            if is_flush:
                self._load_seed(ctx, rand_seed)
                if cutlass.const_expr(self.rmw_loop and self.roll_f):
                    # RMW-loop builds: heads and the RMW as loops (one copy of the RMW
                    # body); same steps as _head (flush)
                    for hh in cutlass.range(HPC, unroll=1):
                        vv = self._d_build(hh, 32, ctx, flush=True)
                        for g in cutlass.range(2, unroll=1):
                            egp, srow, r0, r1, r2, r3 = self._rmw_setup(hh, g, ctx)
                            self._rmw_wait(hh, g, ctx)
                            for gq in cutlass.range(self.ngrp, unroll=1):
                                r0, r1, r2, r3 = self._rmw_group(
                                    hh, g, gq, ctx, egp, srow, r0, r1, r2, r3
                                )
                            cute.arch.fence_view_async_shared()
                            cute.arch.mbarrier_arrive(
                                self._mb(ctx, "f_gdone", 2 * hh + g)
                            )
                        self._epilogue(hh, 0, ctx, vv, False)
                elif cutlass.const_expr(self.rmw_loop):
                    # RMW-loop builds, heads unrolled: one copy of the RMW loop per head
                    for hh in cutlass.range_constexpr(HPC):
                        vv = self._d_build(hh, 32, ctx, flush=True)
                        for g in cutlass.range(2, unroll=1):
                            egp, srow, r0, r1, r2, r3 = self._rmw_setup(hh, g, ctx)
                            self._rmw_wait(hh, g, ctx)
                            for gq in cutlass.range(self.ngrp, unroll=1):
                                r0, r1, r2, r3 = self._rmw_group(
                                    hh, g, gq, ctx, egp, srow, r0, r1, r2, r3
                                )
                            cute.arch.fence_view_async_shared()
                            cute.arch.mbarrier_arrive(
                                self._mb(ctx, "f_gdone", 2 * hh + g)
                            )
                        self._epilogue(hh, 0, ctx, vv, False)
                elif cutlass.const_expr(self.roll):
                    for hh in cutlass.range(HPC, unroll=1):
                        self._head(hh, True, ctx)
                else:
                    for hh in cutlass.range_constexpr(HPC):
                        self._head(hh, True, ctx)
            else:
                if cutlass.const_expr(self.roll):
                    for hh in cutlass.range(HPC, unroll=1):
                        self._head(hh, False, ctx)
                else:
                    for hh in cutlass.range_constexpr(HPC):
                        self._head(hh, False, ctx)
            U.fence_before_sync()
            cute.arch.mbarrier_arrive(tmem_done)

    # ================================================================== persistent
    # kernel_persist: G = min(H * B, 4 x the stream's SMs) CTAs over the (key head, row) items
    # (item i: key head i % H, row i // H, through row_order): each CTA's first item is its own
    # index, the next ones claimed from a self-resetting counter (dynamic load balance: a flush row
    # takes ~2x a verify row; a static stride CTA + n G measured 6 % slower at T = 8, B = 2048).
    # Scope: 1 <= T <= 8, 4 heads per CTA = all of a key head's, every rounding mode (at T <= 4 with
    # the T >= 5 row structure); verify, flush and padded rows in any mix and order. Per CTA, once:
    # warp 1 allocates TMEM, warp 0 initializes the mbarriers (published to warps 1 / 2 and the
    # math warps through one barrier 1 instance of 224 threads), one setmaxnreg split. This
    # one-time set-up (and the TMEM release at the end) is what the persistent kernel saves per
    # item over one CTA per item. Warp 2 (idle in the split) is the
    # scheduler: it loads each item's cursors, skips padded rows and publishes the rows through a
    # 2-entry SMEM mailbox (r_full / r_empty; readers: warps 0, 1 and the four math warps), then a
    # sentinel (valid = 0). Each row runs kernel_early's steps for its kind (flush: P >= flush_min).
    # Every role counts its rows (j) and flush rows (nf); an mbarrier's phase parity is the count of
    # the rows that complete it:
    #   every row:        st_full, u_full, d_ready, epi_done, mma_done, x_ready   -> j & 1
    #   flush rows only:  rd0, f_mma_a / f_mma_b, f_epi_a, f_gdone                -> nf & 1
    #   verify rows only: mma_slot                                                -> (j - nf) & 1
    #   twice per row:    u_free -- in the row (heads 0 / 1 -> heads 2 / 3's windows, parity 0) and
    #                     across rows (the last head -> the next row's heads 0 / 1, parity 1).
    # Every completed phase has a waiter (compute-sanitizer synccheck: "Missing wait" when a barrier
    # is re-arrived after a phase nobody waited for): the epilogue phases a row's own UMMA issue does
    # not wait for (verify: heads 2 / 3, flush: head 3) are waited by warp 1 at the next row's start,
    # mma_slot 6 / 7 (verify) and f_gdone 6 / 7 (flush) by warp 0 across rows. Across rows, warp 0
    # issues the next row's state slots 0 / 1 as the current row's slots 6 / 7 retire (verify: their
    # readouts; flush: their RMW, then their TMA stores have read the stage --
    # cp.async.bulk.wait_group.read), its head 0 / 1 windows once the last head released both
    # buffers, and L2 prefetches of its prologue inputs; warp 1 starts the next row's UMMAs once the
    # current row's epilogues have read TMEM and the next row's X tile is ready (x_ready, arrived
    # after every math thread finished the row: the cross-row fence for TMEM D / X / C / Kt).
    @cute.jit
    def call_persist(
        self,
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state,
        state_idx,
        k_cache,
        u_cache,
        g_cache,
        hist_len,
        cache_base,
        row_order,
        out,
        rand_seed,
        ctr,
        scale: Float32,
        flush_min: Int32,
        hv_base: Int32,
        n_cta: Int32,
        stream,
    ):
        tma_ld = self._make_state_tma(state, cpasync.CopyBulkTensorTileG2SOp())
        tma_st = self._make_state_tma(state, cpasync.CopyBulkTensorTileS2GOp())
        self.kernel_persist(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state_idx,
            k_cache,
            u_cache,
            g_cache,
            hist_len,
            cache_base,
            row_order,
            out,
            rand_seed,
            ctr,
            scale,
            flush_min,
            hv_base,
            tma_ld,
            tma_st,
        ).launch(
            grid=(n_cta, 1, 1),
            block=(self.num_threads, 1, 1),
            min_blocks_per_mp=_MIN_BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel_persist(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        state_idx_t: cute.Tensor,
        k_cache: cute.Tensor,
        u_cache: cute.Tensor,
        g_cache: cute.Tensor,
        hist_len: cute.Tensor,
        cache_base: cute.Tensor,
        row_order: cute.Tensor,
        out: cute.Tensor,
        rand_seed: cute.Tensor,
        ctr: cute.Tensor,
        scale: Float32,
        flush_min: Int32,
        hv_base: Int32,
        tma_ld: cpasync.TmaInfo,
        tma_st: cpasync.TmaInfo,
    ):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        n_cta, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(tid // 32)

        # ------------------------------------------------------------------ SMEM (kernel_early's
        # buffers and mbarriers, + the row mailbox)
        smem = cutlass.utils.SmemAllocator()
        st_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((V_DIM, (64, 2)), stride=(64, (1, V_DIM * 64))),
        )
        s_state = smem.allocate_tensor(
            Float16, st_layout.outer, byte_alignment=1024, swizzle=st_layout.inner
        )
        s_x = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * 128), byte_alignment=1024
        )
        s_kt = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * 128), byte_alignment=1024
        )
        s_c = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * 16), byte_alignment=256
        )
        s_gram = smem.allocate_array(Float32, 32 * 16)
        s_coef = smem.allocate_array(Float32, HPK * _COEF)
        s_rn = smem.allocate_array(Float32, 16)
        s_uwin = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(2 * W_RING * V_DIM), byte_alignment=128
        )
        u_full = self._mbar_list(smem, HPK)
        u_free = self._mbar_list(smem, 2)
        st_full = self._mbar_list(smem, 2 * HPK)
        mma_slot = self._mbar_list(smem, 2 * HPK)
        # rmw_loop (always, in the persistent build): [rd0 | mma_done] and [f_mma_a | f_mma_b] as
        # one array each (the RMW loop's dynamic half index)
        rdm = self._mbar_list(smem, 2 * HPK)
        rd0, mma_done = rdm[:HPK], rdm[HPK:]
        d_ready = self._mbar_list(smem, HPK)
        epi_done = self._mbar_list(smem, HPK)
        fab = self._mbar_list(smem, 2 * HPK)
        f_mma_a, f_mma_b = fab[:HPK], fab[HPK:]
        f_epi_a = self._mbar_list(smem, HPK)
        f_gdone = self._mbar_list(smem, 2 * HPK)
        x_ready = smem.allocate_array(cutlass.Int64, 1)
        tmem_done = smem.allocate_array(cutlass.Int64, 1)
        tmem_addr = smem.allocate(Int32, 4)
        # row mailbox: 2 entries x _PZ_RI words (brow, key head, state index, P, base, valid)
        rinfo = smem.allocate_array(Int32, 2 * _PZ_RI)
        r_full = smem.allocate_array(cutlass.Int64, 2)
        r_empty = smem.allocate_array(cutlass.Int64, 2)

        # TMEM first (the next CTA on the SM waits for this one's allocation permit)
        if warp == 1:
            U.tmem_alloc(tmem_addr, _TMEM_COLS)
            cute.arch.relinquish_tmem_alloc_permit()

        mbar = dict(
            st_full=st_full,
            mma_slot=mma_slot,
            f_gdone=f_gdone,
            u_full=u_full,
            mma_done=mma_done,
            d_ready=d_ready,
            epi_done=epi_done,
            f_mma_a=f_mma_a,
            f_mma_b=f_mma_b,
            f_epi_a=f_epi_a,
            rd0=rd0,
            u_free=u_free,
            x_ready=x_ready,
            tmem_done=tmem_done,
        )
        sm = dict(
            s_state=s_state,
            st_addr=s_state.iterator.toint(),
            x_addr=s_x.iterator.toint(),
            kt_addr=s_kt.iterator.toint(),
            c_addr=s_c.iterator.toint(),
            uw_addr=s_uwin.iterator.toint(),
            s_gram=s_gram,
            s_coef=s_coef,
            s_rn=s_rn,
            tmem_addr=tmem_addr,
            rinfo=rinfo,
            r_full=r_full,
            r_empty=r_empty,
            ctr=ctr.iterator.toint(),
        )
        n_items = Int32(q.shape[0]) * Int32(self.h)
        if warp < self.cw0:
            cute.arch.setmaxregister_decrease(self.regsplit[0])
            if warp == 0:
                with cute.arch.elect_one():
                    for s_ in cutlass.range_constexpr(2):
                        cute.arch.mbarrier_init(r_full + s_, 1)
                        cute.arch.mbarrier_init(r_empty + s_, _PZ_READERS)
                    self._init_mbarriers(mbar, 0)
                cute.arch.fence_acq_rel_cta()
                cute.arch.barrier_arrive(
                    barrier_id=1, number_of_threads=_PZ_INIT, aligned=False
                )
                self._pz_producer(
                    q,
                    k,
                    v,
                    a,
                    b,
                    k_cache,
                    u_cache,
                    g_cache,
                    tma_ld,
                    tma_st,
                    flush_min,
                    mbar,
                    sm,
                )
            elif warp == 1:
                U.bar_sync_ua(1, _PZ_INIT)
                self._pz_mma(flush_min, mbar, sm)
            elif warp == 2:
                U.bar_sync_ua(1, _PZ_INIT)
                self._pz_sched(
                    cta,
                    n_cta,
                    n_items,
                    state_idx_t,
                    hist_len,
                    cache_base,
                    row_order,
                    sm,
                )
        else:
            cute.arch.setmaxregister_increase(self.regsplit[1])
            U.bar_sync_ua(1, _PZ_INIT)
            self._pz_consumers(
                q,
                k,
                v,
                a,
                b,
                a_log,
                dt_bias,
                k_cache,
                u_cache,
                g_cache,
                out,
                rand_seed,
                scale,
                flush_min,
                hv_base,
                mbar,
                sm,
            )

    def _pz_read(self, sm, j):
        """Mailbox entry of row j (wait r_full, read, release r_empty with one arrive per warp):
        [brow, key head, state index, P, base, valid]."""
        s = j & 1
        cute.arch.mbarrier_wait(sm["r_full"] + s, (j >> 1) & 1)
        ri = sm["rinfo"]
        # warp-uniform (every lane reads the same words): ptxas keeps them in uniform registers
        f = [cute.arch.make_warp_uniform(Int32(ri[s * _PZ_RI + c])) for c in range(6)]
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(sm["r_empty"] + s)
        return f

    @cute.jit
    def _pz_sched(
        self,
        cta,
        n_cta,
        n_items,
        state_idx_t,
        hist_len,
        cache_base,
        row_order,
        sm,
    ):
        """Warp 2: the CTA's first item static (item = CTA), then each next item claimed
        (G + an atomic count) once a mailbox slot is free, about two rows ahead of its use (dynamic
        load balance across CTAs: flush rows take ~2x a verify row); padded rows skipped; then the
        sentinel and the counter's done count (the launch's last CTA zeroes it)."""
        ri = sm["rinfo"]
        npub = Int32(0)
        it = Int32(cta)
        fresh = Int32(1)  # `it` holds an item not yet published
        while it < n_items:
            sl = npub & 1
            cute.arch.mbarrier_wait(sm["r_empty"] + sl, ((npub >> 1) & 1) ^ 1)
            if fresh == 0:
                it = n_cta + U.claim_next(sm["ctr"])
            fresh = Int32(0)
            if it < n_items:
                bx = it % self.h
                blk_row = it // self.h
                brow = blk_row
                if cutlass.const_expr(self.use_row_order):
                    brow = Int32(row_order[blk_row])
                sidx = Int32(state_idx_t[brow])
                P = Int32(hist_len[brow])
                base = Int32(cache_base[brow])
                if sidx >= 0:
                    with cute.arch.elect_one():
                        o = sl * _PZ_RI
                        ri[o] = brow
                        ri[o + 1] = bx
                        ri[o + 2] = sidx
                        ri[o + 3] = P
                        ri[o + 4] = base
                        ri[o + 5] = Int32(1)
                        cute.arch.mbarrier_arrive(sm["r_full"] + sl)
                    npub = npub + 1
        U.claim_done(sm["ctr"], n_cta)
        se = npub & 1
        cute.arch.mbarrier_wait(sm["r_empty"] + se, ((npub >> 1) & 1) ^ 1)
        with cute.arch.elect_one():
            ri[se * _PZ_RI + 5] = Int32(0)
            cute.arch.mbarrier_arrive(sm["r_full"] + se)

    # Warp 0 of the persistent kernel runs at 24 registers: its per-token copies and prefetches are
    # rolled loops (unrolled, ptxas hoisted their T addresses out of the row loop and spilled them:
    # 56-121 B STL per cell at T = 5-8 with the flush rows' producer path).
    @cute.jit
    def _pz_v_rows_copy(self, v, uw_addr, mbar, brow, hv, buf):
        """_v_rows_copy as a rolled loop over the T tokens. Call under elect_one."""
        rowb = V_DIM * 2
        dst = uw_addr + buf * (W_RING * rowb) + self.wv * rowb
        src = v.iterator.toint() + Int64(cute.crd2idx((brow, 0, hv, 0), v.layout)) * 2
        tstr = Int64(cute.crd2idx((0, 1, 0, 0), v.layout)) * 2
        for t in cutlass.range(self.T, unroll=1):
            U.bulk_g2s(dst + t * rowb, src + Int64(t) * tstr, Int32(rowb), mbar.toint())

    @cute.jit
    def _pz_v_prefetch(self, v, brow, hv0):
        """Every head's v rows of one row into L2 (as kernel / kernel_early). Under elect_one."""
        src = v.iterator.toint() + Int64(cute.crd2idx((brow, 0, hv0, 0), v.layout)) * 2
        tstr = Int64(cute.crd2idx((0, 1, 0, 0), v.layout)) * 2
        for t in cutlass.range(self.T, unroll=1):
            U.bulk_prefetch_l2(src + Int64(t) * tstr, self.hpc * V_DIM * 2)

    @cute.jit
    def _pz_l2_prefetch(self, q, k, a, b, v, k_cache, g_cache, brow, kh, sidx, base):
        """The next row's prologue inputs into L2 (evict_last), about a head ahead of its consumer
        prologue: X rows (2T), the Gram's 16 ring rows (at most two runs), the heads' g ring rows,
        the row's a / b, every head's v rows. Under elect_one."""
        T = self.T
        hv0 = kh * self.hpk
        rowb = K_DIM * 2
        k_r = k.iterator.toint() + Int64(cute.crd2idx((brow, 0, kh, 0), k.layout)) * 2
        q_r = q.iterator.toint() + Int64(cute.crd2idx((brow, 0, kh, 0), q.layout)) * 2
        kts = Int64(cute.crd2idx((0, 1, 0, 0), k.layout)) * 2
        qts = Int64(cute.crd2idx((0, 1, 0, 0), q.layout)) * 2
        for t in cutlass.range(T, unroll=1):
            U.bulk_prefetch_l2(k_r + Int64(t) * kts, rowb)
            U.bulk_prefetch_l2(q_r + Int64(t) * qts, rowb)
        kc0 = (
            k_cache.iterator.toint()
            + Int64(cute.crd2idx((sidx, kh, 0, 0), k_cache.layout)) * 2
        )
        n1 = cutlass.min(Int32(W_RING), Int32(RING_SLOTS) - base)
        U.bulk_prefetch_l2_n(kc0 + Int64(base) * rowb, n1 * rowb, Int32(1))
        U.bulk_prefetch_l2_n(kc0, (Int32(W_RING) - n1) * rowb, Int32(n1 < W_RING))
        U.bulk_prefetch_l2(
            g_cache.iterator.toint()
            + Int64(cute.crd2idx((sidx, hv0, 0), g_cache.layout)) * 4,
            self.hpc * RING_SLOTS * 4,
        )
        # a / b of the row: the launch's heads of every token (one span; with a head slice its
        # start is 16 B aligned when the slice's head count is a multiple of 8)
        abn = ((T - 1) * self.hv_io + self.hv) * 2
        if abn % 16 == 0 and (self.hv_io == self.hv or self.hv % 8 == 0):
            U.bulk_prefetch_l2(
                a.iterator.toint() + Int64(cute.crd2idx((brow, 0, 0), a.layout)) * 2,
                abn,
            )
            U.bulk_prefetch_l2(
                b.iterator.toint() + Int64(cute.crd2idx((brow, 0, 0), b.layout)) * 2,
                abn,
            )
        self._pz_v_prefetch(v, brow, hv0)

    def _pz_window_copy(self, u_cache, u_g, uw_addr, ubar, hv, sidx, base, buf, fl):
        """Head hv's u window into buffer ``buf`` for a row of kind ``fl`` (dynamic): a flush row's
        16 slots, or a verify row's wv slots + its T v rows (counted on the same u_full)."""
        self._u_window_copy(
            u_cache,
            u_g,
            uw_addr,
            ubar,
            sidx,
            hv,
            base,
            buf,
            _sel(fl != 0, W_RING, self.wv, Int32),
            _sel(fl != 0, 0, self.T * V_DIM * 2, Int32),
        )

    @cute.jit
    def _pz_first_issue(
        self, v, u_cache, u_g, uw_addr, tma_ld, s_state, mbar, brow, kh, sidx, base, fl
    ):
        """Warp 0, the CTA's first row: kernel_early's prologue issue: heads 0 / 1's windows (verify
        rows: + their v rows), head 0's state, every head's v rows into L2."""
        hv0 = kh * self.hpk
        st_full, u_full = mbar["st_full"], mbar["u_full"]
        for hh in cutlass.range_constexpr(2):
            with cute.arch.elect_one():
                self._pz_window_copy(
                    u_cache, u_g, uw_addr, u_full[hh], hv0 + hh, sidx, base, hh, fl
                )
        for g in cutlass.range_constexpr(2):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(st_full[g], 16384)
            U.simple_tma_copy(
                tma_ld.atom,
                tma_ld.tma_tensor[sidx, hv0, None, (None, g)],
                s_state[None, (None, g)],
                st_full[g],
            )
        if fl == 0:
            with cute.arch.elect_one():
                for hh in cutlass.range_constexpr(2):
                    self._pz_v_rows_copy(v, uw_addr, u_full[hh], brow, hv0 + hh, hh)
        with cute.arch.elect_one():
            self._pz_v_prefetch(v, brow, hv0)

    @cute.jit
    def _pz_row_slots(
        self,
        v,
        u_cache,
        u_g,
        uw_addr,
        tma_ld,
        tma_st,
        s_state,
        mbar,
        brow,
        sidx,
        hv0,
        base,
        fl,
        phf,
        phv,
    ):
        """Warp 0, one row's state slots 2..7 (one rolled loop over heads 1..3 for both row kinds;
        warp 0 runs at 24 registers). Slot s takes the stage of slot s - 2 once that slot is done
        with it: a verify row's readout (mma_slot, parity phv = verify rows before this one & 1), a
        flush row's RMW (f_gdone, parity phf = flush rows before this one & 1) and then that slot's
        TMA store has read the stage (wait_group.read). Heads 2 / 3's windows: flush rows behind
        their own state's first half (16 slots), verify rows behind its second half (wv slots + the
        v rows, see _u_window_head), once heads 0 / 1 released the buffers (u_free, in-row phase).
        Slots 6 / 7's stores and the next row's slots 0 / 1: _pz_cross_row."""
        HPC = self.hpc
        st_full, mma_slot, f_gdone = mbar["st_full"], mbar["mma_slot"], mbar["f_gdone"]
        u_full, u_free = mbar["u_full"], mbar["u_free"]
        for hh in cutlass.range(1, HPC, unroll=1):
            for g in cutlass.range_constexpr(2):
                prev = 2 * hh - 2 + g
                if fl != 0:
                    cute.arch.mbarrier_wait(f_gdone[0] + prev, phf)
                    U.simple_tma_copy(
                        tma_st.atom,
                        s_state[None, (None, g)],
                        tma_st.tma_tensor[sidx, hv0 + hh - 1, None, (None, g)],
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                else:
                    cute.arch.mbarrier_wait(mma_slot[0] + prev, phv)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        st_full[0] + (prev + 2), 16384
                    )
                U.simple_tma_copy(
                    tma_ld.atom,
                    tma_ld.tma_tensor[sidx, hv0 + hh, None, (None, g)],
                    s_state[None, (None, g)],
                    st_full[0] + (prev + 2),
                )
                # flush rows (fl = 1): behind half 0; verify rows (fl = 0): behind half 1
                if hh >= 2:
                    if fl == 1 - g:
                        cute.arch.mbarrier_wait(u_free[0] + (hh - 2), 0)
                        with cute.arch.elect_one():
                            self._pz_window_copy(
                                u_cache,
                                u_g,
                                uw_addr,
                                u_full[0] + hh,
                                hv0 + hh,
                                sidx,
                                base,
                                hh & 1,
                                fl,
                            )
                        if fl == 0:
                            with cute.arch.elect_one():
                                self._pz_v_rows_copy(
                                    v, uw_addr, u_full[0] + hh, brow, hv0 + hh, hh & 1
                                )

    @cute.jit
    def _pz_cross_row(
        self,
        q,
        k,
        v,
        a,
        b,
        k_cache,
        u_cache,
        g_cache,
        u_g,
        uw_addr,
        tma_ld,
        tma_st,
        s_state,
        mbar,
        flush_min,
        fl,
        phf,
        phv,
        sidx,
        hv0,
        nx,
    ):
        """Warp 0, behind row j's last state slot. A flush row j first: its slots 6 / 7's TMA stores
        (once their RMW is done, f_gdone), each followed by wait_group.read before the stage takes
        the next row's slot (a TMA load must not overwrite SMEM a TMA store is still reading). Then
        row j + 1 (``nx``, if valid): its prologue inputs into L2, its state slots 0 / 1 (a verify row
        j: as its slots 6 / 7's readouts retire, mma_slot), its heads 0 / 1 windows once row j's last
        head released both buffers (u_free, cross-row phase: parity 1)."""
        HPC = self.hpc
        nbrow, nkh, nsidx, nP, nbase, nvalid = nx[0], nx[1], nx[2], nx[3], nx[4], nx[5]
        nhv0 = nkh * self.hpk
        nfl = Int32(flush_min <= nP)
        st_full, mma_slot, f_gdone = mbar["st_full"], mbar["mma_slot"], mbar["f_gdone"]
        u_full, u_free = mbar["u_full"], mbar["u_free"]
        if nvalid != 0:
            with cute.arch.elect_one():
                self._pz_l2_prefetch(
                    q, k, a, b, v, k_cache, g_cache, nbrow, nkh, nsidx, nbase
                )
        for gx in cutlass.range_constexpr(2):
            prev = 2 * HPC - 2 + gx
            if fl != 0:
                cute.arch.mbarrier_wait(f_gdone[prev], phf)
                U.simple_tma_copy(
                    tma_st.atom,
                    s_state[None, (None, gx)],
                    tma_st.tma_tensor[sidx, hv0 + HPC - 1, None, (None, gx)],
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            if nvalid != 0:
                if fl == 0:
                    cute.arch.mbarrier_wait(mma_slot[prev], phv)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(st_full[gx], 16384)
                U.simple_tma_copy(
                    tma_ld.atom,
                    tma_ld.tma_tensor[nsidx, nhv0, None, (None, gx)],
                    s_state[None, (None, gx)],
                    st_full[gx],
                )
        if nvalid != 0:
            for hx in cutlass.range_constexpr(2):
                cute.arch.mbarrier_wait(u_free[hx], 1)
                with cute.arch.elect_one():
                    self._pz_window_copy(
                        u_cache,
                        u_g,
                        uw_addr,
                        u_full[hx],
                        nhv0 + hx,
                        nsidx,
                        nbase,
                        hx,
                        nfl,
                    )
                if nfl == 0:
                    with cute.arch.elect_one():
                        self._pz_v_rows_copy(
                            v, uw_addr, u_full[hx], nbrow, nhv0 + hx, hx
                        )

    @cute.jit
    def _pz_producer(
        self,
        q,
        k,
        v,
        a,
        b,
        k_cache,
        u_cache,
        g_cache,
        tma_ld,
        tma_st,
        flush_min,
        mbar,
        sm,
    ):
        """Warp 0: the CTA's rows' state slots (flush rows: + the folded halves' TMA stores), u
        windows and v rows."""
        s_state = sm["s_state"]
        uw_addr = sm["uw_addr"]
        u_g = u_cache.iterator.toint()
        cpasync.prefetch_descriptor(tma_ld.atom)
        cpasync.prefetch_descriptor(tma_st.atom)
        f0 = self._pz_read(sm, Int32(0))
        brow = f0[0]
        kh = f0[1]
        sidx = f0[2]
        P = f0[3]
        base = f0[4]
        valid = f0[5]
        if valid != 0:
            self._pz_first_issue(
                v,
                u_cache,
                u_g,
                uw_addr,
                tma_ld,
                s_state,
                mbar,
                brow,
                kh,
                sidx,
                base,
                Int32(flush_min <= P),
            )
        j = Int32(0)
        nf = Int32(0)
        while valid != 0:
            fl = Int32(flush_min <= P)
            hv0 = kh * self.hpk
            self._pz_row_slots(
                v,
                u_cache,
                u_g,
                uw_addr,
                tma_ld,
                tma_st,
                s_state,
                mbar,
                brow,
                sidx,
                hv0,
                base,
                fl,
                nf & 1,
                (j - nf) & 1,
            )
            nx = self._pz_read(sm, j + 1)
            self._pz_cross_row(
                q,
                k,
                v,
                a,
                b,
                k_cache,
                u_cache,
                g_cache,
                u_g,
                uw_addr,
                tma_ld,
                tma_st,
                s_state,
                mbar,
                flush_min,
                fl,
                nf & 1,
                (j - nf) & 1,
                sidx,
                hv0,
                nx,
            )
            brow = nx[0]
            kh = nx[1]
            sidx = nx[2]
            P = nx[3]
            base = nx[4]
            valid = nx[5]
            nf = nf + fl
            j = j + 1
        # the flush rows' TMA stores have completed before the CTA exits
        cute.arch.cp_async_bulk_wait_group(0)

    @cute.jit
    def _pz_mma(self, flush_min, mbar, sm):
        """Warp 1: the UMMA issue of every row (verify and flush rows as kernel_early's), then the
        TMEM dealloc."""
        HPC = self.hpc
        st_full, mma_slot = mbar["st_full"], mbar["mma_slot"]
        d_ready, mma_done, epi_done = (
            mbar["d_ready"],
            mbar["mma_done"],
            mbar["epi_done"],
        )
        f_mma_a, f_mma_b = mbar["f_mma_a"], mbar["f_mma_b"]
        f_epi_a, rd0 = mbar["f_epi_a"], mbar["rd0"]
        x_ready, tmem_done = mbar["x_ready"], mbar["tmem_done"]
        tbase = sm["tmem_addr"][0]
        sd = U.sdesc_sw128(V_DIM * 128)
        x_base = sd | Uint64(sm["x_addr"] >> 4)
        stage_base = [
            sd | Uint64(sm["st_addr"] >> 4),
            sd | Uint64((sm["st_addr"] + 16384) >> 4),
        ]
        c_desc = U.sdesc_sw32(Uint64(sm["c_addr"] >> 4))
        # MN-major SW128: SBO = 1024 B between 8-slot K atoms, N halves 2048 B apart
        kt_desc = U.sdesc_mn_sw128(2048, 1024) | Uint64(sm["kt_addr"] >> 4)
        idesc16 = U.f16_idesc(128, 16)
        idesc64mn = U.f16_idesc(128, 64, b_mn_major=True)
        j = Int32(0)
        nf = Int32(0)
        pfl = Int32(0)  # the previous row was a flush row
        f0 = self._pz_read(sm, j)
        P = f0[3]
        valid = f0[5]
        while valid != 0:
            ph = j & 1
            fl = Int32(flush_min <= P)
            if j > 0:
                # the previous row's epilogues whose TMEM reads (R / Y, which this row's UMMAs
                # overwrite) its own UMMA issue did not wait for: verify heads 2 / 3, flush head 3
                # (flush heads 0..2 gate the next head's UMMAs). Waited every row either way: every
                # phase of every mbarrier has a waiter
                if pfl == 0:
                    cute.arch.mbarrier_wait(epi_done[HPC - 2], ph ^ 1)
                cute.arch.mbarrier_wait(epi_done[HPC - 1], ph ^ 1)
            cute.arch.mbarrier_wait(x_ready, ph)
            U.fence_after_sync()
            if fl != 0:
                # per head (kernel_early's flush issue): fold half 0 (needs only D, Kt) -> readout
                # half 0 (rd0: the RMW of stage 0 may start) -> readout half 1 + history (mma_done)
                # -> fold half 1 once the RMW has read fold half 0 (f_epi_a). R / Y (cols 0..31)
                # and F (64..127) of head h + 1 wait for head h's epilogue.
                phf = nf & 1
                for hh in cutlass.range_constexpr(HPC):
                    cute.arch.mbarrier_wait(d_ready[hh], ph)
                    if cutlass.const_expr(hh > 0):
                        cute.arch.mbarrier_wait(epi_done[hh - 1], ph)
                    U.fence_after_sync()
                    U.umma_ts(tbase + 64, tbase + 32, kt_desc, idesc64mn, False)
                    U.umma_commit(f_mma_a[hh])
                    for g in cutlass.range_constexpr(2):
                        slot = 2 * hh + g
                        cute.arch.mbarrier_wait(st_full[slot], ph)
                        U.fence_after_sync()
                        for ka in cutlass.range_constexpr(4):
                            U.umma_ss(
                                tbase + 0,
                                stage_base[g] | Uint64((ka * 32) >> 4),
                                x_base | Uint64((g * 2048 + ka * 32) >> 4),
                                idesc16,
                                (g > 0) or (ka > 0),
                            )
                        if cutlass.const_expr(g == 0):
                            U.umma_commit(rd0[hh])
                    U.umma_ts(tbase + 16, tbase + 32, c_desc, idesc16, False)
                    U.umma_commit(mma_done[hh])
                    cute.arch.mbarrier_wait(f_epi_a[hh], phf)
                    U.fence_after_sync()
                    U.umma_ts(
                        tbase + 64,
                        tbase + 32,
                        kt_desc + Uint64(2048 >> 4),
                        idesc64mn,
                        False,
                    )
                    U.umma_commit(f_mma_b[hh])
            else:
                for hh in cutlass.range_constexpr(HPC):
                    rcol = (hh % 2) * 32
                    if cutlass.const_expr(hh >= 2):
                        cute.arch.mbarrier_wait(epi_done[hh - 2], ph)
                        U.fence_after_sync()
                    for g in cutlass.range_constexpr(2):
                        slot = 2 * hh + g
                        cute.arch.mbarrier_wait(st_full[slot], ph)
                        U.fence_after_sync()
                        for ka in cutlass.range_constexpr(4):
                            U.umma_ss(
                                tbase + rcol,
                                stage_base[g] | Uint64((ka * 32) >> 4),
                                x_base | Uint64((g * 2048 + ka * 32) >> 4),
                                idesc16,
                                (g > 0) or (ka > 0),
                            )
                        U.umma_commit(mma_slot[slot])
                    cute.arch.mbarrier_wait(d_ready[hh], ph)
                    U.fence_after_sync()
                    U.umma_ts(
                        tbase + rcol + 16,
                        tbase + 64 + (hh % 2) * 8,
                        c_desc,
                        idesc16,
                        False,
                    )
                    U.umma_commit(mma_done[hh])
            nf = nf + fl
            pfl = fl
            j = j + 1
            nx = self._pz_read(sm, j)
            P = nx[3]
            valid = nx[5]
        cute.arch.mbarrier_wait(tmem_done, 0)
        U.fence_after_sync()
        U.tmem_dealloc(tbase, _TMEM_COLS)

    @cute.jit
    def _pz_consumers(
        self,
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        k_cache,
        u_cache,
        g_cache,
        out,
        rand_seed,
        scale,
        flush_min,
        hv_base,
        mbar,
        sm,
    ):
        """Math warps: every row's prologue and heads, then the TMEM release."""
        f0 = self._pz_read(sm, Int32(0))
        brow = f0[0]
        kh = f0[1]
        sidx = f0[2]
        P = f0[3]
        base = f0[4]
        valid = f0[5]
        j = Int32(0)
        nf = Int32(0)
        while valid != 0:
            fl = Int32(flush_min <= P)
            self._pz_cons_row(
                j,
                nf,
                fl,
                brow,
                kh,
                sidx,
                P,
                base,
                q,
                k,
                v,
                a,
                b,
                a_log,
                dt_bias,
                k_cache,
                u_cache,
                g_cache,
                out,
                rand_seed,
                scale,
                hv_base,
                mbar,
                sm,
            )
            # every math warp is done with this row's SMEM (coefficients, Gram, X / C / Kt tiles)
            # before the next row's prologue writes it
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            nf = nf + fl
            j = j + 1
            nx = self._pz_read(sm, j)
            brow = nx[0]
            kh = nx[1]
            sidx = nx[2]
            P = nx[3]
            base = nx[4]
            valid = nx[5]
        U.fence_before_sync()
        cute.arch.mbarrier_arrive(mbar["tmem_done"])

    @cute.jit
    def _pz_flush_heads(self, ctx, rand_seed):
        """Math warps, flush row: kernel_early's rolled flush heads (heads and the RMW as loops)."""
        self._load_seed(ctx, rand_seed)
        for hh in cutlass.range(self.hpc, unroll=1):
            vv = self._d_build(hh, 32, ctx, flush=True)
            for g in cutlass.range(2, unroll=1):
                egp, srow, r0, r1, r2, r3 = self._rmw_setup(hh, g, ctx)
                self._rmw_wait(hh, g, ctx)
                for gq in cutlass.range(self.ngrp, unroll=1):
                    r0, r1, r2, r3 = self._rmw_group(
                        hh, g, gq, ctx, egp, srow, r0, r1, r2, r3
                    )
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(self._mb(ctx, "f_gdone", 2 * hh + g))
            self._epilogue(hh, 0, ctx, vv, False)

    @cute.jit
    def _pz_cons_row(
        self,
        j,
        nf,
        fl,
        brow,
        kh,
        sidx,
        P,
        base,
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        k_cache,
        u_cache,
        g_cache,
        out,
        rand_seed,
        scale,
        hv_base,
        mbar,
        sm,
    ):
        """Math warps, one row: row j, the (nf + 1)-th flush row if ``fl`` (kernel_early's consumer
        path with the cursors from the mailbox, so every entry load issues at once; barrier 3
        without warp 0, no 2/128)."""
        T = self.T
        T2 = 2 * T
        T_ = T
        HPC = self.hpc
        CW0 = self.cw0
        # thread index through a side-effecting read: the row body's per-thread addresses stay in
        # the body (hoisted out of the row loop, ptxas spilled them: 360 B STL at T = 8)
        tid = U.tid_x_volatile()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        ph = j & 1
        hv0 = kh * self.hpk
        s_gram, s_coef, s_rn = sm["s_gram"], sm["s_coef"], sm["s_rn"]
        x_addr, c_addr = sm["x_addr"], sm["c_addr"]
        # ---- entry loads, all at once: X rows (X-tile warps), gate inputs and constants (gate
        # warp), ring rows + own X rows and the window g values (ring-row warps)
        grp_e = lane >> 2
        c4_e = lane & 3
        cw_e = (warp - CW0) & 3
        nt_e = cw_e & 1
        q_g = q.iterator.toint()
        k_g = k.iterator.toint()
        xr_b = nt_e * 8 + grp_e
        is_xt = Int32(warp >= CW0 + 2)
        gate_pred = Int32(warp == CW0 + 2)
        x_own = self._x_row_addr(k, q, k_g, q_g, brow, kh, xr_b)
        x_oth = self._x_row_addr(k, q, k_g, q_g, brow, kh, (1 - nt_e) * 8 + grp_e)
        wb = []
        wo = []
        for p in cutlass.range_constexpr(4):
            koff = Int64((32 * p + 8 * c4_e) * 2)
            # zero-filled where predicated off: no loop-carried registers (see U.ldg_v4_if_z)
            wb.append(U.ldg_v4_if_z(is_xt, x_own + koff))
            wo.append(U.ldg_v4_if_z(is_xt, x_oth + koff))
        ct_e = (tid - 32 * CW0) & 127
        gh = (ct_e >> 3) & 3
        ghc = gh % HPC
        gt = ct_e & 7
        tt = _sel(gt < T_, gt, 0, Int32)
        ab_off = Int64(cute.crd2idx((brow, tt, hv0 + ghc), a.layout)) * 2
        av_raw = U.ldg_u16_if(gate_pred, a.iterator.toint() + ab_off)
        bv_raw = U.ldg_u16_if(gate_pred, b.iterator.toint() + ab_off)
        alog_e = U.ldg_f32_if(
            gate_pred,
            a_log.iterator.toint()
            + Int64(cute.crd2idx((hv0 + ghc,), a_log.layout)) * 4,
        )
        dtb_e = U.ldg_f32_if(
            gate_pred,
            dt_bias.iterator.toint()
            + Int64(cute.crd2idx((hv0 + ghc,), dt_bias.layout)) * 4,
        )
        is_rr = Int32(warp == CW0) + Int32(warp == CW0 + 1)
        kc_g = k_cache.iterator.toint()
        wr1_e = self._ring_row_loads(
            k_cache, kc_g, is_rr, sidx, kh, base, grp_e, c4_e, wo, wb, x_own, zfill=True
        )
        wh, wj, gp_g, gp_w, gj_w = self._window_g_loads(
            g_cache, is_rr, ct_e, sidx, hv0, ghc, base, P
        )
        # ============================================================ consumers
        cw = warp - CW0
        ct = cw * 32 + lane  # TMEM lane address / k index for staging
        row = ((warp & 3) << 5) | lane  # V row held by this thread (= TMEM lane)
        grp = lane >> 2
        c4 = lane & 3
        mt = cw >> 1
        nt = cw & 1
        gc = nt * 8 + 2 * c4
        if mt == 1:
            accx = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
            for p in cutlass.range_constexpr(4):
                for hs in cutlass.range_constexpr(2):
                    w0a = _sel(nt == 0, wb[p][2 * hs], wo[p][2 * hs], Uint32)
                    w1a = _sel(nt == 0, wo[p][2 * hs], wb[p][2 * hs], Uint32)
                    w0b = _sel(nt == 0, wb[p][2 * hs + 1], wo[p][2 * hs + 1], Uint32)
                    w1b = _sel(nt == 0, wo[p][2 * hs + 1], wb[p][2 * hs + 1], Uint32)
                    accx = U.mma_bf16_16816(
                        w0a, w1a, w0b, w1b, wb[p][2 * hs], wb[p][2 * hs + 1], accx
                    )
            s_gram[(16 + grp) * 16 + gc] = accx[0]
            s_gram[(16 + grp) * 16 + gc + 1] = accx[1]
            s_gram[(24 + grp) * 16 + gc] = accx[2]
            s_gram[(24 + grp) * 16 + gc + 1] = accx[3]
            # the ring-row warps read the X x X diagonal: arrive only (barrier 3: these 64 +
            # the ring-row warps' sync)
            cute.arch.barrier_arrive(
                barrier_id=3, number_of_threads=_PZ_B3, aligned=False
            )
            cute.arch.sync_warp()
            rn_x = cute.math.rsqrt(s_gram[(16 + xr_b) * 16 + xr_b] + Float32(_EPS))
            rn_x = _sel(xr_b >= T, rn_x * scale, rn_x, Float32)
            for p in cutlass.range_constexpr(4):
                kc = 4 * p + c4
                xw = []
                for i in cutlass.range_constexpr(4):
                    xw.append(
                        U.pack_f16x2(
                            U.bf16x2_lo_f32(wb[p][i]) * rn_x,
                            U.bf16x2_hi_f32(wb[p][i]) * rn_x,
                        )
                    )
                xb = (kc // 8) * 2048 + xr_b * 128 + (((kc % 8) ^ (xr_b % 8)) * 16)
                U.sts_v4(x_addr + xb, xw[0], xw[1], xw[2], xw[3])
            # the two X-tile warps' Gram halves (the norms below read the other warp's diagonal:
            # kernel_early's barrier 2 / 128 instance, without warps 0 / 1)
            cute.arch.barrier(barrier_id=2, number_of_threads=64)
            # (the X tile's previous readers, row j - 1's readouts, completed before every math
            # thread's mma_done waits of that row)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(mbar["x_ready"])
            cl = ct - 64
            if cl < T2:
                rn_own = cute.math.rsqrt(s_gram[(16 + cl) * 16 + cl] + Float32(_EPS))
                if cl >= T:
                    rn_own = rn_own * scale
                s_rn[cl] = rn_own
            if cl < 32:
                av = U.u16_as_f32(av_raw, True)
                bv = U.u16_as_f32(bv_raw, True)
                xx = av + dtb_e
                sp = xx
                if xx <= Float32(20.0):
                    sp = cute.math.log(
                        Float32(1.0) + cute.math.exp(xx, fastmath=True),
                        fastmath=True,
                    )
                la = -cute.math.exp(alog_e, fastmath=True) * sp
                if gt >= T:
                    la = Float32(0.0)
                beta = Float32(1.0) / (Float32(1.0) + cute.math.exp(-bv, fastmath=True))
                gam = la
                for off in (1, 2, 4):
                    lower = cute.arch.shuffle_sync_up(gam, off, mask_and_clamp=0)
                    if gt >= off:
                        gam = gam + lower
                eg = cute.math.exp(gam, fastmath=True)
                cbg = gh * _COEF
                s_coef[cbg + _CO_BETA + gt] = beta
                s_coef[cbg + _CO_BG + gt] = beta * eg
                s_coef[cbg + _CO_EG + gt] = eg
                s_coef[cbg + _CO_GAM + gt] = gam
            cute.arch.barrier(barrier_id=2, number_of_threads=64)
            NT = HPC * T * T
            for it in cutlass.range_constexpr((NT + 63) // 64):
                idx = cl + it * 64
                if idx < NT:
                    ch = idx // (T * T)
                    t = (idx // T) % T
                    s = idx % T
                    cb = ch * _COEF
                    dec = cute.math.exp(
                        s_coef[cb + _CO_GAM + t] - s_coef[cb + _CO_GAM + s],
                        fastmath=True,
                    )
                    kk = s_gram[(16 + s) * 16 + t] * s_rn[s] * s_rn[t]
                    qk = s_gram[(16 + s) * 16 + T + t] * s_rn[s] * s_rn[T + t]
                    av_ = Float32(0.0)
                    if s < t:
                        av_ = s_coef[cb + _CO_BETA + t] * dec * kk
                    bq_ = Float32(0.0)
                    if s <= t:
                        bq_ = dec * qk
                    s_coef[cb + _CO_A + t * 8 + s] = av_
                    s_coef[cb + _CO_BQ + t * 8 + s] = bq_
        else:
            wr0 = wo
            wr1 = wr1_e
            U.bar_sync_ua(3, _PZ_B3)
            rn_b = cute.math.rsqrt(s_gram[(16 + xr_b) * 16 + xr_b] + Float32(_EPS))
            rn_b = _sel(xr_b >= T, rn_b * scale, rn_b, Float32)
            accr = self._ring_gram(wr0, wr1, wb, rn_b)
            s_gram[grp * 16 + gc] = accr[0]
            s_gram[grp * 16 + gc + 1] = accr[1]
            s_gram[(grp + 8) * 16 + gc] = accr[2]
            s_gram[(grp + 8) * 16 + gc + 1] = accr[3]
            if xr_b < T:
                kdst = (
                    kc_g
                    + Int64(
                        cute.crd2idx(
                            (sidx, kh, (base + P + xr_b) & RING_MASK, 0),
                            k_cache.layout,
                        )
                    )
                    * 2
                )
                for p in cutlass.range_constexpr(4):
                    kw4 = self._k_append_words(wb[p], rn_b)
                    U.stg_v4(
                        kdst + Int64((32 * p + 8 * c4) * 2),
                        kw4[0],
                        kw4[1],
                        kw4[2],
                        kw4[3],
                    )
        cute.arch.barrier(barrier_id=1, number_of_threads=128)

        # -------- C tile (warps mt == 1), Kt (flush rows), window weights, g-ring appends
        if mt == 1:
            cl2 = ct - 64
            for i in cutlass.range_constexpr(4):
                e = cl2 * 4 + i
                xc = e >> 4
                jj = e & 15
                cval = Float32(0.0)
                if jj < P:
                    if xc < T2:
                        if cutlass.const_expr(self.k_f16):
                            cval = s_gram[jj * 16 + xc]  # ring x X̂: normalized already
                        else:
                            rnx = cute.math.rsqrt(
                                s_gram[(16 + xc) * 16 + xc] + Float32(_EPS)
                            )
                            if xc >= T:
                                rnx = rnx * scale
                            cval = s_gram[jj * 16 + xc] * rnx
                cbyte = xc * 32 + (((jj // 8) ^ ((xc >> 2) & 1)) * 16) + (jj % 8) * 2
                U.sts_u16(c_addr + cbyte, U.f32_to_u16(cval, False))
        if fl != 0:
            # K_hist tile (fold B operand), as kernel_early: thread -> (slot row ct / 8, 16 B chunk
            # ct % 8) of both K groups, slots j >= P zeroed. Its previous readers (the last flush
            # row's folds) completed before every math thread's f_mma_b waits of that row.
            jrk = ct >> 3
            c8k = ct & 7
            krow = (
                kc_g
                + Int64(
                    cute.crd2idx(
                        (sidx, kh, (base + jrk) & RING_MASK, c8k * 8), k_cache.layout
                    )
                )
                * 2
            )
            krk = [U.ldg_v4(krow), U.ldg_v4(krow + Int64(128))]
            for gk in cutlass.range_constexpr(2):
                kq4 = self._kt_words(krk[gk], jrk < P)
                U.sts_v4(
                    sm["kt_addr"] + gk * 2048 + jrk * 128 + ((c8k ^ (jrk % 8)) * 16),
                    kq4[0],
                    kq4[1],
                    kq4[2],
                    kq4[3],
                )
        # C / Kt are read by the history / fold UMMAs, which warp 1 issues only after d_ready,
        # arrived by these same threads after this fence. Only their writers fence: the X-tile
        # warps (C) and, on flush rows, everyone (Kt)
        if mt == 1:
            cute.arch.fence_view_async_shared()
        else:
            if fl != 0:
                cute.arch.fence_view_async_shared()
        if ct < 32:
            if gt < T:
                if gh < HPC:
                    gnew = s_coef[gh * _COEF + _CO_GAM + gt]
                    if fl == 0:
                        # (flush rows restart the window: g appends from 0)
                        if P > 0:
                            gnew = gnew + gp_g
                    g_cache[sidx, hv0 + gh, (base + P + gt) & RING_MASK] = gnew
        if ct < 64:
            w = Float32(0.0)
            if wj < P:
                w = cute.math.exp(gp_w - gj_w, fastmath=True)
            s_coef[wh * _COEF + _CO_W + wj] = w
            if wj == 0:
                egp = Float32(1.0)
                if P > 0:
                    egp = cute.math.exp(gp_w, fastmath=True)
                s_coef[wh * _COEF + _CO_EGP] = egp
        cute.arch.barrier(barrier_id=1, number_of_threads=128)

        ctx = dict(
            sidx=sidx,
            base=base,
            P=P,
            brow=brow,
            hv0=hv0,
            ct=ct,
            row=row,
            tbase=sm["tmem_addr"][0],
            st_addr=sm["st_addr"],
            u_cache=u_cache,
            v=v,
            out=out,
            s_coef=s_coef,
            d_ready=mbar["d_ready"],
            mma_done=mbar["mma_done"],
            epi_done=mbar["epi_done"],
            uw_addr=sm["uw_addr"],
            u_full=mbar["u_full"],
            u_free=mbar["u_free"],
            f_mma_a=mbar["f_mma_a"],
            f_mma_b=mbar["f_mma_b"],
            f_epi_a=mbar["f_epi_a"],
            f_gdone=mbar["f_gdone"],
            rd0=mbar["rd0"],
            ph=ph,
            phf=nf & 1,
            xrow=True,
            hv_base=hv_base,
        )
        if fl != 0:
            self._pz_flush_heads(ctx, rand_seed)
        else:
            for hh in cutlass.range_constexpr(HPC):
                self._head(hh, False, ctx)

    def _conv_prephase(
        self,
        q,
        state_idx_t,
        brow,
        kh,
        hv0,
        ct,
        conv_x,
        conv_w,
        conv_s,
        conv_win,
        conv_widx,
        conv_x_row,
    ):
        """fused_conv: the target-verify conv of this CTA's channels, run by the math warps (``ct``
        = 0..127) in place of griddepcontrol.wait. Thread ct < 32 + 16 hpc owns 8 channels: q (ct
        < 16) and k (ct < 32) of key head kh, then v of value heads hv0 .. hv0 + hpc - 1. Weights
        [D, 4] and the prior conv state [slot, D, 3] load before the wait (no kernel in flight
        writes them), the T projection rows after it. Stores, as SGLang's verify conv: the T
        outputs (token-major [B, T, D] at q's base, which q / k / v view), the new state (the last
        3 inputs) and the deduplicated step window [s1, s2, x_0 .. x_{T-1}] of the row's window
        slot. Padded rows (slot < 0) store nothing. Barrier 1 (the math warps) then orders every
        later q / k / v read of the CTA after the stores."""
        T = self.T
        D = self.qk_io * K_DIM
        qd = self.h * K_DIM
        W = T + 2
        xr = Int64(conv_x_row)  # elements per projection row (a token)
        ngrp = 2 * (K_DIM // 8) + self.hpc * (V_DIM // 8)
        g = Int32(ct)
        sidx = Int32(state_idx_t[brow])
        act = Int32(g < ngrp) * Int32(sidx >= 0)
        c0 = _sel(
            g < 16,
            kh * K_DIM + g * 8,
            _sel(
                g < 32,
                qd + kh * K_DIM + (g - 16) * 8,
                2 * qd + hv0 * V_DIM + (g - 32) * 8,
                Int32,
            ),
            Int32,
        )
        c0l = Int64(c0)
        wa = conv_w + c0l * 8
        wq = []
        for i in range(4):
            wq.extend(U.ldg_v4_if(act, wa + Int64(16 * i)))
        sa = conv_s + (Int64(sidx) * (D * 3) + c0l * 3) * 2
        sq = []
        for i in range(3):
            sq.extend(U.ldg_v4_if(act, sa + Int64(16 * i)))
        cute.arch.griddepcontrol_wait()
        xb = U.opaque_i64(conv_x) + (Int64(brow) * T * xr + c0l) * 2
        xq = [U.ldg_v4_if(act, xb + xr * (2 * t)) for t in range(T)]
        widx = U.ldg_s32_if(act, U.opaque_i64(conv_widx) + Int64(brow) * 4)

        def pair(words, e_lo, e_hi):
            # bf16 elements e_lo / e_hi of a packed run as (bits 0..15, bits 16..31)
            sel = (0x10 if e_lo % 2 == 0 else 0x32) | (
                (0x54 if e_hi % 2 == 0 else 0x76) << 8
            )
            return U.prmt(words[e_lo // 2], words[e_hi // 2], sel)

        # per tap k and channel pair j: (w / state of channel c0 + 2j, of c0 + 2j + 1)
        w2 = [[pair(wq, 4 * (2 * j) + k, 4 * (2 * j + 1) + k) for j in range(4)] for k in range(4)]
        s2 = [[pair(sq, 3 * (2 * j) + k, 3 * (2 * j + 1) + k) for j in range(4)] for k in range(3)]
        ob = q.iterator.toint() + (Int64(brow) * (T * D) + c0l) * 2
        out0 = None
        for t in range(T):
            ow = []
            for j in range(4):
                seq = [s2[0][j], s2[1][j], s2[2][j]] + [xq[i][j] for i in range(T)]
                ow.append(
                    U.conv4_silu_bf16x2(
                        seq[t], seq[t + 1], seq[t + 2], seq[t + 3],
                        w2[0][j], w2[1][j], w2[2][j], w2[3][j],
                    )
                )
            if t == 0:
                out0 = ow
            U.stg_v4_if(act, ob + Int64(t * D * 2), ow[0], ow[1], ow[2], ow[3])
        # x rows as one packed run per row: element c of row t is xq[t][c // 2], half c % 2
        xrun = [list(xq[t]) for t in range(T)]

        def x_elem_pair(e_lo, e_hi, elem):
            lo_words, lo_e = elem(e_lo)
            hi_words, hi_e = elem(e_hi)
            sel = (0x10 if lo_e % 2 == 0 else 0x32) | ((0x54 if hi_e % 2 == 0 else 0x76) << 8)
            return U.prmt(lo_words[lo_e // 2], hi_words[hi_e // 2], sel)

        def state_elem(e):  # new state element (c, k) at 3c + k = input row T - 3 + k
            c, k = e // 3, e % 3
            return xrun[T - 3 + k], c

        def window_elem(e):  # window element (c, col) at W c + col
            c, col = e // W, e % W
            if col < 2:
                return sq, 3 * c + 1 + col
            return xrun[col - 2], c

        # the state stores overwrite the prior state this thread loaded: address them through
        # the first outputs, which read every prior-state word
        sa_st = U.addr_after(sa, out0)
        st_words = [x_elem_pair(2 * m, 2 * m + 1, state_elem) for m in range(12)]
        for i in range(3):
            U.stg_v4_if(act, sa_st + Int64(16 * i), *st_words[4 * i : 4 * i + 4])
        wb = conv_win + (Int64(widx) * (D * W) + c0l * W) * 2
        win_words = [x_elem_pair(2 * m, 2 * m + 1, window_elem) for m in range(4 * W)]
        for i in range(W):
            U.stg_v4_if(act, wb + Int64(16 * i), *win_words[4 * i : 4 * i + 4])
        cute.arch.barrier(barrier_id=1, number_of_threads=128)

    def _x_row_addr(self, k, q, k_g, q_g, brow, kh, r):
        """Global byte address of X row r (k_t, then q_t; rows >= 2T -> k_0, never read)."""
        T = self.T
        r = Int32(r)
        is_q = (r >= T) & (r < 2 * T)
        t_k = _sel(r < T, r, 0, Int32)
        t_q = _sel(is_q, r - T, 0, Int32)
        ak = k_g + Int64(cute.crd2idx((brow, t_k, kh, 0), k.layout)) * 2
        aq = q_g + Int64(cute.crd2idx((brow, t_q, kh, 0), q.layout)) * 2
        return _sel(is_q, aq, ak, Int64)

    def _init_mbarriers(self, mb, first):
        """Init every mbarrier of ``mb`` but st_full / u_full [0, first) (initialized already),
        then fence the inits. Call under elect_one."""
        for s_ in range(2 * HPK):
            if s_ >= first:
                cute.arch.mbarrier_init(mb["st_full"][s_], 1)
            cute.arch.mbarrier_init(mb["mma_slot"][s_], 1)
            cute.arch.mbarrier_init(mb["f_gdone"][s_], 128)
        for i in range(HPK):
            if i >= first:
                cute.arch.mbarrier_init(mb["u_full"][i], 1)
            cute.arch.mbarrier_init(mb["mma_done"][i], 1)
            cute.arch.mbarrier_init(mb["d_ready"][i], 128)
            cute.arch.mbarrier_init(mb["epi_done"][i], 128)
            cute.arch.mbarrier_init(mb["f_mma_a"][i], 1)
            cute.arch.mbarrier_init(mb["f_mma_b"][i], 1)
            cute.arch.mbarrier_init(mb["f_epi_a"][i], 128)
            cute.arch.mbarrier_init(mb["rd0"][i], 1)
        cute.arch.mbarrier_init(mb["u_free"][0], 128)
        cute.arch.mbarrier_init(mb["u_free"][1], 128)
        cute.arch.mbarrier_init(mb["x_ready"], 64)
        cute.arch.mbarrier_init(mb["tmem_done"], 128)
        cute.arch.mbarrier_init_fence()

    def _ring_row_loads(
        self,
        k_cache,
        kc_g,
        is_rr,
        sidx,
        kh,
        base,
        grp_e,
        c4_e,
        wo,
        wb=None,
        x_own=None,
        zfill=False,
    ):
        """The ring-row warps' (``is_rr``) Gram ring rows: row grp into ``wo`` (the registers
        that hold the other X half in the X-tile warps, kept there), row grp + 8 returned;
        with ``x_own``, also this lane's own X row into ``wb`` (kept elsewhere)."""
        ra0 = (
            kc_g
            + Int64(
                cute.crd2idx((sidx, kh, (base + grp_e) & RING_MASK, 0), k_cache.layout)
            )
            * 2
        )
        ra1 = (
            kc_g
            + Int64(
                cute.crd2idx(
                    (sidx, kh, (base + grp_e + 8) & RING_MASK, 0), k_cache.layout
                )
            )
            * 2
        )
        wr1 = []
        for p in range(4):
            koff = Int64((32 * p + 8 * c4_e) * 2)
            if x_own is not None:
                wb[p] = U.ldg_v4_if_keep(
                    is_rr, x_own + koff, wb[p][0], wb[p][1], wb[p][2], wb[p][3]
                )
            wo[p] = U.ldg_v4_if_keep(
                is_rr, ra0 + koff, wo[p][0], wo[p][1], wo[p][2], wo[p][3]
            )
            if zfill:  # persistent row loop (see U.ldg_v4_if_z)
                wr1.append(U.ldg_v4_if_z(is_rr, ra1 + koff))
            else:
                wr1.append(U.ldg_v4_if(is_rr, ra1 + koff))
        return wr1

    def _ring_gram(self, wr0, wr1, wb, rn_b):
        """Ring x X Gram tile of this lane: ring rows grp / grp + 8 (``wr0`` / ``wr1``) against its
        own X row ``wb`` (norm ``rn_b``). bf16 k ring: raw X rows, bf16 mma.sync (the C tile
        normalizes). fp16 k ring: the X rows normalized and rounded to fp16 first, the X tile's
        own values, and f16 mma.sync."""
        accr = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
        for p in range(4):
            if self.k_f16:
                xb = self._k_append_words(wb[p], rn_b)
                mma = U.mma_f16_16816
            else:
                xb = wb[p]
                mma = U.mma_bf16_16816
            for hs in range(2):
                accr = mma(
                    wr0[p][2 * hs],
                    wr1[p][2 * hs],
                    wr0[p][2 * hs + 1],
                    wr1[p][2 * hs + 1],
                    xb[2 * hs],
                    xb[2 * hs + 1],
                    accr,
                )
        return accr

    def _k_append_words(self, wbp, rn_b):
        """16 B of normalized k̂ in the k ring's dtype from 16 B of the raw bf16 X row (fp16: the
        X tile's bits, same rn and the same cvt.rn)."""
        pack = U.pack_f16x2 if self.k_f16 else U.pack_bf16x2
        return [
            pack(U.bf16x2_lo_f32(wbp[i]) * rn_b, U.bf16x2_hi_f32(wbp[i]) * rn_b)
            for i in range(4)
        ]

    def _kt_words(self, krg, live):
        """16 B of the fp16 K_hist tile from 16 B of the k ring, zero where not ``live``."""
        if self.k_f16:
            return [_sel(live, krg[i], 0, Uint32) for i in range(4)]
        return [
            U.pack_f16x2(
                _sel(live, U.bf16x2_lo_f32(krg[i]), 0.0, Float32),
                _sel(live, U.bf16x2_hi_f32(krg[i]), 0.0, Float32),
            )
            for i in range(4)
        ]

    def _window_g_loads(self, g_cache, pred, ct_w, sidx, hv0, ghc, base, P):
        """Window g values of weight lane ct_w (loaded where ``pred``): G_P of head ghc (the
        g appends) and, for head wh = ct_w / 16 and slot wj = ct_w % 16, G_P and g_j (the
        window weights). Returns (wh, wj, G_P of ghc, G_P of wh, g_j)."""
        wh = (ct_w >> 4) & 3
        whc = wh % self.hpc
        wj = ct_w & 15
        g_g = g_cache.iterator.toint()
        gp_g = U.ldg_f32_if(
            pred,
            g_g
            + Int64(
                cute.crd2idx(
                    (sidx, hv0 + ghc, (base + P - 1) & RING_MASK), g_cache.layout
                )
            )
            * 4,
        )
        gp_w = U.ldg_f32_if(
            pred,
            g_g
            + Int64(
                cute.crd2idx(
                    (sidx, hv0 + whc, (base + P - 1) & RING_MASK), g_cache.layout
                )
            )
            * 4,
        )
        gj_w = U.ldg_f32_if(
            pred,
            g_g
            + Int64(
                cute.crd2idx((sidx, hv0 + whc, (base + wj) & RING_MASK), g_cache.layout)
            )
            * 4,
        )
        return wh, wj, gp_g, gp_w, gj_w

    def _u_window_head(self, slot):
        """Non-flush rows: the value head whose u window warp 0 bulk-copies right behind
        state slot ``slot``'s load (heads 0 / 1: in the prologue), -1 for none.

        Without vsmem: head hh behind its own state's first half (slot 2 hh, as on flush
        rows). u_free[hh - 2] is arrived in head hh - 2's D build, ahead of the d_ready ->
        history -> readout chain that mma_slot[2 hh - 2] (the next wait) depends on, so the
        wait never delays the state stream.

        With vsmem, u_free[hh - 2] is arrived in head hh - 2's epilogue instead (right after
        its v reads, ahead of its epi_done), so head hh goes behind its own state's second
        half (slot 2 hh + 1): ahead of slots 2 hh / 2 hh + 1 the wait would hold head hh's
        state back to that epilogue; behind them it can delay only slot 2 hh + 2, whose own
        wait (mma_slot[2 hh]: head hh's readout, which waits for epi_done[hh - 2]) returns
        later anyway. No cycle: head hh - 2's epilogue needs only its readout and D build,
        whose state and window were issued before this wait."""
        hh = slot // 2
        if self.vsmem:
            return hh if slot % 2 == 1 and 2 <= hh < self.hpc else -1
        return hh if slot % 2 == 0 and hh >= 2 else -1

    def _u_window_copy(
        self,
        u_cache,
        u_g,
        uw_addr,
        mbar,
        sidx,
        hv,
        base,
        buf,
        nsl=W_RING,
        extra_tx=None,
    ):
        """Bulk-copy logical window slots j = 0..nsl-1 (physical (base + j) & 31; nsl = 16,
        or vsmem's wv on verify rows) of one value head's u ring into SMEM buffer ``buf`` as
        [j][V]; ``extra_tx`` more bytes (vsmem: the v rows, issued separately) are counted
        on ``mbar``. Call under elect_one.

        The window's ring slots wrap at most once: two contiguous copies instead of
        sixteen single-row ones (saves ~0.5 us of warp 0's serial issue per head pair).
        Without a wrap the second copy re-copies row nsl - 1 (the same bytes), so the
        transaction count stays expression-only."""
        rowb = V_DIM * 2
        src_head = u_g + Int64(cute.crd2idx((sidx, hv, 0, 0), u_cache.layout)) * 2
        dst = uw_addr + buf * (W_RING * rowb)
        last = Int32(nsl - 1) if isinstance(nsl, int) else nsl - Int32(1)
        n1 = _sel(base + nsl <= RING_SLOTS, Int32(nsl), Int32(RING_SLOTS) - base, Int32)
        n2 = _sel(n1 < nsl, Int32(nsl) - n1, Int32(1), Int32)
        d2 = _sel(n1 < nsl, n1, last, Int32)
        s2 = _sel(n1 < nsl, Int32(0), (base + last) & RING_MASK, Int32)
        tx = (n1 + n2) * rowb
        if extra_tx is not None:
            tx = tx + extra_tx
        cute.arch.mbarrier_arrive_and_expect_tx(mbar, tx)
        U.bulk_g2s(dst, src_head + Int64(base * rowb), n1 * rowb, mbar.toint())
        U.bulk_g2s(
            dst + d2 * rowb, src_head + Int64(s2 * rowb), n2 * rowb, mbar.toint()
        )

    def _v_rows_copy(self, v, uw_addr, mbar, brow, hv, buf):
        """vsmem, verify rows: value head hv's T v rows v[brow, t, hv, :] (256 B each, HV * V
        elements apart) into slots wv + t of u-window buffer ``buf``, counted on ``mbar``
        (whose expect-tx includes them). Call under elect_one."""
        rowb = V_DIM * 2
        dst = uw_addr + buf * (W_RING * rowb) + self.wv * rowb
        v_g = v.iterator.toint()
        for t in range(self.T):
            U.bulk_g2s(
                dst + t * rowb,
                v_g + Int64(cute.crd2idx((brow, t, hv, 0), v.layout)) * 2,
                Int32(rowb),
                mbar.toint(),
            )

    def _d_build(self, hh, dcol, ctx, vsm=False, flush=False):
        """D row w_j u_j (j < P, fp16 pairs) of head hh into TMEM columns dcol..dcol+7;
        arrives d_ready[hh].  Also returns this row's raw v values of head hh (issued first
        so their latency overlaps the D build and the readout wait); with ``vsm`` (vsmem,
        verify rows) none: the epilogue reads them from the u-window buffer."""
        T = self.T
        P, row, ct, tbase, s_coef = (
            ctx["P"],
            ctx["row"],
            ctx["ct"],
            ctx["tbase"],
            ctx["s_coef"],
        )
        v = ctx["v"]
        cb = hh * _COEF
        vv = []
        if not vsm:
            for t in range(T):
                vv.append(v[ctx["brow"], t, ctx["hv0"] + hh, row])
        cute.arch.mbarrier_wait(self._mb(ctx, "u_full", hh), ctx.get("ph", 0))
        ub = ctx["uw_addr"] + (hh % 2) * (W_RING * V_DIM * 2) + row * 2
        # vsm: only the wv slots that can be live (P <= wv on verify rows); the slots above
        # hold the v rows and their D words are the zeros the masks would give
        nsl = self.wv if vsm else W_RING
        uv = [U.lds_u16(ub + j * (V_DIM * 2)) for j in range(nsl)]
        uf = (
            None  # the window values as fp32, when converted ahead of the u_free arrive
        )
        if not isinstance(hh, int) and not vsm:
            # rolled head loop (dynamic hh): heads 0 / 1 release their buffer (predicated fence +
            # arrive as below); verify rows convert every head's window first (the same values
            # the lazy conversion makes)
            if not flush:
                uf = [U.u16_as_f32(w, self.u_bf16) for w in uv]
            # Flush rows arrive without a proxy fence: warp 0's next async-proxy write over this
            # buffer (heads 2 / 3's window, _pz_row_slots) waits f_gdone of this head's fold
            # (fenced in _rmw) BEFORE it waits u_free, so the fold's fence orders these reads.
            # Keep that wait order (or fence here: measured +0.6 % at T = 8, 30 % flush).
            U.mbar_arrive_if(
                Int32(hh < 2),
                (ctx["u_free"][0] + cutlass.min(hh, Int32(1))).toint(),
                fence=not flush,
            )
        if (
            isinstance(hh, int) and hh < 2 and not vsm
        ):  # vsm: in the epilogue, once the v rows are read
            if not flush:
                # Verify rows: head hh + 2's window is bulk-copied (async proxy) over these
                # slots once u_free completes. The values go to registers first (the
                # conversions use them, so the arrive issues after the loads return), then
                # the generic-proxy reads are ordered before the async-proxy writes (same as
                # the vsmem epilogue; compute-sanitizer 2026.5 racecheck: "WAR hazard
                # (missing fence.proxy.async.shared)" at T <= 4 without it). Flush rows reach
                # that copy only after their fold's fence_view_async_shared (_rmw).
                uf = [U.u16_as_f32(w, self.u_bf16) for w in uv]
                cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(self._mb(ctx, "u_free", hh))
        cv = (
            (lambda j: uf[j])
            if uf is not None
            else (lambda j: U.u16_as_f32(uv[j], self.u_bf16))
        )
        words = []
        for c in range(8):
            if 2 * c >= nsl:
                words.append(Uint32(0))  # vsm: both slots never live
                continue
            hi_live = 2 * c + 1 < nsl
            if cutlass.const_expr(self.d_mask):
                # Slots j >= P (stale ring rows, possibly inf / NaN) are cleared on the packed
                # word: AND with a CTA-uniform mask that keeps half j = 2c + i iff j < P,
                # ~0 >> 16 * clamp(2c + 2 - P, 0, 2) (64-bit shift: by 32 for an empty word).
                # Same bits as selecting 0.0 per value (cvt.rn.f16x2 of 0.0 is 0x0000), but
                # ptxas keeps the 8 masks in uniform registers, where the 16 selects on uniform
                # predicates cost ~40 predicate moves per head on the consumers' critical path.
                lo = cv(2 * c) * s_coef[cb + _CO_W + 2 * c]
                hi = (
                    cv(2 * c + 1) * s_coef[cb + _CO_W + 2 * c + 1]
                    if hi_live
                    else Float32(0.0)
                )
                nz = cutlass.min(cutlass.max(Int32(2 * c + 2) - P, Int32(0)), Int32(2))
                keep = Uint32(Uint64(0xFFFFFFFF) >> Uint64(nz * 16))
                words.append(U.pack_f16x2(lo, hi) & keep)
                continue
            lo = _sel(
                2 * c < P,
                cv(2 * c) * s_coef[cb + _CO_W + 2 * c],
                0.0,
                Float32,
            )
            hi = (
                _sel(
                    2 * c + 1 < P,
                    cv(2 * c + 1) * s_coef[cb + _CO_W + 2 * c + 1],
                    0.0,
                    Float32,
                )
                if hi_live
                else Float32(0.0)
            )
            words.append(U.pack_f16x2(lo, hi))
        U.tmem_st_u32(ct, tbase + dcol, words)
        U.tmem_wait_st()
        U.fence_before_sync()
        cute.arch.mbarrier_arrive(self._mb(ctx, "d_ready", hh))
        if flush and ctx.get("xrow", False):
            # persistent row loop, flush rows: the last head has read its window (its D words used
            # every value), so it releases both u-window buffers to the next row's heads 0 / 1 (the
            # cross-row phase of u_free, parity 1; heads 0 / 1 arrived the in-row one above). The
            # next writer is warp 0's bulk copy (async proxy): fence the generic reads first.
            last = Int32(hh == self.hpc - 1)
            U.mbar_arrive_if(last, ctx["u_free"][0].toint(), fence=True)
            U.mbar_arrive_if(last, ctx["u_free"][1].toint())
        return vv

    def _epilogue(self, hh, rcol, ctx, vv, early_release):
        """Outputs and u appends of head hh from the readout R (cols rcol..) and history Y
        (cols rcol + 16..) TMEM columns; arrives epi_done[hh] once its TMEM reads are done.
        ``vv`` None (vsmem, verify rows): v from the u-window buffer, then u_free[hh]."""
        T = self.T
        ring_t = BFloat16 if self.u_bf16 else Float16  # u ring element type
        ridx, base, P, brow, hv0 = (
            ctx.get("ridx", ctx["sidx"]),  # the u ring's row
            ctx["base"],
            ctx["P"],
            ctx["brow"],
            ctx["hv0"],
        )
        ct, row, tbase = ctx["ct"], ctx["row"], ctx["tbase"]
        u_cache, out, s_coef = ctx["u_cache"], ctx["out"], ctx["s_coef"]
        hv = hv0 + hh
        cb = hh * _COEF
        nld = 1 << (T - 1).bit_length()  # tcgen05.ld widths are powers of two
        rk = U.tmem_ld(ct, tbase + rcol, nld)
        yk = U.tmem_ld(ct, tbase + rcol + 16, nld)
        if vv is None:
            # vsmem (verify rows): this row's v values from head hh's u-window buffer
            # (slots wv..wv + T - 1), read under the TMEM loads; then the buffer may take
            # head hh + 2's window
            vb = (
                ctx["uw_addr"]
                + (hh % 2) * (W_RING * V_DIM * 2)
                + (self.wv * V_DIM) * 2
                + row * 2
            )
            vraw = [U.lds_u16(vb + t * (V_DIM * 2)) for t in range(T)]
            vsf = [U.u16_as_f32(w, True) for w in vraw]  # v is bf16
            if not isinstance(hh, int):
                # rolled head loop (dynamic hh): heads 0 / 1 only, predicated fence + arrive
                U.mbar_arrive_if(
                    Int32(hh < 2),
                    (ctx["u_free"][0] + cutlass.min(hh, Int32(1))).toint(),
                    fence=True,
                )
            elif hh < 2:
                # head hh + 2's window and v rows are bulk-copied (async proxy) over these slots
                # once u_free completes: the v values must be in registers first (vsf uses them,
                # so the arrive issues after the loads return) and the generic-proxy reads must be
                # ordered before the async-proxy writes (without both, Rubin showed a WAR race:
                # 1-2 output rows differing in 7-40 of 40 runs at T = 6 / 8, B >= 2048)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(self._mb(ctx, "u_free", hh))
            elif ctx.get("xrow", False) and hh == self.hpc - 1:
                # persistent row loop: the next row's head 0 / 1 windows (+ v rows) go into both
                # buffers once this last head has read its v rows (heads 2 / 3 read their u and v
                # slots earlier in program order): one proxy fence covers both, then the
                # cross-row completion of u_free[0] / u_free[1] (parity 1; parity 0 is the in-row one)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(ctx["u_free"][0])
                cute.arch.mbarrier_arrive(ctx["u_free"][1])
        U.tmem_wait_ld()
        egp = s_coef[cb + _CO_EGP]
        if early_release:
            # verify rows: all four column groups first, then release the head buffer
            # (the readout two heads ahead waits on it)
            hks = [egp * rk[t] + yk[t] for t in range(T)]
            rq = U.tmem_ld(ct, tbase + rcol + T, nld)
            yq = U.tmem_ld(ct, tbase + rcol + 16 + T, nld)
            U.tmem_wait_ld()
            U.fence_before_sync()
            cute.arch.mbarrier_arrive(self._mb(ctx, "epi_done", hh))
        uu = []
        for t in range(T):
            if early_release:
                hk = hks[t]
            else:
                hk = egp * rk[t] + yk[t]
            acc = (
                s_coef[cb + _CO_BETA + t] * (vsf[t] if vv is None else Float32(vv[t]))
                - s_coef[cb + _CO_BG + t] * hk
            )
            for s_ in range(t):
                acc = acc - s_coef[cb + _CO_A + t * 8 + s_] * uu[s_]
            uu.append(acc)
        if not early_release:
            # flush rows (register-bound): the u solve runs before the q columns are read,
            # so the k-side (hk, v) and q-side (rq, yq) values are never live together
            rq = U.tmem_ld(ct, tbase + rcol + T, nld)
            yq = U.tmem_ld(ct, tbase + rcol + 16 + T, nld)
            U.tmem_wait_ld()
            U.fence_before_sync()
            cute.arch.mbarrier_arrive(self._mb(ctx, "epi_done", hh))

        # u appends at (base + P + t) & 31 wrap at most once: two base rows (s0, s0 - 32)
        # plus immediate offsets address all T appends
        s0 = (base + P) & RING_MASK
        u_a = (
            u_cache.iterator.toint()
            + Int64(cute.crd2idx((ridx, hv, s0, row), u_cache.layout)) * 2
        )
        u_b = u_a - Int64(RING_SLOTS * V_DIM * 2)
        # persistent row loop: y and u stores from one base address each (+ immediates), as the
        # flush path's u appends (the per-store tensor addressing spilled at T = 7 / 8 in the loop)
        pz = ctx.get("xrow", False)
        if pz:
            y_a = (
                out.iterator.toint()
                + Int64(cute.crd2idx((brow, 0, hv, row), out.layout)) * 2
            )
        fnorm = self.fused_norm and ctx.get("s_nrm") is not None
        ys = []
        for t in range(T):
            hq = egp * rq[t] + yq[t]
            y = s_coef[cb + _CO_EG + t] * hq
            for s_ in range(t + 1):
                y = y + s_coef[cb + _CO_BQ + t * 8 + s_] * uu[s_]
            if fnorm:
                ys.append(y)
            elif pz:
                U.stg_u16(
                    y_a + Int64(t * self.hv_io * V_DIM * 2), U.f32_to_u16(y, True)
                )
            else:
                out[brow, t, hv, row] = BFloat16(y)
            if early_release and not pz:
                u_cache[ridx, hv, (base + P + t) & RING_MASK, row] = ring_t(uu[t])
            else:
                ua = _sel(s0 + t < RING_SLOTS, u_a, u_b, Int64)
                U.stg_u16(ua + Int64(t * V_DIM * 2), U.f32_to_u16(uu[t], self.u_bf16))
        if fnorm:
            self._norm_store(hh, ys, ctx)

    def _norm_store(self, hh, ys, ctx):
        """fused_norm: head hh's T outputs as the gated RMSNorm stores them. Each math thread holds
        V row ``row`` of every token; a token's 128 rows reduce as the Triton norm's 16 threads x 8
        elements do (the 8-lane group sums, then the 16 partials through SMEM and barrier 1)."""
        T = self.T
        row, brow, out, s_nrm = ctx["row"], ctx["brow"], ctx["out"], ctx["s_nrm"]
        norm_z, norm_z_row, norm_w, norm_eps = ctx["norm"]
        hv = ctx["hv0"] + hh
        buf = (hh % 2) * (8 * 16)
        grp = row >> 3
        # inputs the norm reads: the stored bf16 output, its weight and the gate z
        xs = [U.u16_as_f32(U.f32_to_u16(y, True), True) for y in ys]
        wv = U.u16_as_f32(U.ldg_u16_if(Int32(1), norm_w + Int64(row) * 2), True)
        za = norm_z + (Int64(brow) * T * Int64(norm_z_row) + Int64(hv * V_DIM + row)) * 2
        zs = [
            U.u16_as_f32(U.ldg_u16_if(Int32(1), za + Int64(norm_z_row) * (2 * t)), True)
            for t in range(T)
        ]
        for t in range(T):
            s_nrm[buf + t * 16 + grp] = U.norm_group_sumsq(xs[t])
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        for t in range(T):
            rstd = U.norm_rstd16([s_nrm[buf + t * 16 + i] for i in range(16)], norm_eps)
            o = U.norm_gate_out(xs[t], rstd, wv, zs[t])
            U.stg_u16(
                out.iterator.toint()
                + Int64(cute.crd2idx((brow, t, hv, row), out.layout)) * 2,
                o,
            )

    def _load_seed(self, ctx, rand_seed):
        """SR variants, flush rows only: the Philox key / LCG mixing key (verify rows never
        read the seed)."""
        if self.sr_rounds > 0 or self.sr == "lcg":
            seed = Int64(rand_seed[0])
            ctx["seed_lo"] = Uint32(seed & Int64(0xFFFFFFFF))
            ctx["seed_hi"] = Uint32(seed >> Int64(32))
        if self.sr == "lcg":
            ctx["lcg_k"] = U.fmix32(
                ctx["seed_lo"] ^ U.fmix32(ctx["seed_hi"] ^ Uint32(0x9E3779B9))
            )
        elif self.sr == "lcg_clock":
            ctx["lcg_k"] = U.fmix32(U.clock32())

    def _mbar_list(self, smem, n):
        """n single mbarriers; with ``roll`` / ``rmw_loop`` one contiguous array (dynamic
        head / half index)."""
        if self.roll or self.rmw_loop:
            arr = smem.allocate_array(cutlass.Int64, n)
            return [arr + i for i in range(n)]
        return [smem.allocate_array(cutlass.Int64, 1) for _ in range(n)]

    @staticmethod
    def _mb(ctx, name, idx):
        """mbarrier ``idx`` of kind ``name`` (a dynamic idx needs ``roll``'s contiguous array)."""
        if isinstance(idx, int):
            return ctx[name][idx]
        return ctx[name][0] + idx

    def _rmw_setup(self, hh, g, ctx):
        """Per (head, half) RMW inputs, ahead of the half's waits: (egp, stage row address,
        r0..r3) with r = the Philox counter (lo, hi) of the half's first chunk, or the four LCG
        chain states (pairs 0..3), or zeros (round-to-nearest)."""
        row, s_coef = ctx["row"], ctx["s_coef"]
        egp = s_coef[hh * _COEF + _CO_EGP]
        srow = ctx["st_addr"] + g * 16384 + row * 128
        z = Uint32(0)
        if self.sr_rounds > 0:
            # flat element offset of S[sidx, hv, row, 64 g] in the [pool, HV, V, K] state pool
            # (FlashInfer Mamba's Philox counter: state offset + row * K + column)
            off_row = (
                (
                    Int64(ctx["sidx"]) * self.hv_io
                    + Int64(ctx["hv_base"] + ctx["hv0"] + hh)
                )
                * V_DIM
                + Int64(row)
            ) * K_DIM + g * 64
            return (
                egp,
                srow,
                Uint32(off_row & Int64(0xFFFFFFFF)),
                Uint32(off_row >> Int64(32)),
                z,
                z,
            )
        if self.sr_lcg:
            # LCG dither (3 instructions per pair): this (state row, half)'s seed = murmur3 of
            # its flat id and the per-flush key; pair n (= 4 chunk + i) takes LCG state n folded
            # as s ^ (s >> 16), run as 4 jump-ahead chains (x_i -> A^4 x_i + C4) so the 4 pairs
            # of a 16-B word are independent instructions. Known weakness: the two 13-bit fields
            # of a word XOR to the state's low bits and consecutive words are consecutive LCG
            # states, so dithers within one (row, half) are not jointly independent. A PCG
            # RXS-M-XS output fixes that but costs ~7 instructions per word -- as much as
            # 5-round Philox (measured), so use "philox" when that matters.
            cid = (
                (
                    Int64(ctx["sidx"]) * self.hv_io
                    + Int64(ctx["hv_base"] + ctx["hv0"] + hh)
                )
                * V_DIM
                + Int64(row)
            ) * 2 + g
            x0 = U.fmix32(
                Uint32(cid & Int64(0xFFFFFFFF)) * Uint32(0x9E3779B9)
                + Uint32(cid >> Int64(32)) * Uint32(0x85EBCA6B)
                + ctx["lcg_k"]
            )
            lx = [x0]
            for _ in range(3):
                lx.append(lx[-1] * Uint32(U.LCG_A) + Uint32(U.LCG_C))
            return (egp, srow, lx[0], lx[1], lx[2], lx[3])
        return (egp, srow, z, z, z, z)

    def _rmw_wait(self, hh, g, ctx):
        """RMW loop: half g's fold and readout (g = 0: f_mma_a, rd0; g = 1: f_mma_b, mma_done
        -- one array each, index g * HPK + hh), as _head's waits. Persistent row loop (``phf`` in
        ctx): the folds and rd0 complete on flush rows only (parity phf), mma_done on every row
        (parity ph)."""
        pf = ctx.get("phf", 0)
        cute.arch.mbarrier_wait(ctx["f_mma_a"][0] + (g * HPK + hh), pf)
        pr = _sel(g == 0, pf, ctx["ph"], Int32) if "phf" in ctx else 0
        cute.arch.mbarrier_wait(ctx["rd0"][0] + (g * HPK + hh), pr)
        U.fence_after_sync()

    def _rmw_chunks(self, gq, ctx, egp, srow, fv, rws, lx):
        """The ldn / 8 chunks of TMEM column group gq (static or dynamic) of one half: load from
        the stage, e^{G_P} S0 + F, round (RN / cvt.rs), store back. Returns the advanced LCG
        chain states."""
        row = ctx["row"]
        nch = self.ldn // 8
        for cc in range(nch):
            chunk = gq * nch + cc
            sa = srow + ((chunk ^ (row % 8)) * 16)
            ow = U.lds_v4(sa)
            if self.sr_lcg:
                rw = [x ^ (x >> Uint32(16)) for x in lx]  # 16-bit fold
                lx = [x * Uint32(U.LCG_A4) + Uint32(U.LCG_C4) for x in lx]
            nw = []
            for i in range(4):
                if self.mixed_fma:
                    lo, hi = U.ffma_f16x2_f32(
                        ow[i], egp, fv[cc * 8 + 2 * i], fv[cc * 8 + 2 * i + 1]
                    )
                else:
                    lo = egp * U.f16x2_lo_f32(ow[i]) + fv[cc * 8 + 2 * i]
                    hi = egp * U.f16x2_hi_f32(ow[i]) + fv[cc * 8 + 2 * i + 1]
                if self.sr_rounds > 0:
                    nw.append(U.pack_f16x2_rs(lo, hi, rws[cc][i]))
                elif self.sr_lcg:
                    nw.append(U.pack_f16x2_rs(lo, hi, rw[i]))
                else:
                    nw.append(U.pack_f16x2(lo, hi))
            U.sts_v4(sa, nw[0], nw[1], nw[2], nw[3])
        return lx

    def _rmw_group(self, hh, g, gq, ctx, egp, srow, r0, r1, r2, r3):
        """RMW loop (``rmw_loop``): TMEM column group gq (dynamic) of half g (dynamic) of head hh;
        the same steps as one group of _rmw. Returns the carried r0..r3 (LCG: advanced)."""
        ct, tbase = ctx["ct"], ctx["tbase"]
        ldn = self.ldn
        fv = U.tmem_ld(ct, tbase + 64 + gq * ldn, ldn)
        rws = None
        if self.sr_rounds > 0:
            # one Philox draw per 8-element chunk, issued under the TMEM load; off_row is a
            # multiple of 64, so the chunk offsets never carry into the high word
            rws = U.philox4x32_streams(
                ctx["seed_lo"],
                ctx["seed_hi"],
                r0 + Uint32(gq * ldn),
                r1,
                ldn // 8,
                self.sr_rounds,
            )
        U.tmem_wait_ld()
        # half 0's last group: F may be overwritten (fold half 1)
        U.fence_before_sync()
        U.mbar_arrive_if(
            Int32(g == 0) & Int32(gq == self.ngrp - 1),
            (ctx["f_epi_a"][0] + hh).toint(),
        )
        lx = self._rmw_chunks(gq, ctx, egp, srow, fv, rws, [r0, r1, r2, r3])
        if self.sr_lcg:
            return lx[0], lx[1], lx[2], lx[3]
        return r0, r1, r2, r3

    def _rmw(self, hh, g, ctx):
        """Flush rows, unrolled RMW (builds without ``rmw_loop``): S_half = e^{G_P} S0_half + F
        (TMEM cols 64..127) in place on state stage g (= half g of head hh), then publish it to
        warp 0's TMA store (f_gdone). Same arithmetic and noise bits as the RMW loop."""
        ct, row, tbase, s_coef = ctx["ct"], ctx["row"], ctx["tbase"], ctx["s_coef"]
        egp = s_coef[hh * _COEF + _CO_EGP]
        srow = ctx["st_addr"] + g * 16384 + row * 128
        if self.sr_rounds > 0:
            # flat element offset of S[sidx, hv, row, 64 g] in the [pool, HV, V, K] state pool
            # (FlashInfer Mamba's Philox counter: state offset + row * K + column)
            off_row = (
                (
                    Int64(ctx["sidx"]) * self.hv_io
                    + Int64(ctx["hv_base"] + ctx["hv0"] + hh)
                )
                * V_DIM
                + Int64(row)
            ) * K_DIM + g * 64
        ldn = self.ldn  # TMEM columns per load (see __init__)
        nch = ldn // 8  # 8-element chunks per load
        nst = nch  # Philox streams per call
        if self.sr_lcg:
            # LCG dither: the same chains as _rmw_setup (see there)
            cid = (
                (
                    Int64(ctx["sidx"]) * self.hv_io
                    + Int64(ctx["hv_base"] + ctx["hv0"] + hh)
                )
                * V_DIM
                + Int64(row)
            ) * 2 + g
            x0 = U.fmix32(
                Uint32(cid & Int64(0xFFFFFFFF)) * Uint32(0x9E3779B9)
                + Uint32(cid >> Int64(32)) * Uint32(0x85EBCA6B)
                + ctx["lcg_k"]
            )
            lx = [x0]
            for _ in range(3):
                lx.append(lx[-1] * Uint32(U.LCG_A) + Uint32(U.LCG_C))
        for grp in range(64 // ldn):
            fv = U.tmem_ld(ct, tbase + 64 + grp * ldn, ldn)
            if self.sr_rounds > 0:
                # one Philox draw per 8-element chunk (word i -> pair i), all of the load's
                # chunks in one call, issued under the TMEM load. off_row is a multiple of 64, so
                # the chunk offsets never carry into the high word.
                off = off_row + grp * ldn
                off_lo, off_hi = (
                    Uint32(off & Int64(0xFFFFFFFF)),
                    Uint32(off >> Int64(32)),
                )
                rws = U.philox4x32_streams(
                    ctx["seed_lo"], ctx["seed_hi"], off_lo, off_hi, nst, self.sr_rounds
                )
            U.tmem_wait_ld()
            if g == 0 and grp == 64 // ldn - 1:
                U.fence_before_sync()
                cute.arch.mbarrier_arrive(
                    self._mb(ctx, "f_epi_a", hh)
                )  # F may be overwritten (fold half 1)
            for cc in range(nch):
                chunk = grp * nch + cc
                sa = srow + ((chunk ^ (row % 8)) * 16)
                ow = U.lds_v4(sa)
                if self.sr_lcg:
                    rw = [x ^ (x >> Uint32(16)) for x in lx]  # 16-bit fold
                    lx = [x * Uint32(U.LCG_A4) + Uint32(U.LCG_C4) for x in lx]
                nw = []
                for i in range(4):
                    lo = egp * U.f16x2_lo_f32(ow[i]) + fv[cc * 8 + 2 * i]
                    hi = egp * U.f16x2_hi_f32(ow[i]) + fv[cc * 8 + 2 * i + 1]
                    if self.sr_rounds > 0:
                        nw.append(U.pack_f16x2_rs(lo, hi, rws[cc % nst][i]))
                    elif self.sr_lcg:
                        nw.append(U.pack_f16x2_rs(lo, hi, rw[i]))
                    else:
                        nw.append(U.pack_f16x2(lo, hi))
                U.sts_v4(sa, nw[0], nw[1], nw[2], nw[3])
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(self._mb(ctx, "f_gdone", 2 * hh + g))

    def _head(self, hh, flush, ctx):
        if flush:
            # Flush rows: the fold F = D . Kt does not need the state, so warp 1 issues fold
            # half 0 before the state lands; each half's RMW runs as soon as the readout has
            # consumed that stage, so the stage is stored and reloaded early and this head's
            # epilogue overlaps the next head's state load.
            vv = self._d_build(hh, 32, ctx, flush=True)
            cute.arch.mbarrier_wait(self._mb(ctx, "f_mma_a", hh), 0)
            cute.arch.mbarrier_wait(self._mb(ctx, "rd0", hh), 0)
            U.fence_after_sync()
            self._rmw(hh, 0, ctx)
            cute.arch.mbarrier_wait(self._mb(ctx, "f_mma_b", hh), 0)
            cute.arch.mbarrier_wait(self._mb(ctx, "mma_done", hh), 0)
            U.fence_after_sync()
            self._rmw(hh, 1, ctx)
            self._epilogue(hh, 0, ctx, vv, False)
        else:
            vv = self._d_build(hh, 64 + (hh % 2) * 8, ctx, self.vsmem)
            cute.arch.mbarrier_wait(self._mb(ctx, "mma_done", hh), ctx.get("ph", 0))
            U.fence_after_sync()
            # vsmem: v from SMEM (vv None)
            self._epilogue(hh, (hh % 2) * 32, ctx, None if self.vsmem else vv, True)

    # ------------------------------------------------------------------ compile
    @staticmethod
    def compile(
        device: torch.device,
        T: int,
        h: int,
        hv: int,
        u_bf16: bool,
        use_row_order: bool,
        hpc: Optional[int] = None,
        sr: str = "",
        sr_rounds: int = 0,
        persistent: bool = False,
        h_io: Optional[int] = None,
        hv_io: Optional[int] = None,
        min_blocks: int = _MIN_BLOCKS_PER_SM,
        parts: int = 1,
        k_f16: bool = False,
        qk_io: Optional[int] = None,
        v_io: Optional[int] = None,
        ab_dyn: bool = False,
        pdl: bool = False,
        ring_rows: bool = False,
        fused_conv: bool = False,
        fused_norm: bool = False,
    ):
        """Compile one specialisation for ``device``'s architecture (callers use ``_get_kernel``).
        ``h_io`` / ``hv_io``: head counts of the input / output tensors' token rows (see
        ``__init__``); q / k / v / a / b / out are then head-slice views with those strides.
        ``min_blocks``: the non-persistent launch bound's CTAs per SM (see ``_launch``).
        ``parts`` > 1: the state and rings are 5-D uGPU buffers (see ``__init__``).
        ``qk_io`` / ``v_io`` / ``ab_dyn`` / ``pdl``: strided inputs and the dependent launch
        (see ``__init__``)."""
        h_io = h if h_io is None else h_io
        hv_io = hv if hv_io is None else hv_io
        qk_io = h_io if qk_io is None else qk_io
        v_io = hv_io if v_io is None else v_io
        batch = cute.sym_int()
        pool = cute.sym_int()
        # ring_rows: the rings have rows of their own (ring_idx), not one per state slot
        ring_pool = cute.sym_int() if ring_rows else pool
        ft = U.make_fake_tensor_lead_dyn
        uring = BFloat16 if u_bf16 else Float16

        def fs(
            dtype, heads, inner, div, lead=None
        ):  # state / ring: [pool, heads, *inner] or 5-D parts
            if parts == 1:
                return ft(dtype, (pool if lead is None else lead, heads, *inner), div)
            dense = [1]
            for d in reversed(inner[1:]):
                dense.insert(0, dense[0] * d)
            hstride = dense[0] * inner[0]
            return cute.runtime.make_fake_tensor(
                dtype,
                (pool, parts, heads // parts, *inner),
                stride=(
                    cute.sym_int64(divisibility=div),
                    cute.sym_int64(divisibility=div),
                    hstride,
                    *dense,
                ),
                assumed_align=max(div * dtype.width // 8, 1),
            )

        def fab():  # a / b: static per-token stride hv_io, or (ab_dyn) a runtime one
            if not ab_dyn:
                return ft(BFloat16, (batch, T, hv), 1, (T, hv_io))
            return cute.runtime.make_fake_tensor(
                BFloat16,
                (batch, T, hv),
                stride=(cute.sym_int64(divisibility=1), cute.sym_int64(divisibility=1), 1),
                assumed_align=2,
            )

        args = [
            ft(BFloat16, (batch, T, h, K_DIM), 8, (T, qk_io, K_DIM)),  # q
            ft(BFloat16, (batch, T, h, K_DIM), 8, (T, qk_io, K_DIM)),  # k
            ft(BFloat16, (batch, T, hv, V_DIM), 8, (T, v_io, V_DIM)),  # v
            fab(),  # a
            fab(),  # b
            ft(Float32, (hv,), 1),  # A_log (fp32)
            ft(Float32, (hv,), 1),  # dt_bias (fp32)
            fs(Float16, hv, (V_DIM, K_DIM), 8),  # state
            ft(Int32, (batch,), 1),  # state indices
            fs(Float16 if k_f16 else BFloat16, h, (RING_SLOTS, K_DIM), 8, ring_pool),  # k ring
            fs(uring, hv, (RING_SLOTS, V_DIM), 8, ring_pool),  # u ring (bf16 / fp16)
            fs(Float32, hv, (RING_SLOTS,), 1, ring_pool),  # g ring
            ft(Int32, (batch,), 1),  # hist_len
            ft(Int32, (batch,), 1),  # cache_base
            ft(Int32, (batch,), 1),  # row order
            ft(BFloat16, (batch, T, hv, V_DIM), 8, (T, hv_io, V_DIM)),  # out
            ft(Int64, (1,), 1),  # rand_seed (read by "philox" / "lcg")
        ]
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        tgt = gdn_device_target(device)
        mixed_fma = (tgt.major, tgt.minor) in _FHFMA2_CC
        small_b = _small_b_build(tgt, h, hv)
        obj = GdnReplayMtpUmmaRubin(
            T,
            h,
            hv,
            u_bf16,
            use_row_order,
            hpc,
            sr,
            sr_rounds,
            mixed_fma,
            persistent,
            h_io,
            hv_io,
            min_blocks,
            small_b,
            parts,
            k_f16,
            qk_io,
            v_io,
            ab_dyn,
            pdl,
            ring_rows,
            fused_conv,
            fused_norm,
        )
        opts = gdn_compile_options(device, cute.EnableTVMFFI(True))
        if persistent:
            return cute.compile[opts](
                obj.call_persist,
                *args,
                ft(Int32, (2,), 1),  # claim counter
                Float32(1.0),  # scale
                Int32(1),  # flush_min
                Int32(0),  # hv_base
                Int32(1),  # n_cta
                stream,
            )
        return cute.compile[opts](
            obj,
            *args,
            ft(Int32, (batch,), 1),  # ring rows (the state indices again without ring_rows)
            Float32(1.0),
            Int32(1),
            Int32(0),
            *([Int64(0)] * 5),  # fused conv addresses (x, weight, state, window, window rows)
            Int32(0),  # fused conv x row stride
            Int64(0),  # fused norm: z address
            Int32(0),  # fused norm: z row stride (elements a token)
            Int64(0),  # fused norm: weight address
            Float32(0.0),  # fused norm: eps
            stream,
        )


_KERNELS: dict = {}


def _get_kernel(
    device: torch.device,
    T: int,
    h: int,
    hv: int,
    u_bf16: bool,
    use_row_order: bool,
    hpc: int,
    sr: str,
    sr_rounds: int,
    persistent: bool = False,
    h_io: Optional[int] = None,
    hv_io: Optional[int] = None,
    min_blocks: int = _MIN_BLOCKS_PER_SM,
    parts: int = 1,
    k_f16: bool = False,
    qk_io: Optional[int] = None,
    v_io: Optional[int] = None,
    ab_dyn: bool = False,
    pdl: bool = False,
    ring_rows: bool = False,
    fused_conv: bool = False,
    fused_norm: bool = False,
):
    """The compiled kernel for this specialisation on ``device``'s architecture: one per
    (compile target, parameters, _SMALL_B_BUILD_CC); batch and pool sizes are dynamic."""
    h_io = h if h_io is None else h_io
    hv_io = hv if hv_io is None else hv_io
    qk_io = h_io if qk_io is None else qk_io
    v_io = hv_io if v_io is None else v_io
    tgt = gdn_device_target(device)
    key = (
        tgt.compile_key,
        (tgt.major, tgt.minor) in _SMALL_B_BUILD_CC,
        T,
        h,
        hv,
        u_bf16,
        use_row_order,
        hpc,
        sr,
        sr_rounds,
        persistent,
        h_io,
        hv_io,
        min_blocks,
        parts,
        k_f16,
        qk_io,
        v_io,
        bool(ab_dyn),
        bool(pdl),
        bool(ring_rows),
        bool(fused_conv),
        bool(fused_norm),
    )
    fn = _KERNELS.get(key)
    if fn is None:
        fn = GdnReplayMtpUmmaRubin.compile(
            device,
            T,
            h,
            hv,
            u_bf16,
            use_row_order,
            hpc,
            sr,
            sr_rounds,
            persistent,
            h_io,
            hv_io,
            min_blocks,
            parts,
            k_f16,
            qk_io,
            v_io,
            bool(ab_dyn),
            bool(pdl),
            bool(ring_rows),
            bool(fused_conv),
            bool(fused_norm),
        )
        _KERNELS[key] = fn
    return fn


# ============================================================================ host side
_F32_CACHE: dict = {}
_ROW_ID: dict = {}


def _cached_f32(t: torch.Tensor) -> torch.Tensor:
    """fp32 copy of a per-layer constant (A_log / dt_bias), cached by object identity and
    refreshed in place when the tensor is updated in place (its version counter, e.g. a weight
    reload; inference-mode tensors have no version counter, so for them only by identity). The
    copy keeps its address, so CUDA graphs captured before the update read the new values.

    The copy is written on the stream that makes it. Another stream (e.g. the other die's
    stream of a uGPU call) must not read it before that write completes, so the first time each
    other stream uses the copy it waits for the conversion's event (and the copy is recorded on
    that stream for the caching allocator). Later launches on that stream are ordered after the
    wait by stream order, so they make no CUDA call. A refresh synchronizes the device first
    (launches on any stream may still read the old values) and makes every other stream wait
    again. Under graph capture a hit makes no CUDA call: the copy was enqueued before the
    capture, and torch.cuda.graph synchronizes the device when it begins. A first sight (or an
    updated tensor) during a capture converts inside the graph and is not cached."""
    if t.dtype == torch.float32 and t.is_contiguous():
        return t
    sid = torch._C._cuda_getCurrentRawStream(t.device.index)
    ver = -1 if t.is_inference() else t._version
    ent = _F32_CACHE.get(id(t))
    if ent is not None and ent[3] == ver and sid in ent[2]:
        return ent[
            0
        ]  # this stream is already ordered after the conversion: no CUDA call
    cur = torch.cuda.current_stream(t.device)
    capturing = torch.cuda.is_current_stream_capturing()
    if ent is None or ent[3] != ver:
        if capturing:
            return t.float().contiguous()
        if ent is not None and ent[0].shape == t.shape:
            # in place: graphs captured earlier keep reading this address
            torch.cuda.synchronize(t.device)
            c = ent[0]
            c.copy_(t)
            ev = torch.cuda.Event()
            ev.record(cur)
            _F32_CACHE[id(t)] = (c, ev, {sid}, ver)
            return c
        c = t.float().contiguous()
        ev = torch.cuda.Event()
        ev.record(cur)
        if ent is None:
            import weakref

            weakref.finalize(t, _F32_CACHE.pop, id(t), None)
        _F32_CACHE[id(t)] = (c, ev, {sid}, ver)
        return c
    c, ev, synced, _ = ent
    if not capturing:
        cur.wait_event(ev)
        c.record_stream(cur)
        synced.add(sid)
    return c


# archs this build was validated on (sm_107a: tuned and measured there; sm_100a: the same
# tests); anything else raises
_SUPPORTED_CC = ((10, 0), (10, 7))
_CVT_RS_CC = (
    (10, 0),
    (10, 3),
    (10, 7),
)  # archs whose ptxas accepts cvt.rs (see mamba/conversion.cuh)
_CTA_SLOTS_PER_SM = _MIN_BLOCKS_PER_SM
# Heads per CTA (auto_heads_per_cta): 1 while the hpc = 1 grid fits cap1 waves of CTA slots, 2
# while the hpc = 2 grid fits cap2 waves, else all of a key head's value heads. On the archs of
# _HPC_TUNED_CC the caps depend on T, flush_hint, the rounding mode and whether the launch covers
# the whole device (fitted on Rubin, see auto_heads_per_cta); elsewhere cap1 = 1 and cap2 =
# _HPC2_MAX_WAVES.
_HPC_TUNED_CC = ((10, 7),)
_HPC1_MAX_WAVES = 1.0
_HPC2_MAX_WAVES = 8  # without T
_HPC2_MAX_WAVES_MTP = 1.5  # T >= 2, round to nearest
# T >= 2 with LCG / Philox of at most _HPC_PHILOX_FAST_ROUNDS rounds (and slower Philox at T >= 5
# with "flush_first" or on a whole-device launch): the RMW loop of the flush rows runs longer
_HPC2_MAX_WAVES_MTP_SR = 1.75
_HPC2_MAX_WAVES_T1_FF = 2.4  # T = 1, "flush_first" on a partial-device launch
_HPC2_MAX_WAVES_T1 = 2.7  # T = 1 otherwise
# Philox with more than _HPC_PHILOX_FAST_ROUNDS rounds: longer CTAs, so hpc = 2 holds longer.
# _HPC2_MAX_WAVES_PHILOX_FF (T = 1, T = 2-4): "flush_first" on a partial-device launch;
# _HPC2_MAX_WAVES_PHILOX_PART (T = 1, T = 2-4, T >= 5): a partial-device launch without it (the
# uGPU path of gated_delta_rule_mtp_ucache_flush, rows in random order), where hpc = 4 did not
# win at T <= 4 by B = 128 per die with a serving-like L2; _HPC2_MAX_WAVES_PHILOX: a
# whole-device launch at T <= 4.
_HPC_PHILOX_FAST_ROUNDS = 6
_HPC2_MAX_WAVES_PHILOX_FF = (2.75, 2.5)
_HPC2_MAX_WAVES_PHILOX_PART = (8.0, 8.0, 2.1)
_HPC2_MAX_WAVES_PHILOX = 3.6
# flush_hint "none" (no flush rows, equal work per row): (cap1, cap2); cap2 = _HPC2_NO_FLUSH_ONE_WAVE
# (hpc = 2 while its grid fits one wave) at T = 1 and on whole-device launches at T = 2-4, where
# hpc = 4 at 0.8-1 waves was slower with a serving-like L2.
_HPC_NO_FLUSH_WAVES = (0.65, 0.8)
_HPC2_NO_FLUSH_ONE_WAVE = 1.0
# Small-batch dispatch on these archs: a non-persistent launch whose grid is small (_small_grid:
# every CTA resident from the start), or without flush rows (equal work per row), ignores
# row_order -- it has nothing to balance there and costs a dependent global round trip.
_SMALL_B_DISPATCH_CC = ((10, 7),)
_SMALL_GRID_CTAS_PER_SM = 2
# Small non-persistent grids (_small_grid) with 4 value heads per key head compile with a
# launch bound of _SMALL_GRID_CTAS_PER_SM CTAs per SM (more registers for latency-bound CTAs);
# not the register-split builds (their setmaxnreg counts assume 4) nor Philox (no gain).
_SMALL_GRID_LB_CC = ((10, 7),)
# small_b build options (GdnReplayMtpUmmaRubin) for the 1- and 2-heads-per-CTA kernels, which with
# 4 value heads per key head run only at small B (few, latency-bound CTAs): the round-to-nearest
# flush RMW loop, the prologue reorder at T >= 5 with 1 head, no rolled head loop with 2 heads
# (round to nearest). Same bits.
_SMALL_B_BUILD_CC = ((10, 7),)


def _small_b_build(target, h: int, hv: int) -> bool:
    """GdnReplayMtpUmmaRubin's small_b build options (_SMALL_B_BUILD_CC): on those archs, with 4
    value heads per key head only."""
    return (target.major, target.minor) in _SMALL_B_BUILD_CC and hv == h * HPK


def _small_grid(n_key_heads: int, hpk: int, hpc: int, batch: int, num_sms: int) -> bool:
    """A non-persistent grid of at most _SMALL_GRID_CTAS_PER_SM CTAs per SM (key heads x
    hpk / hpc CTAs per row): every CTA is resident from the launch on, so the row order buys no
    balance and the CTAs are latency-bound."""
    return n_key_heads * (hpk // hpc) * batch <= _SMALL_GRID_CTAS_PER_SM * int(num_sms)


def auto_heads_per_cta(
    batch: int,
    n_key_heads: int,
    num_sms: int,
    heads_per_key: int = HPK,
    T: Optional[int] = None,
    flush_hint: Optional[str] = None,
    sr: str = "",
    sr_rounds: int = 0,
    whole_device: bool = False,
) -> int:
    """Heads per CTA for ``batch`` rows over ``n_key_heads`` key heads on ``num_sms`` SMs: 1
    while the hpc = 1 grid (rows x key heads x ``heads_per_key`` CTAs) fits cap1 waves of CTA
    slots (4 x ``num_sms``), 2 (when it divides ``heads_per_key``) while the hpc = 2 grid fits
    cap2 waves, else all ``heads_per_key`` value heads of a key head in one CTA.

    One CTA runs its value heads one after another, so splitting a key head over 2 or 4 CTAs
    (each repeating the per-key-head prologue) shortens the per-CTA chain and gives the
    scheduler finer-grained work, until the repeated prologue costs more than that saves.

    Without ``T`` (``_launch`` passes it on _HPC_TUNED_CC only, so on sm_100a): cap1 = 1, cap2 =
    8. B200 (148 SMs), 4 value heads per key head: 1 for B <= 9, 2 for B <= 148, else 4 (3 per
    key head: 1 for B <= 12, else 3); the Rubin caps measured slower on B200 with a cold L2.

    With ``T`` (sm_107a), ``whole_device`` is True for a launch over all of the device's SMs and
    False for a partial-device launch (``num_sms`` below the device's SM count). The only
    partial launches measured are uGPU per-die launches (8 of 16 key heads on one 106-SM Rubin
    die); other partial launches, e.g. a 132-SM green context, get the same caps untimed. Eager
    uGPU calls with B x HV below ``_UGPU_EAGER_MIN_STATES`` (8192) run each die's key heads over
    the whole device, back to back: they are whole-device launches of 8 key heads on 212 SMs, so
    their batch boundaries are twice the per-die ones below and "flush_first" counts as None
    (e.g. round to nearest at T >= 2: hpc = 4 from B = 80, not 40). cap1 = 1, and cap2 =

    * 1.5 at T >= 2 with round to nearest; 1.75 at T >= 2 with stochastic rounding ``sr``: LCG,
      Philox with at most 6 ``sr_rounds``, and Philox with more rounds at T >= 5 on a
      whole-device launch or with "flush_first" on a partial-device launch;
    * 2.4 at T = 1 with ``flush_hint`` "flush_first" on a partial-device launch, else 2.7
      (a whole-device launch treats "flush_first" as None: hpc = 4 overtakes later there with
      cold L2);
    * Philox with more than 6 ``sr_rounds`` (the API default is 10), except at T >= 5 as above:
      with "flush_first" on a partial-device launch 2.75 (T = 1) / 2.5 (T = 2-4); on a
      partial-device launch without it 8 (T <= 4) / 2.1 (T >= 5); on a
      whole-device launch 3.6 (T <= 4);
    * ``flush_hint`` "none" (no flush rows), any rounding: cap1 = 0.65 (1 at T >= 5 with an odd
      ``heads_per_key``, which has no hpc = 2); cap2 = 1 at T = 1 and on a whole-device launch at
      T = 2-4, else 0.8.

    Batch boundaries per Rubin die (8 key heads on 106 SMs, B = rows; a whole-device launch of 16
    key heads on 212 SMs has the same waves and boundaries):

    * T >= 2: hpc = 1 for B <= 13, 2 up to 39 (round to nearest) / 46 (stochastic rounding),
      else 4; T = 1: 2 up to 63 with "flush_first" on a die, else up to 71;
    * Philox above 6 rounds: with "flush_first" on a die hpc = 4 from 73 (T = 1) / 67 (T = 2-4) /
      47 (T >= 5); on a die without it from 213 (T <= 4) / 56 (T >= 5);
      whole-device from 96 (T <= 4) / 47 (T >= 5);
    * no flush rows: hpc = 1 for B <= 8, 2 up to 26 at T = 1 and on whole-device launches at
      T = 2-4, up to 21 otherwise, else 4; an odd heads per key (e.g. 3): hpc = 1 for B <= 11 at
      T <= 4 and 17 at T >= 5, else all of them;
    * the 8-wave cap (no ``T``): 2 up to B = 212.

    The measurements behind the caps are in PR #6791.
    """
    slots = _CTA_SLOTS_PER_SM * num_sms
    hpk = heads_per_key
    # flush rows first moves the caps of partial-device launches only
    ff = flush_hint == "flush_first" and not whole_device
    slow_philox = sr == "philox" and sr_rounds > _HPC_PHILOX_FAST_ROUNDS
    cap1: float = _HPC1_MAX_WAVES
    cap2: float
    if T is None:
        cap2 = _HPC2_MAX_WAVES
    elif flush_hint == "none":
        cap1, cap2 = _HPC_NO_FLUSH_WAVES
        if T == 1 or (whole_device and T <= 4):
            cap2 = _HPC2_NO_FLUSH_ONE_WAVE
        if hpk % 2 and T >= 5:  # no hpc = 2 tier: hpc = 1 up to one wave
            cap1 = _HPC1_MAX_WAVES
    elif slow_philox and not ff and not whole_device:
        cap2 = _HPC2_MAX_WAVES_PHILOX_PART[0 if T == 1 else 1 if T <= 4 else 2]
    elif slow_philox and T <= 4:
        cap2 = (
            _HPC2_MAX_WAVES_PHILOX_FF[0 if T == 1 else 1]
            if ff
            else _HPC2_MAX_WAVES_PHILOX
        )
    elif T >= 2:
        cap2 = _HPC2_MAX_WAVES_MTP_SR if sr else _HPC2_MAX_WAVES_MTP
    else:  # T = 1
        cap2 = _HPC2_MAX_WAVES_T1_FF if ff else _HPC2_MAX_WAVES_T1
    if batch * n_key_heads * hpk <= cap1 * slots:
        return 1
    if hpk % 2 == 0 and batch * n_key_heads * (hpk // 2) <= cap2 * slots:
        return 2
    return hpk


# Dispatch rule for a persistent launch: (rounding mode, T) -> the most (key head, row) items
# per CTA slot (H x B / (4 x num_sms)) at which the launch runs the non-persistent kernel
# instead (``grid_override`` always runs the persistent kernel), per ``flush_hint`` (the host
# cannot see the flush rows: hist_len is a device tensor). Only Philox at T = 6-8, where the
# persistent kernel's longer flush items leave its slots unbalanced at few items per slot.
_PZ_PLAIN_MAX_ITEMS = {
    "some": {("philox", 6): 10, ("philox", 7): 10, ("philox", 8): 10},
    "few": {("philox", 6): 5, ("philox", 7): 5, ("philox", 8): 5},
}
# flush_hint -> the row of _PZ_PLAIN_MAX_ITEMS it uses
_PZ_FLUSH_HINTS = {None: "some", "some": "some", "none": "few", "flush_first": "few"}


def _pz_runs_plain(
    T: int, sr: str, n_items: int, n_sms: int, flush_hint: Optional[str] = None
) -> bool:
    cap = _PZ_PLAIN_MAX_ITEMS[_PZ_FLUSH_HINTS[flush_hint]].get((sr, T), 0)
    return n_items <= cap * _CTA_SLOTS_PER_SM * n_sms


def _pz_auto(
    T: int,
    hpk: int,
    heads_per_cta: Optional[int],
    n_items: int,
    num_sms: Optional[int],
    grid_override: Optional[int],
    target,
    flush_hint: Optional[str] = None,
) -> bool:
    """persistent=None: the persistent kernel where it measured faster (see _PZ_AUTO_T and
    _PZ_AUTO_MIN_ITEMS)."""
    tab = _PZ_AUTO_MIN_ITEMS["none" if flush_hint == "none" else "some"]
    min_items = float(
        tab.get(T, math.inf) if isinstance(tab, dict) else tab
    )  # by T or flat
    return (
        (target.major, target.minor) in _PZ_CC
        and T in _PZ_AUTO_T
        and hpk == HPK
        and heads_per_cta in (None, hpk)
        and grid_override is None
        and num_sms is not None
        and int(num_sms) < target.num_sms
        and n_items >= min_items * _CTA_SLOTS_PER_SM * int(num_sms)
    )


# Claim counters for launches captured into CUDA graphs: per device, chunks of int32[2] slots
# zeroed eagerly (topped up by eager persistent launches), one slot per captured launch, never
# reused (each graph replay's last CTA re-zeroes its slot).
_PZ_ARENA_SLOTS = 4096
_PZ_ARENA: dict = {}  # device index -> {"chunks": [int32[slots, 2], ...], "next": free slot}
_PZ_CTR: dict = {}  # eager launches' claim counters: (device index, stream) -> int32[2]


def persistent_counter_reset() -> None:
    """Zero every cached persistent claim counter, the captured launches' ones included
    (recovery hook: a launch that did not run to completion, e.g. one aborted by an error, leaves
    its counter non-zero; a caller-owned ``persistent_counter`` is the caller's to zero)."""
    for c in _PZ_CTR.values():
        c.zero_()
    for a in _PZ_ARENA.values():
        for c in a["chunks"]:
            c.zero_()


def _pz_arena_topup(device: torch.device) -> None:
    """Eager: make sure the arena of captured launches' claim counters has free slots (a capture
    may pick the persistent kernel where the eager calls before it did not: small eager uGPU
    calls run non-persistent launches back to back on the whole GPU)."""
    a = _PZ_ARENA.get(device.index)
    if _PZ_ARENA_SLOTS and (
        a is None or a["chunks"][-1].shape[0] - a["next"] < _PZ_ARENA_SLOTS // 4
    ):
        chunk = torch.zeros(_PZ_ARENA_SLOTS, 2, dtype=torch.int32, device=device)
        if a is None:
            _PZ_ARENA[device.index] = {"chunks": [chunk], "next": 0}
        else:
            a["chunks"].append(chunk)
            a["next"] = 0


def _pz_counter(device: torch.device) -> torch.Tensor:
    """The persistent launch's claim counter (int32[2], zero between launches: the launch's last
    CTA resets it). Eager: one per (device, stream), so concurrent launches on two streams never
    share one. Under graph capture: a slot of its own from the eagerly zeroed arena (graphs captured
    with torch.cuda.graph's defaults share one capture stream and may be replayed concurrently),
    or, if the arena has no free slot, a new zeroed counter in the graph (a memset node, ~1 us per
    replay)."""
    a = _PZ_ARENA.get(device.index)
    if torch.cuda.is_current_stream_capturing():
        if a is not None and a["next"] < a["chunks"][-1].shape[0]:
            c = a["chunks"][-1][a["next"]]
            a["next"] += 1
            return c
        return torch.zeros(2, dtype=torch.int32, device=device)
    _pz_arena_topup(device)
    key = (device.index, torch._C._cuda_getCurrentRawStream(device.index))
    c = _PZ_CTR.get(key)
    if c is None:
        c = torch.zeros(2, dtype=torch.int32, device=device)
        _PZ_CTR[key] = c
    return c


def flush_first_row_order(hist_len: torch.Tensor, flush_min: int) -> torch.Tensor:
    """Permutation of batch rows with the flushing rows (hist_len >= flush_min) first."""
    return torch.argsort((hist_len < flush_min).to(torch.int32), stable=True).to(
        torch.int32
    )


def gated_delta_rule_mtp_ucache_flush_umma_rubin(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    k_cache: Optional[torch.Tensor] = None,
    u_cache: Optional[torch.Tensor] = None,
    g_cache: Optional[torch.Tensor] = None,
    hist_len: Optional[torch.Tensor] = None,
    cache_base: Optional[torch.Tensor] = None,
    flush_min: Optional[int] = None,
    restart_hist_on_flush: bool = True,
    row_order: Optional[torch.Tensor] = None,
    heads_per_cta: Optional[int] = None,
    stochastic_rounding: Optional[str] = None,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
    persistent: Optional[bool] = None,
    num_sms: Optional[int] = None,
    grid_override: Optional[int] = None,
    persistent_counter: Optional[torch.Tensor] = None,
    flush_hint: Optional[str] = None,
    strided_inputs: bool = False,
    pdl: bool = False,
    ring_indices: Optional[torch.Tensor] = None,
    fused_conv: Optional["GdnFusedConv"] = None,
    fused_norm: Optional["GdnFusedNorm"] = None,
) -> torch.Tensor:
    """Rubin (sm_107a) build of ``gated_delta_rule_mtp_ucache_flush_umma`` (the UMMA kernel
    behind ``gated_delta_rule_mtp_ucache_flush`` for the fp16-state arm), 1 <= T <= 8: same
    tensors, contract and results, plus the persistent kernel (``persistent``,
    ``num_sms``, ``grid_override``, ``persistent_counter``, ``flush_hint``). Runs on sm_107a
    and sm_100a (there without the persistent kernel); any other arch raises
    ``NotImplementedError``.

    uGPU localization (two-die GPUs: Rubin, and B200 on drivers that expose its locality
    domains): with the state pool and rings from ``ugpu_ucache_buffers``
    (5-D: ``[pool, dies, heads / dies, ...]``, part t in die t's memory) the call runs part t's
    key heads on die t's SMs, one launch per die on green-context streams forked from and
    joined back to the current stream (``num_sms`` = each die's SMs, so the automatic
    persistent choice applies; a ``persistent_counter`` is then int32 [dies, 2]). Bitwise the
    same results as the usual 4-D buffers. Small eager calls (B x HV < 8192) and devices
    without that many locality domains run the per-die launches back to back on the current
    stream.

    ``heads_per_cta`` (a divisor of HV / H; None = by batch size, on sm_107a also by T,
    ``flush_hint``, the rounding mode and whether ``num_sms`` restricts the launch, see
    ``auto_heads_per_cta``) sets how many of a key head's 4 value heads one CTA runs.

    ``stochastic_rounding`` rounds the fp16 state written by flush rows stochastically
    (``cvt.rs.f16x2``, sm_100a / sm_103a / sm_107a); outputs, rings and verify rows are unchanged:

    * None (default) or "none": off, round to nearest; ``rand_seed`` is not read.
    * "philox": FlashInfer Mamba's scheme and defaults -- Philox4x32 with ``philox_rounds``
      rounds, key = ``rand_seed``, counter = flat offset of the 8-element chunk in the state
      pool, one draw per chunk feeding 4 pairs. Reproducible per seed.
    * "lcg": an LCG dither -- one LCG step per pair, bits s ^ (s >> 16)
      -- started from a murmur3 hash of the (state row, half) id and the seed. Cheapest; each
      element is unbiased and flushes are independent, but dithers within one state row are
      not jointly independent (see ``_rmw_setup``). With ``rand_seed`` it is reproducible per
      seed; without, the seed is %clock (not reproducible).

    ``persistent`` (None = automatic, the default; True / False force it; sm_107a, all 4 value
    heads of a key head per CTA, every T and rounding mode): one launch of G = min(H x B,
    4 x ``num_sms``) CTAs (``num_sms``: the SMs the launch's stream can use, default the
    device's; ``grid_override`` forces G, for tests) that loop over the (key head, row) items
    -- verify, flush and padded rows in any mix and order -- claiming them from an int32[2]
    counter that the launch's last CTA re-zeroes: by default one cached per (device, current
    stream) for eager launches and one of its own per captured launch under CUDA graph capture
    (from an eagerly zeroed pool, so separately captured graphs can be replayed concurrently),
    or the caller's ``persistent_counter`` (zeroed int32[2] CUDA tensor; give each concurrently
    running launch its own). ``persistent_counter_reset`` re-zeroes the cached counters after
    an aborted launch. It measured faster only on die-localized launches (Rubin: each die's
    rows with their buffers in that die's memory, ``num_sms`` = the die's 106 SMs) with enough
    items per CTA slot, and not at T = 1. Automatic (None): the
    persistent kernel on sm_107a for T = 2-8 with all value heads per CTA when ``num_sms`` is
    given and below the device's SM count (a partial-device launch, e.g. one die's green
    context), ``grid_override`` is unset and there are at least 8 (key head, row) items per CTA slot
    (H x B >= 32 x ``num_sms``: B >= 424 per 106-SM die for the die's 8 key heads of H = 16), or
    with ``flush_hint`` "none" 2.4-5.5 items by T (``_PZ_AUTO_MIN_ITEMS``: B >= 128-292 per die);
    the non-persistent kernel otherwise.

    ``flush_hint`` (performance only, never changes results: both kernels are bit-identical on
    every row mix) tells the host which flush rows the launch has, since it cannot read
    ``hist_len`` (a device tensor) without a sync. "none" lowers the automatic choice's
    minimum from 8 items per CTA slot to 2.4-5.5 by T. On sm_107a the hint also moves the heads per
    CTA of non-persistent launches (``auto_heads_per_cta``): "none" gives all value heads per
    CTA from 0.8 waves of CTA slots (B = 22 per die at H = 16), or from one wave (B = 27) at
    T = 1 and on whole-device launches at T = 2-4; "flush_first" on a partial-device launch
    (``num_sms`` below the device's SMs, e.g. a uGPU per-die launch) from 2.4 waves at T = 1
    (B = 64 per die, not 72) and, with Philox above 6 rounds, from 2.75 / 2.5 / 1.75 waves at
    T = 1 / 2-4 / >= 5 (B = 73 / 67 / 47 per die, not 213 / 213 / 56); a whole-device launch
    treats "flush_first" as None there. ``gated_delta_rule_mtp_ucache_flush`` passes no hint
    (None) and no row order.
    With Philox rounding at T = 6-8 and
    few items per CTA slot a persistent launch (chosen or forced, without ``grid_override``) runs the non-persistent
    kernel instead (measured faster there); how few depends on the hint:

    * None (default) or "some": flush rows may be present, in any order: H x B <= 40 x
      ``num_sms`` (B <= 265 per 106-SM die at H = 16).
    * "none": no row of this launch flushes (``hist_len < flush_min`` on every row), or
    * "flush_first": the flush rows come first in the launch's row order (physically, or via
      ``row_order`` = ``flush_first_row_order``): H x B <= 20 x ``num_sms`` (B <= 132 per die).

    A wrong hint only costs time. The hint is a host value: a captured CUDA graph keeps the
    kernel chosen at capture, so capture with the hint of the regime the graph replays in (e.g.
    a no-flush graph and a flush graph).

    ``rand_seed`` is a single-element int64 CUDA tensor read on device, so a captured CUDA
    graph picks up updates; advance it every step (and use a different seed per layer), or
    repeated flushes of a slot reuse the same random bits.

    Local (SGLang) options, both off by default, bitwise the same results:

    * ``strided_inputs``: read q / k / v and a / b through their strides instead of contiguous
      copies when they are views the kernel can address (``strided_inputs_layout``: q and k with one
      per-token stride, v with one, each a multiple of 128 elements and 16-B aligned, heads
      dense; a and b bf16 with the same strides and dense heads), e.g. the packed post-conv
      ``[q | k | v]`` rows and the projection's ``[b | a]`` columns; otherwise the copies.
      Non-persistent 4-D launches only.
    * ``pdl``: a programmatic dependent launch (see ``GdnReplayMtpUmmaRubin.__init__``).
      Non-persistent 4-D launches only.
    * ``ring_indices`` (int32 [B]): the rings' row of each batch row, when the k / u / g rings
      have rows of their own (``[ring_pool, ...]``, e.g. one per live request) instead of one per
      state slot; the state is still read and written at ``initial_state_indices``. None (the
      default): the rings are indexed by the state slot, as above. Padded rows (negative state
      index) never read theirs. Non-persistent 4-D launches only, and not the T >= 5 builds with
      4 value heads per CTA (``early_split``).

    Same tensors, ring/flush contract and host-side cursor commit as FlashInfer's kernel
    (see the module docstring).  ``row_order`` (int32 [B], a permutation of rows, e.g.
    ``flush_first_row_order``) only changes which CTA runs which request.
    """
    assert softplus_beta == 1.0 and softplus_threshold == 20.0
    assert use_qk_l2norm_in_kernel
    assert q is not None and k is not None and v is not None and b is not None
    h0 = initial_state_source
    assert h0 is not None and h0.dtype == torch.float16, "fp16 state pool required"
    B, T, H, K_ = q.shape
    HV, V_ = v.shape[2], v.shape[3]
    assert K_ == K_DIM and V_ == V_DIM, f"K == V == {K_DIM} required"
    assert HV % H == 0 and 1 <= HV // H <= HPK, (
        f"1 to {HPK} value heads per key head required, got HV={HV}, H={H}"
    )
    assert 1 <= T <= 8, f"T={T} unsupported (1 <= T <= 8)"
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert (
        k_cache is not None
        and u_cache is not None
        and g_cache is not None
        and hist_len is not None
    )
    # bf16 k ring, or fp16 (the k_f16 build: f16 mma.sync on the ring keys and the fp16 X̂
    # rows). The u ring is only read and appended through dtype-aware conversions.
    assert k_cache.dtype in (torch.bfloat16, torch.float16), (
        "bf16 or fp16 k ring required"
    )
    assert u_cache.dtype in (torch.bfloat16, torch.float16), (
        "bf16 or fp16 u ring required"
    )
    assert g_cache.dtype == torch.float32
    pool = h0.shape[0]
    # uGPU buffers (ugpu_ucache_buffers): [pool, parts, heads / parts, ...], one part per die
    ugpu = h0.dim() == 5
    n_parts = h0.shape[1] if ugpu else 1
    if ugpu:
        if not _ugpu_buffers_ok(h0, k_cache, u_cache, g_cache, n_parts, H, HV):
            raise ValueError(
                "5-D state pool: uGPU buffers [pool, parts, HV / parts, V, K] / k ring "
                "[pool, parts, H / parts, 32, K] / u ring [pool, parts, HV / parts, 32, V] / "
                "g ring [pool, parts, HV / parts, 32] with dense parts required (see "
                f"ugpu_ucache_buffers); got state {tuple(h0.shape)}, k ring "
                f"{tuple(k_cache.shape)}, u ring {tuple(u_cache.shape)}, g ring "
                f"{tuple(g_cache.shape)} for H={H}, HV={HV}"
            )
        if n_parts == 1:
            h0, k_cache, u_cache, g_cache = (
                x.select(1, 0) for x in (h0, k_cache, u_cache, g_cache)
            )
    else:
        ring_pool = pool if ring_indices is None else k_cache.shape[0]
        assert tuple(k_cache.shape) == (ring_pool, H, RING_SLOTS, K_DIM)
        assert tuple(u_cache.shape) == (ring_pool, HV, RING_SLOTS, V_DIM)
        assert tuple(g_cache.shape) == (ring_pool, HV, RING_SLOTS)
    if ring_indices is not None:
        if ugpu:
            raise ValueError("ring_indices: 4-D state and rings only (no uGPU buffers)")
        if persistent:
            raise ValueError("ring_indices: the non-persistent kernel only")
        assert (
            ring_indices.dtype == torch.int32
            and tuple(ring_indices.shape) == (B,)
            and ring_indices.is_contiguous()
            and ring_indices.device == q.device
        ), "ring_indices: a contiguous int32 [B] tensor on the inputs' device"
    if scale is None:
        scale = 1.0 / math.sqrt(K_)
    if flush_min is None:
        flush_min = W_RING - T + 1
    assert 1 <= flush_min <= W_RING - T + 1
    if initial_state_indices is None:
        initial_state_indices = torch.arange(B, dtype=torch.int32, device=q.device)
    if cache_base is None:
        assert not restart_hist_on_flush, (
            "restart_hist_on_flush=True needs a caller-owned cache_base"
        )
        cache_base = torch.zeros_like(hist_len)
    io = (
        strided_inputs_layout(q, k, v, a, b) if strided_inputs and not ugpu else None
    )
    if io is None:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        a, b = a.contiguous().to(torch.bfloat16), b.contiguous().to(torch.bfloat16)
    if pdl and ugpu:
        raise ValueError("pdl: 4-D state and rings only (no uGPU buffers)")
    if output is None:
        output = torch.empty(B, T, HV, V_, dtype=torch.bfloat16, device=q.device)
    elif n_parts > 1:
        assert output.is_contiguous(), "uGPU buffers: a contiguous output required"
    use_ro = row_order is not None
    if not use_ro:
        key = (B, str(q.device))
        row_order = _ROW_ID.get(key)
        if row_order is None:
            row_order = torch.arange(B, dtype=torch.int32, device=q.device)
            _ROW_ID[key] = row_order
    target = gdn_device_target(q.device)
    if (target.major, target.minor) not in _SUPPORTED_CC:
        raise NotImplementedError(
            "gated_delta_rule_mtp_ucache_flush_umma_rubin: validated on sm_107a and sm_100a "
            f"only (got {target.arch}); use gated_delta_rule_mtp_ucache_flush_umma"
        )
    hpk = HV // H
    if heads_per_cta is not None and not (
        1 <= int(heads_per_cta) <= hpk and hpk % int(heads_per_cta) == 0
    ):
        raise ValueError(
            f"heads_per_cta={heads_per_cta!r}: must divide the {hpk} value heads per key head"
        )
    mode = "none" if stochastic_rounding is None else stochastic_rounding
    if mode not in ("none", "philox", "lcg"):
        raise ValueError(
            f"stochastic_rounding={stochastic_rounding!r}: use None, 'none', 'philox' or 'lcg'"
        )
    if mode == "philox" and rand_seed is None:
        raise ValueError("stochastic_rounding='philox' needs rand_seed")
    if mode != "none" and (target.major, target.minor) not in _CVT_RS_CC:
        raise NotImplementedError(
            "stochastic rounding needs cvt.rs (sm_100a / sm_103a / sm_107a)"
        )
    if mode == "philox" and philox_rounds <= 0:
        raise ValueError(f"philox_rounds must be > 0, got {philox_rounds}")
    sr = {
        "none": "",
        "philox": "philox",
        "lcg": "lcg" if rand_seed is not None else "lcg_clock",
    }[mode]
    sr_rounds = int(philox_rounds) if sr == "philox" else 0
    if rand_seed is not None and sr in ("philox", "lcg"):
        if not (
            isinstance(rand_seed, torch.Tensor)
            and rand_seed.numel() == 1
            and rand_seed.dtype == torch.int64
            and rand_seed.is_cuda
        ):
            raise ValueError("rand_seed must be a single-element int64 CUDA tensor")
        rand_seed = rand_seed.reshape(1)
    else:
        key = ("seed", str(q.device))
        rand_seed = _ROW_ID.get(key)
        if rand_seed is None:
            rand_seed = torch.zeros(
                1, dtype=torch.int64, device=q.device
            )  # unread placeholder
            _ROW_ID[key] = rand_seed
    if flush_hint is not None and flush_hint not in ("some", "none", "flush_first"):
        raise ValueError(
            f"flush_hint must be None, 'some', 'none' or 'flush_first' (got {flush_hint!r})"
        )
    if n_parts > 1 and persistent_counter is not None:
        if not (
            isinstance(persistent_counter, torch.Tensor)
            and persistent_counter.dtype == torch.int32
            and tuple(persistent_counter.shape) == (n_parts, 2)
            and persistent_counter.is_contiguous()
        ):
            raise ValueError(
                f"uGPU buffers: persistent_counter must be a contiguous int32 [{n_parts}, 2] "
                "tensor (one zeroed counter per die)"
            )
    common = dict(
        T=T,
        u_bf16=u_cache.dtype == torch.bfloat16,
        k_f16=k_cache.dtype == torch.float16,
        use_ro=use_ro,
        heads_per_cta=heads_per_cta,
        sr=sr,
        sr_rounds=sr_rounds,
        persistent=persistent,
        grid_override=grid_override,
        flush_hint=flush_hint,
        target=target,
        scale=float(scale),
        flush_min=int(flush_min),
        h_io=H,
        hv_io=HV,
    )
    if io is not None:
        common.update(io)
    if pdl:
        common["pdl"] = True
    if fused_conv is not None:
        # SGLang: q / k / v are the outputs of the conv this launch computes (see GdnFusedConv)
        reason = (
            None
            if io is not None and pdl and not ugpu and heads_per_cta is None
            else "needs strided q / k / v, pdl, 4-D state and the automatic heads per CTA"
        )
        reason = reason or fused_conv_unsupported_reason(
            fused_conv, num_k_heads=H, num_v_heads=HV
        )
        reason = reason or _fused_conv_qkv_reason(q, k, v, fused_conv.x.shape[2])
        if reason is not None:
            raise ValueError(f"fused_conv: {reason}")
        common["fused_conv"] = dict(
            x_row=fused_conv.x.stride(1),
            addrs=(
                fused_conv.x.data_ptr(),
                fused_conv.weight.data_ptr(),
                fused_conv.conv_state.data_ptr(),
                fused_conv.window.data_ptr(),
                fused_conv.window_indices.data_ptr(),
            ),
        )
    if fused_norm is not None:
        # SGLang: output = the gated RMSNorm of the verify output (see GdnFusedNorm)
        reason = (
            None
            if not ugpu and not persistent and stochastic_rounding in (None, "none")
            else "needs 4-D state, the non-persistent kernel and round to nearest"
        )
        reason = reason or fused_norm_unsupported_reason(fused_norm, batch=B, T=T, HV=HV)
        if reason is not None:
            raise ValueError(f"fused_norm: {reason}")
        common["fused_norm"] = (
            fused_norm.z.data_ptr(),
            fused_norm.z.stride(1) if fused_norm.z.dim() == 4 else fused_norm.z.stride(0),
            fused_norm.weight.data_ptr(),
            float(fused_norm.eps),
        )
    tensors = dict(
        state_idx=initial_state_indices,
        hist_len=hist_len,
        cache_base=cache_base,
        row_order=row_order,
        rand_seed=rand_seed,
    )
    if ring_indices is not None:
        tensors["ring_idx"] = ring_indices
    a32, dt32 = _cached_f32(A_log), _cached_f32(dt_bias)
    if n_parts == 1:
        _launch(
            q,
            k,
            v,
            a,
            b,
            a32,
            dt32,
            h0,
            k_cache,
            u_cache,
            g_cache,
            output,
            0,
            num_sms,
            persistent_counter,
            **tensors,
            **common,
        )
    else:
        _launch_ugpu(
            n_parts,
            q,
            k,
            v,
            a,
            b,
            a32,
            dt32,
            h0,
            k_cache,
            u_cache,
            g_cache,
            output,
            num_sms,
            persistent_counter,
            tensors,
            common,
        )
    if restart_hist_on_flush:
        flushed = hist_len >= flush_min
        cache_base.copy_((cache_base + hist_len * flushed) & RING_MASK)
        hist_len.masked_fill_(flushed, 0)
    return output


class GdnFusedConv(NamedTuple):
    """SGLang (SGLANG_ENABLE_GDN_UCACHE_FUSED_CONV): the target-verify conv the flush launch runs
    in its prologue (the kernel's ``fused_conv`` build). q / k / v are views of the conv output
    rows [B, T, D] (D = 2 H K + HV V channels, q | k | v), which the launch writes; the inputs:

    * ``x``: [B, T, D] projection rows before the conv (bf16, unit channel stride);
    * ``weight``: [D, 4] conv weights (bf16, contiguous), no bias;
    * ``conv_state``: [slots, D, 3] conv state (bf16, dense), the slots of
      ``initial_state_indices``; overwritten with the last 3 inputs, as the separate conv does;
    * ``window``: [rows, T, D, 3] step windows, the deduplicated layout (a [rows, D, T + 2]
      buffer: step t's window is columns t .. t + 2);
    * ``window_indices``: int32 [B], each row's window row."""

    x: torch.Tensor
    weight: torch.Tensor
    conv_state: torch.Tensor
    window: torch.Tensor
    window_indices: torch.Tensor


def fused_conv_unsupported_reason(
    fc: GdnFusedConv, *, num_k_heads: int, num_v_heads: int
) -> Optional[str]:
    """None when a strided, dependent, non-persistent launch with the automatic heads per CTA
    can run ``fc`` (B and T from ``fc.x``) in its prologue: one CTA per key head and row (the
    heads per CTA chosen at this B is every value head of a key head), T 4 only, and every buffer
    in the layout the prologue addresses (16-byte aligned rows)."""
    x, w, cs, win, widx = fc
    if x.dim() != 3:
        return f"x of {x.dim()} dims"
    B, T, D = x.shape
    H, HV = num_k_heads, num_v_heads
    hpk = HV // H
    bf = torch.bfloat16
    if T != 4 or D != 2 * H * K_DIM + HV * V_DIM or HV % H:  # build asserts pro_reorder, not split
        return f"T {T}, D {D}, heads {H} / {HV}"
    target = gdn_device_target(x.device)
    if (target.major, target.minor) not in _HPC_TUNED_CC:
        return f"sm_{target.major}{target.minor} (validated on sm_107a)"
    hpc = auto_heads_per_cta(B, H, target.num_sms, hpk, T, None, "", 0, whole_device=True)
    if hpc != hpk:
        return f"{hpc} of {hpk} value heads per CTA at batch {B}"
    if x.dtype != bf or x.stride(2) != 1 or x.stride(0) != T * x.stride(1):
        return f"x {x.dtype} {tuple(x.shape)} {x.stride()}"
    if x.stride(1) % 8 or x.data_ptr() % 16:
        return f"x rows {x.stride()} at {x.data_ptr() % 16} mod 16"
    if w.dtype != bf or tuple(w.shape) != (D, 4) or w.stride() != (4, 1) or w.data_ptr() % 16:
        return f"weight {w.dtype} {tuple(w.shape)} {w.stride()}"
    if (
        cs.dtype != bf
        or cs.dim() != 3
        or tuple(cs.shape[1:]) != (D, 3)
        or cs.stride() != (3 * D, 3, 1)
        or cs.data_ptr() % 16
    ):
        return f"conv state {cs.dtype} {tuple(cs.shape)} {cs.stride()}"
    if (
        win.dtype != bf
        or win.dim() != 4
        or tuple(win.shape[1:]) != (T, D, 3)
        or win.stride() != ((T + 2) * D, 1, T + 2, 1)
        or win.data_ptr() % 16
    ):
        return f"window {win.dtype} {tuple(win.shape)} {win.stride()} (needs the deduplicated rows)"
    if widx.dtype != torch.int32 or widx.dim() != 1 or widx.shape[0] < B or widx.stride(0) != 1:
        return f"window indices {widx.dtype} {tuple(widx.shape)}"
    return None


class GdnFusedNorm(NamedTuple):
    """SGLang (SGLANG_ENABLE_GDN_UCACHE_FUSED_NORM): the gated RMSNorm the flush applies to its
    output (the kernel's ``fused_norm`` build): out = rmsnorm(y) * weight * sigmoid(z) per (token,
    value head), norm before gate, no bias, eps ``eps``. ``z``: [B * T, HV, V] or [B, T, HV, V]
    bf16 rows (unit element stride, V between heads, any token stride); ``weight``: [V] bf16."""

    z: torch.Tensor
    weight: torch.Tensor
    eps: float


def fused_norm_unsupported_reason(fn: GdnFusedNorm, *, batch: int, T: int, HV: int) -> Optional[str]:
    """None when the epilogue can read ``fn`` for a [batch, T, HV, V] output."""
    z, w = fn.z, fn.weight
    zr = z.reshape(batch * T, HV, V_DIM) if z.dim() == 4 else z
    if z.dtype != torch.bfloat16 or tuple(zr.shape) != (batch * T, HV, V_DIM):
        return f"z {z.dtype} {tuple(z.shape)}"
    if zr.stride(2) != 1 or zr.stride(1) != V_DIM or (z.dim() == 4 and z.stride(1) != zr.stride(0)):
        return f"z strides {z.stride()}"
    if w.dtype != torch.bfloat16 or tuple(w.shape) != (V_DIM,) or w.stride(0) != 1:
        return f"weight {w.dtype} {tuple(w.shape)}"
    return None


def _fused_conv_qkv_reason(q, k, v, D: int) -> Optional[str]:
    """None when q / k / v are the q | k | v column views of one dense [B, T, D] buffer."""
    B, T, H, K_ = q.shape
    if q.stride(1) != D or q.stride(0) != T * D or q.data_ptr() % 16:
        return f"q {tuple(q.shape)} {q.stride()} is not a dense [B, T, D] row view"
    if k.data_ptr() != q.data_ptr() + H * K_ * 2 or k.stride()[:2] != q.stride()[:2]:
        return "k is not the k columns of q's rows"
    if v.data_ptr() != q.data_ptr() + 2 * H * K_ * 2 or v.stride()[:2] != q.stride()[:2]:
        return "v is not the v columns of q's rows"
    return None


def strided_inputs_layout(q, k, v, a, b) -> Optional[dict]:
    """The ``_launch`` layout options under which the kernel reads these views as they are
    (``strided_inputs``), or None if it cannot: q / k [B, T, H, K] and v [B, T, HV, V] with dense
    heads, one per-token stride each (q's = k's) that is a multiple of 128 elements, a batch
    stride and an address 8-element (16-B) aligned, as the fake tensors assume; a / b
    [B, T, HV] bf16 with equal strides and dense heads (their per-token strides are runtime
    values)."""

    def rows_ok(x, dim):
        st = x.stride()
        return (
            st[3] == 1
            and st[2] == dim
            and st[1] % dim == 0
            and st[1] >= x.shape[2] * dim
            and st[0] % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    if not (
        rows_ok(q, K_DIM)
        and rows_ok(k, K_DIM)
        and rows_ok(v, V_DIM)
        and q.stride() == k.stride()
    ):
        return None
    if not (
        a.dtype == b.dtype == torch.bfloat16
        and a.stride() == b.stride()
        and a.stride(2) == 1
        and a.stride(1) >= a.shape[2]
    ):
        return None
    return dict(qk_io=q.stride(1) // K_DIM, v_io=v.stride(1) // V_DIM, ab_dyn=True)


def _launch(
    q,
    k,
    v,
    a,
    b,
    a32,
    dt32,
    h0,
    k_cache,
    u_cache,
    g_cache,
    output,
    hv_base,
    num_sms,
    persistent_counter,
    *,
    state_idx,
    hist_len,
    cache_base,
    row_order,
    rand_seed,
    ring_idx=None,
    T,
    u_bf16,
    use_ro,
    heads_per_cta,
    sr,
    sr_rounds,
    persistent,
    grid_override,
    flush_hint,
    target,
    scale,
    flush_min,
    h_io,
    hv_io,
    parts=1,
    k_f16=False,
    qk_io=None,
    v_io=None,
    ab_dyn=False,
    pdl=False,
    fused_conv=None,
    fused_norm=None,
):
    """One kernel launch over q's key heads on the current stream: all of them, or a uGPU
    head slice (q / k / v / a / b / output: head-slice views of the ``h_io`` / ``hv_io``-head
    tensors; state and rings: the slice's own parts; ``hv_base``: its first value head).
    ``parts`` > 1: all heads of 5-D uGPU state and rings in one launch (non-persistent).
    Picks the persistent or the non-persistent kernel and the heads per CTA."""
    B, H, HV = q.shape[0], q.shape[2], v.shape[2]
    hpk = HV // H
    nsm = int(num_sms) if num_sms is not None else target.num_sms
    if parts > 1:
        assert not persistent and grid_override is None
        persistent = False
    strided = ab_dyn or qk_io is not None or v_io is not None
    if ring_idx is not None:
        if persistent:
            raise ValueError("persistent=True: no ring_indices (the non-persistent kernel only)")
        persistent = False
    if persistent is None:
        persistent = (
            False
            if strided or pdl
            else _pz_auto(
                T, hpk, heads_per_cta, H * B, num_sms, grid_override, target, flush_hint
            )
        )
    if persistent and (strided or pdl):
        raise ValueError(
            "persistent=True: no strided inputs (strided_inputs) or dependent launch (pdl)"
        )
    if heads_per_cta is not None:
        hpc = int(heads_per_cta)
    elif persistent:
        hpc = hpk
    else:  # by batch size, and on _HPC_TUNED_CC by T, flush hint, rounding and launch SMs
        tuned = (target.major, target.minor) in _HPC_TUNED_CC
        hpc = auto_heads_per_cta(
            B,
            H,
            nsm,
            hpk,
            T if tuned else None,
            flush_hint,
            sr,
            sr_rounds,
            whole_device=nsm >= target.num_sms,
        )
    if persistent:
        if (target.major, target.minor) not in _PZ_CC:
            raise ValueError(
                f"persistent=True: enabled on sm_107a only (got {target.arch})"
            )
        if not _persist_ok(T, hpk, hpc, sr):
            raise ValueError(
                f"persistent=True: 4 value heads per key head and per CTA only "
                f"(T={T}, HV/H={hpk}, heads_per_cta={hpc}, "
                f"stochastic_rounding={sr or None!r})"
            )
        if persistent_counter is not None and not (
            isinstance(persistent_counter, torch.Tensor)
            and persistent_counter.dtype == torch.int32
            and persistent_counter.numel() == 2
            and persistent_counter.is_contiguous()
            and persistent_counter.device == q.device
        ):
            raise ValueError(
                "persistent_counter must be a contiguous int32 tensor of 2 elements on "
                "the inputs' device (zeroed before its first launch)"
            )
        if grid_override is None and _pz_runs_plain(T, sr, H * B, nsm, flush_hint):
            persistent = (
                False  # measured dispatch rule: the non-persistent kernel is faster
            )
    if fused_norm is not None and persistent:
        raise ValueError("fused_norm: the non-persistent kernel only")
    if fused_conv is not None and (persistent or hpc != hpk or not pdl):
        raise ValueError(
            f"fused_conv: one CTA per key head and row with a dependent launch only (hpc {hpc} "
            f"of {hpk}, persistent {persistent}, pdl {pdl}); see fused_conv_supported"
        )
    if (
        use_ro
        and not persistent
        and (target.major, target.minor) in _SMALL_B_DISPATCH_CC
        and (flush_hint == "none" or _small_grid(H, hpk, hpc, B, nsm))
    ):
        # the row order is one dependent global round trip ahead of every CTA's cursors and buys
        # nothing on a small grid (all CTAs resident) or without flush rows (equal work per
        # row): rows in batch order, same results (a CTA's row only picks its work)
        use_ro = False
    min_blocks = _MIN_BLOCKS_PER_SM
    if (
        not persistent
        and (target.major, target.minor) in _SMALL_GRID_LB_CC
        and hpk == HPK
        and _small_grid(H, hpk, hpc, B, nsm)
        and sr != "philox"
        and not _use_regsplit(T, hpc, sr)
    ):
        # small grid: a launch bound of _SMALL_GRID_CTAS_PER_SM CTAs per SM (more registers
        # for the latency-bound CTAs, never more CTAs than that per SM anyway); not Philox
        # (no gain)
        min_blocks = _SMALL_GRID_CTAS_PER_SM
    args = (
        q,
        k,
        v,
        a,
        b,
        a32,
        dt32,
        h0,
        state_idx,
        k_cache,
        u_cache,
        g_cache,
        hist_len,
        cache_base,
        row_order,
        output,
        rand_seed,
    )
    fn = _get_kernel(
        q.device,
        T,
        H,
        HV,
        u_bf16,
        use_ro,
        hpc,
        sr,
        sr_rounds,
        bool(persistent),
        h_io,
        hv_io,
        min_blocks=min_blocks,
        parts=parts,
        k_f16=k_f16,
        qk_io=qk_io,
        v_io=v_io,
        ab_dyn=ab_dyn,
        pdl=pdl,
        ring_rows=ring_idx is not None,
        fused_conv=fused_conv is not None,
        fused_norm=fused_norm is not None,
    )
    if persistent:
        ctr = (
            _pz_counter(q.device)
            if persistent_counter is None
            else persistent_counter.reshape(2)
        )
        n_cta = (
            int(grid_override)
            if grid_override is not None
            else min(H * B, _CTA_SLOTS_PER_SM * nsm)
        )
        fn(*args, ctr, scale, flush_min, int(hv_base), max(1, n_cta))
    else:
        # without ring_idx the kernel's ring-row parameter is the state indices, never read
        fn(
            *args,
            state_idx if ring_idx is None else ring_idx,
            scale,
            flush_min,
            int(hv_base),
            *(fused_conv["addrs"] if fused_conv is not None else (0, 0, 0, 0, 0)),
            fused_conv["x_row"] if fused_conv is not None else 0,
            *(fused_norm if fused_norm is not None else (0, 0, 0, 0.0)),
        )


# ============================================================================ uGPU
# uGPU localization (two-die GPUs, Rubin and B200: each die with its own memory and SMs). The
# buffers of ugpu_ucache_buffers keep key heads [t H / 2, (t + 1) H / 2) of every slot (with their
# value heads) in die t's memory, and a call with them runs one launch per die on that die's SMs
# (green-context streams, forked from and joined back to the current stream): every row is
# split the same way, whatever the batch order and slot indices, so no state or ring byte
# crosses the die-to-die link. Each launch is the same kernel on its heads (heads are
# independent; the stochastic-rounding ids use the global head), so the results are bitwise
# equal to one launch over the whole pool.

# Eager calls below this many (batch x value-head) states are host-bound: the per-die fork /
# join costs more host time than the die-local traffic saves, so they run as one fused launch
# (sm_107a) or the per-die launches back to back on the current stream (each over the whole GPU).
# Under CUDA graph capture the per-die streams are used above the fused launch's rows
# (_UGPU_FUSED_MAX_ITEMS).
_UGPU_EAGER_MIN_STATES = 8192

# Launch path of a uGPU call (see _ugpu_path): "dies" = one launch per die on its green-context
# stream (forked / joined), "seq" = the per-die launches back to back on the current stream
# (each over the whole GPU), "fused" = ONE launch over every part's heads on the whole GPU (the
# kernel addresses the 5-D buffers' parts itself), "streams" = the per-die launches forked onto
# two ordinary streams (each over the whole GPU). _UGPU_PATH_OVERRIDE (benchmarks / tests)
# forces one for every call.
_UGPU_PATH_OVERRIDE: Optional[str] = None
# Few rows: one fused launch over both parts while the (key head, row) items per SM of the whole
# GPU, H x B / num_sms, are at most _UGPU_FUSED_MAX_ITEMS[T] (sm_107a, 4 value heads per key
# head; H = 16 on 212 SMs: B <= 12 / 8 / 4 at T = 1-2 / 3-5 / 6-8): there the per-die launches'
# fork / join and second launch cost more than the die-local traffic saves; above, the cross-die
# traffic of the fused launch costs more.
_UGPU_FUSED_CC = ((10, 7),)
_UGPU_FUSED_MAX_ITEMS = {
    1: 0.95,
    2: 0.95,
    3: 0.65,
    4: 0.65,
    5: 0.65,
    6: 0.35,
    7: 0.35,
    8: 0.35,
}


def _ugpu_path(
    B, H, HV, T, num_sms, target, capturing, persistent, grid_override
) -> str:
    """Launch path of a uGPU call (see _UGPU_PATH_OVERRIDE): "fused" for few rows on
    _UGPU_FUSED_CC (and for every small eager call there), else "dies" under graph capture and for
    eager calls of at least _UGPU_EAGER_MIN_STATES states, else "seq"."""
    if _UGPU_PATH_OVERRIDE is not None:
        return _UGPU_PATH_OVERRIDE
    nsm = int(num_sms) if num_sms is not None else target.num_sms
    fused_ok = (
        (target.major, target.minor) in _UGPU_FUSED_CC
        and HV == HPK * H  # the measured shape (4 value heads per key head)
        and not persistent
        and grid_override is None
    )
    n_items, max_items = H * B, _UGPU_FUSED_MAX_ITEMS[T] * nsm
    if fused_ok and n_items <= max_items:
        return "fused"
    if capturing or B * HV >= _UGPU_EAGER_MIN_STATES:
        return "dies"
    return "fused" if fused_ok else "seq"


@functools.cache
def _locality_domains(device_index: int) -> int:
    """Locality domains (dies) of the device; 1 without the CUDA 13.4 locality-domain API."""
    try:
        from flashinfer.green_ctx import get_locality_domain_count
    except ImportError:
        return 1
    return get_locality_domain_count(torch.device("cuda", device_index))


@functools.cache
def _ugpu_streams(device_index: int):
    """One green-context stream per locality domain of the device and its SM count (each
    domain's own SMs plus its share of the rest: 106 per die on Rubin VR200, 74 on B200).
    Cached; the first uGPU call creates them, so make one eager call before CUDA graph
    capture."""
    from flashinfer.green_ctx import split_device_green_ctx_by_locality_domain

    streams, resources = split_device_green_ctx_by_locality_domain(
        torch.device("cuda", device_index)
    )
    return streams, [int(r.sm.smCount) for r in resources]


@functools.cache
def _ugpu_plain_streams(device_index: int, n: int):
    """n ordinary (whole-GPU) streams for the "streams" uGPU path (cached)."""
    return [torch.cuda.Stream(device=device_index) for _ in range(n)]


def _ugpu_buffers_ok(h0, k_cache, u_cache, g_cache, n, H, HV) -> bool:
    """The four buffers are [pool, n, heads / n, ...] with dense parts (any slot stride)."""
    if H % n or HV % n:
        return False
    pool = h0.shape[0]
    for t, heads, inner in (
        (h0, HV, (V_DIM, K_DIM)),
        (k_cache, H, (RING_SLOTS, K_DIM)),
        (u_cache, HV, (RING_SLOTS, V_DIM)),
        (g_cache, HV, (RING_SLOTS,)),
    ):
        shape = (pool, n, heads // n, *inner)
        if tuple(t.shape) != shape:
            return False
        dense = torch.empty(shape[2:], device="meta").stride()
        if tuple(t.stride()[2:]) != tuple(dense):
            return False
    return True


def _launch_ugpu(
    n,
    q,
    k,
    v,
    a,
    b,
    a32,
    dt32,
    h0,
    k_cache,
    u_cache,
    g_cache,
    output,
    num_sms,
    persistent_counter,
    tensors,
    common,
):
    """uGPU buffers: part t (key heads [t H / n, (t + 1) H / n)) on locality domain t's SMs,
    one launch per part forked from and joined back to the current stream. For few rows on
    sm_107a (_ugpu_path), one launch over every part on the whole GPU instead. Without n
    locality domains (older drivers expose one), and for small eager calls elsewhere
    (``_UGPU_EAGER_MIN_STATES``), the launches run back to back on the current stream."""
    B, H, HV = q.shape[0], q.shape[2], v.shape[2]
    hl, hvl = H // n, HV // n

    def part(t, sms):
        _launch(
            q.narrow(2, t * hl, hl),
            k.narrow(2, t * hl, hl),
            v.narrow(2, t * hvl, hvl),
            a.narrow(2, t * hvl, hvl),
            b.narrow(2, t * hvl, hvl),
            a32.narrow(0, t * hvl, hvl),
            dt32.narrow(0, t * hvl, hvl),
            h0.select(1, t),
            k_cache.select(1, t),
            u_cache.select(1, t),
            g_cache.select(1, t),
            output.narrow(2, t * hvl, hvl),
            t * hvl,
            sms,
            None if persistent_counter is None else persistent_counter[t],
            **tensors,
            **common,
        )

    with torch.cuda.device(q.device):  # streams and events of q's device
        if _locality_domains(q.device.index) != n:
            for t in range(n):
                part(t, num_sms)
            return
        # created by the first call (make one eager call before CUDA graph capture)
        streams, sm_counts = _ugpu_streams(q.device.index)
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing and (common["target"].major, common["target"].minor) in _PZ_CC:
            _pz_arena_topup(
                q.device
            )  # for the per-die persistent launches of a later capture
        path = _ugpu_path(
            B,
            H,
            HV,
            common["T"],
            num_sms,
            common["target"],
            capturing,
            common["persistent"],
            common["grid_override"],
        )
        if path == "fused" and (
            common["persistent"] or common["grid_override"] is not None
        ):
            path = "seq"  # the fused launch is non-persistent only
        if path == "seq":
            for t in range(n):
                part(t, num_sms)
            return
        if path == "fused":
            _launch(
                q,
                k,
                v,
                a,
                b,
                a32,
                dt32,
                h0,
                k_cache,
                u_cache,
                g_cache,
                output,
                0,
                num_sms,
                None,
                **tensors,
                **common,
                parts=n,
            )
            return
        main = torch.cuda.current_stream()
        if path == "streams":
            streams, sm_counts = _ugpu_plain_streams(q.device.index, n), [num_sms] * n
        else:
            assert path == "dies", path
        fork = torch.cuda.Event()
        fork.record(main)
        joins = []
        for t in range(n):
            stream = streams[t]
            stream.wait_event(fork)
            with torch.cuda.stream(stream):
                part(t, sm_counts[t])
            done = torch.cuda.Event()
            done.record(stream)
            joins.append(done)
        for done in joins:
            main.wait_event(done)


class UgpuUcacheBuffers(NamedTuple):
    """Die-split state pool and rings (see ``ugpu_ucache_buffers``)."""

    state: torch.Tensor  # fp16 [pool, n, HV / n, V, K]
    k_cache: torch.Tensor  # bf16 [pool, n, H / n, 32, K]
    u_cache: torch.Tensor  # u dtype [pool, n, HV / n, 32, V]
    g_cache: torch.Tensor  # fp32 [pool, n, HV / n, 32]


def ugpu_ucache_buffers(
    pool_size: int,
    num_k_heads: int,
    num_v_heads: int,
    u_dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> UgpuUcacheBuffers:
    r"""Allocate the state pool and the k / u / g rings of
    :func:`gated_delta_rule_mtp_ucache_flush_umma_rubin` for uGPU localization (two-die GPUs).

    Each buffer has a die dimension after the slot: ``[pool, n, heads / n, ...]`` with
    ``n`` = the device's locality domains (2 on Rubin VR200 and B200). Element
    ``[s, t, j, ...]`` is element ``[s, t * heads / n + j, ...]`` of the usual buffer (``heads`` = HV for the state,
    u and g rings, H for the k ring), and part ``t`` lives in locality domain (die) ``t``'s
    memory. Passed to the kernel (``initial_state_source`` = ``state``, and the rings), a call
    runs the key heads of part ``t`` -- with their value heads -- on die ``t``'s SMs, one
    launch per die, so no state or ring byte crosses the die-to-die link, whatever the batch
    order and slot indices. That is for CUDA-graph captured decode (make one eager call before
    capture). Calls with few rows (Rubin, H = 16: B <= 12 at T = 1 / 2, 8 at T = 3-5, 4 at T >= 6) run one
    launch over both dies' heads on the whole GPU instead (latency-bound: the per-die launches'
    fork / join and second launch cost more than the cross-die traffic), and eager calls with
    ``B * HV < 8192`` are host-bound and run that launch too (elsewhere: the per-die launches
    back to back on the current stream). Results are bitwise equal to the usual buffers'.

    The buffers are uninitialized and come from
    :func:`flashinfer.green_ctx.empty_per_locality_domain`, outside PyTorch's caching
    allocator.

    Args:
        pool_size: Number of state slots.
        num_k_heads: Number of key heads (H), a multiple of the locality domain count.
        num_v_heads: Number of value heads (HV).
        u_dtype: u ring dtype (bfloat16 or float16).
        device: CUDA device with at least two locality domains (default: the current device).
    """
    from flashinfer.green_ctx import empty_per_locality_domain

    device = torch.device(device) if device is not None else torch.device("cuda")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    n = _locality_domains(device.index)
    if n < 2:
        raise RuntimeError(
            f"{device} exposes {n} locality domain(s); uGPU buffers need at least two "
            "(e.g. Rubin) and a driver / cuda-python with the locality-domain API"
        )
    if num_k_heads % n or num_v_heads % num_k_heads:
        raise ValueError(
            f"H={num_k_heads} must be a multiple of the {n} locality domains and divide "
            f"HV={num_v_heads}"
        )

    def alloc(heads, inner, dtype):
        parts = empty_per_locality_domain(
            (pool_size, heads // n, *inner), dtype, device
        )
        return parts.transpose(0, 1)

    return UgpuUcacheBuffers(
        alloc(num_v_heads, (V_DIM, K_DIM), torch.float16),
        alloc(num_k_heads, (RING_SLOTS, K_DIM), torch.bfloat16),
        alloc(num_v_heads, (RING_SLOTS, V_DIM), u_dtype),
        alloc(num_v_heads, (RING_SLOTS,), torch.float32),
    )
