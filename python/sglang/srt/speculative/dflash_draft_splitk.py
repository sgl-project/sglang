# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Default-off split-K GEMMs for the DFlash MLA draft layers (SGLANG_OPT_DFLASH_DRAFT_SPLITK).

At decode-block row counts (<= 8 rows; with SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE also the 9-64-row blocks of batches of
2-8 requests listed in WIDE_TACTICS) the draft layers' BF16 linears (conv kernel projections, fused qkv_a,
q_b, o_proj, gate_up, down) run on the megakernel plugin's cluster split-K projection instead of cuBLAS (+ its
splitKreduce), sglang's fused_a_gemm or the CuTe TGV GEMM. Each cluster rank accumulates one K slice in FP32 and
rank 0 sums the partials in rank order before the BF16 store: an accumulation reorder of the stock GEMMs.

The design (drafter linears on the prepared split-K projection, a tactic per weight shape) is based on earlier
NVIDIA work.
"""

from __future__ import annotations

import logging
from itertools import chain
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

logger = logging.getLogger(__name__)

# One B1 draft block (block size 8); larger batches keep the stock GEMMs except the WIDE_TACTICS row counts.
MAX_SPLITK_ROWS = 8
_CTA_K = 128  # K per split-K tile (plugin splitk.CTA_K); each cluster rank takes K / split tiles.

# (M tile over N, N tile over rows, split K, stages) per draft weight [N, K] at TP16 per-rank shapes, tuned on the target GPU.
# (2112, 7168), (7168, 768), (7168, 896): that design's tactics for the same shapes. (1792, 7168) splits K in
# 7 slices like the stock cuBLAS GEMM it replaces (bitwise to it there); fewer slices were less accurate than stock.
# o_proj stays on split-K: alone the stock TGV GEMM is faster, but the split-K form is faster inside the draft layer.
SPLITK_TACTICS: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {
    (1792, 7168): (128, 8, 7, 5),  # conv kernel projections, gate_up
    (2112, 7168): (64, 8, 4, 10),  # fused qkv_a
    (1152, 1536): (64, 8, 4, 3),  # q_b
    (7168, 768): (64, 8, 1, 6),  # o_proj
    (7168, 896): (64, 8, 1, 7),  # down
}


Tactic = Tuple[int, int, int, int]

# SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE: per weight shape, {draft rows: tactic} for the 9-64-row blocks of batches of 2-8
# requests (8 rows per request at K=7, 6 at K=5). Row counts without an entry keep the stock GEMM. The served <= 8-row
# tactic stretched over more rows is slower than the stock GEMM from 16 rows. An entry is a tactic that was faster than the
# stock GEMM in two separate kernel-only measurements on the target GPU (cold L2), with K = 7168 split in at least 4
# slices like the served <= 8-row qkv_a tactic: 2 slices were faster still but round 3-4x more outputs away from the
# correctly rounded value than the stock GEMM (4 slices: ~2x). o_proj and down have no entry: no tactic beat the stock
# GEMM consistently.
_WIDE_K7168 = {
    12: (64, 16, 4, 10), 16: (64, 16, 4, 10), 18: (64, 24, 4, 9), 24: (64, 24, 4, 9), 30: (64, 32, 4, 8), 32: (64, 32, 4, 8),
    36: (64, 40, 4, 7), 40: (64, 40, 4, 7), 42: (64, 48, 4, 6), 48: (64, 48, 4, 6), 56: (64, 56, 4, 6), 64: (128, 24, 4, 4),
}  # fmt: skip
WIDE_TACTICS: Dict[Tuple[int, int], Dict[int, Tactic]] = {
    # conv kernel projections, gate_up (the stock GEMM is as fast at 30 rows and from 42 rows)
    (1792, 7168): {r: _WIDE_K7168[r] for r in (12, 16, 18, 24, 32, 36, 40)},
    # fused qkv_a (as fast at 64 rows)
    (2112, 7168): {r: t for r, t in _WIDE_K7168.items() if r < 64},
    # q_b (as fast from 18 rows)
    (1152, 1536): {12: (64, 8, 3, 4), 16: (64, 16, 3, 4)},
}  # fmt: skip
# The fused gate_up + SiLU-mul (SGLANG_OPT_DFLASH_DRAFT_SPLITK_ACT): it also saves the act kernel, so it is faster than the
# stock GEMM + act pair at every row count with gate_up's tactic, with one stage fewer at 56 rows: the act epilogue stages its
# BF16 tile in shared memory, and 6 stages exceed the per-CTA shared-memory limit (the plugin's stage check refuses them).
# Stages change only the pipeline depth, not the sum.
WIDE_ACT_TACTICS: Dict[int, Tactic] = {**_WIDE_K7168, 56: (64, 56, 4, 5)}

# SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV_WIDE_ROWS: per K = 7168 weight shape, {draft rows: tactic} with the wide tactic's M tile,
# N tile and split (the same partial sums, reduced in the same rank order: bitwise to it) and more pipeline stages than the
# standard per-CTA shared-memory allocation holds, launched in the plugin's extended shared-memory mode: 11 / 11 / 12
# instead of 10 / 9 / 8, so more of the weight stream is issued before the PDL wait. The stages stay below the 14 K tiles
# of a split, so the shipped trigger still comes after the wait (trigger_follows_wait) and the AR site's early release of
# the conv projection is unchanged. Stage counts from a 1-GPU chain sweep (conv projection -> grouped conv -> qkv_a or the
# fused gate_up + SiLU-mul): 11 / 11 / 12 stages were the fastest bitwise forms; staging all 14 tiles with a fenced trigger
# after the wait was slower at 16 and 24 rows. The same tactic serves the fused act form (its BF16 staging tile still
# fits). At 40 / 48 / 56 / 64 rows (batches of 5-8 requests) a second sweep found 11 / 8 / 9 / 6 stages fastest. A shape
# keeps entries only where it has a wide tactic (the conv kernel projections and gate_up stop at 40 rows, qkv_a at 56).
_OV_WIDE_K7168 = {
    16: (64, 16, 4, 11), 24: (64, 24, 4, 11), 32: (64, 32, 4, 12), 40: (64, 40, 4, 11), 48: (64, 48, 4, 8), 56: (64, 56, 4, 9),
}  # fmt: skip
OV_WIDE_TACTICS: Dict[Tuple[int, int], Dict[int, Tactic]] = {
    shape: {r: t for r, t in _OV_WIDE_K7168.items() if r in WIDE_TACTICS[shape]}
    for shape in ((1792, 7168), (2112, 7168))  # conv kernel projections + gate_up, fused qkv_a
}
OV_WIDE_ACT_TACTICS: Dict[int, Tactic] = {**_OV_WIDE_K7168, 64: (128, 24, 4, 6)}

# SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B: the 96-row draft blocks (12 requests x block 8) of the K=7168 linears on SK3B
# (speculative/dflash_sk3b.py), a streaming cluster split-K: the weight K tiles stream through their own shared-memory ring from kernel entry,
# x loads into its own slots right after the PDL wait, and the plugin TM64's DSMEM reduce-scatter sums the 8 K-slice partials in rank order.
# Policy (N tile, split, x slots, ring stages, owners, flags; flags 32 = the owners' mailboxes in their own shared memory, so no epilogue
# cluster barrier: each rank pushes its partial as soon as its own MMAs end). At 96 rows (N96, split 8): the same K slices, MMA order and
# rank-order FP32 sum as the stock cuBLAS GEMM -> bitwise to it. With _ACT the fused gate_up + SiLU-mul uses
# SK3B_ACT_TACTICS instead (SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_ACT); without that knob the fused projection does not cover these rows and the
# MLP runs this unfused gate_up projection + the stock act.
SK3B_TACTICS: Dict[Tuple[int, int], Dict[int, Tuple[int, ...]]] = {
    (1792, 7168): {96: (96, 8, 4, 7, 8, 32)},  # conv kernel projections, gate_up
    (2112, 7168): {96: (96, 8, 4, 7, 8, 32)},  # fused qkv_a
}

# SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_ACT: the fused gate_up + SiLU-mul at these row counts on SK3B
# with an act epilogue (speculative/dflash_sk3b_act.py): each CTA loads gate rows j and up rows I + j of its 64 features, an owner gets both rows of
# its features, rounds them to BF16 as the unfused SK3B does and applies sglang's act_and_mul sequence -> bitwise to SK3B's gate_up output + the
# stock act, so at 96 rows (the SK3B 96-row policy) bitwise to the stock cuBLAS gate_up + act. It saves the act kernel.
SK3B_ACT_TACTICS: Dict[Tuple[int, int], Dict[int, Tuple[int, ...]]] = {
    (1792, 7168): {96: (96, 8, 4, 7, 8, 32)},  # gate_up
}
# SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_LOW: the 24-88-row draft blocks (3-11 requests x block 8)
# of the K=7168 linears on SK3B too (with _SK3B_ACT also the fused gate_up + SiLU-mul, with _SK3B_CONV the conv epilogue): N64 split 8 at 24-64
# rows, the 96-row policy at 72-88. The same K slices and rank-order FP32 sum as the stock cuBLAS GEMM at these rows -> bitwise to it
# (at 16 rows cuBLAS splits differently, so no 16-row entry). These rows take precedence over
# SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE's rows (not bitwise to cuBLAS); WIDE keeps 12-18. In the batch-8 serving configuration they also take
# precedence over SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV_WIDE_ROWS' rows (OV-wide keeps 16 rows and the rows SK3B does not list). Faster than the
# stock GEMMs at 24-88 rows in the 1-GPU drafter chain rig (served conv flag, with act + packed conv).
_SK3B_LOW = {**{r: (64, 8, 4, 7, 8, 32) for r in (24, 32, 40, 48, 56, 64)}, **{r: (96, 8, 4, 7, 8, 32) for r in (72, 80, 88)}}
SK3B_LOW_TACTICS: Dict[Tuple[int, int], Dict[int, Tuple[int, ...]]] = {
    (1792, 7168): dict(_SK3B_LOW),  # conv kernel projections, gate_up
    (2112, 7168): dict(_SK3B_LOW),  # fused qkv_a
}
SK3B_ACT_LOW_TACTICS: Dict[Tuple[int, int], Dict[int, Tuple[int, ...]]] = {
    (1792, 7168): dict(_SK3B_LOW),  # gate_up
}
# SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_B4: the 104-128-row draft blocks (13-16 requests x block 8) of the K=7168 linears on SK3B
# with split 4 (N64 tiles, 4 x slots, 8 ring stages, 4 owners, DMB; with _SK3B_ACT also the fused gate_up + SiLU-mul, with _SK3B_CONV the conv
# epilogue). At these rows the stock cuBLAS GEMM splits K in the same four 1792-K slices and sums them in rank order in FP32 (an emulation search over
# 91 slice layouts matched only this one), so this form is bitwise to it.
# With act + PF and the conv epilogue: faster than the stock GEMMs at these rows.
_SK3B_B4 = {r: (64, 4, 4, 8, 4, 32) for r in (104, 112, 120, 128)}
SK3B_B4_TACTICS: Dict[Tuple[int, int], Dict[int, Tuple[int, ...]]] = {
    (1792, 7168): dict(_SK3B_B4),  # conv kernel projections, gate_up
    (2112, 7168): dict(_SK3B_B4),  # fused qkv_a
}
SK3B_ACT_B4_TACTICS: Dict[Tuple[int, int], Dict[int, Tuple[int, ...]]] = {
    (1792, 7168): dict(_SK3B_B4),  # gate_up
}
# SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_PF: the SK3B rows of the fused qkv_a projection and of the gate_up projection (unfused and
# act-fused) also prefetch the NEXT GEMM's weight into L2 after their own weight stream (o_proj after qkv_a, down after gate_up: the GEMMs that
# stay on cuBLAS at these rows), flags 512 on the same SK3B policy (dflash_sk3b.SplitKStream3b.prefetch_next). A cache hint: outputs unchanged.
# Faster for each of the two links in the 1-GPU drafter chain rig.
PF_FLAG = 512


# SGLANG_OPT_DFLASH_DRAFT_SPLITK_WCONV: {rows: (tactic, extended-SMEM flag)} of the conv kernel projection [1792, 7168] that run the side-0 conv
# in the epilogue (dflash_splitk_conv.py, prefetch form), exactly the tactics selected for the batch-8 configuration (1-GPU rig).
WCONV_TACTICS: Dict[int, Tuple[Tuple[int, int, int, int], bool]] = {8: ((128, 8, 7, 5), False), 16: ((64, 16, 4, 11), True)}
# SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV: per weight shape, the <= 8-row tactic with one stage per K tile of a split (the served
# tactic's M tile, N tile and split, so the same partial sums in the same order), launched in the plugin's extended
# shared-memory mode: the weight warp issues every tile before the PDL wait. Tactics from 1-GPU chain probes (bitwise to
# the served tactic; faster in a PDL chain with the trigger before the wait).
OV_TACTICS: Dict[Tuple[int, int], Tactic] = {
    (1792, 7168): (128, 8, 7, 8),  # conv kernel projections, gate_up: 8 K tiles per split
    (2112, 7168): (64, 8, 4, 14),  # fused qkv_a: 14 K tiles per split
}
# SplitKGemm's fenced trigger modes: 0 = the weight warp triggers after issuing its stages (before the wait when stages >=
# tiles); 2 = the idle warp 3 waits, fences (fence.acq_rel.gpu) and triggers.
_TRIGGER_SHIPPED, _TRIGGER_AFTER_WAIT = 0, 2


class DraftSplitKProjection:
    """The plugin's split-K projection of one draft weight [N, K] for 1..MAX_SPLITK_ROWS rows, plus the row counts in
    ``wide`` ({rows: tactic}, SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE).

    ``act_fuse`` (SGLANG_OPT_DFLASH_DRAFT_SPLITK_ACT): ``weight`` is the merged gate_up [2I, K] and ``run`` returns
    SiLU(gate) * up [rows, I], bitwise equal to this projection's gate_up output followed by sglang's act_and_mul.

    ``ov_wide`` (SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV_WIDE_ROWS, {rows: tactic}): those wide row counts run on a second
    prepared projection in the extended shared-memory mode. Each entry must keep the wide tactic's M tile, N tile and split at its
    row count (the same partial sums, reduced in the same order) and hold fewer stages than K tiles per split (the release
    stays after the wait); only the stages differ.

    ``ov`` (SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV): the <= MAX_SPLITK_ROWS-row tactic instead of ``tactic``, on another
    prepared projection in the extended shared-memory mode with SplitKGemm trigger mode ``ov_trigger`` (0 or 2). It must keep
    ``tactic``'s M tile, N tile and split (the same partial sums, reduced in the same order); the wide, OV-wide and SK3B
    row counts keep their own kernels.

    ``sk3b`` (SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B*, {rows: SK3B policy}): those row counts on SK3B (dflash_sk3b.py; with ``act_fuse``
    dflash_sk3b_act.py); none of them may also be a split-K, wide or OV-wide row. ``prefetch`` (SK3B_PF): the next GEMM's weight, prefetched
    by the SK3B rows. ``conv`` (SK3B_CONV, conv kernel projections): the SK3B rows also on SK3B with the side-0 grouped conv epilogue."""

    def __init__(
        self, weight: torch.Tensor, tactic: Tactic, act_fuse: bool = False, wide: Optional[Dict[int, Tactic]] = None,
        ov_wide: Optional[Dict[int, Tactic]] = None, sk3b: Optional[Dict[int, Tuple[int, ...]]] = None,
        prefetch: Optional[torch.Tensor] = None, conv: bool = False, wconv: bool = False,
        ov: Optional[Tactic] = None, ov_trigger: int = _TRIGGER_SHIPPED,
    ):  # fmt: skip
        from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.projection import PreparedProjection

        n, k = weight.shape
        if ov is not None and tuple(ov[:3]) != tuple(tactic[:3]):
            raise ValueError(f"OV tactic {ov} changes the served tactic {tactic}'s tiles or split (not bitwise)")
        if ov_trigger not in (_TRIGGER_SHIPPED, _TRIGGER_AFTER_WAIT):
            raise ValueError(f"OV trigger mode {ov_trigger}: 0 or 2")
        policies = {rows: (ov if ov is not None else tactic) for rows in range(1, MAX_SPLITK_ROWS + 1)}
        for rows, wide_tactic in (wide or {}).items():
            if rows <= MAX_SPLITK_ROWS:
                raise ValueError(f"wide tactic at {rows} rows: the <= {MAX_SPLITK_ROWS}-row tactic is {tactic}")
            policies[rows] = wide_tactic
        ov_wide = dict(ov_wide or {})
        for rows, ov_tactic in ov_wide.items():
            served = policies.get(rows) if rows > MAX_SPLITK_ROWS else None
            if served is None or tuple(ov_tactic[:3]) != tuple(served[:3]):
                raise ValueError(
                    f"OV-wide tactic {ov_tactic} at {rows} rows does not keep the wide tactic {served}'s tiles and split"
                )
            if ov_tactic[3] >= k // (_CTA_K * ov_tactic[2]):
                raise ValueError(f"OV-wide tactic {ov_tactic} stages every K tile: its trigger would precede its wait")
            policies[rows] = ov_tactic
        # The kernel splits K evenly over the cluster without a tail check.
        for policy in set(policies.values()):
            if k % (_CTA_K * policy[2]):
                raise ValueError(f"split-K tactic {policy} does not divide K={k}")
        self.n, self.k = (n // 2 if act_fuse else n), k
        self.rows = frozenset(policies)
        self._policies = dict(policies)
        self.ov_wide_rows = frozenset(ov_wide)
        # SplitKGemm trigger mode per row count where it is not the shipped mode 0 (the OV rows with ov_trigger 2).
        self._triggers = {rows: ov_trigger for rows in range(1, MAX_SPLITK_ROWS + 1)} if ov is not None and ov_trigger else {}
        # The draft's neighbours are stock kernels that do not fence their stores for TMA readers, and
        # griddepcontrol is not a proxy-preserving boundary: the projection fences around its TMA reads itself.
        served_policies = {
            rows: t for rows, t in policies.items() if rows not in ov_wide and (ov is None or rows > MAX_SPLITK_ROWS)
        }
        self._prepared = (
            PreparedProjection(weight, served_policies, proxy_fences=True, act_fuse=act_fuse).prepare()
            if served_policies
            else None
        )
        self._ov_wide_prepared = None
        if ov_wide:
            self._ov_wide_prepared = PreparedProjection(
                weight, ov_wide, proxy_fences=True, act_fuse=act_fuse, oversized=True
            ).prepare()
        self._ov_prepared = None
        if ov is not None:
            small_rows = {rows: ov for rows in range(1, MAX_SPLITK_ROWS + 1)}
            self._ov_prepared = PreparedProjection(
                weight, small_rows, proxy_fences=True, act_fuse=act_fuse, fenced_trigger=ov_trigger, oversized=True
            ).prepare()
        # SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B*: these row counts on SK3B instead (trigger mode 2: it triggers after its own wait).
        self._sk3b, self._sk3b_rows = None, frozenset()
        # SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_PF: ``prefetch`` = the next GEMM's weight, prefetched by the SK3B rows (flags PF_FLAG).
        self._prefetch = prefetch if sk3b else None
        if self._prefetch is not None:
            sk3b = {r: tuple(t[:5]) + ((int(t[5]) if len(t) > 5 else 0) | PF_FLAG,) for r, t in sk3b.items()}
        if sk3b:
            for rows in sk3b:
                if rows in policies:
                    raise ValueError(f"SK3B at {rows} rows: the split-K projection already covers it")
            if act_fuse:  # SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_ACT: SK3B with the SiLU-mul epilogue
                from sglang.srt.speculative.dflash_sk3b_act import SK3BActProjection

                self._sk3b = SK3BActProjection(weight, dict(sk3b)).prepare()
            else:
                from sglang.srt.speculative.dflash_sk3b import SK3BProjection

                self._sk3b = SK3BProjection(weight, dict(sk3b)).prepare()
            self._sk3b_rows = frozenset(sk3b)
            self.rows = self.rows | self._sk3b_rows
        # SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_CONV (conv kernel projections only): the SK3B rows also on SK3B with the side-0 grouped conv in
        # the epilogue (run_conv); run() keeps the unfused SK3B for any other caller.
        self._sk3b_conv, self._conv_rows = None, frozenset()
        if conv and sk3b:
            from sglang.srt.speculative.dflash_sk3b_conv import SK3BConvProjection

            self._sk3b_conv = SK3BConvProjection(weight, dict(sk3b)).prepare()
            self._conv_rows = frozenset(r for r in sk3b if r % 8 == 0)
        # SGLANG_OPT_DFLASH_DRAFT_SPLITK_WCONV (conv kernel projections only): the WCONV_TACTICS rows that run exactly the gated split-K /
        # OV-wide tactic also on that tactic with the side-0 grouped conv in the epilogue (dflash_splitk_conv.py); others decline with a line.
        self._wconv, self._wconv_rows = None, frozenset()
        if wconv:
            wrows, declined = {}, []
            for rows, (wtactic, over) in WCONV_TACTICS.items():
                if rows in self._sk3b_rows:
                    continue
                if tuple(policies.get(rows, ())) == wtactic and (rows in self.ov_wide_rows) == over:
                    wrows[rows] = wtactic
                else:
                    declined.append((rows, policies.get(rows), rows in self.ov_wide_rows))
            if declined:
                logger.info("DFLASH draft WCONV declined: weight %s, (rows, served tactic, extended-SMEM) %s != gated %s", (n, k), declined, WCONV_TACTICS)
            if wrows:
                from sglang.srt.speculative.dflash_splitk_conv import SplitKConvProjection

                # prefetch form: chosen at 8 and 16 rows (faster than the plain form in a pre-registered A/B).
                self._wconv = SplitKConvProjection(
                    weight, wrows, oversized_rows=[r for r in wrows if WCONV_TACTICS[r][1]], prefetch=True
                ).prepare()
                self._wconv_rows = frozenset(wrows)
                self._conv_rows = self._conv_rows | self._wconv_rows
                # One line per prepared projection (engagement check in the server log).
                logger.info("DFLASH draft WCONV engaged: weight %s, rows %s on split-K + side-0 conv %s", (n, k), sorted(wrows), wrows)

    def covers(self, x: torch.Tensor) -> bool:
        # TMA reads rows at the tensor's pitch: unit inner stride, 16-byte pitch and base.
        return (
            x.dim() == 2
            and x.shape[0] in self.rows
            and x.shape[1] == self.k
            and x.dtype == torch.bfloat16
            and x.stride(1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    def covers_conv(self, x: torch.Tensor) -> bool:
        """Whether run_conv takes ``x`` (SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_CONV rows of a conv kernel projection)."""
        return x.shape[0] in self._conv_rows and self.covers(x)

    def run_conv(self, x: torch.Tensor, base0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(projection [rows, N], side-0 grouped conv of x [rows, K]) in one kernel; ``base0`` = the conv's base_kernel[0] [taps, K] BF16."""
        delta = x.new_empty((x.shape[0], self.n))
        out = torch.empty_like(x)
        if x.shape[0] in self._wconv_rows:
            self._wconv.run(x, delta, base0, out)
        else:
            self._sk3b_conv.run(x, delta, base0, out)
        return delta, out

    def trigger_follows_wait(self, rows: int) -> bool:
        """Whether this projection releases its own PDL dependents only after its griddepcontrol.wait at ``rows`` rows.
        The kernel triggers once its weight warp has issued every K tile of its split. With more tiles per split than
        pipeline stages, the weight load at index ``stages`` waits for the MMA of tile 0 to free its stage, and that
        MMA needs the activation tile, which is loaded only after the wait. With stages >= tiles, every weight load
        can be issued before the wait, so the trigger can come before it. A fenced trigger mode (``ov_trigger`` 2) triggers
        only after the wait whatever the stages; SK3B rows trigger after their own wait. False for a row count without a
        tactic."""
        if rows in self._sk3b_rows:
            return True
        policy = self._policies.get(rows)
        if policy is None:
            return False
        trigger = getattr(self, "_triggers", {}).get(rows, _TRIGGER_SHIPPED)
        return trigger != _TRIGGER_SHIPPED or self.k // (_CTA_K * policy[2]) > policy[3]

    def run(self, x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        if out is None:
            out = x.new_empty((x.shape[0], self.n))
        if x.shape[0] in self._sk3b_rows:
            return self._sk3b.run(x, out, prefetch=self._prefetch)
        if x.shape[0] in self.ov_wide_rows:
            return self._ov_wide_prepared.run(x, out)
        if self._ov_prepared is not None and x.shape[0] <= MAX_SPLITK_ROWS:
            return self._ov_prepared.run(x, out)
        return self._prepared.run(x, out)

    def tensor_maps(self) -> List[torch.Tensor]:
        """The device tensors holding this projection's TMA weight descriptors (one per M tile height)."""
        prepared = (p for p in (self._prepared, self._ov_wide_prepared, self._ov_prepared) if p is not None)
        maps = [m for p in prepared for m in p.maps.values()]
        if self._sk3b is not None:
            maps.append(self._sk3b.map)
        if self._sk3b_conv is not None:
            maps.extend(self._sk3b_conv.tensor_maps())
        if self._wconv is not None:
            maps.extend(self._wconv.tensor_maps())
        return maps


class DraftSplitKLinearMethod(UnquantizedLinearMethod):
    """UnquantizedLinearMethod with covered calls on the split-K projection of the layer's weight."""

    def __init__(self, projection: DraftSplitKProjection):
        super().__init__()
        self.projection = projection

    def apply(self, layer, x, bias=None):
        if bias is None and self.projection.covers(x):
            return self.projection.run(x)
        return super().apply(layer, x, bias)

    def apply_into(self, layer, x, output, bias=None):
        if (
            bias is None
            and self.projection.covers(x)
            and output.is_contiguous()
            and output.dtype == torch.bfloat16
            and output.shape == (x.shape[0], self.projection.n)
        ):
            return self.projection.run(x, output)
        return super().apply_into(layer, x, output, bias)


def _sk3b_rows(shape: Tuple[int, int], act_fuse: bool) -> Dict[int, Tuple[int, ...]]:
    """The SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B* rows of a weight shape ({rows: SK3B policy}; empty when no SK3B knob covers it)."""
    sk3b = {}
    if not act_fuse:
        if envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B.get():
            sk3b.update(SK3B_TACTICS.get(shape, {}))
        if envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_LOW.get():
            sk3b.update(SK3B_LOW_TACTICS.get(shape, {}))
        if envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_B4.get():
            sk3b.update(SK3B_B4_TACTICS.get(shape, {}))
        if sk3b:
            # One line per prepared projection (engagement check in the server log).
            logger.info("DFLASH draft SK3B engaged: weight %s, rows %s on SK3B %s", shape, sorted(sk3b), sk3b)
    elif envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_ACT.get():
        sk3b.update(SK3B_ACT_TACTICS.get(shape, {}))
        if envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_LOW.get():
            sk3b.update(SK3B_ACT_LOW_TACTICS.get(shape, {}))
        if envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_B4.get():
            sk3b.update(SK3B_ACT_B4_TACTICS.get(shape, {}))
        if sk3b:
            # One line per prepared projection (engagement check in the server log).
            logger.info("DFLASH draft SK3B act engaged: weight %s, rows %s on SK3B + SiLU-mul %s", shape, sorted(sk3b), sk3b)
    return sk3b


def _projection(
    weight: torch.Tensor,
    act_fuse: bool = False,
    prefetch: Optional[torch.Tensor] = None,
    conv: bool = False,
    conv_projection: bool = False,
) -> Optional[DraftSplitKProjection]:
    """``conv``: a grouped conv's kernel projection whose conv has the fixed form of the SK3B / WCONV conv epilogues.
    ``conv_projection``: a grouped conv's kernel projection (any form). Its predecessor, the draft AR site, triggers it
    right after the site's own wait only if the projection releases its dependents after its wait
    (DraftArSites._early_for), so its OV form (SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV) takes trigger mode 2; the other OV
    projections keep the shipped trigger (before their wait with OV). Their dependents (the q_a RMSNorm after qkv_a,
    the ring down_proj after gate_up) neither read nor write memory before their own wait."""
    shape = tuple(weight.shape)
    tactic = SPLITK_TACTICS.get(shape)
    if weight.dtype != torch.bfloat16:
        return None
    if tactic is None:
        return None
    wide = None
    if envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE.get():
        wide = WIDE_ACT_TACTICS if act_fuse else WIDE_TACTICS.get(shape)
    ov_table = OV_WIDE_ACT_TACTICS if act_fuse else OV_WIDE_TACTICS.get(shape)
    ov_wide = {rows: ov_table[rows] for rows in ov_wide_rows() if rows in ov_table} if ov_table is not None else None
    sk3b = _sk3b_rows(shape, act_fuse) or None
    if sk3b:
        # SK3B rows take precedence over the WIDE and OV-wide rows (SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_LOW at 24-64 rows).
        if wide:
            wide = {r: t for r, t in wide.items() if r not in sk3b}
        if ov_wide:
            ov_wide = {r: t for r, t in ov_wide.items() if r not in sk3b}
    if not (sk3b and envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_PF.get()):
        prefetch = None
    if prefetch is not None:
        # One line per prepared projection (engagement check in the server log).
        logger.info("DFLASH draft SK3B prefetch engaged: weight %s, rows %s prefetch the next weight %s", shape, sorted(sk3b), tuple(prefetch.shape))
    wconv = bool(conv and envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_WCONV.get())
    conv = bool(conv and sk3b and envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_CONV.get())
    if conv:
        # One line per prepared projection (engagement check in the server log).
        logger.info("DFLASH draft SK3B conv engaged: weight %s, rows %s on SK3B + side-0 conv %s", shape, sorted(sk3b), sk3b)
    ov = OV_TACTICS.get(shape) if envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV.get() else None
    ov_trigger = _TRIGGER_AFTER_WAIT if conv_projection else _TRIGGER_SHIPPED
    return DraftSplitKProjection(
        weight, tactic, act_fuse=act_fuse, wide=wide, ov_wide=ov_wide, sk3b=sk3b, prefetch=prefetch, conv=conv, wconv=wconv,
        ov=ov, ov_trigger=ov_trigger,
    )  # fmt: skip


def ov_wide_rows() -> Tuple[int, ...]:
    """SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV_WIDE_ROWS as row counts; fails closed on a row count without an OV-wide entry
    for any K = 7168 shape, or without SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE (the wide tactics it keeps). A shape without
    an entry at a listed row count keeps its served kernel there."""
    rows = tuple(sorted({int(r) for r in envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV_WIDE_ROWS.get()}))
    if not rows:
        return ()
    if not envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE.get():
        raise ValueError("SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV_WIDE_ROWS needs SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE")
    tables = (*OV_WIDE_TACTICS.values(), OV_WIDE_ACT_TACTICS)
    missing = [r for r in rows if not any(r in table for table in tables)]
    if missing:
        raise ValueError(f"SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV_WIDE_ROWS: no OV-wide tactic at rows {missing}")
    return rows


def _install_mlp_act(mlp) -> int:
    """SGLANG_OPT_DFLASH_DRAFT_SPLITK_ACT: the fused gate_up + SiLU-mul projection DFlashMLP.forward reads for covered rows
    (the unfused split-K gate_up stays for the rest); returns how many projections it installed (0 or 1)."""
    if not (envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_ACT.get() and _fusable_silu(mlp)):
        return 0
    mlp.split_k_gate_up_act = _projection(mlp.gate_up_proj.weight, act_fuse=True, prefetch=mlp.down_proj.weight)
    return int(mlp.split_k_gate_up_act is not None)


def _fusable_silu(mlp) -> bool:
    """The MLP's act_fn is sglang's SiluAndMul on its default dispatch (the act_and_mul kernel the fused epilogue reproduces
    bitwise), not pinned to forward_native (the deterministic RL mode's exact torch SiLU)."""
    from sglang.srt.layers.activation import SiluAndMul

    act = mlp.act_fn
    return type(act) is SiluAndMul and act._forward_method != act.forward_native


def _sk3b_conv_ok(conv) -> bool:
    """SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_CONV applies to this DFlashGroupedConv: the epilogue's fixed form (2 taps, 448 groups of 16 channels
    = hidden 7168, block 8, BF16 base kernel [2, taps, hidden])."""
    base = conv.base_kernel
    return (
        conv.taps == 2
        and conv.group_size == 16
        and conv.num_groups == 448
        and conv.block_size == 8
        and base.dtype == torch.bfloat16
        and tuple(base.shape) == (2, 2, 7168)
        and base.is_contiguous()
    )


def _conv_form_ok(conv) -> bool:
    """_sk3b_conv_ok, plus one decline line per word that asked for a conv epilogue (SK3B_CONV at 24-64 rows, WCONV at 8 / 16 rows) on a
    conv outside that fixed form, so neither word declines silently."""
    ok = _sk3b_conv_ok(conv)
    if not ok:
        form = (conv.taps, conv.group_size, conv.num_groups, conv.block_size, conv.base_kernel.dtype, tuple(conv.base_kernel.shape))
        if envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_CONV.get():
            logger.info("DFLASH draft SK3B conv declined: taps %s group %s groups %s block %s base %s %s", *form)
        if envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_WCONV.get():
            logger.info("DFLASH draft WCONV declined: conv taps %s group %s groups %s block %s base %s %s outside the fixed conv form", *form)
    return ok


def install_draft_splitk(draft_model) -> int:
    """Route the covered draft-layer linears of an MLA DFlash draft to split-K; returns how many."""
    from sglang.srt.models.dflash import DFlashMLAAttention

    installed, projections = 0, []
    for layer in draft_model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        if not isinstance(attn, DFlashMLAAttention) or attn.q_lora_rank is None:
            raise ValueError("SGLANG_OPT_DFLASH_DRAFT_SPLITK needs an MLA draft with q_lora")
        linears = (
            attn.fused_qkv_a_proj_with_mqa,
            attn.q_b_proj,
            attn.o_proj,
            mlp.gate_up_proj,
            mlp.down_proj,
        )
        # SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_PF: the GEMM after each of these runs on cuBLAS at the SK3B rows.
        nexts = {id(attn.fused_qkv_a_proj_with_mqa): attn.o_proj.weight, id(mlp.gate_up_proj): mlp.down_proj.weight}
        for linear in linears:
            if not isinstance(linear.quant_method, UnquantizedLinearMethod):
                continue
            projection = _projection(linear.weight, prefetch=nexts.get(id(linear)))
            if projection is not None:
                linear.quant_method = DraftSplitKLinearMethod(projection)
                installed += 1
                projections.append(projection)
        installed += _install_mlp_act(mlp)
        projections.append(getattr(mlp, "split_k_gate_up_act", None))
        attn.route_projections_through_linears()
        for conv in (layer.attention_conv, layer.mlp_conv):
            if conv is not None:
                conv_ok = _conv_form_ok(conv)
                conv.split_k_projection = _projection(
                    conv.kernel_projection.weight, conv=conv_ok, conv_projection=True
                )
                installed += conv.split_k_projection is not None
                projections.append(conv.split_k_projection)
    logger.info("DFLASH draft split-K GEMMs on %d linears", installed)
    rows = ov_wide_rows()
    if rows:
        ov = sum(p is not None and p._ov_wide_prepared is not None for p in projections)
        logger.info(
            "DFLASH draft split-K OV-wide ENGAGED: %d projections at rows %s (extended-SMEM, stages %s)",
            ov, ",".join(map(str, rows)), ",".join(str(OV_WIDE_ACT_TACTICS[r][3]) for r in rows),
        )  # fmt: skip
    if envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV.get():
        ov = sum(p is not None and p._ov_prepared is not None for p in projections)
        logger.info("DFLASH draft split-K OV: %d projections (all weight tiles before the PDL wait at <= 8 rows)", ov)
    return installed


# Arbitrary bound; larger tensors are GEMM weights their kernels stream anyway.
_L2_PREFETCH_MAX_TENSOR_BYTES = 4 << 20


def _own_tensors(module):
    attrs = (v for v in vars(module).values() if isinstance(v, torch.Tensor))
    return chain(module.parameters(recurse=False), module.buffers(recurse=False), attrs)


def draft_l2_prefetch_tensors(draft_model) -> List[torch.Tensor]:
    """What the draft layers re-read each step besides GEMM weights: every CUDA tensor of at most 4 MiB owned by a
    non-linear module of a layer or the final norm (norm weights, conv base kernels, absorbed w_kc / w_vc, ...), and
    every installed split-K projection's TMA weight maps. Linear layers' own weights are left out."""
    from sglang.srt.layers.linear import LinearBase

    tensors, seen = [], set()

    def add(t):
        if t.is_cuda and t.numel() * t.element_size() <= _L2_PREFETCH_MAX_TENSOR_BYTES and t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            tensors.append(t)

    projections = []
    for layer in draft_model.layers:
        for module in layer.modules():
            if isinstance(module, LinearBase):
                if isinstance(module.quant_method, DraftSplitKLinearMethod):
                    projections.append(module.quant_method.projection)
                continue
            for t in _own_tensors(module):
                add(t)
        projections.append(layer.mlp.split_k_gate_up_act)
        projections.extend(c.split_k_projection for c in (layer.attention_conv, layer.mlp_conv) if c is not None)
    for module in draft_model.norm.modules():
        for t in _own_tensors(module):
            add(t)
    for projection in projections:
        if projection is not None:
            for t in projection.tensor_maps():
                add(t)
    return tensors


class DraftL2Prefetch:
    """SGLANG_OPT_DFLASH_DRAFT_L2_PREFETCH: capture hooks that fork a branch at the head of each draft graph, prefetch
    ``tensors`` into L2 on it and join it at the tail, so the prefetch overlaps the draft layers instead of preceding them.

    The bulk prefetch returns no data and has no completion, so the draft's outputs are unchanged."""

    def __init__(self, tensors: List[torch.Tensor]):
        from sglang.kernels.ops.speculative.l2_prefetch import build_l2_prefetch_table

        self.tensors = tensors
        self.table = build_l2_prefetch_table(tensors)
        self._stream = torch.cuda.Stream(device=self.table.device)

    def capture_head_hook(self, runner, forward_batch) -> None:
        from sglang.kernels.ops.speculative.l2_prefetch import l2_prefetch

        del runner, forward_batch
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            l2_prefetch(self.table)

    def capture_tail_hook(self, runner, out, forward_batch, num_tokens) -> None:
        del runner, out, forward_batch, num_tokens
        torch.cuda.current_stream().wait_stream(self._stream)
