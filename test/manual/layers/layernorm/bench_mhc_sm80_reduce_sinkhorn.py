#!/usr/bin/env python3
"""Diagnostic + benchmark for the V4.1 mHC reduce+Sinkhorn kernel on CUDA SM80.

TARGET
  ``sglang.kernels.ops.layernorm.mhc._hc_mix_reduce_sinkhorn_kernel`` as launched
  by ``hc_mix_stats_sinkhorn`` for the deployed single-request decode geometry:
  M=1, K=20480, MIX=24, HC=4, NUM_SLICES=80, ITERS=20, rms_eps=hc_eps=1e-6.
  Whole-model profiling attributed 0.6057 ms/token to this kernel over 320
  launches (40 layers x 8 steps).  That number is quoted two ways -- 0.6057/40 =
  15.14 us per launch and 0.6057/320 = 1.89 us per launch -- and this benchmark
  exists to decide which of those two the kernel body can actually justify.

WHAT IT MEASURES
  1. ``hot``: baseline (real production kernel), every candidate variant, and
     two attribution kernels (load loop only, Sinkhorn only) plus an empty
     kernel for the launch floor.  Each is measured two ways inside a CUDA
     graph captured on a NON-default stream:
       * ``amortized_us``: K identical launches inside one graph, so consecutive
         CTAs may overlap the way back-to-back model launches do.
       * ``single_us``: one launch per graph, one event pair per replay, which
         is the relevant number for a serialized decode chain.
  2. ``cold``: >= 64 distinct (hc_fn, partial) working sets, each config captured
     once per set, L2 flushed (48 MB > 40 MB A100 L2) before every timed replay,
     sets rotated.  Reports median us/launch, speedup vs baseline, and projected
     ms saved per 40 layers and per 80 calls.
  3. ``numerics``: every variant on every fixture is compared against (a) a
     float64 CPU reference that performs the *literal sequential* slice sum and
     the literal Sinkhorn iteration sequence, and (b) the unmodified production
     kernel.  max_abs, scale-relative error and ULP-ish distance are reported
     per output.  Tolerances are NOT invented here: ATOL=RTOL=1e-5 is the value
     already fixed in test/registered/kernels/ops/layernorm/
     test_mhc_sm80_fused_stats.py.  No variant gets a wider tolerance, and the
     tree-reduction variant is never claimed to be bit-exact.

PRE-REGISTERED CRITERIA (fixed before any measurement, printed as one JSON line)
  * every candidate output must be finite on every fixture;
  * every candidate must satisfy ``|v - ref64| <= ATOL + RTOL*|ref64|``
    pointwise on every QUALIFICATION fixture;
  * the recommended config's max_abs error vs float64 must be <= 4x the
    baseline's max_abs error (never tighter than 4xATOL scaled by the output
    magnitude) on every qualification fixture;
  * to be recommended as a change, the winner must beat the baseline by
    >= 1.15x on ``single_us`` (the serialized-decode metric).  Otherwise the
    printed recommendation is explicitly "keep_production_baseline" and the run
    still reports PASS, because the diagnostic then proved a real negative.
  ``stress`` fixtures (designed for catastrophic FP32 cancellation) are reported
  in full but excluded from the verdict, since the baseline itself fails them by
  construction; their numbers are printed so the association-order risk is
  visible rather than hidden.

OUTPUT / MARKER
  One JSON object per stdout line (``"event"`` names the record).  The last line
  is exactly one of::

      BENCH_RESULT: PASS
      BENCH_RESULT: FAIL
      BENCH_RESULT: SKIP (<reason>)

  Exit code 0 for PASS/SKIP, 1 for FAIL.

RUN
  PYTHONPATH=python python test/manual/layers/layernorm/bench_mhc_sm80_reduce_sinkhorn.py
  PYTHONPATH=python python .../bench_mhc_sm80_reduce_sinkhorn.py --preflight
  PYTHONPATH=python python .../bench_mhc_sm80_reduce_sinkhorn.py --quick --skip-cold
  PYTHONPATH=python python .../bench_mhc_sm80_reduce_sinkhorn.py --simt-stats

  Useful flags: --inner --samples --sets --fixtures --warps --skip-cold --quick
  --preflight --list-configs --json-timings.

HONEST SCOPE OF VERIFICATION
  This file was authored and syntax/statically checked on a machine WITHOUT
  torch or triton, so NO GPU number in it has ever been produced locally.  What
  was verified statically: it compiles under python3.12; it contains no tabs and
  no trailing whitespace; ``_hc_mix_reduce_sinkhorn_kernel``,
  ``_hc_mix_stats_partial_kernel``, ``hc_mix_stats_sinkhorn`` and
  ``hc_mix_stats`` exist in mhc.py with exactly the parameter names this file
  launches with (re-checked at run time by ``--preflight``/``event=arg_check``);
  every variant kernel's parameter list matches its launch site; and the epilogue
  helper is called with all 15 of its parameters at all 5 call sites.
  Every timing, numerical and resource number this file prints is UNTESTED until
  it is run on the A100 host.

FIXTURE DESIGN EVIDENCE (CPU, numpy, summation-order simulation only)
  The fixtures were designed against a CPU simulation of the FP32 summation
  order and of the epilogue, because a fixture that does not discriminate would
  waste the GPU run:
    * partials_cancel_alt: exact a_comb[0] = 77.72; sequential FP32 = 38.80
      (abs err 38.9), pairwise tree = 71.81 (abs err 5.9). Resulting outputs
      differ between the two orders by up to 0.031 (comb) and 0.132 (pre), i.e.
      far beyond ATOL=1e-5. The deployed kernel keeps the sequential order, so
      this fixture is the one that makes the tree variant's association change
      visible, and it is tagged stress and excluded from the verdict.
    * partials_random (realistic scale): sequential vs tree differ by 6e-8 in
      the outputs, i.e. ~500x inside tolerance. On realistic partials the
      association change is not numerically material.
    * partials_zeros / huge 1e30 / tiny 1e-30 / comb_dominant_row /
      comb_dominant_col / comb_uniform are finite in both FP32 and FP64 in the
      same simulation (the 1e30 and 1e-30 cases saturate, so they exercise the
      eps and finiteness paths rather than sensitivity).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import sys
import traceback
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - import guard only
    print(f"FATAL: triton import failed: {exc!r}", flush=True)
    raise SystemExit(3)


# ---------------------------------------------------------------------------
# Geometry and frozen parameters (deployed V4.1 single-request decode)
# ---------------------------------------------------------------------------
HC = 4
K = 20480
MIX = 24
ITERS = 20
NUM_SLICES = 80
M = 1
RMS_EPS = 1e-6
HC_EPS = 1e-6
INV_K = 1.0 / K
MIX_PAD = 32
SBLOCK = 128  # next power of two >= NUM_SLICES

ATOL = 1e-5
RTOL = 1e-5
TOLERANCE_SOURCE = (
    "test/registered/kernels/ops/layernorm/test_mhc_sm80_fused_stats.py "
    "(ATOL=RTOL=1e-5)"
)
ERROR_BUDGET_FACTOR = 4.0
SPEEDUP_GATE = 1.15
SEED = 41041

PROFILE_MS_PER_TOKEN = 0.6057
PROFILE_LAUNCHES = 320
PROFILE_LAYERS = 40
PROFILE_STEPS = 8

BASELINE_CID = "prod_nw1_nsdef"
L2_FLUSH_MB = 48
DEFAULT_SETS = 64

OUTPUT_NAMES = ("pre", "post", "comb")


def emit(**fields):
    """One JSON object per line, sorted keys, flushed."""
    print(json.dumps(fields, sort_keys=True, default=str), flush=True)


# ---------------------------------------------------------------------------
# Kernels: shared Sinkhorn epilogue
# ---------------------------------------------------------------------------
@triton.jit
def _sinkhorn_store(
    a_pre,
    a_post,
    a_comb,
    sq,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    row,
    inv_k,
    rms_eps,
    HC: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    """Byte-for-byte the production epilogue: rsqrt, sigmoid, 2D Sinkhorn.

    Kept in one place so every variant differs only in how a_pre/a_post/a_comb/sq
    are produced, never in the epilogue arithmetic or the iteration count.
    """
    rsqrt = 1.0 / tl.sqrt(sq * inv_k + rms_eps)
    j = tl.arange(0, HC)
    jj = j[:, None]
    kk = j[None, :]
    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)
    pre = tl.sigmoid(a_pre * rsqrt * s0 + tl.load(base_ptr + j)) + EPS
    tl.store(pre_ptr + row * HC + j, pre)
    post = 2.0 * tl.sigmoid(a_post * rsqrt * s1 + tl.load(base_ptr + HC + j))
    tl.store(post_ptr + row * HC + j, post)
    comb = a_comb * rsqrt * s2 + tl.load(base_ptr + 2 * HC + jj * HC + kk)
    comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = comb / (tl.sum(comb, axis=1)[:, None] + EPS)
    comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    for _ in tl.static_range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + EPS)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    tl.store(comb_ptr + row * HC * HC + jj * HC + kk, comb)


# ---------------------------------------------------------------------------
# Variant (a) fidelity copy: structure of the unmodified production kernel
# ---------------------------------------------------------------------------
@triton.jit
def _reduce_sinkhorn_static_copy(
    part_mix_ptr,
    part_sq_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    part_mix_residual_ptr=None,
):
    """Verbatim copy of the production body (static_range, 3 tiny loads/slice).

    Exists so the harness can prove that the fixtures, the launch arguments and
    the output buffers reproduce the production kernel bitwise.  If this copy is
    not bitwise-equal to production, every other comparison in this file is
    suspect and the run says so.
    """
    row = tl.program_id(0)
    if row >= m:
        return
    j = tl.arange(0, HC)
    jj = j[:, None]
    kk = j[None, :]

    a_pre = tl.zeros([HC], dtype=tl.float32)
    a_post = tl.zeros([HC], dtype=tl.float32)
    a_comb = tl.zeros([HC, HC], dtype=tl.float32)
    sq = tl.zeros([], dtype=tl.float32)
    for s in tl.static_range(NUM_SLICES):
        off = (s * m + row) * MIX
        v_pre = tl.load(part_mix_ptr + off + j)
        v_post = tl.load(part_mix_ptr + off + HC + j)
        v_comb = tl.load(part_mix_ptr + off + 2 * HC + jj * HC + kk)
        if part_mix_residual_ptr is not None:
            v_pre += tl.load(part_mix_residual_ptr + off + j)
            v_post += tl.load(part_mix_residual_ptr + off + HC + j)
            v_comb += tl.load(part_mix_residual_ptr + off + 2 * HC + jj * HC + kk)
        a_pre += v_pre
        a_post += v_post
        a_comb += v_comb
        sq += tl.load(part_sq_ptr + s * m + row)
    _sinkhorn_store(
        a_pre, a_post, a_comb, sq, scale_ptr, base_ptr, pre_ptr, post_ptr, comb_ptr,
        row, inv_k, rms_eps, HC=HC, ITERS=ITERS, EPS=EPS,
    )


# ---------------------------------------------------------------------------
# Variant (b) / (d): kernel-level num_stages is a no-op on static_range, so the
# same body is also emitted with tl.range(..., num_stages=NS) to give the
# software pipeliner an actual loop to work with.
# ---------------------------------------------------------------------------
@triton.jit
def _reduce_sinkhorn_tlrange(
    part_mix_ptr,
    part_sq_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    NS: tl.constexpr,
):
    """Production load structure with tl.range(0, NUM_SLICES, num_stages=NS)."""
    row = tl.program_id(0)
    if row >= m:
        return
    j = tl.arange(0, HC)
    jj = j[:, None]
    kk = j[None, :]

    a_pre = tl.zeros([HC], dtype=tl.float32)
    a_post = tl.zeros([HC], dtype=tl.float32)
    a_comb = tl.zeros([HC, HC], dtype=tl.float32)
    sq = tl.zeros([], dtype=tl.float32)
    for s in tl.range(0, NUM_SLICES, num_stages=NS):
        off = (s * m + row) * MIX
        v_pre = tl.load(part_mix_ptr + off + j)
        v_post = tl.load(part_mix_ptr + off + HC + j)
        v_comb = tl.load(part_mix_ptr + off + 2 * HC + jj * HC + kk)
        a_pre += v_pre
        a_post += v_post
        a_comb += v_comb
        sq += tl.load(part_sq_ptr + s * m + row)
    _sinkhorn_store(
        a_pre, a_post, a_comb, sq, scale_ptr, base_ptr, pre_ptr, post_ptr, comb_ptr,
        row, inv_k, rms_eps, HC=HC, ITERS=ITERS, EPS=EPS,
    )


# ---------------------------------------------------------------------------
# Variant (c): the whole (NUM_SLICES, MIX) partial tile in ONE masked 2D load,
# reduced with tl.sum(axis=0) (tree).  Changes FP32 association order.
# ---------------------------------------------------------------------------
@triton.jit
def _reduce_sinkhorn_oneload(
    part_mix_ptr,
    part_sq_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    SBLOCK: tl.constexpr,
    NPAD: tl.constexpr,
):
    """One (SBLOCK, NPAD) load + tl.sum(axis=0), then masked extraction."""
    row = tl.program_id(0)
    if row >= m:
        return
    sv = tl.arange(0, SBLOCK)
    nv = tl.arange(0, NPAD)
    smask = sv < NUM_SLICES
    nmask = nv < MIX

    tile = tl.load(
        part_mix_ptr + (sv[:, None] * m + row) * MIX + nv[None, :],
        mask=smask[:, None] & nmask[None, :],
        other=0.0,
    )
    colsum = tl.sum(tile, axis=0)
    sq = tl.sum(tl.load(part_sq_ptr + sv * m + row, mask=smask, other=0.0), axis=0)

    j = tl.arange(0, HC)
    a_pre = tl.sum(tl.where(nv[None, :] == j[:, None], colsum[None, :], 0.0), axis=1)
    a_post = tl.sum(
        tl.where(nv[None, :] == (HC + j)[:, None], colsum[None, :], 0.0), axis=1
    )
    jj = tl.arange(0, HC)
    kk = tl.arange(0, HC)
    cidx = 2 * HC + jj[:, None] * HC + kk[None, :]
    a_comb = tl.sum(
        tl.where(nv[None, None, :] == cidx[:, :, None], colsum[None, None, :], 0.0),
        axis=2,
    )
    _sinkhorn_store(
        a_pre, a_post, a_comb, sq, scale_ptr, base_ptr, pre_ptr, post_ptr, comb_ptr,
        row, inv_k, rms_eps, HC=HC, ITERS=ITERS, EPS=EPS,
    )


# ---------------------------------------------------------------------------
# Middle ground: keep the three component groups, but load each as a
# slice-axis tile so the reduction is a tree over the slice dimension.
# ---------------------------------------------------------------------------
@triton.jit
def _reduce_sinkhorn_split2d(
    part_mix_ptr,
    part_sq_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    SBLOCK: tl.constexpr,
):
    """Three slice-axis loads (2D/2D/3D) + tl.sum(axis=0), no extraction."""
    row = tl.program_id(0)
    if row >= m:
        return
    sv = tl.arange(0, SBLOCK)
    smask = sv < NUM_SLICES
    j = tl.arange(0, HC)
    base_off = (sv * m + row) * MIX

    v_pre = tl.load(
        part_mix_ptr + base_off[:, None] + j[None, :],
        mask=smask[:, None],
        other=0.0,
    )
    v_post = tl.load(
        part_mix_ptr + base_off[:, None] + HC + j[None, :],
        mask=smask[:, None],
        other=0.0,
    )
    jc = tl.arange(0, HC)
    kc = tl.arange(0, HC)
    v_comb = tl.load(
        part_mix_ptr
        + base_off[:, None, None]
        + 2 * HC
        + jc[None, :, None] * HC
        + kc[None, None, :],
        mask=smask[:, None, None],
        other=0.0,
    )
    a_pre = tl.sum(v_pre, axis=0)
    a_post = tl.sum(v_post, axis=0)
    a_comb = tl.sum(v_comb, axis=0)
    sq = tl.sum(tl.load(part_sq_ptr + sv * m + row, mask=smask, other=0.0), axis=0)
    _sinkhorn_store(
        a_pre, a_post, a_comb, sq, scale_ptr, base_ptr, pre_ptr, post_ptr, comb_ptr,
        row, inv_k, rms_eps, HC=HC, ITERS=ITERS, EPS=EPS,
    )


# ---------------------------------------------------------------------------
# Attribution kernels (never candidates): where does the time actually go?
# ---------------------------------------------------------------------------
@triton.jit
def _reduce_sinkhorn_slice0(
    part_mix_ptr,
    part_sq_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    """Production load structure restricted to slice 0: isolates Sinkhorn cost.

    baseline_single_us - slice0_single_us bounds the cost of the 79 extra slice
    loads; slice0_single_us - empty_single_us bounds the Sinkhorn epilogue cost.
    """
    row = tl.program_id(0)
    if row >= m:
        return
    j = tl.arange(0, HC)
    jj = j[:, None]
    kk = j[None, :]
    off = row * MIX
    a_pre = tl.load(part_mix_ptr + off + j)
    a_post = tl.load(part_mix_ptr + off + HC + j)
    a_comb = tl.load(part_mix_ptr + off + 2 * HC + jj * HC + kk)
    sq = tl.load(part_sq_ptr + row)
    _sinkhorn_store(
        a_pre, a_post, a_comb, sq, scale_ptr, base_ptr, pre_ptr, post_ptr, comb_ptr,
        row, inv_k, rms_eps, HC=HC, ITERS=ITERS, EPS=EPS,
    )


@triton.jit
def _reduce_loads_only(
    part_mix_ptr,
    part_sq_ptr,
    out_mix_ptr,
    out_sq_ptr,
    m,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
):
    """Verbatim production load/accumulate loop, no Sinkhorn, no transcendentals."""
    row = tl.program_id(0)
    if row >= m:
        return
    j = tl.arange(0, HC)
    jj = j[:, None]
    kk = j[None, :]
    a_pre = tl.zeros([HC], dtype=tl.float32)
    a_post = tl.zeros([HC], dtype=tl.float32)
    a_comb = tl.zeros([HC, HC], dtype=tl.float32)
    sq = tl.zeros([], dtype=tl.float32)
    for s in tl.static_range(NUM_SLICES):
        off = (s * m + row) * MIX
        a_pre += tl.load(part_mix_ptr + off + j)
        a_post += tl.load(part_mix_ptr + off + HC + j)
        a_comb += tl.load(part_mix_ptr + off + 2 * HC + jj * HC + kk)
        sq += tl.load(part_sq_ptr + s * m + row)
    out = row * MIX
    tl.store(out_mix_ptr + out + j, a_pre)
    tl.store(out_mix_ptr + out + HC + j, a_post)
    tl.store(out_mix_ptr + out + 2 * HC + jj * HC + kk, a_comb)
    tl.store(out_sq_ptr + row, sq)


@triton.jit
def _empty_kernel(dummy_ptr):
    """Launch floor: one CTA that stores one float."""
    tl.store(dummy_ptr + tl.program_id(0), 1.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@dataclass
class Fixture:
    name: str
    tag: str
    note: str
    part_mix: torch.Tensor
    part_sq: torch.Tensor
    scale: torch.Tensor
    base: torch.Tensor
    x: torch.Tensor
    fn: torch.Tensor
    partials_override: bool


@dataclass
class ReduceBuffers:
    part_mix: torch.Tensor
    part_sq: torch.Tensor
    scale: torch.Tensor
    base: torch.Tensor
    pre: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    out_mix: torch.Tensor
    out_sq: torch.Tensor


FIXTURE_SPECS = (
    ("prod_realistic", "qualification", "production split-K partials from seeded x/fn", False),
    ("partials_random", "qualification", "seeded pseudo-random partials, realistic scale", True),
    ("inputs_mixed_signs", "qualification", "seeded +/- inputs; per-slice partials mix signs", False),
    ("inputs_zero", "qualification", "x and hc_fn exactly zero; partials all zero", False),
    ("inputs_large", "qualification", "x scaled to 1e4; large finite partials", False),
    ("partials_zeros", "qualification", "all-zero part_mix and part_sq (eps-only path)", True),
    ("partials_huge_1e30", "qualification", "1e30 magnitudes: finiteness/eps path; outputs saturate, so not a sensitive discriminator", True),
    ("partials_tiny_1e-30", "qualification", "1e-30 magnitudes: finiteness/eps path; outputs saturate, so not a sensitive discriminator", True),
    ("comb_dominant_row", "qualification", "comb logits: row 0 = +30, other rows = -30", True),
    ("comb_dominant_col", "qualification", "comb logits: col 0 = +30, other cols = -30", True),
    ("comb_uniform", "qualification", "comb logits all zero -> doubly stochastic limit", True),
    ("partials_cancel_alt", "stress", "1e8/-1e8 swamping then cancellation; sequential vs tree association stress", True),
)


def _seeded_inputs(seed, scale_x=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = (
        torch.randn(M, K, generator=gen, device="cuda", dtype=torch.float32)
        .mul_(scale_x)
        .to(torch.bfloat16)
        .contiguous()
    )
    fn = (
        torch.randn(MIX, K, generator=gen, device="cuda", dtype=torch.float32)
        .mul_(0.01)
        .contiguous()
    )
    return x, fn


def _scaled(tensor, target_max):
    peak = tensor.abs().max()
    if float(peak) == 0.0:
        return tensor
    return (tensor / peak * target_max).contiguous()


def build_fixtures(prod, names=None, seed=SEED, simt_predicate=False):
    """Deterministic fixtures; each one carries partials AND a matching x/hc_fn."""
    fixtures = []
    base_seed = seed
    for idx, (name, tag, note, override) in enumerate(FIXTURE_SPECS):
        if names is not None and name not in names:
            continue
        x, fn = _seeded_inputs(base_seed + idx)
        scale = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32, device="cuda")
        gen = torch.Generator(device="cuda").manual_seed(base_seed + 1000 + idx)
        base = (
            torch.randn(MIX, generator=gen, device="cuda", dtype=torch.float32) * 0.1
        ).contiguous()

        if override:
            genp = torch.Generator(device="cuda").manual_seed(base_seed + 2000 + idx)
            # Realistic partial scale: per-slice dot of N(0,1) x with a 256-long
            # column of N(0,0.01) weights, and per-slice sum of squares ~256.
            pm = (
                torch.randn(NUM_SLICES, M, MIX, generator=genp, device="cuda",
                            dtype=torch.float32)
                .mul_(0.16)
                .contiguous()
            )
            ps = (
                torch.rand(NUM_SLICES, M, generator=genp, device="cuda",
                           dtype=torch.float32)
                .sub_(0.5)
                .mul_(25.6)
                .add_(256.0)
                .contiguous()
            )
            if name == "partials_zeros":
                pm = torch.zeros_like(pm)
                ps = torch.zeros_like(ps)
            elif name == "partials_huge_1e30":
                pm = _scaled(pm, 1e30)
                ps = _scaled(ps, 1e30)
            elif name == "partials_tiny_1e-30":
                pm = _scaled(pm, 1e-30)
                ps = _scaled(ps, 1e-30)
            elif name == "partials_cancel_alt":
                # Adversarial association-order fixture.  One +1e8 slice, one
                # -1e8 slice 40 positions later, and ~1.0 in between:
                #   sequential FP32 absorbs every small term added while the
                #   running sum is 1e8 (ulp(1e8) = 8, so +1.0 rounds away), and
                #   when the -1e8 finally arrives it cancels to 0, leaving only
                #   the terms that came after it;
                #   a pairwise tree keeps the small terms in its intermediate
                #   nodes, so it stays far closer to the exact total.
                # rms_eps scaling is chosen (part_sq = 1e5/slice -> rms = 390)
                # so the surviving logits land in the sensitive sigmoid/exp
                # region instead of saturating, i.e. the accumulator difference
                # is visible in pre/post/comb rather than hidden by saturation.
                pm = (
                    torch.randn(NUM_SLICES, M, MIX, generator=genp, device="cuda",
                                dtype=torch.float32)
                    .mul_(0.05)
                    .add_(1.0)
                    .contiguous()
                )
                pm[0] = 1e8
                pm[40] = -1e8
                ps = torch.full(
                    (NUM_SLICES, M), 1e5, device="cuda", dtype=torch.float32)
            elif name in ("comb_dominant_row", "comb_dominant_col", "comb_uniform"):
                pm = torch.zeros_like(pm)
                ps = torch.ones_like(ps)
                rows = torch.arange(HC, device="cuda").view(HC, 1).expand(HC, HC)
                cols = torch.arange(HC, device="cuda").view(1, HC).expand(HC, HC)
                if name == "comb_dominant_row":
                    target = torch.where(rows == 0, 30.0, -30.0)
                elif name == "comb_dominant_col":
                    target = torch.where(cols == 0, 30.0, -30.0)
                else:
                    target = torch.zeros(HC, HC, device="cuda")
                base = base.clone()
                base[2 * HC: 2 * HC + HC * HC] = target.reshape(-1).float()
            partials = (pm, ps)
        else:
            if name == "inputs_zero":
                x = torch.zeros_like(x)
                fn = torch.zeros_like(fn)
            elif name == "inputs_mixed_signs":
                genm = torch.Generator(device="cuda").manual_seed(base_seed + 3000 + idx)
                sign_x = (
                    torch.randint(0, 2, (M, K), generator=genm, device="cuda")
                    .float()
                    .mul_(2.0)
                    .sub_(1.0)
                )
                sign_f = (
                    torch.randint(0, 2, (MIX, K), generator=genm, device="cuda")
                    .float()
                    .mul_(2.0)
                    .sub_(1.0)
                )
                x = (x.float() * sign_x).to(torch.bfloat16).contiguous()
                fn = (fn * sign_f).contiguous()
            elif name == "inputs_large":
                x = x.float().mul_(1e4).to(torch.bfloat16).contiguous()
            partials = produce_partials(prod, x, fn, simt=simt_predicate)

        fixtures.append(
            Fixture(
                name=name,
                tag=tag,
                note=note,
                part_mix=partials[0],
                part_sq=partials[1],
                scale=scale,
                base=base,
                x=x,
                fn=fn,
                partials_override=override,
            )
        )
    return fixtures


# ---------------------------------------------------------------------------
# Production import, argument checks, partial producer
# ---------------------------------------------------------------------------
def import_production():
    import sglang.kernels.ops.layernorm.mhc as mhc

    return mhc


def kernel_param_report(prod):
    """Runtime guard against drift in the production kernel/entry signatures."""
    report = {"kernel": "hc_mix_stats_sinkhorn", "missing": [], "extra": []}
    entry_params = list(inspect.signature(prod.hc_mix_stats_sinkhorn).parameters)
    expected_entry = [
        "x_flat", "hc_fn", "hc_scale", "hc_base", "hc_mult", "sinkhorn_iters",
        "rms_eps", "hc_eps",
    ]
    report["entry_params"] = entry_params
    report["entry_params_match"] = entry_params == expected_entry
    if not report["entry_params_match"]:
        report["missing"] = [p for p in expected_entry if p not in entry_params]
        report["extra"] = [p for p in entry_params if p not in expected_entry]

    wanted = {
        "_hc_mix_reduce_sinkhorn_kernel": [
            "part_mix_ptr", "part_sq_ptr", "scale_ptr", "base_ptr", "pre_ptr",
            "post_ptr", "comb_ptr", "m", "inv_k", "rms_eps", "MIX", "HC",
            "NUM_SLICES", "ITERS", "EPS", "part_mix_residual_ptr",
        ],
        "_hc_mix_stats_partial_kernel": [
            "x_ptr", "w_ptr", "part_mix_ptr", "part_sq_ptr", "M", "K",
            "x_stride_m", "w_stride_n", "MIX", "MIX_PAD", "NUM_SLICES",
            "BLOCK_M", "BLOCK_K", "DOT_PRECISION",
        ],
        "_hc_mix_stats_partial_sm80_simt": ["X", "FN", "PM", "PS"],
    }
    kernel_reports = {}
    for kname, expected in wanted.items():
        fn = getattr(prod, kname, None)
        if fn is None:
            kernel_reports[kname] = {"present": False, "expected": expected}
            report["missing"].append(kname)
            continue
        try:
            params = list(inspect.signature(fn.fn).parameters)
        except Exception:
            params = list(getattr(fn, "arg_names", []) or [])
        kernel_reports[kname] = {
            "present": True,
            "params": params,
            "expected": expected,
            "params_match": params == expected,
        }
        if params != expected:
            report["missing"].append(kname + ":signature")
    report["kernels"] = kernel_reports
    return report


def sm80_simt_predicate_report(prod, x, fn):
    """Read-only mirror of the production SIMT gate, for diagnostics only."""
    pieces = {
        "env_experimental": os.environ.get("SGLANG_DSV41_SM80_EXPERIMENTAL") == "1",
        "env_simt_stats": os.environ.get("SGLANG_DSV41_SM80_MHC_SIMT_STATS") == "1",
        "hip": bool(getattr(prod, "_is_hip", False)),
        "x_cuda": bool(x.is_cuda),
        "x_bf16": x.dtype == torch.bfloat16,
        "fn_fp32": fn.dtype == torch.float32,
        "x_contiguous": bool(x.is_contiguous()),
        "fn_contiguous": bool(fn.is_contiguous()),
        "geometry_ok": (M, K, MIX, HC, NUM_SLICES) == (1, 20480, 24, 4, 80),
        "capability_80": torch.cuda.get_device_capability(x.device) == (8, 0),
    }
    try:
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

        pieces["batch_invariant_mode"] = bool(is_batch_invariant_mode_enabled())
    except Exception as exc:  # pragma: no cover - diagnostics only
        pieces["batch_invariant_mode"] = f"unavailable: {exc!r}"
    active = all(
        value is True
        for key, value in pieces.items()
        if key != "batch_invariant_mode"
    ) and pieces["batch_invariant_mode"] is False
    pieces["simt_active"] = active
    return pieces


def produce_partials(prod, x, fn, simt=False):
    """Run the real partial kernel exactly as production would, return (mix, sq)."""
    part_mix = torch.empty((NUM_SLICES, M, MIX), dtype=torch.float32, device=x.device)
    part_sq = torch.empty((NUM_SLICES, M), dtype=torch.float32, device=x.device)
    if simt:
        prod._hc_mix_stats_partial_sm80_simt[(NUM_SLICES,)](
            x, fn, part_mix, part_sq, num_warps=8, enable_fp_fusion=False
        )
        return part_mix, part_sq
    block_m = prod._block_m_for(M)
    prod._hc_mix_stats_partial_kernel[(triton.cdiv(M, block_m), NUM_SLICES)](
        x,
        fn,
        part_mix,
        part_sq,
        M,
        K,
        x.stride(0),
        fn.stride(0),
        MIX=MIX,
        MIX_PAD=MIX_PAD,
        NUM_SLICES=NUM_SLICES,
        BLOCK_M=block_m,
        BLOCK_K=prod._HC_MIX_BLOCK_K,
        DOT_PRECISION=prod._HC_MIX_DOT_PRECISION,
        num_warps=prod._HC_MIX_NUM_WARPS,
        num_stages=prod._HC_MIX_NUM_STAGES,
    )
    return part_mix, part_sq


# ---------------------------------------------------------------------------
# float64 CPU reference: literal sequential slice sum + literal Sinkhorn loop
# ---------------------------------------------------------------------------
def reference_from_partials(fx, inv_k=INV_K):
    part_mix = fx.part_mix.detach().to("cpu", torch.float64).reshape(NUM_SLICES, MIX)
    part_sq = fx.part_sq.detach().to("cpu", torch.float64).reshape(NUM_SLICES, M)
    scale = fx.scale.detach().to("cpu", torch.float64)
    base = fx.base.detach().to("cpu", torch.float64)
    # Triton passes the Python float 1.0/K as an fp32 kernel argument.
    inv_k32 = float(torch.tensor(inv_k, dtype=torch.float32).to(torch.float64))

    acc = torch.zeros(MIX, dtype=torch.float64)
    sq = torch.zeros(M, dtype=torch.float64)
    for s in range(NUM_SLICES):
        acc = acc + part_mix[s]
        sq = sq + part_sq[s]
    rsqrt = 1.0 / torch.sqrt(sq * inv_k32 + RMS_EPS)

    a_pre = acc[0:HC]
    a_post = acc[HC:2 * HC]
    a_comb = acc[2 * HC:2 * HC + HC * HC].reshape(HC, HC)
    pre = torch.sigmoid(a_pre * rsqrt * scale[0] + base[0:HC]) + HC_EPS
    post = 2.0 * torch.sigmoid(a_post * rsqrt * scale[1] + base[HC:2 * HC])
    comb = a_comb * rsqrt * scale[2] + base[2 * HC:2 * HC + HC * HC].reshape(HC, HC)
    comb = torch.exp(comb - comb.max(dim=1, keepdim=True).values)
    comb = comb / (comb.sum(dim=1, keepdim=True) + HC_EPS)
    comb = comb / (comb.sum(dim=0, keepdim=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(dim=1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(dim=0, keepdim=True) + HC_EPS)
    return {
        "pre": pre.reshape(M, HC),
        "post": post.reshape(M, HC),
        "comb": comb.reshape(M, HC, HC),
    }


def _ordered_fp32(value):
    """Monotone int32 mapping so ULP-ish distance is signed-zero safe."""
    bits = value.contiguous().view(torch.int32).to(torch.int64)
    return torch.where(bits < 0, -(2 ** 31) - bits, bits)


def compare_outputs(actual, expected, *, expected_is_fp64, atol=ATOL, rtol=RTOL):
    """max_abs / scale-relative / ULP-ish comparison, plus the fixed tolerance gate."""
    result = {}
    worst_abs = 0.0
    all_ok = True
    all_finite = True
    for name in OUTPUT_NAMES:
        a = actual[name].detach().to(torch.float32).cpu()
        b_raw = expected[name].detach().cpu()
        b = b_raw.to(torch.float64) if expected_is_fp64 else b_raw.to(torch.float32)
        a64 = a.to(torch.float64)
        b64 = b.to(torch.float64)
        finite = bool(torch.isfinite(a64).all()) and bool(torch.isfinite(b64).all())
        if not finite:
            all_finite = False
            all_ok = False
            result[name] = {
                "finite": False,
                "max_abs": None,
                "max_rel": None,
                "max_ulp": None,
                "tolerance_ok": False,
            }
            continue
        diff = (a64 - b64).abs()
        max_abs = float(diff.max()) if diff.numel() else 0.0
        max_rel = max_abs / max(float(b64.abs().max()) if b64.numel() else 0.0, 1e-300)
        tol_ok = bool((diff <= atol + rtol * b64.abs()).all())
        max_ulp = None
        bitwise = None
        if not expected_is_fp64:
            bitwise = bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
            max_ulp = int((_ordered_fp32(a) - _ordered_fp32(b)).abs().max())
        entry = {
            "finite": True,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "max_ulp": max_ulp,
            "bitwise_equal": bitwise,
            "tolerance_ok": tol_ok,
            "ref_max_abs": float(b64.abs().max()) if b64.numel() else 0.0,
        }
        result[name] = entry
        worst_abs = max(worst_abs, max_abs)
        all_ok = all_ok and tol_ok
    result["worst"] = {
        "max_abs": worst_abs,
        "tolerance_ok": all_ok,
        "finite": all_finite,
    }
    return result


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    cid: str
    kind: str  # "reduce" | "entry" | "attribution"
    variant: str
    num_warps: int
    num_stages: int = -1  # -1 means "let Triton use its default"
    tl_ns: int = 0

    def as_json(self):
        return {
            "config": self.cid,
            "kind": self.kind,
            "variant": self.variant,
            "num_warps": self.num_warps if self.kind != "entry" else None,
            "num_stages": None if self.num_stages == -1 else self.num_stages,
            "tl_range_num_stages": self.tl_ns or None,
        }


def build_configs(warps_filter=None, include_stage_sweep=True):
    cfgs = []
    cfgs.append(Config("prod_nw1_nsdef", "reduce", "prod", 1))
    if include_stage_sweep:
        for ns in (1, 2, 3, 4, 8):
            cfgs.append(Config(f"prod_nw1_ns{ns}", "reduce", "prod", 1, ns))
    for nw in (2, 4):
        cfgs.append(Config(f"prod_nw{nw}", "reduce", "prod", nw))
    cfgs.append(Config("copy_nw1", "reduce", "copy", 1))
    if include_stage_sweep:
        for ns in (1, 2, 4, 8):
            cfgs.append(Config(f"copy_nw1_ns{ns}", "reduce", "copy", 1, ns))
    for ns in (2, 4, 8):
        cfgs.append(Config(f"tlrange_nw1_ns{ns}", "reduce", "tlrange", 1, tl_ns=ns))
    for nw in (1, 2, 4, 8):
        cfgs.append(Config(f"oneload_nw{nw}", "reduce", "oneload", nw))
    for nw in (1, 2, 4, 8):
        cfgs.append(Config(f"split2d_nw{nw}", "reduce", "split2d", nw))
    cfgs.append(Config("sinkhorn_only_nw1", "attribution", "sinkhorn_only", 1))
    cfgs.append(Config("loads_only_nw1", "attribution", "loads_only", 1))
    cfgs.append(Config("loads_only_nw4", "attribution", "loads_only", 4))
    cfgs.append(Config("empty_nw1", "attribution", "empty", 1))
    cfgs.append(Config("entry_prod", "entry", "entry", 0))
    if warps_filter:
        keep = []
        for cfg in cfgs:
            if cfg.kind != "reduce" or cfg.num_warps in warps_filter:
                keep.append(cfg)
        cfgs = keep
    return cfgs


def make_reduce_buffers(fx):
    device = fx.part_mix.device
    return ReduceBuffers(
        part_mix=fx.part_mix,
        part_sq=fx.part_sq,
        scale=fx.scale,
        base=fx.base,
        pre=torch.empty(M, HC, dtype=torch.float32, device=device),
        post=torch.empty(M, HC, dtype=torch.float32, device=device),
        comb=torch.empty(M, HC, HC, dtype=torch.float32, device=device),
        out_mix=torch.empty(M, MIX, dtype=torch.float32, device=device),
        out_sq=torch.empty(M, dtype=torch.float32, device=device),
    )


def launch_config(cfg, prod, fx, rb, dummy):
    """Launch one config once; returns a dict of output tensors."""
    if cfg.kind == "entry":
        pre, post, comb = prod.hc_mix_stats_sinkhorn(
            fx.x, fx.fn, fx.scale, fx.base, HC, ITERS, RMS_EPS, HC_EPS
        )
        return {"pre": pre, "post": post, "comb": comb}
    if cfg.variant == "empty":
        _empty_kernel[(M,)](dummy, num_warps=cfg.num_warps)
        return {}
    if cfg.variant == "sinkhorn_only":
        _reduce_sinkhorn_slice0[(M,)](
            rb.part_mix, rb.part_sq, rb.scale, rb.base, rb.pre, rb.post, rb.comb,
            M, INV_K, RMS_EPS,
            MIX=MIX, HC=HC, ITERS=ITERS, EPS=HC_EPS,
            num_warps=cfg.num_warps,
        )
        return {"pre": rb.pre, "post": rb.post, "comb": rb.comb}
    if cfg.variant == "loads_only":
        _reduce_loads_only[(M,)](
            rb.part_mix, rb.part_sq, rb.out_mix, rb.out_sq, M,
            MIX=MIX, HC=HC, NUM_SLICES=NUM_SLICES,
            num_warps=cfg.num_warps,
            **({} if cfg.num_stages == -1 else {"num_stages": cfg.num_stages}),
        )
        return {"out_mix": rb.out_mix, "out_sq": rb.out_sq}

    if cfg.variant == "prod":
        fn = prod._hc_mix_reduce_sinkhorn_kernel
        extra = {}
    elif cfg.variant == "copy":
        fn = _reduce_sinkhorn_static_copy
        extra = {}
    elif cfg.variant == "tlrange":
        fn = _reduce_sinkhorn_tlrange
        extra = {"NS": cfg.tl_ns}
    elif cfg.variant == "oneload":
        fn = _reduce_sinkhorn_oneload
        extra = {"SBLOCK": SBLOCK, "NPAD": MIX_PAD}
    elif cfg.variant == "split2d":
        fn = _reduce_sinkhorn_split2d
        extra = {"SBLOCK": SBLOCK}
    else:  # pragma: no cover - config table is closed
        raise AssertionError(f"unknown variant {cfg.variant}")

    kwargs = {
        "MIX": MIX,
        "HC": HC,
        "NUM_SLICES": NUM_SLICES,
        "ITERS": ITERS,
        "EPS": HC_EPS,
        "num_warps": cfg.num_warps,
    }
    if cfg.num_stages != -1:
        kwargs["num_stages"] = cfg.num_stages
    kwargs.update(extra)
    fn[(M,)](
        rb.part_mix, rb.part_sq, rb.scale, rb.base, rb.pre, rb.post, rb.comb,
        M, INV_K, RMS_EPS, **kwargs
    )
    return {"pre": rb.pre, "post": rb.post, "comb": rb.comb}


# ---------------------------------------------------------------------------
# Compiled resource introspection (best effort, several Triton layouts)
# ---------------------------------------------------------------------------
def _cache_entries(jit_fn):
    entries = {}
    for attr in ("device_caches", "cache"):
        store = getattr(jit_fn, attr, None)
        if not isinstance(store, dict):
            continue
        for dev, value in store.items():
            inner = value
            if isinstance(value, (tuple, list)):
                inner = value[0] if value else None
            if isinstance(inner, dict):
                for key, compiled in inner.items():
                    entries[(attr, str(dev), str(key))] = compiled
    return entries


def _first_attr(obj, names):
    for name in names:
        if obj is None:
            return None
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def resource_info(compiled):
    md = getattr(compiled, "metadata", None)

    def pick(name):
        value = _first_attr(compiled, (name,))
        return value if value is not None else _first_attr(md, (name,))

    return {
        "name": _first_attr(md, ("name",)) or _first_attr(compiled, ("name",)),
        "num_warps": _first_attr(md, ("num_warps",)),
        "num_stages": _first_attr(md, ("num_stages",)),
        "n_regs": pick("n_regs"),
        "n_spills": pick("n_spills"),
        "shared_bytes": pick("shared"),
    }


def collect_resources(jit_fn, before_keys):
    """Report every compilation that appeared in the cache since before_keys."""
    out = []
    try:
        entries = _cache_entries(jit_fn)
    except Exception as exc:  # pragma: no cover - introspection only
        return {"available": False, "error": repr(exc)}
    for key, compiled in entries.items():
        if key in before_keys:
            continue
        try:
            out.append(resource_info(compiled))
        except Exception as exc:  # pragma: no cover
            out.append({"error": repr(exc)})
    if not out:
        return {"available": False, "error": "no new compiled handle in the JIT cache"}
    return {"available": True, "kernels": out}


# ---------------------------------------------------------------------------
# CUDA graph capture + timing
# ---------------------------------------------------------------------------
def capture_graph(work, stream, warmups=3):
    with torch.cuda.stream(stream):
        for _ in range(warmups):
            work()
    stream.synchronize()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        work()
    torch.cuda.synchronize()
    return graph


def time_graph_amortized(graph, inner, replays, samples, stream):
    for _ in range(replays):
        graph.replay()
    stream.synchronize()
    seen = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
        end.synchronize()
        seen.append(start.elapsed_time(end) * 1000.0 / (inner * replays))
    return seen


def time_graph_single(graph, samples, stream, flush=None, before=None):
    for _ in range(10):
        graph.replay()
    stream.synchronize()
    seen = []
    for _ in range(samples):
        if flush is not None:
            flush()
        if before is not None:
            before()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record()
            graph.replay()
            end.record()
        end.synchronize()
        seen.append(start.elapsed_time(end) * 1000.0)
    return seen


def stats(samples):
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
        "n_samples": len(samples),
    }


def us(entry):
    return None if entry is None else entry["median_us"]


# ---------------------------------------------------------------------------
# Main phases
# ---------------------------------------------------------------------------
def cold_config_selection(hot_rows, limit_primary=3):
    """Cold phase covers entry + baseline + the best few reduce candidates."""
    candidates = [
        row for row in hot_rows
        if row.get("kind") == "reduce"
        and row.get("variant") not in ("empty",)
        and row.get("stats_single") is not None
    ]
    candidates.sort(key=lambda row: (row["stats_single"]["median_us"],
                                     row["stats_amortized"]["median_us"]))
    picked = [row["config"] for row in candidates[:limit_primary]]
    return ["entry_prod", BASELINE_CID] + [cid for cid in picked if cid != BASELINE_CID]


def run(args):
    torch.cuda.init()
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    emit(
        event="env",
        device=torch.cuda.get_device_name(device),
        capability=list(capability),
        cuda_version=torch.version.cuda,
        torch_version=torch.__version__,
        triton_version=getattr(triton, "__version__", "unknown"),
        sm80=capability == (8, 0),
        geometry={
            "M": M, "K": K, "MIX": MIX, "HC": HC, "NUM_SLICES": NUM_SLICES,
            "ITERS": ITERS, "rms_eps": RMS_EPS, "hc_eps": HC_EPS,
        },
        inputs={
            "x_shape": [M, K], "x_dtype": "bfloat16",
            "hc_fn_shape": [MIX, K], "hc_fn_dtype": "float32",
            "hc_scale_shape": [3], "hc_base_shape": [MIX],
        },
    )
    emit(
        event="criteria",
        atol=ATOL,
        rtol=RTOL,
        tolerance_source=TOLERANCE_SOURCE,
        error_budget_factor=ERROR_BUDGET_FACTOR,
        speedup_gate=SPEEDUP_GATE,
        rank_key="single_us",
        l2_flush_mb=L2_FLUSH_MB,
        cold_sets=args.sets,
        baseline_num_stages=(
            "not passed by production (Triton default, 3 in Triton 3.x); the "
            "explicit prod_nw1_ns3 config is the same compile option and is "
            "expected to match prod_nw1_nsdef"),
    )
    emit(event="variants", variants=[
        {"name": "prod", "structure": (
            "unmodified production kernel: tl.static_range over 80 slices with "
            "4 tiny loads (4/4/16 elements + 1 scalar) per slice, num_warps=1")},
        {"name": "copy", "structure": (
            "verbatim copy of prod; harness fidelity check, and the num_stages "
            "sweep that shows kernel-level num_stages cannot pipeline an "
            "already-unrolled static_range")},
        {"name": "tlrange", "structure": (
            "prod body with tl.range(0, NUM_SLICES, num_stages=NS) so the "
            "software pipeliner has a real loop to work with")},
        {"name": "oneload", "structure": (
            "one masked (SBLOCK=128, NPAD=32) 2D load of the whole (80,24) "
            "partial tile, tl.sum(axis=0) tree reduction, masked extraction of "
            "pre/post/comb; changes FP32 association order by construction")},
        {"name": "split2d", "structure": (
            "slice-axis (128,4)/(128,4)/(128,4,4) loads with tl.sum(axis=0) tree "
            "reductions, no masked extraction; also changes association order")},
        {"name": "sinkhorn_only", "structure": (
            "attribution only: identical epilogue, slice 0 load only")},
        {"name": "loads_only", "structure": (
            "attribution only: verbatim 80-slice load loop, epilogue removed")},
        {"name": "empty", "structure": "attribution only: launch-floor probe"},
        {"name": "entry", "structure": (
            "the real hc_mix_stats_sinkhorn entry point, i.e. the partial kernel "
            "plus the reduce kernel; per-launch numbers here cover both")},
    ])

    prod = import_production()
    arg_report = kernel_param_report(prod)
    emit(event="arg_check", **arg_report)
    if arg_report["missing"]:
        emit(event="fatal", reason="production kernel/entry signature drift",
             detail=arg_report)
        return 1

    simt = bool(args.simt_stats)
    if simt:
        os.environ["SGLANG_DSV41_SM80_EXPERIMENTAL"] = "1"
        os.environ["SGLANG_DSV41_SM80_MHC_SIMT_STATS"] = "1"

    x_probe, fn_probe = _seeded_inputs(SEED)
    emit(event="partial_path", mode="simt" if simt else "splitk",
         blocking_env_vars={
             "SGLANG_DSV41_SM80_EXPERIMENTAL": os.environ.get(
                 "SGLANG_DSV41_SM80_EXPERIMENTAL"),
             "SGLANG_DSV41_SM80_MHC_SIMT_STATS": os.environ.get(
                 "SGLANG_DSV41_SM80_MHC_SIMT_STATS"),
         },
         predicate=sm80_simt_predicate_report(prod, x_probe, fn_probe),
         block_k=prod._HC_MIX_BLOCK_K,
         dot_precision=str(prod._HC_MIX_DOT_PRECISION),
         partial_num_warps=prod._HC_MIX_NUM_WARPS,
         partial_num_stages=prod._HC_MIX_NUM_STAGES,
         block_m_for_m1=prod._block_m_for(M),
         num_slices_for_k=prod._num_slices_for(K))

    fixtures = build_fixtures(
        prod, names=args.fixtures, simt_predicate=simt
    )
    if not fixtures:
        emit(event="fatal", reason="no fixtures selected")
        return 1
    for fx in fixtures:
        emit(event="fixture", name=fx.name, tag=fx.tag, note=fx.note,
             partials_override=fx.partials_override,
             part_mix_shape=list(fx.part_mix.shape),
             part_sq_shape=list(fx.part_sq.shape),
             part_mix_contiguous=bool(fx.part_mix.is_contiguous()),
             part_mix_max_abs=float(fx.part_mix.abs().max()),
             part_sq_max_abs=float(fx.part_sq.abs().max()),
             base_max_abs=float(fx.base.abs().max()))

    configs = build_configs(
        warps_filter=args.warps,
        include_stage_sweep=not args.no_stage_sweep,
    )
    for cfg in configs:
        emit(event="config", **cfg.as_json())

    hot_fixture = next(
        (fx for fx in fixtures if fx.name == "prod_realistic"), fixtures[0]
    )
    hot_rb = make_reduce_buffers(hot_fixture)
    dummy = torch.zeros(1, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    bench_stream = torch.cuda.Stream()

    # -------- numerics -----------------------------------------------------
    numerics = {}
    fidelity = {}
    prod_reference_outputs = {}
    if not args.skip_numerics:
        for fx in fixtures:
            ref64 = reference_from_partials(fx)
            emit(event="reference", fixture=fx.name, tag=fx.tag,
                 inv_k32=float(torch.tensor(INV_K, dtype=torch.float32)),
                 pre_abs_max=float(ref64["pre"].abs().max()),
                 post_abs_max=float(ref64["post"].abs().max()),
                 comb_abs_max=float(ref64["comb"].abs().max()))
            rb = make_reduce_buffers(fx)
            baseline_out = None
            for cfg in configs:
                if cfg.kind == "attribution":
                    continue
                if cfg.kind == "entry" and fx.partials_override:
                    continue
                try:
                    out = launch_config(cfg, prod, fx, rb, dummy)
                except Exception as exc:
                    emit(event="numerics_error", config=cfg.cid, fixture=fx.name,
                         error=repr(exc), traceback=traceback.format_exc())
                    numerics[(cfg.cid, fx.name)] = {"error": repr(exc)}
                    continue
                torch.cuda.synchronize()
                if cfg.cid == BASELINE_CID:
                    baseline_out = {k: v.clone() for k, v in out.items()}
                    prod_reference_outputs[fx.name] = baseline_out
                if baseline_out is None:
                    emit(event="numerics_skip", config=cfg.cid, fixture=fx.name,
                         reason="baseline numerics unavailable for this fixture")
                    continue
                vs_ref = compare_outputs(out, ref64, expected_is_fp64=True)
                vs_prod = compare_outputs(out, baseline_out, expected_is_fp64=False)
                row = {
                    "event": "numerics",
                    "config": cfg.cid,
                    "variant": cfg.variant,
                    "num_warps": cfg.num_warps,
                    "num_stages": None if cfg.num_stages == -1 else cfg.num_stages,
                    "fixture": fx.name,
                    "tag": fx.tag,
                    "stress": fx.tag == "stress",
                    "vs_fp64": vs_ref,
                    "vs_production": vs_prod,
                }
                numerics[(cfg.cid, fx.name)] = row
                emit(**row)
                if cfg.variant in ("copy", "entry"):
                    bitwise_all = all(
                        vs_prod[name]["bitwise_equal"] is True
                        for name in OUTPUT_NAMES
                    )
                    fidelity[(cfg.cid, fx.name)] = bitwise_all
                    emit(event="fidelity", config=cfg.cid, fixture=fx.name,
                         kind=cfg.kind,
                         bitwise_equal_to_baseline=bitwise_all,
                         max_abs_vs_baseline=vs_prod["worst"]["max_abs"],
                         note=(
                             "copy/entry must reproduce the unmodified production "
                             "kernel bitwise; anything else means this harness "
                             "(fixtures, launch args, buffers) is not faithful."
                         ))
        failures = []
        for (cid, fname), row in numerics.items():
            if "error" in row:
                continue
            if not row["vs_fp64"]["worst"]["tolerance_ok"]:
                failures.append({
                    "config": cid,
                    "fixture": fname,
                    "tag": row["tag"],
                    "stress": row["stress"],
                    "max_abs_vs_fp64": row["vs_fp64"]["worst"]["max_abs"],
                })
        emit(event="numerics_failures", count=len(failures), failures=failures,
             note=("Rows whose max_abs exceeds ATOL+RTOL*|ref64|. Stress fixtures "
                   "are expected to appear here (the baseline does too) and are "
                   "excluded from the verdict; qualification entries are not."))

    # -------- hot timing ---------------------------------------------------
    hot_rows = []
    for cfg in configs:
        row = {"config": cfg.cid, "kind": cfg.kind, "variant": cfg.variant,
               "num_warps": cfg.num_warps,
               "num_stages": None if cfg.num_stages == -1 else cfg.num_stages}
        try:
            def work(cfg=cfg):
                for _ in range(args.inner):
                    launch_config(cfg, prod, hot_fixture, hot_rb, dummy)

            def work_single(cfg=cfg):
                launch_config(cfg, prod, hot_fixture, hot_rb, dummy)

            if cfg.kind == "entry":
                jit_fn = None
            else:
                jit_fn = {
                    "prod": prod._hc_mix_reduce_sinkhorn_kernel,
                    "copy": _reduce_sinkhorn_static_copy,
                    "tlrange": _reduce_sinkhorn_tlrange,
                    "oneload": _reduce_sinkhorn_oneload,
                    "split2d": _reduce_sinkhorn_split2d,
                    "sinkhorn_only": _reduce_sinkhorn_slice0,
                    "loads_only": _reduce_loads_only,
                    "empty": _empty_kernel,
                }[cfg.variant]

            before_keys = set(_cache_entries(jit_fn).keys()) if jit_fn else set()
            graph = capture_graph(work, stream)
            single_graph = capture_graph(work_single, stream)
            amortized = time_graph_amortized(
                graph, args.inner, args.replays, args.samples, bench_stream
            )
            single = time_graph_single(single_graph, args.samples, bench_stream)
            row["stats_amortized"] = stats(amortized)
            row["stats_single"] = stats(single)
            row["inner_launches"] = args.inner
            if jit_fn is not None:
                row["resources"] = collect_resources(jit_fn, before_keys)
                emit(event="resources", config=cfg.cid,
                     **{k: v for k, v in row["resources"].items()})
        except Exception as exc:
            row["error"] = repr(exc)
            row["traceback"] = traceback.format_exc()
            emit(event="timing_error", config=cfg.cid, error=repr(exc),
                 traceback=traceback.format_exc())
        hot_rows.append(row)
        emit(event="hot", **row)

    by_cid = {row["config"]: row for row in hot_rows}
    baseline = by_cid.get(BASELINE_CID, {})
    base_single = us(baseline.get("stats_single"))
    base_amortized = us(baseline.get("stats_amortized"))

    candidates = [
        row for row in hot_rows
        if row.get("kind") == "reduce" and row.get("stats_single") is not None
    ]
    candidates.sort(key=lambda row: (row["stats_single"]["median_us"],
                                     row["stats_amortized"]["median_us"]))
    winner = candidates[0] if candidates else None

    # attribution: empty floor, Sinkhorn-only, load-loop-only
    empty_single = us(by_cid.get("empty_nw1", {}).get("stats_single"))
    slice0_single = us(by_cid.get("sinkhorn_only_nw1", {}).get("stats_single"))
    loads_single = us(by_cid.get("loads_only_nw1", {}).get("stats_single"))
    emit(
        event="attribution",
        baseline_single_us=base_single,
        empty_single_us=empty_single,
        sinkhorn_only_single_us=slice0_single,
        loads_only_single_us=loads_single,
        launch_floor_us=empty_single,
        sinkhorn_epilogue_us=(
            None if (slice0_single is None or empty_single is None)
            else slice0_single - empty_single
        ),
        extra_79_slice_loads_us=(
            None if (base_single is None or slice0_single is None)
            else base_single - slice0_single
        ),
        note=(
            "launch_floor_us is the single-launch graph overhead with an empty "
            "1-CTA kernel; if baseline_single_us is close to it, the reported "
            "profile number is launch/queue bound and no inner-loop rewrite can "
            "remove it."
        ),
    )

    # -------- cold timing --------------------------------------------------
    # Honest caveat: the reduce kernel touches 7.76 KB per launch, so its own
    # footprint can never exceed a 40 MB L2 by itself.  Two things are therefore
    # done, and both are reported rather than assumed:
    #   * the >=64 sets live at 1 MiB strides inside a >40 MB region, so the
    #     address span (and therefore TLB/sector behaviour) is genuinely large;
    #   * every timed replay is preceded by an explicit 48 MB fill on the same
    #     stream, OUTSIDE the event window, so each timed launch starts cold.
    # The entry-point config additionally has a real L2-exceeding working set
    # (64 distinct 1.97 MB hc_fn matrices = ~126 MB).
    cold_rows = []
    if not args.skip_cold and winner is not None:
        cold_cids = cold_config_selection(hot_rows, limit_primary=args.cold_configs)
        cfg_by_cid = {cfg.cid: cfg for cfg in configs}
        stride_elems = (1 << 20) // 4
        pm_region = torch.empty(
            args.sets * stride_elems, dtype=torch.float32, device="cuda")
        ps_region = torch.empty(
            args.sets * stride_elems, dtype=torch.float32, device="cuda")
        flush_buf = torch.empty(
            (L2_FLUSH_MB * 1024 * 1024) // 4, dtype=torch.float32, device="cuda")
        partial_bytes = NUM_SLICES * MIX * 4 + NUM_SLICES * 4
        emit(
            event="cold_plan",
            configs=cold_cids,
            sets=args.sets,
            l2_flush_mb=L2_FLUSH_MB,
            hc_fn_bytes_per_set=MIX * K * 4,
            partial_bytes_per_set=partial_bytes,
            reduce_region_mb=2.0 * args.sets * stride_elems * 4 / 1024 / 1024,
            reduce_touched_bytes=args.sets * partial_bytes,
            entry_working_set_mb=(
                args.sets * (MIX * K * 4 + partial_bytes) / 1024 / 1024),
            sets_below_requirement=args.sets < DEFAULT_SETS,
            note=(
                "reduce_region_mb is the allocation the sets are spread over "
                "(1 MiB stride); reduce_touched_bytes is what the kernel "
                "actually reads per pass, which is why the explicit "
                "{} MB stream-ordered flush, not the footprint, is what makes "
                "the cold numbers cold."
            ).format(L2_FLUSH_MB),
        )

        for cid in cold_cids:
            cfg = cfg_by_cid.get(cid)
            if cfg is None:
                continue
            row = {"config": cid, "kind": cfg.kind, "variant": cfg.variant,
                   "num_warps": cfg.num_warps, "sets": args.sets,
                   "l2_flushed": True}
            try:
                graphs = []
                keep_alive = []
                for set_idx in range(args.sets):
                    fx = fixtures[set_idx % len(fixtures)]
                    gen = torch.Generator(device="cuda").manual_seed(
                        SEED + 900000 + set_idx)
                    scale = fx.scale
                    base = fx.base
                    if cfg.kind == "entry":
                        # Distinct hc_fn working set per set: the 1.97 MB weight
                        # matrix is a real L2-sized input for the partial kernel.
                        fn = (
                            torch.randn(MIX, K, generator=gen, device="cuda",
                                        dtype=torch.float32)
                            .mul_(0.01)
                            .contiguous()
                        )
                        x = fx.x

                        def work(x=x, fn=fn, scale=scale, base=base):
                            prod.hc_mix_stats_sinkhorn(
                                x, fn, scale, base, HC, ITERS, RMS_EPS, HC_EPS)

                        keep_alive.append(fn)
                        rb = None
                    else:
                        off = set_idx * stride_elems
                        pm = pm_region[
                            off:off + NUM_SLICES * MIX].view(NUM_SLICES, M, MIX)
                        ps = ps_region[off:off + NUM_SLICES].view(NUM_SLICES, M)
                        pm.copy_(
                            torch.randn(NUM_SLICES, M, MIX, generator=gen,
                                        device="cuda", dtype=torch.float32)
                            .mul_(0.16))
                        ps.copy_(
                            torch.rand(NUM_SLICES, M, generator=gen, device="cuda",
                                       dtype=torch.float32)
                            .sub_(0.5).mul_(25.6).add_(256.0))
                        rb = ReduceBuffers(
                            part_mix=pm, part_sq=ps, scale=scale, base=base,
                            pre=torch.empty(M, HC, dtype=torch.float32, device="cuda"),
                            post=torch.empty(M, HC, dtype=torch.float32, device="cuda"),
                            comb=torch.empty(M, HC, HC, dtype=torch.float32, device="cuda"),
                            out_mix=torch.empty(M, MIX, dtype=torch.float32, device="cuda"),
                            out_sq=torch.empty(M, dtype=torch.float32, device="cuda"),
                        )

                        def work(rb=rb):
                            launch_config(cfg, prod, None, rb, dummy)

                        keep_alive.append(rb)
                    graphs.append(
                        capture_graph(work, stream, warmups=args.cold_warmups))

                def flush():
                    # Same stream as the replay, so it is ordered before the
                    # start event and therefore outside the measured window.
                    with torch.cuda.stream(bench_stream):
                        flush_buf.fill_(0.5)

                counter = {"i": 0}
                # Cover every distinct working set at least once, so the reported
                # median is over sets rather than over repeats of one set.
                cold_samples_used = max(args.cold_samples, len(graphs))
                seen = time_graph_single(
                    _FirstOnly(graphs, counter), cold_samples_used, bench_stream,
                    flush=flush,
                )
                row["stats_single"] = stats(seen)
                row["cold_sets_captured"] = len(graphs)
                row["cold_samples_used"] = cold_samples_used
                row["cold_keep_alive"] = len(keep_alive)
            except Exception as exc:
                row["error"] = repr(exc)
                row["traceback"] = traceback.format_exc()
                emit(event="cold_error", config=cid, error=repr(exc),
                     traceback=traceback.format_exc())
            cold_rows.append(row)
            emit(event="cold", **row)

    # -------- reconciliation + verdict -------------------------------------
    per_launch_readings = {
        "per_token_40_launches": PROFILE_MS_PER_TOKEN * 1000.0 / PROFILE_LAYERS,
        "whole_profile_320_launches": PROFILE_MS_PER_TOKEN * 1000.0 / PROFILE_LAUNCHES,
    }
    emit(
        event="profile_reconciliation",
        profile_ms_per_token=PROFILE_MS_PER_TOKEN,
        profile_launches=PROFILE_LAUNCHES,
        profile_layers=PROFILE_LAYERS,
        profile_steps=PROFILE_STEPS,
        implied_us_per_launch=per_launch_readings,
        measured_baseline_single_us=base_single,
        measured_baseline_amortized_us=base_amortized,
        note=(
            "The quoted 15.14 us/launch assumes 0.6057 ms per token covers 40 "
            "launches (one per layer); dividing by the 320 launches actually "
            "profiled gives 1.89 us/launch. Compare measured_baseline_single_us "
            "against both to see which reading the kernel body justifies."
        ),
    )

    def projections(label, baseline_us, winner_us):
        if baseline_us is None or winner_us is None:
            return None
        saved = baseline_us - winner_us
        speedup = baseline_us / winner_us if winner_us else None
        entry = {
            "measurement": label,
            "baseline_us": baseline_us,
            "winner_us": winner_us,
            "saved_us": saved,
            "speedup": speedup,
            "ms_saved_per_40_layers": saved * PROFILE_LAYERS / 1000.0,
            "ms_saved_per_80_calls": saved * 80 / 1000.0,
        }
        if speedup is not None:
            entry["projected_ms_saved_per_token_if_15.14us"] = (
                per_launch_readings["per_token_40_launches"]
                * (1.0 - 1.0 / speedup) * PROFILE_LAYERS / 1000.0
            )
            entry["projected_ms_saved_per_token_if_1.89us"] = (
                per_launch_readings["whole_profile_320_launches"]
                * (1.0 - 1.0 / speedup) * PROFILE_LAYERS / 1000.0
            )
        return entry

    projection_rows = []
    if winner is not None and base_single is not None:
        projection_rows.append(projections(
            "hot_single", base_single, us(winner.get("stats_single"))))
    if winner is not None and base_amortized is not None:
        projection_rows.append(projections(
            "hot_amortized", base_amortized, us(winner.get("stats_amortized"))))
    cold_by_cid = {row["config"]: row for row in cold_rows}
    if winner is not None and us(cold_by_cid.get(BASELINE_CID, {}).get("stats_single")):
        projection_rows.append(projections(
            "cold_single",
            us(cold_by_cid[BASELINE_CID]["stats_single"]),
            us(cold_by_cid.get(winner["config"], {}).get("stats_single")),
        ))
    for row in projection_rows:
        if row is not None:
            emit(event="projection", **row)

    reasons = []
    verdict_ok = True

    if winner is None:
        verdict_ok = False
        reasons.append("no reduce candidate produced a timing")

    # gate 0: harness fidelity.  The verbatim copy and the real entry point must
    # reproduce the unmodified production reduce kernel bitwise; otherwise the
    # fixtures, launch arguments or buffers are not faithful and no other
    # numerical conclusion in this run can be trusted.
    if args.skip_numerics:
        reasons.append("--skip-numerics: numerical gates were not evaluated")
    else:
        for (cid, fname), bitwise in sorted(fidelity.items()):
            if bitwise is not True:
                verdict_ok = False
                reasons.append(
                    f"harness fidelity: {cid}/{fname} is not bitwise-equal to "
                    f"the unmodified production reduce kernel")
        if not fidelity:
            verdict_ok = False
            reasons.append("harness fidelity: no copy/entry comparison was made")

    # A candidate whose kernel never compiled/launched is excluded with an
    # explicit reason rather than treated as a numerical failure -- but it can
    # never win, because it produced no timing either.
    numerically_dead = set()
    if not args.skip_numerics:
        for cfg in configs:
            if cfg.kind != "reduce":
                continue
            rows = [numerics.get((cfg.cid, fx.name)) for fx in fixtures]
            if rows and all(r is None or "error" in r for r in rows):
                numerically_dead.add(cfg.cid)

    # gate 1: finiteness on every fixture for every candidate that ran
    for cfg in configs:
        if cfg.kind != "reduce" or args.skip_numerics:
            continue
        if cfg.cid in numerically_dead:
            reasons.append(
                f"{cfg.cid}: excluded from the numerical gates (failed to "
                f"compile or launch on every fixture)")
            continue
        for fx in fixtures:
            row = numerics.get((cfg.cid, fx.name))
            if row is None or "error" in row:
                verdict_ok = False
                reasons.append(
                    f"{cfg.cid}/{fx.name}: numerics unavailable "
                    f"({(row or {}).get('error', 'missing')})")
                continue
            if not row["vs_fp64"]["worst"]["finite"]:
                verdict_ok = False
                reasons.append(f"{cfg.cid}/{fx.name}: non-finite output")

    # gate 2 + 3: tolerance and error budget on qualification fixtures
    if winner is not None and not args.skip_numerics:
        winner_cid = winner["config"]
        for fx in fixtures:
            if fx.tag != "qualification":
                continue
            wrow = numerics.get((winner_cid, fx.name))
            brow = numerics.get((BASELINE_CID, fx.name))
            if wrow is None or brow is None or "error" in wrow or "error" in brow:
                verdict_ok = False
                reasons.append(f"{winner_cid}/{fx.name}: missing numerics row")
                continue
            if not wrow["vs_fp64"]["worst"]["tolerance_ok"]:
                verdict_ok = False
                reasons.append(
                    f"{winner_cid}/{fx.name}: exceeds ATOL/RTOL vs float64 "
                    f"(max_abs={wrow['vs_fp64']['worst']['max_abs']:.3e})")
            scale = max(
                float(brow["vs_fp64"]["pre"].get("ref_max_abs") or 0.0),
                float(brow["vs_fp64"]["post"].get("ref_max_abs") or 0.0),
                float(brow["vs_fp64"]["comb"].get("ref_max_abs") or 0.0),
            )
            budget = ERROR_BUDGET_FACTOR * max(
                brow["vs_fp64"]["worst"]["max_abs"], ATOL * max(1.0, scale))
            if wrow["vs_fp64"]["worst"]["max_abs"] > budget:
                verdict_ok = False
                reasons.append(
                    f"{winner_cid}/{fx.name}: max_abs "
                    f"{wrow['vs_fp64']['worst']['max_abs']:.3e} exceeds "
                    f"{ERROR_BUDGET_FACTOR}x baseline budget {budget:.3e}")

    # stress-fixture report (never a verdict input, always printed)
    stress_summary = []
    if winner is not None and not args.skip_numerics:
        for fx in fixtures:
            if fx.tag != "stress":
                continue
            wrow = numerics.get((winner["config"], fx.name))
            brow = numerics.get((BASELINE_CID, fx.name))
            if wrow is None or brow is None or "error" in wrow or "error" in brow:
                continue
            w_abs = wrow["vs_fp64"]["worst"]["max_abs"]
            b_abs = brow["vs_fp64"]["worst"]["max_abs"]
            w_vs_prod = wrow["vs_production"]["worst"]["max_abs"]
            summary = {
                "fixture": fx.name,
                "winner": winner["config"],
                "winner_max_abs_vs_fp64": w_abs,
                "baseline_max_abs_vs_fp64": b_abs,
                "winner_max_abs_vs_production": w_vs_prod,
                "ratio_vs_baseline_error": (
                    None if b_abs in (None, 0.0) else w_abs / b_abs),
                "winner_tolerance_ok": wrow["vs_fp64"]["worst"]["tolerance_ok"],
                "baseline_tolerance_ok": brow["vs_fp64"]["worst"]["tolerance_ok"],
            }
            stress_summary.append(summary)
            emit(event="stress_report", **summary,
                 note=(
                     "Stress fixtures are constructed for adversarial FP32 "
                     "association: the baseline itself is expected to fail the "
                     "fixed tolerance here. Excluded from the verdict and "
                     "reported for reproducibility risk assessment only."
                 ))

    speedup_single = (
        base_single / us(winner["stats_single"])
        if winner is not None and base_single and us(winner.get("stats_single"))
        else None
    )
    if speedup_single is not None and speedup_single >= SPEEDUP_GATE:
        recommendation = f"adopt {winner['config']} reduce structure"
    elif speedup_single is not None:
        recommendation = "keep_production_baseline"
    else:
        recommendation = "indeterminate"

    emit(
        event="winner",
        config=None if winner is None else winner["config"],
        variant=None if winner is None else winner["variant"],
        num_warps=None if winner is None else winner["num_warps"],
        num_stages=None if winner is None else winner["num_stages"],
        baseline_single_us=base_single,
        winner_single_us=None if winner is None else us(winner["stats_single"]),
        baseline_amortized_us=base_amortized,
        winner_amortized_us=None if winner is None else us(winner["stats_amortized"]),
        speedup_single=speedup_single,
        speedup_gate=SPEEDUP_GATE,
        recommendation=recommendation,
        verdict_ok=verdict_ok,
        verdict_reasons=reasons,
        stress_summary=stress_summary,
        batch_invariance_note=(
            "prod keeps a fixed slice-order FP32 reduction so a row's bits do "
            "not depend on how work was batched. A tree-reduction variant is "
            "also order-fixed (same SBLOCK/NUM_SLICES) but produces different "
            "bits from the deployed path; stress_summary quantifies that."
        ),
    )

    marker = "PASS" if (verdict_ok and recommendation != "indeterminate") else "FAIL"
    if capability != (8, 0):
        reasons.append(
            f"host capability {capability} is not SM80; numbers are informational")
        emit(event="device_warning", capability=list(capability),
             message="target hardware is A100 SM80 (8, 0)")
    print(f"BENCH_RESULT: {marker}", flush=True)
    if reasons:
        emit(event="verdict_reasons", reasons=reasons)
    return 0 if marker == "PASS" else 1


class _FirstOnly:
    """Small adapter: time_graph_single expects a .replay() target per sample.

    The cold phase rotates across one graph per working set, so the object it
    replays must advance.  The flush/before hooks run outside the timed window.
    """

    def __init__(self, graphs, counter):
        self.graphs = graphs
        self.counter = counter

    def replay(self):
        idx = self.counter["i"] % len(self.graphs)
        self.counter["i"] += 1
        self.graphs[idx].replay()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inner", type=int, default=100,
                        help="kernel launches inside one timing graph")
    parser.add_argument("--replays", type=int, default=20,
                        help="graph replays per amortized timing sample")
    parser.add_argument("--samples", type=int, default=9,
                        help="timing samples per (config, mode)")
    parser.add_argument("--sets", type=int, default=DEFAULT_SETS,
                        help="distinct working sets for the cold benchmark (>=64)")
    parser.add_argument("--cold-samples", type=int, default=15)
    parser.add_argument("--cold-warmups", type=int, default=2)
    parser.add_argument("--cold-configs", type=int, default=3,
                        help="number of hot winners carried into the cold phase")
    parser.add_argument("--fixtures", nargs="*", default=None)
    parser.add_argument("--warps", nargs="*", type=int, default=None)
    parser.add_argument("--skip-numerics", action="store_true")
    parser.add_argument("--skip-cold", action="store_true")
    parser.add_argument("--no-stage-sweep", action="store_true")
    parser.add_argument("--simt-stats", action="store_true",
                        help="enable the SM80 SIMT partial kernel env gate")
    parser.add_argument("--quick", action="store_true",
                        help="small sample counts for a smoke run")
    parser.add_argument("--preflight", action="store_true",
                        help="arg/dtype checks only, then exit")
    parser.add_argument("--list-configs", action="store_true")
    args = parser.parse_args(argv)

    if args.quick:
        args.inner = 20
        args.replays = 5
        args.samples = 5
        args.cold_samples = 5
        args.sets = min(args.sets, 8)

    if args.list_configs:
        for cfg in build_configs(args.warps, not args.no_stage_sweep):
            emit(event="config", **cfg.as_json())
        print("BENCH_RESULT: SKIP (list-configs only)", flush=True)
        return 0

    if not torch.cuda.is_available():
        print("BENCH_RESULT: SKIP (no CUDA device available)", flush=True)
        return 0

    if args.preflight:
        prod = import_production()
        report = kernel_param_report(prod)
        emit(event="arg_check", **report)
        x, fn = _seeded_inputs(SEED)
        emit(event="preflight", x_dtype=str(x.dtype), x_shape=list(x.shape),
             x_contiguous=bool(x.is_contiguous()), fn_dtype=str(fn.dtype),
             fn_shape=list(fn.shape), fn_contiguous=bool(fn.is_contiguous()),
             predicate=sm80_simt_predicate_report(prod, x, fn))
        print("BENCH_RESULT: " + ("PASS" if not report["missing"] else "FAIL"),
              flush=True)
        return 0 if not report["missing"] else 1

    try:
        return run(args)
    except Exception:
        emit(event="fatal", traceback=traceback.format_exc())
        print("BENCH_RESULT: FAIL", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
