# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""K31HT_SOFTMAX_SPLIT (default off): temperature softmax over the vocab, split
into vocab blocks so a large batch of rows fills the GPU.

Off (unset / '0'): SoftmaxTemp.execute is unchanged (flashinfer.sampling.softmax).
On ('1'): two launches, each on a (rows, ceil(V / 4096)) grid:
  1. per vocab block b of row r: m_b = max(x / T), d_b = sum(exp(x / T - m_b));
  2. per vocab block: M = max_b m_b, D = sum_b d_b * exp(m_b - M), with the
     partials combined in block order (every block computes the same M and D),
     then writes exp(x / T - M) / D.
FP32 throughout. Not bitwise to flashinfer's softmax; it is the same
distribution within FP32 rounding (FP64-reference error no worse than the
flashinfer kernel's). flashinfer runs one CTA per row once there are more than
128 rows, which leaves most SMs idle at HT verify batch sizes.
Shapes it does not cover (non-2D logits, vocab > 64 x 4096, other dtypes, 0 rows)
return None, and the caller runs the stock path; each such case is logged once
as 'K31HT_SOFTMAX_SPLIT NOT ENGAGED: <reason>'.

K31HT_SMX_LAZY (default off; needs K31HT_SOFTMAX_SPLIT=1): the chain accept reads the target probabilities only at
(row step - 1, drafted token of step) for its walk and, after it, one full row per request (the residual / bonus row).
With the knob, the verify accept (dspark_accept._accept_sampling_core, lazy=True) gets the [rows, V] buffer UNFILLED
after launch 1, plus a pending _LazyProbs registered under the buffer's storage; chain_speculative_sampling_triton
takes it and fills, with the write kernel's own code, (a) the drafted-token entries before the accept walk and (b)
each request's accept row after it (the SAMP_SPLIT / SPARSEQ paths), or (c) every row before any other sampler (the
stock kernel reads every row). Every value a sampler reads is computed by the same expressions as launch 2 writes
it, so the sampled tokens are bitwise those of the full form; launch 2 itself is skipped.
"""

import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_K31HT_SMX_BLOCK_V = 4096
_K31HT_SMX_MAX_BLOCKS = 64
_K31HT_SMX_LOGGED = set()


def k31ht_softmax_split_requested() -> bool:
    """K31HT_SOFTMAX_SPLIT: unset / '0' = the stock softmax, '1' = the split form."""
    raw = os.environ.get("K31HT_SOFTMAX_SPLIT", "0").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"K31HT_SOFTMAX_SPLIT={raw!r}: expected 0 or 1")


def k31ht_smx_lazy_requested() -> bool:
    """K31HT_SMX_LAZY: unset / '' / '0' off, '1' on (needs K31HT_SOFTMAX_SPLIT=1), anything else raises."""
    raw = os.environ.get("K31HT_SMX_LAZY", "")
    if raw in ("", "0"):
        return False
    if raw != "1":
        raise ValueError(f"K31HT_SMX_LAZY={raw!r}: expected unset, '', '0' or '1'")
    if not k31ht_softmax_split_requested():
        raise ValueError("K31HT_SMX_LAZY=1 needs K31HT_SOFTMAX_SPLIT=1")
    return True


def _k31ht_smx_log_once(key, msg, *args):
    if key not in _K31HT_SMX_LOGGED:
        _K31HT_SMX_LOGGED.add(key)
        logger.info(msg, *args)


# K31HT_SOFTMAX_SPLIT kernels begin
@triton.jit
def _k31ht_smx_part_kernel(
    X,
    T,
    PM,
    PD,
    x_stride,
    rows_per_request,
    V: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    msk = offs < V
    inv = 1.0 / tl.load(T + r // rows_per_request)
    x = (
        tl.load(X + r.to(tl.int64) * x_stride + offs, mask=msk, other=float("-inf")).to(
            tl.float32
        )
        * inv
    )
    m = tl.max(x, axis=0)
    # A block whose entries are all -inf (masked vocab) has m = -inf; exp(x - m) would be NaN there.
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    e = tl.where(msk, tl.exp(x - m_safe), 0.0)
    tl.store(PM + r * NB + b, m)
    tl.store(PD + r * NB + b, tl.sum(e, axis=0))


@triton.jit
def _k31ht_smx_write_kernel(
    X,
    T,
    PM,
    PD,
    O,
    x_stride,
    o_stride,
    rows_per_request,
    V: tl.constexpr,
    NB: tl.constexpr,
    NBP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    b = tl.program_id(1)
    j = tl.arange(0, NBP)
    jm = j < NB
    mp = tl.load(PM + r * NB + j, mask=jm, other=float("-inf"))
    dp = tl.load(PD + r * NB + j, mask=jm, other=0.0)
    M = tl.max(mp, axis=0)
    D = tl.sum(tl.where(jm, dp * tl.exp(mp - M), 0.0), axis=0)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    msk = offs < V
    inv = 1.0 / tl.load(T + r // rows_per_request)
    x = (
        tl.load(X + r.to(tl.int64) * x_stride + offs, mask=msk, other=float("-inf")).to(
            tl.float32
        )
        * inv
    )
    tl.store(O + r.to(tl.int64) * o_stride + offs, tl.exp(x - M) / D, mask=msk)


# K31HT_SMX_LAZY kernels: the write kernel's M / D / p code, at other (row, vocab) positions.
@triton.jit
def _k31ht_smx_rows_kernel(
    X,
    T,
    PM,
    PD,
    O,
    State,
    x_stride,
    o_stride,
    rows_per_request,
    V: tl.constexpr,
    NB: tl.constexpr,
    NBP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One request's accept row (State[req, 0] = the accept walk's cur_prob_row), vocab block b.
    q = tl.program_id(0)
    b = tl.program_id(1)
    r = q * rows_per_request + tl.load(State + q * 4 + 0).to(tl.int32)
    j = tl.arange(0, NBP)
    jm = j < NB
    mp = tl.load(PM + r * NB + j, mask=jm, other=float("-inf"))
    dp = tl.load(PD + r * NB + j, mask=jm, other=0.0)
    M = tl.max(mp, axis=0)
    D = tl.sum(tl.where(jm, dp * tl.exp(mp - M), 0.0), axis=0)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    msk = offs < V
    inv = 1.0 / tl.load(T + r // rows_per_request)
    x = (
        tl.load(X + r.to(tl.int64) * x_stride + offs, mask=msk, other=float("-inf")).to(
            tl.float32
        )
        * inv
    )
    tl.store(O + r.to(tl.int64) * o_stride + offs, tl.exp(x - M) / D, mask=msk)


@triton.jit
def _k31ht_smx_draft_kernel(
    X,
    T,
    PM,
    PD,
    O,
    Cand,
    x_stride,
    o_stride,
    cand_b,
    cand_s,
    rows_per_request,
    NUM_SLOTS: tl.constexpr,
    V: tl.constexpr,
    NB: tl.constexpr,
    NBP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Row r = q * rows_per_request + s - 1 at the drafted token Cand[q, s], one program per (q, s), s = 1 .. NUM_SLOTS - 1: computed as
    # a BLOCK-wide vector (the write kernel's code over the block holding the token) and stored only at the token's lane.
    q = tl.program_id(0)
    s = tl.program_id(1) + 1
    r = q * rows_per_request + (s - 1)
    tok = tl.load(Cand + q * cand_b + s * cand_s).to(tl.int32)
    b = tok // BLOCK
    j = tl.arange(0, NBP)
    jm = j < NB
    mp = tl.load(PM + r * NB + j, mask=jm, other=float("-inf"))
    dp = tl.load(PD + r * NB + j, mask=jm, other=0.0)
    M = tl.max(mp, axis=0)
    D = tl.sum(tl.where(jm, dp * tl.exp(mp - M), 0.0), axis=0)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    msk = offs < V
    inv = 1.0 / tl.load(T + r // rows_per_request)
    x = (
        tl.load(X + r.to(tl.int64) * x_stride + offs, mask=msk, other=float("-inf")).to(
            tl.float32
        )
        * inv
    )
    tl.store(O + r.to(tl.int64) * o_stride + offs, tl.exp(x - M) / D, mask=msk & (offs == tok))


# K31HT_SOFTMAX_SPLIT kernels end

_K31HT_SMX_PENDING = {}  # storage data_ptr of an unfilled output -> _LazyProbs (K31HT_SMX_LAZY)


class _LazyProbs:
    """An unfilled K31HT_SOFTMAX_SPLIT output and the inputs to fill any part of it with launch 2's code."""

    def __init__(self, logits, temps, pm, pd, out, rows_per_request, nb, nbp):
        self.logits, self.temps, self.pm, self.pd, self.out = logits, temps, pm, pd, out
        self.rpr, self.nb, self.nbp = rows_per_request, nb, nbp
        self.rows_done = False

    def fill_draft(self, candidates: torch.Tensor) -> None:
        """Entries (q * rpr + s - 1, candidates[q, s]) for s = 1 .. slots - 1: what the accept walk reads."""
        bs, slots = candidates.shape
        if slots != self.rpr or bs * slots != self.out.shape[0]:
            # Not the chain layout this fill assumes: fill every row instead (logged once).
            _k31ht_smx_log_once(("lazy_shape", slots, self.rpr), "K31HT_SMX_LAZY: candidates %s vs %d rows per request: all rows filled",
                                tuple(candidates.shape), self.rpr)
            self.fill_all()
            self.rows_done = True
            return
        _k31ht_smx_draft_kernel[(bs, slots - 1)](
            self.logits, self.temps, self.pm, self.pd, self.out, candidates,
            self.logits.stride(0), self.out.stride(0), candidates.stride(0), candidates.stride(1), self.rpr,
            NUM_SLOTS=slots, V=self.out.shape[1], NB=self.nb, NBP=self.nbp, BLOCK=_K31HT_SMX_BLOCK_V, num_warps=8,
        )

    def fill_rows(self, state: torch.Tensor) -> None:
        """Each request's accept row (state[q, 0]): what the residual / bonus kernels read."""
        if self.rows_done:
            return
        bs = self.out.shape[0] // self.rpr
        _k31ht_smx_rows_kernel[(bs, self.nb)](
            self.logits, self.temps, self.pm, self.pd, self.out, state,
            self.logits.stride(0), self.out.stride(0), self.rpr,
            V=self.out.shape[1], NB=self.nb, NBP=self.nbp, BLOCK=_K31HT_SMX_BLOCK_V, num_warps=8,
        )

    def fill_all(self, why: str = "") -> None:
        """Every row (launch 2 as K31HT_SOFTMAX_SPLIT runs it): for a sampler that reads every row."""
        if why:
            _k31ht_smx_log_once(("lazy_all", why), "K31HT_SMX_LAZY DECLINED (%s): all rows filled", why)
        num_rows, vocab = self.out.shape
        _k31ht_smx_write_kernel[(num_rows, self.nb)](
            self.logits, self.temps, self.pm, self.pd, self.out, self.logits.stride(0), self.out.stride(0), self.rpr,
            V=vocab, NB=self.nb, NBP=self.nbp, BLOCK=_K31HT_SMX_BLOCK_V, num_warps=8,
        )


def k31ht_smx_lazy_take(target_probs: torch.Tensor):
    """The pending _LazyProbs whose output shares target_probs' storage (removed), else None."""
    if not _K31HT_SMX_PENDING:
        return None
    return _K31HT_SMX_PENDING.pop(target_probs.untyped_storage().data_ptr(), None)


def k31ht_softmax_split(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
    lazy: bool = False,
) -> Optional[torch.Tensor]:
    """softmax(logits / T) as fp32 [rows, V]; T = temperatures[row // rows_per_request].

    Returns None (caller runs the stock softmax) for shapes it does not cover. lazy=True (K31HT_SMX_LAZY; only the
    verify accept passes it): launch 2 is skipped, the returned buffer is UNFILLED and a _LazyProbs is registered for
    chain_speculative_sampling_triton (k31ht_smx_lazy_take) to fill what its sampler reads.
    """
    reason = None
    if logits.dim() != 2:
        reason = f"logits dim {logits.dim()}"
    elif logits.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        reason = f"dtype {logits.dtype}"
    elif logits.shape[0] == 0:
        reason = "0 rows"
    elif triton.cdiv(logits.shape[-1], _K31HT_SMX_BLOCK_V) > _K31HT_SMX_MAX_BLOCKS:
        reason = f"vocab {logits.shape[-1]} > {_K31HT_SMX_MAX_BLOCKS * _K31HT_SMX_BLOCK_V}"
    if reason is not None:
        _k31ht_smx_log_once(
            ("not_engaged", reason), "K31HT_SOFTMAX_SPLIT NOT ENGAGED: %s", reason
        )
        return None
    num_rows, vocab = logits.shape
    bs = num_rows // rows_per_request
    assert bs * rows_per_request == num_rows, (
        f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    )
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    temps = temperatures.reshape(bs).to(torch.float32).contiguous()
    nb = triton.cdiv(vocab, _K31HT_SMX_BLOCK_V)
    nbp = max(triton.next_power_of_2(nb), 2)  # >= 2: a one-element arange is avoided for vocab <= 4096
    device = logits.device
    pm = torch.empty((num_rows, nb), dtype=torch.float32, device=device)
    pd = torch.empty((num_rows, nb), dtype=torch.float32, device=device)
    out = torch.empty((num_rows, vocab), dtype=torch.float32, device=device)
    _k31ht_smx_part_kernel[(num_rows, nb)](
        logits,
        temps,
        pm,
        pd,
        logits.stride(0),
        rows_per_request,
        V=vocab,
        NB=nb,
        BLOCK=_K31HT_SMX_BLOCK_V,
        num_warps=8,
    )
    if lazy:
        if _K31HT_SMX_PENDING:
            # An earlier lazy output was never taken by a sampler: nothing read it; drop it (logged once).
            _k31ht_smx_log_once(("lazy_dropped",), "K31HT_SMX_LAZY: an untaken lazy output was dropped (no sampler read it)")
            _K31HT_SMX_PENDING.clear()
        _K31HT_SMX_PENDING[out.untyped_storage().data_ptr()] = _LazyProbs(logits, temps, pm, pd, out, rows_per_request, nb, nbp)
        _k31ht_smx_log_once(("lazy", vocab), "K31HT_SMX_LAZY ENGAGED: target probs filled on demand (vocab %d, blocks %d)", vocab, nb)
        return out
    _k31ht_smx_write_kernel[(num_rows, nb)](
        logits,
        temps,
        pm,
        pd,
        out,
        logits.stride(0),
        out.stride(0),
        rows_per_request,
        V=vocab,
        NB=nb,
        NBP=nbp,
        BLOCK=_K31HT_SMX_BLOCK_V,
        num_warps=8,
    )
    _k31ht_smx_log_once(
        ("engaged", vocab),
        "K31HT_SOFTMAX_SPLIT engaged (vocab %d, blocks %d)",
        vocab,
        nb,
    )
    return out
