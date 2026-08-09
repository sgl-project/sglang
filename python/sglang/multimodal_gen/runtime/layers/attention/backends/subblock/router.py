# Copyright (c) 2026 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
"""Sub-block block-sparse routing for FlashInfer's ``bsa_attn_blk64_fwd``.

Training-free. Runs *before* attention, produces the ``q2k_block_index`` tensor the
64-token block-sparse kernel consumes.

Why sub-blocks
--------------
The usual proxy score for a 64x64 block is ``mean(Q_block) . mean(K_block)``. Averaging
64 keys into one vector throws away exactly the variation that decides which keys a query
wants. Splitting each 64-token block into ``n`` sub-blocks of ``64/n`` tokens, scoring all
sub-block pairs and combining them with a log-sum-exp recovers most of that:

    score(i, j) = log sum_{a<n_q, b<n_k} exp( qbar_{i,a} . kbar_{j,b} * softmax_scale )

which is a direct estimate of the block's un-normalised softmax mass
``sum_{r in i, c in j} exp(q_r . k_c * scale)`` -- the quantity that decides how much
attention mass is lost when the block is skipped.

Measured on 567 (task x denoise-step x layer x head) samples of MiniMax-H3 DiT attention,
mean recall of the retained softmax mass at 0.9 block sparsity:

    n_q=1 n_k=1     .6513     4 u        <- plain avg pooling
    n_q=1 n_k=2     .6598     8 u
    n_q=1 n_k=4     .6655    16 u        <- default
    n_q=1 n_k=8     .6697    32 u
    n_q=8 n_k=1     .6494    32 u        <- splitting Q *alone* is worse than not splitting
    n_q=8 n_k=8     .6793   256 u        <- but splitting both is the best of them
    oracle          .7355     -

(1 u = one ``[S/128, 128] x [128, S/128]`` GEMM = 1/16384 of the dense attention it gates.)

Splitting Q alone loses: a block's mass sums over its query rows, so with one key vector
to score against, the query detail averages out. Splitting both together is a different
proposition -- the log-sum-exp then runs over query-key sub-block *pairs*, and "some part
of this query block wants some part of that key block" is a signal that survives the
averaging. That is the best row in the table, and at sparsity 0.9 it also measured better
end to end than any ``n_q=1`` setting.

``n_q`` defaults to 4 with ``n_k``. The score matrix grows from ``[Gq, Gk*n_k]`` to
``[Gq*n_q, Gk*n_k]``, which costs 0.5% of the denoise time; it is the only change in
this family that has separated from anything else on the output.

One thing not to retry without new evidence
-------------------------------------------
Summing un-normalised sub-block mass over the query axis lets the hottest query sub-block
own a block's score, and the true per-row attention carries a ``1/Z_r`` the raw sum drops,
which over-weights exactly the rows whose attention is spread widest -- the rows that lose
least from dropping any one block. Turning each query sub-block into a distribution over
key blocks first fixes that, and on 1092 real (cell, head, query block) samples at
n_q=n_k=4, sparsity 0.9 it measured better on both proxies: block mass recall .6741 ->
.6779 and relative L2 of the rebuilt attention output .2043 -> .1982, paired t = +8.0 and
-5.7.

It is worse in the pixels, on **0 of 15** prompts, by 0.107 cosine against the dense render
(paired t = -6.4). Single-layer output error, even measured directly, does not order these
estimators the way 40 denoise steps through 50 layers do. Nothing short of an end-to-end
render has predicted this correctly yet -- neither block mass recall nor single-step output
L2.

Usage
-----
    router = SubBlockRouter(n_k=4)
    plan = router.route(q, k, sparsity=0.9)                 # q, k: [B, S, H, D]
    out, _ = bsa_attn_blk64_fwd(q, k, v, plan.index, plan.topk,
                                block_sizes=SubBlockRouter.block_sizes(S, q.device),
                                q2k_block_nums=plan.block_nums)
"""

from __future__ import annotations

import functools
import importlib.util
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch


@functools.lru_cache(maxsize=1)
def load_bsa_attn_blk64_fwd():
    """FlashInfer's 64-block sparse attention entry point.

    ``flashinfer.cute_dsl.sparse.__init__`` also pulls in the blk128 CuTe-DSL backend,
    which breaks in ways blk64 does not care about: it hard-requires the ``quack``
    package, and it tracks a moving ``cutlass.cute`` API (0.6.15.post1 raises
    ``AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'``). blk64 is
    plain CUDA and needs none of it, so whatever the package import trips over we load
    ``bsa_attn_blk64.py`` under a synthetic parent package instead -- same file, same
    kernel. If blk64 itself is broken or absent, that load raises and the caller sees it.
    """
    try:
        from flashinfer.cute_dsl.sparse import bsa_attn_blk64_fwd

        return bsa_attn_blk64_fwd
    except Exception:
        pass
    import flashinfer

    base = Path(flashinfer.__file__).resolve().parent / "cute_dsl" / "sparse"
    pkg = "_flashinfer_sparse_blk64_only"

    def _load(name: str, path: Path, is_pkg: bool):
        spec = importlib.util.spec_from_file_location(
            name,
            path,
            submodule_search_locations=[str(path.parent)] if is_pkg else None,
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    if pkg not in sys.modules:
        parent = types.ModuleType(pkg)
        parent.__path__ = [str(base)]
        sys.modules[pkg] = parent
        _load(f"{pkg}.blk64", base / "blk64" / "__init__.py", True)
    mod = _load(f"{pkg}.bsa_attn_blk64", base / "bsa_attn_blk64.py", False)
    return mod.bsa_attn_blk64_fwd


LOG2E = 1.4426950408889634
BLOCK = 64  # the kernel's block granularity (kSparseBlockSize=64)
VALID_N = (1, 2, 4, 8)  # sub-blocks per 64-token block -> 64 / 32 / 16 / 8 tokens


def _pool(
    x: torch.Tensor, n: int, seq_len: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """``[B, S, H, D] -> [B, H, G*n, D]``: mean of every ``BLOCK//n`` consecutive tokens.

    Sub-blocks that hold at least one real token are averaged over their real tokens only;
    sub-blocks that are entirely padding come back as zeros and are masked by the caller.

    The ragged tail is handled on the *pooled* tensor, never on the token tensor: padding
    ``x`` up to ``G*BLOCK`` first would copy the whole 300+ MB activation to add a few rows.
    """
    b, s, h, d = x.shape
    g = (seq_len + BLOCK - 1) // BLOCK
    sub = BLOCK // n
    n_cells = g * n
    n_full = s // sub  # complete sub-blocks
    cells = torch.zeros(b, n_cells, h, d, device=x.device, dtype=torch.float32)
    if n_full:
        cells[:, :n_full] = (
            x[:, : n_full * sub]
            .view(b, n_full, sub, h, d)
            .sum(dim=2, dtype=torch.float32)
        )
    if s > n_full * sub and n_full < n_cells:  # ragged last sub-block
        cells[:, n_full] = x[:, n_full * sub : s].sum(dim=1, dtype=torch.float32)
    cnt = _valid_counts(seq_len, n, x.device).view(1, n_cells, 1, 1)
    cells /= cnt.clamp_min(1)
    return cells.permute(0, 2, 1, 3).to(out_dtype).contiguous()


def _valid_counts(seq_len: int, n: int, device) -> torch.Tensor:
    """Real (non-padding) token count of each sub-block, ``[G*n]`` float32."""
    g = (seq_len + BLOCK - 1) // BLOCK
    sub = BLOCK // n
    start = torch.arange(g * n, device=device, dtype=torch.float32) * sub
    return (seq_len - start).clamp(0, sub)


@dataclass
class RoutingPlan:
    """What the kernel needs, plus what produced it."""

    index: torch.Tensor  # [B, H, Gq, max_topk] int32
    block_nums: Optional[torch.Tensor]  # [B, H, Gq] int32, or None when topk is uniform
    topk: int  # max_topk (== block_sparse_num when uniform)
    num_blocks: int
    scores: Optional[torch.Tensor] = None  # [B, H, Gq, Gk] float32, only if keep_scores

    @property
    def mean_density(self) -> float:
        """Key blocks kept per query block, as a fraction, averaged over rows.

        Under a per-head budget the count varies by row, so this is the only
        place the budget the router actually spent is observable.
        """
        if self.block_nums is None:
            return self.topk / self.num_blocks
        return float(self.block_nums.float().mean()) / self.num_blocks


class SubBlockRouter:
    """Builds ``q2k_block_index`` from sub-block-pooled Q/K.

    Args:
        n_k: key sub-blocks per 64-token block (1, 2, 4 or 8). 4 is the recommended
            quality/cost point; 1 reproduces plain avg pooling.
        n_q: query sub-blocks. Splitting Q *alone* (n_q>1 with n_k=1) is worse than not
            splitting, which is what the module docstring's table shows; splitting both
            sides together is the best estimator in that table and measured better end to
            end than n_q=1 at sparsity 0.9. Costs n_q times the score matrix.
        reduce: how the n_q*n_k sub-block pairs collapse into one block score.
            ``"lse"`` (log-sum-exp) estimates the block's softmax mass and is the default;
            ``"max"`` is ~1.7x faster in the reduction and costs 0.0 (n=2) to 0.6 (n=8)
            recall points.
        score_dtype: dtype of the pooled GEMM. float32 (TF32 tensor cores). bfloat16 halves
            the GEMM's traffic but measured *slower* end to end, because the reduction then
            has to upcast; the GEMM is under 5% of the router either way.
        score_out_dtype: dtype of the score matrix. **float32, not bfloat16.** bf16 halves
            what the selection reads, but with 8 mantissa bits many blocks tie exactly at the
            threshold, and the fused selector breaks ties by column index -- which
            systematically prefers early key blocks, i.e. one region of the video. Measured
            +3.9% relative L2 at S=96k. torch.topk happens not to have that bias, so bf16 is
            only safe with ``select="torch"``.
        select: ``"fused"`` runs an exact-enough per-row top-k in one pass over the score
            matrix (1.4-1.6x faster than ``torch.topk``, which makes several passes).
        select_iters: threshold-search steps in the fused selector. ``None`` (default) picks
            them from ``log2(G/k)`` via :func:`.kernels.topk_iters` -- 16 at the usual
            operating points, more past sparsity 0.9 where a flat 16 silently costs +3.2%
            relative L2. Pass an int to pin it.
        backend: ``"fused"`` (default) runs the pooling and the GEMM+segmented-log-sum-exp
            as two Triton kernels, so the ``[B,H,Gq,Gk*n_k]`` intermediate never exists.
            ``"torch"`` is the reference path. Fused is 5-17x faster on the score itself.
        workspace_bytes: cap on the live score matrix (torch backend only). Heads are tiled only if
            ``[B, H, Gq, Gk*n_k]`` fp32 would exceed it -- tiling a matrix that already
            fits measured consistently slower.

    Structural block reservation (an attention sink, or forcing the diagonal
    j == i) was measured on 200 real H3 attention cells and is deliberately
    absent: at a fixed budget the diagonal changed relative L2 by 0.2% and the
    sink only helped in DiT layers 2-32, which did not survive to the pixels.
    """

    def __init__(
        self,
        n_k: int = 4,
        n_q: int = 1,
        reduce: str = "lse",
        backend: str = "fused",
        score_out_dtype: torch.dtype = torch.float32,
        select: str = "fused",
        select_iters: int | None = None,
        score_dtype: torch.dtype = torch.float32,
        workspace_bytes: int = 8 << 30,
    ) -> None:
        if n_k not in VALID_N or n_q not in VALID_N:
            raise ValueError(
                f"n_q/n_k must be one of {VALID_N}, got n_q={n_q}, n_k={n_k}"
            )
        if reduce not in ("lse", "max"):
            raise ValueError("reduce must be 'lse' or 'max'")
        self.n_k, self.n_q = n_k, n_q
        self.reduce = reduce
        self.backend = backend
        self.score_out_dtype = score_out_dtype
        self.select = select
        self.select_iters = select_iters
        self.score_dtype = score_dtype
        self.workspace_bytes = workspace_bytes
        self.last_topk = 0

    # ------------------------------------------------------------------ fused path
    @torch.no_grad()
    def _scores_fused(self, q, k, gq, gk, softmax_scale):
        """Two Triton kernels: pool, then GEMM + segmented log-sum-exp in registers.

        ``softmax_scale*log2e`` is folded into Q so the kernel can use the exp2/log2
        hardware instructions, but it multiplies by ln 2 on the way out: scores come back in
        **natural-log units, the same as the torch path**. Selection is a top-k, which any
        monotone rescale leaves alone, so this only matters to a caller that reads the
        score's magnitude -- via ``keep_scores`` -- rather than its order.
        """
        from .kernels import fused_pool, fused_scores

        b, s, h, d = q.shape
        sk = k.shape[1]
        L = b * h
        nq, nk = self.n_q, self.n_k
        sub_q, sub_k = BLOCK // nq, BLOCK // nk
        q_cells, k_cells = gq * nq, gk * nk
        qp = torch.empty(L, q_cells, d, device=q.device, dtype=torch.bfloat16)
        kp = torch.empty(L, k_cells, d, device=k.device, dtype=torch.bfloat16)
        fused_pool(q, q_cells, sub_q, qp, scale=softmax_scale * LOG2E)
        fused_pool(k, k_cells, sub_k, kp)

        out = torch.empty(L, gq, gk, device=q.device, dtype=self.score_out_dtype)
        # sub-cells holding >= 1 real token; the rest pooled to zero
        fused_scores(
            qp,
            kp,
            out,
            nk,
            -(-sk // sub_k),
            n_q=nq,
            m_valid=-(-s // sub_q),
        )
        return out.view(b, h, gq, gk)

    # ------------------------------------------------------------------ scores
    @torch.no_grad()
    def scores(
        self, q: torch.Tensor, k: torch.Tensor, softmax_scale: Optional[float] = None
    ) -> torch.Tensor:
        """``[B, S, H, D] -> [B, H, Gq, Gk]`` block scores (log-space, higher = keep)."""
        b, s, h, d = q.shape
        sk = k.shape[1]
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(d)
        gq, gk = (s + BLOCK - 1) // BLOCK, (sk + BLOCK - 1) // BLOCK
        nq, nk = self.n_q, self.n_k
        if self.backend == "fused":
            return self._scores_fused(q, k, gq, gk, softmax_scale)

        qp = _pool(q, nq, s, self.score_dtype).mul_(softmax_scale)  # [B,H,Gq*nq,D]
        kp = _pool(k, nk, sk, self.score_dtype)  # [B,H,Gk*nk,D]

        # Sub-blocks that hold no real token must drop out of the reduction entirely.
        # Leaving them in would contribute exp(0)=1 per phantom pair, which both inflates
        # the score and flattens the differences the ranking depends on.
        kmask = _valid_counts(sk, nk, k.device) > 0  # [Gk*nk]
        qmask = _valid_counts(s, nq, q.device) > 0  # [Gq*nq]

        # Head tiling only when the score matrix would not fit the workspace. Measured:
        # splitting a fitting matrix is always slower (more launches, more allocations) --
        # this loop runs once in the common case.
        per_head = b * gq * gk * nq * nk * 4
        hc = max(1, min(h, self.workspace_bytes // max(per_head, 1)))
        out = torch.empty(b, h, gq, gk, device=q.device, dtype=torch.float32)
        for h0 in range(0, h, hc):
            h1 = min(h0 + hc, h)
            z = torch.matmul(qp[:, h0:h1], kp[:, h0:h1].transpose(-2, -1))
            if nq == 1:
                # already [b,hc,Gq,Gk*nk]; the (Gk,nk) split is contiguous, no permute
                zz = z.view(b, h1 - h0, gq, gk, nk)
                mask = kmask.view(1, 1, 1, gk, nk)
            else:
                zz = (
                    z.view(b, h1 - h0, gq, nq, gk, nk)
                    .permute(0, 1, 2, 4, 3, 5)
                    .reshape(b, h1 - h0, gq, gk, nq * nk)
                )
                mask = (
                    qmask.view(1, 1, gq, nq)[..., None, :, None]
                    & kmask.view(1, 1, 1, gk, 1, nk)
                ).reshape(1, 1, gq, gk, nq * nk)
            if self.reduce == "max":
                out[:, h0:h1] = zz.masked_fill(~mask, torch.finfo(zz.dtype).min).amax(
                    -1
                )
            else:
                out[:, h0:h1] = torch.logsumexp(
                    zz.float().masked_fill(~mask, -float("inf")), dim=-1
                )
            del z, zz
        # a fully-padded key block is never selectable; -inf would poison topk ordering
        return torch.nan_to_num(out, neginf=-3.0e38)

    # ------------------------------------------------------------------ routing
    @torch.no_grad()
    def route(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        sparsity: Optional[float] = None,
        topk: Optional[Union[int, torch.Tensor]] = None,
        softmax_scale: Optional[float] = None,
        keep_scores: bool = False,
    ) -> RoutingPlan:
        """Select KV blocks per (batch, head, query block).

        Exactly one of ``sparsity`` (fraction of blocks dropped) or ``topk`` (blocks kept)
        must be given. ``topk`` may be an int, or a ``[H]`` int tensor for a per-head budget
        -- the per-head form is worth more than any estimator upgrade: at a fixed mean
        sparsity it lifts the 5th-percentile recall from .52 to .90, because diffuse heads
        stop being starved while sparse heads stop being over-served.
        """
        b, s, h, d = q.shape
        gk = (k.shape[1] + BLOCK - 1) // BLOCK
        sc = self.scores(q, k, softmax_scale)  # [B,H,Gq,Gk]
        gq = sc.shape[2]

        if (sparsity is None) == (topk is None):
            raise ValueError("pass exactly one of sparsity= or topk=")
        if sparsity is not None:
            kk = max(1, min(gk, math.ceil((1.0 - sparsity) * gk)))
            topk = kk

        if torch.is_tensor(topk):  # per-head budget
            per_head = topk.to(device=q.device, dtype=torch.int32)
            if per_head.dim() == 1:  # [H] -> [H, Gq]
                per_head = per_head.view(h, 1).expand(h, gq)
            per_head = per_head.clamp(1, gk).contiguous()  # [H, Gq]
            kmax = int(per_head.max().item())
            # sorted=True is required here: each head keeps only the first `per_head[h]`
            # entries, so they have to be the *best* ones, not an arbitrary subset of the
            # top-kmax.
            idx = sc.topk(kmax, dim=-1).indices.to(torch.int32)  # [B,H,Gq,kmax]
            # pad each head's row out to kmax by repeating its last real block: the kernel
            # would otherwise attend to whatever integer happens to sit there
            ar = torch.arange(kmax, device=q.device).view(1, 1, 1, -1)
            lim = per_head.view(1, h, gq, 1)
            last = idx.gather(-1, (lim - 1).expand(b, h, gq, 1).long())
            idx = torch.where(ar < lim, idx, last)
            nums = per_head.view(1, h, gq).expand(b, h, gq).contiguous()
            plan = RoutingPlan(idx.contiguous(), nums, kmax, gk)
        else:
            kk = int(topk)
            if self.select == "fused" and sc.is_contiguous():
                from .kernels import fused_topk

                # one pass over the score matrix instead of torch.topk's several
                idx = fused_topk(sc.reshape(-1, gk), kk, iters=self.select_iters).view(
                    b, h, gq, kk
                )
            else:
                # unsorted: the kernel accepts any order and sorting is pure overhead
                idx = sc.topk(kk, dim=-1, sorted=False).indices.to(torch.int32)
            plan = RoutingPlan(idx.contiguous(), None, kk, gk)

        self.last_topk = plan.topk
        if keep_scores:
            plan.scores = sc
        return plan

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def block_sizes(seq_len: int, device) -> torch.Tensor:
        """Real token count per 64-block, for the kernel's tail masking."""
        g = (seq_len + BLOCK - 1) // BLOCK
        start = torch.arange(g, device=device, dtype=torch.int32) * BLOCK
        return (seq_len - start).clamp(0, BLOCK).to(torch.int32)

    @staticmethod
    def cost_units(n_q: int, n_k: int) -> float:
        """Scoring cost in units of 1/16384 of the dense attention being gated."""
        return 4.0 * n_q * n_k
