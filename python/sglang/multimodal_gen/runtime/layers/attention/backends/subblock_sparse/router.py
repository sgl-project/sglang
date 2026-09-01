# SPDX-License-Identifier: Apache-2.0
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
    n_q=1 n_k=4     .6655    16 u
    n_q=1 n_k=8     .6697    32 u
    n_q=8 n_k=1     .6494    32 u        <- splitting Q *alone* is worse than not splitting
    n_q=8 n_k=8     .6793   256 u        <- but splitting both is the best of them
    oracle          .7355     -

(1 u = one ``[S/128, 128] x [128, S/128]`` GEMM = 1/16384 of the dense attention it gates.)

Splitting Q alone loses: a block's mass sums over its query rows, so with one key vector
to score against, the query detail averages out. Splitting both together is a different
proposition -- the log-sum-exp then runs over query-key sub-block *pairs*, and "some part
of this query block wants some part of that key block" is a signal that survives the
averaging. That is the best row in the table, and it is the only estimator change in this
family that has separated from anything else end to end. ``n_q = n_k = 4`` ships.

Not worth retrying without new evidence
---------------------------------------
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

Worth trying, not yet exposed
-----------------------------
A per-head budget beats any estimator upgrade measured here: at a fixed mean sparsity,
spending more blocks on diffuse heads and fewer on peaked ones lifts 5th-percentile mass
recall from .52 to .90. It needs a rule for setting the per-head split, which nothing in
the pipeline currently produces.

Usage
-----
    router = SubBlockRouter(n_k=4, n_q=4)
    plan = router.route(q, k, sparsity=0.8, softmax_scale=d**-0.5)   # q, k: [B, S, H, D]
    out, _ = bsa_attn_blk64_fwd(q, k, v, plan.index, plan.topk,
                                block_sizes=SubBlockRouter.block_sizes(S, q.device),
                                q2k_block_nums=None)
"""

from __future__ import annotations

import functools
import importlib.util
import math
import sys
import types
from pathlib import Path

import msgspec
import torch

from .kernels import fused_pool, fused_scores, fused_topk


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
BUDGET_GRANULARITY = 8  # blocks per query row the kernel bills in, padding to fit
VALID_N = (1, 2, 4, 8)  # sub-blocks per 64-token block -> 64 / 32 / 16 / 8 tokens


def _snap_up_to_8(topk: int, num_blocks: int) -> int:
    """Round a block budget up to what the kernel is going to charge for anyway.

    ``bsa_attn_blk64_fwd`` pads each query row's block count up to a multiple of
    ``BUDGET_GRANULARITY`` with phantom slots that repeat the last real block and
    are then masked out of the softmax. Asking for 148 blocks therefore costs
    exactly what 152 costs, with four of the slots computed and thrown away.
    Measured at S=37.7k on B200, 2 prompts in one session: 152 blocks take
    16.055 s against 16.061 s for 148, and 120 take 15.490 s against 15.496 s
    for 118 -- free, inside the noise. So take the blocks.

    The consequence for the caller is that ``sparsity`` is an upper bound rather
    than an exact figure: 0.75 of 590 blocks becomes 152 kept, 0.7424 dropped.
    """
    return min(num_blocks, max(1, -(-topk // BUDGET_GRANULARITY)) * BUDGET_GRANULARITY)


class RoutingPlan(msgspec.Struct, frozen=True):
    """What the kernel needs, plus the budget that produced it."""

    index: torch.Tensor  # [B, H, Gq, topk] int32
    topk: int  # key blocks kept per query block
    num_blocks: int  # key blocks available

    @property
    def density(self) -> float:
        return self.topk / self.num_blocks


class SubBlockRouter:
    """Builds ``q2k_block_index`` from sub-block-pooled Q/K.

    Args:
        n_k: key sub-blocks per 64-token block (1, 2, 4 or 8). 1 reproduces plain avg
            pooling; 4 is the quality/cost point the recall table above lands on.
        n_q: query sub-blocks, same values. Splitting Q *alone* (n_q>1 with n_k=1) is
            worse than not splitting; splitting both sides together is what the default
            does. Costs n_q times the score matrix, 0.5% of the denoise time.

    Structural block reservation (an attention sink, or forcing the diagonal j == i) was
    measured on 200 real H3 attention cells and is deliberately absent: at a fixed budget
    the diagonal changed relative L2 by 0.2% and the sink only helped in DiT layers 2-32,
    which did not survive to the pixels.
    """

    def __init__(self, n_k: int = 4, n_q: int = 4) -> None:
        if n_k not in VALID_N or n_q not in VALID_N:
            raise ValueError(
                f"n_q/n_k must be one of {VALID_N}, got n_q={n_q}, n_k={n_k}"
            )
        self.n_k, self.n_q = n_k, n_q

    @torch.no_grad()
    def scores(
        self, q: torch.Tensor, k: torch.Tensor, softmax_scale: float
    ) -> torch.Tensor:
        """``[B, S, H, D] -> [B, H, Gq, Gk]`` block scores (log-space, higher = keep).

        Two Triton kernels: pool, then GEMM + segmented log-sum-exp in registers, so the
        ``[B, H, Gq*n_q, Gk*n_k]`` intermediate never reaches memory.

        ``softmax_scale * log2(e)`` is folded into Q so the kernel can use the exp2/log2
        hardware instructions; it multiplies by ln 2 on the way out, so scores come back
        in natural-log units. Selection is a top-k and any monotone rescale leaves that
        alone, so the units only matter to a reader of the magnitudes.

        The scores stay **float32**. bf16 would halve what selection reads, but with 8
        mantissa bits many blocks tie exactly at the threshold and the fused selector
        breaks ties by column index -- which systematically prefers early key blocks, one
        region of the video. Measured +3.9% relative L2 at S=96k.
        """
        b, s, h, d = q.shape
        sk = k.shape[1]
        gq, gk = -(-s // BLOCK), -(-sk // BLOCK)
        nq, nk = self.n_q, self.n_k
        sub_q, sub_k = BLOCK // nq, BLOCK // nk

        # Pooling handles the ragged tail on the *pooled* tensor: padding q/k up to
        # G*BLOCK first would copy the whole 300+ MB activation to add a few rows.
        # Sub-cells past the last real token pool to zero, and `*_valid` tells the score
        # kernel to drop them -- left in, each would contribute an exp(0)=1 term that
        # both inflates the score and flattens the differences the ranking depends on.
        pooled_q = torch.empty(b * h, gq * nq, d, device=q.device, dtype=torch.bfloat16)
        pooled_k = torch.empty(b * h, gk * nk, d, device=k.device, dtype=torch.bfloat16)
        fused_pool(q, gq * nq, sub_q, pooled_q, scale=softmax_scale * LOG2E)
        fused_pool(k, gk * nk, sub_k, pooled_k)

        out = torch.empty(b * h, gq, gk, device=q.device, dtype=torch.float32)
        fused_scores(
            pooled_q,
            pooled_k,
            out,
            n_k=nk,
            n_valid=-(-sk // sub_k),
            n_q=nq,
            m_valid=-(-s // sub_q),
        )
        return out.view(b, h, gq, gk)

    @torch.no_grad()
    def route(
        self, q: torch.Tensor, k: torch.Tensor, sparsity: float, softmax_scale: float
    ) -> RoutingPlan:
        """Select the top ``(1 - sparsity)`` fraction of key blocks per query block."""
        b, s, h, d = q.shape
        gk = -(-k.shape[1] // BLOCK)
        scores = self.scores(q, k, softmax_scale)  # [B, H, Gq, Gk]
        gq = scores.shape[2]
        topk = _snap_up_to_8(math.ceil((1.0 - sparsity) * gk), gk)
        # One pass over the score matrix instead of torch.topk's several. The
        # output order is unspecified: SM100 consumes it directly, while the
        # SM90 backend sorts compact active prefixes before heterogeneous expansion.
        index = fused_topk(scores.reshape(-1, gk), topk).view(b, h, gq, topk)
        return RoutingPlan(index=index, topk=topk, num_blocks=gk)

    @staticmethod
    def block_sizes(seq_len: int, device) -> torch.Tensor:
        """Real token count per 64-block, for the kernel's tail masking."""
        g = -(-seq_len // BLOCK)
        start = torch.arange(g, device=device, dtype=torch.int32) * BLOCK
        return (seq_len - start).clamp(0, BLOCK).to(torch.int32)
