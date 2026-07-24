# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# TokenPrune -- the generic, model-agnostic token-pruning technique.
#
# The model-specific bits (which token span is prunable, and per-rank-local
# selection under SP) live in the ModelSpec, not here.
#
# Method: at active steps, score each prunable token, keep the top-K =
# round(N * keep_ratio) by score, run the transformer blocks on ONLY those K
# tokens (gather in before_blocks), then scatter the K-token result back to the
# full N and fill the dropped tokens with a compensation hidden state -- the
# previous step's ('prev') or zero. keep_ratio >= 1 is byte-identical baseline.

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.efficiency.registry import register_technique
from sglang.multimodal_gen.runtime.efficiency.schedule import Schedule, as_schedule
from sglang.multimodal_gen.runtime.efficiency.technique import (
    Capability,
    Phase,
    Seam,
    Technique,
    TechniqueContext,
)


def _uniform_indices(num_tokens: int, keep: int, device: torch.device) -> torch.Tensor:
    """Deterministic ascending uniform subset: idx[i] = floor(i * N / K)."""
    arange = torch.arange(keep, device=device, dtype=torch.long)
    return ((arange * num_tokens) // keep).clamp_(max=num_tokens - 1)


def keep_indices(
    method: str,
    num_tokens: int,
    keep_ratio: float,
    hidden_states: torch.Tensor,
    prev_velocity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ascending kept-token indices over a [B, S, C] (segment) tensor.

    Content-aware methods score each token (averaged over batch) and take the
    top-K; uniform/random are content-blind. Ascending order keeps the
    attention sequence monotone.
    """
    device = hidden_states.device
    keep = max(1, min(num_tokens, int(round(num_tokens * keep_ratio))))
    if keep >= num_tokens:
        return torch.arange(num_tokens, device=device, dtype=torch.long)

    if method in ("velocity", "vel") and prev_velocity is not None:
        scores = prev_velocity.float().pow(2).sum(-1).mean(0)
    elif method in ("feat_norm", "feat", "norm", "feat_l2"):
        scores = hidden_states.float().pow(2).sum(-1).mean(0)
    elif method == "feat_l1":
        scores = hidden_states.float().abs().sum(-1).mean(0)
    elif method in ("feat_linf", "feat_max"):
        scores = hidden_states.float().abs().amax(-1).mean(0)
    elif method == "feat_var":
        scores = hidden_states.float().var(-1).mean(0)
    elif method in ("random", "rand"):
        gen = torch.Generator(device=device).manual_seed(42)
        return torch.sort(
            torch.randperm(num_tokens, generator=gen, device=device)[:keep]
        ).values
    else:  # uniform / unknown -> content-blind even stride
        return _uniform_indices(num_tokens, keep, device)

    return torch.sort(torch.topk(scores, keep, largest=True).indices).values


@register_technique("token_prune")
class TokenPrune(Technique):
    """Generic mid-loop token pruning.

    Parameters
    ----------
    keep_ratio : Schedule[float] | float  -- fraction kept (>=1 => OFF/identity).
    method     : token-scoring method (feat_norm, feat_l1, uniform, ...).
    compensation : 'prev' (reuse previous step's dropped-token hidden) or 'zero'.
    enabled    : Schedule[bool] -- typically at_steps("1-2", ...) to prune only
                 select steps; step 0 always runs full to seed the 'prev' buffer.
    """

    name = "token_prune"
    phase = Phase.PRE_BLOCKS  # gather pre-loop; scatter is the paired POST work
    reads = frozenset({Seam.HIDDEN_STATES})
    writes = frozenset({Seam.TOKEN_SET, Seam.HIDDEN_STATES})
    required_capabilities = frozenset({Capability.PRUNABLE_TOKENS})

    def __init__(
        self,
        keep_ratio: "Schedule | float" = 1.0,
        method: str = "feat_norm",
        compensation: str = "prev",
        enabled: "Schedule | bool" = True,
    ):
        super().__init__(enabled=enabled)
        self.keep_ratio = as_schedule(keep_ratio)
        self.method = method
        self.compensation = compensation

    def is_active(self, ctx: TechniqueContext) -> bool:
        return super().is_active(ctx) and self.keep_ratio.at(ctx.step, ctx.stage) < 1.0

    def before_blocks(self, ctx: TechniqueContext, hidden):
        """Gather the prunable segment down to K tokens. carry = (start, end,
        keep_idx, full_S, prev_full) for after_blocks to scatter back."""
        spec = ctx.spec
        start, end = spec.segment(hidden, ctx)
        seg = hidden[:, start:end, :] if spec.seq_dim == 1 else hidden  # [B, S, C]
        full_S = seg.shape[1]
        ratio = float(self.keep_ratio.at(ctx.step, ctx.stage))

        prev_full = ctx.scratch.get(ctx.cache_key)
        # step 0 (or first active step with no prev) runs FULL to seed 'prev'.
        if prev_full is None and self.compensation == "prev":
            ctx.scratch[ctx.cache_key] = seg.detach()
            return hidden, None

        idx = keep_indices(self.method, full_S, ratio, seg)
        kept = seg.index_select(1, idx)
        new_hidden = torch.cat([hidden[:, :start, :], kept, hidden[:, end:, :]], dim=1)
        return new_hidden, (start, end, idx, full_S, prev_full)

    def after_blocks(self, ctx: TechniqueContext, hidden, carry):
        """Scatter the K-token result back to full S; fill dropped tokens with
        the compensation hidden; refresh the 'prev' buffer."""
        spec = ctx.spec
        if carry is None:
            # full (seed) step: store the full segment as next step's 'prev'.
            start, end = spec.segment(hidden, ctx)
            ctx.scratch[ctx.cache_key] = hidden[:, start:end, :].detach()
            return hidden

        start, end, idx, full_S, prev_full = carry
        kept_len = idx.shape[0]
        kept_out = hidden[:, start : start + kept_len, :]
        B, _, C = kept_out.shape
        if self.compensation == "zero" or prev_full is None:
            full = kept_out.new_zeros((B, full_S, C))
        else:
            full = prev_full.to(dtype=kept_out.dtype, device=kept_out.device).clone()
        full[:, idx, :] = kept_out
        ctx.scratch[ctx.cache_key] = full.detach()
        return torch.cat(
            [hidden[:, :start, :], full, hidden[:, start + kept_len :, :]], dim=1
        )
