"""DASH Stage-0 summarization for ``head_cls`` checkpoints.

A masked softmax summarizes each KV page before it is scored against index
queries; both stages apply ``1 / sqrt(d)``, matching training.
"""

from __future__ import annotations

import torch

# Same activation bound training clamps with.
ACT_CLAMP = 1.0e4


def resolve_stage0(sparse_cfg) -> str:
    """Return the checkpoint's Stage-0 mode, defaulting to ``indexer``."""
    cfg = sparse_cfg.get("dash_stage0") if isinstance(sparse_cfg, dict) else None
    return str(cfg or "indexer").lower()


def _summary_keys(
    keys: torch.Tensor, valid: torch.Tensor, head_cls: torch.Tensor, inv_sqrt_d: float
) -> torch.Tensor:
    """[nb, bs, d] index keys -> [nb, d] head_cls summary keys.

    NaN-safe masked softmax matching training: row max over VALID tokens only,
    exp, multiplicative padding zeroing, clamped denominator.
    """
    aw = (keys * head_cls).sum(-1) * inv_sqrt_d  # [nb, bs]
    aw = aw.clamp(-ACT_CLAMP, ACT_CLAMP)
    neg = torch.finfo(aw.dtype).min
    row_max = torch.where(valid, aw, aw.new_full((), neg)).amax(dim=-1, keepdim=True)
    row_max = torch.where(
        valid.any(dim=-1, keepdim=True), row_max, torch.zeros_like(row_max)
    )
    e = torch.exp(aw - row_max) * valid.to(aw.dtype)
    w = e / e.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [nb, bs]
    k_bar = (w.unsqueeze(-1) * keys).sum(dim=1)  # [nb, d]
    return torch.nan_to_num(k_bar, nan=0.0, posinf=0.0, neginf=0.0)
