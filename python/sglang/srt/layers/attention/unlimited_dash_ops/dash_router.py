"""DASH router math, matching training.

Returns remote block indices plus centered log-prior biases; unused slots use
index ``-1`` with zero bias, and local-window scores stay unbiased.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Training uses a finite sentinel to avoid NaNs in score reductions.
SCORE_NEG = -1.0e9
# Value fed to entmax for masked-out entries (finite, very negative).
ENTMAX_NEG = -1.0e4
# Checkpoints may override the entmax evaluation width.
TOPM_DEFAULT = 16
_topm_warned = False


def _make_ix_like(X: torch.Tensor, dim: int) -> torch.Tensor:
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _entmax_threshold_and_support(X: torch.Tensor, dim: int = -1):
    Xsrt, _ = torch.sort(X, descending=True, dim=dim)
    rho = _make_ix_like(X, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt**2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean**2)
    delta = (1 - ss) / rho
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)
    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    # Same guard as training: a degenerate row can give support_size == 0, and
    # `support_size - 1 == -1` crashes gather with a device-side assert.
    support_size = support_size.clamp(min=1)
    tau_star = tau.gather(dim, support_size - 1)
    return tau_star, support_size


@torch.no_grad()
def entmax15(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """1.5-entmax forward (Peters, Niculae & Martins 2019), no autograd."""
    max_val, _ = X.max(dim=dim, keepdim=True)
    X = X - max_val  # shift-invariant, numerical stability
    X = X / 2  # alpha=1.5 exponent handling
    tau_star, _ = _entmax_threshold_and_support(X, dim)
    return torch.clamp(X - tau_star, min=0) ** 2


class DashRouterMath:
    """Config-carrying, stateless router math (no parameters).

    Owns only the score -> prior -> (indices, bias) transform that must match
    training; the index projections, norms and RoPE are checkpoint weights and
    live in the attention layer.
    """

    def __init__(
        self,
        *,
        index_dim: int,
        block_size: int,
        local_blocks: int,
        topk_blocks: int,
        scaling_factor: float = 1.0,
        sigma: float = 1.0,
        prior_floor: float = 1.0e-2,
        image_only: bool = True,
        image_fallback_remote: int | None = None,
        budget_remote: int | None = None,
        topm: int | None = None,
    ) -> None:
        self.index_dim = int(index_dim)
        self.block_size = int(block_size)
        self.local_blocks = max(1, int(local_blocks))
        self.local_window = self.local_blocks * self.block_size
        self.topk_blocks = int(topk_blocks)
        # training: remote_blocks = max(1, topk_blocks - local_blocks)
        self.remote_blocks = max(1, self.topk_blocks - self.local_blocks)
        # `remote_blocks` is buffer capacity; `budget_remote` is the trained
        # remote read budget and may be smaller.
        self.budget_remote = (
            max(1, int(budget_remote)) if budget_remote else self.remote_blocks
        )
        # Kept at the remote budget: the image-bucket mask compares against this
        # "min candidate count" to decide when to relax the image constraint.
        self.image_fallback_remote = (
            int(image_fallback_remote) if image_fallback_remote else self.remote_blocks
        )
        self.scaling_factor = float(scaling_factor)
        self.sigma = float(sigma)
        self.prior_floor = float(prior_floor)
        self.image_only = bool(image_only)
        # Stage 0 applies one `1 / sqrt(index_dim)` factor; routing applies the
        # second. Checkpoints store the raw scheduler scale.
        self.idx_scale = self.scaling_factor / (self.index_dim**0.5)
        self.topm = int(topm) if topm else TOPM_DEFAULT
        # Pack the hard top-k result into a power-of-two output width.
        slots = self.budget_remote
        slots = 1 << max(0, int(slots) - 1).bit_length()
        self.out_slots = max(1, min(slots, self.topm))

    @torch.no_grad()
    def _entmax_input(
        self,
        block_scores: torch.Tensor,
        image_block_mask: torch.Tensor | None = None,
    ):
        """Build the entmax input and candidate mask.

        Image filtering falls back to all remote blocks when it empties the set;
        the mask must accompany entmax or an all-masked row gets positive mass.
        """
        remote = block_scores > (SCORE_NEG * 0.5)
        cand = remote
        if self.image_only and image_block_mask is not None:
            cand = remote & image_block_mask
        no_cand = ~cand.any(dim=-1, keepdim=True)
        cand = torch.where(no_cand, remote, cand)
        x = torch.where(
            cand, block_scores * self.idx_scale, block_scores.new_full((), ENTMAX_NEG)
        )
        x = torch.nan_to_num(x, nan=ENTMAX_NEG, posinf=ENTMAX_NEG, neginf=ENTMAX_NEG)
        return x, cand

    @torch.no_grad()
    def route_and_select(
        self,
        block_scores: torch.Tensor,
        nblk: int,
        image_block_mask: torch.Tensor | None = None,
    ):
        """Select remote blocks and compute centered log-prior biases.

        Unused indices are ``-1`` with zero bias; the fixed top-M width supports
        CUDA-graph capture and is exact while the entmax support fits in M.
        """
        x, cand = self._entmax_input(block_scores, image_block_mask)
        M = max(1, min(self.topm, int(nblk), int(x.shape[-1])))
        xv, xi = x.topk(M, dim=-1)  # descending
        prior = entmax15(xv, dim=-1)  # [..., M]
        # Entmax gives an all-masked row positive mass; gather candidacy through
        # top-M so masked entries remain unused.
        is_cand = cand.gather(-1, xi)
        K = max(1, min(self.budget_remote, M))
        rank_ok = (
            torch.arange(M, device=x.device).view(*([1] * (is_cand.dim() - 1)), M) < K
        )
        valid = is_cand & rank_ok

        global _topm_warned
        if not _topm_warned and bool(((prior[..., -1] > 0) & is_cand[..., -1]).any()):
            _topm_warned = True
            logger.warning(
                "[Unlimited-OCR v2] entmax support saturated the checkpoint "
                "dash_infer_topm window (M=%d): the M-th block is still active, "
                "so tau* and bias values use a truncated window. Increase "
                "sparse_attention_config.dash_infer_topm.",
                M,
            )

        block_indices = xi.masked_fill(~valid, -1).to(torch.int32)
        # Center selected log-priors per query, preserving training's signed
        # remote bias while unused slots remain zero.
        logp = torch.log(prior.clamp_min(self.prior_floor))
        cnt = valid.sum(dim=-1, keepdim=True).clamp(min=1)
        mu = (
            torch.where(valid, logp, torch.zeros_like(logp)).sum(dim=-1, keepdim=True)
            / cnt
        )
        blk_bias = torch.where(
            valid, (logp - mu) * (1.0 / self.sigma), torch.zeros_like(logp)
        )
        return block_indices, blk_bias.float()

    # -- masking helper (prefill, dense scores) ----------------------------
    @torch.no_grad()
    def mask_remote(
        self,
        scores: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Apply training's remote-only mask to token-level scores.

        ``scores`` [..., Lq, S]; future keys and the local window are set to
        SCORE_NEG.
        """
        future = key_pos > query_pos[..., None]
        local = (key_pos > (query_pos[..., None] - self.local_window)) & ~future
        return scores.masked_fill(future | local, SCORE_NEG)


def _sched_slot(sparse_cfg: dict, layer_idx: int) -> int:
    """Map a decoder layer to its sparse-layer scheduler slot.

    Slots count enabled sparse layers before ``layer_idx``.
    """
    freq = sparse_cfg.get("sparse_attention_freq")
    if not isinstance(freq, (list, tuple)) or layer_idx >= len(freq):
        return int(layer_idx)
    return sum(1 for f in freq[:layer_idx] if f)


def _per_layer(sparse_cfg: dict, key: str, layer_idx):
    """``sparse_cfg[key][slot]`` when the per-layer array is present, else None."""
    if layer_idx is None:
        return None
    arr = sparse_cfg.get(key)
    if not isinstance(arr, (list, tuple)) or not arr:
        return None
    slot = _sched_slot(sparse_cfg, int(layer_idx))
    if not 0 <= slot < len(arr):
        return None
    return float(arr[slot])


def router_from_sparse_config(sparse_cfg: dict, layer_idx=None) -> DashRouterMath:
    """Build router configuration from checkpoint values.

    Per-layer scale/floor arrays are indexed by sparse-layer slot and both forms
    store the raw scheduler scale (``idx_scale`` applies the sqrt factor).
    """
    index_dim = int(sparse_cfg.get("sparse_index_dim", 128))
    scaling = _per_layer(sparse_cfg, "dash_scale_per_layer", layer_idx)
    floor = _per_layer(sparse_cfg, "dash_floor_per_layer", layer_idx)
    if scaling is None:
        scaling = float(sparse_cfg.get("dash_scaling_factor", 1.0))
    if floor is None:
        floor = float(sparse_cfg.get("dash_prior_floor", 1.0e-2))
    # Use the trained read budget, not the widened output-buffer width.
    # Older checkpoints fall back to `dash_image_fallback_remote`.
    budget = sparse_cfg.get("dash_topk_blocks")
    if budget is None:
        budget = sparse_cfg.get("dash_image_fallback_remote")
    return DashRouterMath(
        index_dim=index_dim,
        block_size=int(sparse_cfg.get("sparse_block_size", 128)),
        local_blocks=int(sparse_cfg.get("sparse_local_block", 1)),
        topk_blocks=int(sparse_cfg.get("sparse_topk_blocks", 16)),
        scaling_factor=scaling,
        sigma=float(sparse_cfg.get("dash_sigma", 1.0)),
        prior_floor=floor,
        image_only=bool(sparse_cfg.get("dash_image_only", True)),
        image_fallback_remote=int(sparse_cfg.get("dash_image_fallback_remote", 0))
        or None,
        budget_remote=int(budget) if budget else None,
        topm=int(sparse_cfg.get("dash_infer_topm", 0)) or None,
    )
