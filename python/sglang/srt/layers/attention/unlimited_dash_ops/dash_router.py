"""checkpoint parameters consumed by DASH routing kernels"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DashRouterConfig:
    idx_scale: float
    sigma: float
    prior_floor: float
    image_only: bool
    budget_remote: int
    topm: int
    out_slots: int


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


def router_from_sparse_config(sparse_cfg: dict, layer_idx=None) -> DashRouterConfig:
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
    remote_blocks = max(
        1,
        int(sparse_cfg.get("sparse_topk_blocks", 16))
        - max(1, int(sparse_cfg.get("sparse_local_block", 1))),
    )
    budget = max(1, int(budget)) if budget else remote_blocks
    topm = int(sparse_cfg.get("dash_infer_topm", 0)) or 16
    slots = 1 << (budget - 1).bit_length()
    return DashRouterConfig(
        idx_scale=scaling / (index_dim**0.5),
        sigma=float(sparse_cfg.get("dash_sigma", 1.0)),
        prior_floor=floor,
        image_only=bool(sparse_cfg.get("dash_image_only", True)),
        budget_remote=budget,
        topm=topm,
        out_slots=max(1, min(slots, topm)),
    )
