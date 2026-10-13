"""Checkpoint naming contract for Unlimited-OCR fused experts."""

import re
from typing import Optional, Tuple


def resolve_expert_weight(name: str) -> Optional[Tuple[str, int, str]]:
    """Map separate checkpoint projections to FusedMoE storage.

    Expert ownership and TP slices belong to the production FusedMoE loader.
    Shared experts are separate dense MLPs and must bypass this mapping.
    """
    match = re.fullmatch(
        r"(.*\.mlp\.)experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name
    )
    if match is None:
        return None
    prefix, expert, projection = match.groups()
    target, shard = {
        "gate_proj": ("w13_weight", "w1"),
        "up_proj": ("w13_weight", "w3"),
        "down_proj": ("w2_weight", "w2"),
    }[projection]
    return f"{prefix}experts.{target}", int(expert), shard


def load_unlimited_ocr_expert_weight(params_dict, name, loaded_weight):
    """Load a checkpoint expert into FusedMoE's own TP/EP weight loader.

    A None return leaves the legacy per-expert ModuleList loader untouched.
    Nonlocal EP experts are intentionally handled by the FusedMoE loader.
    """
    mapping = resolve_expert_weight(name)
    if mapping is None:
        return None
    target, expert_id, shard_id = mapping
    if target not in params_dict:
        return None
    param = params_dict[target]
    param.weight_loader(
        param, loaded_weight, target, shard_id=shard_id, expert_id=expert_id
    )
    return target
