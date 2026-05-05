"""Weight audit utilities for true-on-policy debugging.

Captures lightweight per-stage tensor statistics (shape, stride, sample
checksum) for a configurable set of model parameters and writes them as
JSON.  Enabled only when ``SGLANG_WEIGHT_AUDIT_ENABLE=1``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


def _weight_audit_enabled() -> bool:
    value = os.environ.get("SGLANG_WEIGHT_AUDIT_ENABLE", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _weight_audit_max_versions() -> int:
    value = os.environ.get("SGLANG_WEIGHT_AUDIT_MAX_VERSIONS", "3")
    try:
        return int(value)
    except ValueError:
        return 3


def _weight_audit_version_allowed(weight_version: Optional[str]) -> bool:
    max_versions = _weight_audit_max_versions()
    if max_versions < 0 or weight_version is None:
        return True
    try:
        return int(weight_version) <= max_versions
    except (TypeError, ValueError):
        return True


def _weight_audit_dir() -> str:
    return os.environ.get("SGLANG_WEIGHT_AUDIT_DIR", "/tmp/sglang_weight_audit")


def _weight_audit_family(name: str) -> Optional[str]:
    if name in {
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
    }:
        return "embedding"
    if name == "lm_head.weight":
        return "output_layer"
    if name in {
        "model.norm.weight",
        "model.language_model.norm.weight",
        "language_model.norm.weight",
    }:
        return "final_layernorm"
    if ".mlp.gate.weight" in name:
        return "moe_router"
    if ".mlp.experts.w13_weight" in name:
        return "moe_expert_w13"
    if ".mlp.experts.w2_weight" in name:
        return "moe_expert_w2"
    return None


def _weight_audit_selected_names(names: List[str]) -> List[str]:
    by_family: dict[str, List[str]] = {}
    for name in names:
        family = _weight_audit_family(name)
        if family is not None:
            by_family.setdefault(family, []).append(name)

    selected: list[str] = []
    for names_for_family in by_family.values():
        ordered = sorted(names_for_family)
        candidate_indices = {0, len(ordered) // 2, len(ordered) - 1}
        for index in sorted(candidate_indices):
            if 0 <= index < len(ordered):
                selected.append(ordered[index])
    return sorted(set(selected))


def _weight_audit_tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    with torch.no_grad():
        detached = tensor.detach()
        flat = detached.reshape(-1)
        numel = flat.numel()
        if numel == 0:
            sample = flat
        else:
            sample_size = min(4096, numel)
            midpoint = max(0, (numel - sample_size) // 2)
            sample = torch.cat(
                [
                    flat[:sample_size],
                    flat[midpoint : midpoint + sample_size],
                    flat[-sample_size:],
                ]
            )
        sample_float = sample.float()
        if sample_float.numel() == 0:
            sample_sum = 0.0
            sample_absmax = 0.0
            sample_first = None
            sample_last = None
        else:
            sample_sum = float(sample_float.sum().item())
            sample_absmax = float(sample_float.abs().max().item())
            sample_first = float(sample_float[0].item())
            sample_last = float(sample_float[-1].item())

        return {
            "shape": list(detached.shape),
            "stride": list(detached.stride()),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "numel": int(numel),
            "storage_offset": int(detached.storage_offset()),
            "sample_numel": int(sample_float.numel()),
            "sample_sum_fp32": sample_sum,
            "sample_absmax_fp32": sample_absmax,
            "sample_first_fp32": sample_first,
            "sample_last_fp32": sample_last,
        }


def write_weight_audit(
    model: nn.Module,
    *,
    stage: str,
    weight_version: Optional[str],
    tp_rank: int,
    pp_rank: int,
    extra_tensors: Optional[List[Tuple[str, torch.Tensor]]] = None,
) -> None:
    if not _weight_audit_enabled() or not _weight_audit_version_allowed(weight_version):
        return

    try:
        named_tensors = list(model.named_parameters())
    except Exception as err:
        logger.warning("[WeightAudit] failed to list model parameters: %s", err)
        return
    if extra_tensors:
        named_tensors.extend(
            (f"__load__.{name}", tensor) for name, tensor in extra_tensors
        )

    selected_names = _weight_audit_selected_names([name for name, _ in named_tensors])
    tensor_by_name = {name: tensor for name, tensor in named_tensors}
    selected = {
        name: _weight_audit_tensor_stats(tensor_by_name[name])
        for name in selected_names
        if name in tensor_by_name
    }
    if not selected:
        return

    os.makedirs(_weight_audit_dir(), exist_ok=True)
    version = "unknown" if weight_version is None else str(weight_version)
    path = os.path.join(
        _weight_audit_dir(),
        f"sglang_{stage}_v{version}_tp{tp_rank:03d}_pp{pp_rank:03d}.json",
    )
    payload = {
        "stage": stage,
        "weight_version": version,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "selected": selected,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
