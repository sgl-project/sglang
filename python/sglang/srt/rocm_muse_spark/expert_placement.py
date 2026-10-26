"""Static expert placement for Muse Spark expert parallelism (``SGLANG_MUSE_EXPERT_PLACEMENT``).

A placement relabels each MoE layer's routed experts at load time: slot ``j`` of layer ``l`` holds checkpoint
expert ``perm[l][j]``. The router weight columns and ``expert_bias`` entries are permuted the same way, so routing
returns slot ids naming exactly the checkpoint's selection, and EP rank ``r`` keeps owning the contiguous slots
``[r * E / EP, (r + 1) * E / EP)`` that every expert kernel assumes. Nothing after loading knows about it.

The JSON file holds ``{"num_experts": 256, "layers": {"<layer>": [256 checkpoint ids in slot order], ...}}``;
a layer it does not list keeps the identity order. ``ep_size`` (optional) records the EP size it was balanced for.
"""

from __future__ import annotations

import json
import logging
import re

import torch

logger = logging.getLogger(__name__)

_MOE_PARAM = re.compile(r"\.layers\.(\d+)\.mlp\.")


class ExpertPlacement:
    """Per-layer slot -> checkpoint expert permutations (CPU int64); layers without an entry are identity."""

    def __init__(self, perms: dict[int, torch.Tensor], num_experts: int):
        self.num_experts = num_experts
        self._perms = perms
        self._slot_of = {
            layer: torch.argsort(perm).tolist() for layer, perm in perms.items()
        }

    @classmethod
    def from_json(cls, path: str, num_experts: int, ep_size: int) -> ExpertPlacement:
        with open(path) as f:
            spec = json.load(f)
        if int(spec.get("num_experts", num_experts)) != num_experts:
            raise ValueError(
                f"Expert placement {path} is for {spec['num_experts']} experts, the model has {num_experts}"
            )
        if "ep_size" in spec and int(spec["ep_size"]) != ep_size:
            logger.warning(
                "Expert placement %s was balanced for EP%d, running EP%d (outputs unchanged, balance not)",
                path,
                int(spec["ep_size"]),
                ep_size,
            )
        perms = {}
        for layer, order in spec["layers"].items():
            perm = torch.tensor(order, dtype=torch.int64)
            if perm.shape != (num_experts,) or not torch.equal(
                torch.sort(perm).values, torch.arange(num_experts)
            ):
                raise ValueError(
                    f"Expert placement {path} layer {layer}: not a permutation of {num_experts} experts"
                )
            if not torch.equal(perm, torch.arange(num_experts)):
                perms[int(layer)] = perm
        logger.info(
            "Muse expert placement %s: %d permuted layers (EP%d)",
            path,
            len(perms),
            ep_size,
        )
        return cls(perms, num_experts)

    def layers(self) -> list[int]:
        return sorted(self._perms)

    def perm(self, layer: int) -> torch.Tensor | None:
        """Checkpoint expert id of every slot, or None for the identity."""
        return self._perms.get(layer)

    def slot_of(self, layer: int, expert: int) -> int:
        slots = self._slot_of.get(layer)
        return expert if slots is None else slots[expert]

    def param_layer(self, name: str) -> int | None:
        """The MoE layer of a ``...layers.<l>.mlp.*`` parameter when that layer is permuted."""
        match = _MOE_PARAM.search(name)
        if match is None:
            return None
        layer = int(match.group(1))
        return layer if layer in self._perms else None


def permute_router_weight(weight: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Router ``[hidden, experts]`` with column ``j`` = checkpoint expert ``perm[j]``'s column."""
    return weight.index_select(1, perm.to(weight.device)).contiguous()


def permute_expert_vector(vector: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Per-expert vector (``expert_bias``) in slot order."""
    return vector.index_select(0, perm.to(vector.device)).contiguous()


def copy_expert_slots(
    target: torch.Tensor, stacked: torch.Tensor, experts: torch.Tensor
) -> None:
    """``target[i] = stacked[experts[i]]`` one expert at a time (no gathered copy of the stack on the host)."""
    if target.shape[0] != experts.numel() or target.shape[1:] != stacked.shape[1:]:
        raise RuntimeError(
            f"Expert slot copy shape mismatch: target={tuple(target.shape)}, "
            f"stack={tuple(stacked.shape)}, experts={experts.numel()}"
        )
    for slot, expert in enumerate(experts.tolist()):
        target[slot].copy_(stacked[expert])
