"""Graph-stable route-slot and contribution buffers for exact DSR1 W2."""

from __future__ import annotations

import torch

EXPERTS = 257
TOPK = 9
HIDDEN_SIZE = 7168
SUPPORTED_M = (32,)

_WORKSPACES: dict[
    tuple[torch.device, int],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
] = {}


def get_route_workspaces(
    device: torch.device,
    m: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return all graph-stable route metadata and exact-W2 contribution storage.

    The contribution tensor is intentionally not cleared. Under the fail-closed
    DSR1 contract, every token has eight unique routed experts plus shared
    expert 256, so prep records one route for every slot and the producer
    overwrites every ``[token, slot, hidden]`` element exactly once.
    """
    if m not in SUPPORTED_M:
        raise ValueError(f"exact W2 supports M={SUPPORTED_M}, got M={m}")
    key = (device, m)
    workspaces = _WORKSPACES.get(key)
    if workspaces is None:
        workspaces = (
            torch.empty((EXPERTS, m), dtype=torch.int32, device=device),
            torch.empty((EXPERTS, m), dtype=torch.float32, device=device),
            torch.empty((EXPERTS,), dtype=torch.int32, device=device),
            torch.empty((EXPERTS, m), dtype=torch.int32, device=device),
            torch.empty((m, TOPK, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
        )
        _WORKSPACES[key] = workspaces
    return workspaces


__all__ = ["get_route_workspaces"]
