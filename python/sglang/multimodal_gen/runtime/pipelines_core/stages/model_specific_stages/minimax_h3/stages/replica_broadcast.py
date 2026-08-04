# SPDX-License-Identifier: Apache-2.0
"""Request-replica broadcast helpers for MiniMax H3 encoding stages."""

from __future__ import annotations

from typing import Any


def minimax_h3_replica_ctx() -> tuple[int, int]:
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_world_group,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized():
        return 1, 0
    # ServerArgs currently rejects DP>1 and H3 rejects disaggregation, so the
    # world group contains exactly one request replica (TP/CFG/SP ranks).
    group = get_world_group()
    return int(group.world_size), int(group.rank_in_group)


def minimax_h3_replica_broadcast_extra(batch: Any, key: str) -> None:
    world, rank = minimax_h3_replica_ctx()
    if world <= 1:
        return

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_world_group,
    )

    group = get_world_group()
    payload = {"value": batch.extra.get(key)} if rank == 0 else None
    payload = group.broadcast_tensor_dict(payload, src=0)
    value = payload.get("value") if isinstance(payload, dict) else None
    if value is None:
        raise RuntimeError(f"replica broadcast of batch.extra[{key!r}] got None")
    if rank != 0:
        batch.extra[key] = value


def minimax_h3_replica_broadcast_error(error: str | None) -> str | None:
    world, rank = minimax_h3_replica_ctx()
    if world <= 1:
        return error

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_world_group,
    )

    group = get_world_group()
    return group.broadcast_object(error if rank == 0 else None, src=0)


__all__ = [
    "minimax_h3_replica_broadcast_error",
    "minimax_h3_replica_broadcast_extra",
    "minimax_h3_replica_ctx",
]
