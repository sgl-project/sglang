# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    transfer_dtype = "uint8" if tensor.dtype == torch.bool else dtype_name(tensor.dtype)
    return {
        "shape": tuple(tensor.shape),
        "dtype": dtype_name(tensor.dtype),
        "transfer_dtype": transfer_dtype,
    }


def broadcast_object(
    obj: Any,
    *,
    group: GroupCoordinator,
    rank: int,
    src: int,
) -> Any:
    payload = [obj] if rank == src else []
    return broadcast_pyobj(
        payload,
        rank=rank,
        dist_group=group.cpu_group,
        src=src,
    )[0]


def broadcast_tensor(
    tensor: torch.Tensor | None,
    metadata: dict[str, Any],
    *,
    group: GroupCoordinator,
    rank: int,
    src: int,
    device: torch.device,
) -> torch.Tensor:
    transfer_dtype = dtype_from_name(metadata["transfer_dtype"])
    target_dtype = dtype_from_name(metadata["dtype"])
    if rank == src:
        assert tensor is not None
        payload = tensor.contiguous()
        if payload.dtype == torch.bool:
            payload = payload.to(torch.uint8)
    else:
        payload = torch.empty(
            metadata["shape"],
            dtype=transfer_dtype,
            device=device,
        )

    dist.broadcast(payload, src=src, group=group.device_group)
    if target_dtype == torch.bool and payload.dtype != torch.bool:
        payload = payload.to(torch.bool)
    elif payload.dtype != target_dtype:
        payload = payload.to(target_dtype)
    return payload


def broadcast_optional_tensor(
    tensor: torch.Tensor | None,
    *,
    group: GroupCoordinator,
    rank: int,
    src: int,
    device: torch.device,
) -> torch.Tensor | None:
    metadata = broadcast_object(
        None if tensor is None else tensor_metadata(tensor),
        group=group,
        rank=rank,
        src=src,
    )
    if metadata is None:
        return None
    return broadcast_tensor(
        tensor,
        metadata,
        group=group,
        rank=rank,
        src=src,
        device=device,
    )


def broadcast_metadata(
    metadata: dict[str, Any] | None,
    *,
    group: GroupCoordinator,
    rank: int,
    src: int,
) -> dict[str, Any]:
    return broadcast_object(metadata or {}, group=group, rank=rank, src=src)
