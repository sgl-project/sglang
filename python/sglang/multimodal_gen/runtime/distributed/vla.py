# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from transformers.cache_utils import DynamicCache

from sglang.multimodal_gen.runtime.cache.vla_prefix_cache import PrefixContext
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_rank
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj


@dataclass(frozen=True)
class VLASplitGroup:
    group: GroupCoordinator
    prefix_root: int
    action_root: int
    rank: int

    @property
    def enabled(self) -> bool:
        return self.group.world_size > 1

    @property
    def is_prefix_rank(self) -> bool:
        return self.rank == self.prefix_root

    @property
    def is_action_rank(self) -> bool:
        return self.rank == self.action_root


def get_vla_split_group() -> VLASplitGroup | None:
    if not dist.is_available() or not dist.is_initialized():
        return None
    if not model_parallel_is_initialized():
        return None
    group = get_sp_group()
    if group.world_size <= 1:
        return None
    return VLASplitGroup(
        group=group,
        prefix_root=group.ranks[0],
        action_root=group.ranks[-1],
        rank=get_world_rank(),
    )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def _metadata_for_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    transfer_dtype = (
        "uint8" if tensor.dtype == torch.bool else _dtype_name(tensor.dtype)
    )
    return {
        "shape": tuple(tensor.shape),
        "dtype": _dtype_name(tensor.dtype),
        "transfer_dtype": transfer_dtype,
    }


def _broadcast_object(obj: Any, split: VLASplitGroup, src: int) -> Any:
    payload = [obj] if split.rank == src else []
    return broadcast_pyobj(
        payload,
        rank=split.rank,
        dist_group=split.group.cpu_group,
        src=src,
    )[0]


def _broadcast_tensor(
    tensor: torch.Tensor | None,
    metadata: dict[str, Any],
    split: VLASplitGroup,
    *,
    src: int,
    device: torch.device,
) -> torch.Tensor:
    transfer_dtype = _dtype_from_name(metadata["transfer_dtype"])
    target_dtype = _dtype_from_name(metadata["dtype"])
    if split.rank == src:
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

    dist.broadcast(payload, src=src, group=split.group.device_group)
    if target_dtype == torch.bool and payload.dtype != torch.bool:
        payload = payload.to(torch.bool)
    elif payload.dtype != target_dtype:
        payload = payload.to(target_dtype)
    return payload


def _prefix_context_metadata(context: PrefixContext | None) -> dict[str, Any]:
    if context is None:
        return {"is_none": True}
    layers = []
    for keys, values, sliding_window in context.past_key_values:
        layers.append(
            {
                "keys": _metadata_for_tensor(keys),
                "values": _metadata_for_tensor(values),
                "sliding_window": sliding_window,
            }
        )
    return {
        "is_none": False,
        "prefix_pad_masks": _metadata_for_tensor(context.prefix_pad_masks),
        "prefix_position_ids": _metadata_for_tensor(context.prefix_position_ids),
        "prefix_len": context.prefix_len,
        "dtype": _dtype_name(context.dtype),
        "layout": dict(context.layout),
        "cache_key_digest": context.cache_key_digest,
        "layers": layers,
    }


def broadcast_prefix_context(
    context: PrefixContext | None,
    split: VLASplitGroup,
    *,
    src: int,
    device: torch.device,
) -> PrefixContext | None:
    metadata = _broadcast_object(
        _prefix_context_metadata(context) if split.rank == src else None,
        split,
        src,
    )
    if metadata["is_none"]:
        return None

    prefix_pad_masks = _broadcast_tensor(
        context.prefix_pad_masks if split.rank == src and context is not None else None,
        metadata["prefix_pad_masks"],
        split,
        src=src,
        device=device,
    )
    prefix_position_ids = _broadcast_tensor(
        (
            context.prefix_position_ids
            if split.rank == src and context is not None
            else None
        ),
        metadata["prefix_position_ids"],
        split,
        src=src,
        device=device,
    )

    kv_layers = []
    source_layers = (
        list(context.past_key_values)
        if split.rank == src and context is not None
        else []
    )
    for i, layer_metadata in enumerate(metadata["layers"]):
        source_keys = source_layers[i][0] if source_layers else None
        source_values = source_layers[i][1] if source_layers else None
        keys = _broadcast_tensor(
            source_keys,
            layer_metadata["keys"],
            split,
            src=src,
            device=device,
        )
        values = _broadcast_tensor(
            source_values,
            layer_metadata["values"],
            split,
            src=src,
            device=device,
        )
        kv_layers.append((keys, values, layer_metadata["sliding_window"]))

    return PrefixContext(
        past_key_values=DynamicCache(tuple(kv_layers)),
        prefix_pad_masks=prefix_pad_masks,
        prefix_position_ids=prefix_position_ids,
        prefix_len=int(metadata["prefix_len"]),
        dtype=_dtype_from_name(metadata["dtype"]),
        device=device,
        layout=dict(metadata["layout"]),
        cache_key_digest=metadata["cache_key_digest"],
    )


def broadcast_optional_tensor(
    tensor: torch.Tensor | None,
    split: VLASplitGroup,
    *,
    src: int,
    device: torch.device,
) -> torch.Tensor | None:
    metadata = _broadcast_object(
        None if tensor is None else _metadata_for_tensor(tensor),
        split,
        src,
    )
    if metadata is None:
        return None
    return _broadcast_tensor(tensor, metadata, split, src=src, device=device)


def broadcast_metadata(
    metadata: dict[str, Any] | None,
    split: VLASplitGroup,
    *,
    src: int,
) -> dict[str, Any]:
    return _broadcast_object(metadata or {}, split, src)


def broadcast_timing(
    timings: dict[str, float] | None,
    split: VLASplitGroup,
    *,
    src: int,
) -> dict[str, float]:
    return broadcast_metadata(timings, split, src=src)
