# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.cache.vla_prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_rank


@dataclass(frozen=True)
class VLASplitGroup:
    group: GroupCoordinator
    prefix_root: int
    action_root: int
    action_ranks: tuple[int, ...]
    rank: int

    @property
    def enabled(self) -> bool:
        return self.group.world_size > 1

    @property
    def is_prefix_rank(self) -> bool:
        return self.rank == self.prefix_root

    @property
    def is_action_rank(self) -> bool:
        return self.rank in self.action_ranks

    @property
    def uses_action_sp(self) -> bool:
        return len(self.action_ranks) > 1

    def group_rank_for(self, global_rank: int) -> int:
        return self.group.ranks.index(global_rank)

    def broadcast_object_from_rank(self, obj, *, src: int):
        return self.group.broadcast_object(
            obj if self.rank == src else None,
            src=self.group_rank_for(src),
        )


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
        action_ranks=tuple(group.ranks),
        rank=get_world_rank(),
    )


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, object]:
    transfer_dtype = torch.uint8 if tensor.dtype == torch.bool else tensor.dtype
    return {
        "shape": tuple(tensor.shape),
        "dtype": tensor.dtype,
        "transfer_dtype": transfer_dtype,
    }


def _broadcast_tensor_with_metadata(
    tensor: torch.Tensor | None,
    metadata: dict[str, object],
    split: VLASplitGroup,
    *,
    src: int,
    device: torch.device,
) -> torch.Tensor:
    transfer_dtype = metadata["transfer_dtype"]
    target_dtype = metadata["dtype"]
    if split.rank == src:
        assert tensor is not None
        payload = tensor.contiguous()
        if payload.dtype != transfer_dtype:
            payload = payload.to(transfer_dtype)
    else:
        payload = torch.empty(
            metadata["shape"],
            dtype=transfer_dtype,
            device=device,
        )
    payload = split.group.broadcast(payload, src=split.group_rank_for(src))
    if payload.dtype != target_dtype:
        payload = payload.to(target_dtype)
    return payload


def broadcast_tensor_from_rank(
    tensor: torch.Tensor | None,
    split: VLASplitGroup,
    *,
    src: int,
    device: torch.device,
) -> torch.Tensor | None:
    metadata = split.broadcast_object_from_rank(
        _tensor_metadata(tensor) if tensor is not None else None,
        src=src,
    )
    if metadata is None:
        return None
    return _broadcast_tensor_with_metadata(
        tensor,
        metadata,
        split,
        src=src,
        device=device,
    )


def _prefix_context_metadata(context: PrefixContext | None) -> dict[str, object]:
    if context is None:
        return {"is_none": True}
    layers = []
    for keys, values, sliding_window in context.past_key_values:
        layers.append(
            {
                "keys": _tensor_metadata(keys),
                "values": _tensor_metadata(values),
                "sliding_window": sliding_window,
            }
        )
    return {
        "is_none": False,
        "prefix_pad_masks": _tensor_metadata(context.prefix_pad_masks),
        "prefix_position_ids": _tensor_metadata(context.prefix_position_ids),
        "prefix_len": context.prefix_len,
        "dtype": context.dtype,
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
    metadata = split.broadcast_object_from_rank(
        _prefix_context_metadata(context),
        src=src,
    )
    if metadata["is_none"]:
        return None

    prefix_pad_masks = _broadcast_tensor_with_metadata(
        context.prefix_pad_masks if split.rank == src and context is not None else None,
        metadata["prefix_pad_masks"],
        split,
        src=src,
        device=device,
    )
    prefix_position_ids = _broadcast_tensor_with_metadata(
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
        keys = _broadcast_tensor_with_metadata(
            source_keys,
            layer_metadata["keys"],
            split,
            src=src,
            device=device,
        )
        values = _broadcast_tensor_with_metadata(
            source_values,
            layer_metadata["values"],
            split,
            src=src,
            device=device,
        )
        kv_layers.append((keys, values, layer_metadata["sliding_window"]))

    return PrefixContext(
        past_key_values=VLADensePrefixCache(tuple(kv_layers)),
        prefix_pad_masks=prefix_pad_masks,
        prefix_position_ids=prefix_position_ids,
        prefix_len=int(metadata["prefix_len"]),
        dtype=metadata["dtype"],
        device=device,
        layout=dict(metadata["layout"]),
        cache_key_digest=metadata["cache_key_digest"],
    )
