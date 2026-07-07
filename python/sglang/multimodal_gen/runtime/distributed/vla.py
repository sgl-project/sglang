# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from transformers.cache_utils import DynamicCache

from sglang.multimodal_gen.runtime.cache.vla_prefix_cache import PrefixContext
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.broadcast import (
    broadcast_object,
    broadcast_tensor,
    dtype_from_name,
    dtype_name,
    tensor_metadata,
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


def _prefix_context_metadata(context: PrefixContext | None) -> dict[str, object]:
    if context is None:
        return {"is_none": True}
    layers = []
    for keys, values, sliding_window in context.past_key_values:
        layers.append(
            {
                "keys": tensor_metadata(keys),
                "values": tensor_metadata(values),
                "sliding_window": sliding_window,
            }
        )
    return {
        "is_none": False,
        "prefix_pad_masks": tensor_metadata(context.prefix_pad_masks),
        "prefix_position_ids": tensor_metadata(context.prefix_position_ids),
        "prefix_len": context.prefix_len,
        "dtype": dtype_name(context.dtype),
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
    metadata = broadcast_object(
        _prefix_context_metadata(context) if split.rank == src else None,
        group=split.group,
        rank=split.rank,
        src=src,
    )
    if metadata["is_none"]:
        return None

    prefix_pad_masks = broadcast_tensor(
        context.prefix_pad_masks if split.rank == src and context is not None else None,
        metadata["prefix_pad_masks"],
        group=split.group,
        rank=split.rank,
        src=src,
        device=device,
    )
    prefix_position_ids = broadcast_tensor(
        (
            context.prefix_position_ids
            if split.rank == src and context is not None
            else None
        ),
        metadata["prefix_position_ids"],
        group=split.group,
        rank=split.rank,
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
        keys = broadcast_tensor(
            source_keys,
            layer_metadata["keys"],
            group=split.group,
            rank=split.rank,
            src=src,
            device=device,
        )
        values = broadcast_tensor(
            source_values,
            layer_metadata["values"],
            group=split.group,
            rank=split.rank,
            src=src,
            device=device,
        )
        kv_layers.append((keys, values, layer_metadata["sliding_window"]))

    return PrefixContext(
        past_key_values=DynamicCache(tuple(kv_layers)),
        prefix_pad_masks=prefix_pad_masks,
        prefix_position_ids=prefix_position_ids,
        prefix_len=int(metadata["prefix_len"]),
        dtype=dtype_from_name(metadata["dtype"]),
        device=device,
        layout=dict(metadata["layout"]),
        cache_key_digest=metadata["cache_key_digest"],
    )
