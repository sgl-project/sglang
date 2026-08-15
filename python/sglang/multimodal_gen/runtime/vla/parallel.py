# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_rank
from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
)


@dataclass(frozen=True)
class VLASplitGroup:
    """Runtime view for VLA prefix/action split execution.

    This reuses the existing SP group as the coordination group. It is not a
    separate parallel topology:
    1. `prefix_root` computes/fetches PrefixContext and broadcasts it once.
    2. `action_root` owns fallback action denoise and initial noise broadcast.
    3. `action_ranks` may all participate in action SP when the policy allows it.

    All rank fields are global ranks; GroupCoordinator APIs take group-local
    ranks, so call `group_rank_for` before collective helpers.
    """

    group: GroupCoordinator
    prefix_root: int
    action_root: int
    action_ranks: tuple[int, ...]
    rank: int

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
    # v1 maps the split view onto SP: first rank does prefix encode, last rank
    # is the action fallback/root, and all SP ranks are eligible action ranks.
    return VLASplitGroup(
        group=group,
        prefix_root=group.ranks[0],
        action_root=group.ranks[-1],
        action_ranks=tuple(group.ranks),
        rank=get_world_rank(),
    )


def broadcast_tensor_from_rank(
    tensor: torch.Tensor | None,
    split: VLASplitGroup,
    *,
    src: int,
    device: torch.device,
) -> torch.Tensor | None:
    payload = (
        {"is_none": tensor is None, "tensor": tensor} if split.rank == src else None
    )
    payload = split.group.broadcast_tensor_dict(
        payload,
        src=split.group_rank_for(src),
    )
    if payload["is_none"]:
        return None
    output = payload["tensor"]
    if output.device != device:
        output = output.to(device)
    return output


def broadcast_prefix_context(
    context: PrefixContext | None,
    split: VLASplitGroup,
    *,
    src: int,
) -> PrefixContext | None:
    if split.rank == src and context is None:
        payload = {"is_none": True}
    elif split.rank == src:
        prefix_pad_masks = context.prefix_pad_masks
        prefix_pad_masks_is_bool = prefix_pad_masks.dtype == torch.bool
        if prefix_pad_masks_is_bool:
            prefix_pad_masks = prefix_pad_masks.to(torch.uint8)
        payload = {
            "is_none": False,
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_pad_masks_is_bool": prefix_pad_masks_is_bool,
            "prefix_len": context.prefix_len,
            "layout": dict(context.layout),
            "cache_key_digest": context.cache_key_digest,
            "num_layers": len(context.past_key_values),
        }
        for i, (keys, values, sliding_window) in enumerate(context.past_key_values):
            payload[f"layer_{i}_keys"] = keys
            payload[f"layer_{i}_values"] = values
            payload[f"layer_{i}_sliding_window"] = sliding_window
    else:
        payload = None

    payload = split.group.broadcast_tensor_dict(
        payload,
        src=split.group_rank_for(src),
    )
    if payload["is_none"]:
        return None

    kv_layers = []
    for i in range(int(payload["num_layers"])):
        kv_layers.append(
            (
                payload[f"layer_{i}_keys"],
                payload[f"layer_{i}_values"],
                payload[f"layer_{i}_sliding_window"],
            )
        )

    prefix_pad_masks = payload["prefix_pad_masks"]
    if payload.get("prefix_pad_masks_is_bool"):
        prefix_pad_masks = prefix_pad_masks.to(torch.bool)

    return PrefixContext(
        past_key_values=VLADensePrefixCache(tuple(kv_layers)),
        prefix_pad_masks=prefix_pad_masks,
        prefix_len=int(payload["prefix_len"]),
        layout=dict(payload["layout"]),
        cache_key_digest=payload["cache_key_digest"],
    )
