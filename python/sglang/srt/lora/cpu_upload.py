# SPDX-License-Identifier: Apache-2.0
"""Pinned CPU snapshots in the exact layout of a GPU adapter slot."""

from __future__ import annotations

import copy
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

BufferKey = tuple[str, str, int | None]
LAYER_GROUPS = ("A_buffer", "B_buffer")
EMBEDDING_GROUPS = (
    "embedding_A_buffer",
    "embedding_B_buffer",
    "lm_head_A_buffer",
    "lm_head_B_buffer",
    "new_embeddings_buffer",
)


def buffers(pool: Any) -> Iterator[tuple[BufferKey, torch.Tensor]]:
    for group in LAYER_GROUPS:
        for key, layers in getattr(pool, group).items():
            for layer, tensor in enumerate(layers):
                yield (group, key, layer), tensor
    for group in EMBEDDING_GROUPS:
        for key, tensor in getattr(pool, group).items():
            yield (group, key, None), tensor


@dataclass(frozen=True)
class UploadPlan:
    pool: weakref.ReferenceType
    sources: tuple[tuple[BufferKey, torch.Tensor], ...]

    def resolve(self, pool: Any) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        if self.pool() is not pool:
            return None
        current = dict(buffers(pool))
        if current.keys() != {key for key, _ in self.sources}:
            return None
        result = []
        for key, source in self.sources:
            destination = current[key]
            if (
                destination.shape[1:] != source.shape
                or destination.dtype != source.dtype
            ):
                return None
            result.append((destination, source))
        return result

    def copy_to(self, pool: Any, slot: int) -> bool:
        transfers = self.resolve(pool)
        if transfers is None:
            return False
        for destination, source in transfers:
            destination[slot].copy_(source, non_blocking=True)
        return True


def build_upload_plan(
    manager: Any, adapter: Any, load: Callable[..., None]
) -> UploadPlan:
    pool = manager.memory_pool
    shadow = copy.copy(pool)
    shadow.pin_memory_available = False

    def allocate(tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (1, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )

    with torch.cuda.device(manager.device):
        for group in LAYER_GROUPS:
            setattr(
                shadow,
                group,
                {
                    key: [allocate(tensor) for tensor in layers]
                    for key, layers in getattr(pool, group).items()
                },
            )
        for group in EMBEDDING_GROUPS:
            setattr(
                shadow,
                group,
                {key: allocate(tensor) for key, tensor in getattr(pool, group).items()},
            )
        load(
            shadow,
            adapter.uid,
            0,
            adapter,
            manager.lora_modules,
            manager.embed_tokens_module,
            manager.lm_head_module,
        )
    return UploadPlan(
        weakref.ref(pool), tuple((key, tensor[0]) for key, tensor in buffers(shadow))
    )
