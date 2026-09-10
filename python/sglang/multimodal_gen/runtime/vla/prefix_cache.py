# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PrefixContext:
    """Request-local observation K/V reused by every action denoise step.

    The optional digest identifies an exact server-level cache entry; suffix K/V
    is step-dependent and never becomes part of this context.
    """

    past_key_values: Any
    prefix_pad_masks: torch.Tensor
    prefix_len: int
    layout: dict[str, Any] = field(default_factory=dict)
    cache_key_digest: str | None = None


class VLADensePrefixCache:
    """a lightweight and naive dense per-layer K/V container for prefix fill and suffix attention.

    Mutable instances collect prefix K/V layer by layer. Read-only instances
    prepend that fixed K/V to the current suffix K/V without changing storage.
    """

    def __init__(
        self,
        layers: Iterable[tuple[torch.Tensor, torch.Tensor, Any]] | None = None,
        *,
        read_only: bool = False,
    ):
        # cached_keys, cached_values, sliding_window
        self.layers = list(layers or ())
        self.read_only = read_only

    def __iter__(self):
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int):
        return self.layers[layer_idx]

    def get_seq_length(self) -> int:
        return 0 if not self.layers else int(self.layers[0][0].shape[-2])

    def get_prefix(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_keys, prefix_values, _ = self.layers[layer_idx]
        return prefix_keys, prefix_values

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """update the cache with fresh kv from each layer, return the appended prefix kv"""
        if self.read_only:
            prefix_keys, prefix_values = self.get_prefix(layer_idx)
            return (
                torch.cat([prefix_keys, key_states], dim=-2),
                torch.cat([prefix_values, value_states], dim=-2),
            )

        if layer_idx == len(self.layers):
            self.layers.append((key_states, value_states, None))
            return key_states, value_states
        if layer_idx > len(self.layers):
            raise IndexError(f"Invalid VLA prefix cache layer: {layer_idx}")
        cached_keys, cached_values, sliding_window = self.layers[layer_idx]
        key_states = torch.cat([cached_keys, key_states], dim=-2)
        value_states = torch.cat([cached_values, value_states], dim=-2)
        self.layers[layer_idx] = (key_states, value_states, sliding_window)
        return key_states, value_states


def slice_prefix_context(context: PrefixContext, index: int) -> PrefixContext:
    return PrefixContext(
        past_key_values=VLADensePrefixCache(
            tuple(
                (
                    keys[index : index + 1],
                    values[index : index + 1],
                    sliding_window,
                )
                for keys, values, sliding_window in context.past_key_values
            )
        ),
        prefix_pad_masks=context.prefix_pad_masks[index : index + 1],
        prefix_len=context.prefix_len,
        layout=dict(context.layout),
        cache_key_digest=context.cache_key_digest,
    )


class VLAPrefixCacheManager:
    """Bounded exact-match LRU for server-level VLA PrefixContext reuse.

    Partial-match prefix cache does not work well VLA scenario (with multiple combinations of keys).

    Request-local denoise reuse does not go through this cache. Partial-prefix
    K/V reuse is invalid for VLA prefix blocks that use full attention.
    """

    def __init__(self, max_entries: int = 128):
        self.max_entries = max(0, int(max_entries))
        self._cache: OrderedDict[str, PrefixContext] = OrderedDict()

    @staticmethod
    def make_key(
        *,
        model_revision: str,
        tokenizer_id: str,
        camera_order: tuple[str, ...],
        image_hashes: dict[str, str],
        token_digest: str,
        token_mask_digest: str,
        masks: dict[str, bool],
        positions_version: str,
        dtype: str,
        parallel_layout_version: str,
        cache_namespace: str = "vla",
    ) -> str:
        # hash the effective prefix inputs plus runtime compatibility dimensions
        payload = {
            "cache_namespace": cache_namespace,
            "model_revision": model_revision,
            "tokenizer_id": tokenizer_id,
            "camera_order": list(camera_order),
            "image_hashes": image_hashes,
            "token_digest": token_digest,
            "token_mask_digest": token_mask_digest,
            "masks": masks,
            "positions_version": positions_version,
            "dtype": dtype,
            "parallel_layout_version": parallel_layout_version,
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, key: str) -> PrefixContext | None:
        context = self._cache.get(key)
        if context is not None:
            self._cache.move_to_end(key)
        return context

    def put(self, key: str, context: PrefixContext) -> None:
        if self.max_entries == 0:
            return
        if len(self._cache) >= self.max_entries and key not in self._cache:
            self._cache.popitem(last=False)
        context.cache_key_digest = key
        self._cache[key] = context
        self._cache.move_to_end(key)
