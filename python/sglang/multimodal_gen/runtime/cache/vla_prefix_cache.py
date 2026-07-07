# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


@dataclass(frozen=True)
class VLAPrefixCacheKey:
    digest: str
    radix_key: RadixKey
    full_prefix_len: int
    debug_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrefixContext:
    """Request-local prefix state reused by all VLA denoise steps."""

    past_key_values: Any
    prefix_pad_masks: torch.Tensor
    prefix_position_ids: torch.Tensor
    prefix_len: int
    dtype: torch.dtype
    device: torch.device
    layout: dict[str, Any] = field(default_factory=dict)
    cache_key_digest: str | None = None

    def pin_for_request(self) -> PrefixContext:
        return self


class VLADensePrefixCache:
    def __init__(
        self,
        layers: tuple[tuple[torch.Tensor, torch.Tensor, Any], ...] | None = None,
    ):
        self.layers = list(layers or ())

    def __iter__(self):
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int):
        return self.layers[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return int(self.layers[layer_idx][0].shape[-2])

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int,
    ) -> tuple[int, int]:
        return self.get_seq_length(layer_idx) + int(cache_position.shape[0]), 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        prefix_position_ids=context.prefix_position_ids[index : index + 1],
        prefix_len=context.prefix_len,
        dtype=context.dtype,
        device=context.device,
        layout=dict(context.layout),
        cache_key_digest=context.cache_key_digest,
    )


@dataclass
class VLAPrefixCacheLookup:
    hit: bool
    context: PrefixContext | None
    match_len: int
    full_prefix_len: int
    partial_rejected: bool = False


def _json_dumps_stable(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _digest_to_radix_units(digest: str) -> array:
    raw = bytes.fromhex(digest)
    units = array("q")
    for offset in range(0, len(raw), 8):
        chunk = raw[offset : offset + 8].ljust(8, b"\0")
        value = int.from_bytes(chunk, "big", signed=False) & ((1 << 63) - 1)
        units.append(value)
    return units


class VLAPrefixCacheManager:
    """Exact full-prefix cache for VLA PrefixContext objects.

    The underlying SRT RadixCache is used only for key matching semantics here.
    Partial-prefix reuse is rejected because VLA prefix blocks can mix visual,
    language, and state tokens under full attention.
    """

    def __init__(self, max_entries: int = 128):
        self.max_entries = max(0, int(max_entries))
        self.radix = RadixCache.create_simulated()
        self._contexts: OrderedDict[str, PrefixContext] = OrderedDict()
        self.lookups = 0
        self.hits = 0
        self.partial_rejections = 0

    @staticmethod
    def make_key(
        *,
        model_revision: str,
        tokenizer_id: str,
        normalization_config: dict[str, Any] | None,
        discretization_config: dict[str, Any] | None,
        camera_order: tuple[str, ...],
        image_hashes: dict[str, str],
        prompt: list[str],
        token_digest: str | None,
        state_digest: str | None,
        masks: dict[str, bool],
        positions_version: str,
        dtype: str,
        adapter: str | None,
        parallel_layout_version: str,
        cache_namespace: str = "vla",
    ) -> VLAPrefixCacheKey:
        payload = {
            "cache_namespace": cache_namespace,
            "model_revision": model_revision,
            "tokenizer_id": tokenizer_id,
            "normalization_config": normalization_config,
            "discretization_config": discretization_config,
            "camera_order": list(camera_order),
            "image_hashes": image_hashes,
            "prompt": prompt,
            "token_digest": token_digest,
            "state_digest": state_digest,
            "masks": masks,
            "positions_version": positions_version,
            "dtype": dtype,
            "adapter": adapter,
            "parallel_layout_version": parallel_layout_version,
        }
        digest = hashlib.sha256(_json_dumps_stable(payload).encode("utf-8")).hexdigest()
        namespace = (
            f"{cache_namespace}:{model_revision}:{tokenizer_id}:"
            f"{parallel_layout_version}:{dtype}:{adapter or 'base'}"
        )
        radix_key = RadixKey(_digest_to_radix_units(digest), extra_key=namespace)
        return VLAPrefixCacheKey(
            digest=digest,
            radix_key=radix_key,
            full_prefix_len=len(radix_key),
            debug_payload=payload,
        )

    def get(self, key: VLAPrefixCacheKey) -> VLAPrefixCacheLookup:
        self.lookups += 1
        match = self.radix.match_prefix(MatchPrefixParams(key=key.radix_key))
        match_len = int(match.device_indices.numel())
        if match_len != key.full_prefix_len:
            partial_rejected = match_len > 0
            if partial_rejected:
                self.partial_rejections += 1
            return VLAPrefixCacheLookup(
                hit=False,
                context=None,
                match_len=match_len,
                full_prefix_len=key.full_prefix_len,
                partial_rejected=partial_rejected,
            )

        context = self._contexts.get(key.digest)
        if context is None:
            return VLAPrefixCacheLookup(
                hit=False,
                context=None,
                match_len=match_len,
                full_prefix_len=key.full_prefix_len,
            )

        self.hits += 1
        self._contexts.move_to_end(key.digest)
        return VLAPrefixCacheLookup(
            hit=True,
            context=context.pin_for_request(),
            match_len=match_len,
            full_prefix_len=key.full_prefix_len,
        )

    def put(self, key: VLAPrefixCacheKey, context: PrefixContext) -> None:
        if self.max_entries == 0:
            return
        if len(self._contexts) >= self.max_entries and key.digest not in self._contexts:
            self.clear()

        values = torch.arange(key.full_prefix_len, dtype=torch.int64)
        self.radix.insert(InsertParams(key=key.radix_key, value=values))
        context.cache_key_digest = key.digest
        self._contexts[key.digest] = context
        self._contexts.move_to_end(key.digest)

    def clear(self) -> None:
        self.radix.reset()
        self._contexts.clear()
