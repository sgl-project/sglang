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
class Pi05PrefixCacheKey:
    digest: str
    radix_key: RadixKey
    full_prefix_len: int
    debug_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrefixContext:
    """Request-local prefix state reused by all Pi0.5 denoise steps."""

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


@dataclass
class Pi05PrefixCacheLookup:
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


class Pi05PrefixCacheManager:
    """Exact full-prefix cache for Pi0.5 PrefixContext objects.

    The underlying SRT RadixCache is used only for key matching semantics here.
    Pi0.5 does not accept partial-prefix reuse because the visual/language/state
    prefix is full-attention rather than causal-prefix compatible.
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
    ) -> Pi05PrefixCacheKey:
        payload = {
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
            f"pi05:{model_revision}:{tokenizer_id}:"
            f"{parallel_layout_version}:{dtype}:{adapter or 'base'}"
        )
        radix_key = RadixKey(_digest_to_radix_units(digest), extra_key=namespace)
        return Pi05PrefixCacheKey(
            digest=digest,
            radix_key=radix_key,
            full_prefix_len=len(radix_key),
            debug_payload=payload,
        )

    def get(self, key: Pi05PrefixCacheKey) -> Pi05PrefixCacheLookup:
        self.lookups += 1
        match = self.radix.match_prefix(MatchPrefixParams(key=key.radix_key))
        match_len = int(match.device_indices.numel())
        if match_len != key.full_prefix_len:
            partial_rejected = match_len > 0
            if partial_rejected:
                self.partial_rejections += 1
            return Pi05PrefixCacheLookup(
                hit=False,
                context=None,
                match_len=match_len,
                full_prefix_len=key.full_prefix_len,
                partial_rejected=partial_rejected,
            )

        context = self._contexts.get(key.digest)
        if context is None:
            return Pi05PrefixCacheLookup(
                hit=False,
                context=None,
                match_len=match_len,
                full_prefix_len=key.full_prefix_len,
            )

        self.hits += 1
        self._contexts.move_to_end(key.digest)
        return Pi05PrefixCacheLookup(
            hit=True,
            context=context.pin_for_request(),
            match_len=match_len,
            full_prefix_len=key.full_prefix_len,
        )

    def put(self, key: Pi05PrefixCacheKey, context: PrefixContext) -> None:
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
