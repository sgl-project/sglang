"""Keep duplicate prompts out of prefill while their first copy is in flight.

The radix cache cannot expose a request's KV until its prefill has completed.
This gate lets the normal cache lookup handle followers after that checkpoint,
without publishing unwritten KV or changing the cache's ownership rules.
"""

from __future__ import annotations

from typing import Any


class InflightPromptCache:
    def __init__(self, min_uncached_tokens: int = 32):
        self._leaders: dict[tuple[Any, Any, tuple[int, ...]], Any] = {}
        self._min_uncached_tokens = min_uncached_tokens

    @staticmethod
    def _key(req: Any) -> tuple[Any, Any, tuple[int, ...]] | None:
        # Token equality alone is insufficient for inputs whose embeddings or
        # state differ, even if the same placeholder token IDs are present.
        if (
            req.skip_radix_cache_insert
            or req.return_logprob
            or req.output_ids
            or getattr(req, "multimodal_inputs", None) is not None
            or getattr(req, "input_embeds", None) is not None
            or getattr(req, "positional_embed_overrides", None) is not None
            or getattr(req, "session", None) is not None
            or getattr(req, "beam_group", None) is not None
            or getattr(req, "is_retracted", False)
            or len(req.origin_input_ids) < 2
        ):
            return None
        return (req.extra_key, req.cache_salt, tuple(req.origin_input_ids))

    @staticmethod
    def _done(req: Any) -> bool:
        return (
            req.prefix_len >= len(req.origin_input_ids)
            or getattr(req, "is_retracted", False)
            or (req.finished() and req.kv.is_kv_released)
        )

    def prune(self) -> None:
        for key, leader in list(self._leaders.items()):
            if self._done(leader):
                del self._leaders[key]

    def should_wait(self, req: Any) -> bool:
        key = self._key(req)
        if key is None:
            return False
        leader = self._leaders.get(key)
        if leader is None:
            return False
        if self._done(leader):
            del self._leaders[key]
            return False
        # init_next_round_input has already performed the real radix lookup.
        # A nearly complete hit is cheaper to run than to delay.
        return (
            len(req.origin_input_ids) - 1 - req.prefix_len >= self._min_uncached_tokens
        )

    def admit(self, req: Any, *, is_chunked: bool) -> None:
        if is_chunked:
            return
        key = self._key(req)
        if key is not None:
            self._leaders.setdefault(key, req)
