"""Overrides of existing scheduler knobs and how they resolve against the configured values."""

from __future__ import annotations

from typing import Optional

import msgspec

# Token budgets align like DynamicChunkSizer's chunk sizes: to max(page_size, 64).
_MIN_TOKEN_ALIGNMENT = 64


class KnobOverrides(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Hardware-governor overrides of scheduler knobs; ``None`` keeps the configured value.

    ``chunked_prefill_size``, ``max_prefill_tokens`` and ``max_running_requests``
    are fractions in (0, 1] of the configured value, so an override can only
    shrink them. ``prefill_decode_interval`` is a lower bound in decode rounds,
    so it can only space prefills further apart.
    """

    chunked_prefill_size: Optional[float] = None
    max_prefill_tokens: Optional[float] = None
    max_running_requests: Optional[float] = None
    prefill_decode_interval: Optional[int] = None

    def __post_init__(self) -> None:
        _check_fraction("chunked_prefill_size", self.chunked_prefill_size)
        _check_fraction("max_prefill_tokens", self.max_prefill_tokens)
        _check_fraction("max_running_requests", self.max_running_requests)
        if (
            self.prefill_decode_interval is not None
            and self.prefill_decode_interval < 0
        ):
            raise ValueError(
                "prefill_decode_interval must be >= 0 decode rounds, "
                f"got {self.prefill_decode_interval}"
            )

    def resolve_chunked_prefill_size(
        self, configured: Optional[int], *, page_size: int
    ) -> Optional[int]:
        # None means chunked prefill is disabled; there is no chunk to shrink.
        if configured is None:
            return None
        return _scale_token_budget(
            configured, self.chunked_prefill_size, page_size=page_size
        )

    def resolve_max_prefill_tokens(self, configured: int, *, page_size: int) -> int:
        return _scale_token_budget(
            configured, self.max_prefill_tokens, page_size=page_size
        )

    def resolve_max_running_requests(self, configured: int) -> int:
        if self.max_running_requests is None:
            return configured
        return max(1, int(configured * self.max_running_requests))

    def resolve_prefill_decode_interval(self, configured: int) -> int:
        if self.prefill_decode_interval is None:
            return configured
        return max(configured, self.prefill_decode_interval)

    def describe(self) -> str:
        parts = [
            f"{name} x{value:g}"
            for name, value in (
                ("chunked_prefill_size", self.chunked_prefill_size),
                ("max_prefill_tokens", self.max_prefill_tokens),
                ("max_running_requests", self.max_running_requests),
            )
            if value is not None
        ]
        if self.prefill_decode_interval is not None:
            parts.append(f"prefill_decode_interval>={self.prefill_decode_interval}")
        return ", ".join(parts) or "no overrides"


def _check_fraction(name: str, value: Optional[float]) -> None:
    if value is not None and not 0.0 < value <= 1.0:
        raise ValueError(
            f"{name} override is a fraction of the configured value in (0, 1], got {value}"
        )


def _scale_token_budget(
    configured: int, fraction: Optional[float], *, page_size: int
) -> int:
    """Scale a per-pass token budget, aligned down and floored at one alignment unit."""
    if fraction is None:
        return configured
    alignment = max(page_size, _MIN_TOKEN_ALIGNMENT)
    if configured <= alignment:
        return configured
    scaled = int(configured * fraction) // alignment * alignment
    return max(scaled, alignment)
