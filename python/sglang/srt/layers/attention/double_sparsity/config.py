"""Configuration dataclass for standalone Double Sparsity.

The configuration surface is intentionally narrow: ``top_k``, ``page_size``,
``channel_mask_path``, ``device_buffer_size``, plus a free ``extra`` dict.
No ``selection_mode`` / ``top_p`` / ``min_top_k`` / ``max_top_k`` — top-p
selection (Twilight) is a separate follow-on with its own ABI design.

``top_k`` counts maximum **tokens** per request (not pages).  At Option B
operating point this matches the model's intrinsic ``index_topk=2048``.
``device_buffer_size`` is the score-scratch buffer cap (maximum concurrently
live tokens for the decode scoring scratch tensor).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict


_ALLOWED_FIELDS = {
    "top_k",
    "page_size",
    "channel_mask_path",
    "device_buffer_size",
    "extra",
}


_DEFAULT_TOP_K = 2048           # matches DeepSeek-V3.2 index_topk (max tokens per request)
_DEFAULT_PAGE_SIZE = 64         # FlashMLA KV layout requirement
_DEFAULT_DEVICE_BUFFER_SIZE = 4096  # score-scratch buffer cap in tokens


@dataclass
class DoubleSparsityConfig:
    channel_mask_path: str
    top_k: int = _DEFAULT_TOP_K
    page_size: int = _DEFAULT_PAGE_SIZE
    device_buffer_size: int = _DEFAULT_DEVICE_BUFFER_SIZE
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError(
                f"Double Sparsity 'top_k' must be a positive integer, got {self.top_k!r}."
            )
        if not isinstance(self.page_size, int) or self.page_size <= 0:
            raise ValueError(
                f"Double Sparsity 'page_size' must be a positive integer, got {self.page_size!r}."
            )
        if not isinstance(self.channel_mask_path, str) or not self.channel_mask_path:
            raise ValueError(
                "Double Sparsity 'channel_mask_path' must be a non-empty string."
            )
        if not isinstance(self.device_buffer_size, int) or self.device_buffer_size <= 0:
            raise ValueError(
                f"Double Sparsity 'device_buffer_size' must be a positive integer, "
                f"got {self.device_buffer_size!r}."
            )
        if not isinstance(self.extra, dict):
            raise ValueError(
                f"Double Sparsity 'extra' must be a dict, got {type(self.extra).__name__}."
            )


def parse_double_sparsity_config(payload: str) -> DoubleSparsityConfig:
    """Parse a JSON string into a :class:`DoubleSparsityConfig`.

    Rejects unknown top-level keys so that fields reserved for the (deferred)
    Twilight top-p ABI — ``selection_mode``, ``top_p``, ``min_top_k``,
    ``max_top_k`` — do not silently no-op.
    """

    if not isinstance(payload, str) or not payload.strip():
        raise ValueError(
            "Double Sparsity requires --double-sparsity-config to be a non-empty "
            "JSON string with at least 'channel_mask_path' set."
        )

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Double Sparsity config is not valid JSON: {exc.msg} at line {exc.lineno}, "
            f"column {exc.colno}."
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Double Sparsity config must be a JSON object, got {type(data).__name__}."
        )

    unknown = set(data.keys()) - _ALLOWED_FIELDS
    if unknown:
        raise ValueError(
            f"Double Sparsity config has unknown field(s): {sorted(unknown)}. "
            f"Allowed fields are {sorted(_ALLOWED_FIELDS)}. "
            "Note: 'selection_mode' and 'top_p' belong to the deferred Twilight ABI "
            "and are not accepted by the initial deliverable."
        )

    if "channel_mask_path" not in data:
        raise ValueError(
            "Double Sparsity config is missing required field 'channel_mask_path'."
        )

    return DoubleSparsityConfig(
        channel_mask_path=data["channel_mask_path"],
        top_k=int(data.get("top_k", _DEFAULT_TOP_K)),
        page_size=int(data.get("page_size", _DEFAULT_PAGE_SIZE)),
        device_buffer_size=int(data.get("device_buffer_size", _DEFAULT_DEVICE_BUFFER_SIZE)),
        extra=data.get("extra", {}),
    )
