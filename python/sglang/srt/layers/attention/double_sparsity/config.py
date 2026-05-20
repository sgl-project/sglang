"""Configuration dataclass for standalone Double Sparsity.

The configuration surface is intentionally narrow: ``top_k``, ``page_size``,
``channel_mask_path``, ``device_buffer_size``, plus a free ``extra`` dict.
No ``selection_mode`` / ``top_p`` / ``min_top_k`` / ``max_top_k`` — top-p
selection (Twilight) is a separate follow-on with its own ABI design.
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


@dataclass
class DoubleSparsityConfig:
    top_k: int
    page_size: int
    channel_mask_path: str
    device_buffer_size: int
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

    missing = {"top_k", "page_size", "channel_mask_path", "device_buffer_size"} - set(
        data.keys()
    )
    if missing:
        raise ValueError(
            f"Double Sparsity config is missing required field(s): {sorted(missing)}."
        )

    return DoubleSparsityConfig(
        top_k=data["top_k"],
        page_size=data["page_size"],
        channel_mask_path=data["channel_mask_path"],
        device_buffer_size=data["device_buffer_size"],
        extra=data.get("extra", {}),
    )
