"""Configuration dataclass for standalone Double Sparsity.

The configuration surface is intentionally narrow: ``top_k``, ``page_size``,
``channel_mask_path``, ``device_buffer_size``, plus a free ``extra`` dict.  No
``selection_mode`` / ``top_p`` / ``min_top_k`` / ``max_top_k`` — top-p selection
(Twilight) is a separate follow-on with its own ABI design.

``top_k`` counts maximum **tokens** per request (not pages).  At the DSA
index-topk operating point this matches the model's intrinsic ``index_topk=2048``.
``device_buffer_size`` is the score-scratch buffer cap (maximum concurrently
live tokens for the decode scoring scratch tensor).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


_ALLOWED_FIELDS = {
    "top_k",
    "page_size",
    "channel_mask_path",
    "device_buffer_size",
    "scorer_norm",
    "head_agg",
    "anchor_mode",
    "anchor_budget",
    "selector_width_buckets",
    "selector_width_overflow_policy",
    "score_reduce_dtype",
    "extra",
}

# Flag-gated non-learned selector variants (config-borne, not env, so they reach
# the TP worker processes that run the selector). Each is independent and
# defaults to the production behaviour (byte-identical when all are at default).
#
# scorer_norm: only "off" (raw channel-dot) is supported. The absorbed-latent
#   selection identity (score = max_h v_h · c_kv) holds only for the raw dot;
#   direction-only norms would operate on a materialized per-head signature the
#   selector never builds.
# head_agg: cross-head score reduction, "max" (default) or "mean".
# anchor_mode: which deterministic positions to always force-include in the
#   selection — "off" (default, none), "recency" (most-recent), "global"
#   (earliest stable), or "strided" (evenly spaced over [0, seq_len)).
# anchor_budget: how many anchor positions to force-include; 0 disables.
_ALLOWED_SCORER_NORM = ("off",)
_DEFAULT_SCORER_NORM = "off"
_ALLOWED_HEAD_AGG = ("max", "mean")
_DEFAULT_HEAD_AGG = "max"
_ALLOWED_ANCHOR_MODE = ("off", "recency", "global", "strided")
_DEFAULT_ANCHOR_MODE = "off"
_DEFAULT_ANCHOR_BUDGET = 0


_DEFAULT_TOP_K = 2048           # matches the model's intrinsic index_topk (max tokens per request)
# Default compact selector score width: a prefix window comfortably covering
# the served decode windows while shrinking the per-call cross-TP score
# reduce ~40x vs the full req_to_token width. The full width is always
# captured as the overflow fallback; widths >= the model's full width are
# dropped at the runner, so small-context models degrade to full-width-only.
_DEFAULT_SELECTOR_WIDTH_BUCKETS = (5120,)
_ALLOWED_OVERFLOW_POLICY = ("full_fallback", "fail_closed")
_DEFAULT_OVERFLOW_POLICY = "full_fallback"
_DEFAULT_PAGE_SIZE = 64         # FlashMLA KV layout requirement
_DEFAULT_DEVICE_BUFFER_SIZE = 4096  # score-scratch buffer cap in tokens


@dataclass
class DoubleSparsityConfig:
    channel_mask_path: str
    top_k: int = _DEFAULT_TOP_K
    page_size: int = _DEFAULT_PAGE_SIZE
    device_buffer_size: int = _DEFAULT_DEVICE_BUFFER_SIZE
    scorer_norm: str = _DEFAULT_SCORER_NORM
    head_agg: str = _DEFAULT_HEAD_AGG
    anchor_mode: str = _DEFAULT_ANCHOR_MODE
    anchor_budget: int = _DEFAULT_ANCHOR_BUDGET
    selector_width_buckets: List[int] = field(
        default_factory=lambda: list(_DEFAULT_SELECTOR_WIDTH_BUCKETS)
    )
    selector_width_overflow_policy: str = _DEFAULT_OVERFLOW_POLICY
    score_reduce_dtype: str = "bf16"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scorer_norm not in _ALLOWED_SCORER_NORM:
            raise ValueError(
                f"Double Sparsity 'scorer_norm' must be 'off' (the absorbed-latent "
                f"selection identity only holds for the raw channel-dot score), got "
                f"{self.scorer_norm!r}."
            )
        if self.head_agg not in _ALLOWED_HEAD_AGG:
            raise ValueError(
                f"Double Sparsity 'head_agg' must be one of "
                f"{list(_ALLOWED_HEAD_AGG)}, got {self.head_agg!r}."
            )
        if self.anchor_mode not in _ALLOWED_ANCHOR_MODE:
            raise ValueError(
                f"Double Sparsity 'anchor_mode' must be one of "
                f"{list(_ALLOWED_ANCHOR_MODE)}, got {self.anchor_mode!r}."
            )
        if not isinstance(self.anchor_budget, int) or self.anchor_budget < 0:
            raise ValueError(
                f"Double Sparsity 'anchor_budget' must be a non-negative integer, "
                f"got {self.anchor_budget!r}."
            )
        if not isinstance(self.selector_width_buckets, list) or any(
            not isinstance(w, int) or isinstance(w, bool) or w <= 0
            for w in self.selector_width_buckets
        ):
            raise ValueError(
                f"Double Sparsity 'selector_width_buckets' must be a list of "
                f"positive integers, got {self.selector_width_buckets!r}."
            )
        if self.selector_width_overflow_policy not in _ALLOWED_OVERFLOW_POLICY:
            raise ValueError(
                f"Double Sparsity 'selector_width_overflow_policy' must be one of "
                f"{list(_ALLOWED_OVERFLOW_POLICY)}, got "
                f"{self.selector_width_overflow_policy!r}."
            )
        if (
            self.selector_width_overflow_policy == "fail_closed"
            and not self.selector_width_buckets
        ):
            raise ValueError(
                "Double Sparsity 'selector_width_overflow_policy'='fail_closed' "
                "requires at least one 'selector_width_bucket' (a compact width to "
                "capture); with no compact bucket there is no captured graph and "
                "every sequence would fail closed."
            )
        if self.score_reduce_dtype not in ("fp32", "bf16"):
            raise ValueError(
                f"Double Sparsity 'score_reduce_dtype' must be one of "
                f"['fp32', 'bf16'], got {self.score_reduce_dtype!r}."
            )
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


def _coerce_width_buckets(value: Any) -> List[int]:
    # Fail closed: this knob drives the CUDA-graph capture ladder, so a
    # silently coerced bool/float/string width would capture an unintended
    # selector variant. Only genuine positive JSON integers are accepted.
    if (
        not isinstance(value, list)
        or any(type(w) is not int or w <= 0 for w in value)
    ):
        raise ValueError(
            f"Double Sparsity 'selector_width_buckets' must be a JSON array of "
            f"positive integers, got {value!r}."
        )
    return list(value)


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
        scorer_norm=str(data.get("scorer_norm", _DEFAULT_SCORER_NORM)),
        head_agg=str(data.get("head_agg", _DEFAULT_HEAD_AGG)),
        anchor_mode=str(data.get("anchor_mode", _DEFAULT_ANCHOR_MODE)),
        anchor_budget=int(data.get("anchor_budget", _DEFAULT_ANCHOR_BUDGET)),
        selector_width_buckets=_coerce_width_buckets(
            data.get(
                "selector_width_buckets", list(_DEFAULT_SELECTOR_WIDTH_BUCKETS)
            )
        ),
        selector_width_overflow_policy=str(
            data.get("selector_width_overflow_policy", _DEFAULT_OVERFLOW_POLICY)
        ),
        score_reduce_dtype=str(data.get("score_reduce_dtype", "bf16")),
        extra=data.get("extra", {}),
    )
