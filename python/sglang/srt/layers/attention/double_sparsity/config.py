"""Configuration dataclass for standalone Double Sparsity.

The configuration surface is intentionally narrow: ``channel_mask_path`` (the
only required field), ``top_k``, ``page_size``, the selector variants below, and
a free ``extra`` dict. ``top_k`` counts maximum **tokens** per request (not
pages); at the DSA operating point it matches the model's ``index_topk=2048``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

_ALLOWED_FIELDS = {
    "top_k",
    "page_size",
    "channel_mask_path",
    "scorer_norm",
    "head_agg",
    "selector_width_buckets",
    "selector_width_overflow_policy",
    "score_reduce_dtype",
    "include_current_slot",
    "rope_aware_score",
    "extra",
}

# Selector variants are config-borne (not env) so they reach the TP worker
# processes that run the selector.
# scorer_norm: "cosine" (default, per-head dot divided by query/key norms) or
#   "off" (raw channel-dot). Both share the absorbed-latent numerator
#   (score = max_h v_h · c_kv) and are graph-safe.
# head_agg: cross-head score reduction, "max" (default) or "mean".
_ALLOWED_SCORER_NORM = ("off", "cosine")
_DEFAULT_SCORER_NORM = "cosine"
# Force-include the current decode slot (seq_len-1) in its own selected set.
_DEFAULT_INCLUDE_CURRENT_SLOT = True
_ALLOWED_HEAD_AGG = ("max", "mean")
_DEFAULT_HEAD_AGG = "max"


_DEFAULT_TOP_K = (
    2048  # matches the model's intrinsic index_topk (max tokens per request)
)
# Default compact selector score width: a prefix window comfortably covering
# the served decode windows while shrinking the per-call cross-TP score
# reduce ~40x vs the full req_to_token width. The full width is always
# captured as the overflow fallback; widths >= the model's full width are
# dropped at the runner, so small-context models degrade to full-width-only.
_DEFAULT_SELECTOR_WIDTH_BUCKETS = (5120,)
_ALLOWED_OVERFLOW_POLICY = ("full_fallback", "fail_closed")
_DEFAULT_OVERFLOW_POLICY = "full_fallback"
_DEFAULT_PAGE_SIZE = 64  # FlashMLA KV layout requirement


@dataclass
class DoubleSparsityConfig:
    channel_mask_path: str
    top_k: int = _DEFAULT_TOP_K
    page_size: int = _DEFAULT_PAGE_SIZE
    scorer_norm: str = _DEFAULT_SCORER_NORM
    head_agg: str = _DEFAULT_HEAD_AGG
    selector_width_buckets: List[int] = field(
        default_factory=lambda: list(_DEFAULT_SELECTOR_WIDTH_BUCKETS)
    )
    selector_width_overflow_policy: str = _DEFAULT_OVERFLOW_POLICY
    score_reduce_dtype: str = "bf16"
    # Force-include the current decode token's own slot (seq_len-1), overriding
    # the slot-validity -inf mask for that one slot only.
    include_current_slot: bool = _DEFAULT_INCLUDE_CURRENT_SLOT
    # Add the RoPE term q_pe·k_pe[t] to the absorbed score to recover long-context
    # accuracy. Off by default (byte-identical kernel launch); requires
    # scorer_norm="off" and fails closed at the selection site on any
    # non-validated runtime (bf16 KV, spec/MTP/DCP/NSA) rather than scoring no-PE.
    rope_aware_score: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scorer_norm not in _ALLOWED_SCORER_NORM:
            raise ValueError(
                f"Double Sparsity 'scorer_norm' must be one of "
                f"{list(_ALLOWED_SCORER_NORM)} ('off' = raw channel-dot; 'cosine' = "
                f"direction-normalized absorbed score), got {self.scorer_norm!r}."
            )
        if self.head_agg not in _ALLOWED_HEAD_AGG:
            raise ValueError(
                f"Double Sparsity 'head_agg' must be one of "
                f"{list(_ALLOWED_HEAD_AGG)}, got {self.head_agg!r}."
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
        if not isinstance(self.include_current_slot, bool):
            raise ValueError(
                f"Double Sparsity 'include_current_slot' must be a boolean, "
                f"got {self.include_current_slot!r}."
            )
        if not isinstance(self.rope_aware_score, bool):
            raise ValueError(
                f"Double Sparsity 'rope_aware_score' must be a boolean, "
                f"got {self.rope_aware_score!r}."
            )
        if self.rope_aware_score and self.scorer_norm != "off":
            # The absorbed identity score = max_h v_h·c_kv holds only for the raw dot;
            # the rope term is added on top of it. cosine-normalized no-PE mixed with a
            # raw rope dot is not a defined scorer — fail closed rather than score it.
            raise ValueError(
                "Double Sparsity 'rope_aware_score' requires scorer_norm='off' "
                f"(raw-dot absorbed score), got scorer_norm={self.scorer_norm!r}."
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
        if not isinstance(self.extra, dict):
            raise ValueError(
                f"Double Sparsity 'extra' must be a dict, got {type(self.extra).__name__}."
            )


def _coerce_width_buckets(value: Any) -> List[int]:
    # Fail closed: this knob drives the CUDA-graph capture ladder, so a
    # silently coerced bool/float/string width would capture an unintended
    # selector variant. Only genuine positive JSON integers are accepted.
    if not isinstance(value, list) or any(type(w) is not int or w <= 0 for w in value):
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
        scorer_norm=str(data.get("scorer_norm", _DEFAULT_SCORER_NORM)),
        head_agg=str(data.get("head_agg", _DEFAULT_HEAD_AGG)),
        selector_width_buckets=_coerce_width_buckets(
            data.get("selector_width_buckets", list(_DEFAULT_SELECTOR_WIDTH_BUCKETS))
        ),
        selector_width_overflow_policy=str(
            data.get("selector_width_overflow_policy", _DEFAULT_OVERFLOW_POLICY)
        ),
        score_reduce_dtype=str(data.get("score_reduce_dtype", "bf16")),
        include_current_slot=data.get(
            "include_current_slot", _DEFAULT_INCLUDE_CURRENT_SLOT
        ),
        rope_aware_score=data.get("rope_aware_score", False),
        extra=data.get("extra", {}),
    )
