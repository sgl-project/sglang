"""Configuration dataclass for standalone Double Sparsity.

The configuration surface is intentionally narrow: ``top_k``, ``page_size``,
``channel_mask_path``, ``device_buffer_size``, plus a free ``extra`` dict.  No
``selection_mode`` / ``top_p`` / ``min_top_k`` / ``max_top_k`` — top-p selection
(Twilight) is a separate follow-on with its own ABI design.

``top_k`` counts maximum **tokens** per request (not pages).  At Option B
operating point this matches the model's intrinsic ``index_topk=2048``.
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
    "recall_oracle",
    "selection_capture",
    "latent_capture",
    "score_capture",
    "selector_width_buckets",
    "selector_width_overflow_policy",
    "score_reduce_dtype",
    "enable_lifted_budget_decode",
    "lifted_budget_top_k",
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
# recall_oracle: config-borne enable for the fail-closed NIAH recall-oracle
#   diagnostic (off by default; byte-identical selection). Config-borne so it
#   reaches TP workers; requires --disable-cuda-graph (the hook does host syncs
#   illegal under graph capture). NOTE: this disables graph CAPTURE, but the
#   selector still runs the graph-safe path (retrieve_topk_graph_safe) eagerly —
#   the oracle hook lives there.
# score_capture: config-borne enable for the per-(layer, decode-step) absorbed
#   SCORE-row dump — the post-reduce / post-mask fp32 score tensor the selection
#   top-k consumes (logical-position indexed, same domain as selection_capture).
#   The Q2 instrument: compare cold-vs-warm scores at the positions that flip in
#   the selection. Eager decode only; capture-safe no-op under graph capture.
#   Off by default (one getattr on the hot path when off; byte-identical
#   selection). Requires --disable-cuda-graph (the dump host-copies).
# selection_capture: config-borne enable for the per-(layer, decode-step)
#   selection dump (selected_indices + valid_lengths). When on, the graph state
#   allocates per-layer capture buffers, the selector mirrors every layer's
#   selection into them (a captured device copy, so CUDA-graph replay keeps them
#   current), and the model runner appends one per-rank dump file per decode
#   step under cwd/.sglang_ds_selcap. Diagnostic only; off by default
#   (byte-identical selection, zero hot-path cost when off).
# selector_width_buckets: compact DS selector score widths (prefix windows) to
#   capture as additional CUDA-graph variants alongside the always-present
#   full req_to_token width. Default [5120]; an explicit [] captures full
#   width only. Selection is bit-identical across widths (prefix-window
#   semantics; overflow routes to the full-width variant). Each width must be
#   a positive int; widths at or above the full req_to_token width are
#   dropped at the runner.
# selector_width_overflow_policy: how the DS selector-width CUDA-graph ladder
#   treats a live sequence longer than every compact bucket.
#   "full_fallback" (default, byte-compatible): also capture the full
#   req_to_token width as the overflow fallback (today's behavior). "fail_closed":
#   capture ONLY the compact buckets (no full-width graph — reclaiming its
#   per-batch DS scratch) and raise a clear error if a live sequence exceeds the
#   largest captured compact width, declaring a bounded served-width operating
#   point. Requires >=1 compact bucket below the full width.
# score_reduce_dtype: transport dtype for the cross-TP score SUM-reduce.
#   "bf16" (default): scores are cast fp32->bf16 into preallocated scratch,
#   reduced (custom-all-reduce v2 when the byte size passes its eligibility
#   check, else NCCL on the raw group), and cast back — halving the reduce
#   bytes over the static score width. Scoring and top-k stay fp32; the only
#   numerics change is the reduce transport/output quantization, gated by the
#   selection-recall bound. "fp32": the original in-place NCCL ring reduce.
# enable_lifted_budget_decode: opt-in Tier-2.A adjustable-budget decode (AC-4).
#   When True, the selector may pick more than the DSA index_topk (a wider budget
#   recovers needles that rank in (index_topk, lifted_budget_top_k]); the opt-in
#   backend remaps physical selected slots → compact dequantized-KV indices for
#   flash_mla_sparse_fwd. Default False ⇒ the DSA dsa_index_topk==top_k assert is
#   untouched. This is the NEW, explicit mechanism — NOT max_top_k / Twilight
#   fields / the SGLANG_DS_ALLOW_TOPK_MISMATCH ablation escape.
# lifted_budget_top_k: the fixed (padded) budget for the lifted-budget path; must
#   be > index_topk and is only meaningful when enable_lifted_budget_decode is on.
_DEFAULT_LIFTED_BUDGET_TOP_K = 0  # 0 = unset; required (>top_k) when lifted enabled
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
    recall_oracle: bool = False
    selection_capture: bool = False
    latent_capture: bool = False
    score_capture: bool = False
    selector_width_buckets: List[int] = field(
        default_factory=lambda: list(_DEFAULT_SELECTOR_WIDTH_BUCKETS)
    )
    selector_width_overflow_policy: str = _DEFAULT_OVERFLOW_POLICY
    score_reduce_dtype: str = "bf16"
    enable_lifted_budget_decode: bool = False
    lifted_budget_top_k: int = _DEFAULT_LIFTED_BUDGET_TOP_K
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
        if not isinstance(self.recall_oracle, bool):
            raise ValueError(
                f"Double Sparsity 'recall_oracle' must be a boolean, "
                f"got {self.recall_oracle!r}."
            )
        if not isinstance(self.selection_capture, bool):
            raise ValueError(
                f"Double Sparsity 'selection_capture' must be a boolean, "
                f"got {self.selection_capture!r}."
            )
        if not isinstance(self.latent_capture, bool):
            raise ValueError(
                f"Double Sparsity 'latent_capture' must be a boolean, "
                f"got {self.latent_capture!r}."
            )
        if not isinstance(self.score_capture, bool):
            raise ValueError(
                f"Double Sparsity 'score_capture' must be a boolean, "
                f"got {self.score_capture!r}."
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
        if not isinstance(self.enable_lifted_budget_decode, bool):
            raise ValueError(
                f"Double Sparsity 'enable_lifted_budget_decode' must be a boolean, "
                f"got {self.enable_lifted_budget_decode!r}."
            )
        if not isinstance(self.lifted_budget_top_k, int) or self.lifted_budget_top_k < 0:
            raise ValueError(
                f"Double Sparsity 'lifted_budget_top_k' must be a non-negative "
                f"integer, got {self.lifted_budget_top_k!r}."
            )
        if self.enable_lifted_budget_decode:
            # The lifted budget must exceed the base budget (otherwise it is a
            # no-op and should not opt in to the heavier decode path).
            if self.lifted_budget_top_k <= self.top_k:
                raise ValueError(
                    "Double Sparsity 'lifted_budget_top_k' "
                    f"({self.lifted_budget_top_k}) must be > 'top_k' ({self.top_k}) "
                    "when enable_lifted_budget_decode is set (a lifted budget must "
                    "widen selection beyond the base index_topk)."
                )
            # flash_mla_sparse_fwd tiles the topk index width by 2*B_TOPK (=128);
            # a non-multiple width trips an in-kernel `topk % (2*B_TOPK) == 0`
            # assert. The realistic budgets (4096/8192) satisfy this.
            if self.lifted_budget_top_k % 128 != 0:
                raise ValueError(
                    "Double Sparsity 'lifted_budget_top_k' "
                    f"({self.lifted_budget_top_k}) must be a multiple of 128 (the "
                    "flash_mla_sparse_fwd index-width block constraint "
                    "topk % (2*B_TOPK) == 0)."
                )
        elif self.lifted_budget_top_k > 0:
            # Fail closed: a lifted budget set without the enable flag would
            # silently no-op (the default path keeps top_k == index_topk).
            raise ValueError(
                "Double Sparsity 'lifted_budget_top_k' is set "
                f"({self.lifted_budget_top_k}) but 'enable_lifted_budget_decode' is "
                "false — it would be ignored. Set enable_lifted_budget_decode=true "
                "to use the opt-in lifted-budget decode path."
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


def _coerce_bool(value: Any, field: str = "flag") -> bool:
    """Accept JSON booleans plus the common string/int spellings the serve
    script may emit (``true``/``1``/``yes`` etc.) so a config flag never silently
    no-ops because of a quoting mismatch."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    raise ValueError(
        f"Double Sparsity {field!r} must be a boolean, got {value!r}."
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
        recall_oracle=_coerce_bool(data.get("recall_oracle", False), "recall_oracle"),
        selection_capture=_coerce_bool(
            data.get("selection_capture", False), "selection_capture"
        ),
        latent_capture=_coerce_bool(
            data.get("latent_capture", False), "latent_capture"
        ),
        score_capture=_coerce_bool(data.get("score_capture", False), "score_capture"),
        selector_width_buckets=_coerce_width_buckets(
            data.get(
                "selector_width_buckets", list(_DEFAULT_SELECTOR_WIDTH_BUCKETS)
            )
        ),
        selector_width_overflow_policy=str(
            data.get("selector_width_overflow_policy", _DEFAULT_OVERFLOW_POLICY)
        ),
        score_reduce_dtype=str(data.get("score_reduce_dtype", "bf16")),
        enable_lifted_budget_decode=_coerce_bool(
            data.get("enable_lifted_budget_decode", False), "enable_lifted_budget_decode"
        ),
        lifted_budget_top_k=int(
            data.get("lifted_budget_top_k", _DEFAULT_LIFTED_BUDGET_TOP_K)
        ),
        extra=data.get("extra", {}),
    )
