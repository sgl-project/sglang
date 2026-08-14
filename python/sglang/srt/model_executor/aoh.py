"""AoH sidecar configuration and validation helpers.

The offline effective-rank analysis produces a small JSON sidecar instead of
changing model weights.  The runtime only needs the per-layer decision for
each global KV group.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_VALID_MODES = frozenset(("retrieval", "streaming"))


def normalize_aoh_window(
    sink_size: int, recent_size: int, context_len: int
) -> tuple[int, int]:
    """Clamp an AoH window to an equivalent size within the model context."""
    if sink_size <= 0 or recent_size <= 0 or context_len <= 0:
        raise ValueError("AoH sink, recent, and context sizes must be positive.")
    sink_size = min(sink_size, context_len)
    recent_size = min(recent_size, max(1, context_len - sink_size))
    return sink_size, recent_size


def get_aoh_cacheable_prefix_len(
    sequence_len: int, sink_size: int, page_size: int = 1
) -> int:
    """Return the AoH prefix that can safely be shared by a radix tree.

    Streaming-layer KV after the anchor is request-private: the middle is
    reclaimed and the tail moves as decoding progresses.  Keeping the radix
    tree to the permanent anchor avoids retaining a stale SWA mapping. The
    shared length is rounded to a complete page so a private tail cannot free
    the same physical page; attention still masks tokens beyond ``sink_size``.
    """
    cacheable_len = ((sink_size + page_size - 1) // page_size) * page_size
    return min(sequence_len, cacheable_len)


def get_aoh_kv_groups(
    *, total_kv_heads: int, kv_tp_size: int, kv_tp_rank: int, local_kv_heads: int
) -> tuple[int, ...]:
    """Return global KV groups owned by an attention-TP rank."""
    if min(total_kv_heads, kv_tp_size, local_kv_heads) <= 0:
        raise ValueError("AoH KV-head and tensor-parallel sizes must be positive.")
    if not 0 <= kv_tp_rank < kv_tp_size:
        raise ValueError("AoH KV tensor-parallel rank is out of range.")
    if total_kv_heads <= kv_tp_size:
        if kv_tp_size % total_kv_heads:
            raise ValueError(
                "AoH requires KV tensor-parallel size to be divisible by the "
                "number of replicated KV groups."
            )
        if local_kv_heads != 1:
            raise ValueError(
                "AoH replicated KV groups require one local KV head per "
                "attention tensor-parallel rank."
            )
        return (kv_tp_rank // (kv_tp_size // total_kv_heads),)

    if local_kv_heads * kv_tp_size != total_kv_heads:
        raise ValueError(
            "AoH requires local KV heads to form a complete, non-overlapping "
            "tensor-parallel partition of the model KV groups."
        )
    first_group = kv_tp_rank * local_kv_heads
    groups = tuple(range(first_group, first_group + local_kv_heads))
    if groups[-1] >= total_kv_heads:
        raise ValueError("AoH local KV groups exceed the model KV-head count.")
    return groups


def get_aoh_kv_group(
    *, total_kv_heads: int, kv_tp_size: int, kv_tp_rank: int, local_kv_heads: int
) -> int:
    """Return the first global KV group owned by an attention-TP rank."""
    return get_aoh_kv_groups(
        total_kv_heads=total_kv_heads,
        kv_tp_size=kv_tp_size,
        kv_tp_rank=kv_tp_rank,
        local_kv_heads=local_kv_heads,
    )[0]


def get_aoh_max_kv_pages(sink_size: int, recent_size: int, page_size: int) -> int:
    """Return the fixed decode page-table width for arbitrary AoH sizes."""
    if sink_size <= 0 or recent_size <= 0 or page_size <= 0:
        raise ValueError("AoH sink, recent, and page sizes must be positive.")
    anchor_pages = (sink_size + page_size - 1) // page_size
    max_tail_pages = (recent_size + 2 * page_size - 2) // page_size
    return anchor_pages + max_tail_pages


@dataclass(frozen=True)
class AoHConfig:
    """Validated AoH v1 sidecar.

    Expected JSON shape::

        {
          "version": 1,
          "layers": {
            "3": ["streaming", "retrieval"],
            "7": ["retrieval", "streaming"]
          }
        }

    Each list is indexed by the global KV-head group.  For Qwen3.6-35B-A3B
    there are two entries per full-attention layer.
    """

    layer_modes: dict[int, tuple[str, ...]]

    @classmethod
    def load(cls, path: str) -> AoHConfig:
        config_path = Path(path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"AoH config does not exist: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid AoH JSON in {config_path}: {exc}") from exc

        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("AoH config must be a JSON object with version=1.")
        raw_layers = raw.get("layers")
        if not isinstance(raw_layers, dict) or not raw_layers:
            raise ValueError("AoH config must contain a non-empty 'layers' object.")

        layer_modes: dict[int, tuple[str, ...]] = {}
        for raw_layer_id, raw_modes in raw_layers.items():
            try:
                layer_id = int(raw_layer_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"AoH layer id {raw_layer_id!r} is not an integer."
                ) from exc
            if layer_id < 0 or not isinstance(raw_modes, list) or not raw_modes:
                raise ValueError(
                    f"AoH layer {raw_layer_id!r} must map to a non-empty mode list."
                )
            if layer_id in layer_modes:
                raise ValueError(
                    f"AoH config contains duplicate normalized layer id {layer_id}."
                )
            if not all(isinstance(mode, str) for mode in raw_modes):
                raise ValueError(f"AoH layer {layer_id} modes must all be strings.")
            modes = tuple(raw_modes)
            invalid = set(modes).difference(_VALID_MODES)
            if invalid:
                raise ValueError(
                    f"AoH layer {layer_id} has invalid modes {sorted(invalid)}; "
                    "expected 'retrieval' or 'streaming'."
                )
            layer_modes[layer_id] = modes

        return cls(layer_modes=layer_modes)

    def mode_for(self, layer_id: int, kv_group: int) -> str:
        try:
            modes = self.layer_modes[layer_id]
        except KeyError as exc:
            raise ValueError(
                f"AoH config is missing full-attention layer {layer_id}."
            ) from exc
        if kv_group < 0 or kv_group >= len(modes):
            raise ValueError(
                f"AoH layer {layer_id} has {len(modes)} KV-group modes, "
                f"but rank owns KV group {kv_group}."
            )
        return modes[kv_group]


@dataclass(frozen=True)
class AoHPagePlan:
    """Page-level view of the KV tokens visible to one streaming group."""

    page_starts: tuple[int, ...]
    actual_kv_len: int
    anchor_end: int
    tail_start: int
    total_kv_len: int
    query_start: int
    query_len: int
    can_use_causal_template: bool


def build_aoh_page_plan(
    *,
    total_kv_len: int,
    query_len: int,
    sink_size: int,
    recent_size: int,
    page_size: int,
) -> AoHPagePlan:
    """Build a compact paged-KV view for AoH prefill or decode.

    The selected pages cover the permanent anchor and every key needed by the
    earliest query. Later queries apply their own rolling recent boundary in
    the attention mask. Query tokens are always counted in the recent window.
    """
    if total_kv_len < 0 or query_len < 0 or query_len > total_kv_len:
        raise ValueError("AoH KV and query lengths must satisfy 0 <= query <= KV.")
    if sink_size <= 0 or recent_size <= 0 or page_size <= 0:
        raise ValueError("AoH sink, recent, and page sizes must be positive.")

    query_start = total_kv_len - query_len
    anchor_end = min(sink_size, total_kv_len)
    first_query_pos = query_start if query_len else total_kv_len
    tail_start = max(min(sink_size, first_query_pos), first_query_pos - recent_size + 1)

    selected_pages = set(range(0, anchor_end, page_size))
    if tail_start < total_kv_len:
        first_tail_page = tail_start // page_size * page_size
        selected_pages.update(range(first_tail_page, total_kv_len, page_size))
    page_starts = tuple(sorted(selected_pages))

    actual_kv_len = sum(
        min(page_size, total_kv_len - page_start) for page_start in page_starts
    )

    # A plain causal template is valid only while the anchor and rolling tail
    # overlap. Once a middle gap exists, each query needs an explicit mask.
    can_use_causal_template = tail_start <= anchor_end

    return AoHPagePlan(
        page_starts=page_starts,
        actual_kv_len=actual_kv_len,
        anchor_end=anchor_end,
        tail_start=tail_start,
        total_kv_len=total_kv_len,
        query_start=query_start,
        query_len=query_len,
        can_use_causal_template=can_use_causal_template,
    )
