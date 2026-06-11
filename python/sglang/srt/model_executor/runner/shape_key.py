"""ShapeKey — typed identifier for one captured CUDA-graph shape.

Replaces the previous `int | str` keys produced by the legacy
`_make_graph_key(bs, stream_idx, variant_label)` (which f-string'd
into stringly-typed forms like `"lora_0_8"`). The `size` field
preserves the original per-phase shape semantics (prefill →
num_tokens, decode → bs) and doubles as the leading-dim BCG slices
to in prefill (where size == num_tokens == leading dim of the
captured output).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShapeKey:
    """Identifies one captured CUDA-graph shape across all runners.

    size: the per-phase capture size — what the runner iterates over.
        - prefill: num_tokens
        - decode:  bs
    stream_idx:   pdmux stream index, or None for single-stream runners.
    variant_label: LoRA-variant label ("lora" / "nolora"), or None
        for runners that don't record per-variant graphs.
    """

    size: int
    stream_idx: Optional[int] = None
    variant_label: Optional[str] = None
