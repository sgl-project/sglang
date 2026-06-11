"""ShapeKey — typed identifier for one captured CUDA-graph shape.

Replaces the previous `int | str` keys produced by the legacy
`_make_graph_key(bs, stream_idx, variant_label)` (which f-string'd
into stringly-typed forms like `"lora_0_8"`). Carries the size of
the captured input/output (bs and token count are fused — see
DecodeCudaGraphRunner) so backends that share output buffers across
capture sizes (BCG) don't need a side-channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShapeKey:
    """Identifies one captured CUDA-graph shape across all runners.

    size: number of rows in the captured input/output tensor.
        - prefill: num_tokens
        - decode:  bs * num_tokens_per_bs (bs and token count are fused)
    stream_idx:   pdmux stream index, or None for single-stream runners.
    variant_label: LoRA-variant label ("lora" / "nolora"), or None
        for runners that don't record per-variant graphs.
    """

    size: int
    stream_idx: Optional[int] = None
    variant_label: Optional[str] = None
