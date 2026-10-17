"""Paged experts: serve an MoE model whose routed experts do not fit in GPU memory.

Each MoE layer keeps K of its E routed experts in a K-slot GPU table and pages the rest from
pinned host memory; outputs are bit-identical to the unpaged layer. See
``docs/docs/advanced_features/paged_experts.mdx`` for the design and the execution flow.

Modules: ``formats`` (per quantization method), ``runners`` (per runner backend), ``store``
(host copy and transfers), ``residency`` (slots and waves), ``executor`` (one forward step),
``method`` (the ``FusedMoEMethodBase`` wiring them into a ``FusedMoE`` layer).
"""

from sglang.srt.layers.moe.paged_experts.method import (
    PagedExpertsMoEMethod,
    make_for_layer,
)

__all__ = ["PagedExpertsMoEMethod", "make_for_layer"]
