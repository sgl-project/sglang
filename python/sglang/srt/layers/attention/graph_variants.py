from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class AttentionGraphVariants(Protocol):
    # Capture order is significant when variants share a graph memory pool.
    capture_labels: tuple[str, ...]

    def select(self, forward_batch: ForwardBatch) -> str:
        """Select one of capture_labels from host-side batch metadata."""
        ...


def create_attention_graph_variants(
    hf_config, forward_mode: ForwardMode
) -> Optional[AttentionGraphVariants]:
    from sglang.srt.configs.model_config import get_dsa_index_topk, is_deepseek_dsa
    from sglang.srt.layers.attention.dsa.graph_variants import DsaGraphVariants
    from sglang.srt.utils import is_hip

    if is_hip() and is_deepseek_dsa(hf_config):
        return DsaGraphVariants(get_dsa_index_topk(hf_config))
    return None
