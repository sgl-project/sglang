from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


logger = logging.getLogger(__name__)


class AttentionGraphVariants(Protocol):
    # Capture order is significant when variants share a graph memory pool.
    capture_labels: tuple[str, ...]

    def select(self, forward_batch: ForwardBatch) -> str:
        """Select one of capture_labels from host-side batch metadata."""
        ...


@dataclass(frozen=True)
class DsaGraphVariants:
    index_topk: int
    # Dense comes first: the sparse capture peak subsumes its shared-pool storage.
    capture_labels: ClassVar[tuple[str, ...]] = ("dense", "sparse")

    def __post_init__(self):
        logger.info(
            "[dense-decode] DSA dual-graph enabled: capturing "
            "dense (k-only) + sparse (full indexer) decode graphs; "
            "dispatch on max_kv_len vs index_topk=%d.",
            self.index_topk,
        )

    def select(self, forward_batch: ForwardBatch) -> str:
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
            # Plain decode maintains this host mirror without a D2H sync.
            max_kv_len = int(seq_lens_cpu.max().item())
        elif forward_batch.seq_lens is not None and forward_batch.seq_lens.numel() > 0:
            # Fallback: a single scalar reduction d2h (cheap, per-step).
            max_kv_len = int(forward_batch.seq_lens.max().item())
        else:
            # No length info: be safe and use the correct-for-all sparse graph.
            return "sparse"
        return "dense" if max_kv_len <= self.index_topk else "sparse"


def create_attention_graph_variants(
    hf_config, forward_mode: ForwardMode
) -> Optional[AttentionGraphVariants]:
    from sglang.srt.configs.model_config import get_dsa_index_topk, is_deepseek_dsa
    from sglang.srt.utils import is_hip

    if is_hip() and is_deepseek_dsa(hf_config):
        return DsaGraphVariants(get_dsa_index_topk(hf_config))
    return None
