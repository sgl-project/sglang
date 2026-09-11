from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


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
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
            # Host-side mirror (maintained incrementally for plain decode) — no
            # d2h sync needed.
            max_kv_len = int(seq_lens_cpu.max().item())
        elif forward_batch.seq_lens is not None and forward_batch.seq_lens.numel() > 0:
            # Fallback: a single scalar reduction d2h (cheap, per-step).
            max_kv_len = int(forward_batch.seq_lens.max().item())
        else:
            # No length info: be safe and use the correct-for-all sparse graph.
            return "sparse"
        return "dense" if max_kv_len <= self.index_topk else "sparse"
