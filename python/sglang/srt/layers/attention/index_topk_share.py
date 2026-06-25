from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class IndexTopKShareState:
    """Adapter around the ForwardBatch fields that carry DSA indexer topk across
    an MTP draft iteration (``reuse_mtp_topk_indices`` / ``topk_indices``).

    Replaces raw attribute writes scattered across the EAGLE V2 draft worker and
    the NextN decoder with an explicit begin/store/read/clear lifecycle, plus an
    ``enable_mtp_iteration`` context manager that guarantees the flags are reset
    even if a draft step raises.
    """

    def __init__(self, forward_batch: ForwardBatch):
        self.forward_batch = forward_batch

    @classmethod
    def from_forward_batch(cls, forward_batch: ForwardBatch) -> IndexTopKShareState:
        return cls(forward_batch)

    @classmethod
    def begin_mtp_iteration(cls, forward_batch: ForwardBatch) -> IndexTopKShareState:
        forward_batch.reuse_mtp_topk_indices = True
        forward_batch.topk_indices = None
        return cls(forward_batch)

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.forward_batch, "reuse_mtp_topk_indices", False))

    def prev_topk_indices(self) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        return getattr(self.forward_batch, "topk_indices", None)

    def store_topk_indices(self, topk_indices: Optional[torch.Tensor]) -> None:
        if self.enabled:
            self.forward_batch.topk_indices = topk_indices

    def clear_mtp_iteration(self) -> None:
        self.forward_batch.topk_indices = None
        self.forward_batch.reuse_mtp_topk_indices = False

    @classmethod
    @contextmanager
    def enable_mtp_iteration(
        cls, forward_batch: ForwardBatch
    ) -> Iterator[IndexTopKShareState]:
        state = cls.begin_mtp_iteration(forward_batch)
        try:
            yield state
        finally:
            state.clear_mtp_iteration()
