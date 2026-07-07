from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class IndexTopKShareState:
    """Adapter around the state that carries DSA indexer topk across an MTP
    draft iteration: the ``reuse_dsa_topk_indices`` flag on ForwardBatch and the
    carried topk on ``spec_info.dsa_topk_indices`` (the carry must live on
    spec_info, not ForwardBatch — per-step ForwardBatch copies drop it, #29654).

    Replaces raw attribute writes scattered across the EAGLE V2 draft worker and
    the NextN decoder with an explicit read/store lifecycle, plus the
    ``mtp_iteration`` context manager that guarantees the flags are reset even
    if a draft step raises.
    """

    def __init__(self, forward_batch: ForwardBatch):
        self.forward_batch = forward_batch

    @property
    def enabled(self) -> bool:
        return bool(self.forward_batch.reuse_dsa_topk_indices)

    def prev_topk_indices(self) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        return self.forward_batch.spec_info.dsa_topk_indices

    def store_topk_indices(self, topk_indices: Optional[torch.Tensor]) -> None:
        if self.enabled:
            self.forward_batch.spec_info.dsa_topk_indices = topk_indices

    @classmethod
    @contextmanager
    def mtp_iteration(
        cls,
        forward_batch: ForwardBatch,
        enabled: bool = True,
        keep_carry_seed: bool = False,
    ) -> Iterator[Optional[IndexTopKShareState]]:
        if not enabled:
            yield None
            return
        spec_info = forward_batch.spec_info
        forward_batch.reuse_dsa_topk_indices = True
        # Keep the draft-extend seed so step 0 reuses it; else recompute it.
        if not (keep_carry_seed and spec_info.dsa_topk_indices is not None):
            spec_info.dsa_topk_indices = None
        try:
            yield cls(forward_batch)
        finally:
            spec_info.dsa_topk_indices = None
            forward_batch.reuse_dsa_topk_indices = False
