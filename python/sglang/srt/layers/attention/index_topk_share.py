from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class IndexTopKShareState:
    def __init__(
        self,
        forward_batch: ForwardBatch,
        topk_indices: Optional[torch.Tensor],
    ):
        self._forward_batch = forward_batch
        self._topk_indices = topk_indices

    @classmethod
    def from_mtp_carry(cls, forward_batch: ForwardBatch) -> IndexTopKShareState:
        topk_indices = (
            forward_batch.spec_info.dsa_topk_indices
            if forward_batch.reuse_dsa_topk_indices
            else None
        )
        return cls(forward_batch, topk_indices)

    @property
    def enabled(self) -> bool:
        return bool(self._forward_batch.reuse_dsa_topk_indices)

    @property
    def _seed_buf(self) -> Optional[torch.Tensor]:
        if self._forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            return self._forward_batch.spec_info.dsa_seed_topk_capture
        return None

    @property
    def should_publish(self) -> bool:
        return self.enabled or self._seed_buf is not None

    @property
    def topk_indices(self) -> Optional[torch.Tensor]:
        return self._topk_indices

    def update(self, topk_indices: Optional[torch.Tensor]) -> None:
        self._topk_indices = topk_indices

    def publish(self) -> None:
        if self._topk_indices is None or not self.should_publish:
            return
        if self.enabled:
            self._forward_batch.spec_info.dsa_topk_indices = self._topk_indices
        seed_buf = self._seed_buf
        if seed_buf is not None:
            sel = self._forward_batch.spec_info.dsa_seed_topk_select
            src = (
                self._topk_indices[: seed_buf.shape[0]]
                if sel is None
                else self._topk_indices[sel]
            )
            seed_buf[: src.shape[0]].copy_(src)

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
            yield cls.from_mtp_carry(forward_batch)
        finally:
            spec_info.dsa_topk_indices = None
            forward_batch.reuse_dsa_topk_indices = False
