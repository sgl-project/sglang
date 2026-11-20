"""The algorithm-neutral contract between the engine and a watermark algorithm.

Shared engine code calls only these types and the entry points in ``runtime``;
everything a particular algorithm does lives behind them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol

from torch import Tensor

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


class WatermarkRows(Protocol):
    """One batch's per-row watermark policy.

    Exists only if at least one row is a candidate (enabled and top_k > 1); the
    batch carries None otherwise. Immutable: an overlapped forward may still read
    the old object, so filter and concat return new objects.
    """

    def filter(
        self, keep_indices: list[int], keep_indices_device: Tensor
    ) -> Optional[WatermarkRows]:
        """The kept rows, or None when no candidate row is kept."""
        ...

    def concat(
        self, other: Optional[WatermarkRows], self_len: int, other_len: int
    ) -> WatermarkRows:
        """These rows followed by ``other``'s, or by ``other_len`` off rows."""
        ...

    def prepend_off(self, num_rows: int) -> WatermarkRows:
        """``num_rows`` off rows followed by these rows."""
        ...


def merge_watermark_rows(
    a: Optional[WatermarkRows],
    b: Optional[WatermarkRows],
    a_len: int,
    b_len: int,
) -> Optional[WatermarkRows]:
    if a is None:
        return None if b is None else b.prepend_off(a_len)
    return a.concat(b, a_len, b_len)


@dataclass(frozen=True)
class SampleStep:
    # What the stock sampler must sample with.
    sampling_info: SamplingBatchInfo
    # The pre-watermark logits, iff the caller asked to keep them for logprobs.
    logprob_logits: Optional[Tensor]
    # Set when the algorithm already sampled; the stock sampler is then skipped.
    token_ids: Optional[Tensor] = None


class SampleWatermark(Protocol):
    """One ordinary sampling step's watermark, bound to its forward batch."""

    def apply(self, logits: Tensor, *, keep_logits: bool) -> SampleStep: ...

    def commit(self, token_ids: Tensor) -> None: ...


class VerifyWatermark(Protocol):
    """One speculative verify's watermark, bound to its batch rows."""

    def apply(
        self,
        logits: Tensor,
        *,
        sampling_info: Any,
        req_pool_indices: Tensor,
        draft_tokens: Tensor,
        positions: Tensor,
        draft_token_num: int,
        custom_mask: Optional[Tensor],
        keep_logits: bool,
    ) -> tuple[Tensor, Any]:
        """The logits to verify against and the sampling info to verify with.

        ``custom_mask`` None means a chain draft: ``draft_tokens`` and
        ``positions`` are [batch, draft_token_num].
        """
        ...

    def record(self, accept_index: Tensor, accept_lens: Tensor) -> None: ...


class WatermarkEngine(Protocol):
    """Per-server device state, owned by the target ModelRunner (``model_runner.watermark``)."""

    def seed_payload(self, batch: ScheduleBatch) -> Optional[Any]:
        """Host-side state the next sample seeds from (``ForwardBatch.watermark_seed``)."""
        ...

    def bind_sample(self, forward_batch: ForwardBatch) -> Optional[SampleWatermark]:
        """The batch's sampling watermark, or None when it has no watermark rows."""
        ...

    def bind_verify(
        self, rows: WatermarkRows, *, full_mask: bool
    ) -> VerifyWatermark: ...

    def append_accept_tokens(
        self,
        rows: WatermarkRows,
        req_pool_indices: Tensor,
        accept_tokens: Tensor,
        accept_lens: Tensor,
    ) -> None: ...
